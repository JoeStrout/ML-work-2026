# LeJEPA with ConvNeXt on MEC CloudVolume data
# Based on minimal LeJEPA example from:
# https://github.com/JoeStrout/lejepa/blob/main/MINIMAL.md

import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from cloudvolume import CloudVolume
import matplotlib.pyplot as plt
import matplotlib

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ConvNeXtEncoder(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=False,
            in_chans=1,  # Single-channel input for grayscale EM data
            num_classes=emb_dim,
            drop_path_rate=0.1
        )
        self.proj = MLP(emb_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class MECDataset(torch.utils.data.Dataset):
    """Dataset for MEC CloudVolume data.

    Loads the entire volume into memory and returns random 160x160 XY slices.
    """
    def __init__(self, volume_data, split, V=1, num_samples=10000, p_z_view=0.5):
        """
        Args:
            volume_data: numpy array of shape (X, Y, Z, C) containing the full volume
            split: "train" or "val" (determines which augmentations to use)
            V: number of views to return per sample
            num_samples: number of samples to generate per epoch
            p_z_view: probability of using neighboring Z-slice (vs same Z with augmentation)
        """
        self.volume_data = volume_data
        self.V = V
        self.num_samples = num_samples
        self.is_train = (split == "train")
        self.p_z_view = p_z_view

        # Get volume dimensions
        self.X, self.Y, self.Z = volume_data.shape[:3]

        # Size of each slice to extract (160x160 to match Imagenette)
        self.slice_size = 160

        # Define augmentations (used for both train and val)
        # Note: removed ColorJitter since this is grayscale EM data
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.5, 1.0)),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),  # Added since EM data has no preferred orientation
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),  # Single-channel normalization for grayscale
            ]
        )

    def __getitem__(self, i):
        """Returns V related views of the same region from the volume.

        Views are either:
        - Neighboring Z-slices of the same XY region (with probability p_z_view)
        - The same Z-slice with different augmentations
        All views have augmentations applied.
        """
        # Pick ONE base random XY position and Z
        base_x = np.random.randint(0, self.X - self.slice_size + 1)
        base_y = np.random.randint(0, self.Y - self.slice_size + 1)
        base_z = np.random.randint(0, self.Z)

        slices = []
        for view_idx in range(self.V):
            # Decide whether to use neighboring Z-slice or same Z
            # Note: Z variation happens in both train and val to measure 3D invariance
            if np.random.rand() < self.p_z_view:
                # Use neighboring Z-slice (offset by -2, -1, 0, +1, +2)
                # Make sure we stay within bounds
                max_offset = min(2, base_z, self.Z - base_z - 1)
                if max_offset > 0:
                    z_offset = np.random.randint(-max_offset, max_offset + 1)
                else:
                    z_offset = 0
                z = base_z + z_offset
            else:
                # Use the same Z
                z = base_z

            # Extract 160x160 XY slice at chosen Z
            slice_data = self.volume_data[base_x:base_x+self.slice_size,
                                         base_y:base_y+self.slice_size,
                                         z, 0]

            # Convert to torch tensor (single channel for grayscale)
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0)  # Add channel dim

            # Apply augmentations (different random augmentations for each view)
            # Use augmentations for both train and val to measure same inv_loss
            slice_tensor = self.aug(slice_tensor)
            slices.append(slice_tensor)

        # Return stacked views and dummy label (0, since we're not doing classification)
        return torch.stack(slices), 0

    def __len__(self):
        return self.num_samples


def save_views_to_disk(dataloader, net, epoch, output_dir="view_samples", num_samples=3):
    """Save a grid of views with their projection vectors to disk for visual inspection.

    Args:
        dataloader: DataLoader to sample from
        net: Network to compute projection vectors
        epoch: Current epoch number
        output_dir: Directory to save images
        num_samples: Number of samples to visualize
    """
    matplotlib.use('Agg')  # Use non-interactive backend
    os.makedirs(output_dir, exist_ok=True)

    # Get one batch
    for vs, _ in dataloader:
        # vs shape: (batch_size, V, C, H, W)
        batch_size, V, C, H, W = vs.shape

        # Take first num_samples from batch
        num_samples = min(num_samples, batch_size)

        # Compute projection vectors
        with torch.no_grad():
            vs_subset = vs[:num_samples].to("cuda")
            _, proj = net(vs_subset)  # proj shape: (V, num_samples, proj_dim)
            proj = proj.cpu().numpy()  # Now (V, num_samples, proj_dim)

        # Create figure: num_samples x (2 rows per sample) x V columns
        # Each sample has 2 rows: top for images, bottom for bar charts
        # Use height_ratios to make bar charts half the height of images
        height_ratios = [2, 1] * num_samples  # Alternating image (2) and bar (1) rows
        fig, axes = plt.subplots(num_samples * 2, V, figsize=(V * 2, num_samples * 3),
                                 gridspec_kw={'height_ratios': height_ratios})
        if num_samples == 1:
            axes = axes.reshape(2, -1)

        for sample_idx in range(num_samples):
            for view_idx in range(V):
                # Top row: images
                ax_img = axes[sample_idx * 2, view_idx]
                # Get image and denormalize (was normalized with mean=0.5, std=0.5)
                img = vs[sample_idx, view_idx, 0].cpu().numpy()  # Get single channel
                img = img * 0.5 + 0.5  # Denormalize
                img = np.clip(img, 0, 1)

                ax_img.imshow(img, cmap='gray')
                ax_img.axis('off')
                if sample_idx == 0:
                    ax_img.set_title(f'View {view_idx}')

                # Bottom row: projection vector bar charts
                ax_bar = axes[sample_idx * 2 + 1, view_idx]
                proj_vec = proj[view_idx, sample_idx, :]  # Get projection for this view

                # Generate distinct colors for each dimension
                colors = plt.cm.tab20(np.linspace(0, 1, len(proj_vec)))

                ax_bar.bar(range(len(proj_vec)), proj_vec, color=colors)
                ax_bar.set_xlim(-0.5, len(proj_vec) - 0.5)
                # Remove x-axis labels
                ax_bar.tick_params(labelbottom=False, labelsize=6)
                # Set consistent y-axis limits across all views for comparison
                ax_bar.set_ylim(-2, 2)
                ax_bar.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved view samples to {save_path}")
        break  # Only need one batch


# LeJEPA main loop:


@hydra.main(version_base=None, config_path=".", config_name="config_lejepa")
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA_MEC_Dec2025", config=dict(cfg))
    torch.manual_seed(0)

    # Load CloudVolume data
    print("Loading CloudVolume data from gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/")
    cv_path = "gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/"
    cv = CloudVolume(cv_path, mip=0, use_https=False)

    # Define the bounds of the volume
    x_start, y_start, z_start = 293050, 255714, 1122
    x_size, y_size, z_size = 5904, 5904, 120

    print(f"Loading volume: [{x_start}:{x_start+x_size}, {y_start}:{y_start+y_size}, {z_start}:{z_start+z_size}]")
    print(f"Volume size: {x_size} x {y_size} x {z_size} = {x_size*y_size*z_size:,} voxels")
    print(f"Estimated memory: {x_size*y_size*z_size*cv.dtype.itemsize / (1024**3):.2f} GB")

    # Load the entire volume into memory
    volume_data = cv[
        x_start:x_start+x_size,
        y_start:y_start+y_size,
        z_start:z_start+z_size
    ]
    print(f"Volume loaded! Shape: {volume_data.shape}, dtype: {volume_data.dtype}")

    # Convert to numpy array if needed
    volume_data = np.array(volume_data)

    # Create datasets
    train_ds = MECDataset(volume_data, "train", V=cfg.V, num_samples=cfg.train_samples, p_z_view=cfg.p_z_view)
    test_ds = MECDataset(volume_data, "val", V=cfg.V, num_samples=cfg.val_samples, p_z_view=cfg.p_z_view)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ConvNeXtEncoder(emb_dim=cfg.emb_dim, proj_dim=cfg.proj_dim).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler (no probe for unsupervised learning)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2)
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training
    for epoch in range(cfg.epochs):
        net.train()
        epoch_lejepa_loss = 0
        epoch_sigreg_loss = 0
        epoch_inv_loss = 0

        for vs, _ in tqdm.tqdm(train, total=len(train), desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            with autocast(dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                loss = lejepa_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # Accumulate losses for epoch averaging
            epoch_lejepa_loss += lejepa_loss.item()
            epoch_sigreg_loss += sigreg_loss.item()
            epoch_inv_loss += inv_loss.item()

            wandb.log(
                {
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                }
            )

        # Log epoch averages
        num_batches = len(train)
        wandb.log({
            "epoch": epoch,
            "epoch/lejepa": epoch_lejepa_loss / num_batches,
            "epoch/sigreg": epoch_sigreg_loss / num_batches,
            "epoch/inv": epoch_inv_loss / num_batches,
        })

        # Validation: compute loss on validation set
        net.eval()
        val_lejepa_loss = 0
        val_sigreg_loss = 0
        val_inv_loss = 0
        with torch.inference_mode():
            for vs, _ in test:
                vs = vs.to("cuda", non_blocking=True)
                with autocast(dtype=torch.bfloat16):
                    emb, proj = net(vs)
                    inv_loss = (proj.mean(0) - proj).square().mean()
                    sigreg_loss = sigreg(proj)
                    lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

                    val_lejepa_loss += lejepa_loss.item()
                    val_sigreg_loss += sigreg_loss.item()
                    val_inv_loss += inv_loss.item()

        num_val_batches = len(test)
        avg_val_lejepa = val_lejepa_loss / num_val_batches
        wandb.log({
            "val/lejepa": avg_val_lejepa,
            "val/sigreg": val_sigreg_loss / num_val_batches,
            "val/inv": val_inv_loss / num_val_batches,
        })

        # Save view samples every 50 epochs (including epoch 0)
        if epoch % 50 == 0:
            save_views_to_disk(train, net, epoch)

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0 or epoch == cfg.epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch{epoch:04d}_valloss{avg_val_lejepa:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_lejepa_loss': avg_val_lejepa,
                'config': dict(cfg),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
