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
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class MECDataset(torch.utils.data.Dataset):
    """Dataset for MEC CloudVolume data.

    Loads the entire volume into memory and returns random 160x160 XY slices.
    """
    def __init__(self, volume_data, split, V=1, num_samples=10000):
        """
        Args:
            volume_data: numpy array of shape (X, Y, Z, C) containing the full volume
            split: "train" or "val" (determines which augmentations to use)
            V: number of views to return per sample
            num_samples: number of samples to generate per epoch
        """
        self.volume_data = volume_data
        self.V = V
        self.num_samples = num_samples
        self.is_train = (split == "train")

        # Get volume dimensions
        self.X, self.Y, self.Z = volume_data.shape[:3]

        # Size of each slice to extract (160x160 to match Imagenette)
        self.slice_size = 160

        # Define augmentations for training
        # Note: removed ColorJitter since this is grayscale EM data
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),  # Added since EM data has no preferred orientation
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Simple normalization for grayscale
            ]
        )

        # Define transforms for validation/test
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, i):
        """Returns V random 160x160 XY slices from the volume."""
        slices = []
        for _ in range(self.V):
            # Random position for 160x160 crop
            x = np.random.randint(0, self.X - self.slice_size + 1)
            y = np.random.randint(0, self.Y - self.slice_size + 1)
            z = np.random.randint(0, self.Z)

            # Extract 160x160 XY slice at random Z
            slice_data = self.volume_data[x:x+self.slice_size, y:y+self.slice_size, z, 0]

            # Convert to torch tensor and replicate to 3 channels for RGB compatibility
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0)  # Add channel dim
            slice_tensor = slice_tensor.repeat(3, 1, 1)  # Replicate to 3 channels

            # Apply augmentations
            transform = self.aug if self.is_train else self.test
            slice_tensor = transform(slice_tensor)
            slices.append(slice_tensor)

        # Return stacked views and dummy label (0, since we're not doing classification)
        return torch.stack(slices), 0

    def __len__(self):
        return self.num_samples


# LeJEPA main loop:


@hydra.main(version_base=None, config_path=".", config_name="config")
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
    train_ds = MECDataset(volume_data, "train", V=cfg.V, num_samples=cfg.train_samples)
    test_ds = MECDataset(volume_data, "val", V=1, num_samples=cfg.val_samples)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ConvNeXtEncoder(proj_dim=cfg.proj_dim).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler (no probe for unsupervised learning)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2)
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=torch.cuda.is_available())
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
        wandb.log({
            "val/lejepa": val_lejepa_loss / num_val_batches,
            "val/sigreg": val_sigreg_loss / num_val_batches,
            "val/inv": val_inv_loss / num_val_batches,
        })
    wandb.finish()


if __name__ == "__main__":
    main()
