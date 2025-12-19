"""
Training script for multi-class semantic segmentation with hierarchical supervision.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import wandb
import hydra
from omegaconf import DictConfig
import tqdm
from cloudvolume import CloudVolume
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

from multi_unet_arch import ConvNeXtUNet


# Constants
IGNORE_INDEX = -1

# Class names for visualization and logging
CLASS_NAMES = [
    "extracellular",  # 0
    "tear",           # 1
    "dendrite",       # 2
    "axon",           # 3
    "soma",           # 4
    "glia",           # 5
    "myelin",         # 6
    "myelin_inner",   # 7
    "myelin_outer",   # 8
    "nucleus",        # 9
    "mitochondria",   # 10
    "fat_globule"     # 11
]

COARSE_GROUP_NAMES = [
    "neuron_part",     # 0
    "glia",            # 1
    "myelin_related",  # 2
    "organelle",       # 3
    "non_tissue"       # 4
]

# Coarse group lookup table: maps fine class (0-11) to 5-bit coarse vector
# Each row corresponds to one fine class, columns are: [neuron_part, glia, myelin_related, organelle, non_tissue]
COARSE_LUT = torch.tensor([
    # extracellular space (index 0)
    [0, 0, 0, 0, 1],
    # tear (index 1)
    [0, 0, 0, 0, 1],
    # dendrite (index 2)
    [1, 0, 0, 0, 0],
    # axon (index 3)
    [1, 0, 0, 0, 0],
    # soma (index 4)
    [1, 0, 0, 0, 0],
    # glia (index 5)
    [0, 1, 0, 0, 0],
    # myelin (index 6)
    [0, 0, 1, 0, 0],
    # myelin inner tongue (index 7)
    [0, 0, 1, 0, 0],
    # myelin outer tongue (index 8)
    [0, 0, 1, 0, 0],
    # nucleus (index 9)
    [0, 0, 0, 1, 0],
    # mitochondria (index 10)
    [0, 0, 0, 1, 0],
    # fat globule (index 11)
    [0, 0, 0, 1, 0],
], dtype=torch.float32)


def make_coarse_targets(y_fine: torch.Tensor) -> torch.Tensor:
    """
    Convert fine labels to coarse group targets.

    Args:
        y_fine: [B, H, W] with values 0..11 or -1 (IGNORE_INDEX)

    Returns:
        y_coarse: [B, K, H, W] float {0,1}, where K=5
    """
    B, H, W = y_fine.shape
    K = COARSE_LUT.shape[1]

    y_coarse = torch.zeros((B, K, H, W), device=y_fine.device, dtype=torch.float32)

    valid = (y_fine != IGNORE_INDEX)
    if valid.any():
        yv = y_fine[valid].long()  # [N]
        cv = COARSE_LUT.to(y_fine.device)[yv]  # [N, K]

        # Scatter back
        y_coarse_perm = y_coarse.permute(0, 2, 3, 1)  # [B, H, W, K]
        y_coarse_perm[valid] = cv
        y_coarse = y_coarse_perm.permute(0, 3, 1, 2)  # [B, K, H, W]

    return y_coarse


def compute_losses(logits_fine, logits_coarse, y_fine, lambda_coarse=0.2):
    """
    Compute hierarchical loss with fine and coarse supervision.

    Args:
        logits_fine: [B, 12, H, W]
        logits_coarse: [B, 5, H, W]
        y_fine: [B, H, W] in 0..11 or -1 (IGNORE_INDEX)
        lambda_coarse: Weight for coarse loss

    Returns:
        total_loss: Combined loss
        loss_fine: Fine classification loss
        loss_coarse: Coarse group loss
    """
    valid = (y_fine != IGNORE_INDEX).float()  # [B, H, W]
    denom = valid.sum().clamp_min(1.0)

    # Fine CE with ignore_index
    ce = F.cross_entropy(logits_fine, y_fine, ignore_index=IGNORE_INDEX, reduction='none')
    loss_fine = (ce * valid).sum() / denom

    # Coarse BCE
    y_coarse = make_coarse_targets(y_fine)
    bce = F.binary_cross_entropy_with_logits(logits_coarse, y_coarse, reduction='none')
    bce_pixel = bce.mean(dim=1)  # [B, H, W]
    loss_coarse = (bce_pixel * valid).sum() / denom

    total_loss = loss_fine + lambda_coarse * loss_coarse
    return total_loss, loss_fine, loss_coarse


class MultiClassDataset(Dataset):
    """Dataset for multi-class semantic segmentation."""

    def __init__(self, img_volume, seg_volume, img_bounds, seg_bounds, split, patch_size=128, num_samples=5000):
        """
        Args:
            img_volume: Full image volume (numpy array)
            seg_volume: Full segmentation volume (numpy array)
            img_bounds: CloudVolume bounds for image (minpt, maxpt)
            seg_bounds: CloudVolume bounds for segmentation (minpt, maxpt)
            split: "train" or "val"
            patch_size: Size of square patches to extract
            num_samples: Number of samples per epoch
        """
        self.img_volume = img_volume
        self.seg_volume = seg_volume
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.is_train = (split == "train")

        # Compute valid sampling region (overlap between image and segmentation)
        # Both volumes have same coordinate system
        self.img_min = np.array(img_bounds[0])
        self.seg_min = np.array(seg_bounds[0])
        self.seg_max = np.array(seg_bounds[1])

        # Segmentation volume dimensions
        self.seg_shape = seg_volume.shape[:3]  # (X, Y, Z)

        # For train/val split, divide Z slices
        # First 80% for train, last 20% for val
        z_split = int(0.8 * self.seg_shape[2])
        if self.is_train:
            self.z_range = (0, z_split)
        else:
            self.z_range = (z_split, self.seg_shape[2])

        print(f"{split} dataset: Z range {self.z_range}, {num_samples} samples per epoch")

    def __getitem__(self, idx):
        """
        Returns a random patch and its multi-class segmentation labels.

        Returns:
            image: Tensor of shape (1, patch_size, patch_size)
            labels: Tensor of shape (patch_size, patch_size) with class indices 0-11 or -1
        """
        # Pick random XY position within segmentation volume
        # Need to ensure patch_size doesn't exceed bounds
        max_x = self.seg_shape[0] - self.patch_size
        max_y = self.seg_shape[1] - self.patch_size

        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        z = np.random.randint(self.z_range[0], self.z_range[1])

        # Extract patch from segmentation volume (already aligned)
        seg_patch = self.seg_volume[x:x+self.patch_size, y:y+self.patch_size, z, 0]

        # Extract corresponding patch from image volume
        # Need to convert from segmentation volume coords to image volume coords
        # Both volumes use same coordinate system, but different offsets
        img_x = x
        img_y = y
        # Z coordinate is same in both volumes
        img_patch = self.img_volume[img_x:img_x+self.patch_size, img_y:img_y+self.patch_size, z, 0]

        # Convert labels: keep 0-11 as is, convert 255 (uncertain) to -1 (IGNORE_INDEX)
        labels = seg_patch.astype(np.int64)
        labels[labels == 255] = IGNORE_INDEX

        # Convert to tensors and normalize image
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0).float()  # (1, H, W)
        label_tensor = torch.from_numpy(labels).long()  # (H, W)

        # Normalize image to [0, 1]
        img_tensor = img_tensor / 255.0

        return img_tensor, label_tensor

    def __len__(self):
        return self.num_samples


def compute_metrics(logits_fine, y_fine, num_classes=12):
    """
    Compute multi-class segmentation metrics (ignoring IGNORE_INDEX pixels).

    Args:
        logits_fine: [B, 12, H, W]
        y_fine: [B, H, W] in 0..11 or -1
        num_classes: Number of classes (12)

    Returns:
        metrics: Dictionary with accuracy, mean_iou, per_class_iou
    """
    preds = torch.argmax(logits_fine, dim=1)  # [B, H, W]

    valid_mask = (y_fine != IGNORE_INDEX)

    # Accuracy (only on valid pixels)
    correct = (preds == y_fine) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().clamp_min(1)

    # Per-class IoU (only on valid pixels)
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c) & valid_mask
        target_c = (y_fine == c) & valid_mask

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            iou_c = intersection / union
        else:
            iou_c = torch.tensor(float('nan'))

        ious.append(iou_c.item())

    # Mean IoU (ignoring NaN values from missing classes)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        'accuracy': accuracy.item(),
        'mean_iou': mean_iou,
        'per_class_iou': ious
    }


def save_predictions(dataloader, model, epoch, output_dir="multi_seg_predictions", num_samples=4):
    """
    Save visualization of multi-class predictions vs ground truth.

    Args:
        dataloader: DataLoader to sample from
        model: Trained model
        epoch: Current epoch number
        output_dir: Directory to save images
        num_samples: Number of samples to visualize
    """
    matplotlib.use('Agg')
    os.makedirs(output_dir, exist_ok=True)

    # Create colormap for 12 classes (use tab20 for distinct colors)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    cmap = ListedColormap(colors)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to("cuda")
            labels = labels.to("cuda")

            # Get predictions
            logits_fine, logits_coarse = model(images)
            preds = torch.argmax(logits_fine, dim=1)  # [B, H, W]

            # Move to CPU for visualization
            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()

            # Take first num_samples from batch
            num_samples = min(num_samples, images.shape[0])

            # Create figure: num_samples rows x 4 columns
            # Columns: [Image, Ground Truth, Prediction, Difference]
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                img = images[i, 0]  # (H, W)
                label = labels[i]  # (H, W)
                pred = preds[i]  # (H, W)

                # Mask out IGNORE_INDEX for visualization
                label_vis = label.copy()
                label_vis[label == IGNORE_INDEX] = 0  # Show as extracellular

                # Column 0: Image
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title('EM Image' if i == 0 else '')
                axes[i, 0].axis('off')

                # Column 1: Ground truth labels
                axes[i, 1].imshow(label_vis, cmap=cmap, vmin=0, vmax=11)
                axes[i, 1].set_title('Ground Truth' if i == 0 else '')
                axes[i, 1].axis('off')

                # Column 2: Prediction
                axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=11)
                axes[i, 2].set_title('Prediction' if i == 0 else '')
                axes[i, 2].axis('off')

                # Column 3: Difference map (correct=black, incorrect=red)
                valid_mask = (label != IGNORE_INDEX)
                correct_mask = (pred == label) & valid_mask
                incorrect_mask = (pred != label) & valid_mask

                diff_vis = np.zeros((img.shape[0], img.shape[1], 3))
                diff_vis[correct_mask] = [0, 1, 0]  # Green for correct
                diff_vis[incorrect_mask] = [1, 0, 0]  # Red for incorrect

                axes[i, 3].imshow(img, cmap='gray', alpha=0.5)
                axes[i, 3].imshow(diff_vis, alpha=0.5)
                axes[i, 3].set_title('Difference (G=correct, R=error)' if i == 0 else '')
                axes[i, 3].axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved predictions to {save_path}")
            break  # Only need one batch


@hydra.main(version_base=None, config_path=".", config_name="multi_seg_config")
def main(cfg: DictConfig):
    # Initialize WandB (will be updated if resuming)
    wandb_run_id = None
    if cfg.get('resume_checkpoint') is not None:
        # Try to load run_id from checkpoint
        try:
            checkpoint = torch.load(cfg.resume_checkpoint, map_location='cpu')
            wandb_run_id = checkpoint.get('wandb_run_id', None)
            if wandb_run_id:
                print(f"Resuming WandB run: {wandb_run_id}")
        except:
            pass

    if wandb_run_id:
        wandb.init(project="MultiSeg_Dec2025", id=wandb_run_id, resume="allow", config=dict(cfg))
    else:
        wandb.init(project="MultiSeg_Dec2025", config=dict(cfg))

    torch.manual_seed(42)

    # Load CloudVolume data
    print("Loading image and segmentation volumes...")
    img_path = "gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/"
    seg_path = "gs://joe_exp/mec_training/mec_sem_cls"

    cv_img = CloudVolume(img_path, mip=0, use_https=False)
    cv_seg = CloudVolume(seg_path, mip=0, use_https=False)

    # Get bounds
    img_bounds = (cv_img.bounds.minpt, cv_img.bounds.maxpt)
    seg_bounds = (cv_seg.bounds.minpt, cv_seg.bounds.maxpt)

    print(f"Image bounds: {img_bounds}")
    print(f"Segmentation bounds: {seg_bounds}")

    # Load segmentation volume (small, 1792x1792x16)
    print("Loading segmentation volume...")
    seg_volume = cv_seg[seg_bounds[0][0]:seg_bounds[1][0],
                        seg_bounds[0][1]:seg_bounds[1][1],
                        seg_bounds[0][2]:seg_bounds[1][2]]
    seg_volume = np.array(seg_volume)
    print(f"Segmentation volume shape: {seg_volume.shape}")

    # Load corresponding region from image volume
    print("Loading corresponding image volume region...")
    img_volume = cv_img[seg_bounds[0][0]:seg_bounds[1][0],
                        seg_bounds[0][1]:seg_bounds[1][1],
                        seg_bounds[0][2]:seg_bounds[1][2]]
    img_volume = np.array(img_volume)
    print(f"Image volume shape: {img_volume.shape}")

    # Create datasets
    train_ds = MultiClassDataset(
        img_volume, seg_volume, img_bounds, seg_bounds,
        split="train", patch_size=cfg.patch_size, num_samples=cfg.train_samples
    )
    val_ds = MultiClassDataset(
        img_volume, seg_volume, img_bounds, seg_bounds,
        split="val", patch_size=cfg.patch_size, num_samples=cfg.val_samples
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, num_workers=4)

    # Create model
    print("\nInitializing model...")

    # Check if resuming from checkpoint
    start_epoch = 0
    best_mean_iou = 0.0

    if cfg.get('resume_checkpoint') is not None:
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint)

        # Create model (without loading pretrained weights, we'll load from checkpoint instead)
        model = ConvNeXtUNet(
            num_classes=cfg.num_classes,
            num_coarse_groups=cfg.num_coarse_groups,
            pretrained_path=None,
            freeze_encoder=cfg.freeze_encoder
        ).to("cuda")

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")

        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        best_mean_iou = checkpoint.get('val_mean_iou', 0.0)
        print(f"Continuing from epoch {start_epoch}, best mIoU so far: {best_mean_iou:.4f}")

    else:
        # Create model with pretrained encoder
        model = ConvNeXtUNet(
            num_classes=cfg.num_classes,
            num_coarse_groups=cfg.num_coarse_groups,
            pretrained_path=cfg.pretrained_checkpoint,
            freeze_encoder=cfg.freeze_encoder
        ).to("cuda")

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=1e-4
    )

    # Load optimizer state if resuming
    if cfg.get('resume_checkpoint') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Create checkpoint directory
    checkpoint_dir = "checkpoints_multi_seg"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_loss_fine = 0.0
        epoch_loss_coarse = 0.0

        for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + cfg.epochs}"):
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                logits_fine, logits_coarse = model(images)
                loss, loss_fine, loss_coarse = compute_losses(
                    logits_fine, logits_coarse, labels, lambda_coarse=cfg.lambda_coarse
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_loss_fine += loss_fine.item()
            epoch_loss_coarse += loss_coarse.item()

            wandb.log({
                "train/loss": loss.item(),
                "train/loss_fine": loss_fine.item(),
                "train/loss_coarse": loss_coarse.item()
            })

        # Log epoch averages
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_loss_fine = epoch_loss_fine / num_batches
        avg_loss_coarse = epoch_loss_coarse / num_batches

        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": avg_loss,
            "epoch/train_loss_fine": avg_loss_fine,
            "epoch/train_loss_coarse": avg_loss_coarse
        })

        # Validation
        if epoch % cfg.eval_every == 0 or epoch == start_epoch + cfg.epochs - 1:
            model.eval()
            val_loss = 0.0
            val_loss_fine = 0.0
            val_loss_coarse = 0.0
            val_metrics_sum = {'accuracy': 0.0, 'mean_iou': 0.0, 'per_class_iou': [0.0] * cfg.num_classes}

            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to("cuda", non_blocking=True)
                    labels = labels.to("cuda", non_blocking=True)

                    with autocast(dtype=torch.bfloat16):
                        logits_fine, logits_coarse = model(images)
                        loss, loss_fine, loss_coarse = compute_losses(
                            logits_fine, logits_coarse, labels, lambda_coarse=cfg.lambda_coarse
                        )

                        metrics = compute_metrics(logits_fine, labels, num_classes=cfg.num_classes)

                        val_loss += loss.item()
                        val_loss_fine += loss_fine.item()
                        val_loss_coarse += loss_coarse.item()
                        val_metrics_sum['accuracy'] += metrics['accuracy']
                        val_metrics_sum['mean_iou'] += metrics['mean_iou']
                        for c in range(cfg.num_classes):
                            if not np.isnan(metrics['per_class_iou'][c]):
                                val_metrics_sum['per_class_iou'][c] += metrics['per_class_iou'][c]

            num_val_batches = len(val_loader)
            avg_val_loss = val_loss / num_val_batches
            avg_val_loss_fine = val_loss_fine / num_val_batches
            avg_val_loss_coarse = val_loss_coarse / num_val_batches
            avg_val_accuracy = val_metrics_sum['accuracy'] / num_val_batches
            avg_val_mean_iou = val_metrics_sum['mean_iou'] / num_val_batches

            # Log validation metrics
            val_log = {
                "val/loss": avg_val_loss,
                "val/loss_fine": avg_val_loss_fine,
                "val/loss_coarse": avg_val_loss_coarse,
                "val/accuracy": avg_val_accuracy,
                "val/mean_iou": avg_val_mean_iou
            }

            # Log per-class IoU
            for c, name in enumerate(CLASS_NAMES):
                avg_iou_c = val_metrics_sum['per_class_iou'][c] / num_val_batches
                if not np.isnan(avg_iou_c):
                    val_log[f"val/iou_{name}"] = avg_iou_c

            wandb.log(val_log)

            print(f"Epoch {epoch}: Val Loss={avg_val_loss:.4f}, Acc={avg_val_accuracy:.4f}, mIoU={avg_val_mean_iou:.4f}")

            # Save best model (based on mean IoU)
            if avg_val_mean_iou > best_mean_iou:
                best_mean_iou = avg_val_mean_iou
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_miou{best_mean_iou:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mean_iou': avg_val_mean_iou,
                    'val_accuracy': avg_val_accuracy,
                    'config': dict(cfg),
                    'wandb_run_id': wandb.run.id
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint periodically
        if epoch % cfg.save_every == 0 and epoch > 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch{epoch:04d}_miou{avg_val_mean_iou:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mean_iou': avg_val_mean_iou,
                'config': dict(cfg),
                'wandb_run_id': wandb.run.id
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Visualize predictions
        if epoch % cfg.visualize_every == 0:
            save_predictions(val_loader, model, epoch)

    wandb.finish()


if __name__ == "__main__":
    main()
