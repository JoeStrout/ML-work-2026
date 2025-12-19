"""
Training script for mitochondria binary segmentation using ConvNeXt U-Net.
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

from unet_architecture import ConvNeXtUNet


class MitochondriaDataset(Dataset):
    """Dataset for mitochondria binary segmentation."""

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
        Returns a random patch and its binary mitochondria mask.

        Returns:
            image: Tensor of shape (1, patch_size, patch_size)
            mask: Tensor of shape (1, patch_size, patch_size) with binary labels
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

        # Create binary mask for mitochondria (label 10)
        binary_mask = (seg_patch == 10).astype(np.float32)

        # Convert to tensors and normalize image
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()  # (1, H, W)

        # Normalize image to [0, 1]
        img_tensor = img_tensor / 255.0

        return img_tensor, mask_tensor

    def __len__(self):
        return self.num_samples


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient for binary segmentation.

    Args:
        pred: Predicted probabilities (after sigmoid), shape (N, 1, H, W)
        target: Ground truth binary masks, shape (N, 1, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient (scalar)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute IoU (Intersection over Union) for binary segmentation.

    Args:
        pred: Predicted probabilities (after sigmoid), shape (N, 1, H, W)
        target: Ground truth binary masks, shape (N, 1, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        IoU score (scalar)
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


def save_predictions(dataloader, model, epoch, output_dir="seg_predictions", num_samples=4):
    """
    Save visualization of predictions vs ground truth.

    Args:
        dataloader: DataLoader to sample from
        model: Trained model
        epoch: Current epoch number
        output_dir: Directory to save images
        num_samples: Number of samples to visualize
    """
    matplotlib.use('Agg')
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to("cuda")
            masks = masks.to("cuda")

            # Get predictions
            logits = model(images)
            preds = torch.sigmoid(logits)

            # Move to CPU for visualization
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            preds = preds.cpu().numpy()

            # Take first num_samples from batch
            num_samples = min(num_samples, images.shape[0])

            # Create figure: num_samples rows x 4 columns
            # Columns: [Image, Ground Truth, Prediction, Overlay]
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                img = images[i, 0]  # (H, W)
                mask = masks[i, 0]  # (H, W)
                pred = preds[i, 0]  # (H, W)

                # Column 0: Image
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title('EM Image' if i == 0 else '')
                axes[i, 0].axis('off')

                # Column 1: Ground truth mask
                axes[i, 1].imshow(img, cmap='gray', alpha=0.5)
                axes[i, 1].imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                axes[i, 1].set_title('Ground Truth' if i == 0 else '')
                axes[i, 1].axis('off')

                # Column 2: Prediction
                axes[i, 2].imshow(img, cmap='gray', alpha=0.5)
                axes[i, 2].imshow(pred, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
                axes[i, 2].set_title('Prediction' if i == 0 else '')
                axes[i, 2].axis('off')

                # Column 3: Overlay comparison
                # Show ground truth in red, prediction in blue
                axes[i, 3].imshow(img, cmap='gray', alpha=0.3)
                axes[i, 3].imshow(mask, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
                axes[i, 3].imshow(pred, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
                axes[i, 3].set_title('Overlay (GT=Red, Pred=Blue)' if i == 0 else '')
                axes[i, 3].axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved predictions to {save_path}")
            break  # Only need one batch


@hydra.main(version_base=None, config_path=".", config_name="seg_config")
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
        wandb.init(project="MitoSeg_Dec2025", id=wandb_run_id, resume="allow", config=dict(cfg))
    else:
        wandb.init(project="MitoSeg_Dec2025", config=dict(cfg))

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
    train_ds = MitochondriaDataset(
        img_volume, seg_volume, img_bounds, seg_bounds,
        split="train", patch_size=cfg.patch_size, num_samples=cfg.train_samples
    )
    val_ds = MitochondriaDataset(
        img_volume, seg_volume, img_bounds, seg_bounds,
        split="val", patch_size=cfg.patch_size, num_samples=cfg.val_samples
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, num_workers=4)

    # Create model
    print("\nInitializing model...")

    # Check if resuming from checkpoint
    start_epoch = 0
    best_dice = 0.0

    if cfg.get('resume_checkpoint') is not None:
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint)

        # Create model (without loading pretrained weights, we'll load from checkpoint instead)
        model = ConvNeXtUNet(
            pretrained_path=None,
            freeze_encoder=cfg.freeze_encoder
        ).to("cuda")

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")

        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('val_dice', 0.0)
        print(f"Continuing from epoch {start_epoch}, best Dice so far: {best_dice:.4f}")

    else:
        # Create model with pretrained encoder
        model = ConvNeXtUNet(
            pretrained_path=cfg.pretrained_checkpoint,
            freeze_encoder=cfg.freeze_encoder
        ).to("cuda")

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function: Binary Cross Entropy with positive class weighting
    # pos_weight compensates for class imbalance
    pos_weight = torch.tensor([cfg.pos_weight]).to("cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    checkpoint_dir = "checkpoints_seg"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        for images, masks in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + cfg.epochs}"):
            images = images.to("cuda", non_blocking=True)
            masks = masks.to("cuda", non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, masks)

                # Compute Dice for monitoring (use sigmoid)
                preds = torch.sigmoid(logits)
                dice = dice_coefficient(preds, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_dice += dice.item()

            wandb.log({
                "train/loss": loss.item(),
                "train/dice": dice.item()
            })

        # Log epoch averages
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches

        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": avg_loss,
            "epoch/train_dice": avg_dice
        })

        # Validation
        if epoch % cfg.eval_every == 0 or epoch == start_epoch + cfg.epochs - 1:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0

            with torch.inference_mode():
                for images, masks in val_loader:
                    images = images.to("cuda", non_blocking=True)
                    masks = masks.to("cuda", non_blocking=True)

                    with autocast(dtype=torch.bfloat16):
                        logits = model(images)
                        loss = criterion(logits, masks)
                        preds = torch.sigmoid(logits)

                        dice = dice_coefficient(preds, masks)
                        iou = iou_score(preds, masks)

                        val_loss += loss.item()
                        val_dice += dice.item()
                        val_iou += iou.item()

            num_val_batches = len(val_loader)
            avg_val_loss = val_loss / num_val_batches
            avg_val_dice = val_dice / num_val_batches
            avg_val_iou = val_iou / num_val_batches

            wandb.log({
                "val/loss": avg_val_loss,
                "val/dice": avg_val_dice,
                "val/iou": avg_val_iou
            })

            print(f"Epoch {epoch}: Val Loss={avg_val_loss:.4f}, Dice={avg_val_dice:.4f}, IoU={avg_val_iou:.4f}")

            # Save best model
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_dice{best_dice:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': avg_val_dice,
                    'val_iou': avg_val_iou,
                    'config': dict(cfg),
                    'wandb_run_id': wandb.run.id
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint periodically
        if epoch % cfg.save_every == 0 and epoch > 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch{epoch:04d}_dice{avg_val_dice:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
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
