"""
Utility functions for training: loss functions, metrics, visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss

    BCE handles pixel-wise classification
    Dice handles class imbalance and overlap
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, return_components=False):
        """
        Args:
            logits: (B, 2, H, W) raw predictions
            targets: (B, 2, H, W) ground truth masks (0 or 1)
            return_components: If True, return dict with individual components

        Returns:
            If return_components=False: Combined loss (scalar)
            If return_components=True: dict with {
                'loss': total loss,
                'bce_nuclei': BCE loss for nuclei channel,
                'bce_mito': BCE loss for mito channel,
                'dice_nuclei': Dice loss for nuclei channel,
                'dice_mito': Dice loss for mito channel
            }
        """
        # BCE loss - per channel
        bce_nuclei = self.bce(logits[:, 0], targets[:, 0])
        bce_mito = self.bce(logits[:, 1], targets[:, 1])
        bce_loss = (bce_nuclei + bce_mito) / 2.0

        # Dice loss - per channel
        dice_nuclei, dice_mito = self.dice_loss_per_channel(logits, targets)
        dice_loss = (dice_nuclei + dice_mito) / 2.0

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        if return_components:
            return {
                'loss': total_loss,
                'bce_nuclei': bce_nuclei.item(),
                'bce_mito': bce_mito.item(),
                'dice_nuclei': dice_nuclei.item(),
                'dice_mito': dice_mito.item()
            }
        else:
            return total_loss

    def dice_loss_per_channel(self, logits, targets, smooth=1.0):
        """
        Dice loss per channel

        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, C, H, W) ground truth masks
            smooth: Smoothing factor to avoid division by zero

        Returns:
            (dice_nuclei, dice_mito): Tuple of dice losses for each channel
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (B, C, H*W)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # (B, C, H*W)

        # Compute intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        # Dice coefficient per channel
        dice = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)

        # Average over batch for each channel, then convert to loss
        dice_nuclei = 1.0 - dice[:, 0].mean()
        dice_mito = 1.0 - dice[:, 1].mean()

        return dice_nuclei, dice_mito

    def dice_loss(self, logits, targets, smooth=1.0):
        """
        Dice loss = 1 - Dice coefficient

        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, C, H, W) ground truth masks
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss (scalar)
        """
        dice_nuclei, dice_mito = self.dice_loss_per_channel(logits, targets, smooth)
        return (dice_nuclei + dice_mito) / 2.0


def compute_metrics(logits, targets, threshold=0.5):
    """
    Compute evaluation metrics

    Args:
        logits: (B, 2, H, W) raw predictions
        targets: (B, 2, H, W) ground truth masks

    Returns:
        dict: Dictionary of metrics per class
    """
    # Apply sigmoid and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    metrics = {}

    # Compute per-channel metrics
    for c, name in enumerate(['nuclei', 'mito']):
        pred_c = preds[:, c].flatten()
        target_c = targets[:, c].flatten()

        # True positives, false positives, false negatives, true negatives
        tp = (pred_c * target_c).sum().item()
        fp = (pred_c * (1 - target_c)).sum().item()
        fn = ((1 - pred_c) * target_c).sum().item()
        tn = ((1 - pred_c) * (1 - target_c)).sum().item()

        # Pixel accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # IoU (Jaccard index)
        iou = tp / (tp + fp + fn + 1e-8)

        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        metrics[f'{name}_accuracy'] = accuracy
        metrics[f'{name}_precision'] = precision
        metrics[f'{name}_recall'] = recall
        metrics[f'{name}_f1'] = f1
        metrics[f'{name}_iou'] = iou
        metrics[f'{name}_dice'] = dice

    # Overall metrics (average across classes)
    metrics['mean_accuracy'] = np.mean([metrics['nuclei_accuracy'], metrics['mito_accuracy']])
    metrics['mean_iou'] = np.mean([metrics['nuclei_iou'], metrics['mito_iou']])
    metrics['mean_dice'] = np.mean([metrics['nuclei_dice'], metrics['mito_dice']])

    return metrics


def visualize_prediction(img, target, pred, save_path=None):
    """
    Create visualization of input, ground truth, and prediction

    Args:
        img: (1, H, W) or (H, W) input image tensor, range [0, 1]
        target: (2, H, W) ground truth masks
        pred: (2, H, W) prediction logits
        save_path: Optional path to save visualization

    Returns:
        PIL Image
    """
    # Convert to numpy
    if img.dim() == 3:
        img = img[0]  # Remove channel dim
    img_np = (img.cpu().numpy() * 255).astype(np.uint8)

    target_nuclei = target[0].cpu().numpy()
    target_mito = target[1].cpu().numpy()

    pred_probs = torch.sigmoid(pred)
    pred_nuclei = (pred_probs[0] > 0.5).cpu().numpy()
    pred_mito = (pred_probs[1] > 0.5).cpu().numpy()

    # Create RGB composites
    def composite_rgb(img, nuclei, mito):
        """Composite grayscale + red nuclei + cyan mito"""
        rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
        # Nuclei = red tint (boost red, dim others)
        nuclei_mask = nuclei > 0
        rgb[nuclei_mask, 0] = np.clip(rgb[nuclei_mask, 0] + 150, 0, 255)
        rgb[nuclei_mask, 1] = rgb[nuclei_mask, 1] * 0.5
        rgb[nuclei_mask, 2] = rgb[nuclei_mask, 2] * 0.5

        # Mito = cyan tint (boost green+blue, dim red)
        mito_mask = mito > 0
        rgb[mito_mask, 0] = rgb[mito_mask, 0] * 0.5
        rgb[mito_mask, 1] = np.clip(rgb[mito_mask, 1] + 150, 0, 255)
        rgb[mito_mask, 2] = np.clip(rgb[mito_mask, 2] + 150, 0, 255)

        return rgb.astype(np.uint8)

    # Create composites
    img_only = np.stack([img_np, img_np, img_np], axis=-1)
    gt_composite = composite_rgb(img_np, target_nuclei * 255, target_mito * 255)
    pred_composite = composite_rgb(img_np, pred_nuclei * 255, pred_mito * 255)

    # Debug: print prediction statistics
    #print(f"Prediction stats: nuclei={pred_nuclei.sum()} pixels, mito={pred_mito.sum()} pixels")

    # Stack horizontally: input | ground truth | prediction
    vis = np.concatenate([img_only, gt_composite, pred_composite], axis=1)

    pil_img = Image.fromarray(vis)

    if save_path:
        pil_img.save(save_path)

    return pil_img


if __name__ == "__main__":
    # Test loss and metrics
    print("Testing loss and metrics...")

    # Create dummy data
    batch_size = 2
    logits = torch.randn(batch_size, 2, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 2, 256, 256)).float()

    # Test loss
    criterion = CombinedLoss()
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")

    # Test metrics
    metrics = compute_metrics(logits, targets)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test visualization
    print("\nTesting visualization...")
    img = torch.rand(1, 256, 256)
    target = targets[0]
    pred = logits[0]
    vis_img = visualize_prediction(img, target, pred, save_path="test_vis.png")
    print(f"Visualization saved to test_vis.png (size: {vis_img.size})")

    print("\nUtils test passed!")
