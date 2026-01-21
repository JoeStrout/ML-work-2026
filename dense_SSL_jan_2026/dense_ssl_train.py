# Dense SSL Training Script for EM Images
# Per-pixel self-supervised learning using cross-slice prediction

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import timm
import wandb
import hydra
import tqdm
from omegaconf import DictConfig
import numpy as np
import random
from PIL import Image


# =============================================================================
# Geometric transform utilities for equivariance loss
# =============================================================================

def apply_geometric_transform(x, rot_k, flip_h, flip_v):
    """Apply rotation and flips to a tensor.

    Args:
        x: (B, C, H, W) tensor
        rot_k: number of 90° rotations (0, 1, 2, or 3)
        flip_h: whether to flip horizontally
        flip_v: whether to flip vertically

    Returns:
        Transformed tensor
    """
    # Apply rotation (dims -2, -1 are H, W)
    if rot_k > 0:
        x = torch.rot90(x, k=rot_k, dims=(-2, -1))
    # Apply flips
    if flip_h:
        x = torch.flip(x, dims=(-1,))
    if flip_v:
        x = torch.flip(x, dims=(-2,))
    return x


def apply_inverse_transform(x, rot_k, flip_h, flip_v):
    """Apply inverse of the geometric transform.

    Inverse order: undo flips first (they're self-inverse), then undo rotation.
    """
    # Undo flips (same operations, self-inverse)
    if flip_v:
        x = torch.flip(x, dims=(-2,))
    if flip_h:
        x = torch.flip(x, dims=(-1,))
    # Undo rotation (rotate by -k, which is same as rotate by 4-k)
    if rot_k > 0:
        x = torch.rot90(x, k=-rot_k, dims=(-2, -1))
    return x


def sample_random_transform():
    """Sample a random geometric transform."""
    rot_k = random.randint(0, 3)
    flip_h = random.random() > 0.5
    flip_v = random.random() > 0.5
    return rot_k, flip_h, flip_v


# =============================================================================
# Visualization utilities
# =============================================================================

def create_colormap(K):
    """Create a colormap with K distinct colors using HSV spacing."""
    colors = np.zeros((K, 3), dtype=np.uint8)
    for i in range(K):
        # Use HSV with evenly spaced hues
        hue = i / K
        # Convert HSV to RGB (saturation=0.8, value=0.9 for nice colors)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def save_class_visualization(encoder, dataset, device, epoch, output_dir, K, num_samples=4):
    """Save visualization of encoder outputs as colored class maps.

    For each sample, saves:
    - Original grayscale image
    - Colored class map (argmax over K classes)
    - Side-by-side comparison

    Args:
        encoder: The DenseEncoder model
        dataset: EMSliceDataset to sample from
        device: torch device
        epoch: Current epoch number (for filename)
        output_dir: Directory to save images
        K: Number of classes
        num_samples: Number of samples to visualize
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    encoder.eval()
    colormap = create_colormap(K)

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Get a sample (use fixed indices for consistency across epochs)
            np.random.seed(sample_idx * 12345)
            slices, z_idx = dataset[sample_idx]
            slices = slices.unsqueeze(0).to(device)  # (1, 5, 1, H, W)

            # Encode middle slice
            middle_slice = slices[:, 2]  # (1, 1, H, W)
            enc_output = encoder(middle_slice)  # (1, K, H', W')

            # Get class assignments (argmax)
            class_map = enc_output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H', W')

            # Create colored image
            H, W = class_map.shape
            colored = np.zeros((H, W, 3), dtype=np.uint8)
            for k in range(K):
                mask = class_map == k
                colored[mask] = colormap[k]

            # Get original image (middle slice, trimmed to match encoder output)
            orig = middle_slice.squeeze().cpu().numpy()  # (H_orig, W_orig)
            border_trim = encoder.border_trim
            if border_trim > 0:
                orig = orig[border_trim:-border_trim, border_trim:-border_trim]
            orig_uint8 = (orig * 255).astype(np.uint8)

            # Save images
            # 1. Original grayscale
            orig_img = Image.fromarray(orig_uint8, mode='L')
            orig_img.save(os.path.join(output_dir, f"epoch{epoch:03d}_sample{sample_idx}_orig.png"))

            # 2. Class map
            class_img = Image.fromarray(colored, mode='RGB')
            class_img.save(os.path.join(output_dir, f"epoch{epoch:03d}_sample{sample_idx}_classes.png"))

            # 3. Side-by-side comparison
            orig_rgb = np.stack([orig_uint8] * 3, axis=-1)  # Convert grayscale to RGB
            side_by_side = np.concatenate([orig_rgb, colored], axis=1)
            combined_img = Image.fromarray(side_by_side, mode='RGB')
            combined_img.save(os.path.join(output_dir, f"epoch{epoch:03d}_sample{sample_idx}_combined.png"))

    encoder.train()
    print(f"Saved {num_samples} visualization samples to {output_dir}/")

    # Also log to wandb
    wandb_images = []
    for sample_idx in range(num_samples):
        combined_path = os.path.join(output_dir, f"epoch{epoch:03d}_sample{sample_idx}_combined.png")
        if os.path.exists(combined_path):
            wandb_images.append(wandb.Image(combined_path, caption=f"Sample {sample_idx}"))
    if wandb_images:
        wandb.log({"visualizations": wandb_images, "epoch": epoch})


# =============================================================================
# SIGReg - Adapted for dense outputs
# =============================================================================

class SIGReg(nn.Module):
    """Characteristic function-based regularization to prevent collapse.

    Adapted from LeJEPA. Applied to spatial-mean of dense encoder outputs.
    NOTE: May not be ideal for softmax outputs - see EntropyReg for alternative.
    """
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
        """
        Args:
            proj: (B, K) tensor of spatial-mean encoder outputs
        Returns:
            scalar regularization loss
        """
        device = proj.device
        A = torch.randn(proj.size(-1), 256, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t.to(device)
        err = (x_t.cos().mean(0) - self.phi.to(device)).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights.to(device)) * proj.size(0)
        return statistic.mean()


class DiversityReg(nn.Module):
    """InfoMax-style regularization for dense softmax outputs.

    Based on the approach in regularization.md:
    - L_marg: max(0, H* - H(p̄)) - penalize only if marginal entropy is too low
    - L_cond: (1/N) Σ_i H(p_i) - encourage sharp per-pixel predictions

    This prevents collapse without forcing equipartition, and allows:
    - Highly skewed class distributions
    - Classes to be absent in some images
    - Sharp, segmentation-like outputs
    """
    def __init__(self, H_star=None, min_classes=8, alpha=1.0, beta=0.1):
        """
        Args:
            H_star: Target minimum marginal entropy. If None, computed from min_classes.
            min_classes: Effective number of classes for H* (default 8 → H*≈2.08)
            alpha: Weight for marginal entropy term (anti-collapse)
            beta: Weight for conditional entropy term (sharpness)
        """
        super().__init__()
        self.H_star = H_star if H_star is not None else np.log(min_classes)
        self.alpha = alpha
        self.beta = beta

    def forward(self, probs):
        """
        Args:
            probs: (B, K, H, W) softmax probabilities from encoder
        Returns:
            tuple: (total_loss, marginal_loss, conditional_loss) for logging
        """
        B, K, H, W = probs.shape
        N = H * W
        eps = 1e-8

        # Marginal distribution: average class probabilities across all pixels per image
        p_bar = probs.mean(dim=(2, 3))  # (B, K)

        # Marginal entropy per image
        H_marg = -(p_bar * (p_bar + eps).log()).sum(dim=1)  # (B,)

        # L_marg: penalize only if entropy is below threshold
        L_marg = F.relu(self.H_star - H_marg).mean()

        # Conditional entropy: average entropy per pixel
        H_pixel = -(probs * (probs + eps).log()).sum(dim=1)  # (B, H, W)
        L_cond = H_pixel.mean()

        # Combined loss
        total = self.alpha * L_marg + self.beta * L_cond

        return total, L_marg, L_cond


# =============================================================================
# DenseEncoder - ConvNeXt backbone with FPN-style dense head
# =============================================================================

class DenseEncoder(nn.Module):
    """Encoder that produces K-dimensional softmax output per pixel.

    Uses ConvNeXt backbone with FPN-style decoder for dense prediction.
    """
    def __init__(self, K=50, backbone="convnext_small", border_trim=32):
        super().__init__()
        self.K = K
        self.border_trim = border_trim

        # Create ConvNeXt backbone with feature extraction
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            in_chans=1,  # grayscale input
        )

        # Get feature dimensions from backbone
        # ConvNeXt-small stages output: [96, 192, 384, 768] channels
        # at 1/4, 1/8, 1/16, 1/32 resolution
        feat_dims = self.backbone.feature_info.channels()

        # FPN-style decoder: upsample and fuse features
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, 128, 1) for dim in feat_dims
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(128, 128, 3, padding=1) for _ in feat_dims
        ])

        # Final head: fuse all scales and predict K classes
        self.fuse_conv = nn.Conv2d(128 * len(feat_dims), 256, 3, padding=1)
        self.head = nn.Conv2d(256, K, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) grayscale image
        Returns:
            (B, K, H', W') softmax probabilities per pixel (trimmed borders)
        """
        B, C, H, W = x.shape

        # Extract multi-scale features
        features = self.backbone(x)  # list of feature maps

        # FPN: lateral connections and top-down pathway
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='bilinear', align_corners=False
            )

        # Apply FPN convs and upsample all to same size
        target_size = (H // 4, W // 4)  # 1/4 resolution
        fpn_outs = []
        for conv, lat in zip(self.fpn_convs, laterals):
            out = conv(lat)
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            fpn_outs.append(out)

        # Concatenate and fuse
        fused = torch.cat(fpn_outs, dim=1)
        fused = F.relu(self.fuse_conv(fused))

        # Predict K classes
        logits = self.head(fused)

        # Upsample to full resolution
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        # Trim borders
        t = self.border_trim
        if t > 0:
            logits = logits[:, :, t:-t, t:-t]

        # Softmax over classes
        probs = F.softmax(logits, dim=1)

        return probs


# =============================================================================
# SlicePredictor - Predicts middle slice from 4 context slices
# =============================================================================

class SlicePredictor(nn.Module):
    """Small network to predict middle slice encoding from 4 context slices.

    Uses a shallow ConvNet with dilated convolutions.
    """
    def __init__(self, K=50, hidden_channels=128):
        super().__init__()
        self.K = K

        # Input: 4*K channels (4 context slices stacked)
        self.net = nn.Sequential(
            nn.Conv2d(4 * K, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, K, 1),
        )

    def forward(self, context):
        """
        Args:
            context: (B, 4*K, H, W) - stacked encoder outputs from 4 context slices
        Returns:
            (B, K, H, W) - predicted softmax probabilities for middle slice
        """
        logits = self.net(context)
        return F.softmax(logits, dim=1)


# =============================================================================
# EMSliceDataset - Load 5 consecutive Z-slices from EM volume
# =============================================================================

class EMSliceDataset(Dataset):
    """Dataset that samples 5 consecutive Z-slices from an EM volume.

    Loads data via CloudVolume and samples random XY patches.
    """
    def __init__(self, data_path, patch_size=256, num_samples=10000,
                 z_margin=2, mip_level=0):
        """
        Args:
            data_path: CloudVolume path (gs://...)
            patch_size: Size of XY patch to sample
            num_samples: Number of samples per epoch
            z_margin: Number of slices before/after target (default 2 for 5-slice window)
            mip_level: Resolution level (0 = highest)
        """
        from cloudvolume import CloudVolume

        self.patch_size = patch_size
        self.num_samples = num_samples
        self.z_margin = z_margin

        # Load volume
        print(f"Loading CloudVolume from {data_path}...")
        cv = CloudVolume(data_path, mip=mip_level, fill_missing=True)

        # Get volume dimensions
        self.vol_shape = cv.shape[:3]  # (X, Y, Z)
        print(f"Volume shape: {self.vol_shape}")

        # Load entire volume into memory for fast access
        print("Loading volume into memory...")
        self.volume = cv[:][:, :, :, 0]  # Remove channel dim, shape (X, Y, Z)
        self.volume = np.transpose(self.volume, (2, 1, 0))  # (Z, Y, X)
        print(f"Loaded volume with shape {self.volume.shape}")

        # Valid ranges for sampling
        self.z_range = (z_margin, self.volume.shape[0] - z_margin - 1)
        self.y_range = (0, self.volume.shape[1] - patch_size)
        self.x_range = (0, self.volume.shape[2] - patch_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Sample 5 consecutive Z-slices with random XY crop.

        Returns:
            slices: (5, 1, H, W) tensor of grayscale patches
            z_idx: central Z index
        """
        # Random Z position (valid range)
        z = np.random.randint(self.z_range[0], self.z_range[1] + 1)

        # Random XY position
        y = np.random.randint(self.y_range[0], self.y_range[1] + 1)
        x = np.random.randint(self.x_range[0], self.x_range[1] + 1)

        # Extract 5 consecutive slices
        z_indices = [z - 2, z - 1, z, z + 1, z + 2]
        slices = []
        for zi in z_indices:
            patch = self.volume[zi, y:y+self.patch_size, x:x+self.patch_size]
            slices.append(patch)

        # Stack and convert to tensor
        slices = np.stack(slices, axis=0)  # (5, H, W)
        slices = torch.from_numpy(slices).float() / 255.0  # normalize to [0, 1]
        slices = slices.unsqueeze(1)  # (5, 1, H, W)

        return slices, z


# =============================================================================
# Training Loop
# =============================================================================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()
    wandb.init(project=cfg.project, config=dict(cfg))
    torch.manual_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("Creating dataset...")
    train_ds = EMSliceDataset(
        data_path=cfg.data_path,
        patch_size=cfg.patch_size,
        num_samples=cfg.num_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Create models
    print("Creating models...")
    # Online encoder (updated by gradient descent)
    encoder = DenseEncoder(
        K=cfg.K,
        backbone=cfg.encoder_backbone,
        border_trim=cfg.get("border_trim", 32),
    ).to(device)

    # Target encoder (updated by EMA of online encoder)
    import copy
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False  # No gradients for target encoder

    # EMA momentum (higher = slower update)
    ema_tau = cfg.get("ema_tau", 0.996)

    predictor = SlicePredictor(
        K=cfg.K,
        hidden_channels=cfg.predictor_channels,
    ).to(device)

    # Regularization: DiversityReg prevents collapse without forcing equipartition
    regularizer = DiversityReg(
        min_classes=cfg.get("min_classes", 8),  # H* = log(8) ≈ 2.08
        alpha=cfg.get("alpha", 1.0),   # Weight for marginal entropy (anti-collapse)
        beta=cfg.get("beta", 0.1),     # Weight for conditional entropy (sharpness)
    ).to(device)

    # Optimizer
    params = [
        {"params": encoder.parameters(), "lr": cfg.lr, "weight_decay": cfg.get("weight_decay", 1e-4)},
        {"params": predictor.parameters(), "lr": cfg.lr, "weight_decay": cfg.get("weight_decay", 1e-4)},
    ]
    opt = torch.optim.AdamW(params)

    # Learning rate scheduler
    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=cfg.lr * 0.01)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    # Mixed precision
    scaler = GradScaler(enabled=(device == "cuda"))

    # Training
    print(f"Starting training for {cfg.epochs} epochs...")
    for epoch in range(cfg.epochs):
        encoder.train()
        target_encoder.train()  # Keep in train mode for BatchNorm stats consistency
        predictor.train()

        epoch_pred_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_L_marg = 0.0
        epoch_L_cond = 0.0
        epoch_mean_var = 0.0
        epoch_L_equiv = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (slices, z_idx) in enumerate(pbar):
            slices = slices.to(device, non_blocking=True)  # (B, 5, 1, H, W)
            B = slices.shape[0]

            with autocast('cuda', dtype=torch.bfloat16):
                # Encode context slices (0, 1, 3, 4) with online encoder
                context_outputs = []
                for i in [0, 1, 3, 4]:
                    enc_out = encoder(slices[:, i])  # (B, K, H', W')
                    context_outputs.append(enc_out)

                # Encode target slice (2) with target encoder (no gradients)
                with torch.no_grad():
                    target = target_encoder(slices[:, 2])  # (B, K, H', W')

                # Also encode middle slice with online encoder for regularization
                enc_middle = encoder(slices[:, 2])

                # All encoder outputs for regularization (online encoder only)
                enc_outputs = [context_outputs[0], context_outputs[1], enc_middle,
                               context_outputs[2], context_outputs[3]]

                # Context: slices 0, 1, 3, 4 from online encoder
                context = torch.cat(context_outputs, dim=1)  # (B, 4*K, H', W')

                # Predict middle slice
                prediction = predictor(context)  # (B, K, H', W')

                # Prediction loss (MSE)
                pred_loss = F.mse_loss(prediction, target)

                # Regularization loss (DiversityReg on all encoder outputs)
                # Stack all encoder outputs and compute regularization
                all_enc = torch.stack(enc_outputs, dim=1)  # (B, 5, K, H', W')
                all_enc_flat = all_enc.reshape(B * 5, cfg.K, *all_enc.shape[-2:])  # (B*5, K, H', W')
                reg_loss, L_marg, L_cond = regularizer(all_enc_flat)

                # Cross-sample variance loss: encourage different samples to have
                # different encodings at each spatial position
                # This prevents position-dependent solutions (e.g., horizontal bands)
                var_across_samples = enc_middle.var(dim=0)  # (K, H', W')
                mean_var = var_across_samples.mean()
                L_cross = -mean_var  # Maximize variance → minimize negative

                # Geometric equivariance loss: encoding should depend on image content,
                # not position. If we transform the input and inverse-transform the output,
                # we should get the same result as encoding the original.
                rot_k, flip_h, flip_v = sample_random_transform()

                # Transform the middle slice input
                middle_slice_transformed = apply_geometric_transform(
                    slices[:, 2], rot_k, flip_h, flip_v
                )

                # Encode the transformed input
                enc_transformed = encoder(middle_slice_transformed)

                # Apply inverse transform to the encoding
                enc_transformed_inv = apply_inverse_transform(
                    enc_transformed, rot_k, flip_h, flip_v
                )

                # Compare with original encoding - should match if content-dependent
                L_equiv = F.mse_loss(enc_middle, enc_transformed_inv)

                # Total loss
                gamma = cfg.get("gamma", 1.0)    # Weight for cross-sample variance
                delta = cfg.get("delta", 1.0)    # Weight for equivariance loss
                loss = pred_loss + reg_loss + gamma * L_cross + delta * L_equiv

            # Backward pass
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # EMA update of target encoder
            with torch.no_grad():
                for p_online, p_target in zip(encoder.parameters(), target_encoder.parameters()):
                    p_target.data = ema_tau * p_target.data + (1 - ema_tau) * p_online.data

            # Logging
            epoch_pred_loss += pred_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_L_marg += L_marg.item()
            epoch_L_cond += L_cond.item()
            epoch_mean_var += mean_var.item()
            epoch_L_equiv += L_equiv.item()

            # Compute class usage statistics
            with torch.no_grad():
                enc_mid = enc_middle  # (B, K, H', W') from online encoder

                # Class histogram from middle slice encoder output
                class_probs = enc_mid.mean(dim=(0, 2, 3))  # (K,)
                class_entropy = -(class_probs * (class_probs + 1e-8).log()).sum().item()
                num_active_classes = (class_probs > 0.01).sum().item()

                # NEW: Check actual distribution sharpness
                max_prob_per_pixel = enc_mid.max(dim=1).values  # (B, H', W')
                avg_max_prob = max_prob_per_pixel.mean().item()  # How confident are predictions?

                # NEW: Check diversity of argmax classes
                class_assignments = enc_mid.argmax(dim=1)  # (B, H', W')
                unique_classes = len(torch.unique(class_assignments))

                # NEW: Which class dominates in argmax?
                flat_assignments = class_assignments.flatten()
                dominant_class = flat_assignments.mode().values.item()
                dominant_frac = (flat_assignments == dominant_class).float().mean().item()

            wandb.log({
                "train/pred_loss": pred_loss.item(),
                "train/reg_loss": reg_loss.item(),
                "train/L_marg": L_marg.item(),  # Marginal entropy penalty (anti-collapse)
                "train/L_cond": L_cond.item(),  # Conditional entropy (sharpness)
                "train/L_cross": L_cross.item(),  # Cross-sample variance (negative)
                "train/mean_var": mean_var.item(),  # Actual variance (should increase)
                "train/L_equiv": L_equiv.item(),  # Equivariance loss (should decrease)
                "train/total_loss": loss.item(),
                "train/class_entropy": class_entropy,
                "train/num_active_classes": num_active_classes,
                "train/avg_max_prob": avg_max_prob,  # Should be >> 1/K if confident
                "train/unique_classes_in_batch": unique_classes,
                "train/dominant_class": dominant_class,
                "train/dominant_class_frac": dominant_frac,  # Fraction of pixels assigned to dominant class
                "train/lr": scheduler.get_last_lr()[0],
            })

            pbar.set_postfix({
                "pred": f"{pred_loss.item():.4f}",
                "equiv": f"{L_equiv.item():.4f}",  # Equivariance loss (should decrease)
                "var": f"{mean_var.item():.4f}",   # Cross-sample variance (should increase)
                "maxP": f"{avg_max_prob:.3f}",     # Confidence (1/K=0.02 if uniform)
                "uniq": unique_classes,
            })

        # Epoch summary
        avg_pred_loss = epoch_pred_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        avg_L_marg = epoch_L_marg / len(train_loader)
        avg_L_cond = epoch_L_cond / len(train_loader)
        avg_mean_var = epoch_mean_var / len(train_loader)
        avg_L_equiv = epoch_L_equiv / len(train_loader)
        print(f"Epoch {epoch+1}: pred={avg_pred_loss:.4f}, equiv={avg_L_equiv:.4f}, var={avg_mean_var:.4f}")

        wandb.log({
            "epoch/pred_loss": avg_pred_loss,
            "epoch/reg_loss": avg_reg_loss,
            "epoch/L_marg": avg_L_marg,
            "epoch/L_cond": avg_L_cond,
            "epoch/mean_var": avg_mean_var,
            "epoch/L_equiv": avg_L_equiv,
            "epoch": epoch,
        })

        # Save checkpoint and visualizations every 10 epochs
        if epoch == 1 or (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": opt.state_dict(),
            }
            ckpt_path = f"checkpoint_epoch{epoch+1}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # Save sample visualizations
            save_class_visualization(
                encoder=encoder,
                dataset=train_ds,
                device=device,
                epoch=epoch + 1,
                output_dir="visualizations",
                K=cfg.K,
                num_samples=cfg.get("viz_samples", 4),
            )

    # Final save
    final_ckpt = {
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
    }
    torch.save(final_ckpt, "dense_ssl_final.pt")
    print("Saved final model to dense_ssl_final.pt")

    wandb.finish()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\nTotal training time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
