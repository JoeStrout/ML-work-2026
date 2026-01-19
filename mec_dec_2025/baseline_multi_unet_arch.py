"""
Standard U-Net architecture for multi-class semantic segmentation with hierarchical supervision.
Baseline version without pre-training.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled features from previous layer
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineMultiUNet(nn.Module):
    """
    Standard U-Net for multi-class semantic segmentation with hierarchical supervision.
    Baseline version without pre-training.

    Architecture uses 5 levels with channel progression: 64 -> 128 -> 256 -> 512 -> 1024
    Outputs both fine-grained class predictions and coarse group predictions.
    """

    def __init__(self, in_channels=1, num_classes=12, num_coarse_groups=5):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale EM)
            num_classes: Number of fine-grained classes (12)
            num_coarse_groups: Number of coarse groups (5)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_coarse_groups = num_coarse_groups

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Fine segmentation head (12-way classification)
        self.fine_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Coarse group head (5-way multi-label)
        self.coarse_head = nn.Conv2d(64, num_coarse_groups, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, 1, H, W)

        Returns:
            logits_fine: Logits of shape (N, 12, H, W) for fine classification
            logits_coarse: Logits of shape (N, 5, H, W) for coarse groups
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Dual heads for hierarchical supervision
        logits_fine = self.fine_head(x)      # (N, 12, H, W)
        logits_coarse = self.coarse_head(x)  # (N, 5, H, W)

        return logits_fine, logits_coarse


# Test the architecture
if __name__ == "__main__":
    # Create model
    model = BaselineMultiUNet(in_channels=1, num_classes=12, num_coarse_groups=5)

    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    logits_fine, logits_coarse = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Fine output shape: {logits_fine.shape}")
    print(f"Coarse output shape: {logits_coarse.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
