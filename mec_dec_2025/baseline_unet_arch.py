"""
Standard U-Net architecture for binary segmentation (baseline, no pre-training).
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


class BaselineUNet(nn.Module):
    """
    Standard U-Net for binary segmentation (baseline without pre-training).

    Architecture matches the complexity of the LeJEPA model for fair comparison.
    Uses 5 levels with channel progression: 64 -> 128 -> 256 -> 512 -> 1024
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

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

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, 1, H, W)

        Returns:
            logits: Output logits of shape (N, 1, H, W)
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

        # Output
        logits = self.outc(x)
        return logits


# Test the architecture
if __name__ == "__main__":
    # Create model
    model = BaselineUNet(in_channels=1, out_channels=1)

    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
