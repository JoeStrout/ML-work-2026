"""
Joe Net 1 - U-Net with ConvNeXt-style bottleneck for EM segmentation

Architecture:
- 4-level encoder-decoder U-Net
- Pre-activation residual blocks (ResNet-v2 style)
- ConvNeXt-style large kernel depthwise convs at bottleneck
- Skip connections between encoder and decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActivationResBlock(nn.Module):
    """
    Pre-activation residual block (ResNet-v2 style)

    Flow: BN → GELU → Conv3x3 → BN → GELU → Conv3x3 → Add residual
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # Residual connection (1x1 conv if channels change)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        return out + residual


class ConvNeXtBottleneckBlock(nn.Module):
    """
    ConvNeXt-style bottleneck block with large depthwise kernels

    Flow: BN → GELU → DepthwiseConv7x7 → BN → GELU → Conv1x1 → Add residual
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        # Depthwise convolution (groups=channels means each channel processed independently)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=channels, bias=False)

        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()
        # Pointwise convolution (1x1) to mix channels
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.dwconv(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.pwconv(out)

        return out + residual


class EncoderLevel(nn.Module):
    """Single encoder level with residual blocks + downsampling"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        # First block may change channels
        blocks = [PreActivationResBlock(in_channels, out_channels)]
        # Remaining blocks maintain same channels
        for _ in range(num_blocks - 1):
            blocks.append(PreActivationResBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

        # Downsampling via strided convolution
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        x = self.blocks(x)
        skip = x  # Save for skip connection
        x = self.downsample(x)
        return x, skip


class DecoderLevel(nn.Module):
    """Single decoder level with upsampling + skip connection + residual blocks"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        # Upsampling via transposed convolution
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)

        # After concatenating skip connection, input channels = in_channels + in_channels
        # (because skip has same channels as current level before upsampling)
        blocks = [PreActivationResBlock(in_channels * 2, out_channels)]
        for _ in range(num_blocks - 1):
            blocks.append(PreActivationResBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenate skip connection along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.blocks(x)
        return x


class UNet(nn.Module):
    """
    U-Net for EM segmentation with:
    - 4 encoder levels (32, 64, 128, 256 channels)
    - ConvNeXt-style bottleneck (256 channels, 7x7 depthwise)
    - 4 decoder levels (mirror encoder)
    - 2 output channels (nuclei, mito)
    """
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super().__init__()

        # Initial convolution to go from input channels to base_channels
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # Encoder: 4 levels
        self.enc1 = EncoderLevel(base_channels, base_channels * 1)      # 32 → 32
        self.enc2 = EncoderLevel(base_channels * 1, base_channels * 2)   # 32 → 64
        self.enc3 = EncoderLevel(base_channels * 2, base_channels * 4)   # 64 → 128
        self.enc4 = EncoderLevel(base_channels * 4, base_channels * 8)   # 128 → 256

        # Bottleneck: ConvNeXt-style blocks with large kernels
        self.bottleneck = nn.Sequential(
            ConvNeXtBottleneckBlock(base_channels * 8, kernel_size=7),
            ConvNeXtBottleneckBlock(base_channels * 8, kernel_size=7)
        )

        # Decoder: 4 levels (mirror encoder)
        self.dec4 = DecoderLevel(base_channels * 8, base_channels * 4)  # 256 → 128
        self.dec3 = DecoderLevel(base_channels * 4, base_channels * 2)  # 128 → 64
        self.dec2 = DecoderLevel(base_channels * 2, base_channels * 1)  # 64 → 32
        self.dec1 = DecoderLevel(base_channels * 1, base_channels * 1)  # 32 → 32

        # Output head: 1x1 conv to output channels
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input convolution
        x = self.input_conv(x)

        # Encoder path (save skip connections)
        x, skip1 = self.enc1(x)  # 256x256 → 128x128, skip: 256x256
        x, skip2 = self.enc2(x)  # 128x128 → 64x64, skip: 128x128
        x, skip3 = self.enc3(x)  # 64x64 → 32x32, skip: 64x64
        x, skip4 = self.enc4(x)  # 32x32 → 16x16, skip: 32x32

        # Bottleneck
        x = self.bottleneck(x)   # 16x16 (with large receptive field)

        # Decoder path (use skip connections)
        x = self.dec4(x, skip4)  # 16x16 → 32x32
        x = self.dec3(x, skip3)  # 32x32 → 64x64
        x = self.dec2(x, skip2)  # 64x64 → 128x128
        x = self.dec1(x, skip1)  # 128x128 → 256x256

        # Output
        x = self.output_conv(x)  # 256x256x2 (logits)

        return x


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing UNet architecture...")
    model = UNet(in_channels=1, out_channels=2, base_channels=32)

    # Count parameters
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 256, 256)  # Batch of 2
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 2, 256, 256)")

    assert output.shape == (2, 2, 256, 256), "Output shape mismatch!"
    print("\nModel test passed!")
