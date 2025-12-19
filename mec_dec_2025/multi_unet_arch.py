"""
U-Net architecture for multi-class semantic segmentation with hierarchical supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtUNetEncoder(nn.Module):
    """ConvNeXt encoder that extracts multi-scale features for U-Net."""

    def __init__(self, pretrained_path=None, freeze=True):
        """
        Args:
            pretrained_path: Path to LeJEPA pre-trained checkpoint (optional)
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        # Create ConvNeXt backbone
        # convnext_small has feature dims: [96, 192, 384, 768] at stages [0, 1, 2, 3]
        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=False,
            in_chans=1,
            num_classes=0,  # Remove classification head
            features_only=True,  # Return intermediate features
            out_indices=(0, 1, 2, 3)  # Return features from all 4 stages
        )

        # Load pre-trained weights if provided
        if pretrained_path:
            print(f"Loading pre-trained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Extract backbone weights from checkpoint
            # The checkpoint contains the full ConvNeXtEncoder with projection head
            # We only want the backbone weights
            state_dict = checkpoint['model_state_dict']
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    # Remove 'backbone.' prefix and '.head' suffix if present
                    new_key = key.replace('backbone.', '')
                    if not new_key.startswith('head'):
                        backbone_state_dict[new_key] = value

            # Load the filtered state dict
            missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Loaded backbone weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Freeze encoder if requested
        if freeze:
            print("Freezing encoder weights")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimensions for each stage
        # For convnext_small: [96, 192, 384, 768]
        self.feature_dims = self.backbone.feature_info.channels()
        print(f"Encoder feature dimensions: {self.feature_dims}")

    def forward(self, x):
        """
        Returns multi-scale features from encoder.

        Args:
            x: Input tensor of shape (N, 1, H, W)

        Returns:
            List of features [stage0, stage1, stage2, stage3]
            Spatial resolutions: [H/4, H/8, H/16, H/32]
        """
        features = self.backbone(x)
        return features

    def unfreeze(self):
        """Unfreeze encoder for fine-tuning."""
        print("Unfreezing encoder weights")
        for param in self.backbone.parameters():
            param.requires_grad = True


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Channels from previous decoder layer
            skip_channels: Channels from encoder skip connection
            out_channels: Output channels
        """
        super().__init__()

        # Upsample by 2x
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # After concatenation with skip connection
        concat_channels = in_channels + skip_channels

        # Two conv blocks
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x, skip):
        """
        Args:
            x: Upsampled features from previous decoder layer
            skip: Skip connection from encoder

        Returns:
            Decoded features
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class ConvNeXtUNet(nn.Module):
    """U-Net with ConvNeXt encoder for multi-class semantic segmentation with hierarchical supervision."""

    def __init__(self, num_classes=12, num_coarse_groups=5, pretrained_path=None, freeze_encoder=True):
        """
        Args:
            num_classes: Number of fine-grained classes (12)
            num_coarse_groups: Number of coarse groups (5)
            pretrained_path: Path to LeJEPA pre-trained checkpoint
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_coarse_groups = num_coarse_groups

        # Encoder
        self.encoder = ConvNeXtUNetEncoder(pretrained_path, freeze=freeze_encoder)
        enc_dims = self.encoder.feature_dims  # [96, 192, 384, 768]

        # Decoder
        # Decoder channels (matching encoder in reverse): [384, 192, 96, 48]
        dec_dims = [384, 192, 96, 48]

        # Build decoder blocks
        # Block 3: 768 -> 384 (upsample + concat with stage2:384)
        self.dec3 = DecoderBlock(enc_dims[3], enc_dims[2], dec_dims[0])

        # Block 2: 384 -> 192 (upsample + concat with stage1:192)
        self.dec2 = DecoderBlock(dec_dims[0], enc_dims[1], dec_dims[1])

        # Block 1: 192 -> 96 (upsample + concat with stage0:96)
        self.dec1 = DecoderBlock(dec_dims[1], enc_dims[0], dec_dims[2])

        # Block 0: 96 -> 48 (upsample, no skip)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[2], dec_dims[3], kernel_size=2, stride=2),
            nn.Conv2d(dec_dims[3], dec_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(dec_dims[3]),
            nn.GELU()
        )

        # Final upsample to match input resolution (ConvNeXt stem downsamples by 4x, we've only upsampled 2x)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[3], dec_dims[3], kernel_size=2, stride=2),
            nn.Conv2d(dec_dims[3], dec_dims[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(dec_dims[3]),
            nn.GELU()
        )

        # Fine segmentation head (12-way classification)
        self.fine_head = nn.Conv2d(dec_dims[3], num_classes, kernel_size=1)

        # Coarse group head (5-way multi-label)
        self.coarse_head = nn.Conv2d(dec_dims[3], num_coarse_groups, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, 1, H, W)

        Returns:
            logits_fine: Logits of shape (N, 12, H, W) for fine classification
            logits_coarse: Logits of shape (N, 5, H, W) for coarse groups
        """
        # Encoder
        enc_features = self.encoder(x)
        stage0, stage1, stage2, stage3 = enc_features

        # Decoder with skip connections
        x = self.dec3(stage3, stage2)  # 768+384 -> 384
        x = self.dec2(x, stage1)       # 384+192 -> 192
        x = self.dec1(x, stage0)       # 192+96 -> 96
        x = self.dec0(x)               # 96 -> 48
        x = self.final_upsample(x)     # 48 -> 48 (upsample to match input)

        # Dual heads for hierarchical supervision
        logits_fine = self.fine_head(x)      # (N, 12, H, W)
        logits_coarse = self.coarse_head(x)  # (N, 5, H, W)

        return logits_fine, logits_coarse

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        self.encoder.unfreeze()


# Test the architecture
if __name__ == "__main__":
    # Create model
    model = ConvNeXtUNet(num_classes=12, num_coarse_groups=5, pretrained_path=None, freeze_encoder=False)

    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    logits_fine, logits_coarse = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Fine output shape: {logits_fine.shape}")
    print(f"Coarse output shape: {logits_coarse.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
