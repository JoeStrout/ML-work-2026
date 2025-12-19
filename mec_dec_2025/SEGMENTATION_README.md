# Mitochondria Segmentation with Pre-trained LeJEPA

This folder contains a U-Net architecture for binary mitochondria segmentation using the LeJEPA pre-trained ConvNeXt encoder.

## Files

- `unet_architecture.py` - U-Net model with ConvNeXt encoder and decoder
- `mito_seg_train.py` - Training script for mitochondria segmentation
- `seg_config.yaml` - Configuration for segmentation training
- `check_seg_data.py` - Script to verify segmentation data loading

## Architecture

**ConvNeXtUNet** - U-Net style encoder-decoder:
- **Encoder**: Pre-trained ConvNeXt from LeJEPA checkpoint (frozen by default)
  - 4 stages with feature dims: [96, 192, 384, 768]
  - Spatial resolutions: [H/4, H/8, H/16, H/32]

- **Decoder**: Symmetric upsampling path
  - 4 decoder blocks with skip connections
  - Final upsample to match input resolution
  - Output: Binary segmentation mask (same size as input)

- **Parameters**: ~60M total, ~10M trainable (when encoder is frozen)

## Data

- **Image volume**: `gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/`
  - Full volume: 5904×5904×120 voxels

- **Segmentation volume**: `gs://joe_exp/mec_training/mec_sem_cls`
  - Subset: 1792×1792×16 voxels
  - 13 semantic classes (see table below)
  - Mitochondria: label 10 (~7.78% of volume)

- **Train/val split**: First 80% of Z-slices for train, last 20% for val

## Training

### Start training:
```bash
python mito_seg_train.py
```

### Configuration (seg_config.yaml):
- `pretrained_checkpoint`: Path to LeJEPA checkpoint
- `freeze_encoder`: Whether to freeze encoder (start with true)
- `patch_size`: 128×128 patches
- `bs`: Batch size (32)
- `lr`: Learning rate (1e-3)
- `pos_weight`: 11.86 (compensates for 7.78% vs 92.22% class imbalance)
- `epochs`: 100

### Metrics:
- **Loss**: Weighted Binary Cross-Entropy
- **Dice coefficient**: Primary metric (harmonic mean of precision/recall)
- **IoU**: Intersection over Union

### Outputs:
- Checkpoints saved to `checkpoints_seg/`
- Best model: `best_model_dice{score}.pt`
- Visualizations: `seg_predictions/epoch_XXXX.png` every 10 epochs
- WandB logging: Project "MitoSeg_Dec2025"

## Fine-tuning the Encoder

After initial training with frozen encoder, you can fine-tune:

1. Update `seg_config.yaml`: Set `freeze_encoder: false`
2. Reduce learning rate: `lr: 0.0001` (or use differential learning rates)
3. Continue training from best checkpoint

## Segmentation Classes (for reference)

| Class               | SegID | Percentage |
|---------------------|-------|------------|
| Extracellular space | 0     | 18.07%     |
| Tear                | 1     | 0.22%      |
| Dendrite            | 2     | 24.26%     |
| Axon                | 3     | 33.02%     |
| Soma                | 4     | 0.00%      |
| Glia                | 5     | 8.16%      |
| Myelin              | 6     | 4.28%      |
| Myelin inner tongue | 7     | 0.32%      |
| Myelin outer tongue | 8     | 0.30%      |
| Nucleus             | 9     | 1.83%      |
| **Mitochondria**    | **10**| **7.78%**  |
| Fat globule         | 11    | 0.24%      |
| Uncertain           | 255   | 1.52%      |
