# Joe Net 1 - Network Architecture Design

## Overview
Lightweight U-Net for semantic segmentation of electron microscopy data, predicting nuclei and mitochondria from grayscale EM images.

## Design Philosophy
- **Start simple, iterate:** Begin with a lightweight architecture and add complexity only if needed
- **Modern components:** Use proven techniques from recent research
- **Fast training:** Target 1-3 hours on RTX 4090 with mixed precision
- **Pre-activation design:** ResNet-v2 style (BN → Activation → Conv) for better gradient flow

## Architecture

### Input/Output
- **Input:** 256×256×1 (grayscale EM image, normalized to [0, 1])
- **Output:** 256×256×2 (logits for nuclei and mito channels)

### Encoder-Decoder Structure
```
Input: 256×256×1

Encoder (4 levels with downsampling):
  Level 1: 32 channels   (256×256) → downsample → (128×128)
  Level 2: 64 channels   (128×128) → downsample → (64×64)
  Level 3: 128 channels  (64×64)   → downsample → (32×32)
  Level 4: 256 channels  (32×32)   [bottleneck]

Decoder (4 levels with upsampling + skip connections):
  Level 4: 256 channels  (32×32)   → upsample → (64×64)
  Level 3: 128 channels  (64×64)   → upsample → (128×128)
  Level 2: 64 channels   (128×128) → upsample → (256×256)
  Level 1: 32 channels   (256×256)

Output head: 256×256×2 (1×1 conv to 2 channels)
```

### Building Blocks

#### Pre-Activation Residual Block (ResNet-v2 style)
```
Input
  ↓
BatchNorm → LeakyReLU/GELU → Conv 3×3
  ↓
BatchNorm → LeakyReLU/GELU → Conv 3×3
  ↓
Add with residual connection (identity or 1×1 conv if channels change)
  ↓
Output
```

**Why pre-activation?**
- Better gradient flow (activation before conv means gradients propagate more cleanly)
- Regularization effect (BN before conv acts as regularizer)
- Proven in ResNet-v2 (He et al., 2016)
- No "dead neurons" at block output (output is raw features, not post-ReLU)

#### ConvNeXt-Style Bottleneck Block (Level 4 only)
For large receptive field at the bottleneck, use depthwise large kernels:
```
Input
  ↓
BatchNorm → GELU → Depthwise Conv 7×7 (operates on each channel independently)
  ↓
BatchNorm → GELU → Conv 1×1 (pointwise, mixes channels)
  ↓
Add with residual connection
  ↓
Output
```

**Why large depthwise kernels at bottleneck?**
- **Receptive field:** 7×7 kernel at 32×32 resolution → RF ≈ 176 pixels (vs ~120 with 3×3)
- **Efficiency:** Depthwise conv is cheap (params = kernel_size² × channels, not channels²)
- **Context:** Captures long-range dependencies needed for semantic understanding
- **Modern design:** Inspired by ConvNeXt (Liu et al., 2022)

#### Downsampling
- Strided convolution (2×2 stride) OR
- Max pooling (2×2)
- Decision: Try both, strided conv is more modern but max pool is classic U-Net

#### Upsampling
- Transposed convolution (learnable) OR
- Bilinear upsample + 3×3 conv (often works better, less checkerboard artifacts)
- Decision: Start with transposed conv for simplicity

### Skip Connections
- Concatenate encoder features with decoder features at each level
- Classic U-Net style
- Helps preserve fine spatial details for precise segmentation

## Training Configuration

### Loss Function
**Combined BCE + Dice Loss:**
```python
loss = BCE_loss + Dice_loss
```

- **Binary Cross-Entropy (BCE):** Pixel-wise classification
- **Dice Loss:** Handles class imbalance (most pixels are background)
- Both applied per-channel (nuclei, mito) and averaged

**Why combined?**
- BCE optimizes pixel accuracy
- Dice optimizes overlap (better for sparse objects)
- Together they balance precision and recall

### Optimizer
- **AdamW** (Adam with decoupled weight decay)
- Initial learning rate: 1e-3 or 1e-4
- Weight decay: 1e-4
- Cosine annealing or ReduceLROnPlateau scheduler

### Data Augmentation
Essential for small datasets:
- **Geometric:** Random rotation (0-360°), horizontal/vertical flip
- **Intensity:** Random brightness/contrast adjustment
- **Elastic deformation:** Simulate tissue distortion (optional, biomedical standard)
- **Noise:** Gaussian noise (optional)

### Training Details
- **Batch size:** 16-32 (RTX 4090 can handle it)
- **Epochs:** 50-100 (early stopping on validation loss)
- **Mixed precision:** FP16/BF16 for 2× speedup
- **Gradient clipping:** Optional, if training unstable

### Validation Strategy
- **Hold out Z slices 880-900** (20 slices = 20% of data)
- Tests generalization across Z dimension
- Alternative: Spatial holdout (one corner of XY)

### Evaluation Metrics
- **Pixel Accuracy:** Overall correctness
- **IoU (Intersection over Union):** Per-class overlap
- **Dice Score:** Per-class F1-equivalent
- **Precision/Recall:** Per-class
- **Visual inspection:** Side-by-side (input, ground truth, prediction)

## Data Pipeline

### Dataset Design
```python
class EMSegmentationDataset(Dataset):
    - Load volumes once at init (already in memory)
    - __getitem__:
        - Random sample (x, y, z) coordinates
        - Extract 256×256 slice
        - Apply augmentation
        - Return (image, nuclei_mask, mito_mask)
```

### Train/Val Split
- **Train:** Z slices 800-879 (80 slices)
- **Val:** Z slices 880-900 (20 slices, held out)
- Random 256×256 patches from each slice
- ~1000-2000 patches per epoch

## Future Enhancements (Add Only If Needed)

### If accuracy insufficient:
1. **Attention mechanisms:** CBAM or Squeeze-and-Excitation blocks
2. **Deeper network:** Add level 5 (512 channels at 16×16)
3. **Wider network:** Increase channel counts (32→64, 64→128, etc.)
4. **Deep supervision:** Auxiliary losses at multiple decoder levels
5. **Pre-trained encoder:** ImageNet weights (if transferable)
6. **Test-time augmentation:** Average predictions over augmented versions

### If training too slow:
1. **Depthwise separable convolutions:** MobileNet-style efficiency
2. **Smaller batch size + gradient accumulation**
3. **Mixed precision optimization**

### If overfitting:
1. **More augmentation:** Stronger elastic deformation, noise
2. **Dropout:** Add to bottleneck
3. **More data:** Generate more random patches per epoch

## Estimated Performance

### Training Time (RTX 4090)
- ~1000-2000 patches per epoch
- ~50-100 epochs to convergence
- **Total: 1-3 hours** with mixed precision

### Memory Usage
- Model: ~10-20M parameters → ~40-80 MB
- Batch of 32×256×256: ~200 MB (input + gradients)
- **Total GPU memory: < 2 GB** (plenty of headroom on 24 GB card)

## Code Structure

```
model.py          - UNet class, residual blocks
dataset.py        - EMSegmentationDataset, augmentation
train.py          - Training loop, checkpointing
eval.py           - Evaluation metrics, visualization
config.py/.yaml   - Hyperparameters
utils.py          - Helper functions (loss, metrics)
```

## Receptive Field Analysis

With the chosen architecture:
- **Levels 1-3:** Standard 3×3 convolutions → RF ≈ 88 pixels
- **Level 4 (bottleneck):** 7×7 depthwise convolutions → RF ≈ **176 pixels**
- **Coverage:** ~69% of 256×256 input image
- **Sufficient for:** Understanding cellular context, distinguishing nuclei from shadows, identifying mitochondria clusters

## References

- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **ResNet-v2:** He et al., "Identity Mappings in Deep Residual Networks" (2016)
- **Dice Loss:** Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)
- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s" (2022)

---

**Author:** Joe (with Claude assistance)
**Date:** 2025-11-21
**Hardware:** Development on Intel Mac, Training on RTX 4090
