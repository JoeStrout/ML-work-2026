# Joe Net 1 - Training Guide

## Quick Start

### On Your Mac (Development/Testing)
```bash
# Activate environment
micromamba activate pytorch-wx

# Test model architecture
python model.py

# Test loss functions
python utils.py
```

### On Your PC with RTX 4090 (Training)

1. **Transfer files to PC:**
   ```
   model.py
   dataset.py
   utils.py
   train.py
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install cloud-volume
   pip install numpy scipy pillow
   ```

3. **Start training:**
   ```bash
   python train.py
   ```

## Training Configuration

Edit the `Config` class in `train.py` to adjust hyperparameters:

```python
class Config:
    # Model
    base_channels = 32          # Base number of channels (32→64→128→256)

    # Data
    batch_size = 16             # Increase if you have more GPU memory
    patches_per_epoch = 1000    # Random patches per epoch

    # Training
    num_epochs = 100            # Total epochs
    learning_rate = 1e-4        # Starting learning rate
    weight_decay = 1e-4         # AdamW weight decay

    # Mixed precision
    use_amp = True              # Use FP16 for 2x speedup (recommended on RTX 4090)

    # Validation
    val_every = 5               # Validate every N epochs
    vis_every = 10              # Save visualizations every N epochs
```

## Training Output

### Checkpoints (`checkpoints/`)
- `best.pth` - Best model based on validation loss
- `latest.pth` - Latest checkpoint (for resuming)
- `final.pth` - Final model after training completes

### Visualizations (`visualizations/`)
- `epoch_XXX_sample_Y.png` - Side-by-side: input | ground truth | prediction
- Each image shows: grayscale, nuclei (red), mito (cyan)

### Console Output
```
Epoch [50/100]
  Batch [50/62], Loss: 0.4523
  Train Loss: 0.4521
  Val Loss: 0.4102
  Val Mean IoU: 0.6543
  Val Mean Dice: 0.7891
  Val Nuclei IoU: 0.6723
  Val Mito IoU: 0.6363
  New best model saved!
  Epoch time: 45.23s
```

## Monitoring Training

### Key Metrics to Watch:
- **Train Loss:** Should decrease steadily
- **Val Loss:** Should decrease (if increasing, model is overfitting)
- **Mean IoU:** Intersection over Union (higher is better, target > 0.7)
- **Mean Dice:** Dice coefficient (higher is better, target > 0.8)
- **Per-class IoU:** Check nuclei and mito separately

### Typical Training Progress:
- **Epoch 1-10:** Loss drops rapidly, metrics improve quickly
- **Epoch 10-50:** Steady improvement
- **Epoch 50-100:** Refinement, diminishing returns

### Signs of Good Training:
- Train loss and val loss both decreasing
- Val loss not much higher than train loss (< 10% difference)
- IoU > 0.7, Dice > 0.8 by epoch 50-100

### Signs of Problems:
- **Overfitting:** Train loss decreases but val loss increases
  - Solution: Add more augmentation, increase weight decay
- **Underfitting:** Both losses plateau at high values
  - Solution: Increase model capacity (more channels), train longer
- **Unstable training:** Loss jumps around wildly
  - Solution: Decrease learning rate, add gradient clipping

## Resuming Training

If training is interrupted, it automatically resumes from `checkpoints/latest.pth`:

```bash
python train.py  # Will load latest.pth if it exists
```

## Using the Trained Model

After training, use the best model for inference:

```python
import torch
from model import UNet

# Load model
model = UNet(in_channels=1, out_channels=2, base_channels=32)
checkpoint = torch.load("checkpoints/best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    input_image = torch.randn(1, 1, 256, 256)  # Your image
    output = model(input_image)  # (1, 2, 256, 256) logits

    # Convert to probabilities and threshold
    probs = torch.sigmoid(output)
    nuclei_mask = (probs[0, 0] > 0.5).numpy()
    mito_mask = (probs[0, 1] > 0.5).numpy()
```

## Expected Performance

### Training Time (RTX 4090):
- ~45-60 seconds per epoch (1000 patches, batch size 16)
- ~50-100 epochs to convergence
- **Total: 1-3 hours**

### Memory Usage:
- Model: ~80 MB (5.1M parameters)
- Batch (16×256×256): ~200 MB
- Gradients + optimizer states: ~500 MB
- **Total GPU memory: < 1 GB** (plenty of room on 24 GB card)

### Final Metrics (Expected):
- **Nuclei IoU:** 0.70-0.85 (depends on data quality)
- **Mito IoU:** 0.65-0.80 (mito is harder due to smaller size)
- **Mean Dice:** 0.75-0.90

## Troubleshooting

### CUDA Out of Memory:
```python
# Reduce batch size in train.py
config.batch_size = 8  # or 4
```

### Training Too Slow:
```python
# Ensure mixed precision is enabled
config.use_amp = True

# Increase batch size (if you have memory)
config.batch_size = 32

# Reduce workers if CPU is bottleneck
config.num_workers = 2
```

### Poor Validation Performance:
1. Check visualizations - are predictions reasonable?
2. Try training longer (100+ epochs)
3. Increase augmentation strength in `dataset.py`
4. Try different learning rate (1e-3 or 1e-5)

### CloudVolume Connection Issues:
- Ensure Google Cloud credentials are set up on PC
- Test with `gcloud auth application-default login`
- Check internet connection and firewall

## Next Steps After Training

1. **Evaluate on test set** - Hold out additional Z slices for final evaluation
2. **Full volume inference** - Apply model to entire volume with sliding window
3. **Hyperparameter tuning** - Try different learning rates, model sizes
4. **Architecture improvements** - Add attention, deeper network (see NET_DESIGN.md)
5. **Export predictions** - Save masks back to CloudVolume format

---

**Author:** Joe (with Claude assistance)
**Date:** 2025-11-21
