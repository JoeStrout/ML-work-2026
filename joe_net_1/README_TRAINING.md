# Joe Net 1 - Complete Training System

## Quick Start

### Option 1: GUI Training (Recommended for first run)
```bash
micromamba activate pytorch-wx
python joe_net_1_train.py
```
Click "▶ Start Training" and watch the magic happen!

### Option 2: Headless Training (For long runs)
```bash
python train.py
```

## What You've Built

A complete end-to-end training system for semantic segmentation of EM data:

### 🏗️ Architecture (`model.py`)
- **U-Net** with 4 encoder/decoder levels
- **Pre-activation residual blocks** (ResNet-v2 style)
- **ConvNeXt-style bottleneck** with 7×7 depthwise convolutions
- **5.1M parameters**, lightweight and fast
- **Receptive field:** ~176 pixels (69% of 256×256 image)

### 📊 Data Pipeline (`dataset.py`)
- Loads full 1.2 GB volume from CloudVolume
- Random 256×256 patch sampling
- **Augmentation:** rotation, flip, brightness, contrast, noise
- Train/val split: 80/20 by Z slices

### 📉 Loss & Metrics (`utils.py`)
- **Combined BCE + Dice loss** (handles class imbalance)
- **Comprehensive metrics:** accuracy, precision, recall, F1, IoU, Dice
- **Visualization:** side-by-side comparison of input/GT/prediction

### 🎯 Training System (`trainer.py`)
- **Trainer class** with callback architecture
- **AdamW optimizer** with cosine annealing
- **Mixed precision (FP16)** for 2× speedup on GPU
- **Checkpointing:** best, latest, final models
- **Resume support:** automatically continues from `latest.pth`

### 🖥️ GUI Training (`joe_net_1_train.py`)
- **Live loss graphs:** Train and validation loss over time
- **IoU metric graphs:** Track nuclei and mito performance
- **Prediction visualization:** See results as training progresses
- **Progress tracking:** Real-time status and progress bar
- **Background training:** UI stays responsive during training

### 💻 Headless Training (`train.py`)
- **Console output:** Clean progress updates
- **Batch training:** Perfect for remote servers
- **Same underlying system:** Uses `Trainer` with console callbacks

## File Structure

```
wx-dev/
├── model.py                 - UNet architecture
├── dataset.py               - Data loading and augmentation
├── utils.py                 - Loss functions and metrics
├── trainer.py               - Core training logic with callbacks
├── train.py                 - Headless training script
├── joe_net_1_train.py       - GUI training application
├── joe_net_1.py             - Original data viewer (still works!)
│
├── NET_DESIGN.md            - Architecture design document
├── TRAINING_GUIDE.md        - Training configuration guide
├── TRAINING_GUI.md          - GUI training detailed guide
├── CLAUDE.md                - Project overview
└── README_TRAINING.md       - This file
```

## Training Workflow

### 1. Test Locally (Mac)
```bash
python joe_net_1_train.py
```
- Verify everything works
- Check GUI updates
- Train for 5-10 epochs
- Inspect predictions

### 2. Transfer to PC (RTX 4090)
Copy these files:
- `model.py`
- `dataset.py`
- `utils.py`
- `trainer.py`
- `train.py` (for headless)
- `joe_net_1_train.py` (for GUI)

### 3. Full Training
```bash
# Headless (recommended for long runs)
python train.py

# Or GUI (if you want to watch)
python joe_net_1_train.py
```

### 4. Monitor Progress
**Headless:** Check console output
**GUI:** Watch live graphs and predictions

**Key metrics:**
- Train loss: Should decrease to ~0.3-0.5
- Val loss: Should track train loss
- Nuclei IoU: Target > 0.75
- Mito IoU: Target > 0.65

### 5. Evaluate Results
Best model saved to `checkpoints/best.pth`

```python
import torch
from model import UNet

model = UNet(in_channels=1, out_channels=2, base_channels=32)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference!
```

## Expected Performance

### Training Time (RTX 4090)
- **Per epoch:** ~45-60 seconds (1000 patches, batch size 16)
- **To convergence:** ~50-100 epochs
- **Total time:** 1-3 hours

### Model Performance (Expected)
- **Nuclei IoU:** 0.70-0.85
- **Mito IoU:** 0.65-0.80
- **Mean Dice:** 0.75-0.90

### Resource Usage
- **Model size:** ~80 MB (5.1M parameters)
- **GPU memory:** < 1 GB (batch size 16)
- **Disk space:** ~500 MB (checkpoints + visualizations)

## Customization

### Change Hyperparameters
Edit `trainer.py`:
```python
class TrainerConfig:
    batch_size = 32           # Increase for GPU
    learning_rate = 1e-3      # Try higher
    num_epochs = 200          # Train longer
    val_every = 2             # Validate more often
```

### Change Architecture
Edit `model.py`:
```python
model = UNet(
    in_channels=1,
    out_channels=2,
    base_channels=64    # Wider network (more parameters)
)
```

### Add More Augmentation
Edit `dataset.py` → `augment_patch()`:
```python
# Add elastic deformation
if random.random() > 0.5:
    img = elastic_transform(img)
```

### Custom Callbacks
Create your own in `train.py` or `joe_net_1_train.py`:
```python
class MyCallbacks(TrainingCallbacks):
    def on_epoch_end(self, epoch, train_loss, val_loss, metrics):
        # Log to Weights & Biases
        wandb.log({'loss': train_loss, 'epoch': epoch})
```

## Troubleshooting

### Problem: Out of memory
**Solution:** Reduce `batch_size` in `TrainerConfig`

### Problem: Training too slow
**Solution:**
- Use GPU (RTX 4090)
- Enable `use_amp = True` (mixed precision)
- Reduce `patches_per_epoch`

### Problem: Poor validation performance
**Solution:**
- Train longer (more epochs)
- Add more augmentation
- Check visualizations - are predictions reasonable?
- Try different learning rate

### Problem: Overfitting (val loss > train loss)
**Solution:**
- Add more augmentation
- Increase `weight_decay`
- Train with more data patches

## Next Steps

### Immediate
1. ✅ Test GUI training on Mac (5-10 epochs)
2. ✅ Verify graphs and predictions update
3. ✅ Transfer to PC with RTX 4090

### Short-term
4. Train full 100 epochs on PC
5. Evaluate best model on validation set
6. Visualize predictions on test slices

### Long-term
7. Hyperparameter tuning (learning rate, batch size)
8. Architecture experiments (deeper, wider, attention)
9. Full volume inference (sliding window)
10. Export predictions to CloudVolume

## Key Design Decisions

### Why Pre-Activation?
Better gradient flow, proven in ResNet-v2

### Why ConvNeXt-style Bottleneck?
Large receptive field (176 pixels) with minimal parameters

### Why Combined BCE + Dice Loss?
BCE for pixel accuracy, Dice for class imbalance

### Why Callback Architecture?
Separation of concerns: training logic separate from I/O (console/GUI)

### Why Thread-based GUI?
Keep UI responsive during long training runs

## References

- **Architecture:** `NET_DESIGN.md`
- **Training Guide:** `TRAINING_GUIDE.md`
- **GUI Details:** `TRAINING_GUI.md`
- **Project Overview:** `CLAUDE.md`

## Support

If something doesn't work:
1. Check console output for errors
2. Verify volumes load correctly
3. Test model with `python model.py`
4. Test dataset with `python dataset.py`
5. Test utils with `python utils.py`

---

**You now have a production-ready training system with both headless and GUI modes!**

Enjoy training your network! 🚀

---

**Author:** Joe (with Claude assistance)
**Date:** 2025-11-21
**Status:** Ready for training!
