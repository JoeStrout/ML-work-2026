# Joe Net 1 - Training GUI Guide

## Overview

You now have **two ways** to train your model:

1. **Headless training** (`train.py`) - For batch training on servers/PC
2. **GUI training** (`joe_net_1_train.py`) - For interactive training with live visualization

Both use the same underlying `Trainer` class with callback support.

## Headless Training (train.py)

Use this for unattended training, especially on your RTX 4090 PC.

```bash
# On your PC with RTX 4090
python train.py
```

**Output:**
- Console progress with epoch/batch updates
- Checkpoints saved to `checkpoints/`
- Best model based on validation loss

**Perfect for:**
- Long training runs
- Remote servers
- Batch experimentation

## GUI Training (joe_net_1_train.py)

Use this for interactive training with live feedback.

```bash
# On your Mac or PC
python joe_net_1_train.py
```

**Features:**

### 1. Live Loss Graphs
- Real-time plot of training and validation loss
- See convergence as it happens
- Identify overfitting immediately

### 2. IoU Metrics Graphs
- Track nuclei and mito IoU over time
- Compare performance between classes
- Gauge when training is complete

### 3. Prediction Visualization
- See latest validation predictions
- Side-by-side: Input | Ground Truth | Prediction
- Red = nuclei, Cyan = mito
- Updated every validation epoch

### 4. Progress Tracking
- Progress bar for overall training
- Status updates for current epoch/batch
- Model parameter count

## Architecture: How It Works

### Core Components

```
trainer.py
├── Trainer class          - Main training logic
├── TrainerConfig          - Hyperparameters
└── TrainingCallbacks      - Interface for events

train.py
└── ConsoleCallbacks       - Print to console

joe_net_1_train.py
├── GUICallbacks          - Update GUI widgets
└── TrainingPanel         - WxPython dashboard
```

### Callback Flow

```
Training Event                  → Callback Method           → Action
─────────────────────────────────────────────────────────────────────
Training starts                 → on_training_start()       → Show params
Epoch begins                    → on_epoch_start()          → Update progress
Batch completes                 → on_batch_end()            → Update status
Epoch ends                      → on_epoch_end()            → Plot loss
Validation ends                 → on_validation_end()       → Show predictions
Checkpoint saved                → on_checkpoint_saved()     → Notify user
Training complete               → on_training_end()         → Final status
```

### Thread Safety

The GUI runs training in a background thread to keep the UI responsive:
- Training happens in `training_thread`
- Callbacks use `wx.CallAfter()` to update GUI from worker thread
- UI remains interactive during training

## Creating Your Own Callbacks

You can easily create custom callbacks for other purposes (logging, W&B, etc.):

```python
from trainer import TrainingCallbacks

class CustomCallbacks(TrainingCallbacks):
    def on_epoch_end(self, epoch, train_loss, val_loss=None, metrics=None):
        # Log to file
        with open('training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch}: Loss {train_loss:.4f}\n")

    def on_validation_end(self, epoch, val_loss, metrics, sample_predictions):
        # Send to Weights & Biases
        import wandb
        wandb.log({
            'val_loss': val_loss,
            'nuclei_iou': metrics['nuclei_iou'],
            'mito_iou': metrics['mito_iou']
        })

# Use your callbacks
from trainer import Trainer, TrainerConfig
from dataset import load_volumes

config = TrainerConfig()
volumes = load_volumes(mip=0)
callbacks = CustomCallbacks()

trainer = Trainer(config, volumes, callbacks=callbacks)
trainer.train()
```

## GUI Features in Detail

### Loss Graph Tab
- **Blue line:** Training loss (updated every epoch)
- **Red line:** Validation loss (updated every `val_every` epochs)
- **X-axis:** Epoch number
- **Y-axis:** Combined BCE + Dice loss

**What to look for:**
- Both lines should trend downward
- Val loss should track train loss (not diverge)
- Flattening indicates convergence

### IoU Metrics Tab
- **Red line:** Nuclei IoU (intersection over union)
- **Cyan line:** Mito IoU
- **Target:** IoU > 0.7 for good performance

**What to look for:**
- Steady increase over epochs
- Nuclei typically higher than mito (larger objects)
- Convergence indicates training complete

### Predictions Tab
Shows 3 validation samples:
- **Left:** Grayscale input image
- **Middle:** Ground truth (red=nuclei, cyan=mito)
- **Right:** Model prediction (red=nuclei, cyan=mito)

**What to look for:**
- Early epochs: Prediction mostly wrong
- Middle epochs: Shapes start to match
- Late epochs: Prediction very close to ground truth

## Configuration

Edit `TrainerConfig` in `trainer.py`:

```python
class TrainerConfig:
    # Model
    base_channels = 32          # Base channel count (32→64→128→256)

    # Data
    batch_size = 16             # Samples per batch (reduce if OOM)
    patches_per_epoch = 1000    # Random patches per epoch

    # Training
    num_epochs = 100            # Total training epochs
    learning_rate = 1e-4        # AdamW learning rate
    weight_decay = 1e-4         # L2 regularization

    # Validation
    val_every = 5               # Validate every N epochs
```

For GUI training, you can override in `joe_net_1_train.py`:

```python
config = TrainerConfig()
config.batch_size = 8          # Smaller for CPU
config.val_every = 2           # More frequent for GUI feedback
```

## Tips for Best Results

### On Mac (Development)
- Use GUI mode to test architecture and visualize
- Reduce `batch_size` to 4-8 (CPU limited)
- Train for a few epochs to verify everything works
- Then move to PC for full training

### On PC with RTX 4090 (Production)
- Use headless mode for long runs
- Or use GUI mode to watch progress
- Increase `batch_size` to 16-32
- Enable `use_amp = True` for 2× speedup (FP16)
- Full training: 1-3 hours

### General Tips
- **Stop early if overfitting:** Val loss increases while train loss decreases
- **Increase augmentation if overfitting:** Edit `dataset.py`
- **Check predictions visually:** Trust your eyes, not just metrics
- **Resume from checkpoint:** Trainer automatically loads `latest.pth`

## Troubleshooting

### GUI doesn't start
```bash
# Ensure wx is installed
pip install wxPython
micromamba install wxpython
```

### Training is slow
- Reduce `batch_size`
- Reduce `patches_per_epoch`
- Use GPU (RTX 4090)
- Enable mixed precision

### Out of memory
```python
config.batch_size = 4  # Reduce batch size
config.num_workers = 0  # Reduce dataloader workers
```

### Graphs not updating
- Check console for errors
- Training runs in background thread
- Callbacks use `wx.CallAfter` for thread safety

## Next Steps

1. **Test on Mac:** Run `joe_net_1_train.py` for a few epochs
2. **Verify GUI:** Check that graphs and predictions update
3. **Transfer to PC:** Copy files to RTX 4090 machine
4. **Full training:** Run `train.py` or GUI for 100 epochs
5. **Evaluate:** Check IoU metrics and visual predictions
6. **Iterate:** Adjust hyperparameters, augmentation, architecture

---

**Author:** Joe (with Claude assistance)
**Date:** 2025-11-21
