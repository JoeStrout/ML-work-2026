# Joe Net 1 - Segmentation & Semantic Labeling Network

## Project Overview

This is a machine learning project for training a segmentation and semantic labeling network on electron microscopy data. The goal is to predict nuclei and mitochondria labels from grayscale EM images.

## Current Status

### Completed
- ✅ Set up `pytorch-wx` micromamba environment with PyTorch, wxPython, and cloud-volume
- ✅ Created `joe_net_1.py` - a wxPython GUI for visualizing training data
- ✅ Implemented full volume loading (~1.2 GB) for efficient slice extraction
- ✅ Built compositing system to overlay nuclei (red) and mito (cyan) on grayscale images
- ✅ Added random slice viewer with dice button for exploring training data
- ✅ **Implemented complete training system (2025-11-21):**
  - U-Net architecture with pre-activation residual blocks and ConvNeXt-style bottleneck
  - Data pipeline with augmentation
  - Combined BCE + Dice loss with comprehensive metrics
  - Callback-based trainer supporting both headless and GUI modes
  - wxPython training dashboard with live graphs and predictions
  - Full documentation and guides

### Data Details

**CloudVolume Sources:**
- Image layer: `gs://joe_exp/jarvis/jarvis-1-img/`
- Nuclei layer: `gs://joe_exp/jarvis/jarvis-1-nuclei/`
- Mito layer: `gs://joe_exp/jarvis/jarvis-1-mito/`

**Training Volume Bounds:**
- X: 5852 to 7900 (2048 pixels)
- Y: 4020 to 6068 (2048 pixels)
- Z: 800 to 900 (100 slices)
- Total: ~419 million voxels

**Memory Usage:**
- All 3 layers (uint8): ~1.2 GB
- Nuclei and mito stored as binary masks (uint8) for efficiency
- Image layer normalized to uint8

**Slice Parameters:**
- Fixed size: 256x256 pixels
- Single Z slice per sample
- Compositing done on-the-fly for display only

## Architecture & Code Structure

### Key Functions

**`load_volumes(mip=0)`**
- Downloads entire training volume from CloudVolume
- Converts nuclei/mito to binary masks immediately to save memory
- Normalizes image data to uint8
- Stores in global variables for fast access

**`extract_slice(min_x, min_y, z, size=256)`**
- Extracts 256x256 slice from loaded volumes
- Takes top-left corner coordinates (min_x, min_y) and Z position
- Returns tuple: (img_slice, nuclei_slice, mito_slice)
- All coordinates in original world space

**`composite_images(img_data, nuclei_data, mito_data)`**
- Creates RGB visualization of grayscale + labels
- Uses float32 for calculations to avoid overflow
- Tints: nuclei = red, mito = cyan
- Returns PIL Image for display

### GUI Components

**MainFrame**
- Loads all three volumes at startup
- Displays 256x256 composited slices at fixed size
- Random slice button (🎲) for exploring data
- Status bar shows current slice coordinates

**ImagePanel**
- Fixed 256x256 display (no scaling)
- Updates efficiently with new slices

## Environment Setup

```bash
# Activate environment
micromamba activate pytorch-wx

# Run the viewer
python joe_net_1.py
```

**Installed packages:**
- Python 3.11
- PyTorch 2.2.2 (CPU version for Intel Mac)
- torchvision 0.17.2
- wxPython 4.2.4
- cloud-volume (for accessing GCS neuroglancer volumes)

## Next Steps: Neural Network Training

### Architecture Design
- **Input:** 256x256 grayscale EM image (1 channel)
- **Output:** Two binary masks (nuclei and mito), likely as 2-channel output
- **Architecture options:**
  - U-Net (classic choice for segmentation)
  - ResNet-based encoder-decoder
  - Lightweight CNN for fast iteration

### Training Infrastructure

**Data Pipeline:**
- Random slice sampling from loaded volume
- Online data augmentation (rotation, flip, brightness, contrast)
- Split volume into train/val/test (e.g., by Z slices or spatial regions)
- PyTorch Dataset/DataLoader implementation

**Loss Function:**
- Binary cross-entropy for each channel
- Consider: Dice loss, Focal loss for imbalanced classes
- May need class weights (mito/nuclei are sparse)

**Training Loop:**
- Standard PyTorch training with Adam optimizer
- Learning rate scheduling
- Checkpoint best models
- TensorBoard or simple logging for metrics

**Evaluation Metrics:**
- Pixel accuracy
- IoU (Intersection over Union) per class
- Precision/Recall/F1
- Visualize predictions vs. ground truth

**Visualization During Training:**
- Could extend GUI to show:
  - Current training slice
  - Model predictions in real-time
  - Training loss curves
  - Side-by-side: input, ground truth, prediction

### Technical Considerations

**Memory Management:**
- Full volume fits in RAM (1.2 GB), so can keep loaded
- Batch size limited by available RAM during training
- Consider gradient accumulation if needed

**Data Balance:**
- Nuclei and mito are sparse (most pixels are background)
- May need weighted loss or hard negative mining
- Random sampling should see diverse examples

**Validation Strategy:**
- Hold out Z slices for validation (e.g., Z=850-900)
- Or spatial holdout (e.g., one corner of XY volume)
- Ensure no leakage between train/val

**Inference:**
- Sliding window over full volume for prediction
- Could add prediction visualization to GUI
- Export predictions back to CloudVolume format

## File Structure

```
wx-dev/
├── joe_net_1.py          # Original data viewer GUI
├── joe_net_1_train.py    # Training GUI with live graphs and predictions
├── model.py              # U-Net architecture (5.1M params)
├── dataset.py            # Data loading and augmentation
├── utils.py              # Loss functions, metrics, visualization
├── trainer.py            # Core training logic with callbacks
├── train.py              # Standalone headless training script
│
├── CLAUDE.md             # This file (project overview)
├── NET_DESIGN.md         # Architecture design document
├── TRAINING_GUIDE.md     # Configuration and training guide
├── TRAINING_GUI.md       # GUI training detailed guide
├── README_TRAINING.md    # Quick start guide
│
└── slab_finder_wx.py     # Reference example (XZ viewer)
```

## Notes

- Currently using MIP level 0 (full resolution)
- Compositing only for visualization; training will use raw data
- Intel Mac, so CPU-only PyTorch (no GPU acceleration)
- CloudVolume handles GCS authentication automatically
- wxPython window title shows current slice coordinates
- Print statements in console show data loading progress

## Development History

1. Set up micromamba environment with PyTorch + wxPython
2. Created basic XY slice viewer based on slab_finder_wx.py
3. Added multi-volume support (image, nuclei, mito)
4. Implemented RGB compositing with color tints
5. Fixed overflow issues by using float32 during compositing
6. Switched to full-volume loading for training efficiency
7. Added random slice viewer with dice button

---

**Author:** Joe (with Claude assistance)
**Date:** 2025-11-20
**Environment:** Intel Mac, MacOS 13.6, Python 3.11
