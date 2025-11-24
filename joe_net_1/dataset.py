"""
Dataset and data augmentation for EM segmentation training

Loads volumes once, then samples random 256x256 patches with augmentation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from cloudvolume import CloudVolume, Bbox
import random
from scipy.ndimage import rotate, gaussian_filter


# CloudVolume paths
CLOUDVOLUME_PATH_IMG = "gs://joe_exp/jarvis/jarvis-1-img/"
CLOUDVOLUME_PATH_NUCLEI = "gs://joe_exp/jarvis/jarvis-1-nuclei/"
CLOUDVOLUME_PATH_MITO = "gs://joe_exp/jarvis/jarvis-1-mito/"

# Volume bounds for training data
VOL_MIN_X, VOL_MIN_Y, VOL_MIN_Z = 5852, 4020, 800
VOL_MAX_X, VOL_MAX_Y, VOL_MAX_Z = 7900, 6068, 900

# Slice size
SLICE_SIZE = 256


def load_volumes(mip=0):
    """
    Load the full training volumes for all three layers.

    Args:
        mip: MIP level to use (0 for full resolution)

    Returns:
        tuple: (img_volume, nuclei_volume, mito_volume) as numpy arrays
    """
    # Create bounding box for the training volume
    scale_factor = 2 ** mip
    min_x = int(VOL_MIN_X / scale_factor)
    min_y = int(VOL_MIN_Y / scale_factor)
    min_z = int(VOL_MIN_Z / scale_factor)
    max_x = int(VOL_MAX_X / scale_factor)
    max_y = int(VOL_MAX_Y / scale_factor)
    max_z = int(VOL_MAX_Z / scale_factor)

    vol_bbox = Bbox((min_x, min_y, min_z), (max_x, max_y, max_z))

    print(f"Loading training volume bbox: {vol_bbox}")
    print(f"Volume size: {max_x - min_x} x {max_y - min_y} x {max_z - min_z}")

    # Load image volume
    print(f"\nLoading image volume from: {CLOUDVOLUME_PATH_IMG}")
    vol_img = CloudVolume(CLOUDVOLUME_PATH_IMG, progress=False)
    vol_img.fill_missing = True
    img_data = vol_img.download(vol_bbox, mip=mip)
    print(f"Image data shape: {img_data.shape}, dtype: {img_data.dtype}")

    # Extract and normalize to uint8
    volume_img = img_data[:, :, :, 0]  # Remove channel dimension
    if volume_img.dtype != np.uint8:
        if volume_img.max() > volume_img.min():
            volume_img = ((volume_img - volume_img.min()) / (volume_img.max() - volume_img.min()) * 255).astype(np.uint8)
        else:
            volume_img = np.zeros_like(volume_img, dtype=np.uint8)
    print(f"Image volume stored as: {volume_img.shape}, {volume_img.dtype}")

    # Load nuclei volume
    print(f"\nLoading nuclei volume from: {CLOUDVOLUME_PATH_NUCLEI}")
    vol_nuclei_cv = CloudVolume(CLOUDVOLUME_PATH_NUCLEI, progress=False)
    vol_nuclei_cv.fill_missing = True
    nuclei_data = vol_nuclei_cv.download(vol_bbox, mip=mip)
    print(f"Nuclei data shape: {nuclei_data.shape}, dtype: {nuclei_data.dtype}")

    # Convert to binary mask (uint8)
    volume_nuclei = (nuclei_data[:, :, :, 0] > 0).astype(np.uint8)
    print(f"Nuclei volume stored as: {volume_nuclei.shape}, {volume_nuclei.dtype}")

    # Load mito volume
    print(f"\nLoading mito volume from: {CLOUDVOLUME_PATH_MITO}")
    vol_mito_cv = CloudVolume(CLOUDVOLUME_PATH_MITO, progress=False)
    vol_mito_cv.fill_missing = True
    mito_data = vol_mito_cv.download(vol_bbox, mip=mip)
    print(f"Mito data shape: {mito_data.shape}, dtype: {mito_data.dtype}")

    # Convert to binary mask (uint8)
    volume_mito = (mito_data[:, :, :, 0] > 0).astype(np.uint8)
    print(f"Mito volume stored as: {volume_mito.shape}, {volume_mito.dtype}")

    # Calculate total memory usage
    total_bytes = volume_img.nbytes + volume_nuclei.nbytes + volume_mito.nbytes
    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal memory used: {total_mb:.2f} MB")

    return volume_img, volume_nuclei, volume_mito


class EMSegmentationDataset(Dataset):
    """
    PyTorch Dataset for EM segmentation

    Samples random 256x256 patches from loaded volumes with augmentation.
    """
    def __init__(self, volume_img, volume_nuclei, volume_mito,
                 is_validation=False, patches_per_epoch=1000, augment=True):
        """
        Args:
            volume_img: Image volume (X, Y, Z) as uint8
            volume_nuclei: Nuclei mask volume (X, Y, Z) as uint8
            volume_mito: Mito mask volume (X, Y, Z) as uint8
            is_validation: If True, sample from validation region (top-right quadrant);
                          if False, sample from training region (other 3/4 of XY)
            patches_per_epoch: Number of random patches to sample per epoch
            augment: Whether to apply data augmentation
        """
        self.volume_img = volume_img
        self.volume_nuclei = volume_nuclei
        self.volume_mito = volume_mito

        self.is_validation = is_validation
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        # Get volume dimensions
        self.vol_x, self.vol_y, self.vol_z = volume_img.shape

        # Calculate midpoints for XY split
        self.mid_x = self.vol_x // 2
        self.mid_y = self.vol_y // 2

        print(f"Dataset initialized:")
        print(f"  Volume shape: {self.vol_x} x {self.vol_y} x {self.vol_z}")
        print(f"  Midpoint: X={self.mid_x}, Y={self.mid_y}")
        print(f"  Mode: {'VALIDATION' if is_validation else 'TRAINING'}")
        print(f"  Region: {'X > {}, Y > {} (top-right quadrant, ~1/4 of area)'.format(self.mid_x, self.mid_y) if is_validation else 'Other 3/4 of XY area'}")
        print(f"  Patches per epoch: {patches_per_epoch}")
        print(f"  Augmentation: {augment}")

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        """
        Sample a random 256x256 patch and apply augmentation

        Returns:
            tuple: (image, target) where
                - image: (1, 256, 256) float32 tensor, normalized to [0, 1]
                - target: (2, 256, 256) float32 tensor, nuclei and mito masks
        """
        # Random Z position (use all Z slices)
        z = random.randint(0, self.vol_z - 1)

        # Random X, Y position based on train/validation split
        max_x = self.vol_x - SLICE_SIZE
        max_y = self.vol_y - SLICE_SIZE

        if self.is_validation:
            # Validation: sample where X > mid_x AND Y > mid_y (top-right quadrant)
            x = random.randint(self.mid_x, max_x)
            y = random.randint(self.mid_y, max_y)
        else:
            # Training: sample where X <= mid_x OR Y <= mid_y (other 3/4 of area)
            # We need to avoid the validation quadrant
            # Strategy: randomly choose one of the 3 training quadrants
            quadrant = random.randint(0, 2)
            if quadrant == 0:
                # Bottom-left: X <= mid_x, Y <= mid_y
                x = random.randint(0, self.mid_x)
                y = random.randint(0, self.mid_y)
            elif quadrant == 1:
                # Top-left: X <= mid_x, Y > mid_y
                x = random.randint(0, self.mid_x)
                y = random.randint(self.mid_y, max_y)
            else:
                # Bottom-right: X > mid_x, Y <= mid_y
                x = random.randint(self.mid_x, max_x)
                y = random.randint(0, self.mid_y)

        # Extract slices
        img_slice = self.volume_img[x:x+SLICE_SIZE, y:y+SLICE_SIZE, z].copy()
        nuclei_slice = self.volume_nuclei[x:x+SLICE_SIZE, y:y+SLICE_SIZE, z].copy()
        mito_slice = self.volume_mito[x:x+SLICE_SIZE, y:y+SLICE_SIZE, z].copy()

        # Apply augmentation if enabled
        if self.augment:
            img_slice, nuclei_slice, mito_slice = self.augment_patch(
                img_slice, nuclei_slice, mito_slice
            )

        # Convert to tensors
        # Image: normalize to [0, 1] and add channel dimension
        img_tensor = torch.from_numpy(img_slice).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)

        # Target: stack nuclei and mito as 2 channels
        nuclei_tensor = torch.from_numpy(nuclei_slice).float()
        mito_tensor = torch.from_numpy(mito_slice).float()
        target_tensor = torch.stack([nuclei_tensor, mito_tensor], dim=0)  # (2, H, W)

        return img_tensor, target_tensor

    def augment_patch(self, img, nuclei, mito):
        """
        Apply data augmentation to a patch

        Augmentations:
        - Random rotation (0, 90, 180, 270 degrees)
        - Random horizontal/vertical flip
        - Random brightness/contrast adjustment
        - Random Gaussian noise

        Args:
            img: (H, W) uint8 image
            nuclei: (H, W) uint8 mask
            mito: (H, W) uint8 mask

        Returns:
            tuple: (img, nuclei, mito) augmented
        """
        # Random rotation (90 degree increments for speed)
        k = random.randint(0, 3)  # 0, 1, 2, 3 -> 0, 90, 180, 270 degrees
        if k > 0:
            img = np.rot90(img, k).copy()
            nuclei = np.rot90(nuclei, k).copy()
            mito = np.rot90(mito, k).copy()

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            nuclei = np.fliplr(nuclei).copy()
            mito = np.fliplr(mito).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            nuclei = np.flipud(nuclei).copy()
            mito = np.flipud(mito).copy()

        # Random brightness adjustment (±20%)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        # Random contrast adjustment (±20%)
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, nuclei, mito


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset...")

    # Load volumes
    volume_img, volume_nuclei, volume_mito = load_volumes(mip=0)

    # Create train dataset (3/4 of XY area)
    train_dataset = EMSegmentationDataset(
        volume_img, volume_nuclei, volume_mito,
        is_validation=False,
        patches_per_epoch=10,
        augment=True
    )

    # Create validation dataset (1/4 of XY area - top-right quadrant)
    val_dataset = EMSegmentationDataset(
        volume_img, volume_nuclei, volume_mito,
        is_validation=True,
        patches_per_epoch=10,
        augment=False
    )

    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    # Sample a few patches from training
    print("\nSampling TRAINING patches...")
    for i in range(3):
        img, target = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Target shape: {target.shape}, dtype: {target.dtype}")
        print(f"  Nuclei pixels: {target[0].sum():.0f}, Mito pixels: {target[1].sum():.0f}")

    # Sample a few patches from validation
    print("\nSampling VALIDATION patches...")
    for i in range(3):
        img, target = val_dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Target shape: {target.shape}, dtype: {target.dtype}")
        print(f"  Nuclei pixels: {target[0].sum():.0f}, Mito pixels: {target[1].sum():.0f}")

    print("\nDataset test passed!")
