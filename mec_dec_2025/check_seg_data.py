#!/usr/bin/env python3
"""
Script to check segmentation data loading from CloudVolume for MEC semantic segmentation.
Loads segmentation labels from gs://joe_exp/mec_training/mec_sem_cls
"""

import numpy as np
from cloudvolume import CloudVolume
import matplotlib.pyplot as plt
import matplotlib

def main():
    matplotlib.use('Agg')  # Non-interactive backend

    # Define the CloudVolume paths
    seg_path = "gs://joe_exp/mec_training/mec_sem_cls"
    img_path = "gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/"

    print(f"Loading segmentation CloudVolume from: {seg_path}")
    cv_seg = CloudVolume(seg_path, mip=0, use_https=False)

    print(f"Loading image CloudVolume from: {img_path}")
    cv_img = CloudVolume(img_path, mip=0, use_https=False)

    # Print volume information
    print("\n" + "="*60)
    print("Segmentation Volume Information:")
    print("="*60)
    print(f"Shape: {cv_seg.shape}")
    print(f"Data type: {cv_seg.dtype}")
    print(f"Resolution: {cv_seg.resolution}")
    print(f"Voxel offset: {cv_seg.voxel_offset}")
    print(f"Bounds: {cv_seg.bounds}")
    print(f"Number of channels: {cv_seg.num_channels}")

    print("\n" + "="*60)
    print("Image Volume Information:")
    print("="*60)
    print(f"Shape: {cv_img.shape}")
    print(f"Data type: {cv_img.dtype}")
    print(f"Resolution: {cv_img.resolution}")
    print(f"Voxel offset: {cv_img.voxel_offset}")
    print(f"Bounds: {cv_img.bounds}")
    print(f"Number of channels: {cv_img.num_channels}")

    # Check coordinate overlap
    print("\n" + "="*60)
    print("Coordinate System Check:")
    print("="*60)
    seg_min = cv_seg.bounds.minpt
    seg_max = cv_seg.bounds.maxpt
    img_min = cv_img.bounds.minpt
    img_max = cv_img.bounds.maxpt

    print(f"Segmentation bounds: {seg_min} to {seg_max}")
    print(f"Image bounds: {img_min} to {img_max}")

    # Check if segmentation is subset of image
    overlap_min = [max(seg_min[i], img_min[i]) for i in range(3)]
    overlap_max = [min(seg_max[i], img_max[i]) for i in range(3)]

    print(f"Overlap region: {overlap_min} to {overlap_max}")
    print(f"Overlap size: {[overlap_max[i] - overlap_min[i] for i in range(3)]}")

    # Load a sample chunk of segmentation data
    print("\n" + "="*60)
    print("Loading sample segmentation data...")
    print("="*60)

    # Load a 512×512×16 sample from the middle of the segmentation volume
    x_start, y_start, z_start = seg_min
    x_size = min(512, seg_max[0] - seg_min[0])
    y_size = min(512, seg_max[1] - seg_min[1])
    z_size = min(16, seg_max[2] - seg_min[2])

    # Take from middle of volume
    x_mid = x_start + (seg_max[0] - x_start) // 2 - x_size // 2
    y_mid = y_start + (seg_max[1] - y_start) // 2 - y_size // 2
    z_mid = z_start + (seg_max[2] - z_start) // 2 - z_size // 2

    print(f"Loading region: [{x_mid}:{x_mid+x_size}, {y_mid}:{y_mid+y_size}, {z_mid}:{z_mid+z_size}]")

    try:
        seg_data = cv_seg[
            x_mid:x_mid+x_size,
            y_mid:y_mid+y_size,
            z_mid:z_mid+z_size
        ]

        print(f"\nSuccessfully loaded segmentation data!")
        print(f"Sample shape: {seg_data.shape}")
        print(f"Sample dtype: {seg_data.dtype}")
        print(f"Unique labels: {np.unique(seg_data)}")

        # Count voxels per class
        print(f"\nClass distribution in sample:")
        unique, counts = np.unique(seg_data, return_counts=True)
        total = seg_data.size
        for label, count in zip(unique, counts):
            pct = 100 * count / total
            print(f"  Label {label:3d}: {count:8d} voxels ({pct:5.2f}%)")

        # Check mitochondria specifically
        mito_count = np.sum(seg_data == 10)
        mito_pct = 100 * mito_count / total
        print(f"\nMitochondria (label 10): {mito_count} voxels ({mito_pct:.2f}%)")

        # Load corresponding image data
        print("\n" + "="*60)
        print("Loading corresponding image data...")
        print("="*60)

        img_data = cv_img[
            x_mid:x_mid+x_size,
            y_mid:y_mid+y_size,
            z_mid:z_mid+z_size
        ]

        print(f"Image data shape: {img_data.shape}")
        print(f"Image data dtype: {img_data.dtype}")
        print(f"Image value range: [{img_data.min()}, {img_data.max()}]")

        # Create visualization: show middle Z slice with image, segmentation, and mitochondria mask
        z_vis = z_size // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Image
        axes[0].imshow(img_data[:, :, z_vis, 0], cmap='gray')
        axes[0].set_title('EM Image')
        axes[0].axis('off')

        # Segmentation (all classes)
        seg_slice = seg_data[:, :, z_vis, 0]
        axes[1].imshow(seg_slice, cmap='tab20', vmin=0, vmax=19)
        axes[1].set_title('All Segmentation Labels')
        axes[1].axis('off')

        # Mitochondria mask only
        mito_mask = (seg_slice == 10).astype(np.uint8)
        axes[2].imshow(img_data[:, :, z_vis, 0], cmap='gray', alpha=0.7)
        axes[2].imshow(mito_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f'Mitochondria Overlay ({mito_mask.sum()} pixels)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('seg_data_check.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to seg_data_check.png")
        plt.close()

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Segmentation data check complete!")
    print("="*60)

if __name__ == "__main__":
    main()
