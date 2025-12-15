#!/usr/bin/env python3
"""
Script to check data loading from CloudVolume for MEC December 2025 experiment.
Loads highest-resolution (MIP 0) data from gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/
"""

import numpy as np
from cloudvolume import CloudVolume

def main():
    # Define the CloudVolume path
    cv_path = "gs://hammerschmith-mec-central/img-cutouts/cutout0-45nm/"

    print(f"Loading CloudVolume from: {cv_path}")
    print(f"MIP level: 0 (highest resolution)")

    # Create CloudVolume object
    cv = CloudVolume(cv_path, mip=0, use_https=False)

    # Print volume information
    print("\n" + "="*60)
    print("CloudVolume Information:")
    print("="*60)
    print(f"Shape: {cv.shape}")
    print(f"Data type: {cv.dtype}")
    print(f"Resolution: {cv.resolution}")
    print(f"Voxel offset: {cv.voxel_offset}")
    print(f"Bounds: {cv.bounds}")
    print(f"Available MIP levels: {cv.available_mips}")
    print(f"Number of channels: {cv.num_channels}")

    # Load a sample chunk of data
    print("\n" + "="*60)
    print("Loading sample data chunk...")
    print("="*60)

    # Get the bounds
    x_start, y_start, z_start = cv.bounds.minpt
    x_end, y_end, z_end = cv.bounds.maxpt

    # Load a small sample (first 256x256x32 voxels, or smaller if volume is smaller)
    sample_x = min(256, x_end - x_start)
    sample_y = min(256, y_end - y_start)
    sample_z = min(32, z_end - z_start)

    print(f"Attempting to load: [{x_start}:{x_start+sample_x}, {y_start}:{y_start+sample_y}, {z_start}:{z_start+sample_z}]")

    try:
        data = cv[
            x_start:x_start+sample_x,
            y_start:y_start+sample_y,
            z_start:z_start+sample_z
        ]

        print(f"\nSuccessfully loaded data!")
        print(f"Sample shape: {data.shape}")
        print(f"Sample dtype: {data.dtype}")
        print(f"Value range: [{data.min()}, {data.max()}]")
        print(f"Mean: {data.mean():.2f}")
        print(f"Std: {data.std():.2f}")

        # Print statistics for first Z slice
        if data.shape[2] > 0:
            slice_data = data[:, :, 0, :]
            print(f"\nFirst Z-slice statistics:")
            print(f"  Shape: {slice_data.shape}")
            print(f"  Range: [{slice_data.min()}, {slice_data.max()}]")
            print(f"  Mean: {slice_data.mean():.2f}")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Data check complete!")
    print("="*60)

if __name__ == "__main__":
    main()
