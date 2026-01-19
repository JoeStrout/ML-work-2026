"""
Inference script for multi-class semantic segmentation.

Applies a trained model to a CloudVolume, processing one Z-slice at a time
with overlapping patches. Trims edge predictions to avoid boundary artifacts.
"""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cloudvolume as cv
from tqdm import tqdm
import time

from multi_unet_arch import ConvNeXtUNet


def generate_patch_positions(width, height, patch_size, stride):
    """
    Generate (x, y) positions for sliding window patches.

    Args:
        width: Image width
        height: Image height
        patch_size: Size of input patches
        stride: Stride between patches

    Returns:
        List of (x, y) tuples for patch top-left corners
    """
    positions = []

    # Generate regular grid
    y = 0
    while y + patch_size <= height:
        x = 0
        while x + patch_size <= width:
            positions.append((x, y))
            x += stride
        y += stride

    return positions


def extract_patches(image_slice, positions, patch_size):
    """
    Extract patches from a 2D image at given positions.

    Args:
        image_slice: 2D numpy array [H, W]
        positions: List of (x, y) positions
        patch_size: Size of patches to extract

    Returns:
        Tensor of patches [N, 1, patch_size, patch_size]
    """
    patches = []
    for x, y in positions:
        patch = image_slice[y:y+patch_size, x:x+patch_size]
        patches.append(patch)

    patches = np.stack(patches, axis=0)  # [N, H, W]
    patches = torch.from_numpy(patches).unsqueeze(1).float()  # [N, 1, H, W]

    # Normalize to [0, 1] to match training preprocessing
    patches = patches / 255.0

    return patches


def trim_predictions(predictions, trim_size):
    """
    Trim predictions to center region, removing edge artifacts.

    Args:
        predictions: Tensor [N, H, W] of class predictions
        trim_size: Pixels to trim from each edge

    Returns:
        Trimmed predictions [N, H-2*trim, W-2*trim]
    """
    if trim_size > 0:
        return predictions[:, trim_size:-trim_size, trim_size:-trim_size]
    return predictions


def assemble_predictions(pred_patches, positions, output_shape, trim_size):
    """
    Assemble trimmed patch predictions into full image.

    Args:
        pred_patches: List of prediction patches [N_i, trim_H, trim_W]
        positions: List of (x, y) positions for each patch
        output_shape: (H, W) of output image
        trim_size: Trim size used

    Returns:
        Assembled prediction image [H, W]
    """
    output = np.full(output_shape, 255, dtype=np.uint8)  # Initialize with 255 (void)

    trim_patch_size = pred_patches[0].shape[-1]  # Should be patch_size - 2*trim_size

    for pred, (x, y) in zip(pred_patches, positions):
        # Position in output corresponds to trimmed region
        out_x = x + trim_size
        out_y = y + trim_size

        # Write trimmed prediction to output
        output[out_y:out_y+trim_patch_size, out_x:out_x+trim_patch_size] = pred

    return output


@hydra.main(version_base=None, config_path=".", config_name="multi_seg_infer_config")
def main(cfg: DictConfig):
    print("="*80)
    print("Multi-Class Semantic Segmentation Inference")
    print("="*80)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {cfg.checkpoint}")
    model = ConvNeXtUNet(
        num_classes=cfg.num_classes,
        num_coarse_groups=cfg.num_coarse_groups,
        pretrained_path=None,
        freeze_encoder=False
    )

    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    val_miou_str = f"{checkpoint['val_miou']:.4f}" if 'val_miou' in checkpoint else 'N/A'
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (mIoU: {val_miou_str})")

    # Load input CloudVolume
    print(f"\nLoading input CloudVolume: {cfg.input_cv_path}")
    input_vol = cv.CloudVolume(
        cfg.input_cv_path,
        mip=cfg.input_mip,
        use_https=False,
        fill_missing=True,
        progress=False
    )

    # Get volume bounds from config
    x_start, x_end = cfg.x_start, cfg.x_end
    y_start, y_end = cfg.y_start, cfg.y_end
    z_start, z_end = cfg.z_start, cfg.z_end

    width = x_end - x_start
    height = y_end - y_start
    depth = z_end - z_start

    print(f"Processing volume: X[{x_start}:{x_end}], Y[{y_start}:{y_end}], Z[{z_start}:{z_end}]")
    print(f"Dimensions: {width} x {height} x {depth}")

    # Validate stride
    expected_stride = cfg.patch_size - 2 * cfg.trim_size
    if cfg.stride != expected_stride:
        print(f"WARNING: Configured stride ({cfg.stride}) doesn't match expected ({expected_stride})")

    # Generate patch positions for XY slices
    positions = generate_patch_positions(width, height, cfg.patch_size, cfg.stride)
    num_patches_per_slice = len(positions)
    print(f"\nPatch configuration:")
    print(f"  Input patch size: {cfg.patch_size}x{cfg.patch_size}")
    print(f"  Trim size: {cfg.trim_size} pixels per edge")
    print(f"  Output per patch: {cfg.stride}x{cfg.stride}")
    print(f"  Stride: {cfg.stride}")
    print(f"  Patches per slice: {num_patches_per_slice}")
    print(f"  Batch size: {cfg.batch_size}")

    # Create output CloudVolume
    print(f"\nCreating output CloudVolume: {cfg.output_cv_path}")

    # Always create fresh CloudVolume (delete if exists)
    try:
        existing_vol = cv.CloudVolume(cfg.output_cv_path, use_https=False)
        print(f"WARNING: Output CloudVolume already exists - deleting and recreating...")
        existing_vol.delete()
    except:
        pass

    # Create new CloudVolume
    info = cv.CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint8',
        encoding='raw',
        resolution=input_vol.resolution,
        voxel_offset=[x_start, y_start, z_start],
        volume_size=[width, height, depth],
        chunk_size=cfg.output_chunk_size,
    )

    output_vol = cv.CloudVolume(
        cfg.output_cv_path,
        info=info,
        use_https=False,
        fill_missing=False,
        progress=False
    )
    output_vol.commit_info()

    print(f"Created new CloudVolume with shape {width}x{height}x{depth}")
    print(f"Voxel offset: [{x_start}, {y_start}, {z_start}]")
    print(f"Chunk size: {cfg.output_chunk_size}")

    # Process Z slices in chunks aligned with CloudVolume chunk size
    chunk_z_size = cfg.output_chunk_size[2]  # Z chunk size (e.g., 4)
    num_z_chunks = (depth + chunk_z_size - 1) // chunk_z_size

    print(f"\nProcessing {depth} Z-slices in {num_z_chunks} chunks of {chunk_z_size}...")
    start_time = time.time()

    with torch.no_grad():
        for chunk_idx in tqdm(range(num_z_chunks), desc="Z-chunks"):
            # Determine Z range for this chunk
            z_start_chunk = z_start + chunk_idx * chunk_z_size
            z_end_chunk = min(z_start_chunk + chunk_z_size, z_end)
            num_slices_in_chunk = z_end_chunk - z_start_chunk

            # Accumulate output slices for this chunk
            output_slices = []

            for z_offset in range(num_slices_in_chunk):
                z = z_start_chunk + z_offset

                # Load input slice
                input_slice = input_vol[x_start:x_end, y_start:y_end, z:z+1]
                input_slice = input_slice[:, :, 0, 0].T  # [H, W]

                # Extract all patches for this slice
                patch_tensor = extract_patches(input_slice, positions, cfg.patch_size)

                # Process in batches
                all_predictions = []
                for batch_start in range(0, len(positions), cfg.batch_size):
                    batch_end = min(batch_start + cfg.batch_size, len(positions))
                    batch_patches = patch_tensor[batch_start:batch_end].to(device)

                    # Forward pass
                    logits_fine, logits_coarse = model(batch_patches)

                    # Get predictions
                    preds = torch.argmax(logits_fine, dim=1)  # [B, H, W]

                    # Trim edges
                    preds_trimmed = trim_predictions(preds, cfg.trim_size)  # [B, H-2t, W-2t]

                    # Move to CPU and convert to numpy
                    preds_numpy = preds_trimmed.cpu().numpy().astype(np.uint8)
                    all_predictions.append(preds_numpy)

                # Concatenate all batch predictions
                all_predictions = np.concatenate(all_predictions, axis=0)

                # Assemble into full slice
                output_slice = assemble_predictions(
                    all_predictions,
                    positions,
                    (height, width),
                    cfg.trim_size
                )
                output_slices.append(output_slice)

            # Stack slices and write chunk to CloudVolume
            # CloudVolume expects [X, Y, Z, C] ordering
            # output_slices are [height, width] = [Y_size, X_size]
            output_chunk = np.stack(output_slices, axis=2)  # [Y, X, Z]
            output_chunk = np.transpose(output_chunk, (1, 0, 2))  # [X, Y, Z]
            output_chunk = output_chunk[:, :, :, np.newaxis]  # [X, Y, Z, 1]

            output_vol[x_start:x_end, y_start:y_end, z_start_chunk:z_end_chunk] = output_chunk

    elapsed = time.time() - start_time
    print(f"\nInference complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/depth:.2f}s per slice)")
    print(f"Output saved to: {cfg.output_cv_path}")

    # Print coverage statistics
    output_width = (num_patches_per_slice ** 0.5) * cfg.stride  # Approximate
    coverage_x = output_width / width * 100
    coverage_y = coverage_x
    print(f"\nApproximate coverage: {coverage_x:.1f}% in X and Y")
    print(f"Note: Outer {cfg.trim_size}-pixel border may not have predictions")


if __name__ == "__main__":
    main()
