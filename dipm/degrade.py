"""
Image degradation utilities for DIPM (Degraded Input, Predict Missing).

This module provides functions to apply various degradations to images,
useful for training networks to recognize features in degraded EM data.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import os


def add_noise(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Add grayscale Gaussian noise to an image.

    Args:
        image: Input image as numpy array (values 0-255 for uint8, or 0-1 for float)
        amount: Noise amount from 0.0 (no noise) to 1.0 (maximum noise).
                At 1.0, noise standard deviation equals the full value range.

    Returns:
        Noisy image as numpy array with same dtype as input.
    """
    if amount <= 0:
        return image.copy()

    original_dtype = image.dtype

    # Convert to float for processing
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # Generate grayscale noise (same value for all channels per pixel)
    if image.ndim == 2:
        noise_shape = image.shape
    else:
        noise_shape = image.shape[:2]
    noise = np.random.randn(*noise_shape).astype(np.float32) * amount

    # Broadcast noise to all channels if needed
    if image.ndim == 3:
        noise = noise[:, :, np.newaxis]

    # Add noise and clip to valid range
    noisy = np.clip(img_float + noise, 0.0, 1.0)

    # Convert back to original dtype
    if original_dtype == np.uint8:
        return (noisy * 255).astype(np.uint8)
    else:
        return noisy.astype(original_dtype)


def add_blur(image: np.ndarray, amount: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Args:
        image: Input image as numpy array (values 0-255 for uint8, or 0-1 for float)
        amount: Blur amount from 0.0 (no blur) to 1.0 (heavy blur).
                Maps to Gaussian sigma: 0.0 -> 0, 1.0 -> 10 pixels.

    Returns:
        Blurred image as numpy array with same dtype as input.
    """
    if amount <= 0:
        return image.copy()

    original_dtype = image.dtype
    max_sigma = 10.0
    sigma = amount * max_sigma

    # Apply blur to each channel independently (or just the image if grayscale)
    if image.ndim == 2:
        blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    else:
        # Blur spatial dimensions only, not across channels
        blurred = gaussian_filter(image.astype(np.float32), sigma=(sigma, sigma, 0))

    # Convert back to original dtype
    if original_dtype == np.uint8:
        return np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        return blurred.astype(original_dtype)


def add_blur_patches(image: np.ndarray, quantity: int, radius: float = 40, amount: float = 1.0) -> np.ndarray:
    """
    Apply blur to random circular patches with Gaussian falloff.

    Args:
        image: Input image as numpy array
        quantity: Number of blur patches to add
        radius: Radius of patches (Gaussian sigma for falloff), in pixels
        amount: Blur intensity from 0.0 to 1.0

    Returns:
        Image with blurred patches as numpy array with same dtype as input.
    """
    if quantity <= 0 or amount <= 0:
        return image.copy()

    original_dtype = image.dtype
    h, w = image.shape[:2]

    # Convert to float for blending
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)

    # Create blurred version
    blurred = add_blur(img_float, amount)

    # Create coordinate grids for mask computation
    y_coords, x_coords = np.ogrid[:h, :w]

    # Build combined mask from all patches
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(quantity):
        # Random center point
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        # Gaussian falloff from center
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        patch_mask = np.exp(-dist_sq / (2 * radius ** 2))

        # Combine masks (max blending to avoid over-darkening overlaps)
        mask = np.maximum(mask, patch_mask)

    # Expand mask for broadcasting if needed
    if image.ndim == 3:
        mask = mask[:, :, np.newaxis]

    # Blend original and blurred using mask
    result = img_float * (1 - mask) + blurred * mask

    # Convert back to original dtype
    if original_dtype == np.uint8:
        return (result * 255).astype(np.uint8)
    else:
        return result.astype(original_dtype)


def subsample(image: np.ndarray, scale: int) -> np.ndarray:
    """
    Subsample an image and scale back up to simulate lower resolution.

    Args:
        image: Input image as numpy array
        scale: Integer scale factor (e.g., 2 means half resolution in each dimension)

    Returns:
        Pixelated image at original size with same dtype as input.
    """
    if scale <= 1:
        return image.copy()

    h, w = image.shape[:2]

    # Subsample by taking every nth pixel
    if image.ndim == 2:
        small = image[::scale, ::scale]
    else:
        small = image[::scale, ::scale, :]

    # Scale back up using nearest-neighbor (repeat pixels)
    result = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)

    # Crop to original size (in case dimensions weren't divisible by scale)
    if image.ndim == 2:
        result = result[:h, :w]
    else:
        result = result[:h, :w, :]

    return result


def subsample_patches(image: np.ndarray, quantity: int, size: int = 80, scale: int = 4) -> np.ndarray:
    """
    Apply subsampling to random square patches of the image.

    Args:
        image: Input image as numpy array
        quantity: Number of patches to subsample
        size: Side length of each square patch in pixels
        scale: Subsample scale factor within each patch

    Returns:
        Image with subsampled patches at original size with same dtype as input.
    """
    if quantity <= 0 or scale <= 1:
        return image.copy()

    result = image.copy()
    h, w = image.shape[:2]

    for _ in range(quantity):
        # Random top-left corner for patch
        x = np.random.randint(0, max(1, w - size + 1))
        y = np.random.randint(0, max(1, h - size + 1))

        # Extract patch
        if image.ndim == 2:
            patch = result[y:y+size, x:x+size]
        else:
            patch = result[y:y+size, x:x+size, :]

        # Subsample the patch
        subsampled_patch = subsample(patch, scale)

        # Place back
        if image.ndim == 2:
            result[y:y+size, x:x+size] = subsampled_patch
        else:
            result[y:y+size, x:x+size, :] = subsampled_patch

    return result


def mask(image: np.ndarray, amount: float, rows_cols: int = 8, color: float = 0.5) -> np.ndarray:
    """
    Mask random grid cells with a solid color.

    Args:
        image: Input image as numpy array
        amount: Probability (0.0 to 1.0) of masking each cell
        rows_cols: Number of rows and columns in the grid
        color: Fill color from 0.0 (black) to 1.0 (white)

    Returns:
        Image with masked cells at same dtype as input.
    """
    if amount <= 0 or rows_cols <= 0:
        return image.copy()

    result = image.copy()
    h, w = image.shape[:2]

    cell_h = h // rows_cols
    cell_w = w // rows_cols

    # Determine fill value based on dtype
    if image.dtype == np.uint8:
        fill_value = int(color * 255)
    else:
        fill_value = color

    for row in range(rows_cols):
        for col in range(rows_cols):
            if np.random.random() < amount:
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < rows_cols - 1 else h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < rows_cols - 1 else w

                if image.ndim == 2:
                    result[y1:y2, x1:x2] = fill_value
                else:
                    result[y1:y2, x1:x2, :] = fill_value

    return result


# Configuration for random_combo: easy to tweak thresholds and parameters
# Each entry: (min_difficulty, max_probability, param_func)
# - min_difficulty: difficulty level at which this effect can start appearing
# - max_probability: probability of applying at max difficulty (scales linearly from min_difficulty)
# - param_func: function(difficulty) -> dict of parameters for the degradation
DEGRADATION_CONFIG = {
    'noise': (
        0.0,   # min_difficulty: can appear at any difficulty
        0.8,   # max_probability: 80% chance at difficulty=1
        lambda d: {'amount': d * 0.4}  # noise amount scales 0 to 0.4
    ),
    'blur': (
        0.1,   # min_difficulty: starts at 0.1
        0.5,   # max_probability: 50% chance at difficulty=1
        lambda d: {'amount': d * 0.3}  # blur amount scales 0.03 to 0.3
    ),
    'blur_patches': (
        0.2,   # min_difficulty: starts at 0.2
        0.6,   # max_probability: 60% chance at difficulty=1
        lambda d: {
            'quantity': int(1 + d * 10),  # 1-11 patches
            'radius': 20 + d * 40,        # radius 28-60
            'amount': 0.5 + d * 0.5       # blur amount 0.6-1.0
        }
    ),
    'subsample': (
        0.3,   # min_difficulty: starts at 0.3
        0.4,   # max_probability: 40% chance at difficulty=1
        lambda d: {'scale': int(2 + d * 6)}  # scale 2-8
    ),
    'subsample_patches': (
        0.2,   # min_difficulty: starts at 0.2
        0.5,   # max_probability: 50% chance at difficulty=1
        lambda d: {
            'quantity': int(1 + d * 8),   # 1-9 patches
            'size': int(40 + d * 60),     # size 52-100
            'scale': int(2 + d * 6)       # scale 2-8
        }
    ),
    'mask': (
        0.5,   # min_difficulty: starts at 0.5
        0.5,   # max_probability: 50% chance at difficulty=1
        lambda d: {
            'amount': d * 0.4,            # mask probability 0-0.4
            'rows_cols': int(12 - d * 8), # grid 12x12 to 4x4 (bigger patches = harder)
            'color': np.random.random()   # random gray level
        }
    ),
}


def random_combo(image: np.ndarray, difficulty: float) -> np.ndarray:
    """
    Apply a random combination of degradations based on difficulty.

    Args:
        image: Input image as numpy array
        difficulty: Difficulty level from 0.0 (no degradation) to 1.0 (heavy degradation)

    Returns:
        Degraded image with same dtype as input.
    """
    if difficulty <= 0:
        return image.copy()

    difficulty = min(1.0, difficulty)  # clamp to [0, 1]
    result = image.copy()

    # Map of degradation names to functions
    degrade_funcs = {
        'noise': add_noise,
        'blur': add_blur,
        'blur_patches': add_blur_patches,
        'subsample': subsample,
        'subsample_patches': subsample_patches,
        'mask': mask,
    }

    for name, (min_diff, max_prob, param_func) in DEGRADATION_CONFIG.items():
        if difficulty < min_diff:
            continue

        # Scale probability from 0 at min_difficulty to max_prob at difficulty=1
        prob_scale = (difficulty - min_diff) / (1.0 - min_diff) if min_diff < 1.0 else 1.0
        probability = prob_scale * max_prob

        if np.random.random() < probability:
            params = param_func(difficulty)
            result = degrade_funcs[name](result, **params)

    return result


def load_image(path: str) -> np.ndarray:
    """Load an image file as a numpy array."""
    img = Image.open(path)
    return np.array(img)


def save_image(image: np.ndarray, path: str) -> None:
    """Save a numpy array as an image file."""
    img = Image.fromarray(image)
    img.save(path)


if __name__ == "__main__":
    # Test degradation functions with different levels
    input_path = "sampleImage.png"
    output_dir = "degrade_tests"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load sample image
    print(f"Loading {input_path}...")
    image = load_image(input_path)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    test_levels = [0.0, 0.1, 0.25, 0.5, 1.0]

    # Test noise
    print("\nTesting noise...")
    for level in test_levels:
        output_path = os.path.join(output_dir, f"noise_{level:.2f}.png")
        noisy_image = add_noise(image, level)
        save_image(noisy_image, output_path)
        print(f"  Saved {output_path}")

    # Test blur
    print("\nTesting blur...")
    for level in test_levels:
        output_path = os.path.join(output_dir, f"blur_{level:.2f}.png")
        blurred_image = add_blur(image, level)
        save_image(blurred_image, output_path)
        print(f"  Saved {output_path}")

    # Test blur patches
    print("\nTesting blur patches...")
    patch_counts = [1, 3, 5, 10, 20]
    for count in patch_counts:
        output_path = os.path.join(output_dir, f"blur_patches_{count:02d}.png")
        patched_image = add_blur_patches(image, quantity=count, radius=40, amount=1.0)
        save_image(patched_image, output_path)
        print(f"  Saved {output_path}")

    # Test subsample
    print("\nTesting subsample...")
    scale_factors = [1, 2, 4, 8, 16]
    for scale in scale_factors:
        output_path = os.path.join(output_dir, f"subsample_{scale:02d}x.png")
        subsampled_image = subsample(image, scale)
        save_image(subsampled_image, output_path)
        print(f"  Saved {output_path}")

    # Test subsample patches
    print("\nTesting subsample patches...")
    patch_counts = [1, 3, 5, 10, 20]
    for count in patch_counts:
        output_path = os.path.join(output_dir, f"subsample_patches_{count:02d}.png")
        patched_image = subsample_patches(image, quantity=count, size=80, scale=8)
        save_image(patched_image, output_path)
        print(f"  Saved {output_path}")

    # Test mask
    print("\nTesting mask...")
    mask_amounts = [0.1, 0.25, 0.5, 0.75, 1.0]
    for amt in mask_amounts:
        output_path = os.path.join(output_dir, f"mask_{amt:.2f}.png")
        masked_image = mask(image, amount=amt, rows_cols=8, color=0.5)
        save_image(masked_image, output_path)
        print(f"  Saved {output_path}")

    # Test random_combo
    print("\nTesting random_combo...")
    difficulties = np.arange(0.0, 1.01, 0.1)
    for diff in difficulties:
        output_path = os.path.join(output_dir, f"combo_{diff:.2f}.png")
        combo_image = random_combo(image, difficulty=diff)
        save_image(combo_image, output_path)
        print(f"  Saved {output_path}")

    print("\nDone!")
