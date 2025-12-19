# Plan: Hierarchical supervision + ignore “uncertain” voxels (PyTorch)

## Goals
- Train a voxel-wise segmentation network on **12 fine classes**.
- Add **coarse (hierarchical) supervision** so the model learns broad categories early and provides interpretable group-level confidence.
- Voxels labeled **“uncertain”** by human tracers should be **excluded from *all* losses and metrics** (no gradient contribution), while still being present in the input.

---

## Label set

### Fine classes (12-way softmax)
Index these as `0..11` with the following ordering:
0. extracellular space
1. tear
2. dendrite
3. axon
4. soma (note: 0.00% of dataset - not used by labelers, but kept in label set)
5. glia
6. myelin
7. myelin inner tongue
8. myelin outer tongue
9. nucleus
10. mitochondria
11. fat globule

### Special label
- **uncertain** (void/ignore): store as `IGNORE_INDEX = -1` in the fine label tensor.
  - Do **not** include “uncertain” as a 13th softmax class.

---

## Coarse hierarchy (auxiliary heads)
Use one shared backbone and multiple output heads:

### Coarse group head (example: 5 bits)
A sigmoid (multi-label) head with these groups:
- **neuron_part**: {dendrite, axon, soma}
- **glia**: {glia}
- **myelin_related**: {myelin, myelin inner tongue, myelin outer tongue}
- **organelle**: {nucleus, mitochondria, fat globule}
- **non_tissue**: {extracellular space, tear}

Notes:
- Coarse groups are *derived deterministically* from the fine label.
- If a voxel is `IGNORE_INDEX` (uncertain), **ignore it for coarse losses too** (do not assign coarse targets).

Optional later: add additional levels/heads (e.g., split `myelin_related` into `myelin` vs `{inner, outer}`), but start with one coarse head.

---

## Model outputs
For a batch of 2D patches:
- `logits_fine`: shape **[B, 12, H, W]**
- `logits_coarse`: shape **[B, 5, H, W]** (or [B, K, H, W])

---

## Target tensors
- `y_fine`: shape **[B, H, W]**
  - values in `0..11` for known pixels
  - value `-1` for uncertain pixels

- `valid_mask = (y_fine != IGNORE_INDEX)`: shape **[B, H, W]** boolean

- `y_coarse`: derived from `y_fine` (only meaningful where `valid_mask=True`):
  - shape **[B, 5, H, W]** of {0,1}

### Mapping fine → coarse
Create a lookup table of size 12 → 5 bits, e.g.:

| Fine class | neuron_part | glia | myelin_related | organelle | non_tissue |
|---|---:|---:|---:|---:|---:|
| extracellular space | 0 | 0 | 0 | 0 | 1 |
| tear | 0 | 0 | 0 | 0 | 1 |
| dendrite | 1 | 0 | 0 | 0 | 0 |
| axon | 1 | 0 | 0 | 0 | 0 |
| soma | 1 | 0 | 0 | 0 | 0 |
| glia | 0 | 1 | 0 | 0 | 0 |
| myelin | 0 | 0 | 1 | 0 | 0 |
| myelin inner tongue | 0 | 0 | 1 | 0 | 0 |
| myelin outer tongue | 0 | 0 | 1 | 0 | 0 |
| nucleus | 0 | 0 | 0 | 1 | 0 |
| mitochondria | 0 | 0 | 0 | 1 | 0 |
| fat globule | 0 | 0 | 0 | 1 | 0 |

Implementation options:
- `coarse_lut = torch.tensor([
    # extracellular space (index 0)
    [0,0,0,0,1],
    # tear (index 1)
    [0,0,0,0,1],
    # dendrite (index 2)
    [1,0,0,0,0],
    # axon (index 3)
    [1,0,0,0,0],
    # soma (index 4)
    [1,0,0,0,0],
    # glia (index 5)
    [0,1,0,0,0],
    # myelin (index 6)
    [0,0,1,0,0],
    # myelin inner tongue (index 7)
    [0,0,1,0,0],
    # myelin outer tongue (index 8)
    [0,0,1,0,0],
    # nucleus (index 9)
    [0,0,0,1,0],
    # mitochondria (index 10)
    [0,0,0,1,0],
    # fat globule (index 11)
    [0,0,0,1,0],
], dtype=torch.float32)` of shape [12,5]
- For valid voxels: `y_coarse = coarse_lut[y_fine_valid]` then reshape back.

---

## Losses (mask out “uncertain” voxels everywhere)

### Fine loss: Cross-entropy with ignore index
Use per-pixel loss, then apply `valid_mask` and normalize by number of valid pixels.

PyTorch pattern:
- `ce = F.cross_entropy(logits_fine, y_fine, ignore_index=IGNORE_INDEX, reduction='none')`
  - `ce` has shape [B, H, W]
- `loss_fine = (ce * valid_mask).sum() / valid_mask.sum().clamp_min(1)`

### Coarse loss: BCEWithLogitsLoss masked by valid pixels
Compute elementwise BCE, reduce across coarse channels, apply same pixel mask.

PyTorch pattern:
- `bce = F.binary_cross_entropy_with_logits(logits_coarse, y_coarse, reduction='none')`
  - `bce` shape [B, 5, H, W]
- reduce per pixel: `bce_pixel = bce.mean(dim=1)` → shape [B, H, W]
- `loss_coarse = (bce_pixel * valid_mask).sum() / valid_mask.sum().clamp_min(1)`

### Total loss
- `loss = loss_fine + lambda_coarse * loss_coarse`

Suggested starting weight:
- `lambda_coarse = 0.2` (tune 0.05–0.5 depending on behavior)

Critical rule:
- If a voxel is labeled uncertain (`-1`), it must contribute **zero** loss to both fine and coarse terms.

---

## Metrics (also ignore uncertain)
All evaluation (accuracy, IoU/Dice, confusion matrices) should be computed only on pixels where `valid_mask=True`.

---

## Training loop checklist
1. Forward pass → `logits_fine`, `logits_coarse`
2. Build `valid_mask` from `y_fine`
3. Build `y_coarse` from `y_fine` using LUT (values don’t matter where invalid, but simplest is to fill zeros then mask loss)
4. Compute `loss_fine` with ignore_index and/or explicit masking
5. Compute `loss_coarse` with explicit masking
6. Combine losses, backprop
7. Compute masked metrics for logging

---

## Minimal code skeleton (PyTorch)
```python
import torch
import torch.nn.functional as F

IGNORE_INDEX = -1

# coarse groups (K=5): neuron_part, glia, myelin_related, organelle, non_tissue
coarse_lut = torch.tensor([
    # extracellular space (index 0)
    [0,0,0,0,1],
    # tear (index 1)
    [0,0,0,0,1],
    # dendrite (index 2)
    [1,0,0,0,0],
    # axon (index 3)
    [1,0,0,0,0],
    # soma (index 4)
    [1,0,0,0,0],
    # glia (index 5)
    [0,1,0,0,0],
    # myelin (index 6)
    [0,0,1,0,0],
    # myelin inner tongue (index 7)
    [0,0,1,0,0],
    # myelin outer tongue (index 8)
    [0,0,1,0,0],
    # nucleus (index 9)
    [0,0,0,1,0],
    # mitochondria (index 10)
    [0,0,0,1,0],
    # fat globule (index 11)
    [0,0,0,1,0],
], dtype=torch.float32)


def make_coarse_targets(y_fine: torch.Tensor) -> torch.Tensor:
    """y_fine: [B, H, W] with values 0..11 or -1.
    Returns y_coarse: [B, K, H, W] float {0,1}.
    """
    B, H, W = y_fine.shape
    K = coarse_lut.shape[1]

    y_coarse = torch.zeros((B, K, H, W), device=y_fine.device, dtype=torch.float32)

    valid = (y_fine != IGNORE_INDEX)
    if valid.any():
        yv = y_fine[valid].long()                 # [N]
        cv = coarse_lut.to(y_fine.device)[yv]     # [N, K]
        # scatter back
        # y_coarse[:, :, ...][valid] expects [N, K] if we permute
        y_coarse_perm = y_coarse.permute(0,2,3,1)  # [B, H, W, K]
        y_coarse_perm[valid] = cv
        y_coarse = y_coarse_perm.permute(0,3,1,2)  # [B, K, H, W]

    return y_coarse


def compute_losses(logits_fine, logits_coarse, y_fine, lambda_coarse=0.2):
    """logits_fine: [B, 12, H, W]
       logits_coarse: [B, 5, H, W]
       y_fine: [B, H, W] in 0..11 or -1
    """
    valid = (y_fine != IGNORE_INDEX).float()  # [B, H, W]
    denom = valid.sum().clamp_min(1.0)

    # Fine CE (ignore_index also works, but we still normalize explicitly)
    ce = F.cross_entropy(logits_fine, y_fine, ignore_index=IGNORE_INDEX, reduction='none')
    loss_fine = (ce * valid).sum() / denom

    # Coarse BCE
    y_coarse = make_coarse_targets(y_fine)
    bce = F.binary_cross_entropy_with_logits(logits_coarse, y_coarse, reduction='none')
    bce_pixel = bce.mean(dim=1)  # [B, H, W]
    loss_coarse = (bce_pixel * valid).sum() / denom

    loss = loss_fine + lambda_coarse * loss_coarse
    return loss, loss_fine, loss_coarse
```

---

## Inference and interpretation
- Fine prediction: `argmax(softmax(logits_fine), dim=1)`
- Coarse confidence: `sigmoid(logits_coarse)` gives group-wise probabilities.
- Optional “fallback”: if fine distribution is uncertain but a coarse group is confident, you can report the coarse group as “good enough”.

---

## Common pitfalls
- **Do not** treat “uncertain” as a real class in softmax if you want it excluded from learning.
- Mask out uncertain voxels in **metrics and logging**, not just loss.
- If you oversample patches, ensure sampling/weights are computed from **valid voxels only**.
