# Dense SSL Experiments Log

## Goal
Train a self-supervised encoder that produces meaningful per-pixel class assignments for 3D EM images, using cross-slice prediction as the pretext task.

---

## Experiment 1: Initial Implementation with SIGReg

**Setup:**
- DenseEncoder (ConvNeXt backbone + FPN decoder) → K=50 softmax classes per pixel
- SlicePredictor: predict middle slice encoding from 4 context slices (2 before, 2 after)
- SIGReg regularization (from LeJEPA) on spatial-mean of encoder outputs
- Prediction target was detached (no gradient flow to encoder from prediction loss)

**Results:**
- `pred_loss ≈ 0` (prediction task trivially easy)
- `reg_loss ≈ 32` (very high)
- Outputs were nearly uniform (all one color/class)
- `num_active_classes = 50` but this was misleading - all classes had ~equal tiny probabilities

**Diagnosis:**
SIGReg was designed for continuous embedding vectors, not softmax probability distributions. The softmax constraint (sum to 1) fundamentally changes the distribution properties.

---

## Experiment 2: DiversityReg (Entropy-based Regularization)

**Changes:**
- Replaced SIGReg with DiversityReg based on regularization.md:
  - `L_marg = max(0, H* - H(p̄))` — penalize only if marginal entropy < threshold
  - `L_cond = mean pixel entropy` — encourage sharp per-pixel predictions
- H* = log(8) ≈ 2.08 (require ~8 effective classes minimum)

**Results:**
- Still getting near-uniform outputs
- One class slightly dominating everywhere
- `uniq = 1` (only 1 class in argmax despite "50 active")
- `avg_max_prob ≈ 0.17` (peaked but same class wins everywhere)

**Diagnosis:**
Encoder only received gradients from regularization (prediction target was detached). Not enough learning signal to create meaningful spatial structure.

---

## Experiment 3: EMA Target Encoder

**Changes:**
- Added target_encoder as EMA copy of online encoder (τ = 0.996)
- Online encoder: processes context slices, receives gradients
- Target encoder: processes middle slice, provides prediction target, updated by EMA only
- Prediction loss now flows gradients to online encoder

**Results:**
- Epoch 10: Near-uniform outputs with noise at edges
- Epoch 20: Promising! Complex structure, blobs tracking image features
- Epoch 30-40: Collapsed to **horizontal bands** (~8 bands of varying width)
- Same bands for all inputs regardless of image content

**Diagnosis:**
Network found a positional encoding that satisfies all losses:
- Multiple classes used (satisfies L_marg)
- Sharp predictions (satisfies L_cond)
- Consistent across Z-slices (satisfies prediction loss)
- But completely ignores image content

---

## Experiment 4: Cross-Sample Variance Loss

**Changes:**
- Added L_cross to encourage different samples to have different encodings at each spatial position:
  ```python
  var_across_samples = enc.var(dim=0)  # (K, H', W')
  mean_var = var_across_samples.mean()
  L_cross = -mean_var  # maximize variance
  ```
- Weight: γ = 1.0

**Results:**
- Successfully broke out of horizontal bands
- Epoch progression:
  - Early: mostly monocolor
  - Middle: wavy vertical bands
  - Epoch 50: complex swirly patterns, varying between samples
  - Stats at epoch 51: `pred=0.0151, var=0.0179, cond=0.002, maxP=0.999, uniq=11`
- Epoch 100: Settled into **diagonal stripes** with slight image-based modulation
  - Patterns are swirly/complex but bear no resemblance to image structures
  - Still essentially positional encoding with minor perturbations

**Diagnosis:**
Cross-sample variance loss prevents identical outputs but doesn't force content-dependence. Network found that slight image-based modulation of a positional pattern satisfies variance constraint without actually encoding semantic content.

---

## Experiment 5: Geometric Equivariance Loss (PLANNED)

**Idea:**
Test whether encoding depends on image content vs position by:
1. Create two views: original and randomly rotated/flipped
2. Encode both
3. Apply inverse transform to the transformed encoding
4. Compare - should match if encoding is content-dependent

**Rationale:**
- Positional encoding: rotation doesn't change output, but inverse-rotating that output creates mismatch
- Content-dependent encoding: rotation rotates the encoding, inverse-rotation recovers original → match

**Results:**
The network discovered a grid-like spatial encoding that is robust to rotations and flips.  :|

I'm giving up on this approach for now.  The task we're giving the network is apparently much easier to solve through a clever spatial encoding, than through actually understanding the images.

---

## Key Insights

1. **Prediction loss alone is insufficient** — the prediction task can be solved by positional patterns that are consistent across Z-slices

2. **Entropy/diversity constraints are necessary but not sufficient** — they prevent single-class collapse but allow positional hacks

3. **Cross-sample variance helps but isn't enough** — prevents identical outputs but allows position-based patterns with minor modulation

4. **The core problem:** Nothing in our losses requires class assignments to depend on actual image content. The network finds clever shortcuts.

5. **Geometric equivariance** may be the key — directly tests whether the encoding is "about" the image or "about" the position
