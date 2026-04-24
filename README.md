# DDIM (in progress) — CIFAR10 Diffusion from scratch

A from-scratch PyTorch implementation built up piece by piece toward the
DDIM paper (Song et al., 2020). Currently the noise schedule, U-Net, DDPM
training loop, and DDPM sampler are implemented. The deterministic
DDIM sampler (η=0, skip-step τ subsequence) is the remaining piece.

## Files

| File | What it does |
| --- | --- |
| `noise_schedule.py` | Linear β schedule + precomputed constants. `add_noise` (DDPM Eq. 4) and `denoise_one_step` (DDPM Algorithm 2). |
| `unet.py` | The ε-prediction network. Sinusoidal time embedding → residual blocks with time conditioning → self-attention at 16×16 and 4×4 → symmetric up path with skip connections. |
| `train.py` | CIFAR10 training loop (DDPM Algorithm 1). Periodically generates sample grids and saves checkpoints. |
| `sample.py` | Standalone generation from a trained checkpoint. Includes a progressive-generation mode that snapshots `x̂_0` estimates at regular timesteps (paper Fig. 6 style). |

## Running it

```bash
# Train (downloads CIFAR10 on first run, ~170 MB to ./data)
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py

# Sample from a saved checkpoint (update CHECKPOINT_PATH in sample.py first)
python sample.py
```

Outputs write to the current directory: `samples_step_*.png`,
`checkpoint_step_*.pt`, and for sampling, `generated_samples.png` /
`generated_final.png` / `progressive_generation.png`.

Device detection order: **CUDA → MPS → CPU**. `cudnn.benchmark` and
`pin_memory` only flip on when CUDA is actually available.

## This is a shrunken model — and you can see it in training

The paper's CIFAR10 config is **53.5M parameters** (`base_channels=128`,
`T=1000`, `batch=128`) and trains for ~800k steps on a serious GPU. To
make this runnable on an M4 MacBook Pro, the config in this repo is cut
down significantly:

| | Paper | This repo |
| --- | --- | --- |
| `base_channels` | 128 | **64** |
| Parameters | ~53.5 M | **~13.4 M** |
| `T` (diffusion steps) | 1000 | **500** |
| `BATCH_SIZE` | 128 | **64** |
| Target steps | 800 k | run until Ctrl+C |

**Observed effect on training loss** (M4 MPS, ~2.7 steps/sec, 5 000 steps):

```
Step 100:  0.1328
Step 300:  0.0447   ← fast drop as the model learns the identity-like
Step 500:  0.0657     shortcut for high-t timesteps (where x_t ≈ ε)
Step 1000: 0.0491
Step 2000: 0.0679   ← model enters its actual learning regime,
Step 2400: 0.0330     but the per-batch loss now has enough variance
Step 3000: 0.0384     (~±0.02) that you cannot read a trend off
Step 3700: 0.0340     individual 100-step prints
Step 4700: 0.0369
Step 5000: 0.0464
```

The loss floor is sitting around **0.035–0.05**, where the paper
(full-size model) reaches ~0.02 after extensive training. Two forces are
at work:

- **Capacity limit.** 13.4 M params isn't enough to model fine
  low-noise residuals as well as 53.5 M. You hit the model's floor
  sooner and lower quality caps out earlier.
- **Easier task.** Halving `T` (1000→500) makes the per-step denoising
  jump larger, which is *harder* per step, but the total chain is
  shorter, so cumulative drift is less forgiving — a net wash for loss
  value, but visible in slightly blurrier samples.

### Sample-quality milestones actually observed

With this shrunken config, what the `samples_step_*.png` images look
like:

- **0–2 k steps:** pure Gaussian noise.
- **2 k–10 k steps:** faint color bias, occasional low-frequency blobs.
- **10 k–50 k steps:** distinct color regions, vaguely object-shaped
  blobs if you squint. This is roughly the ceiling for this config on
  an M4 — the model plateaus here.
- **Beyond ~50 k:** diminishing returns. No amount of additional
  training on the shrunken model will reach paper-quality recognizable
  CIFAR10 classes.

For paper-quality samples, you need to flip the config back
(`BASE_CHANNELS=128`, `T=1000`, `BATCH_SIZE=128`) and run on a real
GPU. On an A100 that's ~13 hours for 800k steps; on a 4090 ~18 hours.

## What's not implemented yet

- The **DDIM sampler** (the whole reason the repo is named this).
  The current `sample.py` runs DDPM's Markov-chain Algorithm 2 — 500
  sequential network calls, stochastic noise injection at each step.
  DDIM's contribution is the non-Markovian deterministic sampler
  `x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1}−σ²)·ε_θ + σ·z` (η=0 is
  fully deterministic) that can skip timesteps — e.g., sample in 50
  network calls instead of 500 with minimal quality loss.
- **EMA of weights** (pseudocode hook exists in `train.py`).
- **Checkpoint resume** in `train.py` — right now every run starts
  from step 0.
