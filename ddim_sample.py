import time
import torch
from torchvision.utils import make_grid, save_image

from noise_schedule import NoiseSchedule
from unet import UNet
from sample import (
    pick_device,
    move_schedule_to_device,
    load_model,
    save_samples,
    sample as ddpm_sample,
    T,
    BASE_CHANNELS,
)


# ============================================
# Setup
# ============================================
DEVICE = pick_device()
CHECKPOINT_PATH = "checkpoint_step_4000.pt"
NUM_SAMPLES = 16
DEFAULT_DDIM_STEPS = 50

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


# ============================================
# Timestep subsequences
# ============================================

def make_uniform_timesteps(num_steps, T_total=T, device="cpu"):
    """
    Build a decreasing uniformly-spaced subsequence of [0, T_total - 1] of
    length num_steps. Returned as a Python list of ints, ordered from largest
    (T-1) down to smallest (0). This is the t-sequence DDIM iterates over.
    """
    ts = torch.linspace(0, T_total - 1, num_steps, device=device).long().tolist()
    return list(reversed(ts))


# Backwards-compatible alias used by older call sites.
_ddim_timesteps = make_uniform_timesteps


# ============================================
# DDIM denoise step
# ============================================

def ddim_denoise_step(schedule, x_t, noise_pred, t, t_prev, eta=0.0):
    """
    One step of DDIM sampling. Replaces the DDPM denoise_one_step.

    Key differences from DDPM:
    - Uses TWO timesteps: where you are (t) and where you're jumping to (t_prev)
    - First predicts the clean image x_0, then uses it to jump to t_prev
    - eta controls randomness: 0 = fully deterministic, 1 = similar to DDPM
    """

    alpha_bar_t = schedule.alpha_bars[t]

    # If t_prev is -1, target is the fully clean image (alpha_bar = 1.0)
    if t_prev >= 0:
        alpha_bar_t_prev = schedule.alpha_bars[t_prev]
    else:
        alpha_bar_t_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)

    # Predict x_0 (DDPM Eq. 15)
    predicted_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
    predicted_x0 = predicted_x0.clamp(-1, 1)

    # Stochasticity coefficient: eta=0 → deterministic, eta=1 → DDPM-equivalent
    sigma = eta * torch.sqrt(
        (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        * (1 - alpha_bar_t / alpha_bar_t_prev)
    )

    # "Direction pointing to x_t" — deterministic component
    direction = torch.sqrt(torch.clamp(1 - alpha_bar_t_prev - sigma ** 2, min=0.0)) * noise_pred

    x_prev = torch.sqrt(alpha_bar_t_prev) * predicted_x0 + direction

    if eta > 0:
        x_prev = x_prev + sigma * torch.randn_like(x_t)

    return x_prev


# ============================================
# DDIM sampling
# ============================================

@torch.no_grad()
def ddim_sample(model, schedule, x_T, timesteps, eta=0.0, device=DEVICE):
    """
    Generate images using DDIM sampling.

    Args:
        model:     ε-prediction U-Net (trained with DDPM loss).
        schedule:  NoiseSchedule providing precomputed ᾱ_t.
        x_T:       starting noise, shape (B, 3, 32, 32). Determines the trajectory.
        timesteps: decreasing list/sequence of integer timesteps in [0, T-1].
                   Use make_uniform_timesteps(N) for the standard uniform spacing.
        eta:       0.0 = deterministic DDIM; 1.0 = DDPM-equivalent stochastic step.

    Returns: x_0 of shape (B, 3, 32, 32), values in roughly [-1, 1].
    """
    model.eval()

    x = x_T.to(device)
    batch_size = x.shape[0]
    timesteps = list(timesteps)

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)
        x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

    return x


@torch.no_grad()
def ddim_sample_with_progress(model, schedule, x_T, timesteps, eta=0.0, device=DEVICE):
    snapshots = []
    timesteps = list(timesteps)

    x = x_T.to(device)
    batch_size = x.shape[0]

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)

        alpha_bar_t = schedule.alpha_bars[t]
        predicted_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        snapshots.append({"timestep": t, "predicted_x0": predicted_x0.clone()})

        x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

    return x, snapshots


# ============================================
# Experiments
# ============================================

def compare_steps(model, schedule, num_samples=NUM_SAMPLES, device=DEVICE, seed=42):
    """
    DDIM at various step counts vs full DDPM. Demonstrates the speedup.
    Writes comparison_DDIM_{10,20,50,100}steps.png and comparison_DDPM_500steps.png.
    Prints a markdown timing table to stdout in the same format as the README.
    """
    step_counts = [10, 20, 50, 100, T]  # last entry runs DDPM

    # Fixed x_T so the visual differences across step counts are comparable.
    torch.manual_seed(seed)
    x_T = torch.randn(num_samples, 3, 32, 32)

    rows = []
    ddpm_elapsed = None

    for num_steps in step_counts:
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()

        if num_steps == T:
            images = ddpm_sample(model, schedule, num_samples=num_samples, device=device)
            method = "DDPM"
        else:
            timesteps = make_uniform_timesteps(num_steps)
            images = ddim_sample(
                model, schedule, x_T.clone(), timesteps, eta=0.0, device=device,
            )
            method = "DDIM"

        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - start

        save_samples(images, f"comparison_{method}_{num_steps}steps.png")
        if num_steps == T:
            ddpm_elapsed = elapsed
        rows.append((method, num_steps, elapsed))

    # Print markdown timing table matching the README format.
    print()
    print("| Sampler | Steps | Wall-clock | Speedup vs DDPM |")
    print("| --- | --- | --- | --- |")
    for method, num_steps, elapsed in rows:
        speedup = (ddpm_elapsed / elapsed) if ddpm_elapsed and elapsed > 0 else 1.0
        print(
            f"| {method} | {num_steps:>3} | **{elapsed:.1f} s** | "
            f"{speedup:.1f}× |"
        )
    print()

    return rows


# Backwards-compatible alias.
compare_sampling_speeds = compare_steps


def compare_eta(model, schedule, num_samples=NUM_SAMPLES, num_steps=DEFAULT_DDIM_STEPS,
                device=DEVICE, seed=42):
    """Same starting noise, varying eta. Visualizes the deterministic→stochastic axis."""
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]

    torch.manual_seed(seed)
    fixed_noise = torch.randn(num_samples, 3, 32, 32)
    timesteps = make_uniform_timesteps(num_steps)

    for eta in etas:
        # Re-seed before each run so the η>0 stochastic path is reproducible across calls.
        torch.manual_seed(seed)
        x = ddim_sample(
            model, schedule, fixed_noise.clone(), timesteps, eta=eta, device=device,
        )
        save_samples(x, f"eta_{eta:.2f}.png")
        print(f"eta={eta:.2f} done")


def consistency_experiment(model, schedule, num_samples=NUM_SAMPLES, device=DEVICE, seed=42):
    """
    Same fixed noise, increasing step counts, eta=0. With DDIM the high-level
    features should be stable across step counts — only fine detail changes.
    This is the property that does NOT hold for DDPM.
    """
    torch.manual_seed(seed)
    fixed_noise = torch.randn(num_samples, 3, 32, 32)

    for num_steps in [10, 20, 50, 100, 200]:
        timesteps = make_uniform_timesteps(num_steps)
        x = ddim_sample(
            model, schedule, fixed_noise.clone(), timesteps, eta=0.0, device=device,
        )
        save_samples(x, f"consistency_{num_steps}steps.png")
        print(f"consistency num_steps={num_steps} done")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DEVICE == "mps":
        print("Using Apple Metal (MPS).")

    model = load_model(CHECKPOINT_PATH, DEVICE)
    schedule = NoiseSchedule(T=T)
    schedule = move_schedule_to_device(schedule, DEVICE)

    print(f"\n=== Basic DDIM sampling ({DEFAULT_DDIM_STEPS} steps, eta=0) ===")
    torch.manual_seed(42)
    x_T = torch.randn(NUM_SAMPLES, 3, 32, 32)
    timesteps = make_uniform_timesteps(DEFAULT_DDIM_STEPS)
    images = ddim_sample(model, schedule, x_T, timesteps, eta=0.0)
    save_samples(images, "ddim_samples.png")

    print("\n=== Comparing step counts (DDIM vs DDPM) ===")
    compare_steps(model, schedule, num_samples=NUM_SAMPLES)

    print("\n=== Comparing eta values (eta=0 deterministic, eta=1 ~DDPM) ===")
    compare_eta(model, schedule, num_samples=NUM_SAMPLES, num_steps=DEFAULT_DDIM_STEPS)

    print("\n=== Consistency experiment (same noise, varying step counts) ===")
    consistency_experiment(model, schedule, num_samples=NUM_SAMPLES)

    print("\nDone!")
