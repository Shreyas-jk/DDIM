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


def _ddim_timesteps(num_steps, device):
    """Evenly spaced subsequence of timesteps from T-1 down to 0."""
    ts = torch.linspace(0, T - 1, num_steps, device=device).long().tolist()
    return list(reversed(ts))


# ============================================
# DDIM sampling
# ============================================

@torch.no_grad()
def ddim_sample(model, schedule, num_samples=64, num_steps=50, eta=0.0, device=DEVICE,
                init_noise=None):
    """
    Generate images using DDIM sampling.

    num_steps can be << T. Same trained model as DDPM, just a different sampler.
    """
    model.eval()
    timesteps = _ddim_timesteps(num_steps, device="cpu")

    if init_noise is None:
        x = torch.randn(num_samples, 3, 32, 32, device=device)
    else:
        x = init_noise.to(device)

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)
        x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

    return x


@torch.no_grad()
def ddim_sample_with_progress(model, schedule, num_samples=8, num_steps=50, eta=0.0,
                              device=DEVICE):
    snapshots = []
    timesteps = _ddim_timesteps(num_steps, device="cpu")

    x = torch.randn(num_samples, 3, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)

        alpha_bar_t = schedule.alpha_bars[t]
        predicted_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        snapshots.append({"timestep": t, "predicted_x0": predicted_x0.clone()})

        x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

    return x, snapshots


# ============================================
# Experiments
# ============================================

def compare_sampling_speeds(model, schedule, num_samples=NUM_SAMPLES, device=DEVICE):
    """DDIM at various step counts vs full DDPM. Demonstrates the speedup."""
    step_counts = [10, 20, 50, 100, T]  # last entry runs DDPM

    for num_steps in step_counts:
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()

        if num_steps == T:
            images = ddpm_sample(model, schedule, num_samples=num_samples, device=device)
            method = "DDPM"
        else:
            images = ddim_sample(model, schedule, num_samples=num_samples,
                                 num_steps=num_steps, eta=0.0, device=device)
            method = "DDIM"

        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - start

        save_samples(images, f"comparison_{method}_{num_steps}steps.png")
        print(f"{method} {num_steps:>4} steps: {elapsed:6.1f} s")


def compare_eta(model, schedule, num_samples=8, num_steps=50, device=DEVICE):
    """Same starting noise, varying eta. Visualizes the deterministic→stochastic axis."""
    etas = [0.0, 0.25, 0.5, 0.75, 1.0]

    torch.manual_seed(42)
    fixed_noise = torch.randn(num_samples, 3, 32, 32)

    for eta in etas:
        x = ddim_sample(model, schedule, num_samples=num_samples,
                        num_steps=num_steps, eta=eta, device=device,
                        init_noise=fixed_noise.clone())
        save_samples(x, f"eta_{eta:.2f}.png")
        print(f"eta={eta:.2f} done")


def consistency_experiment(model, schedule, num_samples=4, device=DEVICE):
    """
    Same fixed noise, increasing step counts, eta=0. With DDIM the high-level
    features should be stable across step counts — only fine detail changes.
    This is the property that does NOT hold for DDPM.
    """
    torch.manual_seed(42)
    fixed_noise = torch.randn(num_samples, 3, 32, 32)

    for num_steps in [10, 20, 50, 100, 200]:
        x = ddim_sample(model, schedule, num_samples=num_samples,
                        num_steps=num_steps, eta=0.0, device=device,
                        init_noise=fixed_noise.clone())
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
    images = ddim_sample(model, schedule, num_samples=NUM_SAMPLES,
                         num_steps=DEFAULT_DDIM_STEPS, eta=0.0)
    save_samples(images, "ddim_samples.png")

    print("\n=== Comparing step counts (DDIM vs DDPM) ===")
    compare_sampling_speeds(model, schedule, num_samples=NUM_SAMPLES)

    print("\n=== Comparing eta values (eta=0 deterministic, eta=1 ~DDPM) ===")
    compare_eta(model, schedule, num_samples=8, num_steps=DEFAULT_DDIM_STEPS)

    print("\n=== Consistency experiment (same noise, varying step counts) ===")
    consistency_experiment(model, schedule, num_samples=4)

    print("\nDone!")
