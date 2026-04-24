import torch
from torchvision.utils import make_grid, save_image

from noise_schedule import NoiseSchedule
from unet import UNet


# ============================================
# Setup
# ============================================

# Must match the values used during training.
T = 500
BASE_CHANNELS = 64
CHECKPOINT_PATH = "checkpoint_step_2000.pt"
NUM_SAMPLES = 16


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

# cudnn picks the fastest conv kernels for fixed input shapes (CUDA only)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def load_model(checkpoint_path, device):
    model = UNet(image_channels=3, base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def move_schedule_to_device(schedule, device):
    # NoiseSchedule stores plain tensors (not a Module), so we move them manually
    for attr in [
        "betas", "alphas", "alpha_bars",
        "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar",
        "sqrt_recip_alpha", "beta_over_sqrt_one_minus_alpha_bar",
    ]:
        setattr(schedule, attr, getattr(schedule, attr).to(device))
    return schedule


# ============================================
# Basic sampling — Algorithm 2
# ============================================

@torch.no_grad()
def sample(model, schedule, num_samples=64, device=DEVICE):
    """
    Generate images from pure noise.
    This is the full reverse process Markov chain.

    Returns: generated images, shape (num_samples, 3, 32, 32), range [-1, 1]
    """

    # Step 1: Start from pure random noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)

    # Step 2: Loop from t=999 down to t=0
    # This is the Markov chain — each step only uses current x
    for t in range(T - 1, -1, -1):

        # Create a batch of the same timestep for all images
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        # Network predicts the noise in the current image
        noise_pred = model(x, t_batch)

        # Apply one denoising step
        x = schedule.denoise_one_step(x, noise_pred, t)

        # Optional: print progress
        if t % 100 == 0:
            print(f"Denoising step {T - t}/{T}")

    # x is now generated images in [-1, 1]
    return x


# ============================================
# Progressive sampling — watch the image form
# ============================================

@torch.no_grad()
def sample_with_progress(model, schedule, num_samples=8, save_every=100, device=DEVICE):
    """
    Same as basic sampling, but saves intermediate images
    so you can watch the denoising process happen.

    This creates the progressive generation visualization
    shown in Figure 6 of the paper.
    """

    snapshots = []  # store intermediate images

    x = torch.randn(num_samples, 3, 32, 32, device=device)

    for t in range(T - 1, -1, -1):

        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)

        # Snapshot BEFORE the denoise step so (x, noise_pred, t) are consistent
        if t % save_every == 0 or t == 0:
            x_hat_0 = predict_x0(x, noise_pred, schedule, t)
            snapshots.append({
                "timestep": t,
                "x_t": x.clone(),
                "x_hat_0": x_hat_0.clone(),
            })

        x = schedule.denoise_one_step(x, noise_pred, t)

    return x, snapshots


def predict_x0(x_t, noise_pred, schedule, t):
    """
    Equation 15: Estimate the clean image from a noisy image
    and the network's noise prediction.

    x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    """
    if t == 0:
        return x_t  # already clean

    sqrt_alpha_bar = schedule.sqrt_alpha_bar[t]
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alpha_bar[t]

    return (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar


# ============================================
# Visualization helpers
# ============================================

def save_samples(images, filename):
    """
    Save a batch of generated images as a grid.
    Images come in [-1, 1] range, need to convert to [0, 1].
    """

    images = (images + 1) / 2
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=8)
    save_image(grid, filename)
    print(f"Saved {filename}")


def save_progress(snapshots, filename):
    """
    Save the progressive denoising visualization.
    Rows = samples, columns = timesteps (noise -> clean, left to right).

    snapshots is ordered t=T-1 -> t=0, so its natural order already goes
    noise -> clean when read left to right.
    """

    num_samples = snapshots[0]["x_hat_0"].shape[0]
    num_timesteps = len(snapshots)

    all_images = []
    for i in range(num_samples):
        for snapshot in snapshots:
            all_images.append(snapshot["x_hat_0"][i])

    all_images = torch.stack(all_images, dim=0)
    all_images = (all_images + 1) / 2
    all_images = all_images.clamp(0, 1)

    grid = make_grid(all_images, nrow=num_timesteps)
    save_image(grid, filename)
    print(f"Saved {filename}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif DEVICE == "mps":
        print("Using Apple Metal (MPS).")

    model = load_model(CHECKPOINT_PATH, DEVICE)
    schedule = NoiseSchedule(T=T)
    schedule = move_schedule_to_device(schedule, DEVICE)

    # Basic generation
    print("Generating samples...")
    images = sample(model, schedule, num_samples=NUM_SAMPLES)
    save_samples(images, "generated_samples.png")

    # Progressive generation
    print("Generating with progress visualization...")
    images, snapshots = sample_with_progress(model, schedule, num_samples=8)
    save_samples(images, "generated_final.png")
    save_progress(snapshots, "progressive_generation.png")

    print("Done!")


# ============================================
# NOTES
# ============================================
#
# 1. Sampling is SLOW — 1000 forward passes through the network.
#    For 64 images on a 3090, expect ~20-30 seconds.
#    This is why DDIM was invented (your next project!)
#
# 2. You can generate as many images as you want by adjusting num_samples.
#    GPU memory is the only limit.
#
# 3. Different starting noise = different images every time.
#    To reproduce exact same images, set a random seed before sampling.
#
# 4. If samples look bad, it's almost always because:
#    - Model hasn't trained long enough
#    - Bug in the noise schedule constants
#    - Bug in the denoise_one_step formula
#    - Images weren't normalized to [-1, 1] during training
