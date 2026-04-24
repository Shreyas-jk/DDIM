import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from noise_schedule import NoiseSchedule
from unet import UNet


# ============================================
# Hyperparameters
# ============================================
# Shrunken config so this runs on a laptop GPU (Apple MPS / small CUDA).
# For paper-quality CIFAR10, use: T=1000, BATCH_SIZE=128, BASE_CHANNELS=128.
T = 500
BATCH_SIZE = 64
BASE_CHANNELS = 64         # paper uses 128 (~53M params); 64 is ~13M
LEARNING_RATE = 2e-4
NUM_EPOCHS = 800           # you will Ctrl+C long before this on a laptop
IMAGE_SIZE = 32
EMA_DECAY = 0.9999         # optional: add later for better sample quality
SAMPLE_EVERY = 500         # generate samples every N steps to track progress
SAVE_EVERY = 2000          # save model checkpoint every N steps
SAMPLE_COUNT = 16          # images per periodic sample grid (fewer = faster)


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


def move_schedule_to_device(schedule, device):
    # NoiseSchedule holds plain tensors (not a Module), so move them manually
    for attr in [
        "betas", "alphas", "alpha_bars",
        "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar",
        "sqrt_recip_alpha", "beta_over_sqrt_one_minus_alpha_bar",
    ]:
        setattr(schedule, attr, getattr(schedule, attr).to(device))
    return schedule


# ============================================
# Helper: Generate and save sample images
# ============================================

def generate_and_save_samples(model, schedule, step, device, num_samples=SAMPLE_COUNT):
    """
    Run Algorithm 2 to generate images from pure noise.
    Save the results to track training progress.
    """

    model.eval()  # switch to evaluation mode (disables dropout)

    with torch.no_grad():  # don't track gradients during sampling
        # Start from pure noise
        x = torch.randn(num_samples, 3, 32, 32, device=device)

        # Run the reverse process: denoise step by step
        for t in range(T - 1, -1, -1):  # T-1, T-2, ..., 1, 0
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            noise_pred = model(x, t_batch)
            x = schedule.denoise_one_step(x, noise_pred, t)

        # x is now generated images in [-1, 1] range
        # Convert back to [0, 1] for saving
        x = (x + 1) / 2
        x = x.clamp(0, 1)

        nrow = min(8, num_samples)
        grid = make_grid(x, nrow=nrow)
        save_image(grid, f"samples_step_{step}.png")

    model.train()  # switch back to training mode


# ============================================
# Training Loop — This is Algorithm 1
# ============================================

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif DEVICE == "mps":
        print("Using Apple Metal (MPS). Expect ~1–2 s/step with the shrunken config.")
    else:
        print("WARNING: no GPU backend available — training on CPU will be extremely slow.")

    # Load CIFAR10 dataset
    # Images should be normalized to [-1, 1] range (not [0, 1])
    # The paper scales integers {0, 1, ..., 255} linearly to [-1, 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),   # paper uses this for CIFAR10
        transforms.ToTensor(),               # converts to [0, 1]
        transforms.Normalize(                # shifts to [-1, 1]
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
        drop_last=True,
    )

    # Create model and noise schedule
    model = UNet(image_channels=3, base_channels=BASE_CHANNELS).to(DEVICE)
    schedule = NoiseSchedule(T=T)
    schedule = move_schedule_to_device(schedule, DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        for batch, _ in dataloader:  # CIFAR10 yields (image, label)

            # batch is (batch_size, 3, 32, 32), values in [-1, 1]
            x_0 = batch.to(DEVICE, non_blocking=True)
            batch_size = x_0.shape[0]

            # ---- Algorithm 1, line 3: sample random timesteps ----
            # Each image gets a different random timestep
            t = torch.randint(0, T, (batch_size,), device=DEVICE)

            # ---- Algorithm 1, line 4: sample random noise ----
            noise = torch.randn_like(x_0)

            # ---- Algorithm 1, line 5: compute loss ----
            # Step A: Create noisy images using Equation 4 shortcut
            x_t = schedule.add_noise(x_0, t, noise)

            # Step B: Network predicts the noise
            noise_pred = model(x_t, t)

            # Step C: Loss is MSE between real noise and predicted noise
            loss = F.mse_loss(noise, noise_pred)

            # ---- Backprop and update ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Optional: EMA update ----
            # For each parameter, do:
            #   ema_param = EMA_DECAY * ema_param + (1 - EMA_DECAY) * param
            # This keeps a smoothed version of the model that generates better samples
            # Skip this on first implementation, add later

            # ---- Logging ----
            step += 1

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            # ---- Generate samples periodically ----
            if step % SAMPLE_EVERY == 0:
                generate_and_save_samples(model, schedule, step, DEVICE)

            # ---- Save checkpoint ----
            if step % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f"checkpoint_step_{step}.pt")


if __name__ == "__main__":
    main()


# ============================================
# NOTES
# ============================================
#
# Common issues to watch for:
#
# 1. Make sure images are in [-1, 1] range, NOT [0, 1]
#    The noise schedule assumes data centered at 0
#
# 2. Make sure all tensors are on the same device (CPU or GPU)
#    The schedule's precomputed tensors need to be on GPU too
#
# 3. Loss should start high and decrease over training
#    If it stays flat or increases, something is wrong
#    Typical starting loss is around 0.5-1.0, should drop to ~0.02-0.05
#
# 4. Early samples (first few thousand steps) will look like noise
#    You might see vague color blobs around 10k-50k steps
#    Recognizable images appear around 100k-200k steps
#    Good quality around 500k-800k steps
#
# 5. On a single GPU, training takes roughly:
#    - 3090/4090: ~15-20 hours for 800k steps
#    - A100: ~10 hours for 800k steps
#
# 6. If you run out of GPU memory, reduce batch_size or base_channels
