from noise_schedule import NoiseSchedule
from unet import UNet


# ============================================
# New denoising step for DDIM
# ============================================
# You can add this as a method to your NoiseSchedule class,
# or keep it as a standalone function — your choice.

def ddim_denoise_step(schedule, x_t, noise_pred, t, t_prev, eta=0.0):
    """
    One step of DDIM sampling. Replaces the DDPM denoise_one_step.

    Key differences from DDPM:
    - Uses TWO timesteps: where you are (t) and where you're jumping to (t_prev)
    - First predicts the clean image x_0, then uses it to jump to t_prev
    - eta controls randomness: 0 = fully deterministic, 1 = similar to DDPM

    Args:
        schedule:   your NoiseSchedule object (need access to alpha_bars)
        x_t:        current noisy image,        shape (batch, 3, 32, 32)
        noise_pred: network's noise prediction,  shape (batch, 3, 32, 32)
        t:          current timestep             (integer)
        t_prev:     timestep we're jumping to    (integer, or -1 if final step)
        eta:        controls randomness          (0.0 = deterministic DDIM)

    Returns:
        x_{t_prev}: less noisy image,            shape (batch, 3, 32, 32)
    """

    # --- Step 1: Get alpha bar values for current and target timesteps ---
    alpha_bar_t = schedule.alpha_bars[t]

    # If t_prev is -1, we're at the final step — use alpha_bar = 1.0
    # (meaning the "target" is a fully clean image)
    if t_prev >= 0:
        alpha_bar_t_prev = schedule.alpha_bars[t_prev]
    else:
        alpha_bar_t_prev = 1.0

    # --- Step 2: Predict the clean image x_0 ---
    # This is Equation 15 from the DDPM paper:
    # x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    predicted_x0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

    # Optional: clamp to [-1, 1] for stability
    # Some implementations do this, some don't — experiment with both
    predicted_x0 = clamp(predicted_x0, -1, 1)

    # --- Step 3: Compute sigma (controls how much randomness to add) ---
    # When eta = 0: sigma = 0, fully deterministic
    # When eta = 1: sigma matches DDPM-like stochasticity
    sigma = eta * sqrt(
        (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        * (1 - alpha_bar_t / alpha_bar_t_prev)
    )

    # --- Step 4: Compute "direction pointing to x_t" ---
    # This is the deterministic component that points from x_0 toward noise
    direction = sqrt(1 - alpha_bar_t_prev - sigma ** 2) * noise_pred

    # --- Step 5: Combine predicted x_0 and direction to get x_{t_prev} ---
    x_prev = sqrt(alpha_bar_t_prev) * predicted_x0 + direction

    # --- Step 6: Add noise if eta > 0 ---
    if sigma > 0:
        noise = random_normal(same_shape_as(x_t))
        x_prev = x_prev + sigma * noise

    return x_prev


# ============================================
# DDIM sampling loop
# ============================================

def ddim_sample(model, schedule, num_samples=64, num_steps=50, eta=0.0):
    """
    Generate images using DDIM sampling.

    Instead of 1000 steps like DDPM, we use a subset (e.g. 50 steps).
    Same trained model, same noise schedule — just a faster sampling loop.

    Args:
        model:       trained UNet from DDPM
        schedule:    NoiseSchedule object
        num_samples: how many images to generate
        num_steps:   how many denoising steps (e.g. 50 instead of 1000)
        eta:         randomness control (0 = deterministic, 1 = stochastic)

    Returns:
        generated images, shape (num_samples, 3, 32, 32), range [-1, 1]
    """

    model.eval()

    with no_gradient():

        # --- Step 1: Create the timestep subsequence ---
        # Pick num_steps evenly spaced timesteps from [0, 999]
        # Example with num_steps=50: [999, 979, 959, 939, ..., 19]
        # There are different ways to do this — here's a simple one:

        # Evenly space num_steps values from 0 to 999
        timesteps = linear_space(0, 999, num_steps).long()
        # Reverse so we go from noisy to clean
        timesteps = reversed(timesteps)  # [999, 979, ..., 19, 0] or similar

        # --- Step 2: Start from pure noise ---
        x = random_normal(shape=(num_samples, 3, 32, 32))
        x = x.to(DEVICE)

        # --- Step 3: Denoise using the subsequence ---
        for i in range(len(timesteps)):
            t = timesteps[i]

            # t_prev is the next timestep we're jumping to
            # If we're at the last step, use -1 to signal "final"
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = -1

            # Create batch of same timestep for all images
            t_batch = tensor([t]).repeat(num_samples).to(DEVICE)

            # Network predicts noise (same model from DDPM training!)
            noise_pred = model(x, t_batch)

            # DDIM denoising step (the new formula)
            x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

        return x


# ============================================
# DDIM sampling with progress visualization
# ============================================

def ddim_sample_with_progress(model, schedule, num_samples=8, num_steps=50, eta=0.0):
    """
    Same as ddim_sample but saves intermediate predictions
    so you can visualize the denoising process.
    """

    snapshots = []

    with no_gradient():
        timesteps = linear_space(0, 999, num_steps).long()
        timesteps = reversed(timesteps)

        x = random_normal(shape=(num_samples, 3, 32, 32))
        x = x.to(DEVICE)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            t_batch = tensor([t]).repeat(num_samples).to(DEVICE)
            noise_pred = model(x, t_batch)

            # Save predicted x_0 at this step
            alpha_bar_t = schedule.alpha_bars[t]
            predicted_x0 = (x - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            snapshots.append({
                "timestep": t,
                "predicted_x0": predicted_x0.clone()
            })

            x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

        return x, snapshots


# ============================================
# Comparison script: DDPM vs DDIM at different step counts
# ============================================

def compare_sampling_speeds(model, schedule, num_samples=16):
    """
    Generate images with different numbers of steps to see
    how DDIM quality degrades gracefully while DDPM falls apart.

    This is the key experiment to run — it shows why DDIM matters.
    """

    import time

    # Test different step counts
    step_counts = [10, 20, 50, 100, 200, 1000]

    for num_steps in step_counts:

        start_time = time.time()

        if num_steps == 1000:
            # Full DDPM sampling for comparison
            images = ddpm_sample(model, schedule, num_samples)
            method = "DDPM"
        else:
            # DDIM with reduced steps
            images = ddim_sample(model, schedule, num_samples, num_steps, eta=0.0)
            method = "DDIM"

        elapsed = time.time() - start_time

        # Convert to [0, 1] and save
        images = (images + 1) / 2
        images = clamp(images, 0, 1)
        save_image_grid(images, f"comparison_{method}_{num_steps}steps.png")

        print(f"{method} {num_steps} steps: {elapsed:.1f} seconds")

    # Expected output:
    # DDIM 10 steps:   ~0.5 seconds  — blurry but recognizable
    # DDIM 20 steps:   ~1 second     — decent quality
    # DDIM 50 steps:   ~2.5 seconds  — good quality
    # DDIM 100 steps:  ~5 seconds    — very close to full DDPM
    # DDIM 200 steps:  ~10 seconds   — nearly identical to full DDPM
    # DDPM 1000 steps: ~50 seconds   — best quality, but slow


# ============================================
# Eta comparison: deterministic vs stochastic
# ============================================

def compare_eta(model, schedule, num_samples=8, num_steps=50):
    """
    Show how eta affects the output.
    eta=0 is fully deterministic (same noise -> same image every time)
    eta=1 is stochastic (similar to DDPM)

    Fun experiment: run eta=0 multiple times with the same seed
    and verify you get identical images.
    """

    etas = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Fix the starting noise so we can compare fairly
    set_random_seed(42)
    fixed_noise = random_normal(shape=(num_samples, 3, 32, 32))

    for eta in etas:
        # Start from the SAME noise
        x = fixed_noise.clone().to(DEVICE)

        timesteps = linear_space(0, 999, num_steps).long()
        timesteps = reversed(timesteps)

        with no_gradient():
            for i in range(len(timesteps)):
                t = timesteps[i]
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

                t_batch = tensor([t]).repeat(num_samples).to(DEVICE)
                noise_pred = model(x, t_batch)
                x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta)

        images = (x + 1) / 2
        images = clamp(images, 0, 1)
        save_image_grid(images, f"eta_{eta}.png")

    # Expected: eta=0 images all share same high-level structure
    # As eta increases, more variation appears
    # eta=1 images look different from each other even with same starting noise


# ============================================
# DDIM consistency experiment
# ============================================

def consistency_experiment(model, schedule, num_samples=4):
    """
    Demonstrate DDIM's consistency property:
    Same starting noise with different numbers of steps
    should produce images with the same high-level features.

    This does NOT work with DDPM — only with DDIM (eta=0).
    """

    # Fix starting noise
    set_random_seed(42)
    fixed_noise = random_normal(shape=(num_samples, 3, 32, 32))

    step_counts = [10, 20, 50, 100, 200]

    for num_steps in step_counts:
        # Always start from the SAME noise
        x = fixed_noise.clone().to(DEVICE)

        timesteps = linear_space(0, 999, num_steps).long()
        timesteps = reversed(timesteps)

        with no_gradient():
            for i in range(len(timesteps)):
                t = timesteps[i]
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

                t_batch = tensor([t]).repeat(num_samples).to(DEVICE)
                noise_pred = model(x, t_batch)
                x = ddim_denoise_step(schedule, x, noise_pred, t, t_prev, eta=0.0)

        images = (x + 1) / 2
        images = clamp(images, 0, 1)
        save_image_grid(images, f"consistency_{num_steps}steps.png")

    # Expected: all images should look roughly the same across step counts!
    # Same faces, same colors, same poses — just increasing detail.
    # This is because x_T encodes the high-level features and DDIM
    # deterministically decodes them regardless of how many steps you use.


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    # Setup — load your trained DDPM model
    model = UNet(image_channels=3, base_channels=128)
    model.load_state_dict(load("your_ddpm_checkpoint.pt"))
    model = model.to(DEVICE)
    model.eval()

    schedule = NoiseSchedule(T=1000)
    # Move schedule tensors to DEVICE

    # 1. Basic DDIM sampling
    print("=== Basic DDIM sampling (50 steps) ===")
    images = ddim_sample(model, schedule, num_samples=64, num_steps=50, eta=0.0)
    save_samples(images, "ddim_samples.png")

    # 2. Compare step counts
    print("\n=== Comparing step counts ===")
    compare_sampling_speeds(model, schedule)

    # 3. Compare eta values
    print("\n=== Comparing eta values ===")
    compare_eta(model, schedule)

    # 4. Consistency experiment
    print("\n=== Consistency experiment ===")
    consistency_experiment(model, schedule)

    print("\nDone!")


# ============================================
# SUMMARY OF WHAT CHANGED FROM DDPM
# ============================================
#
# Files unchanged:
#   - noise_schedule.py  (same constants, same add_noise for training)
#   - unet.py            (same architecture)
#   - train.py           (same training, same loss, same everything)
#
# What's new:
#   - ddim_denoise_step: new formula that predicts x_0 first, then jumps
#   - ddim_sample: uses a subset of timesteps instead of all 1000
#   - eta parameter: controls deterministic vs stochastic
#
# The core insight:
#   DDPM says "each step must be small and stochastic"
#   DDIM says "actually you can take big deterministic jumps
#              because the training objective doesn't care about
#              the specific forward process, only the marginals q(x_t|x_0)"
#
# In practice:
#   DDPM 1000 steps: ~50 seconds, stochastic
#   DDIM 50 steps:   ~2.5 seconds, deterministic, nearly same quality
