import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from noise_schedule import NoiseSchedule
from unet import UNet


def test_add_noise_output_shape_matches_input():
    schedule = NoiseSchedule(T=100)
    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 32, 32)
    noise = torch.randn_like(x_0)
    t = torch.randint(0, 100, (batch_size,))

    x_t = schedule.add_noise(x_0, t, noise)

    assert x_t.shape == x_0.shape


def test_noise_schedule_precomputed_tensor_shapes():
    T = 100
    schedule = NoiseSchedule(T=T)

    assert schedule.betas.shape == (T,)
    assert schedule.alphas.shape == (T,)
    assert schedule.alpha_bars.shape == (T,)
    assert schedule.sqrt_alpha_bar.shape == (T,)
    assert schedule.sqrt_one_minus_alpha_bar.shape == (T,)


def test_noise_schedule_boundary_at_t_zero():
    beta_start = 0.0001
    schedule = NoiseSchedule(T=100, beta_start=beta_start, beta_end=0.02)

    # At t=0, alpha_bar equals alphas[0] = 1 - beta_start
    expected_alpha_bar_0 = 1.0 - beta_start
    assert schedule.alpha_bars[0].item() == pytest.approx(expected_alpha_bar_0, abs=1e-6)

    # sqrt(alpha_bar[0]) should be very close to 1
    assert schedule.sqrt_alpha_bar[0].item() == pytest.approx(
        expected_alpha_bar_0 ** 0.5, abs=1e-6
    )

    # sqrt(1 - alpha_bar[0]) should be very close to sqrt(beta_start)
    assert schedule.sqrt_one_minus_alpha_bar[0].item() == pytest.approx(
        beta_start ** 0.5, abs=1e-6
    )

    # add_noise at t=0 leaves the image almost unchanged
    x_0 = torch.randn(2, 3, 32, 32)
    noise = torch.randn_like(x_0)
    t = torch.zeros(2, dtype=torch.long)
    x_t = schedule.add_noise(x_0, t, noise)
    assert torch.allclose(x_t, x_0, atol=0.05)


def test_denoise_one_step_output_shape():
    schedule = NoiseSchedule(T=100)
    x_t = torch.randn(2, 3, 32, 32)
    noise_pred = torch.randn_like(x_t)

    # t > 0: stochastic branch (adds random noise)
    out = schedule.denoise_one_step(x_t, noise_pred, t=50)
    assert out.shape == x_t.shape

    # t == 0: deterministic branch (no random noise added)
    out_zero = schedule.denoise_one_step(x_t, noise_pred, t=0)
    assert out_zero.shape == x_t.shape


def test_denoise_one_step_is_deterministic_at_t_zero():
    schedule = NoiseSchedule(T=100)
    x_t = torch.randn(2, 3, 32, 32)
    noise_pred = torch.randn_like(x_t)

    out_a = schedule.denoise_one_step(x_t, noise_pred, t=0)
    out_b = schedule.denoise_one_step(x_t, noise_pred, t=0)

    assert torch.equal(out_a, out_b)


def test_unet_output_shape_matches_input():
    # base_channels=32 is the smallest valid value (must be divisible by 32
    # for the GroupNorm layers); keeps the test fast on CI CPU runners.
    model = UNet(image_channels=3, base_channels=32)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        out = model(x, t)

    assert out.shape == x.shape
