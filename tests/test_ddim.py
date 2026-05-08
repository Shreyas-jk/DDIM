import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from noise_schedule import NoiseSchedule
from unet import UNet
from ddim_sample import ddim_sample, make_uniform_timesteps


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoint_step_4000.pt"
)
HAS_CHECKPOINT = os.path.exists(CHECKPOINT_PATH)


def _tiny_model(seed=0, base_channels=32):
    """
    A small untrained UNet with a fixed seed. Sufficient for sampler-logic
    tests (determinism, stochasticity, shape, range) where the model just
    needs to be a callable ε-predictor with stable weights between calls.
    """
    torch.manual_seed(seed)
    model = UNet(image_channels=3, base_channels=base_channels)
    model.eval()
    return model


def _trained_model():
    """The committed-but-gitignored 13.4M-parameter checkpoint, if present."""
    from sample import BASE_CHANNELS
    model = UNet(image_channels=3, base_channels=BASE_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def _schedule(T=500):
    return NoiseSchedule(T=T)


# ============================================
# Helper-function tests (no model required)
# ============================================

def test_make_uniform_timesteps_is_decreasing_and_in_range():
    ts = make_uniform_timesteps(num_steps=20, T_total=500)

    assert len(ts) == 20
    assert ts == sorted(ts, reverse=True), "timesteps must be strictly decreasing"
    assert ts[0] == 499, "should start at T-1"
    assert ts[-1] == 0, "should end at 0"
    assert all(0 <= t <= 499 for t in ts)


# ============================================
# Sampler-logic tests (untrained model is fine)
# ============================================

def test_ddim_eta0_is_bit_identical_across_runs():
    """Same x_T + η=0 + same step count → identical output (allclose atol=0)."""
    model = _tiny_model()
    schedule = _schedule()
    timesteps = make_uniform_timesteps(20)

    torch.manual_seed(123)
    x_T = torch.randn(2, 3, 32, 32)

    out_a = ddim_sample(model, schedule, x_T.clone(), timesteps, eta=0.0, device="cpu")
    out_b = ddim_sample(model, schedule, x_T.clone(), timesteps, eta=0.0, device="cpu")

    assert torch.allclose(out_a, out_b, atol=0.0), "η=0 must be deterministic"
    assert torch.equal(out_a, out_b), "η=0 outputs must be bit-identical"


def test_ddim_eta1_is_nondeterministic():
    """η=1 injects per-step noise — two runs from the same x_T must differ."""
    model = _tiny_model()
    schedule = _schedule()
    timesteps = make_uniform_timesteps(20)

    torch.manual_seed(0)
    x_T = torch.randn(2, 3, 32, 32)

    torch.manual_seed(1)
    out_a = ddim_sample(model, schedule, x_T.clone(), timesteps, eta=1.0, device="cpu")
    torch.manual_seed(2)
    out_b = ddim_sample(model, schedule, x_T.clone(), timesteps, eta=1.0, device="cpu")

    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "η=1 must inject randomness; same x_T should produce different x_0"


def test_ddim_output_shape_and_range():
    """Output must be (B, 3, 32, 32) and within [-1, 1] (predicted_x0 is clamped)."""
    model = _tiny_model()
    schedule = _schedule()
    timesteps = make_uniform_timesteps(10)

    torch.manual_seed(0)
    x_T = torch.randn(4, 3, 32, 32)

    out = ddim_sample(model, schedule, x_T, timesteps, eta=0.0, device="cpu")

    assert out.shape == (4, 3, 32, 32)
    # The final step jumps to ᾱ=1 with direction=0, so x_0 = clamp(predicted_x0, -1, 1).
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


# ============================================
# Heavy tests (need the trained checkpoint)
# ============================================

# The 50MB checkpoint is gitignored, so this test only runs locally where
# the file is present. CI skips it.
@pytest.mark.slow
@pytest.mark.skipif(not HAS_CHECKPOINT, reason="checkpoint_step_4000.pt not available")
def test_ddim_step_count_invariance_at_eta0():
    """
    With η=0 and a trained model, the sampler is a deterministic ODE-style
    integrator over the same trajectory — coarse and fine step counts should
    land at structurally similar images.

    Threshold: 0.85 cosine similarity in flattened pixel space.
    Chosen because (a) random images have cos-sim ≈ 0 in expectation, so
    >0.85 is a meaningful structural-overlap signal; (b) at this checkpoint
    (step 4000, shrunken 13.4M-param config) we measured >0.92 between N=20
    and N=100 in practice, so 0.85 leaves a healthy margin without being lax.
    """
    model = _trained_model()
    schedule = _schedule()

    torch.manual_seed(42)
    x_T = torch.randn(4, 3, 32, 32)

    samples = {}
    for n in (20, 50, 100):
        ts = make_uniform_timesteps(n)
        samples[n] = ddim_sample(model, schedule, x_T.clone(), ts, eta=0.0, device="cpu")

    flat20 = samples[20].flatten(1)
    flat50 = samples[50].flatten(1)
    flat100 = samples[100].flatten(1)

    cos_20_50 = F.cosine_similarity(flat20, flat50, dim=1).mean().item()
    cos_20_100 = F.cosine_similarity(flat20, flat100, dim=1).mean().item()
    cos_50_100 = F.cosine_similarity(flat50, flat100, dim=1).mean().item()

    assert cos_20_50 > 0.85, f"N=20 vs N=50 cos sim {cos_20_50:.3f} < 0.85"
    assert cos_20_100 > 0.85, f"N=20 vs N=100 cos sim {cos_20_100:.3f} < 0.85"
    assert cos_50_100 > 0.85, f"N=50 vs N=100 cos sim {cos_50_100:.3f} < 0.85"
