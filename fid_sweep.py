"""
FID sweep: deterministic DDIM (eta=0) vs ancestral DDPM (eta=1) on this repo's
checkpoint, against the CIFAR-10 train set.

Both samplers call the same network and the same ddim_denoise_step; the only
differences are eta and the length of the timestep subsequence. eta=1.0 is the
ancestral/stochastic step -- this is the DDIM paper's own DDPM baseline (Song
et al. 2020, Table 1), which sweeps step counts on the shared subsequence.

Features come from pytorch-fid's TF-ported InceptionV3 (pool3, 2048-d), so the
values are on the same scale as published CIFAR-10 FIDs.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from torchvision import datasets

from noise_schedule import NoiseSchedule
from sample import load_model, move_schedule_to_device, pick_device
from ddim_sample import ddim_sample, make_uniform_timesteps

CHECKPOINT = "checkpoint_step_4000.pt"
T_TOTAL = 500
N_SAMPLES = 10_000
BATCH = 256
INCEPTION_BATCH = 128
SEED = 1234
STEP_GRID = [10, 20, 50, 100, 200, 500]
SUBSET_NS = [1_000, 2_500, 5_000, 10_000]
RESULTS = Path("fid_results.json")

DEV = pick_device()


# ---------------------------------------------------------------- inception

def build_inception():
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([block]).to(DEV).eval()


@torch.no_grad()
def features(images_uint8, inception):
    """images_uint8: (N,3,32,32) uint8 CPU tensor -> (N,2048) float64 numpy."""
    out = []
    for i in range(0, len(images_uint8), INCEPTION_BATCH):
        batch = images_uint8[i : i + INCEPTION_BATCH].to(DEV).float() / 255.0
        feat = inception(batch)[0].squeeze(-1).squeeze(-1)
        out.append(feat.cpu())
    return torch.cat(out).double().numpy()


def stats(feat):
    return feat.mean(axis=0), np.cov(feat, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Standard Frechet distance. Inlined because pytorch-fid 0.3.0 calls
    scipy.linalg.sqrtm(disp=...), and scipy >= 1.18 dropped that kwarg."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


def cached_features(name, loader, inception):
    """Inception over CIFAR-10 costs ~465 s for 50k on this machine; cache it."""
    path = Path(f"fid_cache_{name}.npy")
    if path.exists():
        print(f"  loaded cached {name} features")
        return np.load(path)
    feat = features(loader(), inception)
    np.save(path, feat)
    return feat


# ---------------------------------------------------------------- data

def cifar_uint8(train):
    ds = datasets.CIFAR10(root="data", train=train, download=False)
    arr = ds.data  # (N,32,32,3) uint8 numpy
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def generate(model, sched, num_steps, eta, n, seed):
    """Returns (n,3,32,32) uint8 CPU tensor, quantized exactly like real data."""
    ts = make_uniform_timesteps(num_steps, T_total=T_TOTAL)
    # One fixed x_T pool shared by every config, so curves are paired and any
    # difference is the sampler, not the noise draw.
    gen = torch.Generator().manual_seed(seed)
    out = []
    done = 0
    while done < n:
        bs = min(BATCH, n - done)
        x_T = torch.randn(bs, 3, 32, 32, generator=gen)
        # Seed the device RNG for the eta>0 stochastic term (reproducible per chunk).
        torch.manual_seed(seed + done)
        x0 = ddim_sample(model, sched, x_T, ts, eta=eta, device=DEV)
        img = ((x0.clamp(-1, 1) + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
        out.append(img.cpu())
        done += bs
    return torch.cat(out)


# ---------------------------------------------------------------- main

def main():
    print(f"device={DEV}  samples/config={N_SAMPLES}  seed={SEED}")
    inception = build_inception()

    print("computing reference stats (CIFAR-10 train, 50k)...")
    t0 = time.time()
    ref_feat = cached_features("train50k", lambda: cifar_uint8(train=True), inception)
    mu_ref, sig_ref = stats(ref_feat)
    print(f"  reference: {ref_feat.shape} in {time.time()-t0:.0f}s")

    results = {
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "checkpoint": CHECKPOINT,
        "T_total": T_TOTAL,
        "reference": "CIFAR-10 train, 50000 images",
        "device": DEV,
        "baselines": {},
        "sweep": [],
    }

    # Real-vs-real floor: CIFAR-10 *test* (disjoint from train) vs the same
    # reference. This is what FID reads for two genuine draws of one
    # distribution at each N -- i.e. the small-sample bias, measured.
    print("computing real-vs-real baseline (CIFAR-10 test)...")
    test_feat = cached_features("test10k", lambda: cifar_uint8(train=False), inception)
    for n in SUBSET_NS:
        mu, sig = stats(test_feat[:n])
        fid = calculate_frechet_distance(mu, sig, mu_ref, sig_ref)
        results["baselines"][str(n)] = fid
        print(f"  real-vs-real N={n:>6}: FID={fid:7.3f}")
    RESULTS.write_text(json.dumps(results, indent=2))

    model = load_model(CHECKPOINT, DEV)
    sched = move_schedule_to_device(NoiseSchedule(T=T_TOTAL), DEV)

    configs = [("DDIM", 0.0, s) for s in STEP_GRID] + [("DDPM", 1.0, s) for s in STEP_GRID]
    total_fwd = sum(s * N_SAMPLES for _, _, s in configs)
    print(f"\nsweep: {len(configs)} configs, {total_fwd:,} network forwards\n")

    for name, eta, steps in configs:
        t0 = time.time()
        imgs = generate(model, sched, steps, eta, N_SAMPLES, SEED)
        gen_s = time.time() - t0
        feat = features(imgs, inception)
        mu, sig = stats(feat)
        fid = calculate_frechet_distance(mu, sig, mu_ref, sig_ref)
        sub = {}
        for n in SUBSET_NS:
            if n <= N_SAMPLES:
                m2, s2 = stats(feat[:n])
                sub[str(n)] = calculate_frechet_distance(m2, s2, mu_ref, sig_ref)
        row = {
            "sampler": name, "eta": eta, "steps": steps,
            "fid": fid, "fid_by_n": sub,
            "sample_seconds": gen_s,
            "forwards_per_sec": steps * N_SAMPLES / gen_s,
        }
        results["sweep"].append(row)
        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"{name} eta={eta} steps={steps:>3}: FID={fid:7.3f} "
              f"({gen_s:6.0f}s, {row['forwards_per_sec']:.0f} fwd/s)")

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
