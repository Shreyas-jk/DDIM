"""Turn fid_results.json into the tables: FID vs steps, the measured small-N
bias floor, and the step counts at which the two samplers match on FID."""
import json
import math
from pathlib import Path

R = json.loads(Path("fid_results.json").read_text())
SWEEP = R["sweep"]
BASE = R["baselines"]
N = R["n_samples"]
FLOOR = BASE[str(N)]


def curve(name):
    rows = sorted((r for r in SWEEP if r["sampler"] == name), key=lambda r: r["steps"])
    return [(r["steps"], r["fid"]) for r in rows]


def steps_for_fid(pts, target):
    """Fewest steps reaching FID <= target, log-linear interpolation between
    bracketing points. None if the curve never gets there."""
    # Check the cheapest point first: on a curve that degrades with steps
    # (DDIM here) the fewest steps IS the best FID, and scanning for a
    # bracketing crossing would wrongly report the far side instead.
    if pts[0][1] <= target:
        return float(pts[0][0])
    for i in range(len(pts) - 1):
        (s0, f0), (s1, f1) = pts[i], pts[i + 1]
        if (f0 - target) * (f1 - target) <= 0 and f0 != f1:
            w = (target - f0) / (f1 - f0)
            return math.exp(math.log(s0) + w * (math.log(s1) - math.log(s0)))
    return None


ddim, ddpm = curve("DDIM"), curve("DDPM")

print(f"machine: Apple M4 / MPS   samples/config: {N:,}   seed: {R['seed']}")
print(f"checkpoint: {R['checkpoint']}   T={R['T_total']}   reference: {R['reference']}\n")

print("FID vs sampling steps")
print(f"{'steps':>6} | {'DDIM (eta=0)':>13} | {'DDPM (eta=1)':>13} | {'DDPM-DDIM':>10}")
print("-" * 54)
for (s, fd), (_, fp) in zip(ddim, ddpm):
    d = fp - fd
    mark = "" if abs(d) > FLOOR else "  (< floor)"
    print(f"{s:>6} | {fd:>13.2f} | {fp:>13.2f} | {d:>+10.2f}{mark}")

print(f"\nmeasured noise floor (real-vs-real at N={N:,}): {FLOOR:.2f} FID")
print("differences smaller than this are not distinguishable from sampling noise\n")

print("small-sample bias: CIFAR-10 test vs train-50k (true FID = 0)")
for n, v in sorted(BASE.items(), key=lambda kv: int(kv[0])):
    print(f"  N={int(n):>6,}: {v:7.2f}")

print("\nmonotonicity")
for label, pts in (("DDIM", ddim), ("DDPM", ddpm)):
    fids = [f for _, f in pts]
    shape = ("improves with steps" if fids == sorted(fids, reverse=True)
             else "degrades with steps" if fids == sorted(fids)
             else "non-monotone")
    print(f"  {label}: {shape}  (best FID {min(fids):.2f} @ "
          f"{min(pts, key=lambda p: p[1])[0]} steps)")

print("\nsteps needed to reach a common FID")
lo = max(min(f for _, f in ddim), min(f for _, f in ddpm))
hi = min(max(f for _, f in ddim), max(f for _, f in ddpm))
if lo > hi:
    print("  curves do not overlap in FID; no common target exists")
else:
    print(f"{'target FID':>11} | {'DDIM steps':>11} | {'DDPM steps':>11} | {'speedup':>8}")
    print("-" * 50)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = lo + frac * (hi - lo)
        a, b = steps_for_fid(ddim, t), steps_for_fid(ddpm, t)
        sa = f"{a:.1f}" if a else "never"
        sb = f"{b:.1f}" if b else "never"
        sp = f"{b/a:.2f}x" if (a and b) else "-"
        print(f"{t:>11.2f} | {sa:>11} | {sb:>11} | {sp:>8}")

tot = sum(r["sample_seconds"] for r in SWEEP)
fwd = sum(r["steps"] * N for r in SWEEP)
print(f"\ntotal: {fwd:,} forwards in {tot/3600:.2f} h "
      f"({fwd/tot:.0f} forwards/s mean)")
