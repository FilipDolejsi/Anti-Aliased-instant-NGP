#!/usr/bin/env python3
"""
aggregate_thesis.py — Build the final thesis summary table from completed runs.

Combines baseline runs (any timestamp) with zip runs (any timestamp),
preferring the most recent run per (scene, model, seed) combination.

Usage:
    python aggregate_thesis.py
    python aggregate_thesis.py --scenes "lego chair" --seeds "42 1337 2026"
"""
import argparse
import json
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir",  default="runs")
parser.add_argument("--scenes",    default="lego chair drums ficus hotdog materials mic ship")
parser.add_argument("--models",    default="baseline zip")
parser.add_argument("--seeds",     default="42 1337 2026")
parser.add_argument("--scales",    default="1 2 4 8")
parser.add_argument("--out",       default="results_thesis/thesis_summary_final.json")
args = parser.parse_args()

scenes = args.scenes.split()
models = args.models.split()
seeds  = [int(s) for s in args.seeds.split()]
scales = [int(s) for s in args.scales.split()]

runs_dir = args.runs_dir

def find_runs():
    """Return dict (scene, model, seed) -> (exp_name, timestamp) — newest wins.

    Only considers runs with a numeric YYYYMMDD timestamp to exclude old dev
    runs named with alphabetic suffixes (e.g. _v3, _parallel) that have no
    metrics.json.
    """
    import re
    ts_pattern = re.compile(r'^\d{8}_\d{6}$')
    run_map = {}
    for name in sorted(os.listdir(runs_dir)):
        # thesis_<model>_<scene>_s<seed>_<YYYYMMDD>_<HHMMSS>
        if not name.startswith("thesis_"):
            continue
        parts = name.split("_")
        if len(parts) < 6:
            continue
        model = parts[1]
        scene = parts[2]
        seed_str = parts[3]
        if not seed_str.startswith("s"):
            continue
        try:
            seed = int(seed_str[1:])
        except ValueError:
            continue
        # timestamp must be YYYYMMDD_HHMMSS (two parts)
        timestamp = "_".join(parts[4:])
        if not ts_pattern.match(timestamp):
            continue
        # Skip if no metrics.json
        if not os.path.exists(os.path.join(runs_dir, name, "metrics.json")):
            continue
        key = (scene, model, seed)
        prev = run_map.get(key)
        if prev is None or timestamp > prev[1]:
            run_map[key] = (name, timestamp)
    return {k: v[0] for k, v in run_map.items()}


def load_metrics(exp):
    path = os.path.join(runs_dir, exp, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    mu = sum(vals) / n
    # Sample std (n-1) so error bars are unbiased estimates
    var = sum((v - mu) ** 2 for v in vals) / max(n - 1, 1)
    return mu, math.sqrt(var)


run_map = find_runs()

print()
print("=" * 90)
print("  THESIS RESULTS  (μ ± σ across seeds)  — latest run per (scene/model/seed)")
print("=" * 90)

summary = {}
missing = []

for scene in scenes:
    print(f"\n  Scene: {scene}")
    hdr = f"  {'Scale':<6} {'Model':<12} {'PSNR (μ±σ)':<22} {'SSIM (μ±σ)':<22} {'LPIPS (μ±σ)'}"
    print(hdr)
    print(f"  {'-'*6} {'-'*12} {'-'*22} {'-'*22} {'-'*12}")

    for model in models:
        for scale in scales:
            psnrs, ssims, lpipss = [], [], []
            for seed in seeds:
                key = (scene, model, seed)
                if key not in run_map:
                    missing.append(f"{scene}/{model}/s{seed}")
                    continue
                d = load_metrics(run_map[key])
                if d is None:
                    missing.append(f"{run_map[key]} (no metrics.json)")
                    continue
                ms = d.get("eval_multiscale", {})
                k_p = f"psnr_{scale}x"
                k_s = f"ssim_{scale}x"
                k_l = f"lpips_{scale}x"
                if k_p in ms: psnrs.append(ms[k_p])
                if k_s in ms: ssims.append(ms[k_s])
                if k_l in ms: lpipss.append(ms[k_l])

            pm, ps = mean_std(psnrs)
            qm, qs = mean_std(ssims)
            lm, ls = mean_std(lpipss)

            def fmt(mu, sd):
                if mu is None: return "N/A"
                return f"{mu:.3f} ± {sd:.3f}"

            print(f"  {scale}×{'':<5} {model:<12} {fmt(pm, ps):<22} {fmt(qm, qs):<22} {fmt(lm, ls)}")
            summary[f"{scene}/{model}/{scale}x"] = {
                "psnr_mu": pm, "psnr_sd": ps,
                "ssim_mu": qm, "ssim_sd": qs,
                "lpips_mu": lm, "lpips_sd": ls,
                "n": len(psnrs),
            }

print()
if missing:
    print(f"  Missing runs: {', '.join(missing)}")
print("=" * 90)

# Individual run table
print("\n  Per-run detail (1× PSNR):")
print(f"  {'Scene':<6} {'Model':<10} {'s42':>8} {'s1337':>8} {'s2026':>8}  {'exp (newest timestamp)'}")
for scene in scenes:
    for model in models:
        row = []
        exp_names = []
        for seed in seeds:
            key = (scene, model, seed)
            exp = run_map.get(key, "—")
            d = load_metrics(exp) if exp != "—" else None
            if d:
                ms = d.get("eval_multiscale", {})
                p = ms.get("psnr_1x", 0)
                row.append(f"{p:.2f}")
            else:
                row.append("N/A")
            exp_names.append(exp)
        ts_part = "_".join(exp_names[0].split("_")[4:]) if exp_names[0] != "—" else "—"
        print(f"  {scene:<6} {model:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}  @{ts_part}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved to: {args.out}")
