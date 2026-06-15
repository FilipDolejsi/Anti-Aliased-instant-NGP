"""
Re-run multiscale evaluation on already-trained baseline (and zip) runs.

Usage:
  conda activate ngp_env && export CUDA_VISIBLE_DEVICES=2
  python reeval_baseline.py --suite 20260608_233928 --method baseline
  python reeval_baseline.py --suite 20260608_233928 --method all

For each run directory matching the pattern, loads instant_ngp_best.pth,
rebuilds the occupancy grid, runs evaluate(), and patches metrics.json
with the new eval_multiscale / eval_per_image fields.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys

import numpy as np
import torch

# Import project modules (must be run from work_dir)
from config import Config
from data import NeRFDataset
from model import InstantNGP, ZipInstantNGP
from occupancy import OccupancyGrid
from rendering import render_rays, set_warp_dt
from train import evaluate, EVAL_SCALES, _get_lpips_fn


def rebuild_occ(occ_grid, model, device, chunk_size=262144, n_steps=20):
    for step in range(n_steps):
        occ_grid.update(model, step=step * 16, chunk_size=chunk_size)


def reeval_run(run_dir, dry_run=False):
    mf = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(mf):
        print(f"  SKIP (no metrics.json): {run_dir}")
        return False

    metrics = json.load(open(mf))
    ms = metrics.get('eval_multiscale', {})

    # Check if multiscale already has real values (non-zero)
    if ms and any(ms.get(f'psnr_{s}x', 0) > 1.0 for s in EVAL_SCALES):
        print(f"  SKIP (multiscale already populated): {os.path.basename(run_dir)}")
        return False

    ckpt = os.path.join(run_dir, 'instant_ngp_best.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, 'instant_ngp_last_ema.pth')
    if not os.path.exists(ckpt):
        print(f"  SKIP (no checkpoint): {run_dir}")
        return False

    c = metrics['config']
    print(f"  Re-evaluating: {os.path.basename(run_dir)}")
    print(f"    scene={c['data_root'].split('/')[-1]}  seed={c['seed']}  use_zip={c['use_zip']}")

    if dry_run:
        print("    [dry-run, skipping actual eval]")
        return True

    # Build config
    config = Config()
    config.DATA_ROOT    = c['data_root']
    config.USE_WARP     = c.get('use_warp', True)
    config.N_SAMPLES    = c.get('n_samples', 1024)
    config.L            = c.get('L', 16)
    config.F            = c.get('F', 2)
    config.T            = c.get('T', 524288)
    config.AABB_MIN     = c.get('aabb_min', [-1.3, -1.3, -1.3])
    config.AABB_MAX     = c.get('aabb_max', [1.3, 1.3, 1.3])
    config.SAMPLE_BUDGET = c.get('sample_budget', 262144)

    use_zip = c.get('use_zip', False)
    if use_zip:
        config.RANDOM_BG_TRAIN = False

    if config.USE_WARP:
        set_warp_dt(c.get('warp_dt', math.sqrt(3.0) / 1024.0))

    # Dataset
    from data import NeRFDataset
    test_dataset = NeRFDataset(config.DATA_ROOT, split='test', device=config.DEVICE)
    train_dataset = NeRFDataset(config.DATA_ROOT, split='train', device=config.DEVICE)

    aabb_min = torch.tensor(config.AABB_MIN)
    aabb_max = torch.tensor(config.AABB_MAX)

    cone_radius = (1.0 / (train_dataset.focal * math.sqrt(3.0))) if use_zip else None

    # Model
    if use_zip:
        raw_model = ZipInstantNGP(config).to(config.DEVICE)
        raw_model.encoder.no_downweighting = c.get('zip_no_downweighting', False)
    else:
        raw_model = InstantNGP(config).to(config.DEVICE)

    ema_model = copy.deepcopy(raw_model)
    ema_model.eval()

    print(f"    Loading checkpoint: {os.path.basename(ckpt)}")
    state = torch.load(ckpt, map_location=config.DEVICE)
    ema_model.load_state_dict(state)

    # Occupancy grid
    occ_threshold = c.get('occ_threshold', 0.1)
    occ_grid = OccupancyGrid(
        resolution=c.get('occ_resolution', 128),
        device=config.DEVICE,
        threshold=occ_threshold,
    )
    print(f"    Rebuilding occupancy grid (threshold={occ_threshold})...")
    rebuild_occ(occ_grid, ema_model, config.DEVICE)

    # Evaluate
    zip_n_samples = c.get('zip_eval_n_samples') or c.get('zip_n_samples', 6)
    eval_psnr, eval_ssim, eval_fps, eval_multiscale, eval_per_image = evaluate(
        raw_model, ema_model, test_dataset, aabb_min, aabb_max, config, run_dir,
        occ_grid=occ_grid, cone_radius=cone_radius,
        zip_mode=use_zip, scene_type='synthetic',
        zip_collapse=c.get('zip_collapse_samples', False),
        zip_n_samples=zip_n_samples,
        zip_sigma_scale=c.get('zip_sigma_scale', 1.0),
    )

    # Patch metrics.json
    metrics['eval_mean_psnr']        = eval_psnr
    metrics['eval_mean_ssim']        = eval_ssim
    metrics['eval_fps']              = eval_fps
    metrics['eval_multiscale']       = eval_multiscale
    metrics['eval_per_image_psnr']   = eval_per_image['psnr']
    metrics['eval_per_image_ssim']   = eval_per_image['ssim']
    metrics['eval_per_image_lpips']  = eval_per_image['lpips']
    metrics['eval_render_time_per_frame_s'] = eval_per_image['render_time_per_frame_s']

    if torch.cuda.is_available():
        metrics['peak_gpu_memory_eval_mb'] = (
            torch.cuda.max_memory_allocated(config.DEVICE) / 1e6
        )

    def _serial(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(mf, 'w') as f:
        json.dump(metrics, f, indent=4, default=_serial)
    print(f"    Saved updated metrics.json  (PSNR@1x={eval_psnr:.2f} dB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite',  default='20260608_233928',
                    help='Suite ID (timestamp prefix of run dir names)')
    ap.add_argument('--method', default='baseline',
                    choices=['baseline', 'zip', 'all'],
                    help='Which method runs to re-evaluate')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print which runs would be processed without running eval')
    ap.add_argument('--runs_dir', default='runs',
                    help='Top-level runs directory')
    args = ap.parse_args()

    if args.method == 'all':
        pattern = f'{args.runs_dir}/suite_{args.suite}_*'
    else:
        pattern = f'{args.runs_dir}/suite_{args.suite}_{args.method}_*'

    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"No run directories found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories to check.\n")
    n_done = 0
    for rd in run_dirs:
        if os.path.isdir(rd):
            ok = reeval_run(rd, dry_run=args.dry_run)
            if ok and not args.dry_run:
                n_done += 1

    print(f"\nDone. Re-evaluated {n_done} run(s).")


if __name__ == '__main__':
    main()
