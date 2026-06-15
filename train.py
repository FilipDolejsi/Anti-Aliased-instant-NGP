import copy
import math
import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from config import Config
from data import NeRFDataset, NeRFDataset360
from model import InstantNGP, ZipInstantNGP
from rendering import render_rays, rendering_stats, set_warp_dt
from occupancy import OccupancyGrid
from profiler import profiler

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None

try:
    import lpips as lpips_lib
    _lpips_fn = None   # lazy-init on first use to avoid GPU alloc at import time
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

def _get_lpips_fn(device):
    global _lpips_fn
    if _lpips_fn is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            _lpips_fn = lpips_lib.LPIPS(net='vgg').eval().to(device)
    return _lpips_fn

try:
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_SSIM = True
except Exception:
    compute_ssim = None
    HAS_SSIM = False

import imageio


def compute_scene_bounds(config):
    aabb_min = torch.tensor(config.AABB_MIN)
    aabb_max = torch.tensor(config.AABB_MAX)
    print(f"  Fixed Scene bounds: min={aabb_min.numpy()}, max={aabb_max.numpy()}")
    return aabb_min, aabb_max


def validate_and_save(model, ema_model, dataset, aabb_min, aabb_max, config, step,
                      best_psnr, run_dir, occ_grid=None, cone_radius=None,
                      zip_mode=False, scene_type='synthetic',
                      zip_collapse=False, zip_n_samples=6, zip_sigma_scale=1.0):
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    val_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    psnr_total = 0.0
    ssim_total = 0.0
    val_bg = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)

    n_evaluated = 0
    with torch.no_grad():
        for idx in val_indices:
            if idx >= len(dataset): break

            rays_o, rays_d, target_rgba = dataset.get_rays_for_image(idx)
            H, W = dataset.H, dataset.W

            target_rgb = target_rgba[..., :3] * target_rgba[..., 3:4] + val_bg * (1.0 - target_rgba[..., 3:4])

            # Zip-NeRF: smaller chunks to avoid OOM from K× memory expansion
            chunk_size = 512 if zip_mode else 4096
            rgb_pred_list = []

            flat_o = rays_o.reshape(-1, 3)
            flat_d = rays_d.reshape(-1, 3)

            for k in range(0, flat_o.shape[0], chunk_size):
                chunk_o = flat_o[k : k+chunk_size]
                chunk_d = flat_d[k : k+chunk_size]
                rgb_chunk = render_rays(
                    eval_model, chunk_o, chunk_d,
                    aabb_min.to(config.DEVICE),
                    aabb_max.to(config.DEVICE),
                    background_color=val_bg,
                    num_samples=config.N_SAMPLES,
                    perturb=False,
                    occupancy_grid=occ_grid,
                    cone_radius=cone_radius,
                    zip_mode=zip_mode,
                    scene_type=scene_type,
                    zip_collapse=zip_collapse,
                    zip_n_samples=zip_n_samples,
                    zip_sigma_scale=zip_sigma_scale,
                )
                rgb_pred_list.append(rgb_chunk)

            rgb_pred = torch.cat(rgb_pred_list, 0).reshape(H, W, 3)

            # 1. PSNR
            mse = F.mse_loss(rgb_pred, target_rgb)
            psnr = -10. * torch.log10(mse)
            psnr_total += psnr.item()

            # 2. SSIM
            if HAS_SSIM and compute_ssim is not None:
                img_pred_np = torch.clamp(rgb_pred, 0.0, 1.0).cpu().numpy()
                img_target_np = target_rgb.cpu().numpy()
                ssim_val = compute_ssim(img_target_np, img_pred_np, data_range=1.0, channel_axis=-1)
                ssim_total += ssim_val

            n_evaluated += 1

    avg_psnr = psnr_total / n_evaluated if n_evaluated > 0 else 0.0
    avg_ssim = (ssim_total / n_evaluated) if (HAS_SSIM and compute_ssim is not None and n_evaluated > 0) else 0.0

    is_best = avg_psnr > best_psnr
    if is_best:
        best_psnr = avg_psnr
        ssim_str = f" | SSIM: {avg_ssim:.4f}" if HAS_SSIM and compute_ssim is not None else ""
        print(f"[Step {step}] New Best Validation PSNR: {best_psnr:.2f} dB{ssim_str}. Saving checkpoint.")
        torch.save(eval_model.state_dict(), os.path.join(run_dir, "instant_ngp_best.pth"))
    else:
        ssim_str = f" | SSIM: {avg_ssim:.4f}" if HAS_SSIM and compute_ssim is not None else ""
        print(f"[Step {step}] Validation PSNR: {avg_psnr:.2f} dB{ssim_str}")

    return avg_psnr, avg_ssim, best_psnr


EVAL_SCALES = [1, 2, 4, 8]

def _downsample(img_hwc, scale):
    """Bicubic downsample a (H,W,3) GPU tensor by integer scale factor."""
    if scale == 1:
        return img_hwc
    return F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        scale_factor=1.0 / scale, mode='bicubic', align_corners=False, recompute_scale_factor=False,
    ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def evaluate(model, ema_model, dataset, aabb_min, aabb_max, config, run_dir,
             occ_grid=None, cone_radius=None, zip_mode=False, scene_type='synthetic',
             zip_collapse=False, zip_n_samples=6, zip_sigma_scale=1.0):
    eval_model = ema_model if ema_model is not None else model
    eval_model.eval()
    print(f"Starting Evaluation on {len(dataset)} images...")

    renders_dir = os.path.join(run_dir, "test_renders")
    os.makedirs(renders_dir, exist_ok=True)

    # Per-scale accumulators — lists so we can persist per-image values
    psnr_by_scale  = {s: [] for s in EVAL_SCALES}
    ssim_by_scale  = {s: [] for s in EVAL_SCALES}
    lpips_by_scale = {s: [] for s in EVAL_SCALES}

    all_frames = []
    eval_bg = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)

    lpips_fn = _get_lpips_fn(config.DEVICE) if HAS_LPIPS else None
    start_eval_time = time.time()

    with torch.no_grad():
        for i in range(len(dataset)):
            rays_o, rays_d, target_rgba = dataset.get_rays_for_image(i)
            H, W = dataset.H, dataset.W

            target_rgb = target_rgba[..., :3] * target_rgba[..., 3:4] + eval_bg * (1.0 - target_rgba[..., 3:4])

            eval_chunk_size = 512 if zip_mode else 4096
            rgb_pred_list = []
            flat_o = rays_o.reshape(-1, 3)
            flat_d = rays_d.reshape(-1, 3)

            for k in range(0, flat_o.shape[0], eval_chunk_size):
                rgb_pred_list.append(render_rays(
                    eval_model, flat_o[k:k+eval_chunk_size], flat_d[k:k+eval_chunk_size],
                    aabb_min.to(config.DEVICE), aabb_max.to(config.DEVICE),
                    background_color=eval_bg, num_samples=config.N_SAMPLES,
                    perturb=False, occupancy_grid=occ_grid,
                    cone_radius=cone_radius, zip_mode=zip_mode, scene_type=scene_type,
                    zip_collapse=zip_collapse, zip_n_samples=zip_n_samples,
                    zip_sigma_scale=zip_sigma_scale,
                ))

            rgb_pred = torch.cat(rgb_pred_list, 0).reshape(H, W, 3)

            # Compute metrics at each scale
            for scale in EVAL_SCALES:
                pred_s = _downsample(rgb_pred,   scale)
                tgt_s  = _downsample(target_rgb, scale)

                mse = F.mse_loss(pred_s, tgt_s)
                psnr_by_scale[scale].append((-10. * torch.log10(mse)).item())

                if HAS_SSIM and compute_ssim is not None:
                    p_np = pred_s.cpu().numpy()
                    t_np = tgt_s.cpu().numpy()
                    ssim_by_scale[scale].append(
                        compute_ssim(t_np, p_np, data_range=1.0, channel_axis=-1)
                    )

                if lpips_fn is not None:
                    p_lp = pred_s.permute(2, 0, 1).unsqueeze(0) * 2 - 1   # [0,1]→[-1,1]
                    t_lp = tgt_s.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    lpips_by_scale[scale].append(lpips_fn(p_lp, t_lp).item())

            img_np = (torch.clamp(rgb_pred, 0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            all_frames.append(img_np)
            imageio.imwrite(os.path.join(renders_dir, f"test_{i:03d}.png"), img_np)

            if i % 10 == 0 or i == len(dataset) - 1:
                p1 = psnr_by_scale[1][-1]
                s1 = ssim_by_scale[1][-1] if ssim_by_scale[1] else 0.0
                print(f"  Test Img {i}/{len(dataset)}: PSNR@1x={p1:.2f} dB | SSIM@1x={s1:.4f}")

    total_eval_time = time.time() - start_eval_time
    fps = len(dataset) / total_eval_time

    def _mean(lst): return sum(lst) / len(lst) if lst else 0.0

    mean_psnr = _mean(psnr_by_scale[1])
    mean_ssim = _mean(ssim_by_scale[1])

    print(f"Evaluation Complete. Mean PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | FPS: {fps:.2f}")
    for s in EVAL_SCALES:
        p = _mean(psnr_by_scale[s]); q = _mean(ssim_by_scale[s]); lp = _mean(lpips_by_scale[s])
        lp_str = f" | LPIPS={lp:.4f}" if lpips_by_scale[s] else ""
        print(f"  {s}×: PSNR={p:.2f} SSIM={q:.4f}{lp_str}")

    print("Saving GIF video...")
    imageio.mimsave(os.path.join(run_dir, "video.gif"), all_frames, fps=30)
    print(f"Video saved to '{run_dir}/video.gif'")

    multiscale = {}
    for s in EVAL_SCALES:
        multiscale[f'psnr_{s}x']  = _mean(psnr_by_scale[s])
        multiscale[f'ssim_{s}x']  = _mean(ssim_by_scale[s])
        multiscale[f'lpips_{s}x'] = _mean(lpips_by_scale[s])

    # Per-image metric lists (required for paired significance tests in aggregate_suite.py)
    per_image = {
        'psnr': {f'{s}x': psnr_by_scale[s]  for s in EVAL_SCALES},
        'ssim': {f'{s}x': ssim_by_scale[s]  for s in EVAL_SCALES},
        'lpips': {f'{s}x': lpips_by_scale[s] for s in EVAL_SCALES},
        'render_time_per_frame_s': total_eval_time / max(len(dataset), 1),
    }

    return mean_psnr, mean_ssim, fps, multiscale, per_image


def _write_status(run_dir, status):
    with open(os.path.join(run_dir, 'run_status.json'), 'w') as f:
        json.dump({'status': status}, f)


def generate_graphs(metrics, run_dir):
    if not HAS_MATPLOTLIB or plt is None:
        return

    print("Generating performance graphs...")

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_times"], metrics["train_psnr"], alpha=0.3, label="Train PSNR (per batch)", color='blue')
    if metrics["val_psnr"]:
        plt.plot(metrics["val_times"], metrics["val_psnr"], marker='o', label="Validation PSNR", color='orange', linewidth=2)
    plt.title("Convergence: PSNR vs. Time")
    plt.xlabel("Cumulative Training Time (Seconds)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_psnr_vs_time.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_steps"], metrics["train_psnr"], alpha=0.3, label="Train PSNR (per batch)", color='blue')
    if metrics["val_psnr"]:
        plt.plot(metrics["val_steps"], metrics["val_psnr"], marker='o', label="Validation PSNR", color='orange', linewidth=2)
    plt.title("Convergence: PSNR vs. Step")
    plt.xlabel("Training Step")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_psnr_vs_step.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_steps"], metrics["step_times_ms"], alpha=0.6, color='green')

    avg_step_time = sum(metrics["step_times_ms"]) / len(metrics["step_times_ms"])
    plt.axhline(y=avg_step_time, color='r', linestyle='--', label=f'Avg: {avg_step_time:.2f} ms')

    plt.title("Training Throughput: Time per Step")
    plt.xlabel("Training Step")
    plt.ylabel("Time per Step (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_throughput.png"))
    plt.close()

    if HAS_SSIM and metrics["val_ssim"]:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["val_steps"], metrics["val_ssim"], marker='o', label="Validation SSIM", color='purple', linewidth=2)
        plt.title("Convergence: SSIM vs. Step")
        plt.xlabel("Training Step")
        plt.ylabel("SSIM")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, "graph_ssim_vs_step.png"))
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Instant NGP PyTorch/Warp Implementation")
    parser.add_argument('--exp_name', type=str, default="run_baseline", help="Name of the experiment run folder")
    parser.add_argument('--data_root', type=str, default=Config.DATA_ROOT, help="Path to dataset")
    parser.add_argument('--use_warp', action='store_true', help="Enable Warp backend")
    parser.add_argument('--use_amp', action='store_true',
                        help="Enable automatic mixed precision (fp16 MLP forward via Tensor Cores). "
                             "Roughly 2x MLP speedup on 2080 Ti. Safe to combine with --use_warp.")
    parser.add_argument('--iterations', type=int, default=Config.ITERATIONS, help="Number of training iterations")
    parser.add_argument('--val_interval', type=int, default=Config.VAL_INTERVAL, help="Validation interval")
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help="Ray count fallback for PyTorch mode")
    parser.add_argument('--sample_budget', type=int, default=Config.SAMPLE_BUDGET,
                        help="Fixed total MLP evals per step in Warp+DDA mode (paper: 2^18=262144)")
    parser.add_argument('--lr', type=float, default=Config.LR, help="Learning rate")
    parser.add_argument('--n_samples', type=int, default=Config.N_SAMPLES, help="Number of samples per ray")
    parser.add_argument('--l', type=int, default=Config.L, dest='L', help="Number of hash grid levels")
    parser.add_argument('--f', type=int, default=Config.F, dest='F', help="Feature dimension per level")
    parser.add_argument('--t', type=int, default=Config.T, dest='T', help="Hash table size")
    parser.add_argument('--aabb_min', type=float, nargs=3, default=Config.AABB_MIN, help="AABB min coords (e.g. -1.5 -1.5 -1.5)")
    parser.add_argument('--aabb_max', type=float, nargs=3, default=Config.AABB_MAX, help="AABB max coords (e.g. 1.5 1.5 1.5)")
    parser.add_argument('--profile_interval', type=int, default=0,
                        help="Print CUDA stage timings every N steps (0 = disabled). Suggested: 500")
    # Occupancy grid is ALWAYS on — paper Appendix E always uses it for NeRF.
    parser.add_argument('--no_occupancy_grid', action='store_true',
                        help="Disable occupancy grid (NOT paper-true; for ablation only)")
    parser.add_argument('--occ_resolution', type=int, default=128,
                        help="Occupancy grid resolution per axis (default: 128, paper uses 128)")
    parser.add_argument('--occ_threshold', type=float, default=0.1,
                        help="Occupancy pruning threshold (default: 0.1). Zip-NeRF's blurry density "
                             "needs ~0.5 to keep occupancy at ~9%% and maintain ray diversity.")
    parser.add_argument('--occ_update_interval', type=int, default=16,
                        help="Update occupancy grid every N steps (default: 16, paper uses 16)")
    import math as _math
    parser.add_argument('--warp_dt', type=float, default=_math.sqrt(3.0) / 1024.0,
                        help="Warp ray-march step size (default: sqrt(3)/1024 ≈ 0.00169, paper value). "
                             "Larger = faster but coarser (e.g. sqrt(3)/512 ≈ 0.00338).")
    parser.add_argument('--scene_type', type=str, default='synthetic', choices=['synthetic', '360'],
                        help="'synthetic' for NeRF Blender scenes, '360' for mip-NeRF 360 / LLFF scenes")
    parser.add_argument('--downsample', type=int, default=4,
                        help="Image downsample factor for 360 scenes (default: 4 → images_4/)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (used in multi-seed evaluation).")
    parser.add_argument('--use_zip', action='store_true',
                        help="Enable Zip-NeRF cone multisampling (6 hexagonal samples per DDA point). "
                             "Requires --use_warp. Adds normalized weight decay.")
    parser.add_argument('--zip_nwd_lambda', type=float, default=0.0,
                        help="Normalized weight decay on hash table (mean of squared weights). "
                             "0.0 = disabled (default). Paper uses 5.0 per grid for Blender.")
    # Ablation flags
    parser.add_argument('--zip_collapse_samples', action='store_true',
                        help="Ablation: zero lateral offsets in hex sampling so all K samples "
                             "lie on the ray centre (keeps K, sigma_j, and downweighting unchanged). "
                             "--use_zip required.")
    parser.add_argument('--zip_no_downweighting', action='store_true',
                        help="Ablation: set omega=1.0, disabling scale-aware feature attenuation. "
                             "Full hex spread is kept; only the erf gating is removed. "
                             "--use_zip required.")
    parser.add_argument('--zip_n_samples', type=int, default=6,
                        help="Zip-NeRF multisample count per DDA point (default=6 hex). "
                             "Other values use K uniform angles; moment-matching only holds at K=6. "
                             "--use_zip required.")
    parser.add_argument('--zip_sigma_scale', type=float, default=0.5,
                        help="Scale factor on sigma_j (paper Eq.3 hyperparameter=0.5). "
                             "sigma_j = cone_radius * t / sqrt(2) * zip_sigma_scale. "
                             "--use_zip required.")
    parser.add_argument('--zip_nwd_per_level', action='store_true',
                        help="Use paper-faithful NWD formula: lambda * sum_l(mean(V_l^2)) instead of "
                             "lambda * global_mean(V^2). Paper formula is 16x stronger at same lambda. "
                             "--use_zip required.")
    parser.add_argument('--zip_eval_n_samples', type=int, default=None,
                        help="Override zip_n_samples for evaluation only (default: same as --zip_n_samples). "
                             "Set to 1 for ~6x eval speedup with minimal quality loss. --use_zip required.")
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_float32_matmul_precision('high')

    run_dir = os.path.join("runs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Saving outputs to: {run_dir}/")

    if HAS_SSIM:
        print("SSIM Tracking: ENABLED (scikit-image found)")
    else:
        print("SSIM Tracking: DISABLED (install scikit-image to enable)")

    config = Config()

    config.DATA_ROOT = args.data_root
    config.USE_WARP = args.use_warp
    config.ITERATIONS = args.iterations
    config.VAL_INTERVAL = args.val_interval
    config.BATCH_SIZE = args.batch_size
    config.SAMPLE_BUDGET = args.sample_budget
    config.LR = args.lr
    config.N_SAMPLES = args.n_samples
    config.L = args.L
    config.F = args.F
    config.T = args.T
    config.AABB_MIN = args.aabb_min
    config.AABB_MAX = args.aabb_max

    if not HAS_MATPLOTLIB:
        print("matplotlib not available; graphs disabled.")

    if args.profile_interval > 0:
        profiler.enable(print_interval=args.profile_interval)
        print(f"CUDA Profiling: ENABLED (reporting every {args.profile_interval} steps)")
    else:
        print("CUDA Profiling: DISABLED (use --profile_interval N to enable)")

    if config.USE_WARP:
        set_warp_dt(args.warp_dt)
        print(f"Backend: Warp  |  Warp march Δt = {args.warp_dt:.6f} "
              f"(activates after occ grid prunes below 30%)")
    else:
        print(f"Backend: PyTorch")

    metrics = {
        "config": vars(args),
        "train_steps": [],
        "train_times": [],
        "train_psnr": [],
        "step_times_ms": [],
        "val_steps": [],
        "val_times": [],
        "val_psnr": [],
        "val_ssim": [],
        "eval_mean_psnr": 0.0,
        "eval_mean_ssim": 0.0,
        "eval_fps": 0.0,
        # Ablation config logged for traceability in aggregate_suite.py
        "ablation_config": {
            "zip_collapse_samples":  args.zip_collapse_samples,
            "zip_no_downweighting":  args.zip_no_downweighting,
            "zip_n_samples":         args.zip_n_samples,
        },
    }

    if args.scene_type == '360':
        DS = NeRFDataset360
        ds_kwargs = dict(device=config.DEVICE, downsample=args.downsample)
        # 360 scenes: large AABB in world space; contraction handles the unbounded extent
        config.AABB_MIN = [-2.0, -2.0, -2.0]
        config.AABB_MAX = [ 2.0,  2.0,  2.0]
    else:
        DS = NeRFDataset
        ds_kwargs = dict(device=config.DEVICE)
        # Zip requires fixed white bg for Blender scenes: K=6 averaging blurs the alpha
        # boundary, so a random bg corrupts the gradient there (paper: blender.gin bg_intensity=1)
        if args.use_zip:
            config.RANDOM_BG_TRAIN = False


    train_dataset = DS(config.DATA_ROOT, split='train', **ds_kwargs)
    val_dataset   = DS(config.DATA_ROOT, split='val',   **ds_kwargs)
    test_dataset  = DS(config.DATA_ROOT, split='test',  **ds_kwargs)

    aabb_min, aabb_max = compute_scene_bounds(config)
    train_dataset.shuffle()

    # cone_radius = 1/(focal*sqrt(3)) moment-matches a square pixel to an isotropic Gaussian (Appendix C)
    cone_radius = (1.0 / (train_dataset.focal * math.sqrt(3.0))) if args.use_zip else None
    if args.use_zip:
        print(f"Zip-NeRF: ENABLED  (cone_radius={cone_radius:.6f}, focal={train_dataset.focal:.1f})")
        if args.zip_collapse_samples:
            print("  Ablation: zip_collapse_samples=True  (lateral offsets zeroed)")
        if args.zip_no_downweighting:
            print("  Ablation: zip_no_downweighting=True  (omega fixed to 1)")
        if args.zip_n_samples != 6:
            print(f"  Ablation: zip_n_samples={args.zip_n_samples}  (non-standard K)")
        nwd_formula = "per_level_sum (paper)" if args.zip_nwd_per_level else "global_mean (1/16 paper)"
        print(f"  sigma_scale={args.zip_sigma_scale}  (paper default=0.5)")
        print(f"  nwd_formula={nwd_formula}  lambda={args.zip_nwd_lambda}")
    else:
        print("Zip-NeRF: DISABLED (pass --use_zip to enable)")

    if args.use_zip:
        raw_model = ZipInstantNGP(config).to(config.DEVICE)
        # Apply ablation flags directly on the encoder — no signature threading needed
        raw_model.encoder.no_downweighting = args.zip_no_downweighting
    else:
        raw_model = InstantNGP(config).to(config.DEVICE)

    if not config.USE_WARP:
        print("Compiling model with torch.compile...")
        model = torch.compile(raw_model, dynamic=True)
    else:
        print("Skipping torch.compile (Warp Backend)...")
        model = raw_model

    # EMA shadow copy used for validation/test; decay=0.999 -> ~1000-step window
    ema_model = copy.deepcopy(raw_model)
    ema_model.eval()
    EMA_DECAY = 0.999

    params_no_decay = []
    params_decay = []

    for name, param in raw_model.named_parameters():
        if "embeddings" in name:
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    optimizer = torch.optim.Adam(
        [
            {'params': params_no_decay, 'weight_decay': 0.0},
            {'params': params_decay,    'weight_decay': 1e-6},
        ],
        lr=config.LR,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS
    )

    # LR schedule: 5K cosine warmup, then ×0.33 step-decay every 10K starting at step 20K.
    # Both baseline and Zip converge by ~10K; a later decay let Adam drift an already-converged model.
    _warmup = 5000
    _decay_start = 15000

    def lr_lambda(step):
        if step < _warmup:
            # Cosine warmup from ~0 to full LR
            return 1e-8 + (1.0 - 1e-8) * math.sin(0.5 * math.pi * step / _warmup)
        adj = step - _warmup  # adjusted step after warmup
        if adj < _decay_start:
            return 1.0
        n = (adj - _decay_start) // 10000 + 1
        return 0.33 ** n

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradScaler prevents fp16 underflow during backward; hash-table embeddings stay fp32
    # since they're accessed via Warp kernels / index_select, not autocast linear layers
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP: ENABLED  (fp16 MLP forward via Tensor Cores)")
    else:
        print("AMP: DISABLED (pass --use_amp to enable)")

    ray_idx = 0
    aabb_min_gpu = aabb_min.to(config.DEVICE)
    aabb_max_gpu = aabb_max.to(config.DEVICE)

    # Paper Appendix E: occupancy grid is always used for NeRF (K=1 for synthetic).
    if args.no_occupancy_grid:
        occ_grid = None
        print("Occupancy Grid: DISABLED (ablation mode — not paper-true)")
    else:
        from occupancy import OccupancyGrid as _OCC
        print(f"Occupancy Grid: ENABLED (res={args.occ_resolution}, "
              f"update_every={args.occ_update_interval} steps, "
              f"threshold={args.occ_threshold:.3f})")
        occ_grid = OccupancyGrid(
            resolution=args.occ_resolution,
            device=config.DEVICE,
            threshold=args.occ_threshold,
        )

    best_psnr = -1.0

    # Configuration summary
    from occupancy import OccupancyGrid as _OCC2
    import math as _m
    _warp_components = []
    if config.USE_WARP:
        _warp_components = ["HashGrid-fwd(fp16 read)", "HashGrid-bwd", "SH", "AABB-intersect"]
    _dt_approx = 2.6 / config.N_SAMPLES  # rough Lego ray length / samples
    print(f"\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│  iNGP Configuration                                             │")
    print(f"│  Hash:   L={config.L}, F={config.F}, T=2^{round(_m.log2(config.T))}, Nmin={config.N_MIN}, Nmax={config.N_MAX}       │")
    print(f"│  MLP:    density=1x{config.HIDDEN_DIM_DENSITY}, color=2x{config.HIDDEN_DIM_COLOR}, SH deg=4                   │")
    print(f"│  Optim:  lr={config.LR:.0e}, β=({config.ADAM_BETA1},{config.ADAM_BETA2}), ε={config.ADAM_EPS:.0e}             │")
    _decay_step = 5000 + _decay_start
    print(f"│  LR decay: ×0.33 at {_decay_step//1000}k, every 10k after                      │")
    _budget_str = f"budget={config.SAMPLE_BUDGET//1024}Ki samp → adaptive rays" if (config.USE_WARP and not args.no_occupancy_grid) else f"{config.BATCH_SIZE} rays × {config.N_SAMPLES} samp"
    print(f"│  Batch:  {_budget_str:<53}│")
    print(f"│  Step Δt≈{_dt_approx:.4f} (paper: {_m.sqrt(3)/1024:.4f})                            │")
    print(f"│  Occ:    128^3, K=1, thresh={_OCC2.THRESHOLD:.2f}, decay=0.95, init=100.0    │")
    print(f"│  Loss:   L2 on sRGB/255 pixel values                           │")
    print(f"│  AMP:    {'fp16 MLP fwd (Tensor Cores)' if use_amp else 'disabled'}                      │")
    print(f"│  Warp:   {', '.join(_warp_components) if _warp_components else 'disabled (PyTorch only)'}  │")
    print(f"└─────────────────────────────────────────────────────────────────┘\n")

    print("Starting Training...")
    _write_status(run_dir, 'started')

    # Reset peak memory stats so we measure only the training phase
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.DEVICE)

    train_start_time = time.time()
    cumulative_train_time = 0.0

    for step in range(config.ITERATIONS):
        step_start_t = time.time()

        profiler.start("data_loading")
        # Fixed-sample-budget ray count (paper Appendix E.3): pool_size = SAMPLE_BUDGET / avg_spr,
        # clamped to [1024, 32768]. Do not divide by K for Zip — that collapses pool_size to the
        # 1024 minimum and starves the model of camera angle diversity.
        use_budget = config.USE_WARP and (occ_grid is not None) and (config.SAMPLE_BUDGET > 0)
        if use_budget:
            avg_spr = rendering_stats["avg_samples_per_ray"]
            avg_spr = avg_spr if avg_spr >= 1.0 else float(config.N_SAMPLES)
            pool_size = int(config.SAMPLE_BUDGET / avg_spr)
            pool_size = max(pool_size, 1024)
            pool_size = min(pool_size, 32768)
        else:
            pool_size = config.BATCH_SIZE

        if ray_idx + pool_size > len(train_dataset):
            train_dataset.shuffle()
            ray_idx = 0

        batch_o = train_dataset.rays_o[ray_idx : ray_idx + pool_size].to(config.DEVICE)
        batch_d = train_dataset.rays_d[ray_idx : ray_idx + pool_size].to(config.DEVICE)
        batch_rgba = train_dataset.target_rgba[ray_idx : ray_idx + pool_size].to(config.DEVICE)
        ray_idx += pool_size

        if config.RANDOM_BG_TRAIN:
            bg_color = torch.rand(3, device=config.DEVICE)
        else:
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)

        batch_rgb_pixels = batch_rgba[..., :3]
        if batch_rgba.shape[-1] == 4:
            batch_alpha = batch_rgba[..., 3:4]
            batch_target_rgb = batch_rgb_pixels * batch_alpha + bg_color * (1.0 - batch_alpha)
        else:
            batch_target_rgb = batch_rgb_pixels  # 360 scenes: no alpha
        profiler.stop("data_loading")

        # Update occupancy grid before the forward pass for this step
        if occ_grid is not None and step % args.occ_update_interval == 0:
            model.eval()
            occ_grid.update(model, step)
            model.train()

        # Grid starts fully occupied (full N_SAMPLES/ray), then prunes to ~13% occupied
        # after ~1500 steps, matching the paper's ~25 evals/ray.
        with torch.amp.autocast('cuda', enabled=use_amp):
            rgb_pred = render_rays(
                model, batch_o, batch_d,
                aabb_min_gpu, aabb_max_gpu,
                background_color=bg_color,
                num_samples=config.N_SAMPLES,
                perturb=True,
                occupancy_grid=occ_grid,
                cone_radius=cone_radius,
                zip_mode=args.use_zip,
                scene_type=args.scene_type,
                zip_collapse=args.zip_collapse_samples,
                zip_n_samples=args.zip_n_samples,
                zip_sigma_scale=args.zip_sigma_scale,
            )

            profiler.start("loss")
            loss = F.mse_loss(rgb_pred, batch_target_rgb)
            if args.use_zip and args.zip_nwd_lambda > 0:
                if args.zip_nwd_per_level:
                    # Paper formula: λ · Σ_l mean(V_l²)  (sum of per-level means)
                    nwd = raw_model.encoder.embeddings.pow(2).mean(dim=(1, 2)).sum()
                else:
                    # Global mean — 16× smaller than paper formula at same lambda
                    nwd = raw_model.encoder.embeddings.pow(2).mean()
                loss = loss + args.zip_nwd_lambda * nwd
            profiler.stop("loss")

        profiler.start("backward")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        profiler.stop("backward")

        profiler.start("optimizer")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        profiler.stop("optimizer")

        # EMA update of shadow weights
        with torch.no_grad():
            for p, ep in zip(raw_model.parameters(), ema_model.parameters()):
                ep.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

        profiler.tick(step)

        step_dt = time.time() - step_start_t
        cumulative_train_time += step_dt

        if step % 100 == 0 or step == config.ITERATIONS - 1:
            with torch.no_grad():
                mse_for_log = F.mse_loss(rgb_pred, batch_target_rgb)
            psnr = -10. * torch.log10(mse_for_log)
            metrics["train_steps"].append(step)
            metrics["train_times"].append(cumulative_train_time)
            metrics["train_psnr"].append(psnr.item())
            metrics["step_times_ms"].append(step_dt * 1000)

            spr = rendering_stats["avg_samples_per_ray"]
            n_rays = rendering_stats["n_valid_rays"]
            print(f"[Step {step:05d}] Loss: {loss.item():.5f} | PSNR: {psnr.item():.2f}dB"
                  f" | rays: {pool_size} | samp/ray: {spr:.1f} | Time: {step_dt*1000:.1f}ms/step"
                  f" | LR: {scheduler.get_last_lr()[0]:.2e}")

        if step > 0 and step % config.VAL_INTERVAL == 0:
            model.train(False)
            current_psnr, current_ssim, best_psnr = validate_and_save(
                model, ema_model, val_dataset, aabb_min_gpu, aabb_max_gpu, config,
                step, best_psnr, run_dir, occ_grid=occ_grid, cone_radius=cone_radius,
                zip_mode=args.use_zip, scene_type=args.scene_type,
                zip_collapse=args.zip_collapse_samples, zip_n_samples=args.zip_n_samples,
                zip_sigma_scale=args.zip_sigma_scale,
            )

            metrics["val_steps"].append(step)
            metrics["val_times"].append(cumulative_train_time)
            metrics["val_psnr"].append(current_psnr)
            metrics["val_ssim"].append(current_ssim)

            model.train(True)
            torch.cuda.empty_cache()

    # Final-step validation (runs even if last step isn't a VAL_INTERVAL multiple)
    final_step = config.ITERATIONS - 1
    if final_step % config.VAL_INTERVAL != 0:
        model.train(False)
        _, _, best_psnr = validate_and_save(
            model, ema_model, val_dataset, aabb_min_gpu, aabb_max_gpu, config,
            final_step, best_psnr, run_dir, occ_grid=occ_grid, cone_radius=cone_radius,
            zip_mode=args.use_zip, scene_type=args.scene_type,
            zip_collapse=args.zip_collapse_samples, zip_n_samples=args.zip_n_samples,
            zip_sigma_scale=args.zip_sigma_scale,
        )
        model.train(True)
        torch.cuda.empty_cache()

    print("Training Complete. Saving Last Model...")
    torch.save(model.state_dict(), os.path.join(run_dir, "instant_ngp_last.pth"))
    torch.save(ema_model.state_dict(), os.path.join(run_dir, "instant_ngp_last_ema.pth"))

    # Record peak GPU memory used during training (before eval resets the counter)
    peak_gpu_training_mb = (
        torch.cuda.max_memory_allocated(config.DEVICE) / 1e6
        if torch.cuda.is_available() else 0.0
    )
    metrics["peak_gpu_memory_training_mb"] = peak_gpu_training_mb
    print(f"Peak GPU memory (training): {peak_gpu_training_mb:.0f} MB")

    # Summarise step timing
    if metrics["step_times_ms"]:
        avg_ms = sum(metrics["step_times_ms"]) / len(metrics["step_times_ms"])
        metrics["avg_step_time_ms"]    = avg_ms
        metrics["avg_steps_per_sec"]   = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    metrics["total_training_seconds"] = cumulative_train_time

    # Zip can peak early and degrade via hash-table memorisation; fall back to the
    # best-val EMA checkpoint if it beats the final weights by >0.5 dB.
    eval_ckpt_label = "last EMA"
    if args.use_zip and metrics["val_psnr"]:
        best_val  = max(metrics["val_psnr"])
        last_val  = metrics["val_psnr"][-1]
        best_ckpt = os.path.join(run_dir, "instant_ngp_best.pth")
        if best_val > last_val + 0.5 and os.path.exists(best_ckpt):
            best_step = metrics["val_steps"][metrics["val_psnr"].index(best_val)]
            print(f"Zip overfit detected: best val {best_val:.2f} dB @ step {best_step} "
                  f"> last val {last_val:.2f} dB.  Loading best checkpoint for evaluation.")
            ema_model.load_state_dict(torch.load(best_ckpt, map_location=config.DEVICE))
            # Rebuild occupancy grid from the best model at threshold=0.1 for complete coverage.
            if occ_grid is not None:
                print("Rebuilding occupancy grid at threshold=0.1 for evaluation...")
                occ_grid.THRESHOLD = 0.1
                for _rebuild_step in range(20):
                    occ_grid.update(ema_model, step=_rebuild_step * 16, chunk_size=262144)
            eval_ckpt_label = f"best-val EMA (step {best_step})"

    # Reset peak-memory counter before eval so we capture eval memory separately
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(config.DEVICE)

    _write_status(run_dir, 'training_done')
    print(f"Evaluating with {eval_ckpt_label} model...")
    eval_zip_n = args.zip_eval_n_samples if args.zip_eval_n_samples is not None else args.zip_n_samples
    eval_psnr, eval_ssim, eval_fps, eval_multiscale, eval_per_image = evaluate(
        model, ema_model, test_dataset, aabb_min, aabb_max, config, run_dir,
        occ_grid=occ_grid, cone_radius=cone_radius,
        zip_mode=args.use_zip, scene_type=args.scene_type,
        zip_collapse=args.zip_collapse_samples, zip_n_samples=eval_zip_n,
        zip_sigma_scale=args.zip_sigma_scale,
    )

    metrics["eval_mean_psnr"]  = eval_psnr
    metrics["eval_mean_ssim"]  = eval_ssim
    metrics["eval_fps"]        = eval_fps
    metrics["eval_multiscale"] = eval_multiscale

    # Per-image metric lists (needed for paired t-tests in aggregate_suite.py)
    metrics["eval_per_image_psnr"]  = eval_per_image["psnr"]
    metrics["eval_per_image_ssim"]  = eval_per_image["ssim"]
    metrics["eval_per_image_lpips"] = eval_per_image["lpips"]
    metrics["eval_render_time_per_frame_s"] = eval_per_image["render_time_per_frame_s"]

    # Peak GPU memory during evaluation
    metrics["peak_gpu_memory_eval_mb"] = (
        torch.cuda.max_memory_allocated(config.DEVICE) / 1e6
        if torch.cuda.is_available() else 0.0
    )

    metrics_path = os.path.join(run_dir, "metrics.json")
    def to_serializable(obj):
        # Convert torch tensors / numpy types to Python primitives for JSON
        try:
            import torch as _torch
            if isinstance(obj, _torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        return obj

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, default=to_serializable)
    print(f"Saved run metrics textually to: {metrics_path}")
    _write_status(run_dir, 'complete')

    generate_graphs(metrics, run_dir)


if __name__ == "__main__":
    main()
