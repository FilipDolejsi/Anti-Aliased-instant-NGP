#!/bin/bash
# run_thesis_eval.sh — Multi-seed empirical evaluation for JKU thesis rubric
#
# Runs 3 seeds × 2 models (baseline + Zip-iNGP) × N scenes.
# Computes PSNR/SSIM/LPIPS at 1×/2×/4×/8× resolution scales.
# Prints mean ± std table at the end.
#
# Usage:
#   ./run_thesis_eval.sh                                   # all 8 blender scenes, 3 seeds
#   ./run_thesis_eval.sh --scenes "lego chair"             # subset of scenes
#   ./run_thesis_eval.sh --scenes "drums ficus hotdog"     # remaining scenes
#   ./run_thesis_eval.sh --model zip                       # only Zip-iNGP
#   ./run_thesis_eval.sh --model baseline                  # only baseline
#   ./run_thesis_eval.sh --iters 30000                     # faster (less converged)
#   ./run_thesis_eval.sh --gpu 2
#
# Expected runtime: 3 seeds × 2 models × 8 scenes × 75K iters × ~30ms/step ≈ 60 hours
# For a quick check: --scenes "lego" --iters 30000  ≈ 1 hour

set -euo pipefail

WORK_DIR="<input your working directory here>"
DATA_ROOT="<input your working directory here>"
RUNS_DIR="$WORK_DIR/<input your working directory here>"
SUMMARY_DIR="$WORK_DIR/<input your working directory here>"

# Defaults
SEEDS=(42 1337 2026)
SCENES="lego chair drums ficus hotdog materials mic ship"
ITERS=75000
GPU=2
MODEL_ARG="both"   # baseline | zip | both

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenes) SCENES="$2";     shift 2 ;;
        --iters)  ITERS="$2";      shift 2 ;;
        --gpu)    GPU="$2";        shift 2 ;;
        --model)  MODEL_ARG="$2";  shift 2 ;;
        --help|-h) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

case "$MODEL_ARG" in
    both)     MODELS=(baseline zip) ;;
    baseline) MODELS=(baseline) ;;
    zip)      MODELS=(zip) ;;
    *) echo "Unknown --model value: $MODEL_ARG  (use baseline | zip | both)"; exit 1 ;;
esac

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_DIR/runs_${TIMESTAMP}.txt"
source /system/apps/studentenv/miniconda3/bashrc
conda activate /system/user/studentwork/dolejsi/ngp_env
export CUDA_VISIBLE_DEVICES=$GPU   # must be set AFTER conda activate
cd "$WORK_DIR"

echo "================================================================"
echo "  Thesis Multi-Seed Evaluation"
echo "  Seeds: ${SEEDS[*]}  |  Scenes: $SCENES  |  Iters: $ITERS  |  GPU: $GPU  |  Models: ${MODELS[*]}"
echo "================================================================"
echo ""

# Training loop
declare -A RUN_MAP   # key = scene_model_seed → exp_name

for scene in $SCENES; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="thesis_${model}_${scene}_s${seed}_${TIMESTAMP}"
            echo "──────────────────────────────────────────────"
            echo "  [$model] $scene  seed=$seed  → $exp"
            echo "──────────────────────────────────────────────"

            zip_flag="";     [[ "$model" == "zip" ]] && zip_flag="--use_zip"
            occ_flag="";     [[ "$model" == "zip" ]] && occ_flag="--occ_threshold 0.5"
            nwd_flag="";     [[ "$model" == "zip" ]] && nwd_flag="--zip_nwd_lambda 0.1"

            python train.py \
                --use_warp --use_amp \
                $zip_flag $occ_flag $nwd_flag \
                --seed $seed \
                --exp_name "$exp" \
                --data_root "$DATA_ROOT/$scene" \
                --iterations $ITERS \
                --t 524288 \
                --val_interval 10000

            RUN_MAP["${scene}_${model}_${seed}"]="$exp"
            echo "$scene $model $seed $exp" >> "$LOG"
            echo ""
        done
    done
done

# Aggregate results
python3 - <<PYEOF
import json, os, sys, math
from pathlib import Path

runs_dir  = Path("$RUNS_DIR")
log_file  = "$LOG"
scales    = [1, 2, 4, 8]
scenes    = "$SCENES".split()
models    = "${MODEL_ARG}".replace("both", "baseline zip").split()
seeds     = [${SEEDS[0]}, ${SEEDS[1]}, ${SEEDS[2]}]
timestamp = "$TIMESTAMP"

def load(exp):
    p = runs_dir / exp / "metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))

def mean_std(vals):
    if not vals: return "N/A", "N/A"
    mu = sum(vals) / len(vals)
    var = sum((v - mu)**2 for v in vals) / len(vals)
    return f"{mu:.3f}", f"{math.sqrt(var):.3f}"

# Load exp names from log
run_map = {}
for line in open(log_file):
    sc, mo, se, exp = line.strip().split()
    run_map[(sc, mo, int(se))] = exp

print()
print("=" * 78)
print("  MULTI-SEED RESULTS  (μ ± σ across 3 seeds)")
print("=" * 78)

summary = {}
for scene in scenes:
    print(f"\n  Scene: {scene}")
    print(f"  {'Scale':<6} {'Model':<12} {'PSNR (μ±σ)':<20} {'SSIM (μ±σ)':<20} {'LPIPS (μ±σ)'}")
    print(f"  {'-'*6} {'-'*12} {'-'*20} {'-'*20} {'-'*12}")
    for scale in scales:
        for model in models:
            psnrs, ssims, lpipss = [], [], []
            for seed in seeds:
                key = (scene, model, seed)
                if key not in run_map: continue
                d = load(run_map[key])
                if d is None: continue
                ms = d.get("eval_multiscale", {})
                if f"psnr_{scale}x" in ms: psnrs.append(ms[f"psnr_{scale}x"])
                if f"ssim_{scale}x" in ms: ssims.append(ms[f"ssim_{scale}x"])
                if f"lpips_{scale}x" in ms: lpipss.append(ms[f"lpips_{scale}x"])
            pm, ps = mean_std(psnrs)
            qm, qs = mean_std(ssims)
            lm, ls = mean_std(lpipss)
            psnr_str  = f"{pm} ± {ps}" if pm != "N/A" else "N/A"
            ssim_str  = f"{qm} ± {qs}" if qm != "N/A" else "N/A"
            lpips_str = f"{lm} ± {ls}" if lm != "N/A" else "N/A"
            print(f"  {scale}×{'':<5} {model:<12} {psnr_str:<20} {ssim_str:<20} {lpips_str}")
            summary[f"{scene}/{model}/{scale}x"] = dict(psnr_mu=pm, psnr_sd=ps, ssim_mu=qm, ssim_sd=qs, lpips_mu=lm, lpips_sd=ls)

print()
print("=" * 78)
out = "$RESULTS_DIR/summary_${TIMESTAMP}.json"
json.dump(summary, open(out, "w"), indent=2)
print(f"  Summary saved to: {out}")
print("=" * 78)
PYEOF
