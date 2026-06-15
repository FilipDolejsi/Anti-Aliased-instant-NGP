#!/bin/bash
# eval_blender.sh — Run iNGP baseline vs Zip-iNGP on NeRF Blender synthetic scenes
#
# Usage:
#   ./eval_blender.sh                      # all 8 scenes, both models, 75K iters
#   ./eval_blender.sh --scene lego         # single scene, both models
#   ./eval_blender.sh --model zip          # all scenes, Zip-iNGP only
#   ./eval_blender.sh --model baseline     # all scenes, iNGP baseline only
#   ./eval_blender.sh --iters 50000        # override iteration count
#   ./eval_blender.sh --gpu 2              # GPU index (default: 2)
#
# Expected runtime at 75K iters, ~20ms/step:
#   ~25 min/run × 2 models × 8 scenes = ~7 hours
#   Use --scene lego for a quick ~50 min sanity check first.
#
# Results in runs/blender_{model}_{scene}_{timestamp}/metrics.json
# Summary saved to results_blender/summary_{timestamp}.txt

set -euo pipefail

# Paths
WORK_DIR="<input your working directory here>"
DATA_ROOT="<input your working directory here>"
RUNS_DIR="$WORK_DIR/<input your working directory here>"
SUMMARY_DIR="$WORK_DIR/<input your working directory here>"

# Defaults
ITERS=75000
GPU=2
MODEL_ARG="both"
SCENE_ARG="all"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)  SCENE_ARG="$2";  shift 2 ;;
        --model)  MODEL_ARG="$2";  shift 2 ;;
        --iters)  ITERS="$2";      shift 2 ;;
        --gpu)    GPU="$2";        shift 2 ;;
        --help|-h)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1  (run with --help)"; exit 1 ;;
    esac
done

ALL_SCENES=(chair drums ficus hotdog lego materials mic ship)

if [[ "$SCENE_ARG" == "all" ]]; then
    SCENES=("${ALL_SCENES[@]}")
else
    SCENES=("$SCENE_ARG")
fi

case "$MODEL_ARG" in
    both)     MODELS=(baseline zip) ;;
    baseline) MODELS=(baseline) ;;
    zip)      MODELS=(zip) ;;
    *) echo "Unknown --model: $MODEL_ARG"; exit 1 ;;
esac

get_metric() {
    python3 -c "
import json
try:
    d = json.load(open('$1'))
    v = d.get('$2')
    print(f'{v:.4f}' if v is not None else 'N/A')
except: print('N/A')
" 2>/dev/null
}

source /system/apps/studentenv/miniconda3/bashrc
conda activate /system/user/studentwork/dolejsi/ngp_env
cd "$WORK_DIR"
mkdir -p "$SUMMARY_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$SUMMARY_DIR/summary_${TIMESTAMP}.txt"
declare -A PSNR_MAP SSIM_MAP

echo ""
echo "================================================================"
echo "  Blender Benchmark  |  models: ${MODELS[*]}  |  iters: $ITERS"
echo "  GPU: $GPU  |  scenes: ${SCENES[*]}"
echo "================================================================"
echo ""

for scene in "${SCENES[@]}"; do
    data_root="$DATA_ROOT/$scene"
    if [[ ! -d "$data_root" ]]; then
        echo "WARNING: $data_root not found — skipping $scene"
        continue
    fi

    for model in "${MODELS[@]}"; do
        exp_name="blender_${model}_${scene}_${TIMESTAMP}"
        metrics_json="$RUNS_DIR/$exp_name/metrics.json"

        echo "──────────────────────────────────────────────────────────────"
        echo "  [$model] $scene"
        echo "──────────────────────────────────────────────────────────────"

        zip_flag=""
        occ_flag=""
        [[ "$model" == "zip" ]] && zip_flag="--use_zip"
        [[ "$model" == "zip" ]] && occ_flag="--occ_threshold 0.5"

        export CUDA_VISIBLE_DEVICES=$GPU
        python train.py \
            --use_warp \
            --use_amp \
            $zip_flag $occ_flag \
            --exp_name  "$exp_name" \
            --data_root "$data_root" \
            --iterations $ITERS \
            --t 524288 \
            --profile_interval 10000 \
            --val_interval 10000 \

        if [[ -f "$metrics_json" ]]; then
            psnr=$(get_metric "$metrics_json" eval_mean_psnr)
            ssim=$(get_metric "$metrics_json" eval_mean_ssim)
            PSNR_MAP["${scene}_${model}"]="$psnr"
            SSIM_MAP["${scene}_${model}"]="$ssim"
            echo "  → PSNR: $psnr dB  |  SSIM: $ssim"
        fi
        echo ""
    done
done

# Paper reference numbers (iNGP Müller et al. 2022, Table 1)
declare -A PAPER_INGP PAPER_ZIP
PAPER_INGP[chair]=35.00; PAPER_INGP[drums]=26.02; PAPER_INGP[ficus]=33.51
PAPER_INGP[hotdog]=37.40; PAPER_INGP[lego]=36.39; PAPER_INGP[materials]=29.78
PAPER_INGP[mic]=36.22; PAPER_INGP[ship]=31.10

{
    echo ""
    echo "================================================================"
    echo "  RESULTS SUMMARY  (iters=$ITERS)"
    echo "================================================================"
    printf "%-10s  %-9s  %-9s  %-8s  %-10s\n" \
        "Scene" "Baseline" "Zip-iNGP" "Delta" "iNGP(paper)"
    printf "%-10s  %-9s  %-9s  %-8s  %-10s\n" \
        "-----" "--------" "--------" "-----" "-----------"

    total_base=0; total_zip=0; n_both=0
    for scene in "${ALL_SCENES[@]}"; do
        base="${PSNR_MAP[${scene}_baseline]:-}"
        zip="${PSNR_MAP[${scene}_zip]:-}"
        paper="${PAPER_INGP[$scene]:-}"
        delta=""
        if [[ -n "$base" && -n "$zip" && "$base" != "N/A" && "$zip" != "N/A" ]]; then
            delta=$(python3 -c "print(f'{float(\"$zip\")-float(\"$base\"):+.2f}')" 2>/dev/null)
            total_base=$(python3 -c "print($total_base+float('$base'))" 2>/dev/null)
            total_zip=$(python3 -c "print($total_zip+float('$zip'))" 2>/dev/null)
            n_both=$((n_both+1))
        fi
        printf "%-10s  %-9s  %-9s  %-8s  %-10s\n" \
            "$scene" "${base:-—}" "${zip:-—}" "${delta:-—}" "${paper:-—}"
    done

    if [[ $n_both -gt 0 ]]; then
        avg_b=$(python3 -c "print(f'{$total_base/$n_both:.2f}')" 2>/dev/null)
        avg_z=$(python3 -c "print(f'{$total_zip/$n_both:.2f}')" 2>/dev/null)
        avg_d=$(python3 -c "print(f'{($total_zip-$total_base)/$n_both:+.2f}')" 2>/dev/null)
        printf "%-10s  %-9s  %-9s  %-8s  %-10s\n" \
            "----------" "---------" "---------" "--------" "-----------"
        printf "%-10s  %-9s  %-9s  %-8s  %-10s\n" \
            "avg" "$avg_b" "$avg_z" "$avg_d" "33.18"
    fi
    echo "================================================================"
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
