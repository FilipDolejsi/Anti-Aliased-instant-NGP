# Anti-aliased iNGP (AAiNGP)

Filip Dolejší (k12317783)
Supervisor: Mgr. Eric Volkmann
Institute of Machine Learning, JKU Linz
BSc thesis, Artificial Intelligence, 2026

## Overview

This repository contains the implementation and experiment suite for the
thesis. The code reimplements Instant NGP (Müller et al., 2022) in PyTorch
with an NVIDIA Warp kernel backend, and augments the hash grid encoder with
Zip-NeRF's (Barron et al., 2023) cone-based hexagonal multisampling and
erf-based feature downweighting (AAiNGP). The two models are benchmarked
against each other on the Blender synthetic dataset at four evaluation
scales (1x, 2x, 4x, 8x), three seeds per scene. An ablation suite isolates
the contribution of each mechanism added on top of the iNGP baseline.

All results reported in the thesis were produced on a single NVIDIA Tesla
V100 (16 GB). The ablation suite was additionally run on an RTX 2080 Ti
(11 GB). The Warp backend is the default and the only backend used for
reported numbers; a `--pytorch` fallback path exists in the encoder and
renderer but produces numerically different results due to differences in
random sampling order and is not used for any result in the thesis.

## Requirements

`environment.yml` is the authoritative dependency specification, pinned to
the versions recorded for the runs underlying the thesis results
(Python 3.9, PyTorch 2.8.0+cu128, NumPy 1.26.4, SciPy 1.13.1,
scikit-image 0.24.0, PyYAML 6.0.3, warp-lang 1.11.0, lpips 0.1.4).
`requirements.txt` is a pip-compatible mirror for use without conda; PyTorch
must be installed separately from the CUDA 12.8 wheel index as noted in that
file.

```bash
conda env create -f environment.yml
conda activate torch-warp-ngp-env
```

The Blender synthetic dataset (lego, chair, drums, ficus, hotdog, materials,
mic, ship) is not included in this repository and must be obtained
separately (NeRF, Mildenhall et al., 2020). Set `data_root` in
`experiment_config.yaml`, and `DATA_ROOT` in `eval_blender.sh` and
`run_thesis_eval.sh`, to its location.

## Reproducing Results

The most effective way to run the full pipeline is to:

```bash
python run_experiments.py --profile full
```

Baseline iNGP, all 8 Blender scenes, 3 seeds:

```bash
bash eval_blender.sh --method baseline --seeds 42 1337 2026
```

AAiNGP main configuration (K=6 hexagonal samples, sigma_scale=1.0, hash-norm
weight decay with global-mean reduction):

```bash
bash eval_blender.sh --method zip --seeds 42 1337 2026
```

Each invocation trains and evaluates one (scene, seed) combination per run
and writes `metrics.json` under `runs/`. After all runs complete, aggregate
into the thesis tables:

```bash
python aggregate_thesis.py --results_dir results/
```

## Ablation Studies

Each ablation modifies exactly one flag relative to the AAiNGP main
configuration. All flags are defined in `experiment_config.yaml`; the
per-variant flag table in the thesis appendix corresponds directly to the
`ablations` entries in that file. The variants are:

```
zip_no_dw:     --zip_no_downweighting
zip_collapse:  --zip_collapse_samples
zip_k1:        --zip_n_samples 1
zip_k3:        --zip_n_samples 3
zip_sigma05:   --zip_sigma_scale 0.5
zip_nwd_paper: --zip_nwd_per_level
```

The ablation suite was run on lego and drums, three seeds each:

```bash
python run_experiments.py --config experiment_config.yaml \
    --scenes lego drums --ablations all
```

## Significance Testing

```bash
python scripts/significance_test.py \
    --results_dir results/ \
    --output scripts/significance_pairwise_diffs.csv
```

This reproduces Table 4: paired t-test and Wilcoxon signed-rank test per
metric-scale combination (12 combinations: PSNR/SSIM/LPIPS x 4 scales),
n=8 scenes, with Holm-Bonferroni correction across the 12 comparisons.

## Implementation Notes

`RANDOM_BG_TRAIN` is not exposed as a CLI flag and is set in `train.py`. It
is `True` for the baseline (random background composited per training step)
and `False` for all AAiNGP variants (fixed white background), matching the
Zip-NeRF training protocol. This difference is a confound between baseline
and AAiNGP and is discussed in the thesis Limitations section.

PSNR is computed per image and then averaged over the test set, not from the
mean squared error over the full set. SSIM is computed with `data_range=1.0`.
LPIPS uses the VGG backbone with both images remapped from [0, 1] to
[-1, 1]. For the multiscale evaluation, the prediction and ground truth are
each downsampled by bicubic interpolation by factors {1, 2, 4, 8} before any
metric is computed; scale 1x corresponds to no downsampling.

## Expected Output

For PSNR at scale 1x, averaged over 8 scenes and 3 seeds, the baseline
reached 30.31 dB and AAiNGP reached 27.88 dB. The gap between the two models
is approximately constant across all four evaluation scales. Per-scene
breakdowns are given in thesis Table 2.
