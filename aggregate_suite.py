#!/usr/bin/env python3
"""
aggregate_suite.py — Post-run aggregator for the thesis experiment suite.

Reads every metrics.json in the manifest and writes:
  results.csv    flat table, one row per (method, scene, seed, scale, ablation)
  summary.md     human-readable tables (paste into LaTeX thesis)

Called automatically by run_experiments.py, or standalone:
    python aggregate_suite.py --suite_dir results/<id> --runs_dir runs --manifest results/<id>/manifest.jsonl

Aggregation order: per-image metrics -> scene mean (per seed) -> cross-seed
mean ± std (Bessel-corrected) -> cross-scene mean of per-scene means, each
scale aggregated independently.

Significance tests (need scipy, else skipped): per-scene paired t-test
(df=2, low power), overall paired t-test (8 scenes × 3 seeds = 24 pairs),
and unpaired Mann-Whitney U.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SCALES = [1, 2, 4, 8]
METRICS = ['psnr', 'ssim', 'lpips']


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--suite_dir',  required=True, help="Path to results/<suite_id>/")
    p.add_argument('--runs_dir',   required=True, help="Path to runs/ directory")
    p.add_argument('--manifest',   required=True, help="Path to manifest.jsonl")
    return p.parse_args()


# Data loading

def load_manifest(path):
    """Return list of completed run records from manifest.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('status') == 'complete':
                records.append(r)
    return records


def load_metrics(runs_dir, exp_name):
    p = Path(runs_dir) / exp_name / 'metrics.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# Scalar extraction

def get_scene_mean(m, metric, scale):
    """
    Return the scene-mean value for (metric, scale) from a metrics dict.
    Primary source: eval_multiscale (pre-computed mean over test images).
    """
    key = f'{metric}_{scale}x'
    ms  = m.get('eval_multiscale', {})
    return ms.get(key)


def get_per_image_list(m, metric, scale):
    """Return the per-image list for (metric, scale), or [] if not stored."""
    key = f'{scale}x'
    top = f'eval_per_image_{metric}'
    return (m.get(top) or {}).get(key, [])


# Statistics helpers

def sample_mean_std(vals):
    """Return (mean, sample_std) with Bessel correction.  Returns (None,None) if empty."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    n  = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def fmt(mu, sd, ndigits=3):
    """Format mean ± std, e.g. '32.410 ± 0.123'.  Returns 'N/A' if mu is None."""
    if mu is None:
        return 'N/A'
    f = f'.{ndigits}f'
    return f'{mu:{f}} ± {(sd or 0.0):{f}}'


# CSV production

CSV_FIELDS = [
    'suite_id', 'exp_name', 'method', 'scene', 'seed', 'ablation_tag',
    'scale',
    'psnr', 'ssim', 'lpips',
    'training_seconds', 'avg_step_time_ms', 'avg_steps_per_sec',
    'peak_gpu_memory_training_mb', 'peak_gpu_memory_eval_mb',
    'render_time_per_frame_s',
    'iterations', 'zip_collapse_samples', 'zip_no_downweighting', 'zip_n_samples',
]


def build_csv_rows(records, runs_dir, suite_id):
    rows = []
    for rec in records:
        m = load_metrics(runs_dir, rec['exp_name'])
        if m is None:
            continue

        abl = m.get('ablation_config', {})
        cfg = m.get('config', {})

        # Timing / memory (scale-independent)
        shared = {
            'suite_id':    suite_id,
            'exp_name':    rec['exp_name'],
            'method':      rec['method'],
            'scene':       rec['scene'],
            'seed':        rec['seed'],
            'ablation_tag': rec['ablation_tag'],
            'training_seconds':             m.get('total_training_seconds'),
            'avg_step_time_ms':             m.get('avg_step_time_ms'),
            'avg_steps_per_sec':            m.get('avg_steps_per_sec'),
            'peak_gpu_memory_training_mb':  m.get('peak_gpu_memory_training_mb'),
            'peak_gpu_memory_eval_mb':      m.get('peak_gpu_memory_eval_mb'),
            'render_time_per_frame_s':      m.get('eval_render_time_per_frame_s'),
            'iterations':                   cfg.get('iterations'),
            'zip_collapse_samples':         abl.get('zip_collapse_samples', False),
            'zip_no_downweighting':         abl.get('zip_no_downweighting', False),
            'zip_n_samples':                abl.get('zip_n_samples', 6),
        }

        for scale in SCALES:
            row = {**shared, 'scale': scale}
            for metric in METRICS:
                row[metric] = get_scene_mean(m, metric, scale)
            rows.append(row)

    return rows


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: ('' if v is None else v) for k, v in row.items()})
    print(f"CSV written: {path}  ({len(rows)} rows)")


# Data structure for summary

def build_data_index(records, runs_dir):
    """
    Returns a nested dict:
      data[method][scene][seed][scale][metric] = scene_mean_value (float or None)
      per_image[method][scene][seed][scale][metric] = [float, ...]
    """
    data      = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    per_image = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for rec in records:
        m = load_metrics(runs_dir, rec['exp_name'])
        if m is None:
            continue
        method = rec['method']
        scene  = rec['scene']
        seed   = int(rec['seed'])

        for scale in SCALES:
            for metric in METRICS:
                data[method][scene][seed][scale][metric] = get_scene_mean(m, metric, scale)
                per_image[method][scene][seed][scale][metric] = get_per_image_list(m, metric, scale)

    return data, per_image


# Per-method per-scene summary (mean ± std across seeds)

def scene_summary(data, method, scene, scale, metric):
    """Return (mean, std) across seeds for one (method, scene, scale, metric)."""
    vals = [
        data[method][scene][seed][scale].get(metric)
        for seed in data[method][scene]
    ]
    return sample_mean_std(vals)


# Significance tests

def significance_tests(data, per_image, scenes, scale=1):
    """
    Compare zip vs baseline per scene and overall.

    Returns a dict with keys:
      per_scene[scene] = {t_stat, t_pval, mwu_stat, mwu_pval, n_pairs, direction}
      overall           = {t_stat, t_pval, mwu_stat, mwu_pval, n_pairs}
    """
    if not HAS_SCIPY:
        return None

    results = {'per_scene': {}, 'overall': {}}

    baseline_overall = []
    zip_overall      = []

    for scene in scenes:
        b_seeds = sorted(data['baseline'][scene].keys())
        z_seeds = sorted(data['zip'][scene].keys())
        common  = sorted(set(b_seeds) & set(z_seeds))

        b_vals = [data['baseline'][scene][s][scale].get('psnr') for s in common]
        z_vals = [data['zip'][scene][s][scale].get('psnr')      for s in common]

        # Drop pairs where either value is missing
        pairs = [(b, z) for b, z in zip(b_vals, z_vals) if b is not None and z is not None]
        if len(pairs) < 2:
            results['per_scene'][scene] = {'note': f'only {len(pairs)} pairs — test skipped'}
            continue

        b_arr = [p[0] for p in pairs]
        z_arr = [p[1] for p in pairs]

        # Paired t-test (seed-paired)
        t_stat, t_pval = scipy_stats.ttest_rel(z_arr, b_arr)
        # Mann–Whitney U (unpaired, two-sided)
        mwu_stat, mwu_pval = scipy_stats.mannwhitneyu(z_arr, b_arr, alternative='two-sided')

        mu_b = sum(b_arr) / len(b_arr)
        mu_z = sum(z_arr) / len(z_arr)

        results['per_scene'][scene] = {
            'n_pairs':     len(pairs),
            't_stat':      round(float(t_stat), 4),
            't_pval':      round(float(t_pval), 4),
            'mwu_stat':    round(float(mwu_stat), 4),
            'mwu_pval':    round(float(mwu_pval), 4),
            'mean_baseline': round(mu_b, 3),
            'mean_zip':      round(mu_z, 3),
            'delta':         round(mu_z - mu_b, 3),
        }

        baseline_overall.extend(b_arr)
        zip_overall.extend(z_arr)

    # Overall test: treat all (scene × seed) pairs as independent observations
    if len(baseline_overall) >= 2:
        t_stat, t_pval = scipy_stats.ttest_rel(zip_overall, baseline_overall)
        mwu_stat, mwu_pval = scipy_stats.mannwhitneyu(
            zip_overall, baseline_overall, alternative='two-sided')
        results['overall'] = {
            'n_pairs':   len(baseline_overall),
            't_stat':    round(float(t_stat), 4),
            't_pval':    round(float(t_pval), 4),
            'mwu_stat':  round(float(mwu_stat), 4),
            'mwu_pval':  round(float(mwu_pval), 4),
            'note':      'Pairs are (baseline, zip) per scene per seed; df = n_pairs - 1',
        }
    else:
        results['overall'] = {'note': 'insufficient data'}

    return results


# Markdown summary

def pval_stars(pval):
    if pval is None:
        return ''
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return 'ns'


def write_summary_md(path, data, per_image, records, runs_dir, suite_dir, suite_id):
    """Write the human-readable Markdown summary."""

    # Load git info if available
    git_path = os.path.join(suite_dir, 'git_info.json')
    git = {}
    if os.path.exists(git_path):
        with open(git_path) as f:
            git = json.load(f)

    # Determine scenes, seeds, ablation tags present in data
    all_methods = sorted(data.keys())
    # Scenes = union across all methods
    all_scenes = sorted({sc for m in data.values() for sc in m.keys()})
    main_scenes = [s for s in all_scenes if any(
        'main' == rec['ablation_tag'] and rec['scene'] == s and rec['method'] in ('baseline', 'zip')
        for rec in records
    )]
    # Fall back to all scenes if filtering yields nothing
    if not main_scenes:
        main_scenes = all_scenes

    lines = []

    # Header
    lines += [
        f"# Thesis Experiment Results",
        f"",
        f"**Suite ID:** `{suite_id}`  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Git commit:** `{git.get('commit', 'unknown')}`"
        + (" *(dirty)*" if git.get('dirty') else ""),
        f"",
    ]
    if git.get('library_versions'):
        lines.append("**Library versions:**")
        for lib, ver in git['library_versions'].items():
            lines.append(f"- {lib}: {ver}")
        lines.append("")

    lines += [
        "---",
        "",
        "> **Aggregation note:** PSNR/SSIM/LPIPS are averaged arithmetically over all test images",
        "> per scene, then mean ± std computed across seeds using Bessel-corrected (n−1) sample std.",
        "> Cross-scene rows are arithmetic means of per-scene values.",
        "> Each scale is aggregated independently; no values across scales are mixed.",
        "",
    ]

    # Primary comparison table: baseline vs zip, per scale
    lines += ["## Primary Results: Baseline iNGP vs Zip-iNGP", ""]

    for scale in SCALES:
        lines += [f"### Scale {scale}×", ""]
        lines += [f"| Scene | Baseline PSNR | Zip-iNGP PSNR | Δ PSNR "
                  f"| Baseline SSIM | Zip-iNGP SSIM | Baseline LPIPS | Zip-iNGP LPIPS |",
                  f"|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]

        b_psnrs, z_psnrs = [], []
        b_ssims, z_ssims = [], []
        b_lpips, z_lpips = [], []

        for scene in main_scenes:
            bmu, bsd = scene_summary(data, 'baseline', scene, scale, 'psnr')
            zmu, zsd = scene_summary(data, 'zip',      scene, scale, 'psnr')
            delta_psnr = f"{zmu - bmu:+.3f}" if bmu is not None and zmu is not None else 'N/A'

            bsmu, bssd = scene_summary(data, 'baseline', scene, scale, 'ssim')
            zsmu, zssd = scene_summary(data, 'zip',      scene, scale, 'ssim')
            blmu, blsd = scene_summary(data, 'baseline', scene, scale, 'lpips')
            zlmu, zlsd = scene_summary(data, 'zip',      scene, scale, 'lpips')

            lines.append(
                f"| {scene} "
                f"| {fmt(bmu, bsd)} | {fmt(zmu, zsd)} | {delta_psnr} "
                f"| {fmt(bsmu, bssd)} | {fmt(zsmu, zssd)} "
                f"| {fmt(blmu, blsd, 4)} | {fmt(zlmu, zlsd, 4)} |"
            )

            if bmu is not None: b_psnrs.append(bmu)
            if zmu is not None: z_psnrs.append(zmu)
            if bsmu is not None: b_ssims.append(bsmu)
            if zsmu is not None: z_ssims.append(zsmu)
            if blmu is not None: b_lpips.append(blmu)
            if zlmu is not None: z_lpips.append(zlmu)

        # Cross-scene averages row
        def _avg(lst): return sum(lst)/len(lst) if lst else None
        bp_avg, zp_avg = _avg(b_psnrs), _avg(z_psnrs)
        delta_avg = f"{zp_avg - bp_avg:+.3f}" if bp_avg and zp_avg else 'N/A'
        lines += [
            f"|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **Mean** "
            f"| **{bp_avg:.3f}** | **{zp_avg:.3f}** | **{delta_avg}** "
            f"| **{_avg(b_ssims):.3f}** | **{_avg(z_ssims):.3f}** "
            f"| **{_avg(b_lpips):.4f}** | **{_avg(z_lpips):.4f}** |"
            if bp_avg else "| **Mean** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |",
            "",
        ]

    # Significance tests
    lines += ["## Statistical Significance Tests (Zip-iNGP vs Baseline)", ""]

    if not HAS_SCIPY:
        lines += ["> **scipy not available** — install scipy to enable significance tests.", ""]
    else:
        sig = significance_tests(data, per_image, main_scenes, scale=1)
        if sig:
            lines += [
                "Tests are on 1× PSNR.  ",
                "**Paired t-test** pairs (baseline, zip) by seed per scene.  ",
                "**Mann–Whitney U** is non-parametric and unpaired.  ",
                "Stars: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` p≥0.05.  ",
                "⚠ With only 3 seeds, df=2 — t-test power is low per scene.  ",
                "  The overall test pools all (scene × seed) pairs (n={}).".format(
                    sig.get('overall', {}).get('n_pairs', '?')),
                "",
                "| Scene | n pairs | Δ PSNR | Paired t | p-value | MWU p | Sig |",
                "|-------|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
            for scene in main_scenes:
                r = sig['per_scene'].get(scene, {})
                if 'note' in r and 't_stat' not in r:
                    lines.append(f"| {scene} | — | — | — | — | — | {r.get('note','')} |")
                else:
                    lines.append(
                        f"| {scene} "
                        f"| {r['n_pairs']} "
                        f"| {r['delta']:+.3f} "
                        f"| {r['t_stat']:.3f} "
                        f"| {r['t_pval']:.4f} "
                        f"| {r['mwu_pval']:.4f} "
                        f"| {pval_stars(r['t_pval'])} |"
                    )

            ov = sig.get('overall', {})
            if 't_stat' in ov:
                lines += [
                    "|-------|:---:|:---:|:---:|:---:|:---:|:---:|",
                    f"| **Overall** "
                    f"| {ov['n_pairs']} "
                    f"| — "
                    f"| {ov['t_stat']:.3f} "
                    f"| {ov['t_pval']:.4f} "
                    f"| {ov['mwu_pval']:.4f} "
                    f"| {pval_stars(ov['t_pval'])} |",
                ]
            lines.append("")

    # Ablation table (1× PSNR): determine which ablation methods appear in data
    abl_methods = [m for m in all_methods
                   if m not in ('baseline', 'zip')]
    if abl_methods:
        lines += ["## Ablation Results (1× PSNR, mean ± std across seeds)", ""]

        # Header: scene + one column per ablation method + zip reference
        headers  = ['Scene', 'zip (ref)'] + abl_methods
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

        abl_scenes = sorted({sc for m in abl_methods for sc in data[m].keys()})
        for scene in abl_scenes:
            zmu, zsd = scene_summary(data, 'zip', scene, 1, 'psnr')
            row = [scene, fmt(zmu, zsd)]
            for abl in abl_methods:
                mu, sd = scene_summary(data, abl, scene, 1, 'psnr')
                delta = f" ({mu - zmu:+.2f})" if mu is not None and zmu is not None else ""
                row.append(fmt(mu, sd) + delta)
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append("")

        # n_samples sweep sub-table
        k_methods = sorted([m for m in abl_methods if m.startswith('zip_k')])
        if k_methods:
            lines += ["### Number-of-Samples Sweep (1× PSNR)", ""]
            k_all = ['zip_k1', 'zip_k3', 'zip'] + [m for m in k_methods if m not in ('zip_k1','zip_k3')] + ['zip_k12']
            k_present = [m for m in k_all if m in data]
            k_labels  = [m.replace('zip_k','K=').replace('zip','K=6 (ref)') for m in k_present]
            abl_scenes_for_k = sorted({sc for m in k_present for sc in data[m].keys()})
            lines.append('| Scene | ' + ' | '.join(k_labels) + ' |')
            lines.append('|' + '|'.join(['---']*(len(k_labels)+1)) + '|')
            for scene in abl_scenes_for_k:
                row = [scene]
                for m in k_present:
                    mu, sd = scene_summary(data, m, scene, 1, 'psnr')
                    row.append(fmt(mu, sd))
                lines.append('| ' + ' | '.join(row) + ' |')
            lines.append("")

    # Timing and memory
    lines += ["## Timing and Memory", ""]
    timing_rows = []
    seen_timing = set()
    for rec in records:
        key = (rec['method'], rec['scene'], rec['ablation_tag'])
        if key in seen_timing:
            continue
        m = load_metrics(runs_dir, rec['exp_name'])
        if m is None:
            continue
        seen_timing.add(key)
        timing_rows.append({
            'method':       rec['method'],
            'scene':        rec['scene'],
            'ablation':     rec['ablation_tag'],
            'train_s':      m.get('total_training_seconds'),
            'step_ms':      m.get('avg_step_time_ms'),
            'steps_per_s':  m.get('avg_steps_per_sec'),
            'gpu_train_mb': m.get('peak_gpu_memory_training_mb'),
            'render_fps':   (1.0 / m['eval_render_time_per_frame_s']
                             if m.get('eval_render_time_per_frame_s') else None),
        })
    if timing_rows:
        lines += [
            "| Method | Scene | Ablation | Train time (min) | Step (ms) | GPU mem (MB) | Render FPS |",
            "|--------|-------|----------|:---:|:---:|:---:|:---:|",
        ]
        for r in timing_rows:
            train_min = f"{r['train_s']/60:.1f}" if r['train_s'] else 'N/A'
            step_ms   = f"{r['step_ms']:.1f}"  if r['step_ms'] else 'N/A'
            gpu_mb    = f"{r['gpu_train_mb']:.0f}" if r['gpu_train_mb'] else 'N/A'
            fps       = f"{r['render_fps']:.2f}" if r['render_fps'] else 'N/A'
            lines.append(
                f"| {r['method']} | {r['scene']} | {r['ablation']} "
                f"| {train_min} | {step_ms} | {gpu_mb} | {fps} |"
            )
        lines.append("")

    # Footer
    lines += [
        "---",
        "",
        f"*Generated by `aggregate_suite.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Summary written: {path}")


# Main

def main():
    args = parse_args()

    suite_id  = Path(args.suite_dir).name
    csv_path  = os.path.join(args.suite_dir, 'results.csv')
    md_path   = os.path.join(args.suite_dir, 'summary.md')

    print(f"\n{'='*60}")
    print(f"  aggregate_suite.py  |  suite: {suite_id}")
    print(f"{'='*60}")

    records = load_manifest(args.manifest)
    print(f"  Loaded {len(records)} completed runs from manifest")

    if not records:
        print("  No completed runs found. Nothing to aggregate.")
        return

    # Build flat CSV
    rows = build_csv_rows(records, args.runs_dir, suite_id)
    write_csv(rows, csv_path)

    # Build data index for summary
    data, per_image = build_data_index(records, args.runs_dir)

    # Write Markdown summary
    write_summary_md(md_path, data, per_image, records, args.runs_dir,
                     args.suite_dir, suite_id)

    # Print a quick console preview
    all_scenes = sorted({sc for m in data.values() for sc in m.keys()})
    methods    = sorted(data.keys())
    print(f"\n  Quick preview — 1× PSNR (mean across seeds):")
    print(f"  {'Scene':<10}", end='')
    for meth in methods:
        print(f"  {meth:<12}", end='')
    print()
    for scene in all_scenes:
        print(f"  {scene:<10}", end='')
        for meth in methods:
            vals = [data[meth][scene][s][1].get('psnr')
                    for s in data[meth][scene] if data[meth][scene][s][1].get('psnr') is not None]
            if vals:
                mu = sum(vals)/len(vals)
                print(f"  {mu:>8.3f}    ", end='')
            else:
                print(f"  {'N/A':>8}    ", end='')
        print()

    print(f"\n  CSV:     {csv_path}")
    print(f"  Summary: {md_path}")
    print()

    if not HAS_SCIPY:
        print("  WARNING: scipy not installed — significance tests skipped.")
        print("           Install with: pip install scipy")


if __name__ == '__main__':
    main()
