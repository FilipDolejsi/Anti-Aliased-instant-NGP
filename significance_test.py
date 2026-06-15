"""
Statistical significance: baseline vs Zip-iNGP, NeRF Blender synthetic suite
(suite_20260608_233928, 8 scenes x 3 seeds, 75K-iter runs).

Data below is transcribed directly from runs/suite_20260608_233928_{baseline,zip}_*/metrics.json
('eval_multiscale' dict), scales 1x/2x/4x/8x.
"""

import numpy as np
import pandas as pd
from scipy import stats

# RAW DATA  (scene, seed, psnr_1x..8x, ssim_1x..8x, lpips_1x..8x)

COLUMNS = ['scene', 'seed',
           'psnr_1x', 'psnr_2x', 'psnr_4x', 'psnr_8x',
           'ssim_1x', 'ssim_2x', 'ssim_4x', 'ssim_8x',
           'lpips_1x', 'lpips_2x', 'lpips_4x', 'lpips_8x']

BASELINE = [
    ('chair', 42,   32.841, 34.012, 34.009, 33.970,  0.9751, 0.9840, 0.9870, 0.9885,  0.0451, 0.0278, 0.0206, 0.0141),
    ('chair', 1337, 32.741, 33.898, 33.892, 33.844,  0.9746, 0.9837, 0.9868, 0.9884,  0.0436, 0.0262, 0.0196, 0.0138),
    ('chair', 2026, 32.856, 34.018, 34.021, 33.974,  0.9749, 0.9837, 0.9867, 0.9883,  0.0478, 0.0303, 0.0230, 0.0155),
    ('drums', 42,   25.039, 25.436, 25.424, 25.444,  0.9128, 0.9167, 0.9211, 0.9315,  0.2129, 0.1811, 0.1593, 0.1154),
    ('drums', 1337, 25.102, 25.507, 25.497, 25.511,  0.9161, 0.9210, 0.9256, 0.9354,  0.2019, 0.1698, 0.1492, 0.1074),
    ('drums', 2026, 25.123, 25.528, 25.519, 25.541,  0.9164, 0.9212, 0.9261, 0.9362,  0.2090, 0.1749, 0.1530, 0.1094),
    ('ficus', 42,   30.253, 30.379, 30.363, 30.427,  0.9697, 0.9774, 0.9812, 0.9833,  0.0752, 0.0604, 0.0487, 0.0296),
    ('ficus', 1337, 30.257, 30.385, 30.368, 30.433,  0.9693, 0.9771, 0.9808, 0.9829,  0.0868, 0.0738, 0.0606, 0.0364),
    ('ficus', 2026, 30.237, 30.362, 30.347, 30.408,  0.9694, 0.9774, 0.9812, 0.9833,  0.0738, 0.0583, 0.0456, 0.0277),
    ('hotdog', 42,   34.990, 35.633, 35.626, 35.671, 0.9750, 0.9800, 0.9840, 0.9886,  0.0537, 0.0313, 0.0198, 0.0117),
    ('hotdog', 1337, 34.803, 35.429, 35.428, 35.452, 0.9745, 0.9796, 0.9837, 0.9883,  0.0547, 0.0316, 0.0201, 0.0121),
    ('hotdog', 2026, 34.996, 35.645, 35.646, 35.681, 0.9755, 0.9806, 0.9845, 0.9889,  0.0524, 0.0297, 0.0188, 0.0114),
    ('lego', 42,   32.390, 33.045, 33.038, 33.003,   0.9698, 0.9800, 0.9850, 0.9881,  0.0563, 0.0388, 0.0322, 0.0225),
    ('lego', 1337, 32.374, 33.026, 33.019, 32.993,   0.9695, 0.9801, 0.9853, 0.9884,  0.0547, 0.0348, 0.0279, 0.0191),
    ('lego', 2026, 32.299, 32.945, 32.940, 32.911,   0.9689, 0.9795, 0.9846, 0.9877,  0.0562, 0.0379, 0.0319, 0.0230),
    ('materials', 42,   27.876, 28.218, 28.214, 28.211, 0.9322, 0.9495, 0.9647, 0.9765, 0.1595, 0.1237, 0.0880, 0.0493),
    ('materials', 1337, 27.933, 28.273, 28.272, 28.267, 0.9334, 0.9503, 0.9653, 0.9767, 0.1535, 0.1187, 0.0839, 0.0463),
    ('materials', 2026, 27.772, 28.102, 28.100, 28.109, 0.9303, 0.9474, 0.9630, 0.9753, 0.1575, 0.1233, 0.0886, 0.0496),
    ('mic', 42,   33.013, 33.806, 33.827, 33.854,  0.9839, 0.9888, 0.9913, 0.9932,  0.0317, 0.0174, 0.0136, 0.0094),
    ('mic', 1337, 33.076, 33.885, 33.899, 33.944,  0.9842, 0.9890, 0.9916, 0.9934,  0.0309, 0.0173, 0.0139, 0.0095),
    ('mic', 2026, 33.080, 33.897, 33.915, 33.943,  0.9837, 0.9888, 0.9915, 0.9934,  0.0320, 0.0177, 0.0141, 0.0099),
    ('ship', 42,   26.129, 26.626, 26.620, 26.635,  0.8545, 0.8663, 0.8737, 0.8891,  0.2329, 0.1793, 0.1395, 0.1018),
    ('ship', 1337, 26.147, 26.642, 26.636, 26.654,  0.8537, 0.8653, 0.8728, 0.8883,  0.2337, 0.1802, 0.1414, 0.1041),
    ('ship', 2026, 26.160, 26.660, 26.654, 26.670,  0.8552, 0.8673, 0.8748, 0.8901,  0.2279, 0.1743, 0.1363, 0.1002),
]

ZIP = [
    ('chair', 42,   28.810, 29.666, 29.657, 29.630,  0.9463, 0.9562, 0.9567, 0.9554,  0.0879, 0.0642, 0.0566, 0.0550),
    ('chair', 1337, 28.033, 28.816, 28.794, 28.839,  0.9383, 0.9467, 0.9465, 0.9454,  0.0889, 0.0681, 0.0637, 0.0633),
    ('chair', 2026, 29.292, 30.165, 30.147, 30.136,  0.9528, 0.9646, 0.9662, 0.9658,  0.0752, 0.0510, 0.0444, 0.0437),
    ('drums', 42,   23.404, 23.687, 23.680, 23.694,  0.8975, 0.9073, 0.9179, 0.9319,  0.1397, 0.1004, 0.0890, 0.0686),
    ('drums', 1337, 23.509, 23.798, 23.791, 23.804,  0.8989, 0.9087, 0.9194, 0.9332,  0.1213, 0.0896, 0.0833, 0.0667),
    ('drums', 2026, 23.366, 23.649, 23.645, 23.656,  0.8976, 0.9072, 0.9178, 0.9315,  0.1221, 0.0915, 0.0846, 0.0679),
    ('ficus', 42,   27.536, 27.546, 27.536, 27.558,  0.9512, 0.9611, 0.9669, 0.9713,  0.0851, 0.0537, 0.0415, 0.0294),
    ('ficus', 1337, 27.531, 27.529, 27.515, 27.542,  0.9519, 0.9617, 0.9677, 0.9721,  0.0559, 0.0387, 0.0306, 0.0251),
    ('ficus', 2026, 27.800, 27.800, 27.786, 27.810,  0.9520, 0.9619, 0.9680, 0.9726,  0.0632, 0.0449, 0.0350, 0.0268),
    ('hotdog', 42,   31.000, 31.424, 31.429, 31.447, 0.9568, 0.9575, 0.9584, 0.9648,  0.0809, 0.0658, 0.0577, 0.0454),
    ('hotdog', 1337, 30.645, 31.066, 31.070, 31.106, 0.9554, 0.9559, 0.9568, 0.9634,  0.0827, 0.0672, 0.0589, 0.0461),
    ('hotdog', 2026, 30.839, 31.245, 31.245, 31.300, 0.9557, 0.9559, 0.9566, 0.9635,  0.0805, 0.0654, 0.0577, 0.0465),
    ('lego', 42,   29.241, 29.665, 29.666, 29.654,   0.9492, 0.9613, 0.9684, 0.9733,  0.0791, 0.0443, 0.0347, 0.0302),
    ('lego', 1337, 28.245, 28.813, 28.810, 28.729,   0.9339, 0.9505, 0.9633, 0.9723,  0.1021, 0.0556, 0.0350, 0.0293),
    ('lego', 2026, 28.462, 28.935, 28.932, 28.875,   0.9444, 0.9563, 0.9635, 0.9686,  0.0798, 0.0471, 0.0378, 0.0331),
    ('materials', 42,   26.607, 26.840, 26.837, 26.843, 0.9191, 0.9329, 0.9508, 0.9693, 0.0963, 0.0700, 0.0544, 0.0357),
    ('materials', 1337, 26.512, 26.745, 26.739, 26.755, 0.9154, 0.9287, 0.9480, 0.9684, 0.1013, 0.0759, 0.0618, 0.0399),
    ('materials', 2026, 26.348, 26.570, 26.565, 26.576, 0.9149, 0.9288, 0.9478, 0.9674, 0.1054, 0.0788, 0.0608, 0.0390),
    ('mic', 42,   30.353, 31.225, 31.214, 31.300,  0.9754, 0.9820, 0.9858, 0.9888,  0.0364, 0.0205, 0.0179, 0.0140),
    ('mic', 1337, 30.577, 31.295, 31.300, 31.293,  0.9730, 0.9797, 0.9841, 0.9878,  0.0412, 0.0244, 0.0208, 0.0163),
    ('mic', 2026, 31.328, 32.067, 32.066, 32.087,  0.9749, 0.9820, 0.9866, 0.9902,  0.0352, 0.0197, 0.0167, 0.0129),
    ('ship', 42,   26.431, 27.053, 27.043, 27.076,  0.8481, 0.8619, 0.8719, 0.8887,  0.2217, 0.1558, 0.1153, 0.0799),
    ('ship', 1337, 26.489, 27.113, 27.106, 27.125,  0.8485, 0.8629, 0.8732, 0.8903,  0.2133, 0.1469, 0.1073, 0.0744),
    ('ship', 2026, 26.667, 27.321, 27.316, 27.349,  0.8513, 0.8650, 0.8745, 0.8909,  0.2201, 0.1531, 0.1135, 0.0778),
]

df_base = pd.DataFrame(BASELINE, columns=COLUMNS)
df_zip = pd.DataFrame(ZIP, columns=COLUMNS)

assert len(df_base) == 24 and len(df_zip) == 24
assert (df_base[['scene', 'seed']].values == df_zip[['scene', 'seed']].values).all()

METRICS = ['psnr', 'ssim', 'lpips']
SCALES = ['1x', '2x', '4x', '8x']
LOWER_IS_BETTER = {'psnr': False, 'ssim': False, 'lpips': True}


# Helpers

def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction. Returns corrected p-values in
    the original order of `pvals`."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (n - rank)
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def paired_stats(zip_vals, base_vals):
    """Return dict of paired-difference statistics for (zip - base)."""
    diff = np.asarray(zip_vals) - np.asarray(base_vals)
    n = len(diff)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    sem_diff = sd_diff / np.sqrt(n)

    t_stat, t_p = stats.ttest_rel(zip_vals, base_vals)

    # Wilcoxon: undefined if all differences are zero; not the case here.
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_lo = mean_diff - tcrit * sem_diff
    ci_hi = mean_diff + tcrit * sem_diff

    cohens_d = mean_diff / sd_diff if sd_diff > 0 else np.nan

    return dict(n=n, mean_diff=mean_diff, ci_lo=ci_lo, ci_hi=ci_hi,
                 t_stat=t_stat, t_p=t_p, w_stat=w_stat, w_p=w_p,
                 cohens_d=cohens_d)


def winner_label(metric, mean_diff, p_holm, alpha=0.05):
    """Decide a plain-language 'winner' label for the summary table."""
    lower_better = LOWER_IS_BETTER[metric]
    # positive mean_diff = zip > baseline
    zip_better = (mean_diff < 0) if lower_better else (mean_diff > 0)
    sig = p_holm < alpha
    side = "Zip" if zip_better else "Baseline"
    if sig:
        return f"{side} (p_holm={p_holm:.2g})"
    return f"{side} (n.s., p_holm={p_holm:.2g})"


def build_table(df_b, df_z, metric, label, n_for_holm=12, holm_pvals=None):
    """Build the per-metric summary table (rows = scales)."""
    rows = []
    for s in SCALES:
        col = f"{metric}_{s}"
        st = paired_stats(df_z[col].values, df_b[col].values)
        rows.append(st)
    return rows


# 1. PRIMARY analysis A — n=8, pairing on per-scene MEANS (average 3 seeds)

scene_mean_base = df_base.groupby('scene')[
    [f'{m}_{s}' for m in METRICS for s in SCALES]].mean().reset_index()
scene_mean_zip = df_zip.groupby('scene')[
    [f'{m}_{s}' for m in METRICS for s in SCALES]].mean().reset_index()
scene_mean_base = scene_mean_base.sort_values('scene').reset_index(drop=True)
scene_mean_zip = scene_mean_zip.sort_values('scene').reset_index(drop=True)
assert (scene_mean_base['scene'].values == scene_mean_zip['scene'].values).all()

print("=" * 100)
print("PRIMARY ANALYSIS (n=8): paired across 8 scenes, each value = mean over 3 seeds")
print("=" * 100)

# Collect all 12 raw p-values (t-test) across metric x scale for the joint Holm-Bonferroni correction
order_keys = [(m, s) for m in METRICS for s in SCALES]

stats_n8 = {}
for m, s in order_keys:
    col = f"{m}_{s}"
    stats_n8[(m, s)] = paired_stats(scene_mean_zip[col].values, scene_mean_base[col].values)

raw_p_n8 = [stats_n8[k]['t_p'] for k in order_keys]
holm_p_n8 = holm_bonferroni(raw_p_n8)
for k, hp in zip(order_keys, holm_p_n8):
    stats_n8[k]['t_p_holm'] = hp

for m in METRICS:
    print(f"\n--- {m.upper()}  ({'lower=better' if LOWER_IS_BETTER[m] else 'higher=better'}) ---")
    header = (f"{'scale':<6}{'mean_diff(zip-base)':>20}{'95% CI':>22}"
              f"{'t-p (raw)':>11}{'t-p (Holm)':>12}{'Wilcoxon-p':>12}{'Cohen d':>9}  winner")
    print(header)
    for s in SCALES:
        st = stats_n8[(m, s)]
        ci = f"[{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]"
        winner = winner_label(m, st['mean_diff'], st['t_p_holm'])
        print(f"{s:<6}{st['mean_diff']:>20.4f}{ci:>22}{st['t_p']:>11.2e}"
              f"{st['t_p_holm']:>12.2e}{st['w_p']:>12.2e}{st['cohens_d']:>9.2f}  {winner}")


# 2. SUPPORTING analysis — n=24, paired across all (scene, seed) pairs

print("\n\n" + "=" * 100)
print("SUPPORTING ANALYSIS (n=24): paired across all 24 (scene, seed) pairs")
print("=" * 100)

stats_n24 = {}
for m, s in order_keys:
    col = f"{m}_{s}"
    stats_n24[(m, s)] = paired_stats(df_zip[col].values, df_base[col].values)

raw_p_n24 = [stats_n24[k]['t_p'] for k in order_keys]
holm_p_n24 = holm_bonferroni(raw_p_n24)
for k, hp in zip(order_keys, holm_p_n24):
    stats_n24[k]['t_p_holm'] = hp

for m in METRICS:
    print(f"\n--- {m.upper()}  ({'lower=better' if LOWER_IS_BETTER[m] else 'higher=better'}) ---")
    header = (f"{'scale':<6}{'mean_diff(zip-base)':>20}{'95% CI':>22}"
              f"{'t-p (raw)':>11}{'t-p (Holm)':>12}{'Wilcoxon-p':>12}{'Cohen d':>9}  winner")
    print(header)
    for s in SCALES:
        st = stats_n24[(m, s)]
        ci = f"[{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]"
        winner = winner_label(m, st['mean_diff'], st['t_p_holm'])
        print(f"{s:<6}{st['mean_diff']:>20.4f}{ci:>22}{st['t_p']:>11.2e}"
              f"{st['t_p_holm']:>12.2e}{st['w_p']:>12.2e}{st['cohens_d']:>9.2f}  {winner}")


# 3. SECONDARY — per-scene paired t-test, n=3 (DESCRIPTIVE ONLY)

print("\n\n" + "=" * 100)
print("SECONDARY (DESCRIPTIVE ONLY): per-scene paired t-test across 3 seeds (n=3)")
print("*** WARNING: n=3 gives negligible statistical power. p-values below are")
print("*** included ONLY to show per-scene direction & effect size. Do NOT use")
print("*** them to claim significance for any individual scene.")
print("=" * 100)

scenes = sorted(df_base['scene'].unique())
for m in METRICS:
    print(f"\n--- {m.upper()} @ 1x  ({'lower=better' if LOWER_IS_BETTER[m] else 'higher=better'}) ---")
    print(f"{'scene':<12}{'mean_diff(zip-base)':>20}{'t-p (n=3, descr.)':>20}{'Cohen d':>9}")
    for scene in scenes:
        b = df_base[df_base.scene == scene][f'{m}_1x'].values
        z = df_zip[df_zip.scene == scene][f'{m}_1x'].values
        st = paired_stats(z, b)
        print(f"{scene:<12}{st['mean_diff']:>20.4f}{st['t_p']:>20.2e}{st['cohens_d']:>9.2f}")


# 4. Save tidy CSV of per-pair differences for appendix / reproducibility

diff_rows = []
for m in METRICS:
    for s in SCALES:
        col = f"{m}_{s}"
        for i in range(24):
            diff_rows.append({
                'scene': df_base.loc[i, 'scene'],
                'seed': df_base.loc[i, 'seed'],
                'metric': m,
                'scale': s,
                'baseline': df_base.loc[i, col],
                'zip': df_zip.loc[i, col],
                'diff_zip_minus_base': df_zip.loc[i, col] - df_base.loc[i, col],
            })
pd.DataFrame(diff_rows).to_csv('scripts/significance_pairwise_diffs.csv', index=False)
print("\n\nSaved per-pair differences to scripts/significance_pairwise_diffs.csv")
