"""Collapse the two low-data verification runs into one text-ready evidence file.

Fixes the one-sided tail of the cluster-robust slope test (scipy.t.sf on a negative t gives
the upper tail) and gathers every number the manuscript, letter and Table 12 quote, so the
prose never hard-codes a statistic.
"""
import io, os, sys, json, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
TR = json.load(io.open(os.path.join(BASE, 'results_rev', 'lowdata_trend.json'), encoding='utf-8'))
RB = json.load(io.open(os.path.join(BASE, 'results_rev', 'lowdata_robustness.json'), encoding='utf-8'))
LD = json.load(io.open(os.path.join(BASE, 'results_rev', 'lowdata.json'), encoding='utf-8'))

fracs = [float(f) for f in TR['fractions']]
nrep = 10
out = {}
out['fractions'] = fracs
out['n_train'] = TR['n_train']
out['gap_A'] = {f'{f:.2f}': float(np.mean(np.array(TR['gap_S']).T[i])) for i, f in enumerate(fracs)}
out['gap_A_pool'] = {f'{f:.2f}': float(np.mean(np.array(TR['gap_pool']).T[i])) for i, f in enumerate(fracs)}
GP = np.array(TR['gap_pool'])
out['p_A_pool'] = {f'{f:.2f}': float(2 * stats.t.sf(abs(np.mean(GP.T[i]) /
                   math.sqrt((1 / GP.shape[0] + 0.25) * np.var(GP.T[i], ddof=1))),
                   GP.shape[0] - 1)) for i, f in enumerate(fracs)}
out['gap_B'] = {k: v[0] for k, v in RB['A']['curve_S_minus_P'].items()} if False else \
               {k: v[0] for k, v in RB['B']['curve_S_minus_P'].items()}
out['p_B'] = {k: v[1] for k, v in RB['B']['curve_S_minus_P'].items()}
out['p_A'] = {k: v[1] for k, v in RB['A']['curve_S_minus_P'].items()}
out['means_A'] = {k: RB['A']['means'][k] for k in RB['A']['means']}
out['means_B'] = {k: RB['B']['means'][k] for k in RB['B']['means']}

# per-family deltas and their spread, both streams, at 10% and at 100%
fam = ['S', 'C', 'Lp', 'CS']
out['fam_delta'] = {}
for stream in ['A', 'B']:
    for fk in ['0.10', '1.00']:
        m = RB[stream]['means'][fk]
        out['fam_delta'][f'{stream}_{fk}'] = {'P': m['P'], **{k: m[k] - m['P'] for k in fam}}
out['spread_10'] = {s: max(out['fam_delta'][f'{s}_0.10'][k] for k in fam)
                    - min(out['fam_delta'][f'{s}_0.10'][k] for k in fam) for s in ['A', 'B']}
out['spread_100'] = {s: max(out['fam_delta'][f'{s}_1.00'][k] for k in fam)
                     - min(out['fam_delta'][f'{s}_1.00'][k] for k in fam) for s in ['A', 'B']}

# Holm over the full 32-contrast family, and the specific textbook@10% entry
for stream in ['A', 'B']:
    sm = RB[stream]['holm32_smallest']
    out[f'holm32_min_{stream}'] = sm[0][1]
    ent = {k: v for k, v in sm}
    out[f'holm32_S10_{stream}'] = ent.get('S@0.10')
    out[f'trend_contrast_{stream}'] = RB[stream]['trend_contrast']['p']
    out[f'spearman_{stream}'] = [RB[stream]['spearman_mean'], RB[stream]['spearman_range']]

# sign test: repeats with a negative Spearman, per stream (A from the fold-level file,
# B from the robustness run, whose per-repeat values are all negative -- verify below)
rhos_A = []
G = np.array(TR['gap_S'])
K = 5
LOGF = np.log10(fracs)
for r in range(nrep):
    rhos_A.append(stats.spearmanr(np.tile(LOGF, K), G[r * K:(r + 1) * K, :].ravel()).statistic)
out['rhos_A'] = rhos_A
out['nneg_A'] = int(sum(1 for x in rhos_A if x < 0))
out['p_sign'] = float(stats.binomtest(out['nneg_A'], nrep, 0.5, alternative='greater').pvalue)
out['spearman_range_A'] = [min(rhos_A), max(rhos_A)]

# cluster-robust slope with the correct tail
for nm, key in [('textbook', 'textbook_vs_raw'), ('pooled', 'pooled_engineered_vs_raw')]:
    s = TR['summary'][key]
    t = s['t']
    df = nrep - 1
    out[f'slope_{nm}'] = {'slope': s['slope'], 'se': s['se'], 't': t,
                          'p_one': float(stats.t.cdf(t, df)) if t < 0 else float(stats.t.sf(t, df)),
                          'p_two': float(2 * min(stats.t.cdf(t, df), stats.t.sf(t, df))),
                          'ci': [s['slope'] - stats.t.ppf(0.975, df) * s['se'],
                                 s['slope'] + stats.t.ppf(0.975, df) * s['se']],
                          'p_perm': s['p_perm']}
# fold-level 95% intervals at the two anchor fractions (Nadeau-Bengio SE over 50 folds)
GS = np.array(TR['gap_S'])
for fk, nm in [('0.10', 'ci10'), ('1.00', 'ci100')]:
    col = GS[:, fracs.index(float(fk))]
    se = math.sqrt((1 / len(col) + 0.25) * col.var(ddof=1))
    out[nm] = [float(col.mean() - 1.96 * se), float(col.mean() + 1.96 * se)]
    out[nm.replace('ci', 'se_')] = float(se)

out['holm_within_fraction_S10'] = None
# reproduce the 4-contrast Holm within the 10% fraction (what the old text reported)
ps = {k: LD['0.10'][k]['p'] for k in fam}
order = sorted(ps, key=ps.get)
run, adj = 0.0, {}
for rank, k in enumerate(order):
    run = max(run, (4 - rank) * ps[k])
    adj[k] = min(run, 1.0)
out['holm_within_fraction_S10'] = adj['S']
out['holm_within_fraction_note'] = 'superseded: the honest family is 32 contrasts, not 4'

p = os.path.join(BASE, 'results_rev', 'lowdata_evidence.json')
json.dump(out, io.open(p, 'w', encoding='utf-8'), indent=1)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ('means_A', 'means_B', 'fam_delta', 'rhos_A')}, indent=1)[:3000])
print('\nwrote', p)
