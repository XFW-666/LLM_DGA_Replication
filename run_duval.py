"""Reviewer 2, comment 7: class definitions and Duval Triangle 1 ambiguity.

Triangle 1 uses the relative proportions of CH4, C2H4 and C2H2. We compute the zone of
every sample with the IEC 60599 boundaries and report
  (a) the label x zone contingency table,
  (b) the samples whose label disagrees with the triangle family (thermal vs discharge),
  (c) the accuracy of the three main feature sets on the ambiguous and unambiguous
      subsets, so that the reader can see whether the conclusions survive.
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import (load_primary_clean, textbook_ratios, llm_votes, is_canonical,
                   is_textbook, build_matrix, CANONICAL_RATIOS, GASES, FAULT_NAMES)
from eval2 import evaluate, R_DEFAULT

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))
X, y = load_primary_clean()


def duval_t1(ch4, c2h4, c2h2):
    """IEC 60599 Duval Triangle 1 zones in percentage coordinates."""
    s = ch4 + c2h4 + c2h2
    if s <= 0:
        return 'ND'
    a, b, c = 100 * ch4 / s, 100 * c2h4 / s, 100 * c2h2 / s
    # PD: CH4 >= 98%
    if a >= 98:
        return 'PD'
    # D2: C2H2 >= 29%
    if c >= 29:
        return 'D2'
    # T3: C2H4 >= 36% (below the D2 boundary)
    if b >= 36:
        return 'T3'
    # DT: C2H2 13-29% and C2H4 23-36%
    if 13 <= c < 29 and 23 <= b < 36:
        return 'DT'
    # D1: C2H2 13-29%, C2H4 < 23%
    if 13 <= c < 29 and b < 23:
        return 'D1'
    # remaining: C2H2 < 13%; split T1/T2 at C2H4 = 20%
    if b >= 20:
        return 'T2'
    return 'T1'


zones = np.array([duval_t1(X[i, 1], X[i, 3], X[i, 4]) for i in range(len(X))])
tab = pd.crosstab(pd.Series([FAULT_NAMES[v] for v in y], name='label'),
                  pd.Series(zones, name='zone'))
print(tab.to_string())

THERMAL = {'LTO', 'HTO'}
DISCHARGE = {'LED', 'HED', 'PD'}
TZ = {'T1', 'T2', 'T3'}
DZ = {'D1', 'D2', 'DT'}
labels = np.array([FAULT_NAMES[v] for v in y])
amb_thermal = np.isin(labels, list(THERMAL)) & np.isin(zones, list(DZ))
amb_discharge = np.isin(labels, list(DISCHARGE)) & np.isin(zones, list(TZ))
amb_pd = (labels == 'PD') & (zones != 'PD')
ambiguous = amb_thermal | amb_discharge
print(f'\nthermal-labelled samples in a discharge zone: {int(amb_thermal.sum())}')
print(f'discharge-labelled samples in a thermal zone: {int(amb_discharge.sum())}')
print(f'PD-labelled samples outside the PD zone:      {int(amb_pd.sum())}')
print(f'union (thermal/discharge contradictions):     {int(ambiguous.sum())}')

# accuracy on ambiguous vs unambiguous subsets for the three main feature sets
tb = textbook_ratios()
votes = llm_votes(PARSED)
thr = len(PARSED) / 2
endorsed = sorted([c for c, n in votes.items() if n >= thr], key=lambda c: (-votes[c], c))
novel = [c for c in endorsed if not is_canonical(c) and not is_textbook(c, tb)]
SETS = {
    'P': list(GASES),
    'C': list(GASES) + list(CANONICAL_RATIOS),
    'S': list(GASES) + tb,
    'Lp': list(GASES) + novel,
}
subset_res = {}
for name, mask in [('unambiguous', ~ambiguous), ('ambiguous', ambiguous)]:
    print(f'\n--- {name} subset (n={int(mask.sum())}) ---')
    subset_res[name] = {}
    for key, feats in SETS.items():
        Xm = build_matrix(X[mask], feats)
        d = Xm.shape[1]
        r = evaluate(Xm, y[mask], learner='RF', n_select=(17 if d > 17 else None),
                     repeats=R_DEFAULT, return_oof=False)
        subset_res[name][key] = {'n': int(mask.sum()), 'accuracy': r['accuracy_mean'],
                                 'std': r['accuracy_std'],
                                 'macro_f1': r['macro_f1_mean']}
        print(f'  {key:<3} n={int(mask.sum()):3d} acc={r["accuracy_mean"]:6.2f} '
              f'macroF1={r["macro_f1_mean"]:6.2f}')

out = {
    'zone_table': {f'{lb}|{z}': int(tab.loc[lb, z]) for lb in tab.index for z in tab.columns},
    'zones': tab.columns.tolist(),
    'counts': {
        'thermal_in_discharge_zone': int(amb_thermal.sum()),
        'discharge_in_thermal_zone': int(amb_discharge.sum()),
        'pd_outside_pd_zone': int(amb_pd.sum()),
        'ambiguous_total': int(ambiguous.sum()),
        'n': int(len(y)),
    },
    'ambiguous_indices': np.where(ambiguous)[0].tolist(),
    'subset_results': subset_res,
}
json.dump(out, open(os.path.join(OUT, 'duval.json'), 'w', encoding='utf-8'), indent=1)
print('\nwrote', os.path.join(OUT, 'duval.json'))
