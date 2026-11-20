"""Does the engineered ratio family help in the low-data regime?

Hypothesis (pre-declarable): a domain prior should matter most when training data are scarce.
Protocol: for each fold, subsample the TRAINING portion (stratified) to a fraction f, keep the
test fold intact, and evaluate P / S / C / Lp / CS. Corrected test on the fold-level differences
at each fraction.
"""
import os, sys, math, json
import numpy as np
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\rev')
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data2 import (load_primary_clean, build_matrix, textbook_ratios, llm_votes,
                   is_canonical, is_textbook, CANONICAL_RATIOS, GASES)

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))
X, y = load_primary_clean()
tb = textbook_ratios()
votes = llm_votes(PARSED)
thr = len(PARSED) / 2
endorsed = sorted([c for c, n in votes.items() if n >= thr], key=lambda c: (-votes[c], c))
novel = [c for c in endorsed if not is_canonical(c) and not is_textbook(c, tb)]

SETS = {'P': list(GASES), 'S': list(GASES) + tb, 'C': list(GASES) + list(CANONICAL_RATIOS),
        'Lp': list(GASES) + novel, 'CS': list(GASES) + list(CANONICAL_RATIOS) + tb}
MATS = {k: build_matrix(X, f) for k, f in SETS.items()}

FRACTIONS = [0.10, 0.20, 0.40, 0.60, 1.00]
REPEATS = 10
K = 5


def run(fraction):
    accs = {k: [] for k in SETS}      # per (repeat, fold)
    for r in range(REPEATS):
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42 + r)
        for tr, te in skf.split(np.zeros(len(y)), y):
            if fraction < 1.0:
                rng = np.random.default_rng(1000 + r)
                keep = []
                for c in np.unique(y[tr]):
                    idx = tr[y[tr] == c]
                    m = max(2, int(round(len(idx) * fraction)))
                    keep.extend(rng.choice(idx, size=m, replace=False).tolist())
                tr_f = np.array(sorted(keep))
            else:
                tr_f = tr
            for k, M in MATS.items():
                sc = StandardScaler()
                A, B = sc.fit_transform(M[tr_f]), sc.transform(M[te])
                d = A.shape[1]
                if d > 17:
                    sel = RandomForestClassifier(n_estimators=100, random_state=42).fit(A, y[tr_f])
                    top = np.argsort(sel.feature_importances_)[::-1][:17]
                    A, B = A[:, top], B[:, top]
                clf = RandomForestClassifier(n_estimators=150, max_depth=20,
                                             min_samples_leaf=2, random_state=42)
                clf.fit(A, y[tr_f])
                accs[k].append(accuracy_score(y[te], clf.predict(B)) * 100)
    return accs


def nb(da, db):
    d = np.asarray(da, float) - np.asarray(db, float)
    J = len(d)
    var = np.var(d, ddof=1)
    denom = math.sqrt((1.0 / J + 1.0 / 4.0) * var)
    t = d.mean() / denom if denom else float('nan')
    p = 2 * stats.t.sf(abs(t), df=J - 1) if denom else 1.0
    return d.mean(), t, p


out = {}
for f in FRACTIONS:
    accs = run(f)
    means = {k: float(np.mean(v)) for k, v in accs.items()}
    print(f'\nf = {f:.2f}  ' + '  '.join(f'{k}={means[k]:5.2f}' for k in SETS), flush=True)
    row = {'means': means}
    for k in ['S', 'C', 'Lp', 'CS']:
        d, t, p = nb(accs[k], accs['P'])
        row[k] = {'delta': d, 't': t, 'p': p}
        print(f'   {k} vs P: delta={d:+5.2f} t={t:5.2f} p={p:.3f}')
    out[f'{f:.2f}'] = row

json.dump(out, open(os.path.join(BASE, 'results_rev', 'lowdata.json'), 'w',
                    encoding='utf-8'), indent=1)
print('\nwrote results_rev/lowdata.json')
