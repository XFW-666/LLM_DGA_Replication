"""Is the sample-efficiency effect real? (fixed monotonicity indexing, persistent)

(1) single pre-declared trend contrast: delta(textbook vs raw) at 10% minus at 100%;
(2) monotonicity: per-repeat Spearman(fraction, delta), pooled;
(3) seed robustness: whole curve re-run with an independent subsample seed stream;
(4) Holm over the full 8x4 = 32-contrast family (strictest honest correction).
"""
import os, sys, json, math
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
OUT = os.path.join(BASE, 'results_rev', 'lowdata_robustness.json')
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
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00]
REPEATS, K = 10, 5


def log(*a):
    print(*a, flush=True)


def curve(seed_offset):
    out = {}
    for f in FRACTIONS:
        accs = {k: [] for k in SETS}
        for r in range(REPEATS):
            skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42 + r)
            for tr, te in skf.split(np.zeros(len(y)), y):
                if f < 1.0:
                    rng = np.random.default_rng(seed_offset + r)
                    keep = []
                    for c in np.unique(y[tr]):
                        idx = tr[y[tr] == c]
                        m = max(2, int(round(len(idx) * f)))
                        keep.extend(rng.choice(idx, size=m, replace=False).tolist())
                    tr_f = np.array(sorted(keep))
                else:
                    tr_f = tr
                for k, M in MATS.items():
                    sc = StandardScaler()
                    A, B = sc.fit_transform(M[tr_f]), sc.transform(M[te])
                    if A.shape[1] > 17:
                        sel = RandomForestClassifier(n_estimators=100,
                                                     random_state=42).fit(A, y[tr_f])
                        top = np.argsort(sel.feature_importances_)[::-1][:17]
                        A, B = A[:, top], B[:, top]
                    clf = RandomForestClassifier(n_estimators=150, max_depth=20,
                                                 min_samples_leaf=2, random_state=42)
                    clf.fit(A, y[tr_f])
                    accs[k].append(accuracy_score(y[te], clf.predict(B)) * 100)
        out[f] = accs
        log(f'    fraction {f:.2f} done')
    return out


def nb(d):
    d = np.asarray(d, float)
    J = len(d)
    t = d.mean() / math.sqrt((1.0 / J + 1.0 / 4.0) * np.var(d, ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=J - 1)


def holm(ps):
    ps = sorted(ps, key=lambda x: x[1])
    m, run, adj = len(ps), 0.0, {}
    for i, (k, pv) in enumerate(ps):
        run = max(run, (m - i) * pv)
        adj[k] = min(run, 1.0)
    return adj


report = {}
for name, off in [('A', 1000), ('B', 9000)]:
    log(f'\n===== seed stream {name} (offset {off}) =====')
    cur = curve(off)
    deltas = {f: np.asarray(cur[f]['S']) - np.asarray(cur[f]['P']) for f in FRACTIONS}
    for f in FRACTIONS:
        dm, t, p = nb(deltas[f])
        log(f'  f={f:.2f}  S-P = {dm:+.2f}  p={p:.4f}')
    # (1) trend contrast
    dm, t, p = nb(deltas[0.10] - deltas[1.00])
    log(f'  (1) delta(10%) - delta(100%) = {dm:+.2f}  t={t:.2f}  p={p:.4f}')
    # (2) monotonicity per repeat
    rhos = []
    for r in range(REPEATS):
        xs = np.array([f for f in FRACTIONS for _ in range(K)])
        ys = np.concatenate([deltas[f][r * K:(r + 1) * K] for f in FRACTIONS])
        rhos.append(stats.spearmanr(xs, ys).statistic)
    log(f'  (2) mean Spearman(fraction, delta) = {np.mean(rhos):+.2f} '
        f'(range {min(rhos):+.2f}..{max(rhos):+.2f})')
    # (4) Holm over the full 8x4 family
    ps = []
    for f in FRACTIONS:
        for k in ['S', 'C', 'Lp', 'CS']:
            ps.append((f'{k}@{f:.2f}', nb(np.asarray(cur[f][k]) - np.asarray(cur[f]['P']))[2]))
    adj = holm(ps)
    log('  (4) Holm over all 32 contrasts, five smallest:')
    for k, v in list(adj.items())[:5]:
        log(f'      {k:12s} p_holm={v:.3f}')
    # raw p-values for every contrast, for the record
    raw = {k: {'S': nb(np.asarray(cur[f]['S']) - np.asarray(cur[f]['P']))[2],
               'C': nb(np.asarray(cur[f]['C']) - np.asarray(cur[f]['P']))[2],
               'Lp': nb(np.asarray(cur[f]['Lp']) - np.asarray(cur[f]['P']))[2],
               'CS': nb(np.asarray(cur[f]['CS']) - np.asarray(cur[f]['P']))[2]}
           for f in FRACTIONS}
    report[name] = {
        'offset': off,
        'curve_S_minus_P': {f'{f:.2f}': [float(np.mean(deltas[f])), float(nb(deltas[f])[2])]
                             for f in FRACTIONS},
        'trend_contrast': {'delta': dm, 't': t, 'p': p},
        'spearman_mean': float(np.mean(rhos)),
        'spearman_range': [float(min(rhos)), float(max(rhos))],
        'holm32_smallest': [[k, v] for k, v in list(adj.items())[:5]],
        'raw_p': raw,
        'means': {f'{f:.2f}': {k: float(np.mean(cur[f][k])) for k in SETS} for f in FRACTIONS},
    }
json.dump(report, open(OUT, 'w', encoding='utf-8'), indent=1)
log(f'\nsaved -> {OUT}')
