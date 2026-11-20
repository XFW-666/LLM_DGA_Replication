"""Fold-level trend statistics for the sample-efficiency effect.

Recomputes the learning-curve experiment keeping every fold-level value, then tests the
gradient itself rather than a single point on it:
  (a) per-repeat Spearman(log training fraction, gap) with an exact sign test across repeats;
  (b) the same after pooling the four engineered families (the gap should not depend on which
      ratio set is added, so pooling is the mechanism-consistent summary);
  (c) cluster-robust regression of the fold-level gap on log10 fraction;
  (d) permutation test: the eight fraction labels are permuted as blocks across all folds,
      which preserves the correlation between fractions induced by the shared fold partition.
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
OUT = os.path.join(BASE, 'results_rev', 'lowdata_trend.json')
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
FRACTIONS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00])
ENGINEERED = ['S', 'C', 'Lp', 'CS']
REPEATS, K = 10, 5
NFOLD = REPEATS * K
NF = len(FRACTIONS)
LOGF = np.log10(FRACTIONS)
REPS_OF_FOLD = np.repeat(np.arange(REPEATS), K)

# cell ordering: fold-major (row 0 = fold 0 across the 8 fractions)
XS = np.tile(LOGF, NFOLD)
REP_OF_CELL = np.repeat(REPS_OF_FOLD, NF)


def log(*a):
    print(*a, flush=True)


# ---------------- fold-level curve: matrix[fold, fraction] -------------------------
curves = {k: np.zeros((NFOLD, NF)) for k in SETS}
n_train = np.zeros(NF)
for fi, f in enumerate(FRACTIONS):
    for r in range(REPEATS):
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42 + r)
        for j, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if f < 1.0:
                rng = np.random.default_rng(1000 + r)
                keep = []
                for c in np.unique(y[tr]):
                    idx = tr[y[tr] == c]
                    m = max(2, int(round(len(idx) * f)))
                    keep.extend(rng.choice(idx, size=m, replace=False).tolist())
                tr_f = np.array(sorted(keep))
            else:
                tr_f = tr
            if r == 0 and j == 0:
                n_train[fi] = len(tr_f)
            row = r * K + j
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
                curves[k][row, fi] = accuracy_score(y[te], clf.predict(B)) * 100
    log(f'  fraction {f:.2f} done (n_train = {n_train[fi]:.0f})')

gap_S = curves['S'] - curves['P']
gap_pool = np.mean([curves[k] - curves['P'] for k in ENGINEERED], axis=0)


def nb(d):
    d = np.asarray(d, float)
    J = len(d)
    t = d.mean() / math.sqrt((1.0 / J + 1.0 / 4.0) * np.var(d, ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=J - 1)


def slope_t(G):
    """Cluster-robust (by repeat) slope and t of the fold-level gap on log10 fraction."""
    ys = G.ravel()
    Xd = np.column_stack([np.ones(NFOLD * NF), XS])
    bta, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
    res = ys - Xd @ bta
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((2, 2))
    for g in range(REPEATS):
        sg = REP_OF_CELL == g
        m_g = Xd[sg].T @ res[sg]
        meat += np.outer(m_g, m_g)
    se = math.sqrt((XtX_inv @ meat @ XtX_inv)[1, 1])
    return bta[1], se, bta[1] / se


log('\nper-fraction gaps:')
for i, f in enumerate(FRACTIONS):
    a, b = nb(gap_S[:, i]), nb(gap_pool[:, i])
    log(f'  f={f:.2f}  n_train={n_train[i]:5.0f}   textbook: {a[0]:+.2f} (p={a[2]:.3f})   '
        f'pooled: {b[0]:+.2f} (p={b[2]:.3f})')

summary = {}
for nm, G in [('textbook_vs_raw', gap_S), ('pooled_engineered_vs_raw', gap_pool)]:
    rhos = np.array([stats.spearmanr(np.tile(LOGF, K), G[r * K:(r + 1) * K, :].ravel()).statistic
                     for r in range(REPEATS)])
    n_neg = int((rhos < 0).sum())
    p_sign = stats.binomtest(n_neg, REPEATS, 0.5, alternative='greater').pvalue
    log(f'\n(a) {nm}: Spearman(log fraction, gap) mean {rhos.mean():+.2f} '
        f'range [{rhos.min():+.2f}, {rhos.max():+.2f}]; '
        f'{n_neg}/{REPEATS} repeats negative, exact one-sided p = {p_sign:.5f}')
    slope, se, t = slope_t(G)
    p_t = stats.t.sf(t, REPEATS - 1)
    log(f'(c) {nm}: slope = {slope:+.3f} points per decade of training data, '
        f'cluster-robust SE = {se:.3f}, t = {t:.2f}, one-sided p = {p_t:.6f}')
    rng = np.random.default_rng(20261118)
    M = 20000
    cnt = 0
    for _ in range(M):
        _, _, tp = slope_t(G[:, rng.permutation(NF)])
        if tp <= t:
            cnt += 1
    p_perm = (cnt + 1) / (M + 1)
    log(f'(d) {nm}: block-permutation one-sided p = {p_perm:.6f} ({M} permutations)')
    summary[nm] = {'rho_mean': float(rhos.mean()), 'rho_range': [float(rhos.min()), float(rhos.max())],
                   'n_negative': n_neg, 'p_sign': float(p_sign),
                   'slope': float(slope), 'se': float(se), 't': float(t),
                   'p_cluster': float(p_t), 'p_perm': float(p_perm)}

json.dump({'fractions': FRACTIONS.tolist(), 'n_train': n_train.tolist(),
           'gap_S': gap_S.tolist(), 'gap_pool': gap_pool.tolist(),
           'acc': {k: curves[k].tolist() for k in SETS}, 'summary': summary},
          open(OUT, 'w', encoding='utf-8'), indent=1)
log(f'\nfold-level curves + summary saved -> {OUT}')
