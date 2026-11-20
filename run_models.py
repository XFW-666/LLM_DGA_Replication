"""Per-model re-evaluation on the cleaned data + agreement metrics.

Redesigned roles (reviewer 2, comment 1):
  * weights are removed as an accuracy mechanism (Random Forest is invariant to a
    positive rescaling of individual features); the LLM ranking now acts only through
    SELECTION of a 17-feature subset of the canonical 19;
  * per model we report
      G  - generation: 5 gases + the ratios that model proposed (validated, folded)
      K  - selection: canonical 19 restricted to that model's top-17 ranking
      GK - gases + the model's own proposals + its own selection of the canonical 14
  * agreement metrics (Spearman, Jaccard, Fleiss kappa, intra-model stability) are
    recomputed on the same parsed outputs.
"""
import os, sys, json, itertools
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import (load_primary_clean, build_matrix, consensus_ranking,
                   CANONICAL_RATIOS, GASES, FAULT_NAMES, reciprocal_key, is_canonical)
from eval2 import evaluate, mcnemar_exact, nadeau_bengio, holm, fold_diffs, R_DEFAULT

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))
USAGE = json.load(open(os.path.join(BASE, 'results', 'llm_usage.json'), encoding='utf-8'))

X, y = load_primary_clean()
MODELS = list(PARSED.keys())
print('models:', MODELS)


def model_own_set(model):
    """Union over replicates of the model's validated canonical forms, folded."""
    forms, seen = [], set()
    for rep, v in PARSED[model]['S1']['details'].items():
        for c in v['canonical_forms']:
            r = reciprocal_key(c)
            key = c if (r is None or c <= r) else r
            if key in seen:
                continue
            seen.add(key)
            forms.append(key)
    return sorted(forms)


def model_ranking(model):
    acc = {}
    for rep, v in PARSED[model]['S3']['details'].items():
        for it in v['ranking']:
            acc.setdefault(it['feature'], []).append(it['rank'])
    order = sorted(acc, key=lambda f: np.mean(acc[f]))
    return order


def model_removals(model):
    votes = Counter()
    n = 0
    for rep, v in PARSED[model]['S2']['details'].items():
        n += 1
        for f in v['remove']:
            votes[f] += 1
    return {f for f, c in votes.items() if c > n / 2}


results = {}
for m in MODELS:
    own = model_own_set(m)
    order = model_ranking(m)
    # the ranking covers the 19 canonical features (5 gases + 14 ratios)
    from data2 import CANONICAL_FEATURES as CF
    top17 = [f for f in order if f in CF][:17]
    if len(top17) < 17:
        rest = [f for f in CF if f not in top17]
        top17 += rest[:17 - len(top17)]
    rem = model_removals(m)
    own_kept = [c for c in own if c not in rem] or own

    conds = {
        'G': list(GASES) + own,
        'K': list(top17),
        'GK': list(dict.fromkeys(own_kept + top17)),
    }
    for tag, feats in conds.items():
        Xm = build_matrix(X, feats)
        d = Xm.shape[1]
        r = evaluate(Xm, y, learner='RF', n_select=(17 if d > 17 else None),
                     repeats=5, return_oof=True)
        led, hed = FAULT_NAMES.index('LED'), FAULT_NAMES.index('HED')
        results[f'{m}|{tag}'] = {
            'model': m, 'condition': tag, 'd': d,
            'accuracy_mean': r['accuracy_mean'], 'accuracy_std': r['accuracy_std'],
            'macro_f1_mean': r['macro_f1_mean'],
            'balanced_accuracy_mean': r['balanced_accuracy_mean'],
            'per_class_recall_mean': r['per_class_recall_mean'],
            'ci95': None,
            'n_own': len(own), 'n_own_kept': len(own_kept),
            'own': own, 'top17': top17, 'removed': sorted(rem),
        }
        print(f'  {m:<26} {tag:<3} d={d:2d} acc={r["accuracy_mean"]:6.2f} '
              f'macroF1={r["macro_f1_mean"]:6.2f}', flush=True)
        # keep OOF for paired tests against the shared baselines later
        np.save(os.path.join(OUT, f'oof_{m.replace(".", "_")}_{tag}.npy'),
                np.array(r['oof_pred']))

# ---- agreement metrics ----
def fleiss_binary(mat):
    n = mat.shape[1]
    k = mat.sum(axis=1)
    P_bar = np.mean([(x * (x - 1) + (n - x) * (n - x - 1)) / (n * (n - 1)) for x in k])
    p_i = mat.mean(axis=1)
    P_e = np.mean(p_i) ** 2 + np.mean(1 - p_i) ** 2
    return float((P_bar - P_e) / (1 - P_e)) if P_e != 1 else float('nan')


from scipy import stats
rank_vec = {}
for m in MODELS:
    rs = []
    for rep, v in PARSED[m]['S3']['details'].items():
        pos = {it['feature']: it['rank'] for it in v['ranking']}
        rs.append([pos.get(f, 19) for f in CANONICAL_RATIOS])
    rank_vec[m] = np.mean(rs, axis=0)

spear = {}
for a, b in itertools.combinations(MODELS, 2):
    rho, p = stats.spearmanr(rank_vec[a], rank_vec[b])
    spear[f'{a}~{b}'] = {'rho': float(rho), 'p': float(p)}

jac = {}
for a, b in itertools.combinations(MODELS, 2):
    vals = []
    for ra, va in PARSED[a]['S1']['details'].items():
        for rb, vb in PARSED[b]['S1']['details'].items():
            A = {rec.get('matches_canonical') for rec in va['accepted']} - {None}
            B = {rec.get('matches_canonical') for rec in vb['accepted']} - {None}
            if A | B:
                vals.append(len(A & B) / len(A | B))
    if vals:
        jac[f'{a}~{b}'] = float(np.mean(vals))

from exp_common import STANDARD_RATIOS
from exp_common import canonical_key
raters = [(m, rep) for m in MODELS for rep in PARSED[m]['S1']['details']]
std_defs = {n: (canonical_key(f'{a}/{b}'), canonical_key(f'{b}/{a}'))
            for n, (a, b) in STANDARD_RATIOS.items()}
mat = np.zeros((len(std_defs), len(raters)), dtype=int)
for j, (m, rep) in enumerate(raters):
    forms = set(PARSED[m]['S1']['details'][rep]['canonical_forms'])
    for i, (n_, (c1, c2)) in enumerate(std_defs.items()):
        mat[i, j] = 1 if any(f == c1 or f == c2 for f in forms) else 0
kappa = fleiss_binary(mat)

stability = {}
for m in MODELS:
    s3 = list(PARSED[m]['S3']['details'].values())
    if len(s3) >= 2:
        rhos = []
        for a, b in itertools.combinations(s3, 2):
            pa = {it['feature']: it['rank'] for it in a['ranking']}
            pb = {it['feature']: it['rank'] for it in b['ranking']}
            rhos.append(stats.spearmanr([pa.get(f, 19) for f in CANONICAL_RATIOS],
                                        [pb.get(f, 19) for f in CANONICAL_RATIOS]).statistic)
        stability[m] = float(np.mean(rhos))

agreement = {'spearman_pairwise': spear, 'jaccard_pairwise': jac,
             'fleiss_kappa': kappa, 'intra_model_stability': stability,
             'mean_spearman': float(np.mean([v['rho'] for v in spear.values()])),
             'mean_jaccard': float(np.mean(list(jac.values()))),
             'mean_stability': float(np.mean(list(stability.values()))),
             'standard_code_coverage': {m: int(sum(1 for i in range(len(std_defs))
                                                   for j, (mm, rep) in enumerate(raters)
                                                   if mm == m and mat[i, j]))
                                        for m in MODELS},
             'standard_code_total': len(std_defs)}
print('fleiss kappa', round(kappa, 3), 'mean spearman', round(agreement['mean_spearman'], 3),
      'mean stability', round(agreement['mean_stability'], 3))

json.dump({'results': results, 'agreement': agreement},
          open(os.path.join(OUT, 'models.json'), 'w', encoding='utf-8'), indent=1)
print('wrote', os.path.join(OUT, 'models.json'))
