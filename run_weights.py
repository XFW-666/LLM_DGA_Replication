"""Reviewer 2, comment 1: do LLM feature weights affect the classifier?

We test three learners on the canonical 19-feature pool:
  RF   - scale invariant (Gini splits depend only on the ordering of a feature)
  LR   - L2-regularised, so per-feature scaling changes the solution
  kNN  - Euclidean distance, so per-feature scaling changes the neighbourhoods
Weights: uniform, LLM-consensus rank-decay (alpha=0.2), and a null distribution of
random rank-decay weight vectors. If the LLM weights carry information, they should sit
outside that null for the scale-sensitive learners.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import load_primary_clean, consensus_ranking, build_matrix, CANONICAL_RATIOS, GASES
from eval2 import evaluate, R_DEFAULT

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))

X, y = load_primary_clean()
feats = list(GASES) + list(CANONICAL_RATIOS)
Xm = build_matrix(X, feats)
d = Xm.shape[1]


def rank_decay_weights(ranking, features, alpha=0.2):
    n = len(features)
    w = np.ones(n)
    pos = {f: i for i, f in enumerate(features)}
    ranks = {}
    for item in ranking:
        f, r = item.get('feature'), item.get('rank')
        if f in pos and isinstance(r, (int, float)):
            ranks[f] = min(max(int(r), 1), n)
    used = sorted(ranks.values())
    nxt = 1
    for f in features:
        if f not in ranks:
            while nxt in used:
                nxt += 1
            ranks[f] = nxt
            used.append(nxt)
            used.sort()
    for f, r in ranks.items():
        w[pos[f]] = 1.0 + alpha * (1.0 - (r - 1) / max(n - 1, 1))
    return w


cons = consensus_ranking(PARSED)
w_llm = rank_decay_weights(cons, feats)
print('LLM weights  min %.3f max %.3f' % (w_llm.min(), w_llm.max()))

rng = np.random.default_rng(11)
random_weights = []
for _ in range(20):
    perm = rng.permutation(len(feats))
    ranking = [{'feature': feats[j], 'rank': i + 1} for i, j in enumerate(perm)]
    random_weights.append(rank_decay_weights(ranking, feats))

out = {}
for learner in ['RF', 'LR', 'kNN']:
    print(f'\n--- {learner} ---')
    r_uni = evaluate(Xm, y, learner=learner, n_select=None, repeats=R_DEFAULT, return_oof=False)
    r_llm = evaluate(Xm, y, learner=learner, n_select=None, weights=w_llm,
                     repeats=R_DEFAULT, return_oof=False)
    null = []
    for w in random_weights:
        r = evaluate(Xm, y, learner=learner, n_select=None, weights=w,
                     repeats=2, return_oof=False)
        null.append(r['accuracy_mean'])
    null = np.array(null)
    pct = float((null < r_llm['accuracy_mean']).mean() * 100)
    print(f'  uniform {r_uni["accuracy_mean"]:6.2f}  LLM {r_llm["accuracy_mean"]:6.2f} '
          f'(delta {r_llm["accuracy_mean"] - r_uni["accuracy_mean"]:+.2f})')
    print(f'  random weights: mean {null.mean():.2f} sd {null.std():.2f} '
          f'[{np.percentile(null, 2.5):.2f}, {np.percentile(null, 97.5):.2f}]; '
          f'LLM percentile {pct:.0f}')
    out[learner] = {
        'uniform': r_uni['accuracy_mean'], 'uniform_std': r_uni['accuracy_std'],
        'llm': r_llm['accuracy_mean'], 'llm_std': r_llm['accuracy_std'],
        'delta': r_llm['accuracy_mean'] - r_uni['accuracy_mean'],
        'null_mean': float(null.mean()), 'null_sd': float(null.std()),
        'null_p2.5': float(np.percentile(null, 2.5)),
        'null_p97.5': float(np.percentile(null, 97.5)),
        'llm_percentile': pct,
        'macro_f1_uniform': r_uni['macro_f1_mean'], 'macro_f1_llm': r_llm['macro_f1_mean'],
    }

json.dump({'features': feats, 'llm_weights': w_llm.tolist(), 'results': out},
          open(os.path.join(OUT, 'weights.json'), 'w', encoding='utf-8'), indent=1)
print('\nwrote', os.path.join(OUT, 'weights.json'))
