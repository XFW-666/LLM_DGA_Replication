"""Recompute the random-selection null and store the individual draws."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import load_primary_clean, build_matrix, CANONICAL_RATIOS, GASES
from eval2 import evaluate

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
RES = os.path.join(BASE, 'results_rev')
X, y = load_primary_clean()
Xm = build_matrix(X, list(GASES) + list(CANONICAL_RATIOS))
rng = np.random.default_rng(7)
draws = []
for t in range(100):
    sel = rng.choice(Xm.shape[1], 17, replace=False)
    r = evaluate(Xm[:, sel], y, learner='RF', n_select=None, repeats=1, fast=True)
    draws.append(r['accuracy_mean'])
    if (t + 1) % 20 == 0:
        print('draw', t + 1, 'of 100', flush=True)

main = json.load(open(os.path.join(RES, 'main.json'), encoding='utf-8'))
null = main['random_selection_null']
null['draws'] = [float(v) for v in draws]
null['mean'] = float(np.mean(draws))
null['sd'] = float(np.std(draws))
null['p2.5'] = float(np.percentile(draws, 2.5))
null['p50'] = float(np.percentile(draws, 50))
null['p97.5'] = float(np.percentile(draws, 97.5))
null['llm_percentile'] = float((np.array(draws) < null['llm_top17']).mean() * 100)
json.dump(main, open(os.path.join(RES, 'main.json'), 'w', encoding='utf-8'), indent=1)
print('null updated:', null['mean'], null['p2.5'], null['p97.5'],
      null['llm_percentile'])
