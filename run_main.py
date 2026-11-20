"""Primary evaluation on the cleaned dataset (n=519): feature-set conditions,
secondary learners, random-selection null, Nadeau-Bengio + Holm inference."""
import os, sys, json, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import (load_primary_clean, textbook_ratios, llm_votes, is_canonical,
                   is_textbook, consensus_ranking, build_matrix,
                   CANONICAL_RATIOS, GASES, FAULT_NAMES)
from eval2 import (evaluate, nadeau_bengio, mcnemar_exact, holm, fold_diffs,
                   bootstrap_ci_oof, per_class_recall_ci, R_DEFAULT)

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))

X, y, meta = load_primary_clean(return_meta=True)
tb = textbook_ratios()
votes = llm_votes(PARSED)
thr = len(PARSED) / 2
endorsed = sorted([c for c, n in votes.items() if n >= thr], key=lambda c: (-votes[c], c))
novel = [c for c in endorsed if not is_canonical(c) and not is_textbook(c, tb)]
rank = consensus_ranking(PARSED)
canon_top17 = [r['feature'] for r in rank][:17]

FEATURE_SETS = {
    'P':  ('raw five gases', list(GASES)),
    'C':  ('hand-crafted canonical 14 (19 features)', list(GASES) + list(CANONICAL_RATIOS)),
    'S':  ('textbook IEC/Rogers/Doernenburg ratios, no LLM', list(GASES) + tb),
    'CS': ('canonical + textbook', list(GASES) + list(CANONICAL_RATIOS) + tb),
    'L':  ('LLM-endorsed ratios (all)', list(GASES) + endorsed),
    'Lp': ('LLM-only ratios (non-textbook, non-canonical)', list(GASES) + novel),
    'Lk': ('LLM-consensus top-17 of canonical 19', canon_top17),
    'CL': ('canonical + LLM-only ratios', list(GASES) + list(CANONICAL_RATIOS) + novel),
}

print('endorsed ratios:', len(endorsed), '| llm-only:', len(novel))
print('LLM consensus top-17:', canon_top17)

results = {}
matrices = {}
for key, (label, feats) in FEATURE_SETS.items():
    Xm = build_matrix(X, feats)
    matrices[key] = Xm
    d = Xm.shape[1]
    nsel = 17 if d > 17 else None
    res = evaluate(Xm, y, learner='RF', n_select=nsel, repeats=R_DEFAULT)
    res['label'] = label
    res['feature_names'] = feats
    lo, hi = bootstrap_ci_oof(res['oof_pred'], y)
    res['ci95'] = [lo, hi]
    led = FAULT_NAMES.index('LED')
    hed = FAULT_NAMES.index('HED')
    res['led_recall_ci'] = per_class_recall_ci(res['oof_pred'], y, led)
    res['hed_recall_ci'] = per_class_recall_ci(res['oof_pred'], y, hed)
    results[key] = res
    print(f'{key:<3} {label:<48} d={d:2d} acc={res["accuracy_mean"]:6.2f}'
          f'±{res["accuracy_std"]:.2f}  macroF1={res["macro_f1_mean"]:6.2f}'
          f'  bAcc={res["balanced_accuracy_mean"]:6.2f}  CI={res["ci95"][0]:.2f}-{res["ci95"][1]:.2f}')

# ---- secondary learners (reviewer request: report GBDT/SVM/kNN, plus LR) ----
secondary = {}
for learner in ['GBDT', 'SVM', 'kNN', 'LR']:
    for key in ['P', 'C', 'S', 'Lp']:
        Xm = matrices[key]
        d = Xm.shape[1]
        res = evaluate(Xm, y, learner=learner, n_select=(17 if d > 17 else None),
                       repeats=5, return_oof=False)
        secondary[f'{learner}|{key}'] = {
            'accuracy_mean': res['accuracy_mean'], 'accuracy_std': res['accuracy_std'],
            'macro_f1_mean': res['macro_f1_mean'], 'balanced_accuracy_mean': res['balanced_accuracy_mean'],
        }
        print(f'   {learner:<4} {key:<3} acc={res["accuracy_mean"]:6.2f} '
              f'macroF1={res["macro_f1_mean"]:6.2f}', flush=True)

# ---- class-imbalance-aware variants ----
balanced = {}
for key in ['P', 'C', 'Lp']:
    Xm = matrices[key]
    d = Xm.shape[1]
    for learner in ['RF', 'GBDT', 'SVM']:
        res = evaluate(Xm, y, learner=learner, balanced=True,
                       n_select=(17 if d > 17 else None), repeats=5, return_oof=False)
        balanced[f'{learner}|{key}'] = {
            'accuracy_mean': res['accuracy_mean'], 'accuracy_std': res['accuracy_std'],
            'macro_f1_mean': res['macro_f1_mean'],
            'balanced_accuracy_mean': res['balanced_accuracy_mean'],
            'per_class_recall_mean': res['per_class_recall_mean'],
        }
        print(f'   balanced {learner:<4} {key:<3} acc={res["accuracy_mean"]:6.2f} '
              f'macroF1={res["macro_f1_mean"]:6.2f} bAcc={res["balanced_accuracy_mean"]:6.2f}',
              flush=True)

# ---- random-selection null on the canonical 19 pool (100 draws, 1 repeat, fast) ----
Xc = matrices['C']
rng = np.random.default_rng(7)
null_acc = []
for t in range(100):
    sel = rng.choice(Xc.shape[1], 17, replace=False)
    r = evaluate(Xc[:, sel], y, learner='RF', n_select=None, repeats=1, fast=True)
    null_acc.append(r['accuracy_mean'])
null_acc = np.array(null_acc)
null = {'mean': float(null_acc.mean()), 'sd': float(null_acc.std()),
        'p2.5': float(np.percentile(null_acc, 2.5)),
        'p50': float(np.percentile(null_acc, 50)),
        'p97.5': float(np.percentile(null_acc, 97.5)),
        'llm_top17': results['Lk']['accuracy_mean'],
        'llm_percentile': float((null_acc < results['Lk']['accuracy_mean']).mean() * 100)}
print(f'\nrandom 17-of-19 null: {null["mean"]:.2f} [{null["p2.5"]:.2f}, {null["p97.5"]:.2f}], '
      f'LLM top-17 {null["llm_top17"]:.2f} -> percentile {null["llm_percentile"]:.0f}')

# ---- inference: Nadeau-Bengio corrected t-test, Holm within each family ----
contrasts = {}
for family, ref in [('vs_raw', 'P'), ('vs_canonical', 'C')]:
    keys = [k for k in FEATURE_SETS if k != ref]
    tests = []
    for k in keys:
        nb = nadeau_bengio(fold_diffs(results[k], results[ref]))
        mc = mcnemar_exact(np.array(results[k]['oof_pred'][0]),
                           np.array(results[ref]['oof_pred'][0]), y)
        tests.append({'condition': k, 'delta': results[k]['accuracy_mean'] -
                      results[ref]['accuracy_mean'], 'nb': nb, 'mcnemar': mc})
    padj = holm([t['nb']['p'] for t in tests])
    for t, p in zip(tests, padj):
        t['nb']['p_holm'] = float(p)
    contrasts[family] = tests
    print(f'\n{family} (Nadeau-Bengio, Holm-adjusted):')
    for t in tests:
        print(f"   {t['condition']:<3} d={t['delta']:+.2f} p={t['nb']['p']:.4f} "
              f"p_holm={t['nb']['p_holm']:.4f}")

out = {
    'data': meta,
    'feature_sets': {k: {'label': v[0], 'd': len(v[1])} for k, v in FEATURE_SETS.items()},
    'llm_sets': {'endorsed': endorsed, 'llm_only': novel, 'canon_top17': canon_top17,
                 'votes': dict(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:60])},
    'results': {k: {kk: vv for kk, vv in v.items()
                    if kk not in ('fold_rows', 'oof_proba')} for k, v in results.items()},
    'fold_rows': {k: v['fold_rows'] for k, v in results.items()},
    'secondary_learners': secondary,
    'balanced': balanced,
    'random_selection_null': null,
    'contrasts': contrasts,
}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'main.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('\nwrote', os.path.join(OUT, 'main.json'))
