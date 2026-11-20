"""External validation on the independent subset (n=387), with the corrected and
verified label mapping.

Two protocols are reported:
  (a) transfer: fit the complete pipeline on the 519 primary records, predict the 387
      independent external records (no tuning on the external data);
  (b) internal CV of the external subset, for reference only.
The label mapping 1=LTO, 2=HTO, 3=LED, 4=HED, 5=PD, 6=Normal is verified on the 117
records that appear in both collections (see data_manifest.json).
"""
import os, sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             recall_score, confusion_matrix)
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import (load_primary_clean, load_external_clean, verify_external_mapping,
                   textbook_ratios, llm_votes, is_canonical, is_textbook, build_matrix,
                   consensus_ranking, CANONICAL_RATIOS, GASES, FAULT_NAMES)
from eval2 import evaluate, mcnemar_exact, R_DEFAULT

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))

Xp, yp = load_primary_clean()
Xe, ye, ext_meta = load_external_clean(Xp)
vm = verify_external_mapping(Xp, yp)
print('external independent:', Xe.shape, np.bincount(ye, minlength=6).tolist())

tb = textbook_ratios()
votes = llm_votes(PARSED)
thr = len(PARSED) / 2
endorsed = sorted([c for c, n in votes.items() if n >= thr], key=lambda c: (-votes[c], c))
novel = [c for c in endorsed if not is_canonical(c) and not is_textbook(c, tb)]
rank = consensus_ranking(PARSED)
canon_top17 = [r['feature'] for r in rank][:17]

SETS = {
    'P': list(GASES),
    'C': list(GASES) + list(CANONICAL_RATIOS),
    'S': list(GASES) + tb,
    'Lp': list(GASES) + novel,
    'Lk': canon_top17,
}


def fit_predict_transfer(Xtr, ytr, Xte, feats, n_select=17, seed=42):
    fmt = lambda M: build_matrix(M, feats)
    A, B = fmt(Xtr), fmt(Xte)
    d = A.shape[1]
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    if n_select is not None and n_select < d:
        sel = RandomForestClassifier(n_estimators=100, random_state=seed).fit(A, ytr)
        top = np.argsort(sel.feature_importances_)[::-1][:n_select]
        A, B = A[:, top], B[:, top]
    clf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                 random_state=seed).fit(A, ytr)
    return clf.predict(B), clf.predict_proba(B)


transfer = {}
for key, feats in SETS.items():
    d = len(feats)
    pred, proba = fit_predict_transfer(Xp, yp, Xe, feats, n_select=(17 if d > 17 else None))
    transfer[key] = {
        'n_features': d,
        'accuracy': float(accuracy_score(ye, pred) * 100),
        'macro_f1': float(f1_score(ye, pred, average='macro', zero_division=0) * 100),
        'balanced_accuracy': float(balanced_accuracy_score(ye, pred) * 100),
        'per_class_recall': (recall_score(ye, pred, average=None, zero_division=0,
                                          labels=list(range(6))) * 100).tolist(),
        'confusion': confusion_matrix(ye, pred, labels=list(range(6))).tolist(),
        'pred': pred.tolist(),
    }
    print(f'transfer {key:<3} d={d:2d} acc={transfer[key]["accuracy"]:6.2f} '
          f'macroF1={transfer[key]["macro_f1"]:6.2f} bAcc={transfer[key]["balanced_accuracy"]:6.2f}')

# paired tests vs raw gases on the transfer predictions
preds = {k: np.array(v['pred']) for k, v in transfer.items()}
transfer_tests = {}
for key in ('C', 'S', 'Lp', 'Lk'):
    mc = mcnemar_exact(preds[key], preds['P'], ye)
    transfer_tests[key] = {'delta': transfer[key]['accuracy'] - transfer['P']['accuracy'],
                           'mcnemar': mc}
    print(f'  {key} vs P: delta={transfer_tests[key]["delta"]:+.2f} '
          f'p={mc["p"]:.4f}')

# internal CV of the external subset (reference)
internal = {}
for key, feats in SETS.items():
    Xm = build_matrix(Xe, feats)
    d = Xm.shape[1]
    r = evaluate(Xm, ye, learner='RF', n_select=(17 if d > 17 else None),
                 repeats=R_DEFAULT, return_oof=False)
    internal[key] = {'accuracy': r['accuracy_mean'], 'std': r['accuracy_std'],
                     'macro_f1': r['macro_f1_mean'],
                     'balanced_accuracy': r['balanced_accuracy_mean'],
                     'per_class_recall': r['per_class_recall_mean']}
    print(f'internal-CV {key:<3} acc={r["accuracy_mean"]:6.2f} macroF1={r["macro_f1_mean"]:6.2f}')

out = {'external_meta': ext_meta, 'mapping_check': vm,
       'transfer': {k: {kk: vv for kk, vv in v.items() if kk != 'pred'}
                    for k, v in transfer.items()},
       'transfer_tests': transfer_tests, 'internal_cv': internal}
json.dump(out, open(os.path.join(OUT, 'external.json'), 'w', encoding='utf-8'), indent=1)
print('wrote', os.path.join(OUT, 'external.json'))
