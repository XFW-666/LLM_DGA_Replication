"""Fairly tuned baseline learners (reviewer 2, comment 9; reviewer 3, comment 7).

Every learner is tuned with an inner 3-fold grid search *inside the training fold only*,
so the outer comparison remains leakage-free. Gradient boosting uses the histogram-based
implementation for speed on this small sample.
"""
import os, sys, json
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, recall_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import (load_primary_clean, textbook_ratios, llm_votes, is_canonical, is_textbook,
                   consensus_ranking, build_matrix, CANONICAL_RATIOS, GASES, FAULT_NAMES)

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))
X, y = load_primary_clean()

tb = textbook_ratios()
votes = llm_votes(PARSED)
thr = len(PARSED) / 2
endorsed = sorted([c for c, n in votes.items() if n >= thr], key=lambda c: (-votes[c], c))
novel = [c for c in endorsed if not is_canonical(c) and not is_textbook(c, tb)]
rank = consensus_ranking(PARSED)
top17 = [r['feature'] for r in rank][:17]

SETS = {
    'P': list(GASES),
    'C': list(GASES) + list(CANONICAL_RATIOS),
    'S': list(GASES) + tb,
    'Lp': list(GASES) + novel,
    'Lk': top17,
}

GRIDS = {
    'RF': (RandomForestClassifier(random_state=42),
           {'n_estimators': [150], 'max_depth': [20], 'min_samples_leaf': [2]},
           False),
    'RF-tuned': (RandomForestClassifier(random_state=42),
                 {'n_estimators': [300], 'max_depth': [None, 20],
                  'min_samples_leaf': [1, 2], 'max_features': ['sqrt']}, False),
    'GBDT': (HistGradientBoostingClassifier(random_state=42),
             {'max_iter': [200], 'learning_rate': [0.1], 'max_depth': [None, 6],
              'l2_regularization': [1.0]}, False),
    'SVM': (SVC(probability=True, random_state=42),
            {'C': [1, 10, 100], 'gamma': ['scale', 0.01]}, True),
    'kNN': (KNeighborsClassifier(), {'n_neighbors': [5, 15, 30], 'weights': ['distance']}, True),
    'LR': (LogisticRegression(max_iter=3000),
           {'C': [0.1, 1, 10]}, True),
}

K = 5
REPEATS = 3


def run(feats, learner, balanced=False, n_select=17, repeats=REPEATS):
    Xm = build_matrix(X, feats)
    d = Xm.shape[1]
    accs, mf1, bacc, recs = [], [], [], []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42 + r)
        for tr, te in skf.split(Xm, y):
            sc = StandardScaler()
            A, B = sc.fit_transform(Xm[tr]), sc.transform(Xm[te])
            if n_select is not None and n_select < d:
                sel = RandomForestClassifier(n_estimators=100, random_state=42).fit(A, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:n_select]
                A, B = A[:, top], B[:, top]
            est, grid, needs_scale = GRIDS[learner]
            if balanced:
                est = _balanced_copy(est)
            gs = GridSearchCV(est, grid, cv=3, scoring='accuracy', n_jobs=1)
            gs.fit(A, y[tr])
            pred = gs.predict(B)
            accs.append(accuracy_score(y[te], pred) * 100)
            mf1.append(f1_score(y[te], pred, average='macro', zero_division=0) * 100)
            bacc.append(balanced_accuracy_score(y[te], pred) * 100)
            recs.append(recall_score(y[te], pred, average=None, zero_division=0,
                                     labels=list(range(6))) * 100)
    return {'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(mf1)), 'macro_f1_std': float(np.std(mf1)),
            'balanced_accuracy_mean': float(np.mean(bacc)),
            'per_class_recall_mean': np.mean(recs, axis=0).tolist(),
            'repeats': repeats}


def _balanced_copy(est):
    from sklearn.base import clone
    e = clone(est)
    if hasattr(e, 'class_weight'):
        e.set_params(class_weight='balanced')
    return e


results = {}
for learner in ['RF', 'GBDT', 'SVM', 'kNN', 'LR']:
    for key in ['P', 'C', 'S', 'Lp', 'Lk']:
        d = len(SETS[key])
        r = run(SETS[key], learner, n_select=(17 if d > 17 else None))
        results[f'{learner}|{key}'] = r
        print(f'{learner:<9} {key:<3} acc={r["accuracy_mean"]:6.2f}±{r["accuracy_std"]:.2f} '
              f'macroF1={r["macro_f1_mean"]:6.2f} bAcc={r["balanced_accuracy_mean"]:6.2f}',
              flush=True)

# balanced (class-weight) variants of the two strongest learners
balanced = {}
for learner in ['RF', 'GBDT', 'SVM']:
    for key in ['P', 'C', 'Lp']:
        d = len(SETS[key])
        r = run(SETS[key], learner, balanced=True, n_select=(17 if d > 17 else None))
        balanced[f'{learner}|{key}'] = r
        print(f'balanced {learner:<6} {key:<3} acc={r["accuracy_mean"]:6.2f} '
              f'macroF1={r["macro_f1_mean"]:6.2f} bAcc={r["balanced_accuracy_mean"]:6.2f}',
              flush=True)

json.dump({'results': results, 'balanced': balanced,
           'grids': {k: v[1] for k, v in GRIDS.items()},
           'note': 'inner 3-fold grid search inside each training fold; '
                   '5 repeats x 5 folds; identical outer folds across learners'},
          open(os.path.join(OUT, 'baselines.json'), 'w', encoding='utf-8'), indent=1)
print('wrote', os.path.join(OUT, 'baselines.json'))
