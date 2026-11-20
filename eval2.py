"""Revision evaluation harness.

Design decisions (all reported in the manuscript):
  * repeated stratified K-fold (R repeats x 5 folds), identical fold assignment for
    every condition, so all comparisons are paired;
  * feature selection inside the training fold only (leakage-free);
  * metrics: accuracy, macro-F1, balanced accuracy, per-class recall, macro AUC;
  * inference: Nadeau-Bengio corrected resampled t-test on the R*K fold-level
    differences (accounts for overlapping training sets), Holm correction across the
    pre-declared comparison family; McNemar on pooled out-of-fold predictions is also
    reported for continuity, with its independence caveat stated.
"""
import os, sys, json, math, itertools
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             recall_score, precision_score, roc_auc_score)

N_CLASSES = 6
K = 5
R_DEFAULT = 10
SEED = 42


def make_learner(name, balanced=False):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                      random_state=SEED,
                                      class_weight='balanced' if balanced else None)
    if name == 'GBDT':
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.1, random_state=SEED)
    if name == 'SVM':
        return SVC(C=1.0, gamma='scale', probability=True, random_state=SEED,
                   class_weight='balanced' if balanced else None)
    if name == 'kNN':
        return KNeighborsClassifier(n_neighbors=15, weights='distance')
    if name == 'LR':
        return LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight='balanced' if balanced else None)
    raise ValueError(name)


def fold_indices(n, y, repeats):
    """Yield (repeat, fold, train_idx, test_idx) with identical assignments per repeat."""
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
            yield r, f, tr, te


def evaluate(X, y, learner='RF', balanced=False, n_select=17, repeats=R_DEFAULT,
             weights=None, fixed_idx=None, return_oof=True, fast=False):
    """Run repeated CV for one condition; returns per-fold and per-repeat metrics.

    fast=True skips AUC, per-class metrics, probabilities and OOF storage (used for the
    random-selection null, where only accuracy matters).
    """
    n, d = X.shape
    if fixed_idx is not None:
        X = X[:, fixed_idx]
        d = X.shape[1]
    w = np.ones(d) if weights is None else np.asarray(weights, float)
    fold_rows = []
    oof = None if fast else np.zeros((repeats, n), dtype=int)
    oof_p = None if fast else np.zeros((repeats, n, N_CLASSES))
    for r, f, tr, te in fold_indices(n, y, repeats):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]) * w
        Xte = sc.transform(X[te]) * w
        if n_select is not None and n_select < d:
            sel = RandomForestClassifier(n_estimators=100, random_state=SEED)
            sel.fit(Xtr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:n_select]
            Xtr, Xte = Xtr[:, top], Xte[:, top]
        clf = make_learner(learner, balanced)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        if fast:
            fold_rows.append({'repeat': r, 'fold': f,
                              'accuracy': accuracy_score(y[te], pred) * 100})
            continue
        proba = clf.predict_proba(Xte)
        if proba.shape[1] != N_CLASSES:
            full = np.zeros((proba.shape[0], N_CLASSES))
            for j, c in enumerate(clf.classes_):
                full[:, int(c)] = proba[:, j]
            proba = full
        oof[r, te] = pred
        oof_p[r, te] = proba
        fold_rows.append({
            'repeat': r, 'fold': f,
            'accuracy': accuracy_score(y[te], pred) * 100,
            'macro_f1': f1_score(y[te], pred, average='macro', zero_division=0) * 100,
            'balanced_accuracy': balanced_accuracy_score(y[te], pred) * 100,
            'per_class_recall': (recall_score(y[te], pred, average=None, zero_division=0,
                                              labels=list(range(N_CLASSES))) * 100).tolist(),
        })
    if fast:
        per_repeat = [float(np.mean([row['accuracy'] for row in fold_rows
                                     if row['repeat'] == r])) for r in range(repeats)]
        return {'fold_rows': fold_rows, 'per_repeat_accuracy': per_repeat,
                'accuracy_mean': float(np.mean(per_repeat)),
                'accuracy_std': float(np.std(per_repeat))}
    # per-repeat means
    per_repeat = {}
    for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
        per_repeat[key] = [float(np.mean([row[key] for row in fold_rows if row['repeat'] == r]))
                           for r in range(repeats)]
    # pooled OOF metrics per repeat
    pooled = {'accuracy': [], 'macro_f1': [], 'balanced_accuracy': []}
    per_class_recall = np.zeros((repeats, N_CLASSES))
    per_class_f1 = np.zeros((repeats, N_CLASSES))
    aucs = []
    for r in range(repeats):
        pooled['accuracy'].append(accuracy_score(y, oof[r]) * 100)
        pooled['macro_f1'].append(f1_score(y, oof[r], average='macro', zero_division=0) * 100)
        pooled['balanced_accuracy'].append(balanced_accuracy_score(y, oof[r]) * 100)
        per_class_recall[r] = recall_score(y, oof[r], average=None, zero_division=0,
                                           labels=list(range(N_CLASSES))) * 100
        per_class_f1[r] = f1_score(y, oof[r], average=None, zero_division=0,
                                   labels=list(range(N_CLASSES))) * 100
        try:
            yb = label_binarize(y, classes=list(range(N_CLASSES)))
            aucs.append(roc_auc_score(yb, oof_p[r], average='macro') * 100)
        except Exception:
            aucs.append(np.nan)
    out = {
        'learner': learner, 'balanced': balanced, 'n_select': n_select,
        'n_features_raw': int(d), 'repeats': repeats,
        'fold_rows': fold_rows,
        'per_repeat': per_repeat,
        'accuracy_mean': float(np.mean(per_repeat['accuracy'])),
        'accuracy_std': float(np.std(per_repeat['accuracy'])),
        'macro_f1_mean': float(np.mean(per_repeat['macro_f1'])),
        'macro_f1_std': float(np.std(per_repeat['macro_f1'])),
        'balanced_accuracy_mean': float(np.mean(per_repeat['balanced_accuracy'])),
        'balanced_accuracy_std': float(np.std(per_repeat['balanced_accuracy'])),
        'pooled_oof_accuracy_mean': float(np.mean(pooled['accuracy'])),
        'pooled_oof_macro_f1_mean': float(np.mean(pooled['macro_f1'])),
        'pooled_oof_balanced_accuracy_mean': float(np.mean(pooled['balanced_accuracy'])),
        'auc_mean': float(np.nanmean(aucs)),
        'per_class_recall_mean': per_class_recall.mean(axis=0).tolist(),
        'per_class_recall_std': per_class_recall.std(axis=0).tolist(),
        'per_class_f1_mean': per_class_f1.mean(axis=0).tolist(),
        'per_class_f1_std': per_class_f1.std(axis=0).tolist(),
    }
    if return_oof:
        out['oof_pred'] = oof.tolist()
        out['oof_proba'] = oof_p.tolist()
    return out


def bootstrap_ci_oof(oof_by_repeat, y, n_boot=10000, seed=0):
    """Bootstrap CI of pooled OOF accuracy (samples resampled, repeats averaged)."""
    rng = np.random.default_rng(seed)
    corr = np.mean([(np.asarray(o) == y).astype(float) for o in oof_by_repeat], axis=0)
    n = len(y)
    vals = np.array([corr[rng.integers(0, n, n)].mean() * 100 for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class_recall_ci(oof_by_repeat, y, cls, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    hits = np.mean([(np.asarray(o) == cls).astype(float) for o in oof_by_repeat], axis=0)
    mask = y == cls
    n = int(mask.sum())
    vals = np.array([hits[mask][rng.integers(0, n, n)].mean() * 100 for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def nadeau_bengio(diffs, k=K):
    """Corrected resampled t-test (Nadeau & Bengio 2003) on fold-level differences."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = np.var(d, ddof=1)
    denom = math.sqrt((1.0 / J + 1.0 / (k - 1)) * var)
    if denom == 0:
        return {'t': float('nan'), 'p': 1.0, 'mean_diff': float(np.mean(d)), 'J': J}
    t = float(np.mean(d) / denom)
    p = float(2 * stats.t.sf(abs(t), df=J - 1))
    return {'t': t, 'p': p, 'mean_diff': float(np.mean(d)), 'J': J}


def mcnemar_exact(pred_a, pred_b, y):
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {'n01': n01, 'n10': n10, 'p': p}


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj.tolist()


def fold_diffs(res_a, res_b, metric='accuracy'):
    """Paired fold-level differences (same repeat/fold order by construction)."""
    a = [row[metric] for row in res_a['fold_rows']]
    b = [row[metric] for row in res_b['fold_rows']]
    return [x - y for x, y in zip(a, b)]
