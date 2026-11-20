"""Downstream evaluation: per-model LLM conditions, consensus conditions, baselines,
agreement metrics, and statistical tests (t-test, McNemar, Friedman + Nemenyi).

All conditions share identical CV folds (StratifiedKFold, 5 folds, seed 42).
Outputs results/evaluation.json including OOF predictions for every condition.
"""
import json, os, sys, io, itertools, math
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import (load_expanded_dataset, load_original_dataset, load_external_dataset,
                        CANONICAL_RATIOS, CANONICAL_FEATURES, GASES, N_CLASSES, N_FOLDS,
                        RANDOM_STATE, cv_pipeline, build_feature_matrix, canonical_key,
                        reciprocal_key, is_equivalent, RESULTS_DIR, rank_to_weights,
                        evaluate_fold)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

PARSED_PATH = os.path.join(RESULTS_DIR, 'llm_parsed.json')
OUT_PATH = os.path.join(RESULTS_DIR, 'evaluation.json')


# ---------- helpers ----------
def oof_accuracy(pred, y):
    return float((pred == y).mean() * 100)


def bootstrap_ci(pred, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (pred == y).astype(float)
    n = len(correct)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = correct[idx].mean() * 100
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def mcnemar_exact(pred_a, pred_b, y):
    a_ok = pred_a == y
    b_ok = pred_b == y
    n01 = int(np.sum(~a_ok & b_ok))   # a wrong, b right
    n10 = int(np.sum(a_ok & ~b_ok))
    n = n01 + n10
    if n == 0:
        return {'n01': n01, 'n10': n10, 'p': 1.0}
    p = float(stats.binomtest(n01, n, 0.5).pvalue)
    return {'n01': n01, 'n10': n10, 'p': p}


def cochran_q(preds, y):
    """Cochran's Q omnibus test for k conditions evaluated on the same samples."""
    C = np.array([(np.asarray(p) == y).astype(int) for p in preds])  # k x n
    k, n = C.shape
    T = C.sum(axis=0)                      # per sample number of successes
    Cj = C.sum(axis=1)                     # per condition total successes
    Q = (k - 1) * (k * np.sum(Cj ** 2) - np.sum(Cj) ** 2) / \
        (k * np.sum(T) - np.sum(T ** 2))
    p = float(stats.chi2.sf(Q, k - 1))
    return {'Q': float(Q), 'df': k - 1, 'p': p, 'k': k, 'n': n,
            'condition_totals': Cj.tolist()}


def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def feature_columns_for_condition(X_raw, feats):
    """Return (matrix, names) where feats is a list of canonical display names
    containing raw gases and canonical ratios built as 'A+B/C+D'."""
    cols, names = [], []
    for f in feats:
        if f in GASES:
            cols.append(X_raw[:, GASES.index(f)])
            names.append(f)
    ratio_feats = [f for f in feats if f not in GASES]
    if ratio_feats:
        M = build_feature_matrix(X_raw, ratio_feats)
        for j, f in enumerate(ratio_feats):
            cols.append(M[:, j])
            names.append(f)
    return np.column_stack(cols), names


def _canon_to_name(canon):
    """Map a canonical display form to the manuscript's canonical ratio name (or None)."""
    for r in CANONICAL_RATIOS:
        if is_equivalent(canon, canonical_key(r)):
            return r
    return None


# ---------- condition assembly ----------
def model_conditions(parsed, X, y):
    """Build per-model conditions from validated LLM outputs.

    A1 (generation): 5 gases + the model's own validated S1 ratios, uniform weights.
    A2 (+redundancy): A1 minus canonical ratios the model's S2 recommends removing.
    A3 (+ranking): A2 with rank-decay weights from the model's S3 ranking.
    """
    results = {}
    for model, d in parsed.items():
        s1 = d['S1']['details']
        s2 = d['S2']['details']
        s3 = d['S3']['details']
        if not s1 or not s3:
            print(f'  [skip] {model}: incomplete stage data (S1={len(s1)}, S3={len(s3)})')
            continue
        # --- union over reps of accepted canonical forms (A1 feature space),
        #     folding reciprocal duplicates
        own = []
        seen_keys = set()
        for rep, v in s1.items():
            for c in v['canonical_forms']:
                rr = reciprocal_key(c)
                if c in seen_keys or (rr is not None and rr in seen_keys):
                    continue
                seen_keys.add(c)
                own.append(c)
        own = sorted(own, key=lambda c: CANONICAL_FEATURES.index(_canon_to_name(c))
                     if _canon_to_name(c) else 99)
        if not own:
            continue
        feats = GASES + own
        Xc, names = feature_columns_for_condition(X, feats)

        # A1: own generation, uniform weights
        r1 = cv_pipeline(Xc, y, weights=None, n_select=min(17, Xc.shape[1]))
        results[f'{model}|A1-generation'] = summarize(r1, y, names)

        # A2: A1 minus the channels the model's S2 recommends removing
        remove_votes = {}
        for rep, v in s2.items():
            for f in v['remove']:
                remove_votes[f] = remove_votes.get(f, 0) + 1
        n_rep = max(len(s2), 1)
        removed_names = {f for f, v in remove_votes.items() if v > n_rep / 2}
        keep_idx = [i for i, nm in enumerate(names)
                    if nm in GASES or (_canon_to_name(nm) or nm) not in removed_names]
        Xc2, names2 = Xc[:, keep_idx], [names[i] for i in keep_idx]
        r2 = cv_pipeline(Xc2, y, weights=None, n_select=min(17, Xc2.shape[1]))
        results[f'{model}|A2-redundancy'] = summarize(r2, y, names2)

        # A3: ranking protocol on the shared canonical 19-feature space, with
        # rank-decay weights derived from the model's S3 ranking
        merged_rank = merge_rankings([v['ranking'] for v in s3.values()])
        w19 = rank_to_weights(merged_rank, CANONICAL_FEATURES)
        r3 = cv_pipeline(X, y, weights=w19, n_select=17)
        results[f'{model}|A3-ranking'] = summarize(r3, y, CANONICAL_FEATURES,
                                                   weights=w19.tolist(),
                                                   ranking=merged_rank)
    return results


def merge_rankings(rankings):
    """Average multiple rankings of the same feature set into one (mean rank)."""
    acc = {}
    for rk in rankings:
        for item in rk:
            acc.setdefault(item['feature'], []).append(item['rank'])
    merged = [{'rank': float(np.mean(v)), 'feature': k} for k, v in acc.items()]
    merged.sort(key=lambda x: x['rank'])
    for i, m in enumerate(merged, 1):
        m['rank'] = i
    return merged


def summarize(cvres, y, names, **extra):
    agg = cvres['agg']
    pred = cvres['oof_pred']
    ci = bootstrap_ci(pred, y)
    out = {
        'accuracy': agg['accuracy']['mean'], 'accuracy_std': agg['accuracy']['std'],
        'accuracy_folds': agg['accuracy']['values'],
        'precision': agg['precision']['mean'], 'recall': agg['recall']['mean'],
        'f1': agg['f1']['mean'], 'auc': agg['auc']['mean'],
        'oof_accuracy': oof_accuracy(pred, y), 'ci95': list(ci),
        'n_features': len(names), 'features': names,
        'oof_pred': pred.tolist(),
        'per_class_f1': per_class_f1(pred, y),
    }
    out.update(extra)
    return out


def per_class_f1(pred, y):
    from sklearn.metrics import f1_score
    vals = f1_score(y, pred, average=None, zero_division=0, labels=list(range(N_CLASSES)))
    return {str(i): float(v) for i, v in enumerate(vals)}


def baseline_conditions(X, y):
    res = {}
    # Uniform 19 (top-17 by importance)
    r = cv_pipeline(X, y, weights=None, n_select=17)
    res['Baseline|Uniform19'] = summarize(r, y, CANONICAL_FEATURES)
    # Manuscript fixed weights
    w = np.ones(len(CANONICAL_FEATURES))
    for i, n in enumerate(CANONICAL_FEATURES):
        if '/' in n:
            w[i] = 1.2
        if n in ('H2', 'C2H2'):
            w[i] = 1.15
    r = cv_pipeline(X, y, weights=w, n_select=17)
    res['Baseline|FixedWeight19'] = summarize(r, y, CANONICAL_FEATURES, weights=w.tolist())
    # Raw five gases
    r = cv_pipeline(X[:, :5], y, weights=None, n_select=None)
    res['Baseline|RawGas5'] = summarize(r, y, GASES)
    # Statistical selection
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for selector_name, sel in [('ANOVA-F17', SelectKBest(f_classif, k=17)),
                               ('MutualInfo-17', SelectKBest(mutual_info_classif, k=17))]:
        folds, oof, oof_proba = [], np.zeros(len(y), dtype=int), np.zeros((len(y), N_CLASSES))
        for tr, te in cv.split(X, y):
            sc = StandardScaler()
            X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
            X_tr_s = sel.fit_transform(X_tr, y[tr])
            X_te_s = sel.transform(X_te)
            rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                        random_state=RANDOM_STATE)
            rf.fit(X_tr_s, y[tr])
            pred = rf.predict(X_te_s)
            proba = rf.predict_proba(X_te_s)
            oof[te] = pred
            oof_proba[te] = proba
            folds.append(evaluate_fold(y[te], pred, proba))
        agg = {}
        for k_ in folds[0]:
            vals = [f[k_] for f in folds]
            agg[k_] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                       'values': [float(v) for v in vals]}
        ci = bootstrap_ci(oof, y)
        res[f'Baseline|{selector_name}'] = {
            'accuracy': agg['accuracy']['mean'], 'accuracy_std': agg['accuracy']['std'],
            'accuracy_folds': agg['accuracy']['values'],
            'precision': agg['precision']['mean'], 'recall': agg['recall']['mean'],
            'f1': agg['f1']['mean'], 'auc': agg['auc']['mean'],
            'oof_accuracy': oof_accuracy(oof, y), 'ci95': list(ci),
            'oof_pred': oof.tolist(),
            'n_features': 17, 'features': 'top-17 by ' + selector_name,
            'per_class_f1': per_class_f1(oof, y),
        }
    return res


def consensus_conditions(X, y, parsed):
    """Borda fusion and union-based majority voting across all models."""
    res = {}
    all_rankings, model_sets = [], []
    for model, d in parsed.items():
        for rep, v in d['S3']['details'].items():
            all_rankings.append(v['ranking'])
        # model-level proposal set (union over its replicates)
        S = set()
        for rep, v in d['S1']['details'].items():
            S.update(v['canonical_forms'])
        model_sets.append(S)

    merged = merge_rankings(all_rankings)
    w_cons = rank_to_weights(merged, CANONICAL_FEATURES)

    # ---- Borda weights on the canonical 19-feature space
    r = cv_pipeline(X, y, weights=w_cons, n_select=17)
    res['Consensus|BordaRanking'] = summarize(r, y, CANONICAL_FEATURES,
                                              consensus_ranking=merged,
                                              weights=w_cons.tolist())

    # ---- union-based majority voting over the validated proposals
    canon_key = lambda e: canonical_key(e)
    votes = {}
    for S in model_sets:
        folded = set()
        for e in S:
            rr = reciprocal_key(e)
            key = e if (rr is None or e <= rr) else rr
            folded.add(key)
        for e in folded:
            votes[e] = votes.get(e, 0) + 1
    thr = len(model_sets) / 2  # endorsed by at least half of the models
    # deterministic ordering: vote count first, then the canonical string
    endorsed = sorted([e for e, v in votes.items() if v >= thr],
                      key=lambda e: (-votes[e], e))
    feats = GASES + endorsed
    Xc, names = feature_columns_for_condition(X, feats)
    r = cv_pipeline(Xc, y, weights=None, n_select=min(17, Xc.shape[1]))
    res['Consensus|MajorityVote'] = summarize(r, y, names, votes=votes,
                                              threshold=thr, n_union=len(votes))

    # ---- full consensus: union-vote mask + Borda weights where available
    name_to_w = dict(zip(CANONICAL_FEATURES, w_cons))
    w = np.array([name_to_w.get(_canon_to_name(nm), 1.0) if nm not in GASES else 1.0
                  for nm in names])
    r = cv_pipeline(Xc, y, weights=w, n_select=min(17, Xc.shape[1]))
    res['Consensus|Full'] = summarize(r, y, names, weights=w.tolist())

    # ---- union of all proposals (upper bound of the generation stage)
    union = set()
    for S in model_sets:
        for e in S:
            rr = reciprocal_key(e)
            union.add(e if (rr is None or e <= rr) else rr)
    ufeats = GASES + sorted(union)
    Xu, unames = feature_columns_for_condition(X, ufeats)
    r = cv_pipeline(Xu, y, weights=None, n_select=min(17, Xu.shape[1]))
    res['Consensus|UnionAll'] = summarize(r, y, unames, n_union=len(union))
    return res


def agreement_metrics(parsed):
    """Cross-model agreement: Spearman on rankings, Jaccard on S1 sets, Fleiss kappa."""
    models = list(parsed.keys())
    # Spearman matrix (mean over reps) on the canonical 19
    rank_vec = {}
    for m in models:
        rs = []
        for rep, v in parsed[m]['S3']['details'].items():
            pos = {it['feature']: it['rank'] for it in v['ranking']}
            rs.append([pos[f] for f in CANONICAL_FEATURES])
        rank_vec[m] = np.mean(rs, axis=0) if rs else None
    spear = {}
    for a, b in itertools.combinations(models, 2):
        if rank_vec[a] is None or rank_vec[b] is None:
            continue
        rho, p = stats.spearmanr(rank_vec[a], rank_vec[b])
        spear[f'{a}~{b}'] = {'rho': float(rho), 'p': float(p)}
    # Jaccard on S1 canonical projections
    jac = {}
    for a, b in itertools.combinations(models, 2):
        vals = []
        for ra, va in parsed[a]['S1']['details'].items():
            for rb, vb in parsed[b]['S1']['details'].items():
                A = {rec.get('matches_canonical') for rec in va['accepted']} - {None}
                B = {rec.get('matches_canonical') for rec in vb['accepted']} - {None}
                if A | B:
                    vals.append(len(A & B) / len(A | B))
        if vals:
            jac[f'{a}~{b}'] = float(np.mean(vals))
    # Fleiss kappa on endorsement of the eleven standard ratio codes
    # (IEC 60599 / Rogers / Doernenburg), across all (model, replicate) raters
    from exp_common import STANDARD_RATIOS
    raters = [(m, rep) for m in models for rep in parsed[m]['S1']['details']]
    std_defs = {}
    for name, (a, b) in STANDARD_RATIOS.items():
        std_defs[name] = (canonical_key(f'{a}/{b}'), canonical_key(f'{b}/{a}'))
    mat = np.zeros((len(std_defs), len(raters)), dtype=int)
    for j, (m, rep) in enumerate(raters):
        forms = set(parsed[m]['S1']['details'][rep]['canonical_forms'])
        for i, (name, (c1, c2)) in enumerate(std_defs.items()):
            hit = any(f == c1 or f == c2 for f in forms)
            mat[i, j] = 1 if hit else 0
    n = mat.shape[1]
    p_i = mat.mean(axis=1)
    # Fleiss P_i for binary items: agreeing pairs / total pairs
    P_bar = float(np.mean([(k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
                           for k in mat.sum(axis=1)]))
    P_e = float(np.mean(p_i) ** 2 + np.mean(1 - p_i) ** 2)
    kappa = (P_bar - P_e) / (1 - P_e) if P_e != 1 else float('nan')
    # Intra-model stability
    stability = {}
    for m in models:
        s3r = list(parsed[m]['S3']['details'].values())
        if len(s3r) >= 2:
            rhos = []
            for a, b in itertools.combinations(s3r, 2):
                pa = {it['feature']: it['rank'] for it in a['ranking']}
                pb = {it['feature']: it['rank'] for it in b['ranking']}
                rhos.append(stats.spearmanr([pa[f] for f in CANONICAL_FEATURES],
                                            [pb[f] for f in CANONICAL_FEATURES]).statistic)
            stability[m] = float(np.mean(rhos))
    return {'spearman_pairwise': spear, 'jaccard_pairwise': jac,
            'fleiss_kappa': float(kappa), 'intra_model_stability_spearman': stability,
            'rank_vectors': {m: (rank_vec[m].tolist() if rank_vec[m] is not None else None) for m in models}}


def friedman_nemenyi(cond_acc_folds, alpha=0.05):
    """Friedman test over conditions using per-fold accuracies; Nemenyi CD."""
    names = list(cond_acc_folds.keys())
    data = np.array([cond_acc_folds[n] for n in names])  # k x N (folds)
    k, N = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, data)   # rank within each fold? we need per-block ranking
    # Friedman ranks: rank conditions within each fold (block = fold)
    block_ranks = np.apply_along_axis(stats.rankdata, 0, data)  # N x k ranks per fold
    mean_ranks = block_ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[data[i] for i in range(k)])
    # Nemenyi critical difference
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
               15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}.get(k, 3.6)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * N))
    return {'conditions': names, 'mean_ranks': {n: float(r) for n, r in zip(names, mean_ranks)},
            'friedman_chi2': float(chi2), 'friedman_p': float(p), 'nemenyi_cd': float(cd),
            'k': k, 'N': N}


def main():
    X, y, cols = load_expanded_dataset()
    parsed = json.load(open(PARSED_PATH, encoding='utf-8'))
    DL = json.load(open(os.path.join(RESULTS_DIR, 'dl_baselines.json'), encoding='utf-8'))
    print('models with parsed outputs:', list(parsed.keys()))

    results = {}
    print('running baselines...')
    results.update(baseline_conditions(X, y))
    print('running per-model conditions...')
    results.update(model_conditions(parsed, X, y))
    print('running consensus conditions...')
    results.update(consensus_conditions(X, y, parsed))
    print('computing agreement metrics...')
    agree = agreement_metrics(parsed)

    # ---- statistical tests vs baselines ----
    base_pred = {k: np.array(v['oof_pred']) for k, v in results.items()}
    ref_uniform = base_pred['Baseline|Uniform19']
    ref_fixed = base_pred['Baseline|FixedWeight19']
    tests = {}
    for k, pred in base_pred.items():
        if k.startswith('Baseline'):
            continue
        t_uniform = stats.ttest_rel(results[k]['accuracy_folds'],
                                    results['Baseline|Uniform19']['accuracy_folds'])
        t_fixed = stats.ttest_rel(results[k]['accuracy_folds'],
                                  results['Baseline|FixedWeight19']['accuracy_folds'])
        tests[k] = {
            'delta_vs_uniform': results[k]['accuracy'] - results['Baseline|Uniform19']['accuracy'],
            'ttest_p_vs_uniform': float(t_uniform.pvalue),
            'delta_vs_fixed': results[k]['accuracy'] - results['Baseline|FixedWeight19']['accuracy'],
            'ttest_p_vs_fixed': float(t_fixed.pvalue),
            'mcnemar_vs_uniform': mcnemar_exact(pred, ref_uniform, y),
            'mcnemar_vs_fixed': mcnemar_exact(pred, ref_fixed, y),
        }

    # best single-model A3 condition (computed from the evaluated models only)
    _a3 = [k for k in results if k.endswith('|A3-ranking')]
    BM = max(_a3, key=lambda k: results[k]['oof_accuracy']) if _a3 else None
    print('best single-model condition:', BM, results[BM]['oof_accuracy'] if BM else None)

    # ---- McNemar + Holm across a curated pairwise comparison set, and Cochran Q ----
    mcnemar_pairs = [
        ('Consensus vs raw gases', 'Consensus|BordaRanking', 'Baseline|RawGas5'),
        ('Consensus vs uniform weights', 'Consensus|BordaRanking', 'Baseline|Uniform19'),
        ('Consensus vs fixed LLM weights', 'Consensus|BordaRanking', 'Baseline|FixedWeight19'),
        ('Consensus vs best single model', 'Consensus|BordaRanking', BM),
        ('Best single model vs uniform', BM, 'Baseline|Uniform19'),
        ('Best single model vs raw gases', BM, 'Baseline|RawGas5'),
        ('Best single model vs Transformer', BM, 'DL:Transformer-DL'),
        ('Uniform vs raw gases', 'Baseline|Uniform19', 'Baseline|RawGas5'),
        ('Fixed weights vs uniform', 'Baseline|FixedWeight19', 'Baseline|Uniform19'),
    ]

    def get_pred(key):
        if key.startswith('DL:'):
            return np.array(DL[key[3:]]['oof_pred'])
        return np.array(results[key]['oof_pred'])

    def get_cond(key):
        if key.startswith('DL:'):
            return DL[key[3:]]
        return results[key]

    pairwise = {}
    raw_p = []
    for label, a, b in mcnemar_pairs:
        mc = mcnemar_exact(get_pred(a), get_pred(b), y)
        acc_a, acc_b = get_cond(a)['oof_accuracy'], get_cond(b)['oof_accuracy']
        pairwise[label] = {'a': a, 'b': b, 'acc_a': acc_a, 'acc_b': acc_b,
                           'delta': acc_a - acc_b, 'mcnemar': mc}
        raw_p.append(mc['p'])
    adj = holm_adjust(raw_p)
    for (label, _, _), pa in zip(mcnemar_pairs, adj):
        pairwise[label]['p_holm'] = pa

    omnibus_keys = ['Baseline|RawGas5', 'Baseline|Uniform19', 'Baseline|FixedWeight19',
                    'Consensus|BordaRanking', 'Consensus|Full', BM,
                    'DL:Transformer-DL']
    q = cochran_q([get_pred(k) for k in omnibus_keys], y)
    q['conditions'] = omnibus_keys

    # ---- Friedman across a curated set of conditions (kept for continuity, with
    #      an explicit tie warning when several conditions share per-fold results)
    models_sorted = sorted([m for m in parsed],
                           key=lambda m: -results.get(f'{m}|A3-ranking', {}).get('oof_accuracy', 0))
    curated = ['Baseline|RawGas5', 'Baseline|Uniform19', 'Baseline|FixedWeight19',
               'Consensus|BordaRanking', 'Consensus|MajorityVote', 'Consensus|Full']
    curated += [f'{m}|A3-ranking' for m in models_sorted[:2]]
    fold_map = {k: results[k]['accuracy_folds'] for k in curated if k in results}
    fried = friedman_nemenyi(fold_map)

    # ---- cross-dataset validation for key conditions ----
    print('cross-dataset validation...')
    cross = {}
    for ds_name, loader in [('Original570', load_original_dataset),
                            ('External504', load_external_dataset)]:
        Xd_raw, yd = loader()
        Xd, _ = feature_columns_for_condition(Xd_raw, CANONICAL_FEATURES)
        # Borda consensus weights come from the primary-dataset rankings; reuse
        merged = merge_rankings([it for m in parsed for rep, v in parsed[m]['S3']['details'].items()
                                 for it in [v['ranking']]])
        w_cons = rank_to_weights(merged, CANONICAL_FEATURES)
        r_uni = cv_pipeline(Xd, yd, weights=None, n_select=17)
        r_llm = cv_pipeline(Xd, yd, weights=w_cons, n_select=17)
        r_raw = cv_pipeline(Xd[:, :5], yd, weights=None, n_select=None)
        cross[ds_name] = {
            'Uniform19': summarize(r_uni, yd, CANONICAL_FEATURES),
            'Consensus|BordaRanking': summarize(r_llm, yd, CANONICAL_FEATURES),
            'RawGas5': summarize(r_raw, yd, GASES),
        }

    out = {'conditions': results, 'statistical_tests': tests,
           'pairwise_mcnemar': pairwise, 'cochran_q': q,
           'agreement': agree, 'friedman': fried, 'cross_dataset': cross,
           'meta': {'n_samples': int(len(y)), 'features': cols}}
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)

    # ---- console summary ----
    print()
    print(f"{'condition':<40}{'acc':>7}{'std':>7}{'oof':>7}{'ci95':>16}")
    for k, v in results.items():
        print(f"{k:<40}{v['accuracy']:>7.2f}{v['accuracy_std']:>7.2f}{v['oof_accuracy']:>7.2f}"
              f"{str([round(c,2) for c in v['ci95']]):>16}")
    print()
    print('Cochran Q = %.2f (df=%d, p=%.2e) across %d conditions' %
          (q['Q'], q['df'], q['p'], q['k']))
    print('pairwise McNemar (Holm-adjusted):')
    for label, rec in pairwise.items():
        print(f"  {label:<36} d={rec['delta']:+.2f} p={rec['mcnemar']['p']:.4f} "
              f"p_holm={rec['p_holm']:.4f}")
    print('Friedman chi2=%.2f p=%.2e CD=%.2f' % (fried['friedman_chi2'], fried['friedman_p'],
                                                 fried['nemenyi_cd']))
    print('Fleiss kappa (S1 endorsement):', round(agree['fleiss_kappa'], 3))
    print('saved:', OUT_PATH)


if __name__ == '__main__':
    main()
