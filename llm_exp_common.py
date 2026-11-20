"""Common utilities: data loading, canonical features, model registry, ML pipeline."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc as sk_auc)
from sklearn.preprocessing import label_binarize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
RESULTS_DIR = os.path.join(BASE, 'results')
LOGS_DIR = os.path.join(BASE, 'llm_logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FAULT_NAMES = ['Normal', 'LTO', 'HTO', 'PD', 'LED', 'HED']
N_CLASSES = 6
N_FOLDS = 5
RANDOM_STATE = 42
GASES = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']

# ---- Canonical feature space (from the original manuscript) ----
CANONICAL_RATIOS = [
    'H2/(CH4+C2H6)', 'H2/(C2H4+C2H2)', 'H2/ALL',
    'CH4/(H2+C2H6)', 'CH4/(C2H4+C2H2)', 'CH4/ALL',
    'C2H6/(H2+CH4)', 'C2H6/(C2H4+C2H2)', 'C2H6/ALL',
    'C2H4/(H2+CH4)', 'C2H4/(C2H6+C2H2)', 'C2H4/ALL',
    '(CH4+C2H6)/TH', '(C2H6+C2H4)/TH',
]
CANONICAL_FEATURES = GASES + CANONICAL_RATIOS  # 19
EPS = 1e-8  # matches the stored dataset columns (verified by exact reconstruction)

# IEC 60599 / Rogers / Doernenburg ratio codes for cross-reference
STANDARD_RATIOS = {
    'IEC60599_C2H2/C2H4': ('C2H2', 'C2H4'),
    'IEC60599_CH4/H2': ('CH4', 'H2'),
    'IEC60599_C2H4/C2H6': ('C2H4', 'C2H6'),
    'Rogers_C2H2/C2H4': ('C2H2', 'C2H4'),
    'Rogers_CH4/H2': ('CH4', 'H2'),
    'Rogers_C2H4/C2H6': ('C2H4', 'C2H6'),
    'Rogers_C2H6/CH4': ('C2H6', 'CH4'),
    'Doernenburg_CH4/H2': ('CH4', 'H2'),
    'Doernenburg_C2H2/C2H4': ('C2H2', 'C2H4'),
    'Doernenburg_C2H2/CH4': ('C2H2', 'CH4'),
    'Doernenburg_C2H6/C2H2': ('C2H6', 'C2H2'),
}


def load_expanded_dataset():
    """Load the 550-sample expanded DGA dataset (primary evaluation set).

    The 'category' column stores integer codes that were verified against the
    original labelled dataset (原始.xlsx) by exact gas-concentration row matching:
        1 = LTO, 2 = HTO, 3 = PD, 4 = LED, 5 = HED, 6 = Normal
    Class indices are mapped to FAULT_NAMES order [Normal, LTO, HTO, PD, LED, HED].
    """
    path = os.path.join(DATA_DIR, 'expanded_dga_550.xlsx')
    df = pd.read_excel(path)
    code_to_name = {1: 'LTO', 2: 'HTO', 3: 'PD', 4: 'LED', 5: 'HED', 6: 'Normal'}
    name_to_idx = {n: i for i, n in enumerate(FAULT_NAMES)}
    y = df['category'].map(lambda c: name_to_idx[code_to_name[int(c)]]).values.astype(int)
    feature_cols = [c for c in df.columns if c not in ['category', 'ALL']]
    X = df[feature_cols].values.astype(float)
    return X, y, feature_cols


def load_external_dataset():
    """504-sample external validation dataset (5 raw gases)."""
    path = os.path.join(DATA_DIR, 'external_504.xlsx')
    df = pd.read_excel(path)
    y = df.iloc[:, -1].values.astype(int) - 1
    X = df.iloc[:, :5].values.astype(float)
    return X, y


def load_original_dataset():
    """570-sample original dataset (5 raw gases, merge main 6 classes)."""
    path = os.path.join(DATA_DIR, 'original_570.xlsx')
    df = pd.read_excel(path)
    label_map = {'正常': 'Normal', '中低温过热': 'LTO', '高温过热': 'HTO',
                 '局部放电': 'PD', '低能放电': 'LED', '高能放电': 'HED',
                 '火花放电': 'Spark', '电弧放电': 'Arc'}
    df['label'] = df[df.columns[-1]].map(label_map)
    df = df[df['label'].isin(FAULT_NAMES)].copy()
    # explicit order so that the class indices match load_expanded_dataset
    name_to_idx = {n: i for i, n in enumerate(FAULT_NAMES)}
    y = df['label'].map(name_to_idx).values.astype(int)
    X = df[GASES].values.astype(float)
    return X, y


# ---- Expression grammar for LLM-proposed ratios ----
import re


def _tokenize(s):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in '+-*/()':
            toks.append(c)
            i += 1
            continue
        m = re.match(r'[A-Za-z0-9_]+', s[i:])
        if not m:
            return None
        toks.append(m.group(0))
        i += len(m.group(0))
    return toks


def _parse_side(toks):
    """Parse a sum/difference of gases and parenthesized sub-sums.

    Returns {gas: coefficient} or None if inadmissible (unknown token, '*', etc.).
    """
    coeff = {}
    sign = 1
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == '+':
            sign = 1
            i += 1
            continue
        if t == '-':
            sign = -sign
            i += 1
            continue
        if t == '(':
            depth = 0
            j = i
            while j < len(toks):
                if toks[j] == '(':
                    depth += 1
                elif toks[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= len(toks):
                return None
            inner = _parse_side(toks[i + 1:j])
            if inner is None:
                return None
            for g, c in inner.items():
                coeff[g] = coeff.get(g, 0) + sign * c
            i = j + 1
            sign = 1
        elif t in GASES:
            coeff[t] = coeff.get(t, 0) + sign
            i += 1
            sign = 1
        else:
            return None  # unknown symbol, '*' or numeric coefficient
    return coeff


def parse_expression(expr):
    """Parse 'H2/(CH4+C2H6)' into ({gas: coefficient}, {gas: coefficient}).

    Returns (num_dict, den_dict) or None if inadmissible.
    Grammar: ratio := side ('/' side)? ; side := term (('+'|'-') term)* ;
             term := gas | '(' side ')'
    """
    if not isinstance(expr, str):
        return None
    s = expr.strip()
    s = (s.replace('₂', '2').replace('₄', '4').replace('₆', '6')
          .replace('H₂', 'H2').replace('CH₄', 'CH4').replace('C₂H₆', 'C2H6')
          .replace('C₂H₄', 'C2H4').replace('C₂H₂', 'C2H2'))
    s = re.sub(r'[Hh]ydrogen', 'H2', s)
    s = re.sub(r'[Mm]ethane(?!\w)', 'CH4', s)
    s = re.sub(r'[Ee]thane(?!\w)', 'C2H6', s)
    s = re.sub(r'[Ee]thylene|[Ee]thene', 'C2H4', s)
    s = re.sub(r'[Aa]cetylene', 'C2H2', s)
    toks = _tokenize(s)
    if not toks:
        return None
    # split at the top-level slash (exactly one allowed)
    depth = 0
    slash = -1
    for i, t in enumerate(toks):
        if t == '(':
            depth += 1
        elif t == ')':
            depth -= 1
            if depth < 0:
                return None
        elif t == '/' and depth == 0:
            if slash >= 0:
                return None
            slash = i
    if depth != 0:
        return None
    num_toks = toks[:slash] if slash >= 0 else toks
    den_toks = toks[slash + 1:] if slash >= 0 else None
    num = _parse_side(num_toks)
    den = _parse_side(den_toks) if den_toks is not None else None
    if num is None or (den_toks is not None and den is None):
        return None
    if not num:
        return None
    return (num, den)


GAS_ORDER = {g: i for i, g in enumerate(GASES)}


def _fmt_side(coeff):
    return '+'.join(g for g in sorted(coeff, key=lambda g: GAS_ORDER.get(g, 99))
                    for _ in range(int(coeff[g])))


def canonical_key(expr):
    """Canonical display form 'NUM/DEN' preserving orientation, order-insensitive.

    Returns None for bare gases or inadmissible expressions.
    """
    p = parse_expression(expr)
    if p is None:
        return None
    num, den = p
    if den is None or not den:
        return None
    if any(v <= 0 for v in num.values()) or any(v <= 0 for v in den.values()):
        return None
    return f'{_fmt_side(num)}/{_fmt_side(den)}'


def reciprocal_key(canon):
    """Reciprocal of a canonical form (for equivalence tests)."""
    if not canon or '/' not in canon:
        return None
    a, b = canon.split('/')
    return f'{b}/{a}'


def is_equivalent(c1, c2):
    """True if two canonical forms represent the same or reciprocal ratio."""
    if c1 is None or c2 is None:
        return False
    return c1 == c2 or c1 == reciprocal_key(c2)


def canonical_equivalent(expr):
    """Normalize a valid expression to its canonical display form (or None)."""
    return canonical_key(expr)


def expand_aggregates(expr):
    """Replace the ALL and TH aggregate tokens with explicit gas sums."""
    if not isinstance(expr, str):
        return expr
    s = expr.replace('ALL', '(H2+CH4+C2H6+C2H4+C2H2)')
    s = s.replace('TH', '(CH4+C2H6+C2H4+C2H2)')
    return s


def compute_ratio_column(X_raw, expr):
    """Evaluate one ratio expression (ages ALL/TH) against a (n,5) gas matrix."""
    p = parse_expression(expand_aggregates(expr))
    if p is None:
        raise ValueError(f'inadmissible expression: {expr}')
    num, den = p
    n = len(X_raw)
    num_v = np.zeros(n)
    den_v = np.zeros(n)
    for g, c in num.items():
        num_v += c * X_raw[:, GASES.index(g)]
    for g, c in den.items():
        den_v += c * X_raw[:, GASES.index(g)]
    return num_v / (den_v + EPS)


def build_feature_matrix(X_raw, expr_list):
    """Compute ratio features for a list of expressions (aggregates expanded)."""
    cols = [compute_ratio_column(X_raw, e) for e in expr_list]
    return np.column_stack(cols) if cols else np.zeros((len(X_raw), 0))


# ---- ML pipeline (identical across all LLM conditions) ----
def rank_to_weights(ranking, features, alpha=0.20):
    """Linear rank-decay weights: top rank -> 1+alpha, last -> 1.0."""
    n = len(features)
    w = np.ones(n)
    if not ranking:
        return w
    pos = {f: i for i, f in enumerate(features)}
    ranks = {}
    for item in ranking:
        f = item.get('feature') if isinstance(item, dict) else None
        r = item.get('rank') if isinstance(item, dict) else None
        if f in pos and isinstance(r, (int, float)):
            ranks[f] = min(max(int(r), 1), n)
    # fill missing ranks after the assigned ones, preserving order
    used = sorted(ranks.values())
    next_rank = 1
    for f in features:
        if f not in ranks:
            while next_rank in used:
                next_rank += 1
            ranks[f] = next_rank
            used.append(next_rank)
            used.sort()
    for f, r in ranks.items():
        w[pos[f]] = 1.0 + alpha * (1.0 - (r - 1) / max(n - 1, 1))
    return w


def evaluate_fold(y_true, y_pred, y_proba=None):
    res = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
    }
    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=range(N_CLASSES))
        aucs = []
        for i in range(N_CLASSES):
            if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                aucs.append(sk_auc(fpr, tpr))
        res['auc'] = np.mean(aucs) * 100 if aucs else np.nan
    return res


def cv_pipeline(X_base, y, feature_idx=None, weights=None, n_select=17,
                rf_final=dict(n_estimators=150, max_depth=20, min_samples_leaf=2),
                n_folds=N_FOLDS, seed=RANDOM_STATE, return_oof=True):
    """Run the manuscript's LLM+RF pipeline: z-score -> weight -> RF-importance
    selection -> final RF, evaluated in stratified 5-fold CV.

    X_base: (n, d) raw features
    feature_idx: optional column subset of X_base to use
    weights: optional (d_used,) multiplicative weights applied after scaling
    Returns dict with per-fold metrics, aggregated means/stds, and OOF predictions.
    """
    if feature_idx is not None:
        X = X_base[:, feature_idx]
    else:
        X = X_base
    d = X.shape[1]
    if weights is None:
        weights = np.ones(d)
    weights = np.asarray(weights, dtype=float)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES))
    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr]) * weights
        X_te = scaler.transform(X[te]) * weights
        rf_fs = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_fs.fit(X_tr, y[tr])
        if n_select is not None and n_select < d:
            top = np.argsort(rf_fs.feature_importances_)[::-1][:n_select]
            X_tr_s, X_te_s = X_tr[:, top], X_te[:, top]
        else:
            X_tr_s, X_te_s = X_tr, X_te
        rf = RandomForestClassifier(random_state=seed, **rf_final)
        rf.fit(X_tr_s, y[tr])
        pred = rf.predict(X_te_s)
        proba = rf.predict_proba(X_te_s)
        oof_pred[te] = pred
        oof_proba[te] = proba
        folds.append(evaluate_fold(y[te], pred, proba))
    agg = {}
    for k in folds[0]:
        vals = [f[k] for f in folds]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': [float(v) for v in vals]}
    out = {'folds': folds, 'agg': agg}
    if return_oof:
        out['oof_pred'] = oof_pred
        out['oof_proba'] = oof_proba
    return out
