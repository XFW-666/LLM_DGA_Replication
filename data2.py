"""Revision data layer: cleaned datasets, verified label mappings, feature builders.

Cleaning rules (documented in the manuscript):
  primary (expanded 550 -> 519):
    - exact duplicate 5-gas vectors: keep the first occurrence (29 removed)
    - the one duplicate group with conflicting labels (HTO vs PD) is dropped entirely (2 rows)
  external (504 -> 387):
    - label mapping verified against 117 overlapping records:
        1=LTO, 2=HTO, 3=LED, 4=HED, 5=PD, 6=Normal
    - rows whose gas vector also occurs in the (cleaned) primary set are removed
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
from exp_common import (GASES, CANONICAL_RATIOS, CANONICAL_FEATURES, STANDARD_RATIOS,
                        canonical_key, reciprocal_key, build_feature_matrix)

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'results_rev')
os.makedirs(OUT, exist_ok=True)

FAULT_NAMES = ['Normal', 'LTO', 'HTO', 'PD', 'LED', 'HED']
IDX = {n: i for i, n in enumerate(FAULT_NAMES)}
# expanded dataset codes (verified by row matching in the previous revision)
EXP_CODE = {1: 'LTO', 2: 'HTO', 3: 'PD', 4: 'LED', 5: 'HED', 6: 'Normal'}
# external dataset codes (verified in this revision against 117 overlapping rows)
EXT_CODE = {1: 'LTO', 2: 'HTO', 3: 'LED', 4: 'HED', 5: 'PD', 6: 'Normal'}


def gas_key(X):
    return pd.DataFrame(X[:, :5], columns=GASES).round(8).astype(str).agg('|'.join, axis=1)


def load_primary_clean(return_meta=False):
    df = pd.read_excel(os.path.join(DATA, 'expanded_dga_550.xlsx'))
    y = df['category'].map(lambda c: IDX[EXP_CODE[int(c)]]).values.astype(int)
    X = df[[c for c in df.columns if c not in ['category', 'ALL']]].values.astype(float)
    key = gas_key(X)
    g = pd.DataFrame({'k': key, 'y': y})
    conflicts = [k for k, sub in g.groupby('k')
                 if len(sub) > 1 and sub['y'].nunique() > 1]
    keep, seen = np.ones(len(X), bool), set()
    for i, k in enumerate(key):
        if k in conflicts:
            keep[i] = False          # ambiguous duplicate group -> drop both copies
        elif k in seen:
            keep[i] = False          # duplicate -> keep first
        else:
            seen.add(k)
    n_removed = int((~keep).sum())
    Xc, yc = X[keep], y[keep]
    meta = {
        'n_raw': int(len(X)), 'n_clean': int(len(Xc)),
        'n_removed': n_removed,
        'n_duplicate_groups': int(sum(1 for k, sub in g.groupby('k') if len(sub) > 1)),
        'n_rows_in_duplicate_groups': int(sum(len(sub) for k, sub in g.groupby('k')
                                             if len(sub) > 1)),
        'conflicting_groups': [{'key': k, 'labels': sorted(set(int(v) for v in
                                                               g[g['k'] == k]['y']))}
                               for k in conflicts],
        'class_counts': {FAULT_NAMES[i]: int(v) for i, v in enumerate(np.bincount(yc, minlength=6))},
        'sha256_gas_matrix': hashlib.sha256(np.round(Xc[:, :5], 8).tobytes()).hexdigest(),
    }
    return (Xc, yc, meta) if return_meta else (Xc, yc)


def load_external_clean(primary_X):
    df = pd.read_excel(os.path.join(DATA, 'external_504.xlsx'))
    X = df[GASES].values.astype(float)
    codes = df.iloc[:, -1].astype(int).values
    y = np.array([IDX[EXT_CODE[int(c)]] for c in codes])
    key = gas_key(X)
    pkeys = set(gas_key(primary_X))
    overlap = key.isin(pkeys).values
    dup = key.duplicated(keep='first').values
    keep = (~overlap) & (~dup)
    meta = {
        'n_raw': int(len(X)), 'n_overlap_with_primary': int(overlap.sum()),
        'n_internal_duplicates': int(dup.sum()), 'n_independent': int(keep.sum()),
        'class_counts': {FAULT_NAMES[i]: int(v)
                         for i, v in enumerate(np.bincount(y[keep], minlength=6))},
        'code_mapping': {str(k): v for k, v in EXT_CODE.items()},
        'sha256_gas_matrix': hashlib.sha256(np.round(X[keep][:, :5], 8).tobytes()).hexdigest(),
    }
    return X[keep], y[keep], meta


def verify_external_mapping(primary_X, primary_y):
    """Agreement of the external labels with the primary labels on overlapping rows."""
    df = pd.read_excel(os.path.join(DATA, 'external_504.xlsx'))
    X = df[GASES].values.astype(float)
    y = np.array([IDX[EXT_CODE[int(c)]] for c in df.iloc[:, -1].astype(int).values])
    key = gas_key(X)
    prim = {}
    for k, lab in zip(gas_key(primary_X), primary_y):
        prim.setdefault(k, set()).add(int(lab))
    agree = dis = 0
    examples = []
    for k, lab in zip(key, y):
        if k in prim:
            if int(lab) in prim[k]:
                agree += 1
            else:
                dis += 1
                examples.append({'key': k, 'external': FAULT_NAMES[int(lab)],
                                 'primary': [FAULT_NAMES[i] for i in sorted(prim[k])]})
    # also report what the OLD (wrong) mapping would have produced
    old = np.array([int(c) - 1 for c in df.iloc[:, -1].astype(int).values])
    agree_old = sum(1 for k, lab in zip(key, old) if k in prim and int(lab) in prim[k])
    return {'n_overlap': agree + dis, 'agree': agree, 'disagree': dis,
            'disagreements': examples,
            'agree_under_old_mapping': int(agree_old)}


# ---------- feature set builders ----------
def textbook_ratios():
    out = []
    for name, (a, b) in STANDARD_RATIOS.items():
        c = canonical_key(f'{a}/{b}')
        r = reciprocal_key(c)
        k = min(c, r)
        if k not in out:
            out.append(k)
    return out


def llm_votes(parsed):
    from collections import Counter
    votes = Counter()
    for m, d in parsed.items():
        for rep, v in d['S1']['details'].items():
            for c in v['canonical_forms']:
                r = reciprocal_key(c)
                votes[c if (r is None or c <= r) else r] += 1
    return votes


def canonical_key_set():
    s = []
    for r in CANONICAL_RATIOS:
        c = canonical_key(r)
        s.append((c, reciprocal_key(c)))
    return s


def is_canonical(c):
    return any(c == a or c == b for a, b in canonical_key_set())


def is_textbook(c, tb):
    return any(c == t or c == reciprocal_key(t) for t in tb)


def consensus_ranking(parsed):
    """Mean rank per canonical feature over all models and replicates."""
    acc = {}
    for m, d in parsed.items():
        for rep, v in d['S3']['details'].items():
            for it in v['ranking']:
                acc.setdefault(it['feature'], []).append(it['rank'])
    merged = [{'feature': f, 'rank': float(np.mean(v))} for f, v in acc.items()]
    merged.sort(key=lambda x: x['rank'])
    for i, m in enumerate(merged, 1):
        m['rank'] = i
    return merged


def build_matrix(X, feats):
    """Column matrix for a feature list (raw gas names or canonical ratio strings)."""
    rf = [f for f in feats if f not in GASES]
    M = build_feature_matrix(X, rf) if rf else np.zeros((len(X), 0))
    cols = [X[:, GASES.index(g)] for g in GASES] + [M[:, j] for j in range(M.shape[1])]
    return np.column_stack(cols)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    Xp, yp, mp = load_primary_clean(return_meta=True)
    Xe, ye, me = load_external_clean(Xp)
    vm = verify_external_mapping(Xp, yp)
    print('primary:', Xp.shape, mp['class_counts'])
    print('  removed:', mp['n_removed'], 'duplicate groups:', mp['n_duplicate_groups'],
          'conflicts:', mp['conflicting_groups'])
    print('external:', Xe.shape, me['class_counts'])
    print('  overlap with primary:', me['n_overlap_with_primary'],
          'independent:', me['n_independent'])
    print('mapping verification:', {k: vm[k] for k in
                                    ('n_overlap', 'agree', 'disagree', 'agree_under_old_mapping')})
    print('  disagreements:', vm['disagreements'])
    print('textbook ratios:', textbook_ratios())
    json.dump({'primary': mp, 'external': me, 'mapping_check': vm},
              open(os.path.join(OUT, 'data_manifest.json'), 'w', encoding='utf-8'), indent=1)
    print('wrote results_rev/data_manifest.json')
