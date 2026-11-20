"""Validation layer: parse, physically validate, canonicalize and audit all LLM outputs.

Implements the manuscript's "physically-constrained validation protocol":
  1. JSON schema conformance
  2. Grammar check of ratio expressions (linear sums of the five gases in / form)
  3. Canonicalization (order- and sign-insensitive) and equivalence deduplication
  4. Cross-reference against IEC 60599 / Rogers / Doernenburg ratio codes
  5. Coverage statistics per model (acceptance rate, standard coverage, novelty)
"""
import json, os, sys, io, collections
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import (CANONICAL_RATIOS, CANONICAL_FEATURES, GASES, STANDARD_RATIOS,
                        parse_expression, canonical_key, reciprocal_key, is_equivalent,
                        RESULTS_DIR, LOGS_DIR)
from llm_client import LOG_PATH, MODEL_ORDER, MODELS

RAW_OUT = os.path.join(RESULTS_DIR, 'llm_parsed.json')


def load_latest_parsed():
    """For each (model, stage, rep) keep the last successful parseable record."""
    from llm_client import extract_json
    best = {}
    if not os.path.exists(LOG_PATH):
        return best
    for line in open(LOG_PATH, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('status') != 'ok':
            continue
        parsed = extract_json(r.get('response', ''))
        if parsed is None:
            continue
        key = (r['model_key'], r['stage'], int(r.get('rep') or 0))
        best[key] = {'parsed': parsed, 'usage': r.get('usage', {}),
                     'latency_s': r.get('latency_s'),
                     'temperature': r.get('temperature'), 'max_tokens': r.get('max_tokens'),
                     'model_id': r.get('model_id'), 'developer': r.get('developer'),
                     'ts': r.get('ts')}
    return best


def canon_display(canon):
    """Canonical key 'A+B/C' -> display 'A+B/C' (same)."""
    return canon


def validate_s1(parsed):
    """Validate stage-1 ratio proposals. Returns dict with accepted/rejected lists."""
    out = {'proposed': [], 'accepted': [], 'rejected': [], 'canonical_forms': []}
    ratios = parsed.get('ratios') if isinstance(parsed, dict) else None
    if not isinstance(ratios, list):
        return out
    for item in ratios:
        if not isinstance(item, dict):
            continue
        expr = item.get('expression') or item.get('ratio') or ''
        rec = {'expression': expr, 'category': item.get('category', ''),
               'basis': item.get('basis', ''), 'discriminates': item.get('discriminates', [])}
        out['proposed'].append(rec)
        canon = canonical_key(expr)
        if canon is None:
            rec['status'] = 'rejected_grammar'
            out['rejected'].append(rec)
        else:
            rec['status'] = 'accepted'
            rec['canonical'] = canon
            match = None
            for r in CANONICAL_RATIOS:
                if is_equivalent(canon, canonical_key(r)):
                    match = r
                    break
            rec['matches_canonical'] = match
            out['accepted'].append(rec)
            out['canonical_forms'].append(canon)
    return out


def canonical_set_compare(canon_list):
    """Compare accepted canonical forms against the canonical 14 (reciprocal-aware)."""
    hits, novel = [], []
    seen = set()
    for c in canon_list:
        if c in seen or (reciprocal_key(c) in seen):
            continue  # reciprocal duplicates collapse
        seen.add(c)
        matched = None
        for r in CANONICAL_RATIOS:
            cr = canonical_key(r)
            if is_equivalent(c, cr):
                matched = r
                break
        if matched:
            hits.append(matched)
        else:
            novel.append(c)
    return sorted(set(hits)), sorted(set(novel))


def standard_coverage(canon_list):
    """Check which standard (IEC/Rogers/Doernenburg) ratio codes are recovered."""
    covered = []
    for name, (a, b) in STANDARD_RATIOS.items():
        c1 = canonical_key(f'{a}/{b}')
        c2 = canonical_key(f'{b}/{a}')
        if any(is_equivalent(c, c1) or is_equivalent(c, c2) for c in canon_list):
            covered.append(name)
    return covered


def validate_s2(parsed):
    out = {'retain': [], 'remove': [], 'pairs': [], 'valid': False}
    if not isinstance(parsed, dict):
        return out
    feats = set(CANONICAL_RATIOS)
    retain = [f for f in parsed.get('recommended_retain', []) if f in feats]
    remove = [f for f in parsed.get('recommended_remove', []) if f in feats]
    pairs = parsed.get('high_correlation_pairs', [])
    ok_pairs = []
    if isinstance(pairs, list):
        for p in pairs:
            if isinstance(p, dict) and p.get('a') in feats and p.get('b') in feats:
                ok_pairs.append(p)
    out.update({'retain': retain, 'remove': remove, 'pairs': ok_pairs,
                'valid': bool(retain or remove or ok_pairs)})
    return out


def validate_s3(parsed):
    """Validate the 19-feature ranking; repair duplicates/missing entries."""
    out = {'ranking': [], 'valid': False, 'n_unique': 0, 'repaired': False}
    if not isinstance(parsed, dict):
        return out
    ranking = parsed.get('ranking')
    if not isinstance(ranking, list):
        return out
    seen = set()
    clean = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        f = item.get('feature')
        r = item.get('rank')
        if f in CANONICAL_FEATURES and f not in seen:
            try:
                r = int(r)
            except Exception:
                r = len(clean) + 1
            clean.append({'rank': r, 'feature': f, 'rationale': item.get('rationale', '')})
            seen.add(f)
    missing = [f for f in CANONICAL_FEATURES if f not in seen]
    if missing:
        out['repaired'] = True
        base = max([c['rank'] for c in clean], default=0)
        for i, f in enumerate(missing, 1):
            clean.append({'rank': base + i, 'feature': f, 'rationale': '[not ranked by model]'})
    # renormalize ranks to 1..19 preserving order
    clean.sort(key=lambda c: c['rank'])
    for i, c in enumerate(clean, 1):
        c['rank'] = i
    out['ranking'] = clean
    out['n_unique'] = len(seen)
    out['valid'] = len(seen) >= 5
    return out


def validate_s4(parsed):
    out = {'valid': False}
    if not isinstance(parsed, dict):
        return out
    for k in ('n_estimators', 'max_depth', 'min_samples_leaf'):
        v = parsed.get(k)
        if isinstance(v, dict):
            out[k] = {kk: v.get(kk) for kk in ('min', 'max', 'recommended')}
        elif isinstance(v, (int, float)):
            out[k] = {'recommended': v}
    out['class_weight'] = parsed.get('class_weight')
    out['valid'] = any(k in out for k in ('n_estimators', 'max_depth', 'min_samples_leaf'))
    return out


def main():
    best = load_latest_parsed()
    per_model = collections.defaultdict(lambda: {'S1': {}, 'S2': {}, 'S3': {}, 'S4': {}})
    for (model, stage, rep), rec in sorted(best.items()):
        if stage == 'S1':
            per_model[model]['S1'][rep] = validate_s1(rec['parsed'])
        elif stage == 'S2':
            per_model[model]['S2'][rep] = validate_s2(rec['parsed'])
        elif stage == 'S3':
            per_model[model]['S3'][rep] = validate_s3(rec['parsed'])
        elif stage == 'S4':
            per_model[model]['S4'][rep] = validate_s4(rec['parsed'])

    report = {}
    print(f"{'model':<28}{'S1 prop/acc':<14}{'canon hit':<11}{'novel':<7}{'std cov':<9}"
          f"{'S2 valid':<10}{'S3 valid':<10}{'S4 valid':<10}")
    for model in MODEL_ORDER:
        d = per_model.get(model)
        if not d:
            print(f'{model:<28} (no data)')
            continue
        s1 = d['S1']
        n_prop = sum(len(v['proposed']) for v in s1.values())
        n_acc = sum(len(v['accepted']) for v in s1.values())
        canon_hits = []
        novel = []
        std_cov = set()
        for v in s1.values():
            h, nv = canonical_set_compare(v['canonical_forms'])
            canon_hits.append(len(h))
            novel.extend(nv)
            std_cov.update(standard_coverage(v['canonical_forms']))
        n_s2 = sum(1 for v in d['S2'].values() if v['valid'])
        n_s3 = sum(1 for v in d['S3'].values() if v['valid'])
        n_s4 = sum(1 for v in d['S4'].values() if v['valid'])
        rep = {
            'S1': {'proposed': n_prop, 'accepted': n_acc,
                   'acceptance_rate': round(n_acc / n_prop, 3) if n_prop else 0,
                   'canonical_hits_per_rep': canon_hits,
                   'novel_canonical_forms': sorted(set(novel)),
                   'standard_coverage': sorted(std_cov),
                   'unique_after_fold': None,
                   'reciprocal_folded': None,
                   'details': d['S1']},
            'S2': {'n_valid_reps': n_s2, 'details': d['S2']},
            'S3': {'n_valid_reps': n_s3, 'details': d['S3']},
            'S4': {'n_valid_reps': n_s4, 'details': d['S4']},
        }
        # duplication statistics across all accepted expressions
        all_forms = []
        for v in d['S1'].values():
            all_forms.extend(v['canonical_forms'])
        uniq, seen = [], set()
        for c in all_forms:
            r = reciprocal_key(c)
            if c in seen or (r is not None and r in seen):
                continue
            seen.add(c)
            uniq.append(c)
        rep['S1']['unique_after_fold'] = len(uniq)
        rep['S1']['reciprocal_folded'] = len(all_forms) - len(uniq)
        report[model] = rep
        print(f"{model:<28}{n_prop:>4}/{n_acc:<9}{str(canon_hits):<11}{len(set(novel)):<7}"
              f"{len(std_cov):<9}{n_s2}/{sum(1 for _ in d['S2'].values()):<8}"
              f"{n_s3}/{sum(1 for _ in d['S3'].values()):<8}{n_s4}/{sum(1 for _ in d['S4'].values()):<8}")

    with open(RAW_OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print('\nsaved:', RAW_OUT)


if __name__ == '__main__':
    main()
