"""Run the four-stage LLM feature-engineering protocol across all base models.

Stage 1: feature generation (open-ended ratio proposals)
Stage 2: redundancy analysis over the 14 canonical ratios
Stage 3: ranking of the 19 canonical features
Stage 4: RF hyperparameter guidance
Each stage is repeated K times per model for self-consistency analysis.
"""
import json, os, sys, io, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import MODELS, MODEL_ORDER, call_model, extract_json, LOG_PATH
from exp_common import CANONICAL_RATIOS, CANONICAL_FEATURES, GASES, LOGS_DIR

K_REPEATS = 3

SYSTEM_1 = (
    "You are an expert in dissolved gas analysis (DGA) for power transformer fault diagnosis. "
    "You have deep knowledge of IEC 60599, IEEE C57.104, Duval Triangle, and Rogers ratio methods. "
    "Your task is to identify physically meaningful gas ratio features from five primary DGA gases: "
    "H2, CH4, C2H6, C2H4, and C2H2."
)
USER_1 = (
    "Given the five primary DGA gases (H2, CH4, C2H6, C2H4, C2H2), propose all physically "
    "meaningful gas ratio features that can distinguish between six fault types: Normal, "
    "Low-Temperature Overheating (LTO), High-Temperature Overheating (HTO), Partial Discharge (PD), "
    "Low-Energy Discharge (LED), and High-Energy Discharge (HED). For each ratio, explain: "
    "(1) the physical/chemical basis, (2) which fault types it discriminates, "
    "(3) expected behavior under each fault condition.\n\n"
    "Output a single JSON object with the key \"ratios\": a list of at most 18 objects, each having "
    "keys \"expression\" (a ratio built ONLY from H2, CH4, C2H6, C2H4, C2H2 using + and / with "
    "parentheses, e.g. \"H2/(CH4+C2H6)\"; sums of gases are allowed in numerator and denominator), "
    "\"category\" (one of: H2-based, CH4-based, C2H6-based, C2H4-based, C2H2-based, composite), "
    "\"basis\" (physical/chemical basis, at most 15 words), "
    "\"discriminates\" (list of fault-type names it separates, abbreviations only). "
    "Keep every \"basis\" under 15 words; write the JSON compactly (no line-by-line formatting). "
    "Respond with the JSON object only, no text before or after."
)

SYSTEM_2 = (
    "You are a data scientist specializing in feature engineering for fault diagnosis. "
    "Analyze the following gas ratio features for informational redundancy."
)
USER_2_TMPL = (
    "Given these 14 gas ratio features derived from DGA data, analyze pairwise correlations "
    "and identify: (1) which ratios are highly correlated (r > 0.85) and can be merged/removed, "
    "(2) which ratios provide unique diagnostic information, (3) the recommended minimal "
    "feature subset that preserves 95% of diagnostic information.\n"
    "Feature list: {features}.\n\n"
    "Output a single JSON object with keys: \"high_correlation_pairs\" (list of objects with keys "
    "\"a\", \"b\", \"estimated_r\"), \"recommended_retain\" (list of feature names), "
    "\"recommended_remove\" (list of feature names), \"reasoning\" (at most 40 words). "
    "Use the exact feature names from the list. Respond with the JSON object only."
)

SYSTEM_3 = (
    "You are an expert in transformer diagnostics. Rank the following DGA features by "
    "their expected diagnostic value for distinguishing 6 fault types."
)
USER_3_TMPL = (
    "Rank these 19 DGA features (5 raw gases + 14 ratios) by diagnostic importance for "
    "classifying Normal, LTO, HTO, PD, LED, HED faults. For each feature, provide: "
    "(1) rank, (2) reasoning based on DGA physics, (3) which fault pairs it best discriminates. "
    "Consider that C2H2 is the definitive arcing indicator, C2H4 indicates high-temperature "
    "thermal stress, and H2 indicates discharge activity.\n"
    "Feature list: {features}.\n\n"
    "Output a single JSON object with the key \"ranking\": a list of 19 objects, each with keys "
    "\"rank\" (1 = most important, integer 1-19), \"feature\" (exact name from the list), "
    "\"rationale\" (at most 20 words). Include every feature exactly once. "
    "Respond with the JSON object only."
)

SYSTEM_4 = (
    "You are a machine learning engineer tuning classifiers for engineering diagnostics."
)
USER_4 = (
    "For a Random Forest classifier on a 6-class DGA dataset with 550 samples and 19 features, "
    "recommend optimal hyperparameter ranges for: n_estimators, max_depth, min_samples_leaf. "
    "Consider the moderate dataset size and class imbalance (HED: 45 samples, Normal: 140 samples).\n\n"
    "Output a single JSON object with keys: \"n_estimators\" (object with \"min\", \"max\", "
    "\"recommended\"), \"max_depth\" (object with \"min\", \"max\", \"recommended\"), "
    "\"min_samples_leaf\" (object with \"min\", \"max\", \"recommended\"), "
    "\"class_weight\" (string), \"rationale\" (at most 30 words). "
    "Respond with the JSON object only."
)

STAGES = {
    'S1': {'system': SYSTEM_1, 'user': USER_1, 'temperature': 0.2, 'max_tokens': 16384},
    'S2': {'system': SYSTEM_2, 'user': USER_2_TMPL.format(features=', '.join(CANONICAL_RATIOS)),
           'temperature': 0.1, 'max_tokens': 8192},
    'S3': {'system': SYSTEM_3, 'user': USER_3_TMPL.format(features=', '.join(CANONICAL_FEATURES)),
           'temperature': 0.2, 'max_tokens': 8192},
    'S4': {'system': SYSTEM_4, 'user': USER_4, 'temperature': 0.1, 'max_tokens': 4096},
}

# Models with very long internal reasoning traces need larger base budgets; this is
# documented in the manuscript as an engineering finding about model suitability.
BUDGET_OVERRIDE = {
    'longcat-2.5-preview-free': {'S1': 32768, 'S2': 32768, 'S3': 32768, 'S4': 32768},
    'hy4-preview': {'S1': 32768, 'S2': 32768, 'S3': 32768, 'S4': 16384},
}


def base_budget(model_key, stage):
    return BUDGET_OVERRIDE.get(model_key, {}).get(stage, STAGES[stage]['max_tokens'])


def load_done():
    """Return set of (model, stage, rep) already logged successfully."""
    done = set()
    if os.path.exists(LOG_PATH):
        for line in open(LOG_PATH, encoding='utf-8'):
            try:
                r = json.loads(line)
                if r.get('status') == 'ok':
                    done.add((r['model_key'], r['stage'], r.get('rep')))
            except Exception:
                pass
    return done


def one_call(args):
    model_key, stage, rep = args
    cfg = STAGES[stage]
    budget = base_budget(model_key, stage)
    rec = call_model(model_key, cfg['system'], cfg['user'], stage, rep,
                     cfg['temperature'], budget)
    ok = rec.get('status') == 'ok'
    parsed = extract_json(rec.get('response', '')) if ok else None
    extended = False
    # reasoning-token truncation fallback: double the budget up to twice (cap 65536)
    for _ in range(2):
        if parsed or not ok:
            break
        budget *= 2
        if budget > 65536:
            break
        extended = True
        rec = call_model(model_key, cfg['system'], cfg['user'], stage, rep,
                         cfg['temperature'], budget, extra={'extended': True})
        ok = rec.get('status') == 'ok'
        parsed = extract_json(rec.get('response', '')) if ok else None
    return model_key, stage, rep, ok, bool(parsed), rec.get('latency_s'), \
        (rec.get('usage') or {}).get('total_tokens'), extended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODEL_ORDER))
    ap.add_argument('--stages', default='S1,S2,S3,S4')
    ap.add_argument('--reps', type=int, default=K_REPEATS)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    models = args.models.split(',')
    stages = args.stages.split(',')
    done = load_done()
    tasks = [(m, s, r) for m in models for s in stages for r in range(1, args.reps + 1)
             if (m, s, r) not in done]
    print(f'{len(tasks)} calls to run ({len(done)} already done)')
    t0 = time.time()
    okc = parseok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one_call, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            m, s, r, ok, parsed, lat, tok, ext = fut.result()
            okc += ok
            parseok += parsed
            print(f'[{i}/{len(tasks)}] {m} {s} rep{r}: ok={ok} json={parsed} '
                  f'{lat}s {tok}tok ext={ext}', flush=True)
    print(f'DONE: {okc}/{len(tasks)} ok, {parseok} parsed, {time.time()-t0:.0f}s')
    print('log:', LOG_PATH)


if __name__ == '__main__':
    main()
