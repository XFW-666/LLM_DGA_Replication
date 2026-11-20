# Replication data and code - second revision (v2.1.0)

Companion to: *LLM-Enhanced Feature Engineering for Power Transformer Fault Diagnosis Using
Dissolved Gas Analysis: A Leakage-Free Multi-Model Evaluation*
(International Journal of Computational Intelligence Systems, submission
f3eae009-8683-4b74-89e2-19f172f96e70).

This package contains the data, the LLM interaction log, the analysis code and every result
artefact the manuscript is generated from. The manuscript and the response letter are
submission files and are not part of this package.

## Where this archive lives

This archive is self-contained and deliberately carries no web links: identifiers and
repository URLs change when a record is revised or mirrored, and every copy of the package
should stay correct wherever it ends up. The canonical location of this archive, and the
citable identifier for the version you are reading, are given by the manuscript's Data
Availability statement and by the landing page of the repository from which you obtained it.
This build is version 2.1.0.

## Layout
Data/      raw collections as distributed, the two cleaned analysis matrices, and
           DATA_PROVENANCE.md sources plus data_manifest.json (duplicate policy, class
           counts, SHA-256 of the matrices)
Results/   main.json (feature sets, null, contrasts), models.json (per-model and
           agreement), external.json (transfer and mapping verification), duval.json,
           weights.json, baselines.json, dl.json, llm_parsed.json, llm_usage.json,
           reference_audit.json
Logs/      llm_calls.jsonl - one record per API call (prompt, response, tokens, latency,
           timestamp, model identifier)
Code/      data2.py (cleaning), eval2.py (cross-validation and tests), run_*.py
           (experiments), figs_rev.py (figures), and the LLM client plus prompts
Figures    not shipped: run Code/figs_rev.py to regenerate all nine from Results/
Source_Documents/  the two dissertations from which the data derive (see
           DATA_PROVENANCE.md)
Appendix_A_prompts.txt  the four prompt templates, verbatim

## Reproduction
Python 3.12 with numpy, pandas, scikit-learn 1.6, scipy, matplotlib, openpyxl and
torch 2.13 (CPU); seeds are fixed.

    python data2.py          # cleaning + data_manifest.json
    python run_main.py       # main comparison
    python run_external.py   # independent validation
    python run_duval.py      # label ambiguity
    python run_weights.py    # weight experiment
    python run_models.py     # per-model conditions and agreement
    python run_baselines.py  # tuned learners
    python run_dl.py         # deep baselines
    python figs_rev.py       # figures

The hosted models are not required to reproduce the evaluation: `Results/llm_parsed.json`
contains the parsed output of every call and `Logs/llm_calls.jsonl` the verbatim exchanges.

## Licence
Code: MIT (see LICENSE). Data and figures: CC BY 4.0. The underlying collections are
third-party material (see DATA_PROVENANCE.md).
