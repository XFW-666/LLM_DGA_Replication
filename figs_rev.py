"""Figures 3-8 for the second revision (layout-hardened version)."""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'llm'))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import figstyle  # noqa: F401
from figstyle import FAULT_NAMES, FAULT_COLORS, set_inflation

from data2 import load_primary_clean, build_matrix, CANONICAL_RATIOS, GASES

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
RES = os.path.join(BASE, 'results_rev')
FIG = os.path.join(BASE, 'figures')
set_inflation(1.0)

D = json.load(open(os.path.join(RES, 'main.json'), encoding='utf-8'))
E = json.load(open(os.path.join(RES, 'external.json'), encoding='utf-8'))
M = json.load(open(os.path.join(RES, 'models.json'), encoding='utf-8'))
V = json.load(open(os.path.join(RES, 'duval.json'), encoding='utf-8'))
W = json.load(open(os.path.join(RES, 'weights.json'), encoding='utf-8'))
B = json.load(open(os.path.join(RES, 'baselines.json'), encoding='utf-8'))
LD = json.load(open(os.path.join(RES, 'lowdata.json'), encoding='utf-8'))
TR = json.load(open(os.path.join(RES, 'lowdata_trend.json'), encoding='utf-8'))
EV = json.load(open(os.path.join(RES, 'lowdata_evidence.json'), encoding='utf-8'))
PARSED = json.load(open(os.path.join(BASE, 'results', 'llm_parsed.json'), encoding='utf-8'))

X, y = load_primary_clean()
SET_ORDER = ['P', 'S', 'C', 'CS', 'L', 'Lp', 'CL', 'Lk']
SET_LABEL = {'P': 'raw gases', 'S': 'textbook (no LLM)', 'C': 'canonical 14',
             'CS': 'canonical+textbook', 'L': 'LLM endorsed', 'Lp': 'LLM-only',
             'CL': 'canonical+LLM', 'Lk': 'LLM top-17'}
SET_COLOR = {'P': '#64748B', 'S': '#0EA5E9', 'C': '#F59E0B', 'CS': '#B45309',
             'L': '#10B981', 'Lp': '#047857', 'CL': '#065F46', 'Lk': '#7C3AED'}
SHORT = {'mimo-v2.6-flash': 'MiMo-2.6F', 'mimo-v2.5': 'MiMo-2.5',
         'qwen3.8-flash': 'Qwen3.8F', 'deepseek-v4.1-flash': 'DS-4.1F',
         'deepseek-v4-flash': 'DS-4F', 'glm-5.3-flash': 'GLM-5.3F',
         'hy3': 'HY3', 'longcat-2.5-preview-free': 'LongCat'}
MODELS = list(PARSED.keys())


VECDIR = os.path.join(BASE, 'figures', 'vector')
os.makedirs(VECDIR, exist_ok=True)
# TrueType embedding so the vector files satisfy "fonts must be embedded"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=900, bbox_inches='tight', pad_inches=0.05)
    # journal naming: Fig<n>.pdf, vector with embedded fonts (EPS too large)
    m = re.match(r'fig(\d+)_', name)
    if m:
        for ext in ('pdf',):
            fig.savefig(os.path.join(VECDIR, 'Fig%s.%s' % (m.group(1), ext)),
                        bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print('saved', name)


def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    ax = axes[0, 0]
    for ci in range(6):
        vals = np.log10(X[y == ci, 0] + 1)
        xs = np.linspace(-0.2, 4.0, 200)
        bw = max(vals.std(), 0.15)
        kde = np.exp(-0.5 * ((xs - vals.mean()) / bw) ** 2)
        ax.fill_between(xs, ci * 0.9, ci * 0.9 + kde / kde.max() * 0.72,
                        color=FAULT_COLORS[ci], alpha=0.7, lw=0)
        ax.text(4.03, ci * 0.9 + 0.16, FAULT_NAMES[ci], fontsize=7.6, va='center',
                color=FAULT_COLORS[ci], fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-0.2, 4.65)
    ax.set_xlabel('log$_{10}$ H$_2$ (ppm)')
    ax.set_title('(a) H$_2$ distribution by class', loc='left', fontweight='bold')

    ax = axes[0, 1]
    Z = (X[:, :5] - X[:, :5].mean(0)) / X[:, :5].std(0)
    Z = np.clip(Z, -3.5, 7.0)
    for ci in range(6):
        idx = np.where(y == ci)[0][:35]
        for i in idx:
            ax.plot(range(5), Z[i], color=FAULT_COLORS[ci], alpha=0.18, lw=0.55)
    for ci in range(6):
        ax.plot(range(5), np.median(Z[y == ci], axis=0), color=FAULT_COLORS[ci], lw=2.2,
                marker='o', ms=3.4, label=FAULT_NAMES[ci])
    ax.set_xticks(range(5))
    ax.set_xticklabels(GASES, fontsize=8)
    ax.set_ylabel('standardised concentration (clipped at 7)')
    ax.set_title('(b) Parallel coordinates', loc='left', fontweight='bold')
    ax.legend(fontsize=6.4, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              frameon=False, handlelength=1.2, columnspacing=0.9)

    ax = axes[1, 0]
    order = ['PD', 'T1', 'T2', 'T3', 'D1', 'D2', 'DT', 'ND']
    zones = [z for z in order if z in V['zones']]
    mat = np.zeros((6, len(zones)))
    for i, lb in enumerate(FAULT_NAMES):
        for j, z in enumerate(zones):
            mat[i, j] = V['zone_table'].get(f'{lb}|{z}', 0)
    im = ax.imshow(mat, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels(zones, fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(FAULT_NAMES, fontsize=8)
    for i in range(6):
        for j in range(len(zones)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha='center', va='center', fontsize=6.8,
                        color='white' if mat[i, j] > mat.max() * 0.6 else '#1F2937')
    ax.set_title('(c) Label vs Duval Triangle-1 zone', loc='left', fontweight='bold')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1, 1]
    Xm = build_matrix(X, list(CANONICAL_RATIOS))
    C = np.corrcoef(Xm.T)
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('(d) Correlation of the 19 canonical channels', loc='left',
                 fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, 'fig3_dataset_features.png')


def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    ax = axes[0, 0]
    null = D['random_selection_null']
    ax.axvspan(null['p2.5'], null['p97.5'], color='#E2E8F0', alpha=0.9, lw=0,
               label='random-selection null (95%)')
    for i, k in enumerate(SET_ORDER):
        r = D['results'][k]
        ax.errorbar(r['accuracy_mean'], i,
                    xerr=[[r['accuracy_mean'] - r['ci95'][0]],
                          [r['ci95'][1] - r['accuracy_mean']]],
                    fmt='o', color=SET_COLOR[k], ms=6, lw=1.6, capsize=3)
        ax.text(r['ci95'][1] + 0.2, i, f"{r['accuracy_mean']:.2f}", va='center', fontsize=7.4)
    ax.set_yticks(range(len(SET_ORDER)))
    ax.set_yticklabels([SET_LABEL[k] for k in SET_ORDER], fontsize=8)
    ax.set_xlabel('out-of-fold accuracy (%)')
    ax.set_xlim(77.0, 91.0)
    ax.set_xticks([78, 80, 82, 84, 86, 88])
    ax.set_title('(a) Accuracy with 95% CI', loc='left', fontweight='bold')
    # the widest interval plus its value label ends near 86.6, so the axis is extended to
    # 91 and the legend lives in that dedicated right-hand margin: no data under the box
    ax.legend(fontsize=6.8, loc='upper right', frameon=True, framealpha=0.95)
    ax.invert_yaxis()

    ax = axes[0, 1]
    for k in SET_ORDER:
        ax.scatter(D['results'][k]['macro_f1_mean'],
                   D['results'][k]['balanced_accuracy_mean'],
                   s=70, color=SET_COLOR[k], edgecolor='white', zorder=3,
                   label=SET_LABEL[k])
    ax.set_xlabel('macro-F1 (%)')
    ax.set_ylabel('balanced accuracy (%)')
    ax.set_xlim(75.6, 80.2)
    ax.set_ylim(76.2, 80.2)
    ax.set_title('(b) Imbalance-aware metrics', loc='left', fontweight='bold')
    ax.legend(fontsize=6.4, loc='lower right', ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=0.8)

    ax = axes[1, 0]
    rec = np.array([D['results'][k]['per_class_recall_mean'] for k in SET_ORDER])
    im = ax.imshow(rec, cmap='viridis', aspect='auto', vmin=25, vmax=95)
    ax.set_xticks(range(6))
    ax.set_xticklabels(FAULT_NAMES, fontsize=8)
    ax.set_yticks(range(len(SET_ORDER)))
    ax.set_yticklabels([SET_LABEL[k] for k in SET_ORDER], fontsize=8)
    for i in range(rec.shape[0]):
        for j in range(6):
            ax.text(j, i, f'{rec[i, j]:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if rec[i, j] < 70 else '#111827')
    ax.set_title('(c) Per-class recall (%)', loc='left', fontweight='bold')
    ax.grid(False)

    ax = axes[1, 1]
    tests = {t['condition']: t for t in D['contrasts']['vs_raw']}
    order = [k for k in SET_ORDER if k in tests]
    deltas = [tests[k]['delta'] for k in order]
    ps = [tests[k]['nb']['p'] for k in order]
    ax.barh(range(len(order)), deltas, color=[SET_COLOR[k] for k in order], alpha=0.85)
    for i, (dlt, pv) in enumerate(zip(deltas, ps)):
        ax.text(dlt + 0.06, i, f'p={pv:.2f}', va='center', fontsize=7)
    ax.axvline(0, color='#334155', lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SET_LABEL[k] for k in order], fontsize=8)
    ax.set_xlim(0, 3.6)
    ax.set_xlabel('accuracy gain over raw gases (points)')
    ax.set_title('(d) Gain and corrected p-value', loc='left', fontweight='bold')
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, 'fig4_featuresets.png')


def fig5():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    ax = axes[0, 0]
    null = D['random_selection_null']
    draws = np.array(null.get('draws', []))
    if draws.size:
        ax.hist(draws, bins=18, color='#CBD5E1', edgecolor='white')
    ax.axvline(null['llm_top17'], color='#7C3AED', lw=2.2,
               label=f"LLM top-17 ({null['llm_top17']:.2f}%)")
    ax.axvline(null['mean'], color='#334155', lw=1.2, ls='--', label='null mean')
    ax.set_xlabel('accuracy of a random 17-channel subset (%)')
    ax.set_ylabel('draws')
    ax.set_title('(a) Random-selection null (100 subsets)', loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', frameon=False)

    ax = axes[0, 1]
    dropped = np.zeros((len(MODELS), len(CANONICAL_RATIOS)))
    for i, m in enumerate(MODELS):
        top = set(M['results'].get(f'{m}|K', {}).get('top17', []))
        for j, r in enumerate(CANONICAL_RATIOS):
            dropped[i, j] = 0 if r in top else 1
    im = ax.imshow(dropped, cmap=ListedColormap(['#F1F5F9', '#DC2626']), aspect='auto',
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_xticks(range(len(CANONICAL_RATIOS)))
    ax.set_xticklabels(CANONICAL_RATIOS, rotation=90, fontsize=5.4)
    ax.set_title('(b) Channels dropped by each model\u2019s ranking', loc='left',
                 fontweight='bold')
    ax.grid(False)
    ax.text(1.02, 0.5, 'red = dropped', transform=ax.transAxes, rotation=90,
            va='center', fontsize=6.6, color='#DC2626')

    ax = axes[1, 0]
    wr = W['results']
    learners = ['RF', 'LR', 'kNN']
    for i, L in enumerate(learners):
        r = wr[L]
        ax.plot([r['null_p2.5'] - r['uniform'], r['null_p97.5'] - r['uniform']], [i, i],
                color='#CBD5E1', lw=7, solid_capstyle='round',
                label='random-weight 95% range' if i == 0 else None)
        ax.plot(0, i, 'o', color='#334155', ms=6.5, label='uniform (reference)' if i == 0 else None)
        ax.plot(r['delta'], i, 'o', color='#DC2626', ms=6.5,
                label='LLM weights' if i == 0 else None)
        ax.text(r['delta'] + 0.04, i + 0.16, f"{r['delta']:+.2f}", fontsize=6.8,
                color='#B91C1C')
        ax.text(0.04, i - 0.22, f"pct {r['llm_percentile']:.0f}", fontsize=6.4,
                color='#475569')
    ax.axvline(0, color='#94A3B8', lw=0.8)
    ax.set_yticks(range(len(learners)))
    ax.set_yticklabels(learners, fontsize=9)
    ax.set_xlabel('accuracy change vs uniform weights (points)')
    ax.set_title('(c) Weight experiment', loc='left', fontweight='bold')
    # headroom above the top row so the legend cannot sit on the kNN annotations
    ax.set_ylim(-0.52, 3.05)
    ax.legend(fontsize=6.4, loc='upper left', frameon=False, ncol=3)

    ax = axes[1, 1]
    rows = [['Learner', 'uniform', 'LLM weights', '\u0394', 'percentile']]
    for L in learners:
        r = wr[L]
        rows.append([L, f"{r['uniform']:.2f}", f"{r['llm']:.2f}", f"{r['delta']:+.2f}",
                     f"{r['llm_percentile']:.0f}"])
    ax.axis('off')
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.6)
    tb.scale(0.95, 1.4)
    ax.set_title('(d) Effect size and rank of the LLM weights', loc='left',
                 fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig5_selection.png')


def fig6():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    ax = axes[0, 0]
    rank_mat = np.zeros((len(MODELS), len(CANONICAL_RATIOS)))
    for i, m in enumerate(MODELS):
        acc = {}
        for rep, v in PARSED[m]['S3']['details'].items():
            for it in v['ranking']:
                acc.setdefault(it['feature'], []).append(it['rank'])
        for j, r in enumerate(CANONICAL_RATIOS):
            rank_mat[i, j] = np.mean(acc.get(r, [19]))
    im = ax.imshow(rank_mat, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=19)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_xticks(range(len(CANONICAL_RATIOS)))
    ax.set_xticklabels(CANONICAL_RATIOS, rotation=90, fontsize=5.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('mean rank', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title('(a) Ranking of the canonical channels', loc='left', fontweight='bold')
    ax.grid(False)

    ax = axes[0, 1]
    sp = M['agreement']['spearman_pairwise']
    mat = np.eye(len(MODELS))
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if i != j:
                key = f'{a}~{b}' if f'{a}~{b}' in sp else f'{b}~{a}'
                mat[i, j] = sp[key]['rho']
    im = ax.imshow(mat, cmap='PuBu', vmin=0, vmax=1)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=45, fontsize=7)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if i != j:
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6.2,
                        color='white' if mat[i, j] > 0.66 else '#1F2937')
    ax.set_title('(b) Pairwise rank agreement (Spearman)', loc='left', fontweight='bold')
    ax.grid(False)

    ax = axes[1, 0]
    std6 = ['CH4/H2', 'C2H2/C2H4', 'C2H4/C2H6', 'C2H6/CH4', 'C2H2/CH4', 'C2H2/C2H6']
    from data2 import reciprocal_key
    from exp_common import canonical_key
    endorse = np.zeros((len(MODELS), len(std6)))
    for i, m in enumerate(MODELS):
        forms = set()
        for rep, v in PARSED[m]['S1']['details'].items():
            forms |= set(v['canonical_forms'])
        for j, s in enumerate(std6):
            c = canonical_key(s)
            endorse[i, j] = 1 if (c in forms or reciprocal_key(c) in forms) else 0
    ax.imshow(endorse, cmap=ListedColormap(['#F8FAFC', '#059669']), aspect='auto',
              vmin=0, vmax=1)
    ax.set_xticks(range(len(std6)))
    ax.set_xticklabels(std6, fontsize=7, rotation=25)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_title('(c) Endorsement of the six distinct standard ratios\n'
                 '(green = proposed by the model)', loc='left', fontweight='bold',
                 fontsize=9.5)
    ax.grid(False)

    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(MODELS)]
    for i, m in enumerate(MODELS):
        vals = [M['results'][f'{m}|{t}']['accuracy_mean'] for t in ['G', 'K', 'GK']]
        ax.plot([0, 1, 2], vals, 'o-', color=colors[i], lw=1.2, ms=4,
                label=SHORT[m])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['G: own proposals', 'K: own selection', 'GK: union'], fontsize=7.4)
    ax.set_ylabel('accuracy (%)')
    ax.set_title('(d) Per-model feature conditions', loc='left', fontweight='bold')
    ax.legend(fontsize=6.6, ncol=4, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    save(fig, 'fig6_models.png')


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    oof = np.array(D['results']['Lp']['oof_pred'][0])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, oof, labels=list(range(6)))
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100
    ax = axes[0]
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(6))
    ax.set_xticklabels(FAULT_NAMES, rotation=30, fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(FAULT_NAMES, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i, j]}\n{cmn[i, j]:.0f}%', ha='center', va='center',
                    fontsize=6.2, color='white' if cmn[i, j] > 55 else '#111827')
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('(a) Confusion matrix, LLM-only', loc='left',
                 fontweight='bold')
    ax.grid(False)

    ax = axes[1]
    xs = np.arange(6)
    for off, key, color, lab in [(-0.18, 'P', '#64748B', 'raw gases'),
                                 (0.18, 'Lp', '#047857', 'LLM-only')]:
        rec = D['results'][key]['per_class_recall_mean']
        ax.bar(xs + off, rec, width=0.34, color=color, alpha=0.85, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(FAULT_NAMES, fontsize=8)
    ax.set_ylabel('recall (%)')
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7, loc='upper left', frameon=False)
    ax.set_title('(b) Per-class recall', loc='left', fontweight='bold')
    # vertical arrow straight down onto the top of the LLM-only bar for LED
    ax.annotate('smallest class', xy=(4.18, rec[4] + 1.0), xytext=(4.18, 70),
                fontsize=6.8, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#334155',
                                shrinkA=1, shrinkB=0))
    fig.tight_layout()
    save(fig, 'fig7_errors.png')


def fig8():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    ax = axes[0, 0]
    keys = ['P', 'S', 'C', 'Lp', 'Lk']
    vals = [E['transfer'][k]['accuracy'] for k in keys]
    tests = E['transfer_tests']
    ax.barh(range(len(keys)), vals, color=[SET_COLOR[k] for k in keys], alpha=0.85)
    for i, k in enumerate(keys):
        p = tests.get(k, {}).get('mcnemar', {}).get('p')
        txt = f'{vals[i]:.1f}' + ('' if p is None else f'  (p={p:.2f})')
        ax.text(vals[i] + 0.3, i, txt, va='center', fontsize=7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([SET_LABEL[k] for k in keys], fontsize=8)
    ax.set_xlim(70, 90)
    ax.set_xlabel('transfer accuracy (%)')
    ax.set_title('(a) Train on primary, test on independent records', loc='left',
                 fontweight='bold')
    ax.invert_yaxis()

    ax = axes[0, 1]
    for i, k in enumerate(keys):
        tr = E['transfer'][k]['accuracy']
        cv = E['internal_cv'][k]['accuracy']
        ax.plot([cv, tr], [i, i], color='#CBD5E1', lw=3, solid_capstyle='round')
        ax.plot(cv, i, 'o', color='#334155', ms=5)
        ax.plot(tr, i, 'o', color=SET_COLOR[k], ms=5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([SET_LABEL[k] for k in keys], fontsize=8)
    ax.set_xlabel('accuracy (%)')
    ax.set_title('(b) Within-collection CV vs transfer', loc='left', fontweight='bold')
    ax.plot([], [], 'o', color='#334155', label='within-collection CV')
    ax.plot([], [], 'o', color='#0F172A', label='transfer')
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.16))

    ax = axes[1, 0]
    cm = np.array(E['transfer']['Lp']['confusion'])
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cmn, cmap='Purples', vmin=0, vmax=100)
    ax.set_xticks(range(6))
    ax.set_xticklabels(FAULT_NAMES, rotation=30, fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(FAULT_NAMES, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=6.6,
                    color='white' if cmn[i, j] > 55 else '#111827')
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('(c) Transfer confusion, LLM-only (counts)', loc='left',
                 fontweight='bold')
    ax.grid(False)

    ax = axes[1, 1]
    mc = E['mapping_check']
    bars = [mc['agree'], mc['disagree'], mc['agree_under_old_mapping']]
    labs = ['verified mapping\n(agrees)', 'verified mapping\n(disagrees)',
            'old mapping\n(agrees)']
    ax.bar(range(3), bars, color=['#059669', '#DC2626', '#94A3B8'], alpha=0.9)
    for i, b in enumerate(bars):
        ax.text(i, b + 2, str(b), ha='center', fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel(f'records (of {mc["n_overlap"]} overlapping)')
    ax.set_title('(d) Label-mapping verification', loc='left', fontweight='bold')
    ax.set_ylim(0, mc['n_overlap'] * 1.12)
    fig.tight_layout()
    save(fig, 'fig8_transfer.png')


def fig9():
    """Learning-curve evidence for the sample-efficiency gradient (Section 5.9).

    Two panels on an 11-inch canvas: the figure is placed at 6.5 inches in the
    manuscript, so every glyph is sized to stay above ~5.5 pt on the page.
    """
    from scipy import stats as st
    fr = np.array(EV['fractions'])
    kf = [f'{f:.2f}' for f in fr]
    gA = np.array([EV['gap_A'][k] for k in kf])
    gB = np.array([EV['gap_B'][k] for k in kf])
    pA = np.array([EV['p_A'][k] for k in kf])
    pB = np.array([EV['p_B'][k] for k in kf])
    pool = np.array([EV['gap_A_pool'][k] for k in kf])
    G = np.array(TR['gap_S'])                       # 50 folds x 8 fractions
    SE_A = np.sqrt((1.0 / G.shape[0] + 0.25) * G.var(axis=0, ddof=1))
    SE_B = np.abs(gB) / np.abs(st.t.isf(pB / 2, G.shape[0] - 1))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.25),
                             gridspec_kw={'width_ratios': [1.42, 1.0]})

    ax = axes[0]
    ax.axhline(0, color='#94A3B8', lw=1.0, ls=':')
    ax.fill_between(fr, gA - 1.96 * SE_A, gA + 1.96 * SE_A, color=SET_COLOR['S'],
                    alpha=0.16, lw=0, label='95% band (stream A)')
    ax.plot(fr, gA, '-o', color=SET_COLOR['S'], ms=6.5, lw=2.0,
            label='textbook $-$ raw, stream A')
    ax.plot(fr, gB, '--s', color='#B45309', ms=5.6, lw=1.7, mfc='white',
            label='textbook $-$ raw, stream B')
    ax.plot(fr, pool, ':^', color='#047857', ms=6.0, lw=1.7,
            label='pooled engineered $-$ raw')
    ax.set_xscale('log')
    ax.set_xticks(fr)
    ax.set_xticklabels([f'{f * 100:g}%' for f in fr], fontsize=10)
    ax.set_xlabel('fraction of the training fold retained', fontsize=11)
    ax.set_ylabel('accuracy advantage (points)', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('(a) The advantage is a function of data volume', loc='left',
                 fontweight='bold', fontsize=12)
    ax.set_ylim(-7.8, 11.0)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(axis='y', lw=0.6, alpha=0.35)
    # the legend belongs in the band-free zone below zero, not over the curves
    ax.legend(fontsize=8.6, loc='lower right', framealpha=0.95, borderpad=0.5,
              labelspacing=0.4, handlelength=2.2)
    st_txt = EV['slope_textbook']
    ax.text(0.052, -2.9,
            f"slope {st_txt['slope']:+.2f} pts/decade\n"
            f"cluster-robust p = {st_txt['p_one']:.1e}\n"
            f"permutation p = {st_txt['p_perm']:.3f}\n"
            f"negative in {EV['nneg_A']}/{10} repeats",
            fontsize=9.2, va='top', ha='left', color='#111827',
            bbox=dict(boxstyle='round,pad=0.38', fc='white', ec='#CBD5E1', lw=0.8,
                      alpha=0.96))

    ax = axes[1]
    fam = ['S', 'C', 'Lp', 'CS']
    names = {'S': 'textbook', 'C': 'canonical 14', 'Lp': 'LLM-only',
             'CS': 'canon. + textbook'}
    for j, k in enumerate(fam):
        a10, a100 = EV['fam_delta'][f'A_0.10'][k], EV['fam_delta'][f'A_1.00'][k]
        b10 = EV['fam_delta'][f'B_0.10'][k]
        y = len(fam) - 1 - j
        ax.plot([a100, a10], [y, y], '-', color=SET_COLOR[k], lw=2.0, alpha=0.85)
        ax.plot([a100], [y], 'o', ms=9.0, color='white', mec=SET_COLOR[k], mew=1.8)
        ax.plot([a10], [y], 'o', ms=10.5, color=SET_COLOR[k])
        ax.plot([b10], [y + 0.27], 's', ms=6.2, mfc='white', mec=SET_COLOR[k], mew=1.5)
        ax.text(a10 + 0.5, y, f'{a10:+.2f}', fontsize=9.2, va='center', color=SET_COLOR[k])
        ax.text(a100 - 0.5, y, f'{a100:+.2f}', fontsize=9.2, va='center', ha='right',
                color='#475569')
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels([names[k] for k in fam][::-1], fontsize=10)
    ax.set_xlabel('advantage over the raw gases (points)', fontsize=11)
    ax.axvline(0, color='#94A3B8', lw=1.0, ls=':')
    ax.set_xlim(-1.4, 10.6)
    ax.set_ylim(-2.15, 3.6)
    ax.margins(x=0)
    fig.subplots_adjust(left=0.06, right=0.985, wspace=0.32)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_title('(b) Every engineered family behaves alike', loc='left',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', lw=0.6, alpha=0.35)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ms=8.0, ls='', mfc='white',
                              mec='#475569', mew=1.6,
                              label='open circle: full sample'),
                       Line2D([], [], marker='o', ms=8.5, ls='', mfc='#475569',
                              mec='#475569', label='filled circle: 10% of training data'),
                       Line2D([], [], marker='s', ms=6.5, ls='', mfc='white',
                              mec='#475569', mew=1.4,
                              label='square: replicate stream B')],
              fontsize=8.0, loc='lower center', framealpha=0.95, borderpad=0.5,
              labelspacing=0.3, handlelength=1.2, handleheight=0.9, ncol=1,
              bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, 'fig9_lowdata.png')


if __name__ == '__main__':
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print('all figures rebuilt')
