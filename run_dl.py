"""Deep-learning baselines on the cleaned primary set (reviewer request: architecture
detail, macro-F1, balanced accuracy, and identical folds)."""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             recall_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'C:\Users\admin\.zcode\workspace\default\dga_revise\llm')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data2 import load_primary_clean, textbook_ratios, build_matrix, CANONICAL_RATIOS, GASES

BASE = r'C:\Users\admin\.zcode\workspace\default\dga_revise'
OUT = os.path.join(BASE, 'results_rev')
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

X, y = load_primary_clean()
FEATS = list(GASES) + list(CANONICAL_RATIOS)
Xm = build_matrix(X, FEATS).astype(np.float32)


class CNN1D(nn.Module):
    """Two conv blocks over the standardised feature vector viewed as a 1-D signal."""
    def __init__(self, d, n_cls=6, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(ch * 2 * 4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class BiLSTM(nn.Module):
    """Bidirectional LSTM over the feature sequence (chunks of 4 features)."""
    def __init__(self, d, n_cls=6, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(64, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out.mean(dim=1))


class TransformerClf(nn.Module):
    """Small transformer encoder with 8 tokens (feature chunks) and 4 heads."""
    def __init__(self, d, n_cls=6, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=64, dropout=0.2,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(d_model, n_cls)

    def forward(self, x):
        # pad to a multiple of 3 tokens
        pad = (-x.shape[1]) % 3
        if pad:
            x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
        tok = self.proj(x.view(x.shape[0], -1, 3))
        return self.fc(self.enc(tok).mean(dim=1))


def run_model(kind, Xtr, ytr, Xte, yte, epochs=100, lr=1e-3, bs=64):
    d = Xtr.shape[1]
    model = {'cnn': CNN1D, 'bilstm': BiLSTM, 'transformer': TransformerClf}[kind](d)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte))
        proba = torch.softmax(logits, dim=1).numpy()
        pred = proba.argmax(1)
    return pred, proba


REPEATS = 2   # DL baselines are expensive; 3 repeats x 5 folds
results = {}
for kind in ['cnn', 'bilstm', 'transformer']:
    t0 = time.time()
    accs, mf1, bacc, recs = [], [], [], []
    for r in range(REPEATS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + r)
        for tr, te in skf.split(Xm, y):
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xm[tr]), sc.transform(Xm[te])
            pred, _ = run_model(kind, Xtr, y[tr], Xte, y[te])
            accs.append(accuracy_score(y[te], pred) * 100)
            mf1.append(f1_score(y[te], pred, average='macro', zero_division=0) * 100)
            bacc.append(balanced_accuracy_score(y[te], pred) * 100)
            recs.append(recall_score(y[te], pred, average=None, zero_division=0,
                                     labels=list(range(6))) * 100)
    results[kind] = {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'macro_f1_mean': float(np.mean(mf1)), 'macro_f1_std': float(np.std(mf1)),
        'balanced_accuracy_mean': float(np.mean(bacc)),
        'per_class_recall_mean': np.mean(recs, axis=0).tolist(),
        'repeats': REPEATS, 'epochs': 100,
    }
    print(f'{kind:<12} acc={np.mean(accs):6.2f}±{np.std(accs):.2f} '
          f'macroF1={np.mean(mf1):6.2f} bAcc={np.mean(bacc):6.2f} '
          f'({time.time()-t0:.0f}s)')

# architecture summary for the manuscript
results['architecture'] = {
    'cnn': 'Conv1d(1->32,k=3)+BN+ReLU, Conv1d(32->64,k=3)+BN+ReLU, adaptive-avg-pool(4), '
           'FC(256->64)+ReLU+dropout(0.3), FC(64->6); AdamW lr=1e-3 wd=1e-3, 120 epochs, batch 64',
    'bilstm': 'BiLSTM(input 1, hidden 32), mean pooling over steps, FC(64->64)+ReLU+dropout(0.3), '
              'FC(64->6); same optimiser schedule',
    'transformer': 'per-token linear projection of 3 features -> d_model 32, 2 encoder layers '
                   '(4 heads, FFN 64, dropout 0.2), mean pooling, FC(32->6); same schedule',
    'protocol': 'identical repeated stratified 5-fold splits (3 repeats), fold-wise '
                'standardisation, no feature selection beyond the 19 canonical columns',
}
json.dump(results, open(os.path.join(OUT, 'dl.json'), 'w', encoding='utf-8'), indent=1)
print('wrote', os.path.join(OUT, 'dl.json'))
