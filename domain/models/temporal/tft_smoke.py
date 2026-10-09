"""Synthetic TFT-lite smoke metrics (CPU, seconds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from domain.models.temporal.tft_lite import TFTLite

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "d3" / "tft_lite.json"


def _synthetic_batch(n: int = 96, t: int = 6, f: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, t, f)).astype(np.float32)
    m = (rng.random((n, t, f)) > 0.35).astype(np.float32)
    d = rng.integers(0, 4, size=(n, t, f)).astype(np.float32)
    signal = (x * m).mean(axis=(1, 2))
    logits = 1.2 * signal + rng.normal(scale=0.4, size=n)
    y = (logits > np.median(logits)).astype(np.float32)
    return x, m, d, y


def run_smoke(*, epochs: int = 12, write: bool = True) -> dict[str, Any]:
    x, m, d, y = _synthetic_batch()
    n = len(y)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    X = torch.tensor(x)
    M = torch.tensor(m)
    D = torch.tensor(d)
    Y = torch.tensor(y)
    model = TFTLite(input_size=x.shape[-1], d_model=16, nhead=4, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X[:n_tr], M[:n_tr], D[:n_tr]), Y[:n_tr])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te = torch.sigmoid(model(X[n_tr + n_va :], M[n_tr + n_va :], D[n_tr + n_va :])).numpy()
        yt = y[n_tr + n_va :]
    payload: dict[str, Any] = {
        "status": "ok",
        "note": (
            "TFT-lite 合成数据烟测：注意力时序对照轨，不是 Lim TFT 全文实现，"
            "也不是默认床旁模型。真实 MIMIC 对照仍走 GRU-D D3 脚本。"
        ),
        "n_test": int(len(yt)),
        "pr_auc": float(average_precision_score(yt, te)) if len(np.unique(yt)) > 1 else None,
        "brier": float(brier_score_loss(yt, te)),
        "roc_auc": float(roc_auc_score(yt, te)) if len(np.unique(yt)) > 1 else None,
        "primary_metric": "pr_auc",
    }
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
