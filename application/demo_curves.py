"""Read-only demo curves: net benefit on a sampled test split (no retrain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from domain.features.build import FEATURE_COLS
from domain.models.lgbm import _load_training_frame
from domain.models.split import split_frame_by_stay

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "models" / "lgbm_mortality_12h.txt"
MAX_TEST_ROWS = 5000


def net_benefit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decision-curve analysis style net benefit vs threshold probability."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.80, 16)
    n = max(len(y), 1)
    prevalence = float(y.mean()) if n else 0.0
    nb_model: list[float] = []
    nb_all: list[float] = []
    for t in thresholds:
        t = float(t)
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        # NB = TP/n - FP/n * (t/(1-t))
        odds = t / max(1.0 - t, 1e-9)
        nb_model.append(tp / n - fp / n * odds)
        nb_all.append(prevalence - (1.0 - prevalence) * odds)
    return {
        "thresholds": [float(x) for x in thresholds],
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": [0.0] * len(thresholds),
        "n": int(n),
        "prevalence": prevalence,
        "status": "ok",
    }


def compute_demo_net_benefit(*, max_rows: int = MAX_TEST_ROWS) -> dict[str, Any]:
    """Load feat/label + saved booster; score a capped test slice."""
    if not MODEL_PATH.exists():
        return {"status": "no_model", "message": "缺少 lgbm_mortality_12h.txt"}
    try:
        df = _load_training_frame()
        if len(df) < 50:
            return {"status": "too_few", "message": f"样本过少 n={len(df)}"}
        _tr, _va, test_df, _m = split_frame_by_stay(df)
        if len(test_df) > max_rows:
            test_df = test_df.sample(n=max_rows, random_state=42)
        booster = lgb.Booster(model_file=str(MODEL_PATH))
        X = test_df[FEATURE_COLS].to_numpy(dtype=float)
        # LightGBM booster.predict returns margin or prob depending on model
        prob = np.asarray(booster.predict(X), dtype=float)
        if prob.min() < 0 or prob.max() > 1:
            # logistic if raw score
            prob = 1.0 / (1.0 + np.exp(-prob))
        y = test_df["label"].to_numpy(dtype=int)
        out = net_benefit_curve(y, prob)
        out["sampled_n"] = int(len(test_df))
        return out
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}
