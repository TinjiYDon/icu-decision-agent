"""H6 · Sparse Utility Fusion Head (SUFH) + Recency-Weighted trajectory (RWSG).

Named algorithmic contributions beyond plain averaging:

1. **RWSG trajectory**: exponential recency weights on multi-hour scores.
2. **SUFH**: a tiny logistic meta-model on
   ``[p_cur, p_rw, c, p_cur - p_rw]`` trained only on validation stays.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from domain.models.decision_indices import index_bundle
from domain.models.evaluation import binary_metrics
from domain.models.sparsity_fusion import feature_completeness, fuse_scores


def recency_weighted_trajectory(
    stay_ids: Sequence[int],
    hour_index: int,
    hour_prob: dict[tuple[int, int], float],
    *,
    hours: Sequence[int] = (0, 1, 2, 4, 6),
    beta: float = 0.35,
) -> np.ndarray:
    """p_rw = Σ exp(-β·(h−h')) p(h') / Z for h' ≤ h."""
    h_eval = int(hour_index)
    b = float(beta)
    out = np.full(len(stay_ids), np.nan, dtype=float)
    for i, sid in enumerate(stay_ids):
        pairs: list[tuple[float, float]] = []
        for h in hours:
            hi = int(h)
            if hi > h_eval:
                continue
            key = (int(sid), hi)
            if key not in hour_prob:
                continue
            w = float(np.exp(-b * (h_eval - hi)))
            pairs.append((w, float(hour_prob[key])))
        if not pairs:
            continue
        sw = sum(w for w, _ in pairs)
        out[i] = sum(w * p for w, p in pairs) / max(sw, 1e-12)
    return out


def build_sufh_design(
    p_cur: np.ndarray,
    p_rw: np.ndarray,
    completeness: np.ndarray,
) -> np.ndarray:
    pc = np.asarray(p_cur, dtype=float).reshape(-1)
    pr = np.asarray(p_rw, dtype=float).reshape(-1)
    c = np.clip(np.asarray(completeness, dtype=float).reshape(-1), 0.0, 1.0)
    return np.column_stack([pc, pr, c, pc - pr])


def train_sufh(
    p_cur: np.ndarray,
    p_rw: np.ndarray,
    completeness: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> Pipeline:
    """Fit logistic SUFH on validation design matrix only."""
    X = build_sufh_design(p_cur, p_rw, completeness)
    y = np.asarray(y, dtype=int)
    mask = np.isfinite(X).all(axis=1)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=300,
                    C=1.0,
                    # Unbalanced: preserve calibration for NB/UEI; balanced
                    # over-alerts on ~2% prevalence and collapses utility.
                    class_weight=None,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    clf.fit(X[mask], y[mask])
    return clf


def predict_sufh(
    model: Pipeline,
    p_cur: np.ndarray,
    p_rw: np.ndarray,
    completeness: np.ndarray,
) -> np.ndarray:
    X = build_sufh_design(p_cur, p_rw, completeness)
    out = np.full(len(X), np.nan, dtype=float)
    mask = np.isfinite(X).all(axis=1)
    if mask.any():
        out[mask] = model.predict_proba(X[mask])[:, 1]
    return out


def run_cugew_compare(
    *,
    y_val: np.ndarray,
    p_cur_val: np.ndarray,
    p_rw_val: np.ndarray,
    c_val: np.ndarray,
    y_te: np.ndarray,
    p_cur_te: np.ndarray,
    p_rw_te: np.ndarray,
    c_te: np.ndarray,
    threshold: float,
    cost_ratio: float = 4.0,
    gate_power: float = 1.0,
    beta: float = 0.35,
) -> dict[str, Any]:
    """Compare legacy / RWSG-gate / SUFH policies with DEI·ANB·UEI."""
    # Legacy uniform trajectory ≈ mean; here p_rw is already recency-weighted input
    p_gate_te, w_te = fuse_scores(
        p_cur_te, p_rw_te, c_te, gate_power=gate_power
    )
    head = train_sufh(p_cur_val, p_rw_val, c_val, y_val)
    p_sufh_te = predict_sufh(head, p_cur_te, p_rw_te, c_te)

    policies = {
        "legacy_current": np.asarray(p_cur_te, dtype=float),
        "rwsg_gated": p_gate_te,
        "sufh_meta": p_sufh_te,
    }

    rows: list[dict[str, Any]] = []
    for name, p in policies.items():
        mask = np.isfinite(p)
        y = np.asarray(y_te, dtype=int)[mask]
        pp = p[mask]
        m = binary_metrics(y, pp, threshold=float(threshold))
        idx = index_bundle(y, pp, threshold=float(threshold), cost_ratio=cost_ratio)
        rows.append(
            {
                "policy": name,
                "n": int(len(y)),
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "roc_auc": m["roc_auc"],
                "f1": m["f1"],
                "net_benefit": m.get("net_benefit_model"),
                **{k: idx[k] for k in ("DEI", "ANB", "UEI", "alert_rate")},
            }
        )

    # coefficient snapshot for transparency
    logit = head.named_steps["logit"]
    coefs = {
        "p_cur": float(logit.coef_[0, 0]),
        "p_rw": float(logit.coef_[0, 1]),
        "completeness": float(logit.coef_[0, 2]),
        "delta_p": float(logit.coef_[0, 3]),
        "intercept": float(logit.intercept_[0]),
    }

    best_uei = max(rows, key=lambda r: (r["UEI"] if r["UEI"] == r["UEI"] else -1e18))
    return {
        "status": "ok",
        "framework": "CUGEW",
        "contributions": ["DEI", "ANB", "UEI", "RWSG", "SUFH"],
        "note": (
            "CUGEW = Causal Utility-Gated Early Warning. "
            "RWSG: recency-weighted multi-hour trajectory + completeness gate. "
            "SUFH: logistic fusion head trained on val only. "
            "New indices: DEI=NB−NB_all, ANB=NB/alert, UEI=DEI·(1−alert)."
        ),
        "beta_recency": float(beta),
        "gate_power": float(gate_power),
        "cost_ratio": float(cost_ratio),
        "threshold": float(threshold),
        "sufh_coefficients": coefs,
        "mean_gate": float(np.nanmean(w_te)),
        "rows": rows,
        "best_by_UEI": best_uei["policy"],
    }
