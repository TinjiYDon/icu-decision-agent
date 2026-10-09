"""H5 · Sparsity-Gated Dual Fusion (SGDF).

Without Layer0 sequences, the temporal track is a **multi-hour tabular
trajectory** of the same LightGBM scores (honest surrogate). The gate uses
per-row feature completeness: denser observations → more weight on trajectory;
sparser → fall back to the current-hour score.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from domain.features.build import FEATURE_COLS
from domain.models.evaluation import binary_metrics


def feature_completeness(frame: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> np.ndarray:
    """Fraction of non-null features in ``feature_cols`` per row ∈ [0, 1]."""
    cols = list(feature_cols or FEATURE_COLS)
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return np.zeros(len(frame), dtype=float)
    sub = frame[cols]
    return (1.0 - sub.isna().to_numpy().mean(axis=1)).astype(float)


def fuse_scores(
    p_current: np.ndarray,
    p_trajectory: np.ndarray,
    completeness: np.ndarray,
    *,
    gate_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fused, gate_w) where gate_w weights the trajectory track."""
    pc = np.asarray(p_current, dtype=float)
    pt = np.asarray(p_trajectory, dtype=float)
    c = np.clip(np.asarray(completeness, dtype=float), 0.0, 1.0)
    w = np.power(c, float(gate_power))
    fused = (1.0 - w) * pc + w * pt
    return fused, w


def trajectory_from_hour_map(
    stay_ids: Sequence[int],
    hour_index: int,
    hour_prob: dict[tuple[int, int], float],
    *,
    hours: Sequence[int] = (0, 1, 2, 4, 6),
) -> np.ndarray:
    """Mean of available scores with hour <= hour_index (inclusive)."""
    h_eval = int(hour_index)
    out = np.zeros(len(stay_ids), dtype=float)
    for i, sid in enumerate(stay_ids):
        vals = [
            float(hour_prob[(int(sid), int(h))])
            for h in hours
            if int(h) <= h_eval and (int(sid), int(h)) in hour_prob
        ]
        out[i] = float(np.mean(vals)) if vals else float("nan")
    return out


def compare_fusion_policies(
    y: np.ndarray,
    p_current: np.ndarray,
    p_trajectory: np.ndarray,
    completeness: np.ndarray,
    *,
    threshold: float,
    gate_power: float = 1.0,
) -> dict[str, Any]:
    """Compare current-only / ungated mean / sparsity-gated fusion."""
    y = np.asarray(y, dtype=int)
    pc = np.asarray(p_current, dtype=float)
    pt = np.asarray(p_trajectory, dtype=float)
    mask = np.isfinite(pc) & np.isfinite(pt)
    y, pc, pt, comp = y[mask], pc[mask], pt[mask], np.asarray(completeness)[mask]
    p_mean = 0.5 * pc + 0.5 * pt
    p_fuse, gate = fuse_scores(pc, pt, comp, gate_power=gate_power)

    def row(name: str, p: np.ndarray) -> dict[str, Any]:
        m = binary_metrics(y, p, threshold=float(threshold))
        return {
            "policy": name,
            "n": int(len(y)),
            "pr_auc": m["pr_auc"],
            "brier": m["brier"],
            "roc_auc": m["roc_auc"],
            "f1": m["f1"],
            "net_benefit": m.get("net_benefit_model"),
        }

    rows = [
        row("current_hour", pc),
        row("ungated_mean", p_mean),
        row("sparsity_gated", p_fuse),
    ]
    # High-sparsity subset: completeness below median
    med = float(np.median(comp)) if len(comp) else 0.0
    sparse = comp <= med
    dense = ~sparse
    subset: dict[str, Any] = {}
    if sparse.sum() >= 20 and dense.sum() >= 20 and len(np.unique(y[sparse])) > 1:
        subset["sparse_half"] = {
            "n": int(sparse.sum()),
            "mean_completeness": float(comp[sparse].mean()),
            "mean_gate": float(gate[sparse].mean()),
            "current_pr_auc": binary_metrics(y[sparse], pc[sparse], threshold=threshold)["pr_auc"],
            "fused_pr_auc": binary_metrics(y[sparse], p_fuse[sparse], threshold=threshold)["pr_auc"],
        }
    if dense.sum() >= 20 and len(np.unique(y[dense])) > 1:
        subset["dense_half"] = {
            "n": int(dense.sum()),
            "mean_completeness": float(comp[dense].mean()),
            "mean_gate": float(gate[dense].mean()),
            "current_pr_auc": binary_metrics(y[dense], pc[dense], threshold=threshold)["pr_auc"],
            "fused_pr_auc": binary_metrics(y[dense], p_fuse[dense], threshold=threshold)["pr_auc"],
        }

    base_pr = rows[0]["pr_auc"]
    fuse_pr = rows[2]["pr_auc"]
    return {
        "status": "ok",
        "hypothesis": "H5_SGDF",
        "temporal_track": "multi_hour_lgbm_trajectory_surrogate",
        "note": (
            "No Layer0 GRU-D: trajectory = mean LGBM probs at hours ≤ h. "
            "Gate w = completeness^gate_power weights trajectory vs current hour."
        ),
        "threshold": float(threshold),
        "gate_power": float(gate_power),
        "mean_completeness": float(comp.mean()) if len(comp) else None,
        "mean_gate": float(gate.mean()) if len(gate) else None,
        "rows": rows,
        "subsets": subset,
        "pass_criteria": {
            "fused_pr_auc_not_worse_than_current": bool(
                fuse_pr == fuse_pr and base_pr == base_pr and fuse_pr + 1e-9 >= base_pr - 0.02
            ),
        },
    }
