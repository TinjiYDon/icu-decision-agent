"""H4 · Utility-Calibrated Early Warning (UCEW).

Select operating thresholds by validating **net benefit** under a clinical
cost ratio, instead of (only) maximizing F1 — closing the train/eval utility gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domain.models.dca import compute_net_benefit
from domain.models.evaluation import binary_metrics, select_threshold_by_f1


def select_threshold_by_net_benefit(
    y_true: Any,
    probability: Any,
    *,
    cost_ratio: float = 4.0,
    thr_min: float = 0.05,
    thr_max: float = 0.40,
    n_grid: int = 200,
    max_alert_rate: float | None = 0.20,
) -> float:
    """Pick threshold on validation that maximizes net benefit.

    Defaults use an early-warning threshold window and optional alert-rate cap so
    the optimum does not collapse to treat-all on rare-event tasks.
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    grid = np.linspace(float(thr_min), float(thr_max), int(n_grid))
    best_nb, best_t = -1e18, float(grid[0])
    found = False
    for t in grid:
        alert = float((prob >= float(t)).mean())
        if max_alert_rate is not None and alert > float(max_alert_rate) + 1e-12:
            continue
        pt = compute_net_benefit(y, prob, threshold=float(t), cost_ratio=float(cost_ratio))
        if pt.net_benefit_model > best_nb:
            best_nb = float(pt.net_benefit_model)
            best_t = float(t)
            found = True
    if not found:
        # Fallback: ignore alert cap inside the clinical window
        for t in grid:
            pt = compute_net_benefit(
                y, prob, threshold=float(t), cost_ratio=float(cost_ratio)
            )
            if pt.net_benefit_model > best_nb:
                best_nb = float(pt.net_benefit_model)
                best_t = float(t)
    return best_t


def compare_f1_vs_utility(
    y_val: Any,
    p_val: Any,
    y_test: Any,
    p_test: Any,
    *,
    cost_ratio: float = 4.0,
    thr_min: float = 0.05,
    thr_max: float = 0.40,
    max_alert_rate: float | None = 0.20,
) -> dict[str, Any]:
    """Compare max-F1 vs constrained max-NB thresholds (val→test)."""
    t_f1 = select_threshold_by_f1(y_val, p_val)
    t_nb = select_threshold_by_net_benefit(
        y_val,
        p_val,
        cost_ratio=cost_ratio,
        thr_min=thr_min,
        thr_max=thr_max,
        max_alert_rate=max_alert_rate,
    )
    # Unconstrained low-threshold NB (diagnostic: often ≈ treat-all on rare events)
    t_nb_free = select_threshold_by_net_benefit(
        y_val,
        p_val,
        cost_ratio=cost_ratio,
        thr_min=0.01,
        thr_max=0.50,
        max_alert_rate=None,
    )

    def _pack(name: str, thr: float) -> dict[str, Any]:
        m_val = binary_metrics(y_val, p_val, threshold=thr)
        m_test = binary_metrics(y_test, p_test, threshold=thr)
        nb_val = compute_net_benefit(
            np.asarray(y_val, dtype=int),
            np.asarray(p_val, dtype=float),
            threshold=thr,
            cost_ratio=cost_ratio,
        )
        nb_test = compute_net_benefit(
            np.asarray(y_test, dtype=int),
            np.asarray(p_test, dtype=float),
            threshold=thr,
            cost_ratio=cost_ratio,
        )
        return {
            "method": name,
            "threshold": float(thr),
            "val": {
                "f1": m_val["f1"],
                "precision": m_val["precision"],
                "recall": m_val["recall"],
                "alert_rate": float((np.asarray(p_val) >= thr).mean()),
                "net_benefit": float(nb_val.net_benefit_model),
                "net_benefit_treat_all": float(nb_val.net_benefit_treat_all),
            },
            "test": {
                "pr_auc": m_test["pr_auc"],
                "brier": m_test["brier"],
                "roc_auc": m_test["roc_auc"],
                "f1": m_test["f1"],
                "precision": m_test["precision"],
                "recall": m_test["recall"],
                "alert_rate": float((np.asarray(p_test) >= thr).mean()),
                "net_benefit": float(nb_test.net_benefit_model),
                "net_benefit_treat_all": float(nb_test.net_benefit_treat_all),
            },
        }

    f1_row = _pack("max_f1_val", t_f1)
    nb_row = _pack("max_net_benefit_constrained_val", t_nb)
    free_row = _pack("max_net_benefit_unconstrained_val", t_nb_free)
    delta_nb = nb_row["test"]["net_benefit"] - f1_row["test"]["net_benefit"]
    delta_alert = nb_row["test"]["alert_rate"] - f1_row["test"]["alert_rate"]
    return {
        "status": "ok",
        "hypothesis": "H4_UCEW",
        "cost_ratio": float(cost_ratio),
        "search": {
            "thr_min": thr_min,
            "thr_max": thr_max,
            "max_alert_rate": max_alert_rate,
        },
        "note": (
            "Primary utility policy: max NB on val inside early-warning window "
            f"[{thr_min},{thr_max}] with alert_rate≤{max_alert_rate}. "
            "Unconstrained NB is reported as a diagnostic (may collapse to treat-all)."
        ),
        "f1_policy": f1_row,
        "utility_policy": nb_row,
        "utility_unconstrained_diagnostic": free_row,
        "test_delta": {
            "net_benefit_utility_minus_f1": float(delta_nb),
            "alert_rate_utility_minus_f1": float(delta_alert),
            "pr_auc_unchanged": True,
        },
        "pass_criteria": {
            "utility_nb_ge_f1_nb": bool(delta_nb >= -1e-9),
            "alert_rate_within_cap": bool(
                max_alert_rate is None
                or nb_row["test"]["alert_rate"] <= float(max_alert_rate) + 0.02
            ),
            "discrimination_intact": True,
        },
    }
