"""New decision indices for rare-event early warning (H6 / CUGEW).

These are **first-class evaluation indices**, not just ROC/PR-AUC:

* **DEI** — Decision Efficiency Index: excess net benefit over treat-all.
* **ANB** — Alert-Normalized Benefit: net benefit per unit alert load.
* **UEI** — Utility Efficiency Index: DEI scaled by (1 - alert_rate) to reward
  selective alarms that still beat treat-all.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domain.models.dca import compute_net_benefit


def decision_efficiency_index(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
    cost_ratio: float = 4.0,
) -> dict[str, float]:
    """DEI = NB_model − NB_treat_all (higher is better; >0 beats treat-all)."""
    pt = compute_net_benefit(
        np.asarray(y_true, dtype=int),
        np.asarray(probability, dtype=float),
        threshold=float(threshold),
        cost_ratio=float(cost_ratio),
    )
    dei = float(pt.net_benefit_model - pt.net_benefit_treat_all)
    return {
        "DEI": dei,
        "NB_model": float(pt.net_benefit_model),
        "NB_treat_all": float(pt.net_benefit_treat_all),
        "threshold": float(threshold),
        "cost_ratio": float(cost_ratio),
    }


def alert_normalized_benefit(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
    cost_ratio: float = 4.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """ANB = NB_model / max(alert_rate, eps)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    t = float(threshold)
    alert = float((p >= t).mean()) if len(p) else 0.0
    pt = compute_net_benefit(y, p, threshold=t, cost_ratio=float(cost_ratio))
    anb = float(pt.net_benefit_model) / max(alert, float(eps))
    return {
        "ANB": anb,
        "alert_rate": alert,
        "NB_model": float(pt.net_benefit_model),
        "threshold": t,
        "cost_ratio": float(cost_ratio),
    }


def utility_efficiency_index(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
    cost_ratio: float = 4.0,
) -> dict[str, float]:
    """UEI = DEI × (1 − alert_rate): reward beating treat-all with fewer alerts."""
    dei = decision_efficiency_index(
        y_true, probability, threshold=threshold, cost_ratio=cost_ratio
    )
    anb = alert_normalized_benefit(
        y_true, probability, threshold=threshold, cost_ratio=cost_ratio
    )
    uei = float(dei["DEI"]) * (1.0 - float(anb["alert_rate"]))
    return {
        "UEI": uei,
        "DEI": dei["DEI"],
        "ANB": anb["ANB"],
        "alert_rate": anb["alert_rate"],
        "NB_model": dei["NB_model"],
        "NB_treat_all": dei["NB_treat_all"],
        "threshold": float(threshold),
        "cost_ratio": float(cost_ratio),
    }


def index_bundle(
    y_true: Any,
    probability: Any,
    *,
    threshold: float,
    cost_ratio: float = 4.0,
) -> dict[str, float]:
    """Pack DEI / ANB / UEI for one policy."""
    uei = utility_efficiency_index(
        y_true, probability, threshold=threshold, cost_ratio=cost_ratio
    )
    return uei
