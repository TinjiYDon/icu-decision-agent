from __future__ import annotations

import numpy as np

from domain.models.temporal.lab_temporal import (
    LAB_FEATURE_NAMES,
    _json_safe,
    _metrics,
)


def test_lab_feature_count():
    assert len(LAB_FEATURE_NAMES) >= 8
    assert "lactate" in LAB_FEATURE_NAMES
    assert "wbc" in LAB_FEATURE_NAMES


def test_metrics_handles_single_class():
    y = np.zeros(20, dtype=int)
    p = np.linspace(0.01, 0.2, 20)
    m = _metrics(y, p)
    assert m["pr_auc"] != m["pr_auc"]  # nan
    assert m["brier"] == m["brier"]


def test_json_safe_nan():
    assert _json_safe({"pr_auc": float("nan"), "ok": 1.0}) == {"pr_auc": None, "ok": 1.0}
