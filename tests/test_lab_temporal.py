from __future__ import annotations

import numpy as np

from domain.features.sequence_build import build_mask_delta
from domain.models.temporal.lab_temporal import (
    LAB_FEATURE_NAMES,
    _feature_window_hours,
    _json_safe,
    _metrics,
    _quota_caps,
    events_to_raw_grid,
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


def test_quota_caps_never_exceed_limit():
    for limit in (1, 10, 40, 80, 119, 120, 500):
        pos, neg = _quota_caps(limit)
        assert pos + neg == limit
        assert pos >= 1 or limit == 0
        assert neg >= 0


def test_feature_window_clamps_lookback_past_prediction():
    # lookback=12 with h=6 must end at 6 (no future labs), start at 0
    start, end = _feature_window_hours(hour_index=6, lookback_hours=12)
    assert start == 0.0
    assert end == 6.0
    start2, end2 = _feature_window_hours(hour_index=6, lookback_hours=3)
    assert start2 == 3.0
    assert end2 == 6.0


def test_events_to_raw_grid_no_locf_into_mask():
    """Gaps stay NaN in raw so build_mask_delta can mark them unobserved."""
    grid = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=float)
    # itemid 50813 = lactate; observe only at t=0
    events = [(0.1, 50813, 2.5)]
    name_by_id = {50813: "lactate"}
    feature_index = {"lactate": 0}
    raw = events_to_raw_grid(
        events,
        grid=grid,
        name_by_id=name_by_id,
        feature_index=feature_index,
        n_features=1,
    )
    assert raw[0, 0] == 2.5
    assert np.isnan(raw[1, 0]) and np.isnan(raw[2, 0]) and np.isnan(raw[3, 0])
    _x, m, d = build_mask_delta(raw, grid)
    assert m[0, 0] == 1.0
    assert m[1, 0] == 0.0
    assert d[2, 0] > d[1, 0]  # recency grows across gap


def test_events_reject_after_prediction_grid_end():
    grid = np.asarray([0.0, 3.0, 6.0], dtype=float)
    events = [(6.5, 50813, 9.0)]  # after prediction
    raw = events_to_raw_grid(
        events,
        grid=grid,
        name_by_id={50813: "lactate"},
        feature_index={"lactate": 0},
        n_features=1,
    )
    assert np.all(np.isnan(raw))
