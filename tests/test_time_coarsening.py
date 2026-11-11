"""受控粗化标签单测（D1 纯函数部分）。"""
import numpy as np
import pytest

from domain.labels.time_coarsening import coarse_interval, coarse_label, coarsen_labels


def test_interval_floor():
    assert coarse_interval(5.0, 6.0) == (0.0, 6.0)
    assert coarse_interval(6.1, 6.0) == (6.0, 12.0)
    with pytest.raises(ValueError):
        coarse_interval(1.0, 0.0)


def test_label_inside_outside_ambiguous():
    # 窗口 [0, 12)，6h 分辨率
    assert coarse_label(3.0, 0.0, horizon_hours=12.0, resolution_hours=6.0) == 1.0
    assert coarse_label(15.0, 0.0, horizon_hours=12.0, resolution_hours=6.0) == 0.0
    assert coarse_label(30.0, 0.0, horizon_hours=12.0, resolution_hours=6.0) == 0.0
    # 日级：6h 事件落在 [0,24) 跨窗，不可确定
    assert coarse_label(6.0, 0.0, horizon_hours=12.0, resolution_hours=24.0) is None


def test_coarsen_labels_stats():
    events = [3.0, 10.0, 30.0, 6.0]
    preds = [0.0, 0.0, 0.0, 0.0]
    out = coarsen_labels(events, preds, horizon_hours=12.0, resolution_hours=8.0)
    assert out["labels"].shape == (4,)
    assert out["n_ambiguous"] == 1
    assert out["ambiguity_rate"] == pytest.approx(0.25)
    assert out["labels"][0] == 1.0
    assert out["labels"][2] == 0.0
    assert np.isnan(out["labels"][1])


def test_coarsen_labels_validation():
    with pytest.raises(ValueError):
        coarsen_labels([], [])
    with pytest.raises(ValueError):
        coarsen_labels([1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        coarsen_labels([np.nan], [0.0])
