from __future__ import annotations

import numpy as np
import pandas as pd

from domain.models.sparsity_fusion import (
    compare_fusion_policies,
    feature_completeness,
    fuse_scores,
    trajectory_from_hour_map,
)


def test_fuse_scores_respects_completeness():
    pc = np.array([0.1, 0.1])
    pt = np.array([0.9, 0.9])
    fused, w = fuse_scores(pc, pt, np.array([0.0, 1.0]))
    assert abs(fused[0] - 0.1) < 1e-9
    assert abs(fused[1] - 0.9) < 1e-9
    assert w[0] == 0.0 and w[1] == 1.0


def test_trajectory_means_past_hours_only():
    hour_prob = {(1, 0): 0.2, (1, 1): 0.4, (1, 6): 0.8}
    out = trajectory_from_hour_map([1], hour_index=1, hour_prob=hour_prob, hours=(0, 1, 2, 4, 6))
    assert abs(out[0] - 0.3) < 1e-9  # (0.2+0.4)/2, excludes h=6


def test_feature_completeness():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    c = feature_completeness(df, ["a", "b"])
    assert abs(c[0] - 1.0) < 1e-9
    assert abs(c[1] - 0.5) < 1e-9


def test_compare_fusion_policies_runs():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    pc = np.clip(y * 0.5 + rng.random(100) * 0.3, 0, 1)
    pt = np.clip(y * 0.55 + rng.random(100) * 0.3, 0, 1)
    comp = rng.random(100)
    out = compare_fusion_policies(y, pc, pt, comp, threshold=0.3)
    assert out["status"] == "ok"
    assert len(out["rows"]) == 3
