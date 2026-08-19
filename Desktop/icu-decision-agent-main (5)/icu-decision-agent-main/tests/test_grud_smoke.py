"""Tests for sequence → GRU-D smoke path."""

import numpy as np

from domain.features.sequence_build import apply_recency_weights, build_mask_delta, pad_truncate
from domain.models.temporal.grud import grud_forward_numpy, smoke_grud_batch


def test_mask_delta_shapes():
    raw = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    times = np.array([0.0, 1.0, 2.0])
    x, m, d = build_mask_delta(raw, times)
    assert x.shape == (3, 2)
    assert m.shape == (3, 2)
    assert d.shape == (3, 2)
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0


def test_recency_weights_monotone():
    w = apply_recency_weights(np.array([0.0, 1.0, 2.0]), lam=0.5)
    assert w[-1] >= w[0]


def test_grud_smoke_batch():
    x = np.zeros((2, 4, 3))
    m = np.ones((2, 4, 3))
    d = np.ones((2, 4, 3))
    out = smoke_grud_batch(x, m, d)
    assert out["n"] == 2
    h = grud_forward_numpy(x[0], m[0], d[0], hidden_size=8)
    assert h.shape == (8,)


def test_pad_truncate():
    x = np.ones((2, 3))
    m = np.ones((2, 3))
    d = np.ones((2, 3))
    xp, mp, dp = pad_truncate(x, m, d, 5)
    assert xp.shape == (5, 3)
