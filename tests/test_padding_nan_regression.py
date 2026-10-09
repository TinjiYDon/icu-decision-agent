"""Regression tests for numeric-safety bugs found during review.

Root cause: `pad_truncate` used `np.inf` as the "never observed" delta
sentinel. Downstream decay computes `w * delta` with learnable `w` whose
initial value is 0, and `0 * inf == NaN`, which silently turned the entire
GRU-D / TFT-lite forward pass into NaN. These tests lock the fix in place.
"""

from __future__ import annotations

import numpy as np
import torch

from domain.features.sequence_build import PAD_DELTA, pad_truncate
from domain.models.temporal.grud_model import GRUD
from domain.models.temporal.tft_lite import TFTLite


def test_pad_truncate_uses_finite_delta_sentinel():
    """Padding must stay finite so that 0 * delta cannot produce NaN."""
    x = np.ones((2, 3))
    m = np.ones((2, 3))
    d = np.ones((2, 3))
    xp, mp, dp = pad_truncate(x, m, d, 5)

    assert xp.shape == (5, 3)
    assert np.isfinite(dp).all(), "delta padding must be finite, not inf"
    assert (dp[2:] == PAD_DELTA).all()
    # The critical invariant: 0 * delta must stay 0.
    assert (np.zeros(1) * dp).sum() == 0.0


def test_pad_truncate_right_pads():
    """Real observations stay at the FRONT; filler is appended at the END."""
    x = np.array([[1.0], [2.0]])
    m = np.array([[1.0], [1.0]])
    d = np.array([[0.0], [0.5]])
    xp, mp, dp = pad_truncate(x, m, d, 4)

    np.testing.assert_allclose(xp[:2].ravel(), [1.0, 2.0])
    np.testing.assert_allclose(mp[:2].ravel(), [1.0, 1.0])
    assert (mp[2:] == 0.0).all()
    assert dp[0, 0] == 0.0  # untouched real rows


def test_grud_forward_is_finite_with_padded_delta():
    """GRU-D must not emit NaN when delta carries the padding sentinel."""
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    m = torch.ones(4, 6, 3)
    d = torch.ones(4, 6, 3)
    m[:, 3:] = 0.0
    d[:, 3:] = float("inf")  # worst case: legacy inf padding still accepted

    model = GRUD(input_size=3, hidden_size=8)
    model.eval()
    out = model(x, m, d)

    assert torch.isfinite(out).all(), f"non-finite output: {out}"


def test_grud_trains_on_padded_batch():
    """End-to-end training on right-padded data keeps parameters finite."""
    torch.manual_seed(0)
    x = torch.randn(8, 6, 3)
    m = torch.ones(8, 6, 3)
    d = torch.ones(8, 6, 3)
    m[:, 4:] = 0.0
    d[:, 4:] = PAD_DELTA
    y = torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0])

    model = GRUD(input_size=3, hidden_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.BCELoss()
    model.train()
    for _ in range(3):
        opt.zero_grad()
        loss = crit(model(x, m, d).clamp(1e-6, 1 - 1e-6), y)
        loss.backward()
        opt.step()

    assert not any(torch.isnan(p).any() for p in model.parameters())
    assert torch.isfinite(loss)


def test_tft_lite_is_finite_with_inf_padding():
    """TFT-lite must survive inf padding (log1p(inf) previously poisoned it)."""
    torch.manual_seed(0)
    x = torch.randn(3, 5, 4)
    m = torch.ones(3, 5, 4)
    d = torch.ones(3, 5, 4)
    m[0, 3:] = 0.0
    d[0, 3:] = float("inf")

    model = TFTLite(input_size=4, d_model=16, nhead=4, num_layers=1)
    model.eval()
    out = model(x, m, d)

    assert torch.isfinite(out).all(), f"non-finite output: {out}"