from __future__ import annotations

import torch

from domain.models.temporal.tft_lite import TFTLite


def test_tft_lite_logits_shape():
    model = TFTLite(input_size=4, d_model=16, nhead=4, num_layers=1)
    x = torch.randn(3, 6, 4)
    m = torch.ones(3, 6, 4)
    m[0, 4:] = 0
    d = torch.ones(3, 6, 4)
    logits = model(x, m, d)
    assert logits.shape == (3,)
    loss = logits.sum()
    loss.backward()


def test_tft_lite_smoke_metrics():
    from domain.models.temporal.tft_smoke import run_smoke

    payload = run_smoke(epochs=2, write=False)
    assert payload["status"] == "ok"
    assert payload["n_test"] > 0
    assert payload["primary_metric"] == "pr_auc"


def test_tft_lite_rejects_bad_heads():
    try:
        TFTLite(input_size=2, d_model=10, nhead=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
