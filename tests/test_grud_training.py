# Test GRU-D training end-to-end (mock data path)
"""Tests for GRU-D training pipeline with mock data."""

import numpy as np
import pytest
import torch

from domain.models.temporal.grud_model import GRUD, build_train_dataset
from domain.models.temporal.attribution import gradient_input_attribution


def test_build_train_dataset():
    """Dataset builder produces correct tensor shapes (pads to max_timesteps)."""
    seqs = [
        {"x": np.ones((6, 4)), "m": np.ones((6, 4)), "delta": np.ones((6, 4)) * 0.5},
        {"x": np.zeros((6, 4)), "m": np.zeros((6, 4)), "delta": np.zeros((6, 4))},
    ]
    labels = [1, 0]
    X, M, D, Y = build_train_dataset(seqs, labels)
    # pad_truncate pads to max_timesteps (default 12 from config, but test uses mock)
    assert X.shape[0] == 2
    assert X.shape[2] == 4  # F=4


def test_grud_mini_training():
    """Can train for 1 epoch on tiny mock dataset."""
    seqs = [
        {"x": np.random.randn(6, 4), "m": np.ones((6, 4)), "delta": np.ones((6, 4)) * 0.5}
        for _ in range(20)
    ]
    labels = [1 if i % 5 == 0 else 0 for i in range(20)]
    X, M, D, Y = build_train_dataset(seqs, labels)

    model = GRUD(input_size=4, hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(X, M, D)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

    assert not any(torch.isnan(p).any() for p in model.parameters())


def test_attribution_returns_dict():
    """gradient_input_attribution returns expected structure."""
    model = GRUD(input_size=4, hidden_size=8)
    model.eval()
    x = np.random.randn(6, 4)
    m = np.ones((6, 4))
    delta = np.ones((6, 4)) * 0.5
    result = gradient_input_attribution(model, x, m, delta)
    assert isinstance(result, dict)
    assert "feature_importance" in result
    assert "top_features" in result
    assert len(result["feature_importance"]) == 4
    assert len(result["top_features"]) == 4
