# Test GRU-D model forward pass and shapes
"""Tests for PyTorch GRU-D model."""

import numpy as np
import pytest
import torch

from domain.models.temporal.grud_model import GRUD


def test_grud_forward_shapes():
    """Basic forward pass shape check."""
    model = GRUD(input_size=8, hidden_size=16, dropout=0.1)
    B, T, F = 4, 12, 8
    x = torch.randn(B, T, F)
    m = torch.ones(B, T, F)
    delta = torch.ones(B, T, F) * 0.5
    out = model(x, m, delta)
    assert out.shape == (B,)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid output


def test_grud_missing_value_handling():
    """Model should handle all-zero mask gracefully."""
    model = GRUD(input_size=4, hidden_size=8)
    B, T, F = 2, 6, 4
    x = torch.randn(B, T, F)
    m = torch.zeros(B, T, F)  # all missing
    delta = torch.ones(B, T, F) * 1.0
    out = model(x, m, delta)
    assert out.shape == (B,)
    # Should not NaN or Inf
    assert torch.isfinite(out).all()


def test_grud_zero_delta():
    """When delta=0 (consecutive observations), decay should be 1."""
    model = GRUD(input_size=4, hidden_size=8)
    B, T, F = 2, 6, 4
    x = torch.randn(B, T, F)
    m = torch.ones(B, T, F)
    delta = torch.zeros(B, T, F)
    out = model(x, m, delta)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def test_grud_gradient_flow():
    """Model must produce gradients for attribution."""
    model = GRUD(input_size=4, hidden_size=8)
    x = torch.randn(1, 6, 4, requires_grad=True)
    m = torch.ones(1, 6, 4)
    delta = torch.ones(1, 6, 4) * 0.5
    out = model(x, m, delta)
    out.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()
