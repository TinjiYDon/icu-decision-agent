# Test GRU-D attribution output format
"""Tests for gradient-based attribution module."""

import numpy as np
import pytest
import torch

from domain.models.temporal.attribution import FEATURE_NAMES, gradient_input_attribution
from domain.models.temporal.grud_model import GRUD


def test_attribution_shape():
    """Attribution output matches input dimensions."""
    model = GRUD(input_size=len(FEATURE_NAMES), hidden_size=16)
    T, F = 12, len(FEATURE_NAMES)
    x = np.random.randn(T, F)
    m = (np.random.random((T, F)) > 0.3).astype(np.float64)
    delta = np.abs(np.random.randn(T, F)) * 0.5
    result = gradient_input_attribution(model, x, m, delta)
    assert result["per_timestep"].shape == (T, F)
    assert set(result["feature_importance"].keys()) == set(FEATURE_NAMES)


def test_attribution_zero_input():
    """Attribution on all-zero input should not crash."""
    model = GRUD(input_size=4, hidden_size=8)
    x = np.zeros((6, 4))
    m = np.zeros((6, 4))
    delta = np.ones((6, 4))
    result = gradient_input_attribution(model, x, m, delta)
    assert isinstance(result["feature_importance"], dict)
    assert all(isinstance(v, float) for v in result["feature_importance"].values())


def test_attribution_top_features_sorted():
    """top_features should be sorted by importance descending."""
    model = GRUD(input_size=4, hidden_size=8)
    x = np.random.randn(6, 4)
    m = np.ones((6, 4))
    delta = np.ones((6, 4)) * 0.5
    result = gradient_input_attribution(model, x, m, delta)
    vals = [f["value"] for f in result["top_features"]]
    assert vals == sorted(vals, reverse=True)
