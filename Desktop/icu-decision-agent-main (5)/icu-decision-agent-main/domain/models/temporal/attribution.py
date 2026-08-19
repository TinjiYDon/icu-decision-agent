"""Gradient × Input attribution for GRU-D.

Produces feature-level importance scores and per-timestep attribution,
mirroring the interpretability contract expected by the explain page.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


FEATURE_NAMES = ["hr", "sbp", "lactate", "creatinine", "resp_rate", "temperature", "spo2", "bun"]


def gradient_input_attribution(
    model: torch.nn.Module,
    x: np.ndarray,
    m: np.ndarray,
    delta: np.ndarray,
) -> dict[str, Any]:
    """Compute Gradient×Input attribution for a single sample.

    Args:
        model:   trained GRUD instance (eval mode)
        x:       (T, F) forward-filled values
        m:       (T, F) observation mask
        delta:   (T, F) time-since-last-observation
    Returns:
        {
            "feature_importance": {name: float},  # mean |grad·x| per feature
            "per_timestep": np.ndarray (T, F),     # |grad·x| at each timestep
            "top_features": list[dict],            # sorted top factors
        }
    """
    model.eval()
    X = torch.FloatTensor(x).unsqueeze(0).requires_grad_(True)
    M = torch.FloatTensor(m).unsqueeze(0)
    D = torch.FloatTensor(delta).unsqueeze(0)
    pred = model(X, M, D)
    pred.backward()

    grad = X.grad  # (1, T, F)
    attr = torch.abs(grad * X).squeeze(0).detach().numpy()  # (T, F)

    # Per-feature importance = mean across timesteps
    feat_imp = {
        FEATURE_NAMES[i]: float(attr[:, i].mean())
        for i in range(attr.shape[1])
    }

    # Build top_factors aligned with LightGBM output format
    # Sort by mean importance, keep correct feature names and values
    feat_means = {FEATURE_NAMES[i]: float(attr[:, i].mean()) for i in range(attr.shape[1])}
    top = sorted(feat_means.items(), key=lambda kv: kv[1], reverse=True)
    top_factors = [
        {"feature": name, "value": round(val, 4)}
        for name, val in top
    ]

    return {
        "feature_importance": feat_imp,
        "per_timestep": attr,
        "top_features": top_factors,
    }
