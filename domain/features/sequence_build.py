"""Sequence tensor helpers for irregular EHR → GRU-D (mask + Δt)."""

from __future__ import annotations

from typing import Any

import numpy as np

from infra.config import load_yaml


def load_temporal_cfg() -> dict[str, Any]:
    return load_yaml("temporal.yaml").get("temporal", {})


def build_mask_delta(
    values: np.ndarray,
    times_hours: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From irregular observations build (X, M, delta).

    values: (T, F) with NaN for missing
    times_hours: (T,) absolute hours from prediction anchor (usually increasing)

    Returns:
      X: forward-filled values (T, F)
      M: mask 1 if observed (T, F)
      delta: hours since last observation per feature (T, F)
    """
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times_hours, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be (T, F)")
    T, F = x.shape
    m = (~np.isnan(x)).astype(np.float64)
    x_ff = x.copy()
    delta = np.zeros_like(x_ff)
    last_t = np.full(F, t[0] if T else 0.0, dtype=np.float64)
    last_val = np.zeros(F, dtype=np.float64)
    for i in range(T):
        dt = float(t[i] - (t[i - 1] if i else t[i]))
        for f in range(F):
            if i == 0:
                delta[i, f] = 0.0
            else:
                delta[i, f] = (t[i] - last_t[f]) if m[i - 1, f] < 0.5 else dt
            if m[i, f] > 0.5:
                last_val[f] = x[i, f]
                last_t[f] = t[i]
                x_ff[i, f] = x[i, f]
            else:
                x_ff[i, f] = last_val[f] if i else 0.0
                if i:
                    delta[i, f] = t[i] - last_t[f]
    return x_ff, m, delta


def apply_recency_weights(times_hours: np.ndarray, lam: float) -> np.ndarray:
    """Weights emphasizing recent timesteps: exp(-lam * age_hours)."""
    t = np.asarray(times_hours, dtype=np.float64)
    if t.size == 0:
        return t
    age = t[-1] - t
    if lam <= 0:
        return np.ones_like(age)
    return np.exp(-float(lam) * np.maximum(age, 0.0))


def pad_truncate(
    x: np.ndarray, m: np.ndarray, d: np.ndarray, max_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, F = x.shape
    if T >= max_t:
        return x[-max_t:], m[-max_t:], d[-max_t:]
    pad = max_t - T
    return (
        np.vstack([np.zeros((pad, F)), x]),
        np.vstack([np.zeros((pad, F)), m]),
        np.vstack([np.zeros((pad, F)), d]),
    )
