"""Sequence tensor helpers for irregular EHR → GRU-D (mask + Δt)."""

from __future__ import annotations

from typing import Any

import numpy as np

from infra.config import load_yaml

# Finite stand-in for "never observed" when right-padding delta.
# Must stay well below float32 overflow and well above any real gap in hours.
PAD_DELTA = 1e4


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
    """Right-pad short sequences to `max_t`.

    Padding convention: real observations keep their order at the FRONT, and
    zero/inf rows are appended at the END. Callers that build per-timestep
    labels must follow the same convention (see `predict_grud.time_labels`).

    delta padding uses a large finite sentinel rather than ``inf``: downstream
    decay computes ``w * delta`` with learnable ``w`` initialised to 0, and
    ``0 * inf`` yields NaN, which silently poisons the whole forward pass.
    ``PAD_DELTA`` is far beyond any real inter-observation gap, so
    ``exp(-max(0, w * PAD_DELTA))`` still saturates the decay to ~0 exactly as
    ``inf`` would, while staying numerically safe.
    """
    T, F = x.shape
    if T >= max_t:
        return x[-max_t:], m[-max_t:], d[-max_t:]
    pad = max_t - T
    return (
        np.vstack([x, np.zeros((pad, F))]),
        np.vstack([m, np.zeros((pad, F))]),
        np.vstack([d, np.full((pad, F), PAD_DELTA)]),
    )
