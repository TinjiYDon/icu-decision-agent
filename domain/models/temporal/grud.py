"""GRU-D style cell (Che et al.): mask + time-decay into GRU gates.

Research track — train only after sequence dump is available.
LightGBM + TreeSHAP remains the deployable explanation path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GRUDConfig:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout


def grud_forward_numpy(
    x: np.ndarray,
    m: np.ndarray,
    delta: np.ndarray,
    *,
    hidden_size: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Differentiable-free smoke forward for CI without torch.

    Implements the *shape contract* of GRU-D:
      x_hat = m * x + (1-m) * (gamma_x * x_prev + (1-gamma_x) * x_mean)
      h decays with gamma_h(delta) then standard GRU-like update (linear approx).

    Returns last hidden state (H,).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    T, F = x.shape
    # trainable-like decay rates (fixed random for smoke)
    alpha_x = np.abs(rng.normal(0.1, 0.05, size=F)) + 1e-3
    alpha_h = np.abs(rng.normal(0.1, 0.05, size=hidden_size)) + 1e-3
    x_mean = np.nanmean(np.where(m > 0.5, x, np.nan), axis=0)
    x_mean = np.nan_to_num(x_mean, nan=0.0)
    W = rng.normal(0, 0.1, size=(F, hidden_size))
    h = np.zeros(hidden_size)
    x_prev = x_mean.copy()
    for t in range(T):
        gamma_x = np.exp(-alpha_x * np.maximum(delta[t], 0.0))
        x_hat = m[t] * x[t] + (1.0 - m[t]) * (gamma_x * x_prev + (1.0 - gamma_x) * x_mean)
        gamma_h = np.exp(-alpha_h * float(np.mean(delta[t])))
        h = gamma_h * h
        h = np.tanh(h + x_hat @ W)
        x_prev = np.where(m[t] > 0.5, x[t], x_prev)
    return h


def predict_logit_from_hidden(h: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, size=h.shape[0])
    z = float(h @ w)
    return 1.0 / (1.0 + np.exp(-z))


def smoke_grud_batch(batch_x: np.ndarray, batch_m: np.ndarray, batch_d: np.ndarray) -> dict[str, Any]:
    """Run forward on a batch; used by tests / dry-run without GPU torch."""
    outs = []
    for i in range(batch_x.shape[0]):
        h = grud_forward_numpy(batch_x[i], batch_m[i], batch_d[i])
        outs.append(predict_logit_from_hidden(h, seed=i))
    probs = np.asarray(outs)
    return {"n": int(probs.shape[0]), "prob_mean": float(probs.mean()), "probs": probs.tolist()}
