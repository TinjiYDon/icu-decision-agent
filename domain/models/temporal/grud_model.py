"""PyTorch GRU-D model (Che et al. 2017, Scientific Reports).

Replaces the numpy smoke test in grud.py.  Produces proper gradients
for attribution and supports GPU training via DataLoader.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUD(nn.Module):
    """Gated Recurrent Unit with Decay for irregular clinical time series.

    Input shape:  (B, T, F)
      x : observed values (NaN → masked)
      m : observation mask  (1 = observed, 0 = missing)
      d : time delta since last observation (hours)

    Output: probability of mortality in (B,)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable imputation mean
        self.x_mean = nn.Parameter(torch.zeros(input_size))

        # Input-decay parameters: γ_x = exp(-max(0, W·δ + b))
        self.gamma_x_w = nn.Parameter(torch.zeros(input_size))
        self.gamma_x_b = nn.Parameter(torch.zeros(input_size))

        # Hidden-state-decay parameters: γ_h = exp(-max(0, W·δ̄ + b))
        self.gamma_h_w = nn.Parameter(torch.zeros(hidden_size))
        self.gamma_h_b = nn.Parameter(torch.zeros(hidden_size))

        # GRU gates
        gate_input = input_size + hidden_size
        self.W_z = nn.Linear(gate_input, hidden_size)   # update gate
        self.W_r = nn.Linear(gate_input, hidden_size)   # reset gate
        self.W_h = nn.Linear(gate_input, hidden_size)   # new state

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def _decay(self, w: torch.Tensor, b: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """γ = exp(-max(0, W·delta + b)), delta shape must broadcast with (w, b)."""
        log_decay = w.unsqueeze(0) * delta + b.unsqueeze(0)
        return torch.exp(-torch.clamp(log_decay, min=0.0))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, T, F)  raw values (NaN → masked by m)
            m:      (B, T, F)  observation mask (1 = observed)
            delta:  (B, T, F)  hours since last observation per feature
        Returns:
            prob:   (B,) prediction probability
        """
        B, T, F = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_size, device=device)
        x_hat = self.x_mean.unsqueeze(0).expand(B, -1)  # (B, F) previous imputed value

        for t in range(T):
            x_t = x[:, t, :]       # (B, F)
            m_t = m[:, t, :]       # (B, F)
            d_t = delta[:, t, :]   # (B, F)

            # ── Input decay ──────────────────────────────────────────
            gamma_x = self._decay(self.gamma_x_w, self.gamma_x_b, d_t)  # (B, F)
            x_imputed = (
                m_t * x_t
                + (1.0 - m_t) * (gamma_x * x_hat + (1.0 - gamma_x) * self.x_mean)
            )

            # ── Hidden-state decay ───────────────────────────────────
            d_bar = d_t.mean(dim=-1, keepdim=True)  # (B, 1) avg delta
            gamma_h = self._decay(self.gamma_h_w, self.gamma_h_b, d_bar)  # (B, H)
            h_decayed = gamma_h * h

            # ── GRU gates ────────────────────────────────────────────
            combined = torch.cat([x_imputed, h_decayed], dim=-1)  # (B, F+H)
            z = torch.sigmoid(self.W_z(combined))   # update gate
            r = torch.sigmoid(self.W_r(combined))   # reset gate
            h_tilde = torch.tanh(self.W_h(
                torch.cat([x_imputed, r * h_decayed], dim=-1)
            ))
            h = (1.0 - z) * h + z * h_tilde

            # Update running mean for next step
            x_hat = torch.where(m_t > 0.5, x_t, x_hat)

        return torch.sigmoid(self.classifier(h)).squeeze(-1)


def build_train_dataset(
    sequences: list[dict[str, Any]],
    labels: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw sequence dicts to tensors.

    Each dict has keys 'x', 'm', 'delta' as np.ndarray of shape (T, F).
    Pads/truncates to max_timesteps using sequence_build.pad_truncate.
    """
    from domain.features.sequence_build import pad_truncate

    cfg = {}
    try:
        from infra.config import load_yaml
        cfg = load_yaml("temporal.yaml").get("temporal", {})
    except Exception:  # noqa: BLE001
        pass
    max_t = int(cfg.get("max_timesteps", 12))

    xs, ms, ds = [], [], []
    for seq in sequences:
        x_np = seq["x"]
        m_np = seq["m"]
        d_np = seq["delta"]
        xp, mp, dp = pad_truncate(x_np, m_np, d_np, max_t)
        xs.append(xp)
        ms.append(mp)
        ds.append(dp)

    X = torch.FloatTensor(xs)   # (N, T, F)
    M = torch.FloatTensor(ms)   # (N, T, F)
    D = torch.FloatTensor(ds)   # (N, T, F)
    Y = torch.FloatTensor(labels)  # (N,)
    return X, M, D, Y
