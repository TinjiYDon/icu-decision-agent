"""TFT-lite: temporal attention over irregular EHR steps (ablation, not full TFT).

Uses the same (x, mask, delta) tensors as GRU-D. This is a small Transformer
encoder over time — not Lim et al. 2021 Temporal Fusion Transformer (no
static covariates, no VSN, no quantile heads). Default bedside model remains
LightGBM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_positions(n_steps: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard transformer positional encodings (T, d_model)."""
    pos = torch.arange(n_steps, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_steps, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TFTLite(nn.Module):
    """Observed-value + mask + log-delta → d_model → TransformerEncoder → logit."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.input_size = input_size
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size * 3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits of shape (B,)."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = m.nan_to_num(nan=0.0).clamp(0.0, 1.0)
        # log1p(inf) -> inf -> poisons input_proj; clamp to the same finite
        # sentinel used by sequence_build.pad_truncate.
        delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0)
        feat = torch.cat([x * m, m, torch.log1p(delta)], dim=-1)
        h = self.input_proj(feat)
        # Order-aware: without PE, mean-pooled Transformer is permutation-invariant.
        h = h + _sinusoidal_positions(h.size(1), self.d_model, h.device).unsqueeze(0)
        observed = m.sum(dim=-1) > 0
        pad_mask = ~observed
        all_pad = pad_mask.all(dim=1)
        if bool(all_pad.any()):
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False
        enc = self.encoder(h, src_key_padding_mask=pad_mask)
        weights = observed.float().unsqueeze(-1)
        pooled = (enc * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(self.norm(pooled)).squeeze(-1)
