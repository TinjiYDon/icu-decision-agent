"""Dry-run GRU-D shape contract (no Layer0 required)."""

from __future__ import annotations

import argparse

import numpy as np

from domain.features.sequence_build import apply_recency_weights, build_mask_delta, pad_truncate
from domain.models.temporal.grud import smoke_grud_batch
from infra.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU-D smoke (synthetic sequences)")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    cfg = load_yaml("temporal.yaml").get("temporal", {})
    F = len(cfg.get("feature_keys", ["hr", "sbp"]))
    max_t = int(cfg.get("max_timesteps", 24))
    xs, ms, ds = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(args.batch):
        T = int(rng.integers(4, max_t + 1))
        raw = rng.normal(0, 1, size=(T, F))
        raw[rng.random(size=(T, F)) < 0.3] = np.nan
        times = np.linspace(0, float(cfg.get("lookback_hours", 6)), T)
        x, m, d = build_mask_delta(raw, times)
        _ = apply_recency_weights(times, float(cfg.get("recency_lambda", 0.0)))
        x, m, d = pad_truncate(x, m, d, max_t)
        xs.append(x)
        ms.append(m)
        ds.append(d)
    out = smoke_grud_batch(np.stack(xs), np.stack(ms), np.stack(ds))
    print({"status": "grud_smoke_ok", **{k: out[k] for k in ("n", "prob_mean")}})


if __name__ == "__main__":
    main()
