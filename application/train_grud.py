"""GRU-D CLI: synthetic smoke (default) or real Layer0 train (--real)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from domain.features.sequence_build import apply_recency_weights, build_mask_delta, pad_truncate
from domain.models.temporal.grud import smoke_grud_batch
from infra.config import load_yaml


def _smoke(batch: int) -> dict:
    cfg = load_yaml("temporal.yaml").get("temporal", {})
    F = len(cfg.get("feature_keys", ["hr", "sbp"]))
    max_t = int(cfg.get("max_timesteps", 24))
    xs, ms, ds = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(batch):
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
    return {"status": "grud_smoke_ok", **{k: out[k] for k in ("n", "prob_mean")}}


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU-D smoke or real training")
    parser.add_argument("--batch", type=int, default=8, help="synthetic batch size for smoke")
    parser.add_argument(
        "--real",
        action="store_true",
        help="train PyTorch GRU-D on Layer0 sequences (same-split vs LGBM when manifest exists)",
    )
    args = parser.parse_args()
    if args.real:
        from domain.models.temporal.train_grud import train_grud

        result = train_grud()
        print(json.dumps({k: result[k] for k in result if k in (
            "model_path", "metrics_path", "roc_auc", "pr_auc", "brier", "split_mode", "test_n"
        )}, ensure_ascii=False, indent=2))
        return
    print(_smoke(args.batch))


if __name__ == "__main__":
    main()
