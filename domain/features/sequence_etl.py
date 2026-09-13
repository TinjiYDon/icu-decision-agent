"""Sequence ETL helpers: sparse filters + optional on-disk cache (D1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from domain.features.sequence_build import load_temporal_cfg
from domain.models.temporal.attribution import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]


def feature_names() -> list[str]:
    cfg = load_temporal_cfg()
    keys = cfg.get("feature_keys")
    if keys:
        return [str(k) for k in keys]
    return list(FEATURE_NAMES)


def passes_sparse_gate(mask: np.ndarray, *, feature_names_list: list[str] | None = None) -> bool:
    """Reject near-empty sequences per temporal.yaml sparse thresholds."""
    cfg = load_temporal_cfg()
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2 or m.size == 0:
        return False
    min_cells = int(cfg.get("min_observed_cells", 3))
    min_ratio = float(cfg.get("min_obs_ratio", 0.05))
    observed = float(m.sum())
    if observed < min_cells:
        return False
    if observed / float(m.size) < min_ratio:
        return False
    if cfg.get("require_any_vital", True):
        names = feature_names_list or feature_names()
        vital_keys = {str(k) for k in cfg.get("vital_keys", ["hr", "sbp", "spo2", "resp_rate", "temperature"])}
        vital_idx = [i for i, n in enumerate(names) if n in vital_keys]
        if vital_idx and float(m[:, vital_idx].sum()) < 1.0:
            return False
    return True


def cache_dir() -> Path:
    cfg = load_temporal_cfg()
    rel = str(cfg.get("sequence_cache_dir", "artifacts/sequences"))
    path = ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_sequence_manifest(
    stay_ids: list[int],
    *,
    n_valid: int,
    n_dropped_sparse: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a small JSON manifest (no large tensors in Git)."""
    cfg = load_temporal_cfg()
    payload = {
        "n_requested": len(stay_ids),
        "n_valid": int(n_valid),
        "n_dropped_sparse": int(n_dropped_sparse),
        "lookback_hours": cfg.get("lookback_hours"),
        "feature_keys": feature_names(),
        "min_obs_ratio": cfg.get("min_obs_ratio"),
        "min_observed_cells": cfg.get("min_observed_cells"),
        **(extra or {}),
    }
    path = cache_dir() / "sequence_etl_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
