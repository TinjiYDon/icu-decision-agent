"""Train / val / test split by stay_id (no val+test merge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from infra.config import load_yaml

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "models"


def split_frame_by_stay(
    df: pd.DataFrame,
    *,
    stay_col: str = "stay_id",
    label_col: str = "label",
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return train, val, test frames and manifest meta.

    Ratios from configs/labels.yaml split (default 0.7/0.1/0.2).
    Stratify on label when possible at stay level.
    """
    if stay_col not in df.columns:
        raise ValueError(f"missing {stay_col}")
    cfg = load_yaml("labels.yaml").get("split", {})
    train_r = float(cfg.get("train", 0.7))
    val_r = float(cfg.get("val", 0.1))
    test_r = float(cfg.get("test", 0.2))
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    seed = int(cfg.get("seed", 42) if seed is None else seed)

    stays = df[[stay_col, label_col]].drop_duplicates(stay_col)

    # Stratified shuffle of stay ids when both classes exist
    parts: list[pd.DataFrame] = []
    for _, grp in stays.groupby(label_col):
        idx = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        parts.append(idx)
    shuffled = pd.concat(parts, ignore_index=True)
    # Re-shuffle blocks so order is not label-sorted
    shuffled = shuffled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val

    train_ids = set(shuffled.iloc[:n_train][stay_col].tolist())
    val_ids = set(shuffled.iloc[n_train : n_train + n_val][stay_col].tolist())
    test_ids = set(shuffled.iloc[n_train + n_val :][stay_col].tolist())

    train_df = df[df[stay_col].isin(train_ids)].copy()
    val_df = df[df[stay_col].isin(val_ids)].copy()
    test_df = df[df[stay_col].isin(test_ids)].copy()

    # Disjoint check
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)

    manifest = {
        "seed": seed,
        "ratios": {"train": train_r, "val": val_r, "test": test_r},
        "n_stays": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "assignment": {
            **{str(s): "train" for s in train_ids},
            **{str(s): "val" for s in val_ids},
            **{str(s): "test" for s in test_ids},
        },
    }
    return train_df, val_df, test_df, manifest


def save_split_manifest(manifest: dict[str, Any], name: str = "split_manifest_mortality_12h.json") -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    # Store compact assignment
    slim = {
        "seed": manifest["seed"],
        "ratios": manifest["ratios"],
        "n_stays": manifest["n_stays"],
        "assignment": manifest["assignment"],
    }
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
