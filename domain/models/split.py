"""Train / val / test split by stay_id (no val+test merge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

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
    missing = {stay_col, label_col}.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    cfg = load_yaml("labels.yaml").get("split", {})
    train_r = float(cfg.get("train", 0.7))
    val_r = float(cfg.get("val", 0.1))
    test_r = float(cfg.get("test", 0.2))
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    seed = int(cfg.get("seed", 42) if seed is None else seed)

    label_counts = df.groupby(stay_col, dropna=False)[label_col].nunique(dropna=False)
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        raise ValueError(f"{len(conflicting)} stays have conflicting labels")

    stays = df[[stay_col, label_col]].drop_duplicates(stay_col).reset_index(drop=True)
    if stays[label_col].isna().any():
        raise ValueError("labels must not be null")
    unexpected_labels = set(stays[label_col].unique()).difference({0, 1})
    if unexpected_labels:
        raise ValueError(f"labels must be binary 0/1, got {sorted(unexpected_labels)}")
    n = len(stays)
    if n < 3:
        raise ValueError("at least 3 unique stays are required for train/val/test split")
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    class_counts = stays[label_col].value_counts()
    can_stratify = (
        len(class_counts) > 1
        and int(class_counts.min()) >= 3
        and min(n_train, n_val, n_test) >= len(class_counts)
    )
    if can_stratify:
        outer = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_train,
            test_size=n_val + n_test,
            random_state=seed,
        )
        train_idx, holdout_idx = next(outer.split(stays[[stay_col]], stays[label_col]))
        holdout = stays.iloc[holdout_idx].reset_index(drop=True)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_val,
            test_size=n_test,
            random_state=seed + 1,
        )
        val_rel, test_rel = next(inner.split(holdout[[stay_col]], holdout[label_col]))
        train_ids = set(stays.iloc[train_idx][stay_col].tolist())
        val_ids = set(holdout.iloc[val_rel][stay_col].tolist())
        test_ids = set(holdout.iloc[test_rel][stay_col].tolist())
    else:
        shuffled = stays.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_ids = set(shuffled.iloc[:n_train][stay_col].tolist())
        val_ids = set(shuffled.iloc[n_train : n_train + n_val][stay_col].tolist())
        test_ids = set(shuffled.iloc[n_train + n_val :][stay_col].tolist())

    train_df = df[df[stay_col].isin(train_ids)].copy()
    val_df = df[df[stay_col].isin(val_ids)].copy()
    test_df = df[df[stay_col].isin(test_ids)].copy()

    # Disjoint check
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)

    frames = {
        "train": train_df[[stay_col, label_col]].drop_duplicates(stay_col),
        "val": val_df[[stay_col, label_col]].drop_duplicates(stay_col),
        "test": test_df[[stay_col, label_col]].drop_duplicates(stay_col),
    }
    split_class_counts = {
        name: {
            "negative": int((frame[label_col] == 0).sum()),
            "positive": int((frame[label_col] == 1).sum()),
        }
        for name, frame in frames.items()
    }
    positive_rate = {
        name: float(frame[label_col].mean()) if len(frame) else None
        for name, frame in frames.items()
    }
    manifest = {
        "seed": seed,
        "stratified": can_stratify,
        "ratios": {"train": train_r, "val": val_r, "test": test_r},
        "n_stays": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "class_counts": split_class_counts,
        "positive_rate": positive_rate,
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
        "stratified": manifest.get("stratified", False),
        "ratios": manifest["ratios"],
        "n_stays": manifest["n_stays"],
        "class_counts": manifest.get("class_counts", {}),
        "positive_rate": manifest.get("positive_rate", {}),
        "assignment": manifest["assignment"],
    }
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
