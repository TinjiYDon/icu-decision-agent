"""Train P0 LightGBM mortality model."""

from __future__ import annotations

import argparse

from domain.features.build import build_features
from domain.labels.mortality_12h import build_labels
from domain.models.lgbm import train_and_save


def _mlflow_metrics(meta: dict) -> dict[str, float]:
    out: dict[str, float] = {
        "auc_val": float(meta.get("auc_val") or 0),
        "auc_test": float(meta.get("auc_test") or 0),
        "pr_auc_val": float(meta.get("pr_auc_val") or 0),
        "pr_auc_test": float(meta.get("pr_auc_test") or 0),
        "brier_val": float(meta.get("brier_val") or 0),
        "brier_test": float(meta.get("brier_test") or 0),
        "pos_rate": float(meta.get("pos_rate") or 0),
        "positive": float(meta.get("positive") or 0),
        "operating_threshold": float(meta.get("operating_threshold") or 0),
    }
    by_hour = meta.get("metrics_by_hour_test") or {}
    if isinstance(by_hour, dict):
        for h, m in by_hour.items():
            if not isinstance(m, dict):
                continue
            if m.get("pr_auc") is not None:
                out[f"pr_auc_h{h}"] = float(m["pr_auc"])
            if m.get("brier") is not None:
                out[f"brier_h{h}"] = float(m["brier"])
            if m.get("roc_auc") is not None:
                out[f"roc_auc_h{h}"] = float(m["roc_auc"])
    return out


def run_train(*, rebuild_data: bool = True) -> dict:
    """Train the model, optionally reusing restored feat/label tables."""
    meta = {"data_mode": "rebuild_from_layer0" if rebuild_data else "existing_feat_label"}
    if rebuild_data:
        meta.update(build_features())
        meta.update(build_labels())
    meta.update(train_and_save())
    meta["status"] = "train_ok"
    try:
        from infra.mlflow_util import log_run

        rid = log_run(
            "icu-decision",
            "train_mortality_12h",
            {
                "feat_rows": meta.get("feat_rows") or meta.get("total_n"),
                "label_rows": meta.get("label_rows") or meta.get("total_n"),
                "train_n": meta.get("train_n"),
                "val_n": meta.get("val_n"),
                "test_n": meta.get("test_n"),
                "data_mode": meta.get("data_mode"),
                "stratified": meta.get("stratified"),
                "prediction_hours": meta.get("prediction_hours"),
            },
            _mlflow_metrics(meta),
        )
        if rid:
            meta["mlflow_run_id"] = rid
    except Exception as exc:  # noqa: BLE001 — tracking must not break train
        meta["mlflow_error"] = str(exc)
    return meta


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="reuse feat.sample_matrix and label.mortality_12h from a restored full dump",
    )
    args = parser.parse_args()
    print(json.dumps(run_train(rebuild_data=not args.from_existing), indent=2, ensure_ascii=False))
