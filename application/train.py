"""Train P0 LightGBM mortality model."""

from __future__ import annotations

from domain.features.build import build_features
from domain.labels.mortality_12h import build_labels
from domain.models.lgbm import train_and_save


def run_train() -> dict:
    meta = {}
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
                "feat_rows": meta.get("feat_rows"),
                "label_rows": meta.get("label_rows"),
                "train_n": meta.get("train_n"),
                "val_n": meta.get("val_n"),
                "test_n": meta.get("test_n"),
            },
            {
                "auc_val": float(meta.get("auc_val") or 0),
                "auc_test": float(meta.get("auc_test") or 0),
                "pos_rate": float(meta.get("pos_rate") or 0),
                "positive": float(meta.get("positive") or 0),
            },
        )
        if rid:
            meta["mlflow_run_id"] = rid
    except Exception as exc:  # noqa: BLE001 — tracking must not break train
        meta["mlflow_error"] = str(exc)
    return meta


if __name__ == "__main__":
    import json

    print(json.dumps(run_train(), indent=2, ensure_ascii=False))
