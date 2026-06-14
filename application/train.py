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
    return meta


if __name__ == "__main__":
    import json

    print(json.dumps(run_train(), indent=2, ensure_ascii=False))
