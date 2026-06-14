"""P0 end-to-end: ETL → features → labels → train."""

from __future__ import annotations

from application.etl_pipeline import run_etl
from application.train import run_train


def run_p0() -> dict:
    meta = {"steps": []}
    etl = run_etl()
    meta["steps"].append("etl")
    meta["etl"] = etl
    train = run_train()
    meta["steps"].append("train")
    meta["train"] = train
    meta["status"] = "p0_ok"
    return meta


if __name__ == "__main__":
    import json

    print(json.dumps(run_p0(), indent=2, ensure_ascii=False))
