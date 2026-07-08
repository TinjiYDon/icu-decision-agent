"""Data pipeline checkpoint: ETL only (stops before model train/simulate)."""

from __future__ import annotations

from application.etl_pipeline import run_etl


def run_etl_stage() -> dict:
    result = run_etl()
    result["checkpoint"] = "etl_only"
    result["next"] = "application.train / application.run_p0"
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run_etl_stage(), indent=2, ensure_ascii=False))
