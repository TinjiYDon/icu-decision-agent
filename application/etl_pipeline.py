"""P0 ETL entry — Layer0/mock → staging + feat."""

from __future__ import annotations

from infra.config import get_data_phase, get_data_source, get_mimic_source


def run_etl() -> dict:
    source = get_data_source()
    meta: dict = {
        "phase": get_data_phase(),
        "source": source,
        "mimic_source": get_mimic_source() if source == "mimic" else None,
    }
    from domain.etl.pipeline import run_pipeline

    stats = run_pipeline()
    meta.update(stats)
    meta["icustays"] = stats["staging_icustays"]
    meta["status"] = "mimic_ok" if source == "mimic" else "mock_ok"
    return meta


if __name__ == "__main__":
    import json

    print(json.dumps(run_etl(), indent=2, ensure_ascii=False))
