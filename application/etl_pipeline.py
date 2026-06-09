"""P0 ETL entry (stub until MIMIC Layer0 ready)."""

from __future__ import annotations

from infra.config import get_data_phase, get_data_source, get_mimic_source


def run_etl() -> dict:
    """Run ETL pipeline. Full impl after MIMIC import."""
    meta = {
        "phase": get_data_phase(),
        "source": get_data_source(),
        "mimic_source": get_mimic_source() if get_data_source() == "mimic" else None,
        "status": "stub",
        "message": "MIMIC Layer0 not ready; implement domain/etl after import",
    }
    if get_data_source() == "mock":
        from data_access.mimic_repo import count_icustays

        meta["mock_icustays"] = count_icustays()
        meta["status"] = "mock_ok"
    return meta


if __name__ == "__main__":
    import json

    print(json.dumps(run_etl(), indent=2, ensure_ascii=False))
