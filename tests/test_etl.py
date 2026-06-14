def test_etl_smoke():
    from application.etl_pipeline import run_etl
    from infra.config import get_data_source

    r = run_etl()
    if get_data_source() == "mock":
        assert r["status"] == "mock_ok"
    else:
        assert r["status"] == "mimic_ok"
    assert r["staging_icustays"] >= 1
    assert r["icustays"] == r["staging_icustays"]


def test_config_phase():
    from infra.config import get_data_phase

    assert get_data_phase() == "P0"
