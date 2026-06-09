def test_etl_mock():
    from application.etl_pipeline import run_etl

    r = run_etl()
    assert r["status"] == "mock_ok"
    assert r["mock_icustays"] >= 1


def test_config_phase():
    from infra.config import get_data_phase

    assert get_data_phase() == "P0"
