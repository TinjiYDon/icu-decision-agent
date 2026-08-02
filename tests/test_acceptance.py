def test_acceptance_helpers_import():
    from application.acceptance import EXPECTED_FEAT_ROWS, load_metrics_artifact

    assert EXPECTED_FEAT_ROWS == 472290
    # May be None in CI without artifacts; must not raise
    load_metrics_artifact()
