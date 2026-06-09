def test_labels_config():
    from infra.config import load_yaml

    labels = load_yaml("labels.yaml")
    assert labels["primary"]["horizon_hours"] == 12
    assert labels["primary"]["name"] == "mortality_12h"
