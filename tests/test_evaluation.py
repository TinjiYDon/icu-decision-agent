import pytest

from domain.models.evaluation import binary_metrics, select_threshold_by_f1


def test_binary_metrics_include_imbalance_and_calibration_outputs():
    metrics = binary_metrics(
        [0, 0, 0, 1, 1],
        [0.05, 0.10, 0.40, 0.60, 0.90],
        threshold=0.5,
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}
    assert len(metrics["calibration"]["mean_predicted"]) > 0


def test_threshold_is_selected_from_validation_probabilities():
    threshold = select_threshold_by_f1(
        [0, 0, 0, 1, 1],
        [0.05, 0.10, 0.40, 0.60, 0.90],
    )
    assert 0.4 < threshold <= 0.6


def test_binary_metrics_reject_invalid_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        binary_metrics([0, 1], [0.2, 1.2])
