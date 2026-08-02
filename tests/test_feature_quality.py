from domain.models.lgbm import _feature_quality


def test_placeholder_quality():
    q = _feature_quality({"los_hours": 1.2, "first_careunit": "MICU"})
    assert q["is_placeholder"] is True
    assert q["usable"] is False


def test_usable_lab_heavy_row():
    q = _feature_quality(
        {
            "anchor_age": 67,
            "lab_lactate": 1.2,
            "lab_creatinine": 1.1,
            "lab_hematocrit": 35.0,
            "vital_heart_rate": None,
        }
    )
    assert q["is_placeholder"] is False
    assert q["usable"] is True
    assert q["lab_present"] >= 2
