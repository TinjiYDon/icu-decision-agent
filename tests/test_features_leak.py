from domain.features.build import FEATURE_COLS, row_to_features
from domain.models.split import split_frame_by_stay
from infra.config import load_yaml
import pandas as pd


def test_feature_cols_have_no_leakage():
    denied = {"hospital_expire_flag", "los_hours", "dod", "outtime"}
    assert denied.isdisjoint(set(FEATURE_COLS))
    cfg = load_yaml("features.yaml")
    assert set(FEATURE_COLS) == set(cfg["allowed"])
    assert denied.issubset(set(cfg["denied"]))


def test_row_to_features_careunit():
    feat = row_to_features(
        {"anchor_age": 70, "gender": "F", "first_careunit": "Medical Intensive Care Unit (MICU)"}
    )
    assert feat["gender_m"] == 0
    assert feat["careunit_micu"] == 1
    assert "los_hours" not in feat
    assert "hospital_expire_flag" not in feat


def test_split_three_folds_disjoint():
    rows = []
    for i in range(100):
        rows.append({"stay_id": i, "label": i % 2, "anchor_age": 60, "gender_m": 1,
                     "careunit_micu": 1, "careunit_sicu": 0, "careunit_ccu": 0, "careunit_other": 0})
    df = pd.DataFrame(rows)
    train, val, test, meta = split_frame_by_stay(df, seed=42)
    ids = (
        set(train["stay_id"]),
        set(val["stay_id"]),
        set(test["stay_id"]),
    )
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    n = len(df)
    assert abs(len(ids[0]) / n - 0.7) < 0.08
    assert abs(len(ids[1]) / n - 0.1) < 0.08
    assert abs(len(ids[2]) / n - 0.2) < 0.08
    assert meta["seed"] == 42
