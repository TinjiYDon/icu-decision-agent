"""LightGBM trainer for P0 tabular mortality model."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from infra.config import load_yaml
from infra.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "models"
FEATURE_COLS = ["anchor_age", "gender_m", "los_hours", "hospital_expire_flag"]


def _load_training_frame() -> pd.DataFrame:
    engine = get_engine()
    sql = """
        SELECT f.stay_id, f.feature_json, l.label
        FROM feat.sample_matrix f
        JOIN label.mortality_12h l ON f.stay_id = l.stay_id AND f.hour_index = l.hour_index
        WHERE f.hour_index = 0
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    records = []
    for row in rows:
        feat = row["feature_json"] if isinstance(row["feature_json"], dict) else json.loads(row["feature_json"])
        rec = {c: feat.get(c, 0) for c in FEATURE_COLS}
        rec["label"] = int(row["label"])
        rec["stay_id"] = row["stay_id"]
        records.append(rec)
    return pd.DataFrame(records)


def train_and_save() -> dict:
    df = _load_training_frame()
    if len(df) < 10:
        raise RuntimeError(f"too few samples for training: {len(df)}")

    split_cfg = load_yaml("labels.yaml").get("split", {})
    test_size = float(split_cfg.get("test", 0.2)) + float(split_cfg.get("val", 0.1))

    X = df[FEATURE_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    scale = neg / pos

    model = lgb.LGBMClassifier(
        n_estimators=64,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    model.booster_.save_model(str(model_path))

    metrics: dict = {"train_n": len(X_train), "test_n": len(X_test), "pos_rate": float(y.mean())}
    if y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, proba))
    else:
        metrics["auc"] = None

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model.registry (name, version, path, metrics)
                VALUES ('lgbm_mortality_12h', 'p0.1', :path, CAST(:metrics AS jsonb))
                """
            ),
            {"path": str(model_path), "metrics": json.dumps(metrics)},
        )

    return {"model_path": str(model_path), **metrics}


_model_bundle: tuple[lgb.Booster, object] | None = None


def _get_model_bundle() -> tuple[lgb.Booster, object]:
    global _model_bundle
    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if _model_bundle is None:
        import shap

        booster = lgb.Booster(model_file=str(model_path))
        explainer = shap.TreeExplainer(booster)
        _model_bundle = (booster, explainer)
    return _model_bundle


def _load_feature_row(stay_id: int) -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT feature_json FROM feat.sample_matrix
                WHERE stay_id = :stay_id AND hour_index = 0
                """
            ),
            {"stay_id": stay_id},
        ).mappings().first()
    if not row:
        return None
    feat = row["feature_json"] if isinstance(row["feature_json"], dict) else json.loads(row["feature_json"])
    return {c: feat.get(c, 0) for c in FEATURE_COLS}


def predict_stay(stay_id: int) -> dict:
    """L3: single-stay mortality risk score + SHAP top factors."""
    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.exists():
        return {
            "stay_id": stay_id,
            "status": "no_model",
            "message": "Run `python -m application.train` first.",
        }

    feat = _load_feature_row(stay_id)
    if feat is None:
        return {
            "stay_id": stay_id,
            "status": "no_features",
            "message": "Stay not found in feat.sample_matrix; run ETL + build_features.",
        }

    import shap

    booster, explainer = _get_model_bundle()
    row = [[feat[c] for c in FEATURE_COLS]]
    raw_score = float(booster.predict(row)[0])
    # LGBM binary classifier: treat as probability when in [0,1], else raw logit label.
    risk_score = raw_score if 0.0 <= raw_score <= 1.0 else raw_score
    score_kind = "probability" if 0.0 <= raw_score <= 1.0 else "raw"
    shap_row = explainer.shap_values(row)
    values = shap_row[0] if isinstance(shap_row, list) else shap_row[0]
    pairs = sorted(zip(FEATURE_COLS, values), key=lambda x: abs(x[1]), reverse=True)
    top_factors = [
        {"feature": name, "value": feat[name], "shap": round(float(contrib), 4)}
        for name, contrib in pairs[:4]
    ]
    return {
        "stay_id": stay_id,
        "status": "ok",
        "risk_score": round(risk_score, 4),
        "score_kind": score_kind,
        "top_factors": top_factors,
        "features": feat,
    }
