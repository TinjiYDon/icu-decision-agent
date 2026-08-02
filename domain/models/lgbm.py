"""LightGBM trainer for S1 early-warning mortality model (t=intime+offset)."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from domain.features.build import FEATURE_COLS, prediction_hour_index, prediction_hours
from domain.models.evaluation import binary_metrics, select_threshold_by_f1
from domain.models.split import save_split_manifest, split_frame_by_stay
from infra.config import load_yaml
from infra.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "models"


def _load_training_frame() -> pd.DataFrame:
    hours = prediction_hours()
    engine = get_engine()
    sql = text(
        """
        SELECT f.stay_id, f.hour_index, f.feature_json, l.label
        FROM feat.sample_matrix f
        JOIN label.mortality_12h l ON f.stay_id = l.stay_id AND f.hour_index = l.hour_index
        WHERE f.hour_index IN :hours
        """
    ).bindparams(bindparam("hours", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(sql, {"hours": list(hours)}).mappings().all()
    records = []
    for row in rows:
        feat = row["feature_json"] if isinstance(row["feature_json"], dict) else json.loads(row["feature_json"])
        rec = {}
        for c in FEATURE_COLS:
            v = feat.get(c, 0)
            if v is None:
                v = np.nan
            rec[c] = v
        rec["label"] = int(row["label"])
        rec["stay_id"] = row["stay_id"]
        rec["hour_index"] = int(row["hour_index"])
        records.append(rec)
    return pd.DataFrame(records)


def train_and_save() -> dict:
    df = _load_training_frame()
    if len(df) < 10:
        raise RuntimeError(f"too few samples for training: {len(df)}")

    train_df, val_df, test_df, manifest = split_frame_by_stay(df)
    save_split_manifest(manifest)

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

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
    fit_kwargs: dict = {}
    if len(val_df) > 0 and y_val.nunique() > 1:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(10, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    model.booster_.save_model(str(model_path))

    val_probability = model.predict_proba(X_val)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]
    operating_threshold = (
        select_threshold_by_f1(y_val, val_probability)
        if len(X_val) > 0 and y_val.nunique() > 1
        else 0.5
    )
    val_default = binary_metrics(y_val, val_probability, threshold=0.5)
    test_default = binary_metrics(y_test, test_probability, threshold=0.5)
    val_operating = binary_metrics(y_val, val_probability, threshold=operating_threshold)
    test_operating = binary_metrics(y_test, test_probability, threshold=operating_threshold)

    metrics: dict = {
        "total_n": int(len(df)),
        "positive": int(df["label"].sum()),
        "train_n": int(len(X_train)),
        "val_n": int(len(X_val)),
        "test_n": int(len(X_test)),
        "pos_rate": float(df["label"].mean()),
        "feature_cols": FEATURE_COLS,
        "split": manifest["n_stays"],
        "split_class_counts": manifest["class_counts"],
        "split_positive_rate": manifest["positive_rate"],
        "stratified": manifest["stratified"],
        "operating_threshold": operating_threshold,
        "threshold_selection": "maximum F1 on validation split",
        "metrics_at_0_5": {"val": val_default, "test": test_default},
        "metrics_at_val_threshold": {"val": val_operating, "test": test_operating},
    }
    metrics["auc_val"] = val_default["roc_auc"]
    metrics["auc_test"] = test_default["roc_auc"]
    metrics["auc"] = metrics["auc_test"]  # backward-compatible key
    metrics["pr_auc_val"] = val_default["pr_auc"]
    metrics["pr_auc_test"] = test_default["pr_auc"]
    metrics["brier_val"] = val_default["brier"]
    metrics["brier_test"] = test_default["brier"]
    metrics["prediction_hours"] = prediction_hours()
    metrics["n_stays"] = int(df["stay_id"].nunique())

    # Per-hour test metrics (same model, sliced)
    by_hour: dict = {}
    test_with_h = test_df.copy()
    test_with_h["prob"] = test_probability
    for h, part in test_with_h.groupby("hour_index"):
        if len(part) < 5 or part["label"].nunique() < 2:
            continue
        by_hour[str(int(h))] = binary_metrics(
            part["label"], part["prob"].to_numpy(), threshold=operating_threshold
        )
    metrics["metrics_by_hour_test"] = by_hour

    metrics_path = ARTIFACT_DIR / "metrics_mortality_12h.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model.registry (name, version, path, metrics)
                VALUES ('lgbm_mortality_12h', 'p0.3-stratified', :path, CAST(:metrics AS jsonb))
                """
            ),
            {"path": str(model_path), "metrics": json.dumps(metrics)},
        )

    return {"model_path": str(model_path), "metrics_path": str(metrics_path), **metrics}


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


# Probe keys: S2 dump is lab-heavy; chart vitals are often null in export.
_CLINICAL_PROBE = (
    "anchor_age",
    "lab_lactate",
    "lab_creatinine",
    "lab_hematocrit",
    "lab_bun",
    "lab_sodium",
    "lab_ph",
    "vital_heart_rate",
)


def _nonzero(raw: dict, key: str) -> bool:
    if key not in raw or raw[key] is None:
        return False
    try:
        return float(raw[key]) != 0.0
    except (TypeError, ValueError):
        return True


def _feature_quality(raw: dict) -> dict:
    """Assess completeness of stored feature_json (before model zero-fill)."""
    present = sum(1 for k in _CLINICAL_PROBE if _nonzero(raw, k))
    lab_present = sum(
        1 for k in raw if k.startswith("lab_") and _nonzero(raw, k)
    )
    placeholder = set(raw.keys()) <= {"los_hours", "first_careunit"}
    has_age = _nonzero(raw, "anchor_age")
    usable = (not placeholder) and has_age and lab_present >= 2
    return {
        "clinical_present": present,
        "clinical_total": len(_CLINICAL_PROBE),
        "lab_present": lab_present,
        "is_placeholder": placeholder,
        "usable": usable,
    }


def _load_feature_row(stay_id: int, hour_index: int | None = None) -> dict | None:
    """Load features for model + display.

    Returns dict with:
      - model_features: FEATURE_COLS filled with 0 for missing (inference)
      - features_display: same keys, None when absent/null in JSON
      - feature_quality: completeness flags
    """
    h = prediction_hour_index() if hour_index is None else int(hour_index)
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT feature_json FROM feat.sample_matrix
                WHERE stay_id = :stay_id AND hour_index = :hour_index
                """
            ),
            {"stay_id": stay_id, "hour_index": h},
        ).mappings().first()
    if not row:
        return None
    raw = row["feature_json"] if isinstance(row["feature_json"], dict) else json.loads(row["feature_json"])
    model_features = {
        c: (raw.get(c, 0) if raw.get(c) is not None else 0) for c in FEATURE_COLS
    }
    features_display = {
        c: (raw[c] if c in raw and raw[c] is not None else None) for c in FEATURE_COLS
    }
    return {
        "model_features": model_features,
        "features_display": features_display,
        "feature_quality": _feature_quality(raw),
        "raw_keys": sorted(raw.keys()),
    }


def recommend_action(risk_score: float, score_kind: str = "probability") -> dict:
    """Map score to clinical band using configs/labels.yaml recommend thresholds."""
    cfg = load_yaml("labels.yaml").get("recommend", {})
    observe = float(cfg.get("observe", 0.2))
    recheck = float(cfg.get("recheck", 0.4))
    monitor = float(cfg.get("monitor", 0.7))
    if score_kind != "probability":
        return {
            "band": "unknown",
            "label": "分数非概率，暂不分级",
            "thresholds": {"observe": observe, "recheck": recheck, "monitor": monitor},
        }
    if risk_score < observe:
        band, label = "observe", "观察（低风险）"
    elif risk_score < recheck:
        band, label = "recheck", "复查（中低风险）"
    elif risk_score < monitor:
        band, label = "monitor", "加强监护（中高风险）"
    else:
        band, label = "escalate", "升级处置（高风险）"
    return {
        "band": band,
        "label": label,
        "thresholds": {"observe": observe, "recheck": recheck, "monitor": monitor},
    }


def predict_stay(stay_id: int, hour_index: int | None = None) -> dict:
    """L3: single-stay mortality risk score + SHAP top factors."""
    h = prediction_hour_index() if hour_index is None else int(hour_index)
    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.exists():
        return {
            "stay_id": stay_id,
            "hour_index": h,
            "status": "no_model",
            "message": "Run `python -m application.train` first.",
        }

    loaded = _load_feature_row(stay_id, hour_index=h)
    if loaded is None:
        return {
            "stay_id": stay_id,
            "hour_index": h,
            "status": "no_features",
            "message": "Stay not found in feat.sample_matrix; run ETL + build_features.",
        }

    feat = loaded["model_features"]
    display = loaded["features_display"]
    quality = loaded["feature_quality"]
    booster, explainer = _get_model_bundle()
    row = np.asarray([[feat[c] for c in FEATURE_COLS]], dtype=float)
    raw_score = float(booster.predict(row)[0])
    risk_score = raw_score if 0.0 <= raw_score <= 1.0 else raw_score
    score_kind = "probability" if 0.0 <= raw_score <= 1.0 else "raw"
    shap_row = explainer.shap_values(row)
    if isinstance(shap_row, list):
        # binary: prefer positive-class attributions when present
        values = shap_row[1] if len(shap_row) > 1 else shap_row[0]
    else:
        values = shap_row[0]
    pairs = sorted(zip(FEATURE_COLS, values), key=lambda x: abs(float(x[1])), reverse=True)
    # Prefer explaining features that were actually observed (not zero-filled missing).
    observed = [
        (name, contrib)
        for name, contrib in pairs
        if display.get(name) is not None
    ]
    ranked = observed if len(observed) >= 2 else pairs
    top_factors = [
        {
            "feature": name,
            "value": display.get(name),
            "shap": round(float(contrib), 4),
        }
        for name, contrib in ranked[:4]
    ]
    return {
        "stay_id": stay_id,
        "hour_index": h,
        "status": "ok",
        "risk_score": round(risk_score, 4),
        "score_kind": score_kind,
        "recommend": recommend_action(risk_score, score_kind),
        "top_factors": top_factors,
        "features": display,
        "features_model": feat,
        "feature_quality": quality,
    }


def predict_stay_trajectory(stay_id: int, hours: list[int] | None = None) -> dict:
    """Predict risk at each hour in the S2 grid (or provided list)."""
    grid = list(hours) if hours is not None else prediction_hours()
    points = []
    for h in grid:
        out = predict_stay(int(stay_id), hour_index=int(h))
        points.append(
            {
                "hour_index": int(h),
                "status": out.get("status"),
                "risk_score": out.get("risk_score"),
                "score_kind": out.get("score_kind"),
                "recommend": out.get("recommend"),
            }
        )
    ok = [p for p in points if p["status"] == "ok"]
    return {
        "stay_id": int(stay_id),
        "status": "ok" if ok else (points[0]["status"] if points else "empty"),
        "points": points,
    }
