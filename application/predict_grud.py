"""L4 GRU-D inference interface.

Alignd with predict_patient output contract:
  {stay_id, status, risk_score, recommend, top_factors, ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from domain.features.sequence_build import build_mask_delta, pad_truncate
from domain.models.lgbm import recommend_action
from domain.models.temporal.grud_model import GRUD
from domain.models.temporal.attribution import FEATURE_NAMES, gradient_input_attribution
from infra.config import load_yaml, get_layer0_dsn
from infra.db import get_engine
from sqlalchemy import create_engine, text

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"

_CFG = load_yaml("temporal.yaml").get("temporal", {})
LOOKBACK_HOURS = int(_CFG.get("lookback_hours", 6))
MAX_TIMESTEPS = int(_CFG.get("max_timesteps", 12))
HIDDEN_SIZE = int(_CFG.get("hidden_size", 32))

_model_cache: dict[str, tuple[GRUD, torch.device]] | None = None


def _load_model() -> tuple[GRUD, torch.device]:
    global _model_cache
    if _model_cache is not None:
        return _model_cache  # type: ignore[return-value]

    model_path = ARTIFACT_DIR / "grud_mortality.pt"
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    input_size = ckpt.get("input_size", len(FEATURE_NAMES))
    model = GRUD(input_size=input_size, hidden_size=HIDDEN_SIZE, dropout=0.1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_cache = (model, device)
    return _model_cache  # type: ignore[return-value]


def _fetch_raw_sequence(stay_id: int) -> dict | None:
    """Fetch time-series for a single stay, constructing the input EXACTLY
    like training (train_grud.fetch_grud_sequences → _build_sequence_from_raw):
    carry-forward fill (+0.25h tolerance), temperature °F→°C, value clipping.
    Any train/inference mismatch here shifts the model output distribution."""
    dsn = get_layer0_dsn()
    if not dsn:
        return None
    engine = create_engine(dsn, pool_pre_ping=True)

    from domain.models.temporal.train_grud import (
        _build_sequence_from_raw,
        _clip_value,
        _itemids_for_features,
    )

    itemid_map = _itemids_for_features()
    vital_iids = [itemid_map[k] for k in ("hr", "sbp", "resp_rate", "temperature", "spo2")]
    lab_iids = [itemid_map[k] for k in ("lactate", "creatinine", "bun")]

    sql_vital = f"""
        SELECT c.itemid, c.valuenum,
               EXTRACT(EPOCH FROM (c.charttime - i.intime)) / 3600.0 AS hrs
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE i.stay_id = :sid
          AND c.charttime >= i.intime
          AND c.charttime < i.intime + INTERVAL '{LOOKBACK_HOURS} hours'
          AND c.valuenum IS NOT NULL
          AND c.itemid IN ({",".join(str(v) for v in vital_iids)})
    """
    sql_lab = f"""
        SELECT l.itemid, l.valuenum,
               EXTRACT(EPOCH FROM (l.charttime - i.intime)) / 3600.0 AS hrs
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
        WHERE i.stay_id = :sid
          AND l.charttime >= i.intime
          AND l.charttime < i.intime + INTERVAL '{LOOKBACK_HOURS} hours'
          AND l.valuenum IS NOT NULL
          AND l.itemid IN ({",".join(str(v) for v in lab_iids)})
    """
    charts: dict[int, list[tuple[float, float]]] = {}
    labs: dict[int, list[tuple[float, float]]] = {}
    with engine.connect() as conn:
        for r in conn.execute(text(sql_vital), {"sid": stay_id}).mappings().all():
            iid, val, hrs = int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
            if iid == 223761:
                val = (val - 32) * 5 / 9  # °F → °C, same as training
            val = _clip_value(iid, val)
            charts.setdefault(iid, []).append((hrs, val))
        for r in conn.execute(text(sql_lab), {"sid": stay_id}).mappings().all():
            iid, val, hrs = int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
            labs.setdefault(iid, []).append((hrs, val))

    seq = _build_sequence_from_raw(charts, labs, itemid_map)
    if seq is None:
        return None

    x_p, m_p, d_p = pad_truncate(seq["x"], seq["m"], seq["delta"], MAX_TIMESTEPS)
    return {"x": x_p, "m": m_p, "delta": d_p}


def _mock_sequence() -> dict:
    rng = np.random.default_rng(42)
    T = MAX_TIMESTEPS
    x = rng.normal(0, 1, size=(T, len(FEATURE_NAMES)))
    x[:, 0] = 75 + rng.normal(0, 10, T)   # hr
    x[:, 1] = 120 + rng.normal(0, 15, T)  # sbp
    m = (rng.random((T, len(FEATURE_NAMES))) > 0.25).astype(np.float64)
    times = np.arange(T) * 0.5
    _, m_out, d_out = build_mask_delta(x, times)
    return {"x": m_out, "m": m_out, "delta": d_out}


def predict_grud(stay_id: int) -> dict:
    """GRU-D prediction for a single stay. Output aligned with predict_stay()."""
    try:
        model, device = _load_model()
    except FileNotFoundError:
        return {
            "stay_id": stay_id,
            "status": "no_model",
            "message": "GRU-D model not found. Run `python -m application.train --train-gru` first.",
        }

    seq = _fetch_raw_sequence(stay_id)
    if seq is None:
        return {
            "stay_id": stay_id,
            "status": "no_sequence",
            "message": "No time-series data available for this stay.",
        }

    X = torch.FloatTensor(seq["x"]).unsqueeze(0).to(device)
    M = torch.FloatTensor(seq["m"]).unsqueeze(0).to(device)
    D = torch.FloatTensor(seq["delta"]).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(X, M, D).cpu().numpy()[0]

    prob = float(np.clip(prob, 0.0, 1.0))

    # Attribution
    attr = gradient_input_attribution(model, seq["x"], seq["m"], seq["delta"])

    # Build top_factors in same format as LightGBM.
    # Raw Gradient×Input values are ~1e-5; round to 4 decimals would zero them out.
    # Use relative importance normalized to max=1 so bars remain readable.
    raw_vals = [f["value"] for f in attr["top_features"]]
    max_val = max(raw_vals) if raw_vals else 0.0
    top_factors = [
        {"feature": f["feature"], "value": f["value"], "shap": (f["value"] / max_val if max_val > 0 else 0.0)}
        for f in attr["top_features"]
    ]

    # Time labels: pad_truncate prepends (max_t - n_steps) zero rows, so real
    # observations (hours 0..lookback) live at the END of the sequence. Label
    # padded columns as "pad" and real columns with actual hour offsets.
    n_total = seq["x"].shape[0]
    pad_len = max(n_total - int(LOOKBACK_HOURS * 2), 0)
    time_labels = [f"pad({i - pad_len}h)" for i in range(pad_len)] + [
        f"{(i - pad_len) * 0.5:.1f}h" for i in range(pad_len, n_total)
    ]

    recommend = recommend_action(prob, score_kind="probability")

    return {
        "stay_id": stay_id,
        "status": "ok",
        "risk_score": round(prob, 4),
        "score_kind": "probability",
        "recommend": recommend,
        "top_factors": top_factors,
        "hour_index": LOOKBACK_HOURS,
        "gru_attribution": {
            "feature_importance": attr["feature_importance"],
            "per_timestep_mean": {
                FEATURE_NAMES[i]: round(float(seq["x"][:, i].mean()), 4)
                for i in range(len(FEATURE_NAMES))
            },
            "per_timestep_values": seq["x"].tolist(),
            "per_timestep_attribution": attr["per_timestep_signed"].tolist(),
            "per_timestep_magnitude": attr["per_timestep"].tolist(),
            "feature_names": FEATURE_NAMES,
            "time_labels": time_labels,
        },
        "features": {FEATURE_NAMES[i]: round(float(seq["x"][-1, i]), 4) if not np.isnan(seq["x"][-1, i]) else None
                     for i in range(len(FEATURE_NAMES))},
        "model_type": "grud",
    }
