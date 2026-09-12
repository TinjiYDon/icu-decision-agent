"""GRU-D training pipeline for ICU mortality prediction.

End-to-end: fetch sequences → build dataset → train → evaluate → save.
Uses label.mortality_12h from icu_decision DB and raw MIMIC data from layer0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy import bindparam, create_engine, text

from domain.features.sequence_build import build_mask_delta, pad_truncate
from domain.models.split import save_split_manifest, split_frame_by_stay
from domain.models.temporal.grud_model import GRUD, build_train_dataset
from domain.models.temporal.attribution import FEATURE_NAMES
from infra.config import get_data_source, get_layer0_dsn, load_yaml
from infra.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "models"
MODEL_PATH = ARTIFACT_DIR / "grud_mortality.pt"
METRICS_PATH = ARTIFACT_DIR / "metrics_grud.json"

CFG: dict[str, Any] = load_yaml("temporal.yaml").get("temporal", {})
LOOKBACK_HOURS = int(CFG.get("lookback_hours", 6))
MAX_TIMESTEPS = int(CFG.get("max_timesteps", 12))
HIDDEN_SIZE = int(CFG.get("hidden_size", 32))
DROPOUT = float(CFG.get("dropout", 0.1))
EPOCHS = int(CFG.get("epochs", 20))
BATCH_SIZE = int(CFG.get("batch_size", 64))
LR = float(CFG.get("lr", 0.001))

TS_FEATURES = CFG.get("feature_keys", ["hr", "sbp", "lactate", "creatinine", "resp_rate", "temperature", "spo2", "bun"])
F = len(TS_FEATURES)


def _itemids_for_features() -> dict[str, int]:
    return {
        "hr": 220045, "sbp": 220179, "lactate": 50813, "creatinine": 50912,
        "resp_rate": 220210, "temperature": 223761, "spo2": 220277, "bun": 51006,
    }


def _clip_value(itemid: int, val: float) -> float:
    if itemid == 220179: return max(30.0, min(300.0, val))
    if itemid == 220045: return max(30.0, min(250.0, val))
    if itemid == 220277: return max(0.0, min(100.0, val))
    if itemid == 223761: return max(30.0, min(43.0, val))
    return val


def fetch_grud_sequences(stay_ids: list[int], batch_size: int = 500) -> tuple[list[dict], list[int]]:
    """Fetch time-series for stays, batching to avoid PostgreSQL param limit."""
    source = get_data_source()
    itemid_map = _itemids_for_features()

    if source == "mock":
        return _mock_sequences(stay_ids)

    layer0_dsn = get_layer0_dsn()
    if not layer0_dsn:
        raise RuntimeError("layer0 DSN not configured")
    layer0_eng = create_engine(layer0_dsn, pool_pre_ping=True)
    app_eng = get_engine()

    seqs: list[dict] = []
    labels: list[int] = []
    vital_iids = [_itemids_for_features()[k] for k in ("hr", "sbp", "resp_rate", "temperature", "spo2")]
    lab_iids = [_itemids_for_features()[k] for k in ("lactate", "creatinine", "bun")]
    vital_sql = f"""
        SELECT c.stay_id, c.itemid, c.valuenum,
               EXTRACT(EPOCH FROM (c.charttime - i.intime)) / 3600.0 AS hrs
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{LOOKBACK_HOURS} hours'
          AND c.valuenum IS NOT NULL AND c.itemid IN ({",".join(str(v) for v in vital_iids)})
          AND i.stay_id IN :sids ORDER BY c.stay_id, c.charttime"""
    lab_sql = f"""
        SELECT i.stay_id, l.itemid, l.valuenum,
               EXTRACT(EPOCH FROM (l.charttime - i.intime)) / 3600.0 AS hrs
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
        WHERE l.charttime >= i.intime AND l.charttime < i.intime + INTERVAL '{LOOKBACK_HOURS} hours'
          AND l.valuenum IS NOT NULL AND l.itemid IN ({",".join(str(v) for v in lab_iids)})
          AND i.stay_id IN :sids ORDER BY i.stay_id, l.charttime"""
    label_sql = """SELECT stay_id, label FROM label.mortality_12h WHERE hour_index=:h AND stay_id IN :sids"""

    for bs in range(0, len(stay_ids), batch_size):
        batch = stay_ids[bs:bs + batch_size]
        b = tuple(batch)
        with app_eng.connect() as c:
            lmap = {int(r["stay_id"]): int(r["label"])
                    for r in c.execute(text(label_sql).bindparams(bindparam("sids", expanding=True)),
                                       {"h": LOOKBACK_HOURS, "sids": b}).mappings().all()}
        charts: dict[int, dict[int, list[tuple[float, float]]]] = {}
        labs: dict[int, dict[int, list[tuple[float, float]]]] = {}
        with layer0_eng.connect() as c:
            for r in c.execute(text(vital_sql).bindparams(bindparam("sids", expanding=True)), {"sids": b}).mappings().all():
                sid, iid, val, hrs = int(r["stay_id"]), int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
                if iid == 223761: val = (val - 32) * 5 / 9
                val = _clip_value(iid, val)
                charts.setdefault(sid, {}).setdefault(iid, []).append((hrs, val))
            for r in c.execute(text(lab_sql).bindparams(bindparam("sids", expanding=True)), {"sids": b}).mappings().all():
                sid, iid, val, hrs = int(r["stay_id"]), int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
                labs.setdefault(sid, {}).setdefault(iid, []).append((hrs, val))
        for sid in batch:
            seq = _build_sequence_from_raw(charts.get(sid, {}), labs.get(sid, {}), itemid_map)
            if seq is not None:
                seqs.append(seq)
                labels.append(lmap.get(sid, 0))
        if bs % 5000 == 0:
            print(f"  [grud] progress: {min(bs + batch_size, len(stay_ids))}/{len(stay_ids)} stays, {len(seqs)} valid seqs")

    return seqs, labels


def _build_sequence_from_raw(
    charts: dict[int, list[tuple[float, float]]],
    labs: dict[int, list[tuple[float, float]]],
    itemid_map: dict[str, int],
) -> dict | None:
    events: list[tuple[float, int, float]] = []
    for iid, tlist in {**charts, **labs}.items():
        fi = list(itemid_map.values()).index(iid) if iid in itemid_map.values() else -1
        if fi < 0: continue
        for hrs, val in tlist:
            events.append((hrs, fi, val))
    if not events: return None
    events.sort(key=lambda e: e[0])
    n_steps = int(LOOKBACK_HOURS * 2)
    uniform_times = np.arange(n_steps) * 0.5
    uniform_x = np.full((n_steps, F), np.nan, dtype=np.float64)
    by_feat: dict[int, list[tuple[float, float]]] = {}
    for hrs, fi, val in events:
        by_feat.setdefault(fi, []).append((hrs, val))
    for t_idx in range(n_steps):
        t_hr = uniform_times[t_idx]
        for fi in range(F):
            cands = [(h, v) for h, v in by_feat.get(fi, []) if h <= t_hr + 0.25]
            if cands: uniform_x[t_idx, fi] = cands[-1][1]
    m = (~np.isnan(uniform_x)).astype(np.float64)
    x_ff, m_out, d_out = build_mask_delta(uniform_x, uniform_times)
    return {"x": x_ff, "m": m_out, "delta": d_out}


def _mock_sequences(stay_ids: list[int]) -> tuple[list[dict], list[int]]:
    rng = np.random.default_rng(42)
    seqs, labels = [], []
    for sid in stay_ids[:200]:
        T = MAX_TIMESTEPS
        x = rng.normal(0, 1, size=(T, F)).astype(np.float64)
        x[:, 0] = 70 + rng.normal(0, 15, T)
        x[:, 1] = 120 + rng.normal(0, 20, T)
        x[:, 2] = np.maximum(0, 1.5 + rng.exponential(1.0, T))
        m = (rng.random((T, F)) > 0.3).astype(np.float64)
        delta = np.cumsum(rng.exponential(0.5, size=(T, F)), axis=0)
        seqs.append({"x": x, "m": m, "delta": delta})
        labels.append(int(rng.random() < 0.15))
    return seqs, labels


def train_grud() -> dict:
    """Train GRU-D model end-to-end."""
    dsn = get_layer0_dsn()
    if not dsn: raise RuntimeError("layer0 DSN not configured")
    layer0_eng = create_engine(dsn, pool_pre_ping=True)
    app_eng = get_engine()

    with app_eng.connect() as conn:
        stay_rows = conn.execute(text(
            "SELECT DISTINCT stay_id FROM label.mortality_12h WHERE hour_index = :h"),
            {"h": LOOKBACK_HOURS}).mappings().all()
    stay_ids = [int(r["stay_id"]) for r in stay_rows]
    if len(stay_ids) < 100:
        raise RuntimeError(f"too few stays: {len(stay_ids)}")

    print(f"  [grud] fetching {len(stay_ids)} sequences (lookback={LOOKBACK_HOURS}h)...")
    seqs, labels = fetch_grud_sequences(stay_ids)
    print(f"  [grud] got {len(seqs)} valid sequences, pos_rate={np.mean(labels):.3f}")
    if len(seqs) < 50: raise RuntimeError(f"too few sequences: {len(seqs)}")

    X_all, M_all, D_all, Y_all = build_train_dataset(seqs, labels)
    n = X_all.shape[0]
    import pandas as pd
    df = pd.DataFrame({"stay_id": [stay_ids[i] for i in range(n)], "label": Y_all.numpy().astype(int)})
    train_df, val_df, test_df, manifest = split_frame_by_stay(df)
    save_split_manifest(manifest)

    train_ids, val_ids, test_ids = set(train_df["stay_id"]), set(val_df["stay_id"]), set(test_df["stay_id"])
    mk = {"train": [stay_ids[i] in train_ids for i in range(n)],
          "val": [stay_ids[i] in val_ids for i in range(n)],
          "test": [stay_ids[i] in test_ids for i in range(n)]}
    X_tr, Y_tr = X_all[mk["train"]], Y_all[mk["train"]]
    X_va, Y_va = X_all[mk["val"]], Y_all[mk["val"]]
    X_te, Y_te = X_all[mk["test"]], Y_all[mk["test"]]
    M_tr, D_tr = M_all[mk["train"]], D_all[mk["train"]]
    M_va, D_va = M_all[mk["val"]], D_all[mk["val"]]
    M_te, D_te = M_all[mk["test"]], D_all[mk["test"]]

    pos, neg = max(int(Y_tr.sum()), 1), max(int(len(Y_tr) - Y_tr.sum()), 1)
    scale_pos = neg / pos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUD(input_size=F, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos], device=device))

    print(f"  [grud] device={device}, train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}, scale={scale_pos:.1f}")
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, M_tr, D_tr, Y_tr),
                                         batch_size=BATCH_SIZE, shuffle=True)
    best_auc, wait, best_state = 0.0, 0, None

    for epoch in range(EPOCHS):
        model.train()
        for xb, mb, db, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device), mb.to(device), db.to(device)), yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        va_p = _eval_probs(model, X_va, M_va, D_va, device)
        va_auc = _roc_auc(Y_va.numpy(), va_p)
        if va_auc > best_auc:
            best_auc, wait, best_state = va_auc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                print(f"  [grud] early stop epoch {epoch+1}, best val AUC={best_auc:.4f}"); break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    te_p = _eval_probs(model, X_te, M_te, D_te, device)
    from domain.models.evaluation import binary_metrics
    test_m = binary_metrics(Y_te.numpy(), te_p)
    val_m = binary_metrics(Y_va.numpy(), _eval_probs(model, X_va, M_va, D_va, device)) if best_state else {}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "input_size": F, "hidden_size": HIDDEN_SIZE, "config": CFG}, str(MODEL_PATH))
    metrics = {**test_m, **{f"val_{k}": v for k, v in val_m.items()},
               "total_n": int(n), "train_n": int(len(X_tr)), "val_n": int(len(X_va)), "test_n": int(len(X_te)),
               "positive": int(Y_all.sum()), "pos_rate": float(Y_all.mean()), "model": "grud",
               "lookback_hours": LOOKBACK_HOURS, "max_timesteps": MAX_TIMESTEPS,
               "hidden_size": HIDDEN_SIZE, "scale_pos_weight": float(scale_pos), "feature_names": TS_FEATURES}
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with app_eng.begin() as conn:
        conn.execute(text("INSERT INTO model.registry (name, version, path, metrics) VALUES ('grud_mortality', 'p3.0', :path, CAST(:metrics AS jsonb))"),
                     {"path": str(MODEL_PATH), "metrics": json.dumps(metrics)})
    print(f"  [grud] saved → {MODEL_PATH}")
    print(f"  [grud] test ROC-AUC={test_m.get('roc_auc',0):.4f} PR-AUC={test_m.get('pr_auc',0):.4f} Brier={test_m.get('brier',0):.4f}")
    return {"model_path": str(MODEL_PATH), "metrics_path": str(METRICS_PATH), **metrics}


def _eval_probs(model, X, M, D, device):
    model.eval()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, M, D), batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, mb, db in loader:
            probs.extend(model(xb.to(device), mb.to(device), db.to(device)).cpu().numpy().tolist())
    return np.array(probs)


def _roc_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
