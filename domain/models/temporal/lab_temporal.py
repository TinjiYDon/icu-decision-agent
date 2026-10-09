"""Lab-only irregular sequences for temporal deepen (Layer0 often has no chartevents).

Default bedside model remains LightGBM. This track answers H2 with richer labs
than lactate/creatinine/bun alone, and adds TFT-lite as a third peer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sqlalchemy import bindparam, create_engine, text

from domain.features.sequence_build import build_mask_delta, pad_truncate
from domain.features.sequence_etl import passes_sparse_gate
from domain.models.split import split_frame_by_stay
from domain.models.temporal.grud_model import GRUD
from domain.models.temporal.tft_lite import TFTLite
from infra.config import get_layer0_dsn
from infra.db import get_engine

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "d3"

# Labs aligned with decision CDSS feature story (MIMIC itemids).
LAB_FEATURE_NAMES: list[str] = [
    "lactate",
    "creatinine",
    "bun",
    "wbc",
    "hemoglobin",
    "platelets",
    "sodium",
    "potassium",
    "bicarbonate",
    "bilirubin",
]

LAB_ITEMIDS: dict[str, int] = {
    "lactate": 50813,
    "creatinine": 50912,
    "bun": 51006,
    "wbc": 51300,
    "hemoglobin": 50811,
    "platelets": 51265,
    "sodium": 50983,
    "potassium": 50971,
    "bicarbonate": 50882,
    "bilirubin": 50885,
}


@dataclass
class TripleMetrics:
    model: str
    n_test: int
    pr_auc: float
    brier: float
    roc_auc: float
    primary_metric: str = "pr_auc"


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    out = {
        "pr_auc": float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "roc_auc": float("nan"),
    }
    if len(np.unique(y)) > 1:
        out["pr_auc"] = float(average_precision_score(y, p))
        out["roc_auc"] = float(roc_auc_score(y, p))
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _fmt_metric(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{float(v):.4f}"


def _quota_caps(limit: int) -> tuple[int, int]:
    """Split ``limit`` into (pos_cap, neg_cap) with pos≈15%; never exceed limit."""
    limit = max(int(limit), 1)
    pos_cap = max(1, int(round(limit * 0.15)))
    if limit >= 120:
        # Prefer denser positives when the budget allows, still ≤ limit.
        pos_cap = min(max(pos_cap, 40), limit - 1)
    pos_cap = min(pos_cap, limit)
    neg_cap = limit - pos_cap
    return pos_cap, neg_cap


def sample_stay_ids(*, limit: int, hour_index: int = 6) -> list[int]:
    """Mix positives + negatives so rare labels are not wiped by ORDER BY stay_id."""
    app = get_engine()
    pos_cap, neg_cap = _quota_caps(limit)
    h = int(hour_index)
    with app.connect() as c:
        pos = [
            int(r[0])
            for r in c.execute(
                text(
                    "SELECT stay_id FROM label.mortality_12h "
                    "WHERE hour_index = :h AND label = 1 "
                    "ORDER BY stay_id LIMIT :lim"
                ),
                {"h": h, "lim": pos_cap},
            ).fetchall()
        ]
        neg = [
            int(r[0])
            for r in c.execute(
                text(
                    "SELECT stay_id FROM label.mortality_12h "
                    "WHERE hour_index = :h AND label = 0 "
                    "ORDER BY stay_id LIMIT :lim"
                ),
                {"h": h, "lim": neg_cap},
            ).fetchall()
        ]
    ids = sorted(set(pos + neg))[: int(limit)]
    if len(ids) < min(40, int(limit)):
        with app.connect() as c:
            ids = [
                int(r[0])
                for r in c.execute(
                    text(
                        "SELECT DISTINCT stay_id FROM label.mortality_12h "
                        "WHERE hour_index = :h ORDER BY stay_id LIMIT :lim"
                    ),
                    {"h": h, "lim": int(limit)},
                ).fetchall()
            ]
    return ids[: int(limit)]


def events_to_raw_grid(
    events: Sequence[tuple[float, int, float]],
    *,
    grid: np.ndarray,
    name_by_id: dict[int, str],
    feature_index: dict[str, int],
    n_features: int,
) -> np.ndarray:
    """Map each lab event onto at most one grid cell; **no** LOCF into ``raw``.

    Mask/delta/forward-fill are derived later by ``build_mask_delta``, so sparsity
    and recency stay faithful to actual measurement times.
    """
    raw = np.full((len(grid), n_features), np.nan, dtype=np.float64)
    if len(grid) == 0:
        return raw
    t0 = float(grid[0])
    t1 = float(grid[-1])
    for hrs, iid, val in events:
        if hrs < t0 - 1e-9 or hrs > t1 + 1e-9:
            continue
        name = name_by_id.get(int(iid))
        if name is None:
            continue
        col = feature_index.get(name)
        if col is None:
            continue
        t_i = int(np.searchsorted(grid, float(hrs), side="right") - 1)
        t_i = max(0, min(t_i, len(grid) - 1))
        raw[t_i, col] = float(val)
    return raw


def _feature_window_hours(
    *, hour_index: int, lookback_hours: int
) -> tuple[float, float]:
    """Causal window ``[start_h, end_h)`` ending at prediction time ``intime+h``."""
    end_h = float(max(int(hour_index), 0))
    lb = float(max(int(lookback_hours), 0))
    start_h = max(0.0, end_h - lb) if lb > 0 else 0.0
    if end_h <= 0:
        # Degenerate: prediction at intime — empty open interval; keep a point grid.
        return 0.0, 0.0
    return start_h, end_h


def fetch_lab_sequences(
    stay_ids: Sequence[int],
    *,
    hour_index: int = 6,
    lookback_hours: int = 12,
    max_timesteps: int = 12,
) -> tuple[list[dict[str, np.ndarray]], list[int], list[int]]:
    """Build (x,m,delta) from labevents only. Returns seqs, labels, valid_ids.

    Features use ``charttime < intime + hour_index`` (same ``h`` as the label row).
    ``lookback_hours`` only widens the *start* of the window; it must not extend
    past the prediction time (fixes future-lab leakage when lookback > h).
    """
    layer0 = create_engine(get_layer0_dsn(), pool_pre_ping=True)
    app = get_engine()
    names = list(LAB_FEATURE_NAMES)
    f = len(names)
    itemids = [LAB_ITEMIDS[n] for n in names]
    name_by_id = {v: k for k, v in LAB_ITEMIDS.items()}
    idx = {n: i for i, n in enumerate(names)}
    h = int(hour_index)
    start_h, end_h = _feature_window_hours(
        hour_index=h, lookback_hours=int(lookback_hours)
    )

    with app.connect() as c:
        lab_rows = c.execute(
            text(
                "SELECT stay_id, label FROM label.mortality_12h "
                "WHERE hour_index = :h AND stay_id IN :sids"
            ).bindparams(bindparam("sids", expanding=True)),
            {"h": h, "sids": [int(s) for s in stay_ids]},
        ).mappings().all()
    label_map = {int(r["stay_id"]): int(r["label"]) for r in lab_rows}

    seqs: list[dict[str, np.ndarray]] = []
    labels: list[int] = []
    valid: list[int] = []
    batch = 400
    ids = [int(s) for s in stay_ids if int(s) in label_map]
    # Open interval end at prediction time; start may be >0 when lookback < h.
    start_sql = (
        f"i.intime + INTERVAL '{int(start_h)} hours'"
        if start_h > 0
        else "i.intime"
    )
    for bs in range(0, len(ids), batch):
        chunk = ids[bs : bs + batch]
        sql = f"""
            SELECT i.stay_id, l.itemid, l.valuenum,
                   EXTRACT(EPOCH FROM (l.charttime - i.intime))/3600.0 AS hrs
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE i.stay_id IN :sids
              AND l.valuenum IS NOT NULL
              AND l.itemid IN ({",".join(str(x) for x in itemids)})
              AND l.charttime >= {start_sql}
              AND l.charttime < i.intime + INTERVAL '{h} hours'
            ORDER BY i.stay_id, l.charttime
        """
        with layer0.connect() as c:
            rows = c.execute(
                text(sql).bindparams(bindparam("sids", expanding=True)),
                {"sids": chunk},
            ).mappings().all()
        by_stay: dict[int, list[tuple[float, int, float]]] = {}
        for r in rows:
            by_stay.setdefault(int(r["stay_id"]), []).append(
                (float(r["hrs"]), int(r["itemid"]), float(r["valuenum"]))
            )
        grid = np.linspace(start_h, end_h, max_timesteps)
        for sid in chunk:
            events = by_stay.get(sid) or []
            if not events:
                continue
            raw = events_to_raw_grid(
                events,
                grid=grid,
                name_by_id=name_by_id,
                feature_index=idx,
                n_features=f,
            )
            x, m, d = build_mask_delta(raw, grid)
            xp, mp, dp = pad_truncate(x, m, d, max_timesteps)
            if not passes_sparse_gate(
                mp, feature_names_list=names, require_any_vital=False
            ):
                continue
            seqs.append({"x": xp, "m": mp, "delta": dp})
            labels.append(label_map[sid])
            valid.append(sid)
    layer0.dispose()
    return seqs, labels, valid


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _train_torch_classifier(
    model: torch.nn.Module,
    X: torch.Tensor,
    M: torch.Tensor,
    D: torch.Tensor,
    Y: torch.Tensor,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    epochs: int,
    lr: float,
    returns_logits: bool,
) -> torch.nn.Module:
    device = torch.device("cpu")
    model = model.to(device)
    tr = tr_idx.tolist()
    va = va_idx.tolist()
    pos = max(float(Y[tr].sum().item()), 1.0)
    neg = max(float(len(tr) - pos), 1.0)
    if returns_logits:
        crit: torch.nn.Module = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
        )
    else:
        w = torch.ones_like(Y[tr])
        w = torch.where(Y[tr] > 0.5, w * (neg / pos), w)
        crit = torch.nn.BCELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_pr = -1.0
    wait = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X[tr], M[tr], D[tr])
        if returns_logits:
            loss = crit(out, Y[tr])
        else:
            loss = (crit(out.clamp(1e-4, 1 - 1e-4), Y[tr]) * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(X[va], M[va], D[va])
            if returns_logits:
                vp = torch.sigmoid(vo).numpy()
            else:
                vp = vo.numpy()
            yt = Y[va].numpy().astype(int)
            pr = (
                float(average_precision_score(yt, vp))
                if len(np.unique(yt)) > 1
                else 0.0
            )
        if pr >= best_pr:
            best_pr = pr
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _predict_probs(
    model: torch.nn.Module,
    X: torch.Tensor,
    M: torch.Tensor,
    D: torch.Tensor,
    *,
    returns_logits: bool,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(X, M, D)
        if returns_logits:
            return torch.sigmoid(out).numpy()
        return out.numpy()


def _index_arrays(
    valid: Sequence[int],
    labels: Sequence[int],
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    df = pd.DataFrame({"stay_id": list(valid), "label": list(labels)})
    train_df, val_df, test_df, manifest = split_frame_by_stay(df, seed=seed)
    sid_to_i = {int(s): i for i, s in enumerate(valid)}
    tr = np.asarray([sid_to_i[int(s)] for s in train_df["stay_id"]], dtype=int)
    va = np.asarray([sid_to_i[int(s)] for s in val_df["stay_id"]], dtype=int)
    te = np.asarray([sid_to_i[int(s)] for s in test_df["stay_id"]], dtype=int)
    return tr, va, te, manifest


def train_eval_grud(
    seqs: list[dict[str, np.ndarray]],
    labels: list[int],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    *,
    epochs: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [s["x"] for s in seqs]
    ms = [s["m"] for s in seqs]
    ds = [s["delta"] for s in seqs]
    X = torch.FloatTensor(np.asarray(xs))
    M = torch.FloatTensor(np.asarray(ms))
    D = torch.FloatTensor(np.asarray(ds))
    Y = torch.FloatTensor(np.asarray(labels, dtype=np.float32))
    model = GRUD(input_size=X.shape[-1], hidden_size=48, dropout=0.1)
    model = _train_torch_classifier(
        model,
        X,
        M,
        D,
        Y,
        tr_idx,
        va_idx,
        epochs=epochs,
        lr=1e-3,
        returns_logits=False,
    )
    full = _predict_probs(model, X, M, D, returns_logits=False)
    return Y[te_idx].numpy().astype(int), full[te_idx]


def train_eval_tft(
    seqs: list[dict[str, np.ndarray]],
    labels: list[int],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    *,
    epochs: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [s["x"] for s in seqs]
    ms = [s["m"] for s in seqs]
    ds = [s["delta"] for s in seqs]
    X = torch.FloatTensor(np.asarray(xs))
    M = torch.FloatTensor(np.asarray(ms))
    D = torch.FloatTensor(np.asarray(ds))
    Y = torch.FloatTensor(np.asarray(labels, dtype=np.float32))
    model = TFTLite(input_size=X.shape[-1], d_model=32, nhead=4, num_layers=2, dropout=0.1)
    model = _train_torch_classifier(
        model,
        X,
        M,
        D,
        Y,
        tr_idx,
        va_idx,
        epochs=epochs,
        lr=1e-3,
        returns_logits=True,
    )
    full = _predict_probs(model, X, M, D, returns_logits=True)
    return Y[te_idx].numpy().astype(int), full[te_idx]


def lgbm_probs_on_stays(
    stay_ids: Sequence[int], hour_index: int = 6
) -> tuple[dict[int, float], dict[str, Any]]:
    """Score stays with LightGBM; surface failures instead of silent omission."""
    from application.predict_patient import predict_patient

    out: dict[int, float] = {}
    errors: list[dict[str, Any]] = []
    for sid in stay_ids:
        sid_i = int(sid)
        try:
            r = predict_patient(sid_i, hour_index=hour_index, model_type="lgbm")
            if r.get("status") == "ok":
                out[sid_i] = float(r.get("risk_score", 0.0))
            else:
                errors.append(
                    {
                        "stay_id": sid_i,
                        "status": r.get("status"),
                        "message": r.get("message") or r.get("error"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"stay_id": sid_i, "status": "exception", "message": str(exc)})
    meta = {
        "n_requested": len(list(stay_ids)),
        "n_ok": len(out),
        "n_failed": len(errors),
        "errors_head": errors[:20],
    }
    return out, meta


def run_lab_triple_compare(
    *,
    limit: int = 500,
    hour_index: int = 6,
    lookback_hours: int = 12,
    epochs: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    h = int(hour_index)
    stays = sample_stay_ids(limit=limit, hour_index=h)
    seqs, labels, valid = fetch_lab_sequences(
        stays,
        hour_index=h,
        lookback_hours=lookback_hours,
        max_timesteps=12,
    )
    if len(valid) < 40:
        raise RuntimeError(f"lab sequences too few: {len(valid)}")

    tr_idx, va_idx, te_idx, split_meta = _index_arrays(valid, labels, seed=seed)
    y_g, p_g = train_eval_grud(seqs, labels, tr_idx, va_idx, te_idx, epochs=epochs)
    y_t, p_t = train_eval_tft(seqs, labels, tr_idx, va_idx, te_idx, epochs=epochs)

    test_ids = [valid[i] for i in te_idx.tolist()]
    test_y = np.asarray([labels[i] for i in te_idx.tolist()], dtype=int)
    lmap, lgbm_meta = lgbm_probs_on_stays(test_ids, hour_index=h)
    keep = [i for i, sid in enumerate(test_ids) if sid in lmap]
    if len(keep) < 10:
        raise RuntimeError(
            "LGBM overlap on test too small: "
            f"ok={lgbm_meta['n_ok']} failed={lgbm_meta['n_failed']} "
            f"requested={lgbm_meta['n_requested']}"
        )
    y_common = test_y[keep]
    p_l = np.asarray([lmap[test_ids[i]] for i in keep], dtype=float)
    p_g2 = p_g[keep]
    p_t2 = p_t[keep]

    rows = [
        TripleMetrics("lgbm", len(y_common), **_metrics(y_common, p_l)),
        TripleMetrics("grud_labs", len(y_common), **_metrics(y_common, p_g2)),
        TripleMetrics("tft_lite_labs", len(y_common), **_metrics(y_common, p_t2)),
    ]
    both_classes = int(len(np.unique(y_common)) > 1)
    primary = "pr_auc" if both_classes else "brier"
    for r in rows:
        r.primary_metric = primary

    lgbm_pr = rows[0].pr_auc
    lgbm_brier = rows[0].brier
    diffs: list[dict[str, Any]] = []
    for r in rows[1:]:
        r_dict = asdict(r)
        r_dict["vs_lgbm_pr_diff"] = (
            float(r.pr_auc - lgbm_pr) if both_classes and r.pr_auc == r.pr_auc else None
        )
        r_dict["vs_lgbm_brier_diff"] = float(r.brier - lgbm_brier)
        diffs.append(r_dict)

    start_h, end_h = _feature_window_hours(hour_index=h, lookback_hours=lookback_hours)
    payload: dict[str, Any] = {
        "status": "ok",
        "note": (
            "Lab-only temporal deepen: Layer0 chartevents empty; "
            "features=" + ",".join(LAB_FEATURE_NAMES) + ". "
            "Default bedside remains LightGBM. TFT-lite ≠ Lim 2021 full TFT. "
            "Stay pool mixes positives+negatives; split is stratified by stay. "
            f"Causal cutoff: labs with charttime < intime+{h}h "
            f"(window [{start_h:g}, {end_h:g}) h)."
        ),
        "n_sequences": len(valid),
        "hour_index": h,
        "lookback_hours": lookback_hours,
        "feature_window_hours": {"start": start_h, "end": end_h},
        "positive_rate": float(np.mean(labels)),
        "test_positive_rate": float(np.mean(y_common)),
        "n_test_pos": int(y_common.sum()),
        "lgbm_score_meta": lgbm_meta,
        "split": {
            "stratified": bool(split_meta.get("stratified")),
            "n_stays": split_meta.get("n_stays"),
            "class_counts": split_meta.get("class_counts"),
        },
        "rows": [asdict(r) for r in rows],
        "primary_metric": primary,
        "diffs": diffs,
    }
    if not both_classes:
        payload["caveat"] = "test set single-class → PR-AUC/ROC undefined; primary=Brier"
    if lgbm_meta["n_failed"]:
        payload["lgbm_caveat"] = (
            f"LGBM scored {lgbm_meta['n_ok']}/{lgbm_meta['n_requested']} test stays; "
            "metrics use the scored overlap only (see lgbm_score_meta.errors_head)."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "lab_triple_compare.json"
    md = [
        "# Lab temporal triple compare",
        "",
        payload["note"],
        "",
        f"- sequences: {payload['n_sequences']}",
        f"- hour_index (prediction t): {h}",
        f"- lookback_h (window start only): {lookback_hours}",
        f"- feature_window_h: [{start_h:g}, {end_h:g})",
        f"- positive_rate (all): {payload['positive_rate']:.4f}",
        f"- test_positive_rate: {payload['test_positive_rate']:.4f}",
        f"- n_test_pos: {payload['n_test_pos']}",
        f"- lgbm_ok/failed: {lgbm_meta['n_ok']}/{lgbm_meta['n_failed']}",
        f"- primary_metric: {primary}",
        f"- stratified: {payload['split']['stratified']}",
        "",
        "| model | n_test | PR-AUC | Brier | ROC (ref) |",
        "|-------|-------:|-------:|------:|----------:|",
    ]
    for r in rows:
        md.append(
            f"| {r.model} | {r.n_test} | {_fmt_metric(r.pr_auc)} | "
            f"{_fmt_metric(r.brier)} | {_fmt_metric(r.roc_auc)} |"
        )
    md_text = "\n".join(md) + "\n"
    json_text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
    # Stage both artifacts then publish together (avoid half-written / split runs).
    _atomic_write_text(path.with_suffix(".json.staging"), json_text)
    _atomic_write_text(path.with_suffix(".md.staging"), md_text)
    path.with_suffix(".json.staging").replace(path)
    path.with_suffix(".md.staging").replace(OUT_DIR / "lab_triple_compare.md")
    payload["artifact"] = str(path).replace("\\", "/")
    return payload
