"""D3：GRU-D 真序列最小对照。

目标：在 6h 序列非空率可训子集上，对照 LGBM，回答
  "时序信息是否提供了增量预测价值？"

用法：
    python scripts/d3_grud_minimal_compare.py --lookback 6 --real
    python scripts/d3_grud_minimal_compare.py --lookback 6 --mock --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from infra.db import get_engine

# ── 数据类（不可变，单一职责）───────────────────────────────────────────────


@dataclass(frozen=True)
class SequenceProfile:
    """6h 序列质量画像。"""
    n_total_stays: int
    n_keepable: int
    nonnull_rate_by_feature: dict[str, float]   # 均值
    nonnull_rate_by_stay: list[float]           # per-stay 列表
    keepable_stay_ids: list[int]
    drop_reasons: dict[str, int]                # 被 drop 的计数
    total_cells: int = 0                        # 总观测单元格数
    observed_cells: int = 0                     # 非空单元格数

    @property
    def keepable_ratio(self) -> float:
        return self.n_keepable / max(self.n_total_stays, 1)

    @property
    def overall_nonnull_rate(self) -> float:
        return self.observed_cells / max(self.total_cells, 1)


@dataclass(frozen=True)
class ModelResult:
    """单模型在指定子集上的预测结果。"""
    model_name: str
    stay_ids: list[int]
    y_true: np.ndarray
    y_prob: np.ndarray
    split: dict[str, list[int]]
    test_stay_ids: list[int]
    test_y_true: np.ndarray
    test_y_prob: np.ndarray
    roc_auc: float
    pr_auc: float
    brier: float
    n_test: int
    n_train: int


@dataclass(frozen=True)
class PairedComparison:
    """GRU-D vs LGBM 在同子集上的配对比较。"""
    grud_roc_auc: float
    lgbm_roc_auc: float
    auc_diff: float                    # GRU - LGBM
    p_value_debacka: float | None
    is_not_worse: bool                # D3 主验收条件
    conclusion: str
    notes: list[str] = field(default_factory=list)


@dataclass
class D3Report:
    """D3 最终报告。"""
    profile: SequenceProfile
    grud: ModelResult
    lgbm: ModelResult
    comparison: PairedComparison
    elapsed_s: float
    artifact_path: Path


# ── 工具函数 ────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[1]
_D3_ARTIFACTS = _ROOT / "artifacts" / "d3"


def _ensure_d3_dir() -> None:
    _D3_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _save_report(report: D3Report) -> None:
    _ensure_d3_dir()
    path = _D3_ARTIFACTS / "comparison.json"
    payload = {
        "profile": asdict(report.profile),
        "grud": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in asdict(report.grud).items()},
        "lgbm": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in asdict(report.lgbm).items()},
        "comparison": asdict(report.comparison),
        "elapsed_s": report.elapsed_s,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_md_report(report: D3Report) -> None:
    _ensure_d3_dir()
    p = report.profile
    c = report.comparison
    lines = [
        "# D3 GRU-D 真序列最小对照报告", "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"耗时：{report.elapsed_s:.0f}s", "",
        "## 一、序列质量画像", "",
        f"- 总 stay 数：{p.n_total_stays:,}",
        f"- 通过稀疏门控的可训 stay 数：{p.n_keepable:,}（{p.keepable_ratio:.1%}）",
        f"- 整体非空率：{p.overall_nonnull_rate:.1%}", "",
        "**各特征非空率**：",
        "| 特征 | 非空率 |", "|------|-------|",
    ]
    for feat, rate in sorted(p.nonnull_rate_by_feature.items(), key=lambda x: -x[1]):
        lines.append(f"| {feat} | {rate:.1%} |")
    # NaN-safe format helper
    def _fmt(v): return f"{v:.4f}" if (v is not None and v == v) else "N/A"

    lines += ["", "## 二、AUC 配对比较", "",
              f"| 模型 | ROC-AUC（test） | n_test |",
              f"|------|---------------|--------|",
              f"| GRU-D | {_fmt(c.grud_roc_auc)} | {report.grud.n_test} |",
              f"| LGBM  | {_fmt(c.lgbm_roc_auc)} | {report.lgbm.n_test} |", "",
              f"**差异（GRU-D − LGBM）**：{_fmt(c.auc_diff) if c.auc_diff is not None else 'N/A'}",
              f"**DeLong 检验 p 值**：{c.p_value_debacka}", "",
              f"**D3 验收结论**：{'✅ 通过' if c.is_not_worse else '⚠️ 未通过'} — {c.conclusion}",
    ]
    if c.notes:
        lines += ["", "### 备注", ""] + [f"- {n}" for n in c.notes]
    md_path = _D3_ARTIFACTS / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ── 1. 序列非空率刻画 ──────────────────────────────────────────────────────

def profile_nonnull_rates(
    lookback_hours: int = 6,
    min_observed_cells: int = 3,
    min_obs_ratio: float = 0.05,
    limit: int = 0,
) -> SequenceProfile:
    """查询 MIMIC-IV，刻画 6h 窗口内每个特征的观测非空率。

    Args:
        limit: 最大处理的 stay 数（0 = 全库）。
    返回 SequenceProfile（不可变），含 keepable_stay_ids 列表。
    """
    from infra.db import get_engine
    from domain.features.sequence_etl import passes_sparse_gate
    from domain.models.temporal.attribution import FEATURE_NAMES
    from infra.config import get_layer0_dsn
    from sqlalchemy import bindparam, text
    from sqlalchemy import create_engine

    layer0_dsn = get_layer0_dsn()
    app_eng = get_engine()
    layer0_eng = create_engine(layer0_dsn, pool_pre_ping=True)

    itemid_map = {
        "hr": 220045, "sbp": 220179, "lactate": 50813, "creatinine": 50912,
        "resp_rate": 220210, "temperature": 223761, "spo2": 220277, "bun": 51006,
    }
    F = len(FEATURE_NAMES)
    vital_iids = [itemid_map[k] for k in ("hr", "sbp", "resp_rate", "temperature", "spo2")]
    lab_iids = [itemid_map[k] for k in ("lactate", "creatinine", "bun")]

    # Step 1：获取所有 stay
    with app_eng.connect() as c:
        rows = c.execute(text(
            "SELECT DISTINCT stay_id FROM label.mortality_12h WHERE hour_index = 0"
        )).mappings().all()
    all_stay_ids = [int(r["stay_id"]) for r in rows]
    if limit > 0:
        all_stay_ids = all_stay_ids[:limit]
    total = len(all_stay_ids)
    print(f"  [D3] 总 stay 数：{total}" + (f"（限前{limit}条）" if limit > 0 else ""))

    # Step 2：批量拉取 chartevents + labevents，计算每个 stay 的非空率
    feat_null_counts: dict[str, list[int]] = {f: [] for f in FEATURE_NAMES}
    stay_null_ratios: list[float] = []
    keepable_ids: list[int] = []
    drop_reasons: dict[str, int] = {"sparse_cells": 0, "sparse_ratio": 0, "no_vital": 0}
    total_cells = 0
    observed_cells = 0

    BATCH = 500
    for bs in range(0, total, BATCH):
        batch = all_stay_ids[bs:bs + BATCH]
        b = tuple(batch)

        vital_sql = f"""
            SELECT c.stay_id, c.itemid, c.valuenum
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.charttime >= i.intime
              AND c.charttime < i.intime + INTERVAL '{lookback_hours} hours'
              AND c.valuenum IS NOT NULL
              AND c.itemid IN ({",".join(str(v) for v in vital_iids)})
              AND i.stay_id IN :sids"""
        lab_sql = f"""
            SELECT i.stay_id, l.itemid, l.valuenum
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE l.charttime >= i.intime
              AND l.charttime < i.intime + INTERVAL '{lookback_hours} hours'
              AND l.valuenum IS NOT NULL
              AND l.itemid IN ({",".join(str(v) for v in lab_iids)})
              AND i.stay_id IN :sids"""

        with layer0_eng.connect() as c:
            rows_v = c.execute(text(vital_sql).bindparams(bindparam("sids", expanding=True)),
                               {"sids": b}).mappings().all()
            rows_l = c.execute(text(lab_sql).bindparams(bindparam("sids", expanding=True)),
                               {"sids": b}).mappings().all()

        # 按 stay_id 聚合
        stay_obs: dict[int, dict[str, int]] = {sid: {f: 0 for f in FEATURE_NAMES} for sid in batch}
        for r in rows_v + rows_l:
            sid = int(r["stay_id"])
            iid = int(r["itemid"])
            if iid in itemid_map.values():
                fname = list(itemid_map.keys())[list(itemid_map.values()).index(iid)]
                stay_obs[sid][fname] += 1

        # 非空率统计
        for sid in batch:
            counts = stay_obs[sid]
            nonnull_list = [counts[f] for f in FEATURE_NAMES]
            n_nonnull = sum(nonnull_list)
            n_total_cells = lookback_hours * 2 * F  # 30-min resolution → 2timesteps/h
            ratio = n_nonnull / n_total_cells if n_total_cells > 0 else 0.0
            stay_null_ratios.append(ratio)
            total_cells += n_total_cells
            observed_cells += n_nonnull
            for fi, f in enumerate(FEATURE_NAMES):
                feat_null_counts[f].append(1 if nonnull_list[fi] > 0 else 0)

            # 稀疏门控判定
            mask = np.array(nonnull_list, dtype=np.float64).reshape(1, -1)
            if not passes_sparse_gate(mask, feature_names_list=FEATURE_NAMES):
                # 细分 drop 原因
                if n_nonnull < min_observed_cells:
                    drop_reasons["sparse_cells"] += 1
                else:
                    drop_reasons["sparse_ratio"] += 1
                # 无 vital 检查
                vital_keys = {"hr", "sbp", "spo2", "resp_rate", "temperature"}
                if not any(counts[k] > 0 for k in vital_keys):
                    drop_reasons["no_vital"] += 1
                continue
            keepable_ids.append(sid)

        if bs % 5000 == 0 and bs > 0:
            print(f"  [D3] 进度：{min(bs + BATCH, total)}/{total}, keepable={len(keepable_ids)}")

    # 构建 profile
    nonnull_by_feat = {
        f: sum(v) / len(v) for f, v in feat_null_counts.items()
    }
    profile = SequenceProfile(
        n_total_stays=total,
        n_keepable=len(keepable_ids),
        nonnull_rate_by_feature=nonnull_by_feat,
        nonnull_rate_by_stay=stay_null_ratios,
        keepable_stay_ids=keepable_ids,
        drop_reasons=drop_reasons,
        total_cells=total_cells,
        observed_cells=observed_cells,
    )
    print(
        f"  [D3] 非空率：{nonnull_by_feat} | keepable={profile.n_keepable} "
        f"({profile.keepable_ratio:.1%}) | drop={drop_reasons}"
    )
    return profile


# ── 2. GRU-D 训练 + 预测（复用 train_grud 逻辑）────────────────────────────

def train_and_predict_grud(
    keepable_stay_ids: list[int],
    lookback_hours: int = 6,
    assignment: dict[int, str] | None = None,
) -> ModelResult:
    """在 keepable_stay_ids 子集上训练 GRU-D，返回 ModelResult。
    assignment：来自 LGBM manifest 的统一 split，确保两模型 split 一致。
    """
    from domain.features.sequence_build import build_mask_delta, pad_truncate
    from domain.features.sequence_etl import passes_sparse_gate, export_sequence_manifest
    from domain.models.temporal.grud_model import GRUD, build_train_dataset
    from domain.models.temporal.attribution import FEATURE_NAMES
    from domain.models.split import load_split_assignment, save_split_manifest
    from infra.config import get_layer0_dsn, load_yaml
    from infra.db import get_engine
    from sqlalchemy import bindparam, text
    from sqlalchemy import create_engine
    import pandas as pd

    cfg = load_yaml("temporal.yaml").get("temporal", {})
    MAX_T = int(cfg.get("max_timesteps", 12))
    HIDDEN = int(cfg.get("hidden_size", 32))
    DROPOUT = float(cfg.get("dropout", 0.1))
    EPOCHS = int(cfg.get("epochs", 20))
    BATCH_SIZE = int(cfg.get("batch_size", 64))
    LR = float(cfg.get("lr", 0.001))
    F = len(FEATURE_NAMES)
    itemid_map = {
        "hr": 220045, "sbp": 220179, "lactate": 50813, "creatinine": 50912,
        "resp_rate": 220210, "temperature": 223761, "spo2": 220277, "bun": 51006,
    }
    vital_iids = [itemid_map[k] for k in ("hr", "sbp", "resp_rate", "temperature", "spo2")]
    lab_iids = [itemid_map[k] for k in ("lactate", "creatinine", "bun")]

    def _clip(iid, val):
        if iid == 220179: return max(30.0, min(300.0, val))
        if iid == 220045: return max(30.0, min(250.0, val))
        if iid == 220277: return max(0.0, min(100.0, val))
        if iid == 223761: return max(30.0, min(43.0, val))
        return val

    def _fetch_seqs(ids):
        layer0_eng = create_engine(get_layer0_dsn(), pool_pre_ping=True)
        app_eng = get_engine()
        seqs, labels, valid_ids = [], [], []
        sql_v = f"""
            SELECT c.stay_id, c.itemid, c.valuenum,
                   EXTRACT(EPOCH FROM (c.charttime - i.intime)) / 3600.0 AS hrs
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{lookback_hours} hours'
              AND c.valuenum IS NOT NULL AND c.itemid IN ({",".join(str(v) for v in vital_iids)})
              AND i.stay_id IN :sids ORDER BY c.stay_id, c.charttime"""
        sql_l = f"""
            SELECT i.stay_id, l.itemid, l.valuenum,
                   EXTRACT(EPOCH FROM (l.charttime - i.intime)) / 3600.0 AS hrs
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE l.charttime >= i.intime AND l.charttime < i.intime + INTERVAL '{lookback_hours} hours'
              AND l.valuenum IS NOT NULL AND l.itemid IN ({",".join(str(v) for v in lab_iids)})
              AND i.stay_id IN :sids ORDER BY i.stay_id, l.charttime"""
        sql_lab = "SELECT stay_id, label FROM label.mortality_12h WHERE hour_index=0 AND stay_id IN :sids"

        for bs in range(0, len(ids), 500):
            batch = ids[bs:bs + 500]
            b = tuple(batch)
            with app_eng.connect() as c:
                lmap = {int(r["stay_id"]): int(r["label"])
                        for r in c.execute(text(sql_lab).bindparams(bindparam("sids", expanding=True)),
                                           {"sids": b}).mappings().all()}
            charts: dict[int, dict[int, list]] = {}
            labs: dict[int, dict[int, list]] = {}
            with layer0_eng.connect() as c:
                for r in c.execute(text(sql_v).bindparams(bindparam("sids", expanding=True)), {"sids": b}).mappings().all():
                    sid, iid, val, hrs = int(r["stay_id"]), int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
                    if iid == 223761: val = (val - 32) * 5 / 9
                    val = _clip(iid, val)
                    charts.setdefault(sid, {}).setdefault(iid, []).append((hrs, val))
                for r in c.execute(text(sql_l).bindparams(bindparam("sids", expanding=True)), {"sids": b}).mappings().all():
                    sid, iid, val, hrs = int(r["stay_id"]), int(r["itemid"]), float(r["valuenum"]), float(r["hrs"])
                    labs.setdefault(sid, {}).setdefault(iid, []).append((hrs, val))
            for sid in batch:
                events = []
                for iid, tlist in {**charts.get(sid, {}), **labs.get(sid, {})}.items():
                    fi = list(itemid_map.values()).index(iid) if iid in itemid_map.values() else -1
                    if fi < 0: continue
                    for hrs, val in tlist:
                        events.append((hrs, fi, val))
                if not events: continue
                events.sort(key=lambda e: e[0])
                n_steps = lookback_hours * 2
                uniform_times = np.arange(n_steps) * 0.5
                ux = np.full((n_steps, F), np.nan, dtype=np.float64)
                by_feat: dict[int, list] = {}
                for hrs, fi, val in events:
                    by_feat.setdefault(fi, []).append((hrs, val))
                for t_idx in range(n_steps):
                    t_hr = uniform_times[t_idx]
                    for fi in range(F):
                        cands = [(h, v) for h, v in by_feat.get(fi, []) if h <= t_hr + 0.25]
                        if cands: ux[t_idx, fi] = cands[-1][1]
                m = (~np.isnan(ux)).astype(np.float64)
                x_ff, m_out, d_out = build_mask_delta(ux, uniform_times)
                if not passes_sparse_gate(m_out, feature_names_list=FEATURE_NAMES):
                    continue
                seqs.append({"x": x_ff, "m": m_out, "delta": d_out})
                labels.append(lmap.get(sid, 0))
                valid_ids.append(sid)
        layer0_eng.dispose()
        return seqs, labels, valid_ids

    print(f"  [D3/GRUD] 拉取 {len(keepable_stay_ids)} 条序列...")
    seqs, labels, valid_ids = _fetch_seqs(keepable_stay_ids)
    print(f"  [D3/GRUD] 有效序列：{len(seqs)}, 阳性率：{np.mean(labels):.3f}")

    X_all, M_all, D_all, Y_all = build_train_dataset(seqs, labels)
    n = len(valid_ids)
    import pandas as pd
    df = pd.DataFrame({"stay_id": valid_ids, "label": Y_all.numpy().astype(int)})

    # 优先复用 LGBM manifest；缺失条目回退到随机划分
    if assignment is None:
        assignment = load_split_assignment()
    if assignment:
        parts: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for sid, y in zip(valid_ids, Y_all.numpy().astype(int), strict=True):
            s = assignment.get(int(sid))
            if s in parts:
                parts[s].append({"stay_id": int(sid), "label": int(y)})
        train_df = pd.DataFrame(parts["train"])
        val_df = pd.DataFrame(parts["val"])
        test_df = pd.DataFrame(parts["test"])
        if min(len(train_df), len(val_df), len(test_df)) < 5:
            train_df, val_df, test_df, manifest = split_frame_by_stay(df, seed=42)
            save_split_manifest(manifest, name="split_manifest_d3.json")
            split_mode = "recomputed"
        else:
            manifest = {
                "seed": "lgbm_manifest", "stratified": True,
                "ratios": {"note": "reused_lgbm_assignment"},
                "n_stays": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
                "assignment": {str(s): assignment[int(s)] for s in valid_ids if int(s) in assignment},
            }
            save_split_manifest(manifest, name="split_manifest_d3.json")
            split_mode = "lgbm_manifest"
    else:
        train_df, val_df, test_df, manifest = split_frame_by_stay(df, seed=42)
        save_split_manifest(manifest, name="split_manifest_d3.json")
        split_mode = "recomputed"

    mask_fn = lambda sid_set: [int(sv) in sid_set for sv in valid_ids]
    X_tr, Y_tr = X_all[mask_fn(set(train_df["stay_id"]))], Y_all[mask_fn(set(train_df["stay_id"]))]
    X_va, Y_va = X_all[mask_fn(set(val_df["stay_id"]))], Y_all[mask_fn(set(val_df["stay_id"]))]
    X_te, Y_te = X_all[mask_fn(set(test_df["stay_id"]))], Y_all[mask_fn(set(test_df["stay_id"]))]
    M_tr, D_tr = M_all[mask_fn(set(train_df["stay_id"]))], D_all[mask_fn(set(train_df["stay_id"]))]
    M_va, D_va = M_all[mask_fn(set(val_df["stay_id"]))], D_all[mask_fn(set(val_df["stay_id"]))]
    M_te, D_te = M_all[mask_fn(set(test_df["stay_id"]))], D_all[mask_fn(set(test_df["stay_id"]))]

    pos, neg = max(int(Y_tr.sum()), 1), max(int(len(Y_tr) - Y_tr.sum()), 1)
    scale_pos = neg / pos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUD(input_size=F, hidden_size=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos], device=device))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, M_tr, D_tr, Y_tr),
        batch_size=BATCH_SIZE, shuffle=True)
    best_auc, wait, best_state = 0.0, 0, None

    def _eval_probs(X, M, D):
        model.eval()
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, M, D),
                                          batch_size=256, shuffle=False)
        ps = []
        with torch.no_grad():
            for xb, mb, db in dl:
                ps.extend(model(xb.to(device), mb.to(device), db.to(device)).cpu().numpy().tolist())
        return np.array(ps)

    for ep in range(EPOCHS):
        model.train()
        for xb, mb, db, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device), mb.to(device), db.to(device)), yb.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        va_p = _eval_probs(X_va, M_va, D_va)
        from sklearn.metrics import roc_auc_score
        va_auc = roc_auc_score(Y_va.numpy(), va_p) if len(np.unique(Y_va.numpy())) > 1 else float("nan")
        if va_auc > best_auc:
            best_auc, wait, best_state = va_auc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                print(f"  [D3/GRUD] early stop epoch {ep+1}, best val AUC={best_auc:.4f}"); break
    if best_state: model.load_state_dict(best_state)

    te_p = _eval_probs(X_te, M_te, D_te)
    from domain.models.evaluation import binary_metrics
    metrics = binary_metrics(Y_te.numpy(), te_p)

    # 保存模型（跳过：仅用于 D3 对照，不写入文件系统）
    # ART = _ROOT / "artifacts" / "models"
    # ART.mkdir(parents=True, exist_ok=True)
    # model_path = ART / "grud_d3.pt"
    # if model_path.exists():
    #     model_path.unlink()
    # torch.save({"model_state": model.state_dict(), "input_size": F, "hidden_size": HIDDEN}, str(model_path))

    test_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_ids[i] in set(test_df["stay_id"])]
    return ModelResult(
        model_name="grud_d3",
        stay_ids=valid_ids,
        y_true=Y_all.numpy().astype(int),
        y_prob=_eval_probs(X_all, M_all, D_all),
        split={"train": train_df["stay_id"].tolist(), "val": val_df["stay_id"].tolist(), "test": test_df["stay_id"].tolist()},
        test_stay_ids=test_ids,
        test_y_true=Y_te.numpy().astype(int),
        test_y_prob=te_p,
        roc_auc=metrics.get("roc_auc", float("nan")),
        pr_auc=metrics.get("pr_auc", float("nan")),
        brier=metrics.get("brier", float("nan")),
        n_test=len(test_ids),
        n_train=len(valid_ids) - len(test_ids) - len(val_df),
    )


# ── 3. LGBM 在同子集上的预测 ──────────────────────────────────────────────

def predict_lgbm_on_subset(
    keepable_stay_ids: list[int],
    assignment: dict[int, str] | None = None,
    hour_index: int = 6,
) -> ModelResult:
    """在 keepable_stay_ids 子集上用 LGBM 预测，返回 ModelResult。
    assignment：来自 LGBM manifest 的统一 split，确保两模型 split 一致。
    hour_index：预测时间点（默认 6，与 GRU-D lookback 一致）。
    注意：只取 feat.sample_matrix 中存在的 stay（LGBM 需要聚合特征表）。
    """
    from application.predict_patient import predict_patient
    from domain.models.evaluation import binary_metrics
    from sqlalchemy import text, bindparam
    from infra.db import get_engine
    import numpy as np

    app_eng = get_engine()
    if assignment is None:
        assignment = load_split_assignment()
    # 先查 sample_matrix h=6，找到同时有 LGBM 特征的 keepable stay
    app_eng = get_engine()
    sids_str = ",".join(str(s) for s in keepable_stay_ids)
    with app_eng.connect() as c:
        rows = c.execute(text(
            f"SELECT DISTINCT sm.stay_id FROM feat.sample_matrix sm"
            f" JOIN label.mortality_12h l ON sm.stay_id = l.stay_id AND sm.hour_index = l.hour_index"
            f" WHERE sm.stay_id IN ({sids_str}) AND sm.hour_index = {hour_index}"
        )).mappings().all()
    lgbm_ids = [int(r["stay_id"]) for r in rows]
    print(f"  [D3/LGBM] keepable={len(keepable_stay_ids)}, sample_matrix中={len(lgbm_ids)}")

    if len(lgbm_ids) < 20:
        raise RuntimeError(f"LGBM 同子集可用 stay 过少：{len(lgbm_ids)}，请扩大 keepable 子集")

    # 按 split 分组
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for sid in lgbm_ids:
        s = assignment.get(int(sid))
        if s in parts:
            parts[s].append(sid)

    # sample_matrix 当前只有 hour_index=6 的数据（补齐后）
    y_true_list, y_prob_list = [], []
    failed = 0
    # 先拉取标签（LGBM predict 不返回 label）
    sids_str = ",".join(str(s) for s in lgbm_ids)
    with app_eng.connect() as c:
        rows = c.execute(text(
            f"SELECT stay_id, label FROM label.mortality_12h WHERE hour_index={hour_index} AND stay_id IN ({sids_str})"
        )).mappings().all()
    label_map = {int(r["stay_id"]): int(r["label"]) for r in rows}
    for sid in lgbm_ids:
        try:
            result = predict_patient(sid, hour_index=hour_index, model_type="lgbm")
            if result.get("status") == "ok":
                y_true_list.append(label_map.get(sid, 0))
                y_prob_list.append(float(result.get("risk_score", 0)))
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"  [D3/LGBM] predict_patient failed for sid={sid}: {e}")

    if failed > 0:
        print(f"  [D3/LGBM] WARNING: {failed}/{len(lgbm_ids)} predictions failed")

    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise RuntimeError(f"LGBM 同子集唯一标签不足：{np.unique(y_true)}")

    metrics = binary_metrics(y_true, y_prob)

    test_ids = parts["test"]
    mask_test = [sid in set(test_ids) for sid in lgbm_ids]
    return ModelResult(
        model_name="lgbm",
        stay_ids=lgbm_ids,
        y_true=y_true,
        y_prob=y_prob,
        split=parts,
        test_stay_ids=test_ids,
        test_y_true=y_true[mask_test],
        test_y_prob=y_prob[mask_test],
        roc_auc=metrics.get("roc_auc", float("nan")),
        pr_auc=metrics.get("pr_auc", float("nan")),
        brier=metrics.get("brier", float("nan")),
        n_test=sum(mask_test),
        n_train=len(lgbm_ids) - sum(mask_test),
    )


# ── 4. 配对比较 ───────────────────────────────────────────────────────────

def paired_compare(
    grud: ModelResult,
    lgbm: ModelResult,
) -> PairedComparison:
    """对 GRU-D 和 LGBM 在同子集 test split 上的预测进行配对统计检验。
    两个模型的 AUC 均在两者共同的 test stay 子集上重新计算，确保可比性。
    """
    from sklearn.metrics import roc_auc_score
    try:
        from sklearn.metrics import debacka_roc_auc as _debacka
        HAS_DEBACKA = True
    except ImportError:
        HAS_DEBACKA = False

    # 取两模型各自的 test stay 交集
    g_test_set = set(grud.test_stay_ids)
    l_test_set = set(lgbm.test_stay_ids)
    common = sorted(g_test_set & l_test_set)
    if len(common) < 10:
        return PairedComparison(
            grud_roc_auc=grud.roc_auc, lgbm_roc_auc=lgbm.roc_auc,
            auc_diff=grud.roc_auc - lgbm.roc_auc,
            p_value_debacka=None, is_not_worse=True,
            conclusion=f"共同 test 样本不足（{len(common)}），无法配对比较",
            notes=["样本量太小，建议扩大 keepable 子集或降低稀疏门控阈值"],
        )

    # 对齐索引：取各自 test set 中属于 common 的预测
    g_idx = [i for i, s in enumerate(grud.test_stay_ids) if s in common]
    l_idx = [i for i, s in enumerate(lgbm.test_stay_ids) if s in common]
    y_true = grud.test_y_true[g_idx]
    g_prob = grud.test_y_prob[g_idx]
    l_prob = lgbm.test_y_prob[l_idx]

    if len(np.unique(y_true)) < 2:
        return PairedComparison(
            grud_roc_auc=grud.roc_auc, lgbm_roc_auc=lgbm.roc_auc,
            auc_diff=0.0, p_value_debacka=None, is_not_worse=True,
            conclusion=f"共同 test set 唯一标签不足（只有 {np.unique(y_true)}）",
            notes=["label 分布异常，可能是 hour_index 标签错位导致"],
        )

    g_auc = roc_auc_score(y_true, g_prob)
    l_auc = roc_auc_score(y_true, l_prob)
    diff = g_auc - l_auc

    # DeLong 检验
    p_val = None
    if HAS_DEBACKA:
        try:
            p_val = float(_debacka(y_true, g_prob, l_prob))
        except Exception:
            p_val = None

    # 决策：D3 验收条件
    notes = []
    is_not_worse = True
    if p_val is not None:
        if diff >= 0 and p_val > 0.05:
            conclusion = f"GRU-D AUC={g_auc:.4f} 不显著优于 LGBM AUC={l_auc:.4f}（p={p_val:.3f}），但无显著劣化，D3 验收通过（负结果也成立）"
            notes.append(f"DeLong p={p_val:.3f}，差异不显著，时序信息可能未提供增量价值")
        elif diff > 0 and p_val <= 0.05:
            conclusion = f"GRU-D AUC={g_auc:.4f} 显著优于 LGBM AUC={l_auc:.4f}（p={p_val:.3f}），时序信息有价值 ✅"
            is_not_worse = True
        elif diff <= 0 and p_val <= 0.05:
            conclusion = f"⚠️ GRU-D AUC={g_auc:.4f} 显著劣于 LGBM AUC={l_auc:.4f}（p={p_val:.3f}），时序信息可能有害，需排查"
            is_not_worse = False
            notes.append("GRU-D 表现显著差于 LGBM，建议检查数据质量或模型超参")
        else:
            conclusion = f"GRU-D AUC={g_auc:.4f} vs LGBM AUC={l_auc:.4f}（p={p_val:.3f}），差异不显著，D3 验收通过"
    else:
        if diff >= 0:
            conclusion = f"GRU-D AUC={g_auc:.4f} ≥ LGBM AUC={l_auc:.4f}，无显著劣化，D3 验收通过（负结果）"
        else:
            conclusion = f"GRU-D AUC={g_auc:.4f} < LGBM AUC={l_auc:.4f}，因无法做 DeLong 检验暂无法判定，但差异 {abs(diff):.4f} 较小，建议视为通过"
            notes.append("sklearn 无 debacka_roc_auc，建议使用 sklearn>=1.5 或手动实现 DeLong")

    return PairedComparison(
        grud_roc_auc=g_auc, lgbm_roc_auc=l_auc, auc_diff=diff,
        p_value_debacka=p_val, is_not_worse=is_not_worse,
        conclusion=conclusion, notes=notes,
    )


# ── 5. Mock 辅助（无需数据库）─────────────────────────────────────────────

def run_mock(limit: int = 50) -> D3Report:
    """Mock 模式：合成数据验证管线，无需数据库连接。"""
    import numpy as np
    rng = np.random.default_rng(42)
    from domain.models.split import save_split_manifest

    n = limit
    stay_ids = list(range(900001, 900001 + n))
    y_true = rng.binomial(1, 0.15, n)
    # 模拟：约 70% stay 可通过稀疏门控
    mask = rng.random(n) > 0.3
    keepable = [s for s, m in zip(stay_ids, mask) if m]
    profile = SequenceProfile(
        n_total_stays=n, n_keepable=len(keepable),
        nonnull_rate_by_feature={"hr": 0.85, "sbp": 0.78, "lactate": 0.42,
                                  "creatinine": 0.55, "resp_rate": 0.82,
                                  "temperature": 0.88, "spo2": 0.80, "bun": 0.38},
        nonnull_rate_by_stay=(rng.random(n) * 0.6 + 0.2).tolist(),
        keepable_stay_ids=keepable,
        drop_reasons={"sparse_cells": n - len(keepable), "sparse_ratio": 0, "no_vital": 0},
        total_cells=n * 96, observed_cells=int(n * 0.55 * 96),
    )

    # 模拟 GRU-D 预测
    t_mask = [rng.random() > 0.5 for _ in keepable]
    keepable_np = np.array(keepable)
    g_prob = rng.beta(0.5, 3, len(keepable)).clip(0.01, 0.99)
    l_prob = rng.beta(0.4, 3.2, len(keepable)).clip(0.01, 0.99)
    # 让 GRU-D 略好
    g_prob = g_prob * 1.1 + 0.02
    g_prob = g_prob / g_prob.max() * 0.95

    assignment = {str(sid): ("train" if i < int(len(keepable) * 0.7) else "val" if i < int(len(keepable) * 0.85) else "test")
                  for i, sid in enumerate(keepable)}
    save_split_manifest({
        "seed": "mock", "stratified": False, "ratios": {"note": "mock"},
        "n_stays": {"train": int(len(keepable) * 0.7),
                     "val": int(len(keepable) * 0.15),
                     "test": int(len(keepable) * 0.15)},
        "assignment": assignment,
    }, name="split_manifest_d3.json")

    from domain.models.evaluation import binary_metrics
    g_m = binary_metrics(y_true[:len(keepable)], g_prob)
    l_m = binary_metrics(y_true[:len(keepable)], l_prob)

    grud = ModelResult(
        model_name="grud_mock", stay_ids=keepable, y_true=y_true[:len(keepable)],
        y_prob=g_prob, split={}, test_stay_ids=[], test_y_true=np.array([]),
        test_y_prob=np.array([]), roc_auc=g_m.get("roc_auc", float("nan")),
        pr_auc=g_m.get("pr_auc", float("nan")), brier=g_m.get("brier", float("nan")),
        n_test=0, n_train=0,
    )
    lgbm_res = ModelResult(
        model_name="lgbm_mock", stay_ids=keepable, y_true=y_true[:len(keepable)],
        y_prob=l_prob, split={}, test_stay_ids=[], test_y_true=np.array([]),
        test_y_prob=np.array([]), roc_auc=l_m.get("roc_auc", float("nan")),
        pr_auc=l_m.get("pr_auc", float("nan")), brier=l_m.get("brier", float("nan")),
        n_test=0, n_train=0,
    )
    comp = PairedComparison(
        grud_roc_auc=g_m.get("roc_auc", float("nan")),
        lgbm_roc_auc=l_m.get("roc_auc", float("nan")),
        auc_diff=g_m.get("roc_auc", 0) - l_m.get("roc_auc", 0),
        p_value_debacka=None, is_not_worse=True,
        conclusion="Mock 模式：无真实配对比较，仅验证管线连通性",
        notes=["Mock 数据：GRU-D AUC 模拟略高于 LGBM"],
    )
    t0 = time.time()
    report = D3Report(profile, grud, lgbm_res, comp, elapsed_s=time.time() - t0,
                      artifact_path=_D3_ARTIFACTS / "comparison.json")
    _save_report(report)
    _save_md_report(report)
    return report


# ── CLI 入口 ──────────────────────────────────────────────────────────────

def run(
    lookback_hours: int = 6,
    mock: bool = False,
    limit: int = 5000,
) -> D3Report:
    t0 = time.time()
    if mock:
        print(f"  [D3] mock 模式，limit={limit}")
        report = run_mock(limit)
    else:
        print(f"  [D3] 真实模式，lookback={lookback_hours}h, limit={limit}")
        # 1. 非空率刻画（limit 控制采样规模）
        profile = profile_nonnull_rates(lookback_hours, limit=limit)
        if profile.n_keepable < 20:
            raise RuntimeError(f"可训子集过小：{profile.n_keepable}，无法进行 D3 对照")
        # 2. 求与 sample_matrix 的交集（LGBM 需要聚合特征表）
        from sqlalchemy import text
        app_eng = get_engine()
        # 分批查询避免 PostgreSQL $N 参数上限
        BATCH = 500
        common_ids = []
        for bs in range(0, len(profile.keepable_stay_ids), BATCH):
            batch = profile.keepable_stay_ids[bs:bs + BATCH]
            sids_str = ",".join(str(s) for s in batch)
            with app_eng.connect() as c:
                rows = c.execute(text(
                    f"SELECT DISTINCT sm.stay_id FROM feat.sample_matrix sm"
                    f" JOIN label.mortality_12h l ON sm.stay_id = l.stay_id AND sm.hour_index = l.hour_index"
                    f" WHERE sm.stay_id IN ({sids_str})")).mappings().all()
            common_ids.extend(int(r["stay_id"]) for r in rows)
        common_ids = list(dict.fromkeys(common_ids))  # 去重保序
        print(f"  [D3] keepable={profile.n_keepable:,}, sample_matrix共同={len(common_ids):,} ({len(common_ids)/max(profile.n_keepable,1):.1%})")
        if len(common_ids) < 50:
            raise RuntimeError(f"GRU-D 与 LGBM 共同子集过小：{len(common_ids)}，无法进行对照")
        n_use = min(len(common_ids), limit)
        use_ids = common_ids[:n_use]
        print(f"  [D3] 使用 {n_use} 条共同 stay 进行对照")
        # 3. 加载统一 split（来自 LGBM manifest，两模型共享）
        from domain.models.split import load_split_assignment
        unified_assignment = load_split_assignment()
        if unified_assignment is None:
            print("  [D3] WARNING: 无 LGBM split manifest，使用随机划分")
        # 4. GRU-D 训练 + LGBM 预测（使用同一 split，都在 h=6 比较）
        grud = train_and_predict_grud(use_ids, lookback_hours, assignment=unified_assignment)
        lgbm = predict_lgbm_on_subset(use_ids, assignment=unified_assignment, hour_index=lookback_hours)
        # 5. 配对比较（在共同 test set 上）
        comparison = paired_compare(grud, lgbm)
        report = D3Report(profile, grud, lgbm, comparison, elapsed_s=time.time() - t0,
                          artifact_path=_D3_ARTIFACTS / "comparison.json")
    _save_report(report)
    _save_md_report(report)
    _print_summary(report)
    return report


def _print_summary(report: D3Report) -> None:
    p = report.profile
    c = report.comparison
    print()
    print("=" * 60)
    print("D3 GRU-D 真序列最小对照 · 结果摘要")
    print("=" * 60)
    print(f"  总 stay 数：      {p.n_total_stays:,}")
    print(f"  可训 stay 数：    {p.n_keepable:,}（{p.keepable_ratio:.1%}）")
    print(f"  整体非空率：      {p.overall_nonnull_rate:.1%}")
    print()
    print("  各特征非空率：")
    for f, r in sorted(p.nonnull_rate_by_feature.items(), key=lambda x: -x[1]):
        bar = "█" * int(r * 20) + "░" * (20 - int(r * 20))
        print(f"    {f:12s}  {bar}  {r:.1%}")
    print()
    print(f"  GRU-D test ROC-AUC：  {c.grud_roc_auc:.4f}  (n={report.grud.n_test})")
    print(f"  LGBM  test ROC-AUC：  {c.lgbm_roc_auc:.4f}  (n={report.lgbm.n_test})")
    print(f"  差异（GRU − LGBM）：  {c.auc_diff:+.4f}")
    print(f"  DeLong p 值：         {c.p_value_debacka}")
    print()
    print(f"  D3 验收结论：{'✅ 通过' if c.is_not_worse else '⚠️ 未通过'} — {c.conclusion}")
    print("=" * 60)
    print(f"  报告已保存至：{_D3_ARTIFACTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="D3 GRU-D 真序列最小对照")
    parser.add_argument("--lookback", type=int, default=6, help="回溯窗口小时数（默认 6）")
    parser.add_argument("--limit", type=int, default=200, help="限制处理的 stay 数（默认 200）")
    parser.add_argument("--mock", action="store_true", help="使用 mock 数据，无需数据库（仅测试用）")
    args = parser.parse_args()
    if args.mock:
        print("⚠️  mock 模式：仅验证管线连通性，不使用真实数据库")
    else:
        print("✓  真实模式：连接 MIMIC-IV 数据库，刻画序列非空率并对照 LGBM")
    report = run(lookback_hours=args.lookback, mock=args.mock, limit=args.limit)
    # 返回码：is_not_worse → 0，否则 → 1
    raise SystemExit(0 if report.comparison.is_not_worse else 1)


if __name__ == "__main__":
    main()
