from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from application.acceptance import load_metrics_artifact, layer1_counts
from application.predict_patient import (
    get_label_config,
    list_stays,
    predict_patient,
    predict_patient_trajectory,
)
from domain.features.build import prediction_hours
from infra.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "STATUS.md"
FEATURES_YAML = ROOT / "configs" / "features.yaml"
LABELS_YAML = ROOT / "configs" / "labels.yaml"

st.set_page_config(page_title="ICU Decision", layout="wide")
st.title("ICU 临床恶化预警 · 交互台")
st.caption("icu-decision-agent · S2 多时刻 · 细调 / 运行 / 验收")

labels_cfg = get_label_config()
st.sidebar.subheader("标签配置")
st.sidebar.json(labels_cfg.get("primary", {}))
st.sidebar.markdown("### 路径")
st.sidebar.code(str(ROOT), language=None)
st.sidebar.markdown("- dump：`dumps/icu_decision_S2-full_*`")
st.sidebar.markdown("- MLflow：`mlflow ui --backend-store-uri sqlite:///./mlflow.db`")

tab_pred, tab_tune, tab_accept, tab_lab = st.tabs(
    ["多时刻预测", "细调运行", "验收门禁", "STATUS"]
)

with tab_pred:
    stays = list(list_stays(limit=500))
    if not stays:
        st.warning("未找到 ICU stays。请先 restore S2 dump 或跑 ETL。")
        st.stop()

    options = {
        f"stay {s['stay_id']} · LOS {float(s.get('los_hours') or 0):.1f}h": s["stay_id"]
        for s in stays
    }
    label = st.selectbox("选择 ICU stay", list(options.keys()))
    stay_id = options[label]
    hours = prediction_hours()
    hour = st.selectbox("预测时刻 hour_index", hours, index=min(1, len(hours) - 1))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("计算该时刻风险", type="primary"):
            result = predict_patient(stay_id, hour_index=int(hour))
            if result.get("status") != "ok":
                st.error(result.get("message", result.get("status")))
            else:
                score = result["risk_score"]
                kind = result.get("score_kind", "raw")
                if kind == "probability":
                    st.metric(f"h={hour} · 12h mortality risk", f"{score:.2%}")
                else:
                    st.metric(f"h={hour} · model score (raw)", f"{score:.4f}")
                rec = result.get("recommend") or {}
                if rec:
                    st.info(f"建议档位：**{rec.get('label', rec.get('band'))}**")
                st.subheader("Top 影响因素 (SHAP)")
                st.dataframe(pd.DataFrame(result["top_factors"]), use_container_width=True)
                with st.expander("特征向量"):
                    st.json(result.get("features", {}))
    with c2:
        if st.button("多时刻风险曲线"):
            traj = predict_patient_trajectory(stay_id)
            pts = [p for p in traj.get("points", []) if p.get("status") == "ok"]
            if not pts:
                st.error(traj.get("status", "无可用时刻"))
            else:
                df = pd.DataFrame(pts)
                st.line_chart(df.set_index("hour_index")["risk_score"])
                show = df[["hour_index", "risk_score"]].copy()
                show["band"] = [
                    (p.get("recommend") or {}).get("band") for p in pts
                ]
                st.dataframe(show, use_container_width=True)

with tab_tune:
    st.markdown("侧栏参数写入 yaml 后，用 **from-existing** 重训（不碰 Layer0）。")
    feat_cfg = load_yaml("features.yaml")
    lab_cfg = load_yaml("labels.yaml")
    rec = dict(lab_cfg.get("recommend") or {})
    col_a, col_b = st.columns(2)
    with col_a:
        observe = st.number_input("recommend.observe", 0.0, 1.0, float(rec.get("observe", 0.2)), 0.05)
        recheck = st.number_input("recommend.recheck", 0.0, 1.0, float(rec.get("recheck", 0.4)), 0.05)
        monitor = st.number_input("recommend.monitor", 0.0, 1.0, float(rec.get("monitor", 0.7)), 0.05)
    with col_b:
        default_h = int(feat_cfg.get("hour_index", 1))
        hour_index = st.number_input("features.hour_index（单点默认）", 0, 24, default_h, 1)
        st.caption(f"S2 网格 prediction_hours = {feat_cfg.get('prediction_hours')}")

    if st.button("保存细调配置"):
        feat_cfg["hour_index"] = int(hour_index)
        FEATURES_YAML.write_text(
            yaml.safe_dump(feat_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        lab_cfg["recommend"] = {
            **rec,
            "observe": float(observe),
            "recheck": float(recheck),
            "monitor": float(monitor),
        }
        LABELS_YAML.write_text(
            yaml.safe_dump(lab_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        st.success("已写入 configs/features.yaml 与 configs/labels.yaml")

    st.divider()
    if st.button("运行训练 --from-existing", type="primary"):
        with st.spinner("训练中（复用 dump feat/label）…"):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            proc = subprocess.run(
                [sys.executable, "-m", "application.train", "--from-existing"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
        st.code(proc.stdout or "(no stdout)", language="json")
        if proc.returncode != 0:
            st.error(proc.stderr or f"exit {proc.returncode}")
        else:
            st.success("训练完成 · 见下方验收 Tab / MLflow")
            try:
                st.json(json.loads(proc.stdout))
            except json.JSONDecodeError:
                pass

with tab_accept:
    st.subheader("Layer1 dump 行数门禁")
    try:
        counts = layer1_counts()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("feat_rows", f"{counts['feat_rows']:,}")
        g2.metric("label_rows", f"{counts['label_rows']:,}")
        g3.metric("stays", f"{counts['stay_count']:,}")
        g4.metric("gate", counts["status"])
        st.json({"by_hour": counts["by_hour"], "expected_hours": counts["expected_hours"]})
        if counts["gate_ok"]:
            st.success("S2 行数门禁通过（≈472k × 五时刻）")
        else:
            st.error("门禁失败：请 restore `icu_decision_S2-full_*20260802.dump`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法查询 Layer1：{exc}")

    st.subheader("模型指标 artifact")
    metrics = load_metrics_artifact()
    if not metrics:
        st.warning("无 artifacts/models/metrics_mortality_12h.json，请先训练")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC test", f"{metrics.get('pr_auc_test', 0):.3f}")
        m2.metric("Brier test", f"{metrics.get('brier_test', 0):.3f}")
        m3.metric("ROC test（对照）", f"{metrics.get('auc_test', 0):.3f}")
        m4.metric("工作点阈值", f"{metrics.get('operating_threshold', 0):.3f}")
        by_h = metrics.get("metrics_by_hour_test") or {}
        if by_h:
            rows = []
            for h, mm in sorted(by_h.items(), key=lambda x: int(x[0])):
                rows.append(
                    {
                        "h": int(h),
                        "roc_auc": mm.get("roc_auc"),
                        "pr_auc": mm.get("pr_auc"),
                        "brier": mm.get("brier"),
                        "precision": mm.get("precision"),
                        "recall": mm.get("recall"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        op = (metrics.get("metrics_at_val_threshold") or {}).get("test") or {}
        if op.get("calibration"):
            cal = op["calibration"]
            pred = list(cal.get("mean_predicted") or [])
            # evaluation.py uses fraction_positive (sklearn calibration_curve)
            actual = list(
                cal.get("fraction_positive")
                or cal.get("mean_actual")
                or []
            )
            n = min(len(pred), len(actual))
            if n > 0:
                cal_df = pd.DataFrame(
                    {
                        "mean_predicted": pred[:n],
                        "fraction_positive": actual[:n],
                    }
                )
                st.subheader("校准（test · 工作点指标包）")
                st.line_chart(cal_df.set_index("mean_predicted")["fraction_positive"])
            else:
                st.caption("校准曲线数据为空或不齐，已跳过绘图")
        if op.get("confusion_matrix"):
            st.subheader("混淆矩阵 @工作点")
            st.json(op["confusion_matrix"])

with tab_lab:
    if STATUS.exists():
        st.markdown(STATUS.read_text(encoding="utf-8"))
    else:
        st.info("无 STATUS.md")
