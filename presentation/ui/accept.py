"""Acceptance / metrics secondary page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact, layer1_counts
from presentation.ui.charts import fig_calibration
from presentation.ui.theme import disclaimer

STATUS = Path(__file__).resolve().parents[2] / "docs" / "STATUS.md"


def render_accept() -> None:
    st.title("验收门禁")
    st.caption("Layer1 dump 行数 + 模型指标 artifact")

    try:
        counts = layer1_counts()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("特征行数", f"{counts['feat_rows']:,}")
        g2.metric("标签行数", f"{counts['label_rows']:,}")
        g3.metric("住院数", f"{counts['stay_count']:,}")
        g4.metric("门禁", counts["status"])
        st.json({"by_hour": counts["by_hour"], "expected_hours": counts["expected_hours"]})
        if counts["gate_ok"]:
            st.success("S2 行数门禁通过（约 472k × 5 时刻）")
        else:
            st.error("门禁失败 — 请 restore icu_decision_S2-full_*20260802.dump")
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法查询 Layer1：{exc}")

    metrics = load_metrics_artifact()
    if not metrics:
        st.warning("缺少 artifacts/models/metrics_mortality_12h.json，请先训练")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC（测试）", f"{metrics.get('pr_auc_test', 0):.3f}")
        m2.metric("Brier（测试）", f"{metrics.get('brier_test', 0):.3f}")
        m3.metric("ROC（对照）", f"{metrics.get('auc_test', 0):.3f}")
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        op = (metrics.get("metrics_at_val_threshold") or {}).get("test") or {}
        if op.get("calibration"):
            cal = op["calibration"]
            pred = list(cal.get("mean_predicted") or [])
            actual = list(cal.get("fraction_positive") or cal.get("mean_actual") or [])
            n = min(len(pred), len(actual))
            if n > 0:
                st.plotly_chart(
                    fig_calibration(pred[:n], actual[:n]),
                    use_container_width=True,
                )
        if op.get("confusion_matrix"):
            st.subheader("混淆矩阵（工作点）")
            st.json(op["confusion_matrix"])

    with st.expander("项目状态 STATUS.md"):
        if STATUS.exists():
            st.markdown(STATUS.read_text(encoding="utf-8"))
        else:
            st.info("无 STATUS.md")
    disclaimer()
