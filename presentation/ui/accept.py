"""验收门禁 + 校准 + 决策净受益（含 Bootstrap CI）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact, layer1_counts
from application.demo_curves import compute_demo_net_benefit
from domain.models.dca import ART as DCA_ART
from presentation.ui.charts import fig_calibration, fig_net_benefit, fig_net_benefit_with_ci
from presentation.ui.theme import disclaimer

STATUS = Path(__file__).resolve().parents[2] / "docs" / "STATUS.md"


def render_accept() -> None:
    st.title("验收门禁")
    st.caption("Layer1 行数 · 主指标 · 校准 · 决策净受益（演示抽样）")

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
        st.subheader("工作点价值叙事")
        if op:
            cm = op.get("confusion_matrix") or {}
            st.markdown(
                f"- 工作点精确率 **{op.get('precision', 0):.1%}** · "
                f"召回率 **{op.get('recall', 0):.1%}** · "
                f"特异度 **{op.get('specificity', 0):.1%}**\n"
                f"- 混淆：TP={cm.get('tp')} FP={cm.get('fp')} "
                f"FN={cm.get('fn')} TN={cm.get('tn')}\n"
                "- 解读：在稀有阳性结局下，优先看 **PR-AUC / 召回-精确权衡**，勿单报 ROC。"
            )
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

    st.subheader("决策曲线（净受益 + Bootstrap 95% CI）")
    with st.spinner("加载 DCA 报告（含 CI 带）…"):
        dca_files = sorted(DCA_ART.glob("dca_*.json")) if DCA_ART.exists() else []
    if dca_files:
        import json
        latest = dca_files[-1]
        dca_data = json.loads(latest.read_text(encoding="utf-8"))
        st.caption(
            f"来源：{latest.name} · "
            f"样本 n={dca_data['curve']['n']:,} · "
            f"阳性率={dca_data['curve']['prevalence']:.2%} · "
            f"代价比 FP:FN={dca_data['curve']['cost_ratio']}:1 · "
            f"Bootstrap={dca_data['curve']['n_bootstrap']} 次"
        )
        wp = dca_data["curve"].get("working_point")
        if wp:
            st.markdown(
                f"**工作点**：阈值 **{wp['threshold']:.4f}** "
                f"（{wp['method']}）· "
                f"精确率 {wp['precision']:.2%} · 召回率 {wp['recall']:.2%} · "
                f"F1={wp['f1']:.4f} · 净受益 **{wp['net_benefit']:.4f}**"
            )
        has_ci = dca_data["curve"].get("nb_model_ci_lower") is not None
        chart_fn = fig_net_benefit_with_ci if has_ci else fig_net_benefit
        st.plotly_chart(chart_fn(dca_data["curve"]), use_container_width=True)
        if has_ci:
            st.caption(
                "阴影区域为 Bootstrap 95% CI；"
                "曲线在阈值范围内高于「全部干预」和「全不干预」基线，表示该阈值下存在临床净受益。"
            )
    else:
        # fallback: 使用旧的 demo 曲线
        curve = compute_demo_net_benefit()
        if curve.get("status") == "ok":
            st.caption(f"抽样 n={curve.get('sampled_n')} · 阳性率={curve.get('prevalence', 0):.2%}")
            st.plotly_chart(fig_net_benefit(curve), use_container_width=True)
        else:
            st.info(f"净受益曲线暂不可用：{curve.get('message', curve.get('status'))}")

    with st.expander("项目状态 STATUS.md"):
        if STATUS.exists():
            st.markdown(STATUS.read_text(encoding="utf-8"))
        else:
            st.info("无 STATUS.md")
    disclaimer()
