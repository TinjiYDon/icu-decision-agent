"""验收门禁 + 校准 + 决策净受益。"""

from __future__ import annotations

import lightgbm as lgb
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact, layer1_counts
from application.demo_curves import compute_demo_net_benefit
from domain.features.build import FEATURE_COLS
from domain.models.dca import dca_curve_with_ci, dca_working_point
from domain.models.lgbm import _get_model_bundle, _load_training_frame
from domain.models.split import split_frame_by_stay
from presentation.ui.charts import fig_calibration, fig_net_benefit, fig_net_benefit_with_ci
from presentation.ui.theme import disclaimer

STATUS = Path(__file__).resolve().parents[2] / "docs" / "STATUS.md"
_ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts" / "models"


def _compute_dca_with_ci(*, max_rows: int = 5000) -> dict:
    """Compute DCA with bootstrap CI on the LGBM test set."""
    try:
        booster, _ = _get_model_bundle()
    except FileNotFoundError:
        return {"status": "no_model", "message": "缺少 lgbm_mortality_12h.txt，请先训练"}
    try:
        df = _load_training_frame()
        if len(df) < 50:
            return {"status": "too_few", "message": f"样本过少 n={len(df)}"}
        _, _, test_df, _ = split_frame_by_stay(df)
        if len(test_df) > max_rows:
            test_df = test_df.sample(n=max_rows, random_state=42)
        X = test_df[FEATURE_COLS].to_numpy(dtype=float)
        y = test_df["label"].to_numpy(dtype=int)
        raw = booster.predict(X)
        if np.any((raw < -50) | (raw > 50)):
            prob = 1.0 / (1.0 + np.exp(-raw))
        else:
            prob = raw
        cbr = 1.0 / 9.0
        ci_result = dca_curve_with_ci(y, prob, n_boot=500, seed=42)
        wp = dca_working_point(y, prob, cost_benefit_ratio=cbr)
        return {
            "status": "ok",
            "n_boot": 500,
            "cost_benefit_ratio": cbr,
            "working_threshold": wp.threshold,
            "working_nb": wp.net_benefit_model,
            "treat_all_nb": wp.net_benefit_treat_all,
            "n_samples": int(len(y)),
            "positive_rate": float(y.mean()),
            "ci_lower": ci_result.ci_lower_model,
            "ci_upper": ci_result.ci_upper_model,
            **({
                "thresholds": ci_result.thresholds,
                "net_benefit_model": ci_result.net_benefit_model,
                "net_benefit_treat_all": ci_result.net_benefit_treat_all,
            } if ci_result.thresholds else {}),
            "conclusion": (
                "✅ 模型净受益 > 全干预（CBR=1/9）"
                if wp.net_benefit_model > wp.net_benefit_treat_all
                else "⚠️ 模型净受益 ≤ 全干预，阈值需调整"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


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

    st.subheader("决策曲线（净受益）")
    with st.spinner("抽样测试集计算净受益（≤5k）…"):
        curve = compute_demo_net_benefit()
    if curve.get("status") == "ok":
        st.caption(
            f"抽样 n={curve.get('sampled_n')} · 阳性率={curve.get('prevalence', 0):.2%} · "
            "曲线高于「全不干预」且尽量高于「全部干预」的区间，表示阈值有临床净受益。"
        )
        st.plotly_chart(fig_net_benefit(curve), use_container_width=True)

    st.subheader("决策曲线 · Bootstrap 95% 置信区间（D2）")
    try:
        ci_curve = _compute_dca_with_ci(max_rows=5000)
        if ci_curve.get("status") == "ok":
            st.caption(
                f"Bootstrap n_boot={ci_curve.get('n_boot', 500)} · "
                f"CBR={ci_curve.get('cost_benefit_ratio', 1/9):.4f} → 工作阈值 p_t={ci_curve.get('working_threshold', 0.1):.2%} · "
                f"工作点 NB_model={ci_curve.get('working_nb', 0):.4f} vs NB_treat_all={ci_curve.get('treat_all_nb', 0):.4f}"
            )
            st.plotly_chart(fig_net_benefit_with_ci(ci_curve), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("模型净受益", f"{ci_curve.get('working_nb', 0):.4f}")
            col2.metric("全干预净受益", f"{ci_curve.get('treat_all_nb', 0):.4f}")
            col3.metric("净受益增益", f"{(ci_curve.get('working_nb', 0) or 0) - (ci_curve.get('treat_all_nb', 0) or 0):.4f}")
            if ci_curve.get("conclusion"):
                st.success(ci_curve["conclusion"])
        else:
            st.info(f"Bootstrap DCA 暂不可用：{ci_curve.get('message', ci_curve.get('status'))}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Bootstrap DCA 计算出错：{exc}")

    with st.expander("项目状态 STATUS.md"):
        if STATUS.exists():
            st.markdown(STATUS.read_text(encoding="utf-8"))
        else:
            st.info("无 STATUS.md")
    disclaimer()
