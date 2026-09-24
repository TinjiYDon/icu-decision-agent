"""D1 解释链安全对照 · Streamlit 页面。

展示 4 种模式的指标汇总 + 单病例逐模式对比。
兼容旧版报告（7项指标）和最新版（10项指标）。
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from domain.explain.audit import MODE_FULL, MODE_LLM_NO_RAG, MODE_RULE, MODES

MODE_LABELS = {
    MODE_RULE: "A. 规则模板",
    MODE_LLM_NO_RAG: "B. 单 LLM 无 RAG",
    MODE_FULL: "D. SHAP+RAG+LLM",
}


def _safe_get(agg: dict, key: str, mode: str) -> float | None:
    """安全获取指标值，兼容旧版报告（缺少新字段时返回 None）。"""
    return (agg.get(key) or {}).get(mode)


def _load_latest_report() -> dict | None:
    """从 artifacts/explain_audit/ 加载最新的 JSON 报告。"""
    audit_dir = Path(__file__).resolve().parents[2] / "artifacts" / "explain_audit"
    if not audit_dir.exists():
        return None
    files = sorted(audit_dir.glob("report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def render_explain_audit() -> None:
    st.title("解释链安全对照（D1）")
    st.caption("四模式批量评测：规则模板 / 单LLM无RAG / SHAP+RAG+LLM · 十项安全指标")

    summary = _load_latest_report()
    if summary is None:
        st.warning("未找到评测报告。请先运行：`python scripts/audit_safety_metrics.py --mock --limit 20`")
        with st.expander("评测命令参考"):
            st.code("""
# 使用 mock 数据快速验证（无需数据库/LLM）
python scripts/audit_safety_metrics.py --mock --limit 20

# 使用真实数据评测
python scripts/audit_safety_metrics.py --hours 1 --limit 20
            """, language="powershell")
        return

    agg = summary["aggregate"]
    per_stay = summary.get("per_stay", [])
    n = summary["n_stays"]
    h = summary["hour_index"]

    st.caption(f"评测 stay 数：{n} · 预测时刻 h={h} · 耗时：{summary['elapsed_s']}s")

    # D1.2/D1.3 兼容性检查：旧报告可能缺少新指标
    has_new_metrics = "risk_consistency" in agg
    if not has_new_metrics:
        st.warning("⚠️ 当前报告为旧版（7项指标），D1.2/D1.3 新增的 3 项指标（风险一致性/术语密度/可读性）未显示。请重新运行 `python scripts/audit_safety_metrics.py --mock --limit 20` 生成新版报告。")

    # ── 指标汇总表 ───────────────────────────────────────────
    st.subheader("指标汇总（10项）")
    rows = []
    for m in MODES:
        label = MODE_LABELS.get(m, m)
        rows.append({
            "模式": label,
            "SHAP对齐": _safe_get(agg, "shap_align", m),
            "引用有效": _safe_get(agg, "ref_validity", m),
            "过度承诺": _safe_get(agg, "overcommit", m),
            "特征覆盖": _safe_get(agg, "feature_coverage", m),
            "事实grounding": _safe_get(agg, "factual_grounding", m),
            "证据具体度": _safe_get(agg, "evidence_specificity", m),
            "风险一致性": _safe_get(agg, "risk_consistency", m),
            "术语密度": _safe_get(agg, "clinical_term_density", m),
            "可读性": _safe_get(agg, "readability_score", m),
        })
    df = __import__("pandas").DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── 柱状图对比 ───────────────────────────────────────────
    labels = [MODE_LABELS.get(m, m) for m in MODES]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 特征覆盖密度")
        vals_cov = [_safe_get(agg, "feature_coverage", m) or 0 for m in MODES]
        fig1 = go.Figure(data=[go.Bar(x=labels, y=vals_cov, marker_color=["#94a3b8", "#f59e0b", "#0f766e"])])
        fig1.update_layout(yaxis_range=[0, 1], yaxis_title="覆盖率", height=220)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("### 事实 Grounding 率")
        vals_grd = [_safe_get(agg, "factual_grounding", m) or 0 for m in MODES]
        fig2 = go.Figure(data=[go.Bar(x=labels, y=vals_grd, marker_color=["#94a3b8", "#f59e0b", "#0f766e"])])
        fig2.update_layout(yaxis_range=[0, 1], yaxis_title="Grounding率", height=220)
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.markdown("### 证据具体度")
        vals_evi = [_safe_get(agg, "evidence_specificity", m) or 0 for m in MODES]
        fig3 = go.Figure(data=[go.Bar(x=labels, y=vals_evi, marker_color=["#94a3b8", "#f59e0b", "#0f766e"])])
        fig3.update_layout(yaxis_range=[0, 1], yaxis_title="具体度", height=220)
        st.plotly_chart(fig3, use_container_width=True)

    # ── 原有指标图 ───────────────────────────────────────────
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("### SHAP Top-k 对齐率")
        vals = [_safe_get(agg, "shap_align", m) or 0 for m in MODES]
        fig4 = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=["#94a3b8", "#f59e0b", "#0f766e"])])
        fig4.update_layout(yaxis_range=[0, 1], yaxis_title="Jaccard相似度", height=220)
        st.plotly_chart(fig4, use_container_width=True)
    with col5:
        st.markdown("### 过度承诺率")
        vals_oc = [_safe_get(agg, "overcommit", m) or 0 for m in MODES]
        fig5 = go.Figure(data=[go.Bar(x=labels, y=vals_oc, marker_color=["#94a3b8", "#f59e0b", "#ef4444"])])
        fig5.update_layout(yaxis_range=[0, 1], yaxis_title="触发占比", height=220)
        st.plotly_chart(fig5, use_container_width=True)

    # ── 引用有效率 ────────────────────────────────────────────
    ref_val = _safe_get(agg, "ref_validity", MODE_FULL)
    if ref_val is not None:
        st.markdown(f"### 引用有效率（D 模式）：**{ref_val:.2%}**")
        st.caption("有效引用 = 数值在引用片段中能被校验通过的比例")
    else:
        st.caption("引用有效率：暂无有效数据（RAG 知识库可能未构建）")

    # ── 单病例对比 ───────────────────────────────────────────
    if per_stay:
        st.subheader("单病例对比")
        stay_options = [f"stay={s['stay_id']} h={s['hour_index']}" for s in per_stay]
        sel = st.selectbox("选择病例", stay_options, key=f"audit_stay_sel_{n}")
        if sel and " stay=" in sel:
            sid = int(sel.split(" stay=")[1].split(" ")[0])
            sr = next((s for s in per_stay if s["stay_id"] == sid), None)
            if sr is None:
                return

            cols = st.columns(len(MODES))
            for i, m in enumerate(MODES):
                with cols[i]:
                    label = MODE_LABELS.get(m, m)
                    ex = sr.get("mode_results", {}).get(m, {})
                    st.markdown(f"**{label}**")
                    if ex.get("status") == "ok":
                        structured = ex.get("structured", {})
                        factors = structured.get("factor_analysis", [])
                        st.caption(f"风险分数：{sr['risk_score']:.1%}")
                        st.caption(f"因子数：{len(factors)}")
                        st.caption(f"SHAP 对齐：{_safe_get(sr, 'shap_align', m) or '—'}")
                        st.caption(f"引用有效：{_safe_get(sr, 'ref_validity', m) or '—'}")
                        st.caption(f"过度承诺：{_safe_get(sr, 'overcommit', m) or '—'}")
                        st.caption(f"特征覆盖：{_safe_get(sr, 'feature_coverage', m) or '—'}")
                        st.caption(f"事实grounding：{_safe_get(sr, 'factual_grounding', m) or '—'}")
                        st.caption(f"证据具体度：{_safe_get(sr, 'evidence_specificity', m) or '—'}")
                        st.caption(f"风险一致性：{_safe_get(sr, 'risk_consistency', m) or '—'}")
                        st.caption(f"术语密度：{_safe_get(sr, 'clinical_term_density', m) or '—'}")
                        st.caption(f"可读性：{_safe_get(sr, 'readability_score', m) or '—'}")
                        # 展开第一个因子的解释
                        if factors:
                            with st.expander("查看首个因子解释"):
                                fa = factors[0]
                                st.text(fa.get("clinical_interpretation", ""))
                    elif ex.get("status") == "error":
                        st.error(f"错误：{ex.get('error', 'unknown')}")
                    else:
                        st.info("未评测")

    # ── 完整 JSON 报告 ───────────────────────────────────────
    with st.expander("原始评测数据（JSON）"):
        st.json(summary)
