"""解释页：SHAP+LLM 临床可解释性报告，含 RAG 引用溯源。"""

from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy import text

from application.predict_patient import predict_patient_with_explanation
from domain.explain.config import get_config
from domain.explain.rag_retriever import is_available as rag_is_available
from domain.features.build import prediction_hours
from infra.config import get_settings
from infra.db import get_engine
from presentation.ui.charts import fig_shap_bars
from presentation.ui.theme import apply_theme, risk_badge_html, disclaimer

# ── 页面内嵌 CSS：医学证据报告风 ─────────────────────────────
_CSS = """
<style>
/* 状态徽章 */
.explain-badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.badge-ok      { background: #dcfce7; color: #166534; }
.badge-fallback { background: #fef3c7; color: #92400e; }
.badge-norag   { background: #fee2e2; color: #991b1b; }

/* 患者信息卡 */
.patient-card {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  color: #f1f5f9;
  margin-bottom: 1rem;
  border: 1px solid #334155;
}
.patient-card .patient-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #94a3b8;
  margin-bottom: 0.1rem;
}
.patient-card .patient-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: #f8fafc;
}
.patient-card .patient-row {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

/* 因子卡片 */
.factor-card {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  padding: 0.8rem 1rem;
  margin-bottom: 0.55rem;
  transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.factor-card:hover {
  box-shadow: 0 4px 12px rgba(15,23,42,0.08);
  border-color: #94a3b8;
}
.factor-card.positive { border-left: 4px solid #0f766e; }
.factor-card.negative { border-left: 4px solid #2563eb; }
.factor-card.abnormal { background: #fffaf0; border-left: 4px solid #b45309; }

.factor-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.25rem;
}
.factor-name {
  font-weight: 700;
  font-size: 0.95rem;
  color: #1e293b;
}
.factor-meta {
  font-size: 0.78rem;
  color: #64748b;
}
.factor-interp {
  font-size: 0.88rem;
  color: #334155;
  line-height: 1.55;
  margin-bottom: 0.3rem;
}
.factor-meta-row {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  flex-wrap: wrap;
  margin-top: 0.2rem;
}
.shap-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.12rem 0.5rem;
  border-radius: 4px;
}
.shap-up   { background: #dcfce7; color: #166534; }
.shap-down { background: #dbeafe; color: #1e40af; }
.ref-tag   {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  background: #f0fdf4;
  color: #166534;
  border: 1px solid #bbf7d0;
}
.ref-tag.invalid {
  background: #fef9c3;
  color: #92400e;
  border-color: #fde68a;
}

/* 引用卡片 */
.ref-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.55rem 0.75rem;
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
}
.ref-card .ref-id {
  font-weight: 700;
  font-size: 0.72rem;
  color: #0f766e;
  letter-spacing: 0.05em;
}
.ref-card .ref-valid {
  font-size: 0.7rem;
  padding: 0.08rem 0.3rem;
  border-radius: 3px;
  margin-left: 0.35rem;
}
.ref-valid.yes { background: #dcfce7; color: #166534; }
.ref-valid.no  { background: #fee2e2; color: #991b1b; }

/* Token 统计 */
.metric-row {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
  margin-top: 0.4rem;
  padding: 0.75rem 1rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}
.metric-item { display: flex; flex-direction: column; }
.metric-item .metric-val { font-size: 1.15rem; font-weight: 700; color: #1e293b; }
.metric-item .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

/* 底部说明 */
.explain-footer {
  margin-top: 1.5rem;
  padding: 0.65rem 1rem;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
  font-size: 0.78rem;
  color: #64748b;
  border-radius: 0 0 8px 8px;
}
</style>
"""


def _hours_for_stay(stay_id: int) -> list[int]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT hour_index FROM feat.sample_matrix "
                "WHERE stay_id = :sid ORDER BY hour_index"
            ),
            {"sid": int(stay_id)},
        ).fetchall()
    return [int(r[0]) for r in rows]


def _kb_chunk_count() -> int:
    """获取知识库片段数；faiss 未安装时返回 0。"""
    cfg = get_config()
    try:
        import faiss  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return 0
    try:
        if cfg.rag.index_path.exists():
            with open(cfg.rag.index_path, "rb") as f:
                index = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8))
            return index.ntotal
    except Exception:
        pass
    return 0


@st.cache_data(show_spinner=False, ttl=60)
def _cached_explain(stay_id: int, hour_index: int) -> dict[str, Any]:
    return predict_patient_with_explanation(stay_id, hour_index)


def render_explain() -> None:
    apply_theme()
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("ICU 风险解释")
    st.caption("SHAP + LLM 临床可解释性报告 · RAG 医学知识库证据溯源 · Agnes-2.5-flash")

    # ── 侧栏：患者选择 + 系统状态 ─────────────────────────────
    with st.sidebar:
        st.markdown("### 患者选择")
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT f.stay_id "
                    "FROM feat.sample_matrix f "
                    "ORDER BY f.stay_id LIMIT 500"
                )
            ).mappings().all()
        stays = [int(r["stay_id"]) for r in rows]
        if not stays:
            st.warning("数据库无 stay 记录")
            st.stop()
        stay_id = st.selectbox("ICU 住院 Stay ID", stays, key="exp_stay")

        hours = sorted(set(prediction_hours()) & set(_hours_for_stay(stay_id)))
        if not hours:
            hours = sorted(_hours_for_stay(stay_id))[:5]
        if not hours:
            hours = list(prediction_hours())
        hour = st.selectbox("预测时刻（入科后小时）", hours, key="exp_hour")

        st.divider()
        st.markdown("### 系统状态")
        cfg = get_config()
        st.caption(f"LLM：`{cfg.llm.model}`")
        st.caption(f"端点：`{cfg.llm.base_url}`")
        rag_ok = rag_is_available()
        kb_cnt = _kb_chunk_count()
        st.caption(f"RAG 知识库：{'✅ 可用' if rag_ok else '❌ 不可用'}（{kb_cnt} 个片段）")

    # ── 主内容区 ───────────────────────────────────────────────
    with st.spinner("正在调用 LLM 生成临床解释…"):
        result = _cached_explain(stay_id, hour)

    if result.get("status") != "ok":
        st.error(f"预测失败：{result.get('status')} — {result.get('message', '')}")
        disclaimer()
        return

    score = float(result["risk_score"])
    kind = result.get("score_kind", "raw")
    rec = result.get("recommend") or {}
    band = str(rec.get("band", "unknown"))
    band_label = str(rec.get("label", band))

    explanation = result.get("explanation") or {}
    structured = explanation.get("structured") or {}
    factors = structured.get("factor_analysis", [])
    references = structured.get("references", []) or []
    disclaimers = structured.get("disclaimers", []) or []
    fallback = structured.get("fallback_mode", True)
    rag_on = structured.get("rag_available", False)
    llm_usage = structured.get("llm_usage") or {}
    summary_text = structured.get("summary", "")

    # ── 顶部信息条 ─────────────────────────────────────────────
    col_info, col_risk = st.columns([2.2, 1])
    with col_info:
        mode_badge = ("badge-ok" if not fallback else ("badge-norag" if not rag_on else "badge-fallback"))
        mode_txt = "AI 解释" if (not fallback and rag_on) else ("无 RAG 降级" if not rag_on else "降级模式")
        st.markdown(
            f"""
            <div class="patient-card">
              <div class="patient-row">
                <div>
                  <div class="patient-label">ICU 住院 Stay ID</div>
                  <div class="patient-value">{stay_id}</div>
                </div>
                <div>
                  <div class="patient-label">预测时刻</div>
                  <div class="patient-value">入科后 {hour} 小时</div>
                </div>
                <div>
                  <div class="patient-label">报告模式</div>
                  <div class="patient-value">
                    <span class="explain-badge {mode_badge}">{mode_txt}</span>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_risk:
        st.markdown(risk_badge_html(score, kind, band, band_label), unsafe_allow_html=True)

    # ── 两列布局：左主内容 / 右证据卡片 ───────────────────────
    col_main, col_side = st.columns([1.55, 1])

    # ── 左列：风险概述 + 完整解释 ───────────────────────────
    with col_main:
        if summary_text or fallback:
            st.markdown("### 风险概述")
            if summary_text:
                card_color = "#dcfce7" if not fallback else "#fef3c7"
                card_border = "#0f766e" if not fallback else "#b45309"
                st.markdown(
                    f'<div style="padding:0.7rem 1rem;background:{card_color};'
                    f'border-radius:6px;border-left:3px solid {card_border};'
                    f'font-size:0.92rem;line-height:1.65;">{summary_text}</div>',
                    unsafe_allow_html=True,
                )
            elif fallback:
                st.info("当前处于降级模式，请检查 RAG 知识库配置。")

        st.markdown("### 完整解释报告")
        md_text = explanation.get("explanation", "")
        if md_text:
            st.markdown(md_text)
        else:
            st.caption("（暂无 LLM 生成文本，查看右侧因子卡片了解具体因素）")

    # ── 右列：SHAP 图 + 因子证据卡 + 引用溯源 ───────────────
    with col_side:
        top_factors = result.get("top_factors", [])
        if top_factors:
            st.markdown("### SHAP 贡献分布")
            st.plotly_chart(
                fig_shap_bars(top_factors),
                use_container_width=True,
                key=f"shap_exp_{stay_id}_{hour}",
            )

        st.markdown("### 关键因子证据")
        if not factors:
            st.caption("无因子数据")
        else:
            for fa in factors:
                feature = fa.get("feature", "")
                interp = fa.get("clinical_interpretation", "")
                value = fa.get("value")
                unit = fa.get("unit", "")
                shap_val = fa.get("shap", 0)
                shap_dir = fa.get("shap_direction", "positive")
                ref_id = fa.get("reference_id", "")
                ref_valid = fa.get("reference_valid", False)
                evidence_level = fa.get("evidence_level", "C")

                is_pos = shap_dir == "positive"
                card_cls = "positive" if is_pos else "negative"
                if fa.get("is_abnormal"):
                    card_cls += " abnormal"

                shap_arrow = "▲" if is_pos else "▼"
                shap_cls = "shap-up" if is_pos else "shap-down"
                value_str = f"{value} {unit}".strip() if value is not None else "缺失"

                ref_html = ""
                if ref_id:
                    valid_cls = "ref-tag" if ref_valid else "ref-tag invalid"
                    ref_html = f' <span class="{valid_cls}">{ref_id}</span>'

                card_html = (
                    f'<div class="factor-card {card_cls}">'
                    f'<div class="factor-header">'
                    f'  <span class="factor-name">{feature}</span>'
                    f'  <span class="factor-meta">{value_str}</span>'
                    f'</div>'
                    f'<div class="factor-interp">{interp}</div>'
                    f'<div class="factor-meta-row">'
                    f'  <span class="shap-badge {shap_cls}">{shap_arrow} SHAP {abs(float(shap_val)):.4f}</span>'
                    f'  <span style="font-size:0.7rem;color:#94a3b8;">证据级 {evidence_level}</span>'
                    f"  {ref_html}"
                    f"</div></div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        if references:
            st.markdown(f"### 引用溯源（{len(references)} 条）")
            for ref in references:
                ref_id_val = ref.get("id", "")
                source = ref.get("source", "")
                title = ref.get("title", "")
                valid = ref.get("valid", False)
                score_r = ref.get("score", 0)
                snippet = ref.get("snippet", "")
                valid_cls = "ref-valid yes" if valid else "ref-valid no"
                valid_txt = "✅ 有效" if valid else "✗ 无效"

                snippet_html = ""
                if snippet:
                    snippet_escaped = snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    snippet_html = (
                        f'<details style="margin-top:0.2rem;">'
                        f'<summary style="font-size:0.72rem;color:#0f766e;cursor:pointer;">查看引用片段</summary>'
                        f'<div style="font-size:0.78rem;color:#475569;padding:0.2rem 0.5rem;'
                        f'background:#f8fafc;border-radius:4px;margin-top:0.15rem;'
                        f'white-space:pre-wrap;max-height:120px;overflow-y:auto;">{snippet_escaped[:350]}</div>'
                        f"</details>"
                    )

                card_html = (
                    f'<div class="ref-card">'
                    f'<div>'
                    f'  <span class="ref-id">{ref_id_val}</span>'
                    f'  <span class="{valid_cls}">{valid_txt}</span>'
                    f"</div>"
                    f'<div style="font-weight:600;margin:0.15rem 0;font-size:0.85rem;">{title}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b;">📄 {source} · 相似度 {float(score_r):.3f}</div>'
                    f"{snippet_html}"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.caption("暂无引用（可能处于降级模式）")

    # ── 底部：Token 用量 + 免责声明 ──────────────────────────
    if llm_usage:
        pts = llm_usage
        st.markdown(
            f"""
            <div class="metric-row">
              <div class="metric-item">
                <span class="metric-val">{pts.get('prompt_tokens', 0):,}</span>
                <span class="metric-label">Prompt Tokens</span>
              </div>
              <div class="metric-item">
                <span class="metric-val">{pts.get('completion_tokens', 0):,}</span>
                <span class="metric-label">Completion Tokens</span>
              </div>
              <div class="metric-item">
                <span class="metric-val">{pts.get('total_tokens', 0):,}</span>
                <span class="metric-label">Total Tokens</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if disclaimers:
        st.markdown(
            f"""
            <div class="explain-footer">
              <strong>⚠️ 医疗免责声明</strong>
              <ul style="margin:0.3rem 0 0 1.2rem;padding:0;">
                {''.join(f'<li>{d}</li>' for d in disclaimers)}
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    disclaimer()
