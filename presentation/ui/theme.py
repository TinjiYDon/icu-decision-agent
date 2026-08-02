"""临床监测台主题（青绿 / 石板灰；侧栏高对比）。"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
  --icu-ink: #1e293b;
  --icu-muted: #64748b;
}
.stApp { background: linear-gradient(165deg, #eef5f3 0%, #f8fafc 45%, #f1f5f9 100%); }

/* 侧栏：深底 + 明确文字色，勿用 * 覆盖按钮 */
[data-testid="stSidebar"] {
  background: #0f172a !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: #1e293b !important;
  color: #f8fafc !important;
  border-color: #475569 !important;
}

/* 侧栏按钮：青绿底 + 深色字（解决看不清） */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[kind="primary"] {
  background-color: #2dd4bf !important;
  color: #0f172a !important;
  border: 1px solid #0f766e !important;
  font-weight: 700 !important;
}
[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span,
[data-testid="stSidebar"] button div {
  color: #0f172a !important;
  font-weight: 700 !important;
}
[data-testid="stSidebar"] button:hover {
  background-color: #5eead4 !important;
  color: #0f172a !important;
}

/* 导航项保持可读 */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span {
  color: #e2e8f0 !important;
}
[data-testid="stSidebarNav"] [aria-selected="true"] {
  background-color: #334155 !important;
}

.risk-hero {
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}
.risk-badge {
  display: inline-block;
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--icu-ink);
  line-height: 1.1;
}
.risk-band {
  display: inline-block;
  margin-top: 0.5rem;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.95rem;
}
.band-observe { background: #dcfce7; color: #166534; }
.band-recheck { background: #fef9c3; color: #854d0e; }
.band-monitor { background: #ffedd5; color: #9a3412; }
.band-escalate { background: #fee2e2; color: #991b1b; }
.band-unknown { background: #e2e8f0; color: #475569; }
.disclaimer {
  margin-top: 2rem;
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  color: var(--icu-muted);
  border-top: 1px solid #e2e8f0;
}
.section-label {
  font-size: 0.8rem;
  letter-spacing: 0.04em;
  color: var(--icu-muted);
  margin-bottom: 0.25rem;
}
</style>
"""

BAND_CLASS = {
    "observe": "band-observe",
    "recheck": "band-recheck",
    "monitor": "band-monitor",
    "escalate": "band-escalate",
}

STATUS_ZH = {
    "no_model": "尚未训练模型。请先运行：python -m application.train --from-existing",
    "no_features": "该 stay 在所选预测时刻无特征。请改选时刻，或 restore S2 dump（需含 h=0,1,2,4,6）。",
    "ok": "ok",
}


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def risk_badge_html(score: float, score_kind: str, band: str, band_label: str) -> str:
    if score_kind == "probability":
        score_txt = f"{score:.1%}"
    else:
        score_txt = f"{score:.4f}"
    cls = BAND_CLASS.get(band, "band-unknown")
    return f"""
    <div class="risk-hero">
      <div class="section-label">12 小时死亡风险</div>
      <div class="risk-badge">{score_txt}</div>
      <div class="risk-band {cls}">{band_label or band}</div>
    </div>
    """


def status_message(result: dict) -> str:
    st_code = str(result.get("status", ""))
    if st_code in STATUS_ZH and st_code != "ok":
        return STATUS_ZH[st_code]
    return str(result.get("message") or st_code)


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer">研究演示 · 非临床决策设备。标签仍可能为日期级死亡精度；'
        "主报 PR-AUC / Brier / 工作点，ROC 仅对照。</div>",
        unsafe_allow_html=True,
    )
