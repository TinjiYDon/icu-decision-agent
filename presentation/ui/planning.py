"""仓内处置规划建议页（D3）— 非床位调度。"""

from __future__ import annotations

import streamlit as st

from application.care_plan import care_plan_for_stay
from application.predict_patient import list_stays
from presentation.ui.theme import apply_theme, disclaimer


def render_planning() -> None:
    apply_theme()
    st.title("处置规划建议")
    st.caption("仓内临床动作规划（监护/复查/升级）。不调用床位调度引擎。")

    stays = list(list_stays(limit=200))
    if not stays:
        st.warning("无可用 stay")
        return
    labels = [f"{s.get('stay_id')} · {s.get('first_careunit', '')}" for s in stays]
    idx = st.selectbox("选择 stay", range(len(labels)), format_func=lambda i: labels[i])
    stay_id = int(stays[idx]["stay_id"])
    model_type = st.radio("模型", ["lgbm", "grud"], horizontal=True)

    if st.button("生成规划", type="primary"):
        plan = care_plan_for_stay(stay_id, model_type=model_type)
        st.session_state["care_plan"] = plan

    plan = st.session_state.get("care_plan")
    if not plan:
        return
    if plan.get("status") == "care_plan_unavailable":
        st.error(plan)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("风险分", f"{float(plan.get('risk_score', 0)):.3f}")
    c2.metric("档位", plan.get("band", "—"))
    c3.metric("stay", plan.get("stay_id", "—"))

    st.subheader("建议动作")
    for a in plan.get("actions") or []:
        st.markdown(
            f"- **P{a.get('priority')}** {a.get('label_zh')} "
            f"（`{a.get('action')}` · 复查约 {a.get('recheck_hours')}h）"
        )

    if plan.get("drivers"):
        st.subheader("驱动因子")
        st.write(", ".join(plan["drivers"]))

    st.subheader("升级触发")
    st.write(plan.get("escalate_triggers"))
    st.info(plan.get("disclaimer", ""))
    disclaimer()
