"""监测页：选患者即预测 + Plotly。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from application.predict_patient import (
    list_stays,
    predict_patient,
    predict_patient_trajectory,
)
from domain.features.build import prediction_hours
from infra.db import get_engine
from presentation.ui.charts import fig_risk_trajectory, fig_shap_bars
from presentation.ui.theme import disclaimer, risk_badge_html, status_message

VITAL_LABELS = {
    "vital_heart_rate": "心率",
    "vital_resp_rate": "呼吸",
    "vital_temp": "体温",
    "vital_nbps": "血压",
    "lab_lactate": "乳酸",
    "lab_ph": "pH",
    "shock_index": "休克指数",
    "spo2_fio2_ratio": "SpO2/FiO2",
}


def _stay_options(stays: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"住院 {s['stay_id']} · 住院时长 {float(s.get('los_hours') or 0):.1f} 小时": int(
            s["stay_id"]
        )
        for s in stays
    }


@st.cache_data(show_spinner=False, ttl=60)
def _feat_stay_ids(limit: int = 2000) -> tuple[int, ...]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT stay_id FROM feat.sample_matrix "
                "ORDER BY stay_id LIMIT :lim"
            ),
            {"lim": int(limit)},
        ).fetchall()
    return tuple(int(r[0]) for r in rows)


@st.cache_data(show_spinner=False, ttl=60)
def _hours_for_stay(stay_id: int) -> tuple[int, ...]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT hour_index FROM feat.sample_matrix "
                "WHERE stay_id = :sid ORDER BY hour_index"
            ),
            {"sid": int(stay_id)},
        ).fetchall()
    return tuple(int(r[0]) for r in rows)


@st.cache_data(show_spinner=False, ttl=60)
def _cached_predict(stay_id: int, hour_index: int) -> dict[str, Any]:
    return predict_patient(int(stay_id), hour_index=int(hour_index))


@st.cache_data(show_spinner=False, ttl=60)
def _cached_traj(stay_id: int) -> dict[str, Any]:
    return predict_patient_trajectory(int(stay_id))


def _pick_demo_stays(stays: list[dict[str, Any]], hour: int) -> tuple[int | None, int | None]:
    key = f"demo_pick_h{hour}"
    if key in st.session_state:
        return st.session_state[key]
    sample = stays[:40]
    scored: list[tuple[float, int]] = []
    for s in sample:
        sid = int(s["stay_id"])
        out = _cached_predict(sid, hour)
        if out.get("status") == "ok" and out.get("score_kind") == "probability":
            scored.append((float(out["risk_score"]), sid))
    hi = max(scored, key=lambda x: x[0])[1] if scored else None
    lo = min(scored, key=lambda x: x[0])[1] if scored else None
    st.session_state[key] = (hi, lo)
    return hi, lo


def render_monitor() -> None:
    st.title("ICU 早期恶化预警 · 监测台")
    st.caption("S2 多时刻 · LightGBM + SHAP · 选择住院记录后自动刷新")

    stays = list(list_stays(limit=800))
    feat_ids = set(_feat_stay_ids(3000))
    if feat_ids:
        stays = [s for s in stays if int(s["stay_id"]) in feat_ids] or stays
    if not stays:
        st.warning("未找到 ICU 住院记录。请先 restore S2 dump 或运行 ETL。")
        st.stop()

    cfg_hours = prediction_hours()
    with st.sidebar:
        st.markdown("### 患者")
        options = _stay_options(stays)
        labels = list(options.keys())
        default_ix = 0
        if "force_stay_id" in st.session_state:
            fid = st.session_state.pop("force_stay_id")
            for i, lab in enumerate(labels):
                if options[lab] == fid:
                    default_ix = i
                    break
        choice = st.selectbox("ICU 住院（stay）", labels, index=default_ix, key="mon_stay")
        stay_id = options[choice]
        avail = list(_hours_for_stay(stay_id))
        hours = avail if avail else list(cfg_hours)
        # 优先 h=1（S2 主叙事）；若库中只有 h=0 则自动回退
        prefer = 1 if 1 in hours else hours[0]
        hour = st.selectbox(
            "预测时刻（入科后小时）",
            hours,
            index=hours.index(prefer),
            key=f"mon_hour_{stay_id}",
        )
        if set(hours) != set(cfg_hours):
            st.caption(
                f"提示：当前库仅有时刻 {hours}（配置为 {cfg_hours}）。"
                "完整 S2 请 restore `icu_decision_S2-full_*20260802.dump`。"
            )
        st.markdown("### 演示快捷")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("高风险样例", use_container_width=True, type="primary"):
                with st.spinner("正在扫描样例…"):
                    hi, _ = _pick_demo_stays(stays, int(hour))
                if hi is not None:
                    st.session_state["force_stay_id"] = hi
                    st.rerun()
                else:
                    st.warning("未找到可评分样例")
        with c2:
            if st.button("低风险样例", use_container_width=True, type="primary"):
                with st.spinner("正在扫描样例…"):
                    _, lo = _pick_demo_stays(stays, int(hour))
                if lo is not None:
                    st.session_state["force_stay_id"] = lo
                    st.rerun()
                else:
                    st.warning("未找到可评分样例")

    result = _cached_predict(stay_id, int(hour))
    traj = _cached_traj(stay_id)

    if result.get("status") != "ok":
        st.error(status_message(result))
        disclaimer()
        return

    score = float(result["risk_score"])
    kind = result.get("score_kind", "raw")
    rec = result.get("recommend") or {}
    band = str(rec.get("band", "unknown"))
    band_label = str(rec.get("label", band))

    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown(
            risk_badge_html(score, kind, band, band_label),
            unsafe_allow_html=True,
        )
        st.markdown("")
        feats = result.get("features") or {}
        cols = st.columns(4)
        for i, k in enumerate(list(VITAL_LABELS)[:8]):
            with cols[i % 4]:
                v = feats.get(k)
                try:
                    txt = f"{float(v):.2f}" if v is not None else "—"
                except (TypeError, ValueError):
                    txt = str(v)
                st.metric(VITAL_LABELS[k], txt)

    with right:
        pts = [
            p
            for p in traj.get("points", [])
            if p.get("status") == "ok" and p.get("risk_score") is not None
        ]
        if pts:
            st.plotly_chart(fig_risk_trajectory(pts), use_container_width=True)
        else:
            st.info("该住院暂无多时刻轨迹")

    st.subheader("可解释性（SHAP）")
    factors = result.get("top_factors") or []
    if factors:
        c_a, c_b = st.columns([1.2, 1])
        with c_a:
            st.plotly_chart(fig_shap_bars(factors), use_container_width=True)
        with c_b:
            st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)
            with st.expander("完整特征向量"):
                st.json(feats)
    else:
        st.caption("无 SHAP 因子")

    disclaimer()
