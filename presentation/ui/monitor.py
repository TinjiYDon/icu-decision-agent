"""Monitor page: patient workspace with auto-predict + Plotly."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from application.predict_patient import (
    list_stays,
    predict_patient,
    predict_patient_trajectory,
)
from domain.features.build import prediction_hours
from presentation.ui.charts import fig_risk_trajectory, fig_shap_bars
from presentation.ui.theme import disclaimer, risk_badge_html

VITAL_KEYS = [
    "vital_heart_rate",
    "vital_resp_rate",
    "vital_temp",
    "vital_nbps",
    "lab_lactate",
    "lab_ph",
    "shock_index",
    "spo2_fio2_ratio",
]


def _stay_options(stays: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"stay {s['stay_id']} · LOS {float(s.get('los_hours') or 0):.1f}h": int(s["stay_id"])
        for s in stays
    }


@st.cache_data(show_spinner=False, ttl=300)
def _cached_predict(stay_id: int, hour_index: int) -> dict[str, Any]:
    return predict_patient(int(stay_id), hour_index=int(hour_index))


@st.cache_data(show_spinner=False, ttl=300)
def _cached_traj(stay_id: int) -> dict[str, Any]:
    return predict_patient_trajectory(int(stay_id))


def _pick_demo_stays(stays: list[dict[str, Any]], hour: int) -> tuple[int | None, int | None]:
    """Scan a shortlist for high/low risk demos (cached via session)."""
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
    st.title("ICU Early-Warning Monitor")
    st.caption("S2 multi-hour · LightGBM + SHAP · select a stay to refresh")

    stays = list(list_stays(limit=500))
    if not stays:
        st.warning("No ICU stays. Restore S2 dump or run ETL.")
        st.stop()

    hours = prediction_hours()
    with st.sidebar:
        st.markdown("### Patient")
        options = _stay_options(stays)
        labels = list(options.keys())
        default_ix = 0
        if "force_stay_id" in st.session_state:
            fid = st.session_state.pop("force_stay_id")
            for i, lab in enumerate(labels):
                if options[lab] == fid:
                    default_ix = i
                    break
        choice = st.selectbox("ICU stay", labels, index=default_ix, key="mon_stay")
        stay_id = options[choice]
        hour = st.selectbox(
            "Prediction hour",
            hours,
            index=min(1, len(hours) - 1),
            key="mon_hour",
        )
        st.markdown("### Demo shortcuts")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("High-risk sample", use_container_width=True):
                with st.spinner("Scanning shortlist…"):
                    hi, _ = _pick_demo_stays(stays, int(hour))
                if hi is not None:
                    st.session_state["force_stay_id"] = hi
                    st.rerun()
                else:
                    st.warning("No scored sample")
        with c2:
            if st.button("Low-risk sample", use_container_width=True):
                with st.spinner("Scanning shortlist…"):
                    _, lo = _pick_demo_stays(stays, int(hour))
                if lo is not None:
                    st.session_state["force_stay_id"] = lo
                    st.rerun()
                else:
                    st.warning("No scored sample")

    result = _cached_predict(stay_id, int(hour))
    traj = _cached_traj(stay_id)

    if result.get("status") != "ok":
        st.error(result.get("message", result.get("status")))
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
        for i, k in enumerate(VITAL_KEYS[:8]):
            with cols[i % 4]:
                v = feats.get(k)
                try:
                    txt = f"{float(v):.2f}" if v is not None else "—"
                except (TypeError, ValueError):
                    txt = str(v)
                st.metric(k.replace("vital_", "").replace("lab_", ""), txt)

    with right:
        pts = [p for p in traj.get("points", []) if p.get("status") == "ok" and p.get("risk_score") is not None]
        if pts:
            st.plotly_chart(fig_risk_trajectory(pts), use_container_width=True)
        else:
            st.info("Trajectory unavailable for this stay")

    st.subheader("Explainability")
    factors = result.get("top_factors") or []
    if factors:
        c_a, c_b = st.columns([1.2, 1])
        with c_a:
            st.plotly_chart(fig_shap_bars(factors), use_container_width=True)
        with c_b:
            st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)
            with st.expander("Full feature vector"):
                st.json(feats)
    else:
        st.caption("No SHAP factors")

    disclaimer()
