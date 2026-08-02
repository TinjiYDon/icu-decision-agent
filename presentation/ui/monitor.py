"""监测页：选患者即预测 + Plotly。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

from application.acceptance import load_metrics_artifact
from application.predict_patient import predict_patient, predict_patient_trajectory
from domain.features.build import prediction_hours
from infra.config import load_yaml
from infra.db import get_engine
from presentation.ui.charts import fig_risk_trajectory, fig_shap_bars
from presentation.ui.theme import disclaimer, risk_badge_html, status_message

# S2 dump 以化验+年龄为主；chart 生命体征在导出中常为 null，故面板以可得信号优先。
DISPLAY_FIELDS = {
    "anchor_age": "年龄",
    "lab_bun": "BUN",
    "lab_creatinine": "肌酐",
    "lab_hematocrit": "Hct",
    "lab_sodium": "钠",
    "lab_potassium": "钾",
    "lab_lactate": "乳酸",
    "lab_glucose": "血糖",
    "lab_inr": "INR",
    "lab_ph": "pH",
    "vital_heart_rate": "心率",
    "vital_nbps": "血压",
}


def _fmt_val(v: Any) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv == 0.0:
        return "—"
    if abs(fv - round(fv)) < 1e-9:
        return str(int(round(fv)))
    return f"{fv:.2f}"


@st.cache_data(show_spinner=False, ttl=120)
def _rich_stays(limit: int = 2500) -> tuple[dict[str, Any], ...]:
    """Stays with age + ≥2 labs at any hour (usable for demo)."""
    engine = get_engine()
    sql = """
        SELECT f.stay_id,
               MAX(COALESCE(s.los_hours, 0)) AS los_hours,
               MIN(f.hour_index) AS first_hour
        FROM feat.sample_matrix f
        LEFT JOIN staging.icustays s ON s.stay_id = f.stay_id
        WHERE f.hour_index = 1
          AND COALESCE((f.feature_json->>'anchor_age')::float, 0) > 0
          AND (
            (CASE WHEN COALESCE((f.feature_json->>'lab_creatinine')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_hematocrit')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_bun')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_lactate')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_sodium')::float, 0) <> 0 THEN 1 ELSE 0 END)
          ) >= 2
        GROUP BY f.stay_id
        ORDER BY f.stay_id
        LIMIT :lim
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"lim": int(limit)}).mappings().all()
    return tuple(
        {
            "stay_id": int(r["stay_id"]),
            "los_hours": float(r["los_hours"] or 0),
            "first_hour": int(r["first_hour"] or 0),
        }
        for r in rows
    )


@st.cache_data(show_spinner=False, ttl=120)
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


@st.cache_data(show_spinner=False, ttl=120)
def _cached_predict(stay_id: int, hour_index: int) -> dict[str, Any]:
    return predict_patient(int(stay_id), hour_index=int(hour_index))


@st.cache_data(show_spinner=False, ttl=120)
def _cached_traj(stay_id: int) -> dict[str, Any]:
    return predict_patient_trajectory(int(stay_id))


def _stay_options(stays: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"住院 {s['stay_id']} · 住院时长 {float(s.get('los_hours') or 0):.1f} 小时": int(
            s["stay_id"]
        )
        for s in stays
    }


def _pick_demo_stays(stays: list[dict[str, Any]], hour: int) -> tuple[int | None, int | None]:
    key = f"demo_pick_h{hour}_v3"
    if key in st.session_state:
        return st.session_state[key]
    scored: list[tuple[float, int]] = []
    for s in stays[:100]:
        sid = int(s["stay_id"])
        out = _cached_predict(sid, hour)
        q = out.get("feature_quality") or {}
        if (
            out.get("status") == "ok"
            and out.get("score_kind") == "probability"
            and q.get("usable")
        ):
            scored.append((float(out["risk_score"]), sid))
    hi = max(scored, key=lambda x: x[0])[1] if scored else None
    lo = min(scored, key=lambda x: x[0])[1] if scored else None
    st.session_state[key] = (hi, lo)
    return hi, lo


def _apply_forced_stay(options: dict[str, int]) -> None:
    if "force_stay_id" not in st.session_state:
        return
    fid = int(st.session_state.pop("force_stay_id"))
    for lab, sid in options.items():
        if sid == fid:
            st.session_state["mon_stay"] = lab
            return


def render_monitor() -> None:
    st.title("ICU 早期恶化预警 · 监测台")
    st.caption("S2 多时刻 · LightGBM + SHAP · 切换住院后按 stay/h 重新预测")

    import os

    for k in ("DATABASE_URL", "SQLALCHEMY_DATABASE_URI"):
        os.environ.pop(k, None)
    from infra.config import get_settings

    get_settings.cache_clear()

    metrics = load_metrics_artifact() or {}
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("训练样本", f"{metrics.get('total_n', '—'):,}" if metrics.get("total_n") else "—")
    mc2.metric("PR-AUC（测试）", f"{float(metrics.get('pr_auc_test') or 0):.3f}" if metrics else "—")
    mc3.metric("Brier（测试）", f"{float(metrics.get('brier_test') or 0):.3f}" if metrics else "—")
    mc4.metric("工作点", f"{float(metrics.get('operating_threshold') or 0):.3f}" if metrics else "—")

    stays = list(_rich_stays(2500))
    if not stays:
        st.error(
            "未找到可用特征行。请 restore S2 dump，并确认未再跑会清空 feat 的 P0 ETL。"
        )
        st.stop()

    cfg_hours = prediction_hours()
    with st.sidebar:
        st.markdown("### 患者")
        options = _stay_options(stays)
        labels = list(options.keys())
        _apply_forced_stay(options)
        if "mon_stay" not in st.session_state or st.session_state.get("mon_stay") not in options:
            st.session_state["mon_stay"] = labels[0]
        choice = st.selectbox("ICU 住院（stay）", labels, key="mon_stay")
        stay_id = int(options[choice])

        avail = list(_hours_for_stay(stay_id))
        hours = avail if avail else list(cfg_hours)
        prefer = 1 if 1 in hours else hours[0]
        hour_key = f"mon_hour_{stay_id}"
        if hour_key not in st.session_state or st.session_state[hour_key] not in hours:
            st.session_state[hour_key] = prefer
        hour = int(
            st.selectbox("预测时刻（入科后小时）", hours, key=hour_key)
        )

        if set(hours) != set(cfg_hours):
            st.caption(f"提示：当前库仅有时刻 {hours}（配置 {cfg_hours}）。")
        st.caption(f"DB：{get_settings().database_url.split('@')[-1]}")
        st.caption(f"当前 stay={stay_id} · h={hour}")
        st.caption("ui v4.1 · 列表仅含年龄+化验可用者")

        st.markdown("### 演示快捷")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("高风险样例", use_container_width=True, type="primary"):
                with st.spinner("扫描可用样例…"):
                    hi, _ = _pick_demo_stays(stays, hour)
                if hi is not None:
                    st.session_state["force_stay_id"] = hi
                    st.rerun()
                else:
                    st.warning("未找到可用高风险样例")
        with c2:
            if st.button("低风险样例", use_container_width=True, type="primary"):
                with st.spinner("扫描可用样例…"):
                    _, lo = _pick_demo_stays(stays, hour)
                if lo is not None:
                    st.session_state["force_stay_id"] = lo
                    st.rerun()
                else:
                    st.warning("未找到可用低风险样例")

    result = _cached_predict(stay_id, hour)
    traj = _cached_traj(stay_id)

    if result.get("status") != "ok":
        st.error(status_message(result))
        disclaimer()
        return

    # Guard against stale widget/cache mismatch
    if int(result.get("stay_id") or 0) != stay_id or int(result.get("hour_index") or -1) != hour:
        st.warning("预测结果与当前选择不一致，正在重算…")
        _cached_predict.clear()
        result = predict_patient(stay_id, hour_index=hour)

    score = float(result["risk_score"])
    kind = result.get("score_kind", "raw")
    rec = result.get("recommend") or {}
    band = str(rec.get("band", "unknown"))
    band_label = str(rec.get("label", band))
    thr = dict((load_yaml("labels.yaml").get("recommend") or {}))
    quality = result.get("feature_quality") or {}
    feats = result.get("features") or {}

    if quality.get("is_placeholder"):
        st.error(
            "当前为 P0 ETL 占位特征（仅 los/careunit）。请 restore "
            "`icu_decision_S2-full_mimic_*20260802.dump`。"
        )
    elif not quality.get("usable"):
        st.warning(
            f"该时刻化验偏少（lab_present={quality.get('lab_present', 0)}）。"
            "可改选其他住院，或换预测时刻。"
        )
    else:
        st.caption(
            f"可用特征 · 探针 {quality.get('clinical_present', 0)}/"
            f"{quality.get('clinical_total', 0)} · 化验非空 {quality.get('lab_present', 0)} · "
            f"stay={stay_id} · h={hour}"
        )
        st.caption(
            "说明：本 dump 生命体征（心率/血压等）多为缺测；模型主信号来自年龄 + 化验 + 科室等。"
        )

    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown(
            risk_badge_html(score, kind, band, band_label),
            unsafe_allow_html=True,
        )
        st.markdown("#### 建议动作")
        st.info(
            f"**{band_label}**（band=`{band}`）\n\n"
            f"阈值：观察 < {thr.get('observe', 0.2)} · "
            f"复查 < {thr.get('recheck', 0.4)} · "
            f"加强监护 < {thr.get('monitor', 0.7)} · 以上升级处置"
        )
        shown = list(DISPLAY_FIELDS.items())
        missing = sum(1 for k, _ in shown if _fmt_val(feats.get(k)) == "—")
        st.caption(f"面板缺测 {missing}/{len(shown)}（— = JSON 中无值；非把 0 当实测）")
        cols = st.columns(4)
        for i, (k, label) in enumerate(shown):
            with cols[i % 4]:
                st.metric(label, _fmt_val(feats.get(k)))

    with right:
        pts = [
            p
            for p in traj.get("points", [])
            if p.get("status") == "ok" and p.get("risk_score") is not None
        ]
        if pts:
            st.plotly_chart(
                fig_risk_trajectory(pts),
                use_container_width=True,
                key=f"traj_{stay_id}",
            )
        else:
            st.info("该住院暂无多时刻轨迹")

    st.subheader("可解释性（SHAP）")
    factors = result.get("top_factors") or []
    if quality.get("usable") and factors:
        c_a, c_b = st.columns([1.2, 1])
        with c_a:
            st.plotly_chart(
                fig_shap_bars(factors),
                use_container_width=True,
                key=f"shap_{stay_id}_{hour}",
            )
        with c_b:
            st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)
            with st.expander("完整特征向量（缺测为 null）"):
                st.json(feats)
    elif not quality.get("usable"):
        st.warning("特征不足，已隐藏 SHAP，避免用填充零值误导解释。")
    else:
        st.caption("无 SHAP 因子")

    disclaimer()
