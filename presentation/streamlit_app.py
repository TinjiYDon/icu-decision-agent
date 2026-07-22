import pandas as pd
import streamlit as st

from application.predict_patient import get_label_config, list_stays, predict_patient

st.set_page_config(page_title="ICU Decision", layout="wide")
st.title("ICU 临床恶化预警")
st.caption("icu-decision-agent · LightGBM + SHAP · L4 `predict_patient`")

labels = get_label_config()
st.sidebar.subheader("标签配置")
st.sidebar.json(labels.get("primary", {}))

stays = list(list_stays(limit=500))
if not stays:
    st.warning("未找到 ICU stays。请先运行 ETL：`scripts/run_data_pipeline.ps1`")
    st.stop()

options = {f"stay {s['stay_id']} · LOS {float(s.get('los_hours') or 0):.1f}h": s["stay_id"] for s in stays}
label = st.selectbox("选择 ICU stay", list(options.keys()))
stay_id = options[label]

if st.button("计算 12h 恶化风险", type="primary"):
    result = predict_patient(stay_id)
    if result.get("status") != "ok":
        st.error(result.get("message", result.get("status")))
    else:
        score = result["risk_score"]
        kind = result.get("score_kind", "raw")
        if kind == "probability":
            st.metric("12h mortality risk", f"{score:.2%}")
        else:
            st.metric("12h mortality model score (raw)", f"{score:.4f}")
        rec = result.get("recommend") or {}
        if rec:
            st.info(f"建议档位：**{rec.get('label', rec.get('band'))}**（band=`{rec.get('band')}`）")
            st.caption(f"阈值 observe/recheck/monitor = {rec.get('thresholds')}")
        st.subheader("Top 影响因素 (SHAP)")
        st.dataframe(pd.DataFrame(result["top_factors"]), use_container_width=True)
        with st.expander("特征向量与数据故事"):
            st.markdown(
                "- `anchor_age` 年龄 · `gender_m` 男性指示 · `los_hours` 已住 ICU 时长\n"
                "- `hospital_expire_flag` **可能泄漏标签**（与院内死亡强相关），B 训练时应评估剔除"
            )
            st.json(result.get("features", {}))

st.divider()
st.caption("训练模型：`python -m application.train` · 全链路：`python -m application.run_p0`")
