import streamlit as st

from infra.config import load_yaml

st.set_page_config(page_title="ICU Decision", layout="wide")
st.title("ICU 临床恶化预警")
st.caption("icu-decision-agent · LightGBM + SHAP")

labels = load_yaml("labels.yaml")
st.subheader("当前标签配置")
st.json(labels.get("primary", {}))
st.info("骨架已就绪。MIMIC 导入 PostgreSQL 后运行 ETL 与训练。")
