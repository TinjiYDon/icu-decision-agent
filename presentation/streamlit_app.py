from pathlib import Path

import pandas as pd
import streamlit as st

from application.predict_patient import get_label_config, list_stays, predict_patient

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "STATUS.md"

st.set_page_config(page_title="ICU Decision", layout="wide")
st.title("ICU 临床恶化预警")
st.caption("icu-decision-agent · LightGBM + SHAP · L4 · 本地调参 / MLflow")

labels = get_label_config()
st.sidebar.subheader("标签配置")
st.sidebar.json(labels.get("primary", {}))
st.sidebar.markdown("### 路径")
st.sidebar.code(str(ROOT), language=None)
st.sidebar.markdown("- 划分：`configs/labels.yaml` · 特征：`configs/features.yaml`")
st.sidebar.markdown("- 进度：`d:/project/_local-data/mimic/PROGRESS.md`")
st.sidebar.markdown("- MLflow：`mlflow ui --backend-store-uri sqlite:///./mlflow.db`")
st.sidebar.markdown("- 训练：`python -m application.train`（尝试写 mlflow.db）")

tab_pred, tab_lab = st.tabs(["单患者预测", "指标与学习路径"])

with tab_pred:
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
                    "- `anchor_age` / `gender_m` / `careunit_*`：入科可知（Wave1 无泄漏）\n"
                    "- 已剔除 `hospital_expire_flag`、总 `los_hours`（结局泄漏）\n"
                    "- 划分：train/val/test=0.7/0.1/0.2 · stay_id · seed=42"
                )
                st.json(result.get("features", {}))

with tab_lab:
    st.markdown(
        """
### 环境与路径
| 项 | 路径 |
|----|------|
| 仓库根 | `icu-decision-agent/` |
| 特征契约 | `configs/features.yaml` |
| 划分 | `configs/labels.yaml` → split |
| artifact | `artifacts/models/lgbm_mortality_12h.txt`（不入 Git） |
| dump | `dumps/*P0-full*`（线下分发） |
| MLflow | `sqlite:///./mlflow.db` |

### 命令
```powershell
$env:PYTHONPATH = (Get-Location)
.\\.venv\\Scripts\\pip.exe install mlflow streamlit
.\\.venv\\Scripts\\python.exe -m application.train
.\\.venv\\Scripts\\python.exe -m mlflow ui --backend-store-uri sqlite:///./mlflow.db
streamlit run presentation/streamlit_app.py
```
"""
    )
    if STATUS.exists():
        st.subheader("docs/STATUS.md")
        st.markdown(STATUS.read_text(encoding="utf-8"))
    else:
        st.info("无 STATUS.md")
