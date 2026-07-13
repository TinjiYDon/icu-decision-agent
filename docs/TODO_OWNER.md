# 成员 C · 集成负责人待办

## 已完成（本阶段）

| 任务 | 文件 | 状态 |
|------|------|------|
| L4 单患者预测 | `application/predict_patient.py` | ✅ |
| L3 predict 接口 | `domain/models/lgbm.py` · `predict_stay` | ✅（与 B 联调） |
| Streamlit 演示 | `presentation/streamlit_app.py` | ✅ 选 stay → 风险 + SHAP |
| 测试 | `tests/test_predict.py` | ✅ smoke |

## 待完成

| 任务 | 说明 |
|------|------|
| MCP `predict_risk` | S4 · P2 |
| 时序模型 GRU-D/TFT | P1 · B 主责 |
| STATUS 随训练指标更新 | 与 B 联调后填 AUC |

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
python -m application.train          # 需 Layer1 数据
python -m application.predict_patient
streamlit run presentation/streamlit_app.py
pytest tests/test_predict.py -q
```
