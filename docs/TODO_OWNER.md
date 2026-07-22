# 成员 C · 集成负责人待办

## 已完成（本阶段）

| 任务 | 文件 | 状态 |
|------|------|------|
| L4 单患者预测 | `application/predict_patient.py` | ✅ 缓存 list_stays + label config |
| L3 predict 接口 | `domain/models/lgbm.py` · `predict_stay` | ✅ SHAP 缓存（与 B 联调） |
| Streamlit 演示 | `presentation/streamlit_app.py` | ✅ 分数展示修正 |
| 测试 | `tests/test_predict.py` | ✅ smoke |

## 待完成

| 任务 | 说明 |
|------|------|
| MCP `predict_risk` | ✅ 骨架 · `python -m presentation.mcp_server` |
| 时序模型 GRU-D/TFT | P1 · B 主责 |
| STATUS 随训练指标更新 | 与 B 联调后填 AUC |
| Bugbot | ✅ 已开 2026-07-22 · `docs/BUGBOT.md` |

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
python -m application.train          # 需 Layer1 数据
python -m application.predict_patient
streamlit run presentation/streamlit_app.py
pytest tests/test_predict.py -q
```
