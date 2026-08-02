# 项目状态

> 更新：2026-08-02 · **S2 多时刻流式样本** · 主指标 PR-AUC / Brier

## 定位

ICU **实时早期恶化预警**：预测时刻 `t=intime+h`，`h∈{0,1,2,4,6}`。见 [`S1_EARLY_WARNING.md`](S1_EARLY_WARNING.md) · 工作区 `POSITIONING.md`。

## 对照简表

| 模型 | 样本 | ROC test | **PR-AUC test** | Brier test |
|------|------|----------|-----------------|------------|
| Wave2 6-feat h=0 | 94k stays | 0.693 | **0.041** | 0.048 |
| S1 单时刻 h=1 | 94k | 0.770 | **0.149** | 0.040 |
| **S2 多时刻网格** | **472k**（94k×5） | **0.779** | **0.139** | **0.040** |

## S2 主模型（2026-08-02）

> `prediction_hours: [0,1,2,4,6]` · stay 级分层 split · `python -m application.train`

| 指标 | val / test |
|------|------------|
| n_stays / rows | 94,458 / **472,290** |
| **ROC-AUC（对照）** | 0.772 / **0.779** |
| **PR-AUC（主）** | 0.163 / **0.139** |
| **Brier（主）** | 0.040 / **0.040** |

### test 按时刻切片（同一模型）

| h | ROC | PR-AUC | P / R @工作点 |
|---|-----|--------|----------------|
| 0 | 0.694 | 0.066 | 23% / 7% |
| 1 | 0.778 | 0.131 | 27% / 18% |
| 2 | 0.802 | 0.155 | 28% / 23% |
| 4 | 0.808 | 0.166 | 28% / 26% |
| 6 | 0.809 | 0.164 | 27% / 25% |

完整 `metrics_by_hour_test` 见 `artifacts/models/metrics_mortality_12h.json`。

**结论**：S2 用多时刻训练后，整体 PR-AUC 与 S1 同量级；较晚时刻（h=4/6）切片 ROC/PR 更高，符合「观察越久信息越多」。禁止单报 ROC。

## 数据

| 项 | 状态 |
|----|------|
| Layer0 `:5432` | ✅ |
| S2 feat/label | ✅ hour_index∈网格 |
| 标签精度 | ⚠️ 日期级 dod |

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m application.train
```
