# 项目状态

> 更新：2026-08-14 · **标签 v2（deathtime）重训** · 主指标 PR-AUC / Brier  
> **演示台**：总览 / 监测 / **解释（PR#9）** / 方法 / 调参 / 验收

## 定位

ICU **实时早期恶化预警**：预测时刻 `t=intime+h`，`h∈{0,1,2,4,6}`。

## 对照简表

| 模型 | 样本 | ROC test | **PR-AUC test** | Brier test |
|------|------|----------|-----------------|------------|
| Wave2 6-feat h=0 | 94k stays | 0.693 | **0.041** | 0.048 |
| S1 单时刻 h=1（旧 dod） | 94k | 0.770 | **0.149** | 0.040 |
| S2 多时刻（旧 dod） | 472k | 0.779 | **0.139** | 0.040 |
| **S2 + deathtime 标签 v2** | **472k** | **0.776** | **0.092** | **0.151** |

> v2 阳性率约 **1.19%**（旧 dod 约 2.34%）。精确死亡时间后标签更稀、更严；PR-AUC 下降符合预期，**不以刷高旧指标为理由退回 dod**。

## S2 主模型（2026-08-14 · label_version=`mortality_12h_v2_deathtime`）

> 同特征矩阵 · 重算 label · `python -m application.train --from-existing`

| 指标 | val / test |
|------|------------|
| n_stays / rows | 94,458 / **472,290** |
| 阳性（全表） | **5,638**（旧 11,039） |
| **ROC-AUC（对照）** | 0.796 / **0.776** |
| **PR-AUC（主）** | 0.118 / **0.092** |
| **Brier（主）** | 0.148 / **0.151** |

标签来源计数（按 stay×hour 行）：deathtime 命中约 12%；其余回退 dod / 无死亡时间。

完整 metrics：`artifacts/models/metrics_mortality_12h.json`。

## 数据 / 功能

| 项 | 状态 |
|----|------|
| Layer0 `:5432/mimic` | ✅ 含 `admissions.deathtime` |
| S2 feat | ✅ 472,290（未改特征） |
| Label v2 | ✅ 已重算并重训 |
| 解释链 | ✅ PR #9 合入 · Streamlit「解释」 |
| GRU-D | 🟡 骨架+synthetic smoke；Layer0 chart 前 6h 多为空，真序列 ETL 仍待 |
| S2 dump 文件 | ⚠️ 磁盘 dump 仍为旧标签版；**库内已是 v2**（需另打新 dump 才线下同步） |

### 监测台注意

- 清脏 `DATABASE_URL`：`.\scripts\run_console.ps1`
- 勿在 restore 后跑会 TRUNCATE feat 的 P0 ETL

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_labels_deathtime.py tests/test_grud_smoke.py -q
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe -m application.train_grud --batch 8
```
