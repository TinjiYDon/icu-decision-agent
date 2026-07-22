# 参数与数据故事（人机可读）

> Owner：C 维护 · B 补全训练指标 · A 补全 dump 完整性  
> 更新：2026-07-22 · 路线：**方案 C**（监督学习 + 可解释，不做 PPO）

## 标签

| 字段 | 含义 | 来源 |
|------|------|------|
| `mortality_12h` | 入科后 12h 内 ICU/院内死亡 | `configs/labels.yaml` → `label.mortality_12h` |
| `horizon_hours` | 预测窗 | 当前实现 12；规划 4/8/12/24 |

## 特征（P0）

| 特征 | 含义 | 风险 |
|------|------|------|
| `anchor_age` | 年龄 | — |
| `gender_m` | 男性=1 | — |
| `los_hours` | 已住 ICU 小时 | — |
| `hospital_expire_flag` | 院内死亡标志 | **可能标签泄漏**，训练应评估剔除 |

## 建议档位（`recommend`）

| band | 默认阈值（概率） | 含义 |
|------|------------------|------|
| observe | <0.2 | 观察 |
| recheck | <0.4 | 复查 |
| monitor | <0.7 | 加强监护 |
| escalate | ≥0.7 | 升级处置 |

L4：`predict_patient` → `recommend` 字段；UI：Streamlit 展示。

## dump 说明

本地 `dumps/DATA_VERSION_*.json` 若 `schemas_only: true`，**不足以**做强化学习或完整故事分析；需 A 提供含数据的 Layer1 dump 或跑 `run_data_pipeline.ps1`。

## 验收

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -q
```
