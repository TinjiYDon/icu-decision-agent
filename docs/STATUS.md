# 项目状态

> 更新：2026-08-02 · **S1 早期预警**（`t=intime+1h`）· 多指标主叙事

## 定位

ICU **实时早期恶化预警**（非入科瞬间分诊）。详见工作区 [`POSITIONING.md`](../../docs/POSITIONING.md) · [`S1_EARLY_WARNING.md`](S1_EARLY_WARNING.md)。

## 数据检查点

| 项 | 状态 |
|----|------|
| Layer0 `mimic` @ :5432 | ✅ |
| S1 feat/label `hour_index=1` | ✅ 94,458（本机重建） |
| Wave2 dump（6-feat 对照） | ✅ `…20260726.dump` |
| 标签 | ⚠️ 日期级 `dod` · [`LABEL_AUDIT.md`](LABEL_AUDIT.md) |

## 对照：Wave2 弱基线（hour_index=0 · 6 特征）

| 指标 | val / test |
|------|------------|
| ROC-AUC | 0.673 / 0.693 |
| **PR-AUC** | **0.039 / 0.041** |
| Brier | 0.048 / 0.048 |
| 工作点 P/R（test） | 5.3% / 18.3% |

## S1 主模型（hour_index=1 · Model-A 精简特征 · 2026-08-02）

> `python -m application.train` · Layer0 rebuild · seed=42 stratified

| 指标 | val / test |
|------|------------|
| n / 阳性 | 94,458 / **2,162**（~2.29%）；split 66,121 / 9,446 / 18,891 |
| **ROC-AUC（对照）** | **0.769 / 0.770** |
| **PR-AUC（主）** | **0.147 / 0.149** |
| **Brier（主）** | **0.039 / 0.040** |
| 工作点阈值（val max-F1） | 0.573 |
| 工作点 P/R（test@阈值） | 30.6% / 18.2% |
| 特征 | 19 列（含 `vasopressor_1h` 治疗暴露） |
| artifact | `artifacts/models/lgbm_mortality_12h.txt`（不入 Git） |

**结论：S1 相对 Wave2 基线，PR-AUC 约 3.6×，ROC 亦提升；仍非临床可单独部署，须多指标与阈值叙事。**

## GitHub

- S1 分支 → PR merge（本轮）
- `liujiawei` 裸分支不合；已移植
- Issue #3/#4 已关

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m application.train   # 需 Layer0
```
