# 给老师的创新口径 · icu-decision-agent

> 2026-10-09 · 旧 S2 机制主表 · 不宣称外部 SOTA / 新网络结构  
> **总纲领**：[MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) · **公式/架构**：[ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md)

## 一句话

**总目标**：可决策的早期预警 CDSS。  
**请讲 CUGEW（H6）**：我们提出 **新指数 DEI/ANB/UEI**、**新算法 RWSG（近因加权门控）**、**新二级模型 SUFH（逻辑融合头）**——不是「又用了 LightGBM」。骨干 LGBM 只负责可复现打分。

## 不要这样讲

- 「我们用了 LightGBM / GRU，所以模型新。」
- 只报 ROC，不报 PR-AUC / Brier / 净受益 / **UEI**。
- 把 GRU-D / TFT-lite 说成已经替换默认床旁模型。
- 把无 Layer0 的多时刻轨迹说成「真序列深度学习」。

## 要这样讲

| 假设 | 相对谁 | 改了什么 | 旧 S2 证据（本机 2026-10-09） |
|------|--------|----------|-------------------------------|
| H1 | 粗标签 + 单报 AUC | DCA / 工作点协议 | `artifacts/dca/` |
| H4 | val 上 max-F1 | **约束 max-NB** | thr 0.537→0.384；NB −0.053→−0.035 |
| **H6 CUGEW** | 无决策指数；均匀轨迹；无融合头 | **DEI/ANB/UEI + RWSG + SUFH** | PR-AUC **0.164→0.177**；Brier **0.043→0.022**；**UEI 2.300→2.410** |
| H3 | 仅 SHAP 条 | 引用校验 + HITL | DX-1 |

## 底座 vs 创新（答辩一页）

| 层 | 是什么 | 对老师怎么说 |
|------|--------|----------------|
| 底座 | MIMIC + LightGBM | 可复现打分，**不是**论文卖点 |
| **新指数** | DEI / ANB / UEI | 把「优于全干预」与「少报警」写进指标 |
| **新算法 RWSG** | 近因加权 + 完备度门控 | 比均匀平均更合理的轨迹融合 |
| **新模型 SUFH** | 验证集训练的逻辑融合头 | 二级决策模型，系数可解释 |
| H3 | RAG + HITL | 可信交付 |

## 诚实负结果（加分）

- 无约束 max-NB 会塌到 treat-all → 必须加窗与报警帽。  
- SUFH 若用 class_weight=balanced 会过报警、毁校准 → 默认不用 balanced（写进方法）。

## 复现

```powershell
python scripts/h6_cugew_suite.py --hour-index 6 --beta 0.35 --cost-ratio 4
```
