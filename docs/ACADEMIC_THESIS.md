# 学术问题与创新纲领 · icu-decision-agent

> 2026-10-09 · 旧 S2 dump（dod 口径）为机制实验主表 · **无** Layer0 真序列 · 不宣称外部 SOTA  
> **总叙事入口**：[MASTER_NARRATIVE.md](MASTER_NARRATIVE.md)（本文展开假设与主表，不另立总目标）

## 1. 痛点

1. ICU **超早期**（入科后 \(h\in\{0,1,2,4,6\}\)）预测未来 12h 死亡，阳性极稀（旧 S2 ≈2.3%），单报 ROC 几乎无决策含义。  
2. 特征易混入预测时刻之后信息 → 指标虚高（时间因果）。  
3. 训练常优化 logloss/F1，评价才画 DCA → **训练目标与临床效用错位**。  
4. 时序模型与表格模型常「各训各测」不可比；缺测时不知该信谁。  
5. 解释不可追溯、不可人审 → 难进 CDSS 叙事。

## 2. 需求

| 角色 | 需求 |
|------|------|
| 临床/演示 | 风险分 + 工作点建议 + 可采纳/驳回 |
| 学术评审 | 相对谁、改了什么、同协议指标如何更好 |
| 工程 | MIMIC dump 可复现、stay 级划分、主报 PR-AUC/Brier/NB |

## 3. 约束

- 单中心 MIMIC；本机机制实验用 **旧 S2**；无 Layer0 → 真 GRU 序列后置  
- 默认床旁保持 LightGBM（可 TreeSHAP）  
- 独立 CDSS，不耦合床位调度  
- 禁止无外部验证宣称泛化 SOTA

## 4. 目标

在严格时间因果与 stay 公平协议下，交付：**决策效用可量化 + 表格/多时刻轨迹边界可证伪 + 解释可追溯** 的预警—协同系统；并在训练—阈值环补上「为净受益而选」的机制。

## 5. 价值主张

| 不是 | 是 |
|------|----|
| 又一个更炫的深度模型 | 可复现骨干上的 **评价—阈值—融合机制** |
| 刷高 AUC | 稀有事件下 **PR-AUC / Brier / 净受益** |
| 不可审计 LLM | SHAP→RAG→LLM + HITL |

## 6. 创新假设（答辩绑定）

| ID | 假设 | 相对谁 | 机制改动 | 主验收 |
|----|------|--------|----------|--------|
| **H1** | 精确标签叙事 + DCA | 粗 dod + 只报 ROC | 标签协议 + 决策曲线 | Brier / NB（v2 数字另表） |
| **H2** | 公平双轨 | 乱比两模型 AUC | 同 h/split/stay | PR-AUC / Brier |
| **H3** | 解释可信 + HITL | 仅 SHAP 条 | 引用校验 / RAG / audit | 引用有效率、采纳率 |
| **H4** | **效用校准工作点（UCEW）** | val 上 max-F1 再事后 DCA | val 上 **max net benefit(cost_ratio)** 选阈值 | test NB↑ 或同 NB 报警更少；PR-AUC/Brier 不崩 |
| **H5** | **稀疏门控多时刻融合（SGDF）** | 单时刻分数 / 无门控平均 | 用特征完备度门控「当前时刻」vs「多时刻轨迹」 | 同 split 下 PR-AUC/Brier 不劣；高稀疏子集门控可解释 |

> H5 在无 Layer0 时：时序轨 = **同模型多 `hour_index` 概率轨迹**（tabular trajectory surrogate），**不是** GRU-D 真序列；有 Layer0 后再换真序列轨，协议不变。

## 7. 非宣称

- 未外部验证 → 不写 SOTA / 首创网络结构  
- TFT-lite / GRU-D 非默认床旁  
- 旧 S2 与 v2 数字不得混表

## 8. 复现命令（旧 S2 + Docker :5433）

```powershell
$env:PYTHONPATH = (Get-Location)
$env:DATABASE_URL = "postgresql+psycopg://icu_dev:icu_dev@localhost:5433/icu_decision"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe scripts/h4_utility_compare.py --cost-ratio 4
.\.venv\Scripts\python.exe scripts/h5_sparsity_fusion.py --hour-index 6
```

## 9. 旧 S2 机制主表结果（2026-10-09）

### H4

| 策略 | thr | test NB | alert_rate | PR-AUC |
|------|----:|--------:|-----------:|-------:|
| max F1 | 0.537 | −0.053 | 1.7% | 0.139 |
| **max NB 约束** | **0.384** | **−0.035** | **2.2%** | 0.139 |
| max NB 无约束（诊断） | 0.01 | −0.016 | 100% | 0.139 |

→ 约束效用阈值在几乎不增加报警负荷下提升净受益；无约束会塌到 treat-all（方法学必写）。

### H5（h=6）

| 策略 | PR-AUC | Brier | NB |
|------|-------:|------:|---:|
| 当前时刻 | 0.164 | 0.043 | −0.0062 |
| 无门控平均 | 0.153 | 0.040 | −0.0059 |
| **稀疏门控** | 0.153 | **0.040** | **−0.0058** |

稀疏半：PR-AUC 0.075→0.086；完备半：0.215→0.192（门控非万能）。
