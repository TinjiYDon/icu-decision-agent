"""GRU-D 方法论文档（RAG 知识库）"""

# GRU-D 模型详解

## GRU-D 核心原理（Che et al. 2017）

【适用：时序模型解释】

GRU-D（Gated Recurrent Unit with Decay）是专为医疗时间序列设计的神经网络，
核心创新是将"缺失值模式"转化为预测信号。

**双衰减机制**：
1. **输入衰减（γ_x）**：当某个特征长时间缺失时，其插补值会趋向可学习的均值 x_mean
   - γ_x = exp(-max(0, W_x·δ + b_x))，δ = 距上次观测的时间
   - 临床含义：血压 3 小时未测 → 模型自动降低血压的权重
2. **隐藏状态衰减（γ_h）**：记忆单元随时间衰减
   - γ_h = exp(-max(0, W_h·δ̄ + b_h))，δ̄ = 平均时间间隔
   - 临床含义：昨天的生命体征对当前预测的影响小于今天的

**完整前向传播**：
```
x̃_t = m_t · x_t + (1-m_t) · [γ_x · x̂_{t-1} + (1-γ_x) · x_mean]
h̃_t = γ_h · h_{t-1}
r_t = σ(W_r [x̃_t; h̃_t])        # reset gate
z_t = σ(W_z [x̃_t; h̃_t])        # update gate
ĥ_t = tanh(W_h [x̃_t; r_t · h̃_t])
h_t = (1-z_t) · h_{t-1} + z_t · ĥ_t
p = σ(W_clf · h_t + b_clf)
```

**与 LightGBM 的关键区别**：
- LightGBM：每行独立，丢弃时序连续性
- GRU-D：完整利用时间顺序、缺失模式、不规则采样间隔

来源：Che Z, et al. Recurrent Neural Networks for Multivariate Time Series with Missing Values. Sci Rep. 2018;8:6085.

---

## GRU-D 在 ICU 预测中的优势

【适用：模型选型依据】

**为何 GRU-D 优于 LightGBM**：

1. **趋势捕捉**：乳酸从 1.5→4.0→6.0 的上升趋势，比单点值 6.0 更具预测价值
2. **缺失即信号**：护士不测血压可能是因为患者稳定，而非数据丢失
3. **不规则采样**：化验不是每小时都抽，GRU-D 原生支持 variable Δt
4. **端到端学习**：无需手动特征工程（如 shock_index, spo2_fio2_ratio）

**文献基准**：
- Che et al. (2018) PhysioNet 挑战：AUC=0.842±0.012（vs LR 0.799）
- Giesa et al. (2024) MIMIC-IV：AUROC=0.780，AUPRC=0.810（年龄分类）
- 本项目预期：ROC-AUC ≥ 0.80（对比 LightGBM 0.779）

---

## 梯度归因 vs SHAP

【适用：解释方法对比】

| 维度 | TreeSHAP（LightGBM） | Gradient×Input（GRU-D） |
|------|---------------------|------------------------|
| 原理 | Shapley 加性分解 | ∂pred/∂x · x |
| 计算速度 | 快（O(n·d)） | 快（单次反向传播） |
| 保和性 | ✅ 严格满足 | ✅ 近似满足 |
| 局部解释 | ✅ 强 | ✅ 强 |
| 全局解释 | ✅ SHAP摘要图 | ⚠️ 需聚合多个样本 |
| 对树模型 | 精确 | N/A |
| 对神经网络 | N/A | 标准方法 |

**注意**：Gradient×Input 是局部线性近似，不如 TreeSHAP 严格。
临床解读时仍需结合医学知识判断。

---

## 滑动窗口 vs 累积窗口

【适用：时间窗口设计】

**当前 LightGBM（累积窗口近似）**：
- h=1：用 [0,1h) 内所有数据的聚合值
- h=2：用 [0,2h) 内所有数据的聚合值
- 问题：h=2 包含 h=1 的全部信息，样本间高度相关

**GRU-D（滑动窗口）**：
- lookback=6h：固定使用最近 6 小时的原始时序
- 12 个时间步（30min 间隔），每个样本独立
- 优势：捕捉近期变化趋势，避免历史冗余

**训练效率对比**：
- LightGBM：472k 行 × 19 特征，训练 ~5 min
- GRU-D：94k stays × 12 步 × 8 特征，训练 ~2-4h（CPU）/~30min（GPU）

---

## 特征映射表

【适用：时序特征理解】

| GRU-D 特征 | MIMIC itemid | 来源表 | 正常范围 | 危急值 |
|-----------|-------------|--------|---------|-------|
| hr | 220045 | chartevents | 60-100 bpm | <40 或 >130 |
| sbp | 220179 | chartevents | 90-140 mmHg | <90 或 >180 |
| lactate | 50813 | labevents | 0.5-1.7 mmol/L | >2.5 或 >4.0 |
| creatinine | 50912 | labevents | 0.6-1.2 mg/dL | >3.0 |
| resp_rate | 220210 | chartevents | 12-20 次/分 | <8 或 >30 |
| temperature | 223761 | chartevents | 36.0-37.5°C | <35.0 或 >41.0 |
| spo2 | 220277 | chartevents | 95-100% | <90% |
| bun | 51006 | labevents | 7-20 mg/dL | >60 |
