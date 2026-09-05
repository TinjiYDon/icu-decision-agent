# 特征工程与临床数据挖掘最佳实践

## 特征选择策略对比

【适用：本项目的19特征选择依据】

**过滤法（Filter）**：
- 基于统计学方法（卡方、互信息、相关性）预选特征
- 优点：计算快，与模型无关
- 缺点：忽略特征交互，可能遗漏重要特征
- 本项目：部分使用（如基于临床知识筛选候选特征池）

**包裹法（Wrapper）**：
- 基于模型性能迭代选择特征子集
- 优点：考虑特征交互，性能通常更优
- 缺点：计算成本高，可能过拟合
- 本项目：未使用（样本量大，计算成本高）

**嵌入法（Embedded）**：
- 模型训练过程中自动选择特征
- LightGBM原生特征选择（基于分裂增益）
- 优点：高效，考虑特征交互
- 本项目：主要使用的方法（LightGBM内置特征重要性）

**本项目最终特征集（19个）**：
经临床知识+模型重要性双重筛选：
- 基线：anchor_age, careunit_other
- 生命体征：vital_temp, vital_gcs_total, vital_fio2, vital_heart_rate, vital_nbps, vital_resp_rate
- 化验：lab_lactate, lab_bun, lab_ph, lab_potassium, lab_inr, lab_albumin
- 衍生：pre_icu_los_hours, gcs_total, vasopressor_1h, shock_index, spo2_fio2_ratio

---

## 缺失值处理与临床意义

【适用：临床数据的特殊性】

**缺失机制分类**：
1. **MCAR（完全随机缺失）**：与任何变量无关（罕见）
2. **MAR（随机缺失）**：与已观测变量相关（常见）
3. **MNAR（非随机缺失）**：与未观测值相关（临床最常见）

**临床缺失的特殊性**：
- 未检测 = 临床判断不需要 → MNAR
- 例如：轻度患者不做乳酸检测 → 乳酸缺失可能提示病情较轻
- 直接删除或简单填充会丢失此信息

**本项目处理方法**：
- pre-ICU优先策略：优先使用入ICU前的数据
- 多重插补（Multiple Imputation）：用于关键实验室指标
- 指示变量：为关键缺失特征创建missing indicator
- 不插补的情况：特征本身代表"未检测"（如vasopressor_1h=0表示未使用）

**Missing Not At Random（MNAR）的SHAP解释**：
- 缺失指示特征本身可能具有预测价值
- 解释缺失特征时需注意：缺失 ≠ 正常值

来源：little RJ, rubin db. Statistical Analysis with Missing Data. 3rd ed. Wiley, 2019.

---

## 特征交叉与临床交互效应

【适用：模型性能优化方向】

**已实现的特征交叉**：
- shock_index = heart_rate / nbps（心率×血压交互）
- spo2_fio2_ratio = spo2 / fio2（氧合指标）
- gcs_total = eye + verbal + motor（已有）
- SOFA衍生特征：vasopressor_1h（心血管维度）

**潜在可探索的交叉特征**：
- age × lactate：高龄+高乳酸的风险叠加
- gcs × temperature：神经+感染双重评估
- sodium × albumin：渗透压与胶体渗透压关联
- BUN/Cr比值：肾前性氮质血症标记

**交互效应的SHAP解释**：
- SHAP interaction values可分解交互贡献
- 未来可展示特征对之间的交互效应图

---

## 特征标准化与模型鲁棒性

【适用：特征缩放策略】

**树模型 vs 距离模型**：
- 树模型（LightGBM）对特征尺度不敏感，无需标准化
- 本项目使用LightGBM，因此未对特征进行标准化
- 但SHAP值的解释不受标准化影响（TreeSHAP自动处理）

**标准化对下游分析的影响**：
- SHAP摘要图的数值范围受标准化影响
- 本报告直接使用原始SHAP值，不进行标准化
- 特征重要性排序不受影响

**鲁棒性测试建议**：
- 添加噪声测试：特征值±10%扰动下SHAP稳定性
- 跨数据库测试：同一特征在不同数据库中的一致性
- 时间稳定性测试：不同时间段数据的SHAP一致性

来源：Goldstein A, et al. arXiv:1602.02679. 2015.