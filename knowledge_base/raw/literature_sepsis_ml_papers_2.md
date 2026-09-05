# 脓毒症ML预测顶刊论文摘要（二）：AKI、ARDS、多器官衰竭

## 论文6：ICU急性肾损伤AI预测系统综述（JAMIA Open，2025）

【适用：AKI预测与肾功能评估】

**标题**：Artificial intelligence models for predicting acute kidney injury in the intensive care unit: a systematic review
**期刊**：JAMIA Open（AMI，IF=6.2）
**年份**：2025年8月
**样本**：47项研究，仅14项进行了外部验证

**核心结果**：
- AKI预测模型AUC范围：0.76-0.94
- 深度学习（LSTM/Transformer）在时序预测方面优于静态模型
- 多数模型（94%）存在高风险偏倚（PROBAST评估）
- 仅30%模型报告了临床实用性指标（DCA/net benefit）
- BUN、肌酐、尿量是最常用的3个预测特征

**对本项目的意义**：
- 本项目BUN纳入特征，符合文献共识
- 提示需要更严格的外部验证（本项目使用eICU-CRD作为外部验证目标）
- 未来工作可探索LSTM时序建模以提升AKI预测能力

---

## 论文7：ML辅助脓毒症脓毒性休克预测（Sci Rep，2024）

【适用：脓毒性休克预测与SHAP解释】

**标题**：Development and validation of an interpretable model for predicting sepsis mortality across care settings
**期刊**：Scientific Reports（Nature Portfolio）
**年份**：2024年6月
**样本**：韩国多中心队列，脓毒症诊断标准

**核心结果**：
- 11个独立预测因子：年龄、功能状态评分（CFS）、恶性肿瘤、SOFA评分、呼吸道感染来源、CRRT使用、体温、白蛋白、INR、CRP、乳酸
- SHAP局部可解释性展示个体患者的风险驱动因素
- 乳酸每升高1 mmol/L，死亡风险OR增加约1.3倍
- SOFA评分每增加1分，死亡风险增加约10-15%

**对本项目的对应**：
- 本项目乳酸、SOFA衍生特征（vasopressor_1h）、年龄均获SHAP重要性确认
- INR已作为项目特征纳入，与文献高度一致
- 体温异常（低体温）在本项目中通过vital_temp特征覆盖

---

## 论文8：出血性卒中氧合变异度ML预测（Computers in Biology and Medicine，2026）

【适用：氧合指标与呼吸衰竭预测】

**标题**：利用氧合与呼吸变异度构建可解释的机器学习模型以预测出血性脑卒中死亡率
**期刊**：Computer Methods and Programs in Biomedicine（IF=6.0）
**年份**：2026年7月
**样本**：MIMIC-IV（n=2,262）+ eICU-CRD（n=2,076）多中心验证

**核心结果**：
- 梯度提升机模型内部AUC=0.911，外部AUC=0.792
- SHAP关键特征：心率分钟值、血糖平均值、钠离子平均值、TPE-RR、Slope-PaO₂、Mean-SpO₂、SD-SpO₂
- 氧合变异度（SpO₂标准差、PaO₂斜率）比单次测量值预测价值更高
- "U形"氧合关系：SpO₂<94%或>98%均与不良预后相关

**对本项目的意义**：
- 本项目SpO2/FiO2 ratio特征与"氧合指数"概念一致
- 呼吸频率变异度可作为未来特征扩展方向
- SpO₂/FiO2作为ARDS替代指标，与Berlin定义对应

---

## 论文9：入院衰弱与夜间睡眠碎片化预测ICU结局（Journal of Clinical Medicine，2026）

【适用：基线状态与ICU预后的关联】

**标题**：利用可解释人工智能根据入院时的虚弱状况和夜间睡眠片段化程度预测重症监护室（ICU）的不良结局
**期刊**：Journal of Clinical Medicine（MDPI，IF=4.5）
**年份**：2026年7月
**样本**：MIMIC-IV v3.1，n=31,139

**核心结果**：
- 三种梯度提升模型（XGBoost/CatBoost/LightGBM）AUC≈0.82-0.83
- SHAP关键特征：年龄、住院衰弱风险评分（HFRS）、夜间图表事件计数、夜间RASS评分
- 衰弱和睡眠碎片化独立贡献于死亡风险
- 复合终点：30天死亡率、机械通气>7天、出院至护理机构（发生率56.1%）

**对本项目的借鉴**：
- 年龄作为最强预测因子之一，与本项目SHAP结果一致
- HFRS（住院衰弱风险评分）概念可转化为临床合并症评分
- LightGBM表现与其他梯度提升方法统计等效，支持本项目方法选择

---

## 论文10：SepsisAI深度学习脓毒症预警（PLOS Digital Health，2025）

【适用：深度学习脓毒症预警与假阳性控制】

**标题**：Improving sepsis prediction in intensive care with SepsisAI: A clinical decision support system with a focus on minimizing false alarms
**期刊**：PLOS Digital Health（IF=4.1）
**年份**：2025年
**样本**：PhysioNet Challenge数据，n=40,336

**核心结果**：
- LSTM深度学习模型AUROC=0.95，AUPRC=0.96，敏感度88.19%，特异度96.75%
- 中位提前6小时发出警告，4小时发出警报
- 假警报率仅3.18%，远低于传统系统（>20%）
- 警告后88.19%的患者最终确诊脓毒症，正预测价值较高
- 输入特征：心率、血压、呼吸频率、体温、SpO₂、乳酸、白细胞计数

**对本项目的意义**：
- 本项目LightGBM的AUC=0.779低于深度学习，但在类别不平衡场景下PR-AUC=0.139更具参考价值
- 假阳性控制是本系统的重要临床考量——可通过调整风险阈值实现
- 生命体征（HR、NIBP、RR、Temp、SpO2）作为核心输入，与本项目特征高度重合
