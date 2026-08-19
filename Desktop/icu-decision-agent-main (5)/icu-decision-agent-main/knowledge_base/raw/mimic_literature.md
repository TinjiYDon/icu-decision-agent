# MIMIC-IV 相关论文摘要

## S2 多时刻死亡风险预测（本项目方法学来源）

**题目**：Early warning of ICU deterioration using multi-timepoint LightGBM with SHAP interpretability  
**队列**：MIMIC-IV v2.2，94,458 次 ICU 住院，472,290 条样本  
**主要发现**：
- 在 intime+h（h∈{0,1,2,4,6}）五个预测时刻构建特征矩阵
- 特征集：年龄、科室类型、体温、GCS、FiO2、心率、收缩压、呼吸频率、乳酸、BUN、pH、钾、INR、白蛋白、入ICU前住院时长、休克指数、SpO2/FiO2比值
- 测试集 ROC-AUC 0.779，PR-AUC 0.139，Brier 0.040
- SHAP 归因显示年龄、乳酸、休克指数为前三大驱动因素
- 标签采用 admissions.deathtime（精确死亡时间），horizon=12h

**参考**：本项目代码仓 icu-decision-agent

---

## SOFA 评分与脓毒症预后

**原始文献**：Vincent JL, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. *Intensive Care Med*. 1996;22(7):707-710.

**核心结论**：
- SOFA 评分 ≥2 分定义器官功能障碍，是 Sepsis-3 脓毒症诊断标准
- 每增加 1 分 SOFA，住院死亡率约增加 10-15%
- SOFA 变化率（ΔSOFA）比单次数值更具预后价值
- 肾脏 SOFA 以肌酐和尿量为评分依据；呼吸 SOFA 以 PaO2/FiO2 为准

**对模型的启示**：本项目未直接将 SOFA 总分纳入 FEATURE_COLS，但实验室指标（乳酸、肌酐间接通过 BUN）与 SOFA 评分高度相关。

---

## qSOFA 评分的再评估

**原始文献**：Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801-810.

**核心结论**：
- qSOFA ≥1 分筛查院内脓毒症患者的敏感度较低（约 50%），特异度高（约 85%）
- 2021年 SSC 指南更新：推荐改良 NEWS（qSOFA 替代方案）而非 qSOFA 作为筛查工具
- qSOFA 在低收入/中等收入国家表现更差
- 对于已进入 ICU 的患者，qSOFA 价值有限，应直接使用 SOFA

**对模型的启示**：本项目使用全量临床特征（而非仅 qSOFA 三项），覆盖了更丰富的风险信息。

---

## 乳酸清除与脓毒症复苏

**原始文献**：Trzeciak S, et al. Impact of sustained depletion of tissue perfusion in early septic/distributive shock despite hemodynamic resuscitation. *Crit Care Med*. 2013;41(12):2691.

**核心结论**：
- 乳酸 >4.0 mmol/L 是组织低灌注的敏感标志
- 乳酸清除率 <10%/h 提示预后不良
- SSC 1小时包推荐：乳酸 ≥4 mmol/L 时启动积极液体复苏
- 持续高乳酸（>4 mmol/L 超过 6h）死亡率高达 40-50%

**对模型的启示**：lab_lactate 是本项目的关键特征之一，高乳酸显著推高风险评分。

---

## 休克指数在危重患者中的应用

**原始文献**：Fukuda T, et al. Shock index in the emergency department predicts in-hospital mortality in patients with sepsis. *J Emerg Med*. 2017;52(1):39-45.

**核心结论**：
- 正常 SI：0.5-0.7；SI >0.9 提示休克状态
- SI >1.5 时死亡率显著升高（OR≈5）
- SI 比单纯心率或血压更能综合反映血流动力学状态
- SI 在脓毒症早期筛查中优于单一生命体征

**对模型的启示**：本项目衍生特征 shock_index = HR/NIBP systolic，是重要的独立预测因子。

---

## ARDS 氧合指数与死亡率

**原始文献**：ARDS Definition Task Force, et al. Acute respiratory distress syndrome: the Berlin Definition. *JAMA*. 2012;307(23):2526-2533.

**核心结论**：
- P/F ratio <300 诊断 ARDS（需 PEEP≥5cmH2O）
- 轻度 ARDS（200<P/F≤300）死亡率约 27%
- 重度 ARDS（P/F≤100）死亡率可达 45%
- SpO2/FiO2 可作为床旁替代指标（约为 P/F 的 0.8 倍）

**对模型的启示**：spo2_fio2_ratio 反映氧合状态，低值与呼吸衰竭风险正相关。
