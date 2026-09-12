# 特征阈值对照表（基于 feature_meta.yaml）

> 自动生成于 feature_meta.yaml · 2026-08-18

## 监测阈值汇总

| 特征键 | 标准名 | 单位 | 正常下限 | 正常上限 | 危急下限 | 危急上限 | 证据级 |
|-------|--------|------|---------|---------|---------|---------|-------|
| anchor_age | Age（年龄） | 岁 | — | 65 | — | 80 | A |
| careunit_other | ICU Type（非 SICU/MICU/CCU） | 分类变量 | — | — | — | — | B |
| vital_temp | Body Temperature | °C | 36.0 | 37.5 | 35.0 | 41.0 | A |
| vital_gcs_total | GCS Total | 分 | 13 | 15 | 8 | — | A |
| vital_fio2 | FiO2 | 小数 | 0.21 | 0.40 | — | 1.0 | A |
| vital_heart_rate | Heart Rate | bpm | 60 | 100 | 40 | 130 | A |
| vital_nbps | NIBP Systolic | mmHg | 90 | 140 | 90 | 180 | A |
| vital_resp_rate | Respiratory Rate | 次/分 | 12 | 20 | 8 | 30 | A |
| lab_lactate | Lactate | mmol/L | 0.5 | 1.7 | — | 4.0 | A |
| lab_bun | BUN | mg/dL | 7 | 20 | — | 60 | B |
| lab_ph | pH | 无单位 | 7.35 | 7.45 | 7.20 | 7.60 | A |
| lab_potassium | Potassium | mmol/L | 3.5 | 5.0 | 2.5 | 6.0 | A |
| lab_inr | INR | 无单位 | 0.8 | 1.2 | — | 3.0 | A |
| lab_albumin | Albumin | g/dL | 3.5 | 5.0 | 2.0 | — | B |
| pre_icu_los_hours | Pre-ICU Length of Stay | 小时 | — | — | — | — | C |
| gcs_total | GCS Total | 分 | 13 | 15 | 8 | — | A |
| vasopressor_1h | Vasopressor Use in 1h | 0/1 | — | — | — | — | A |
| shock_index | Shock Index | 无量纲 | 0.5 | 0.7 | — | 0.9 | A |
| spo2_fio2_ratio | SpO2/FiO2 Ratio | 比值 | 400 | 500 | 100 | — | A |

## 阈值临床解读

### 高危及异常触发逻辑

- **乳酸 >4.0**：强烈提示组织低灌注，需立即评估脓毒症/休克
- **GCS <8**：昏迷状态，需气管插管保护气道
- **休克指数 >0.9**：血流动力学不稳定，SSC bundle 指征
- **SpO2/FiO2 <300**：符合 ARDS 轻度标准
- **pH <7.20**：严重酸中毒，需紧急纠正
- **血钾 >6.0 或 <2.5**：致命性心律失常风险，心电监护
- **INR >3.0**：凝血功能障碍，活动性出血风险

### 年龄与科室风险

- **年龄 >80 岁**：高龄是独立死亡风险因素
- **CCU（冠心病监护室）**：心血管事件为主，注意心源性休克
- **非 SICU/MICU/CCU**：其他 ICU 类型风险基线不同

## 派生指标说明

### 休克指数（Shock Index）
- 计算公式：SI = Heart Rate / NIBP Systolic
- 来源：Fukuda et al., J Emerg Med 2017
- 正常值：0.5-0.7；危险阈值：>0.9
- 优势：比单一心率或血压更能综合反映血流动力学状态

### SpO2/FiO2 比值
- 计算公式：SpO2（百分数）/ FiO2（小数）
- 来源：ARDS Berlin Definition 2012
- 正常值：400-500；轻度 ARDS：<300
- 注意：与 PaO2/FiO2 比值近似（约为 P/F 的 0.8 倍）
