# MIMIC-IV 数据库使用指南与本项目数据映射

## MIMIC-IV 数据库概览

【适用：数据库结构理解】

**数据来源**：
MIMIC-IV（Medical Information Mart for Intensive Care）是MIT实验室发布的公开ICU数据库，包含Beth Israel Deaconess Medical Center 2008-2022年的去标识化临床数据。

**核心表结构**：
- `hosp` schema：住院相关数据（patients, admissions, diagnoses_icd, procedures_icd, labevents, prescripions）
- `icu` schema：ICU专属数据（icustays, outputevents, chartevents）
- `em` schema：急诊数据（emergencymedicine）

**本项目使用的MIMIC表**：
- `hosp.patients`：患者 demographics
- `hosp.admissions`：住院信息（admit_time, disch_time, deathtime）
- `icu.icustays`：ICU入住记录（stay_id, intime, outtime, first_careunit）
- `hosp.labevents`：实验室检查（itemid, valuenum, flags）
- `icu.chartevents`：生命体征监护数据（itemid, value, valuenum）

---

## stay_id 与 subject_id 的关系

【适用：数据理解】

- `subject_id`：患者唯一标识（一个患者可能有多个 subject_id，因数据合并问题）
- `hadm_id`：住院唯一标识
- `stay_id`：ICU入住唯一标识（一个 hadm_id 可能有多个 stay_id，即多次ICU入住）

**本项目聚焦 stay_id 级别**：
- 每个 stay_id 产生5个预测时刻（intime+1h, +2h, +4h, +6h）
- 共94,458 stays，25%用于测试集（94,455 stays）
- 按 stay_id 分层划分，防止同一患者的不同时刻出现在不同fold

---

## 数据模型时间窗口设计

【适用：特征时间对齐理解】

**label 时间窗口**：
- 起点：预测时刻 intime + h
- 终点：预测时刻 intime + h + 12h
- 终点判定：in-hospital death within 12h

**特征时间窗口**：
- pre-ICU：第一次入院到ICU入住前的所有历史数据
- intra-ICU：intime 到 intime+h（预测时刻之前）
- 优先使用 pre-ICU 数据，仅在 pre-ICU 缺失时回退到 intra-ICU

**数据截断原则**：
- 预测时刻之后不可用的数据（look-ahead bias）
- 时间窗口确保因果关系方向正确

来源：Johnson A, et al. Sci Data. 2023;10:265.
