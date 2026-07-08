# ICU 临床恶化预警决策智能体

> 仓库：`icu-decision-agent`  
> 数据库：`icu_decision`（PostgreSQL 16）

本仓库**独立运行**：自带 ETL、模型与演示，不依赖其他 ICU 项目或共享 MCP 服务。

---

## 1. 定位与边界

**做什么**

- 单患者 ICU 恶化/死亡风险预警；Level 2–3 人机协同（辅助、不替代）。  
- 输出风险分、SHAP 归因、分级临床建议；Streamlit 单患者演示。

**不做什么**

- 不依赖其他 ICU 项目的数据库、API 或 MCP 服务。

**模型路线（并行）**

- **LightGBM**：表格聚合特征 + SHAP + Isotonic 校准（先打通 pipeline）。  
- **GRU-D → TFT**：时序/波形特征（依赖 Waveform 123），与 LightGBM 并存，非二选一。

**输入 / 输出**

| 方向 | 内容 |
|------|------|
| 输入 | 基线、聚合时序特征、干预特征（来自 MIMIC 表） |
| 输出 | 0–100 风险分、Top3–5 因素、观察/复查/关注/评估 |

---

## 2. 领域架构

### 2.1 业务流水线（单向）

```
ETL 对齐 → 特征工程 → 标签构建 → 训练/校准 → 推理 → 解释 → 建议 → 展示
```

| 阶段 | 模块 | 持久化 |
|------|------|--------|
| ETL | `domain/etl` | `staging.*` |
| 特征 | `domain/features` | `feat.sample_matrix` |
| 标签 | `domain/labels` | `label.outcomes` |
| 模型 | `domain/models` | 文件 `artifacts/*.pkl` + `model.registry` 表 |
| 解释 | `domain/explain` | 内存 / 可选 `app.predictions` |
| 建议 | `domain/recommend` | 规则，无独立表 |

### 2.2 代码五层与调用关系

```
streamlit_app  (L5)
      │ 调用
application/   (L4)  run_etl · train · predict_patient · evaluate
      │ 调用
domain/        (L3)  etl · features · labels · models · explain · recommend
      │ 调用
data_access/   (L2)  MimicRepository · FeatureRepository · LabelRepository · ModelRepository
      │ 调用
infra/         (L1)  db · config · logging · paths
```

**推理路径（单患者）**

```
L5 选 stay_id
  → L4 predict_patient(stay_id)
    → L2 读 feat.*
    → L3 models.predict → calibrate → explain.shap → recommend.rules
  → L5 渲染
```

---

## 3. 技术栈与工具职责

| 工具 | 职责 | 不使用于 |
|------|------|----------|
| **PostgreSQL 16** | 特征、标签、预测记录、实验元数据 | — |
| **SQLAlchemy + psycopg** | Repository | 训练循环内逐行 INSERT |
| **Polars** | 特征宽表构建、train/test 矩阵 | 替代 PG 做主库 |
| **LightGBM** | 表格特征分类 | 数据存储 |
| **PyTorch** | GRU-D / TFT 时序模型 | 数据存储 |
| **sklearn** | Isotonic 校准、划分 | — |
| **SHAP** | 个体/全局解释 | 训练 |
| **Streamlit** | 演示 | 训练脚本 |
| **本地 GPU** | MVP 不用；DL 扩展时用 | LightGBM MVP |

---

## 4. 目录结构

```
icu-decision/
├── PLAN.md
├── configs/
│   ├── database.yaml
│   ├── data.yaml           # source: mock | mimic
│   ├── features.yaml       # 变量列表、时间窗
│   └── labels.yaml         # 恶化/死亡、horizon 定义
├── infra/
├── data_access/
│   ├── mimic_repo.py
│   ├── feature_repo.py
│   ├── label_repo.py
│   └── model_repo.py
├── domain/
│   ├── etl/
│   ├── features/
│   ├── labels/
│   ├── models/             # train · predict · calibrate
│   ├── explain/            # shap
│   └── recommend/          # 分级规则
├── application/
│   ├── etl_pipeline.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── presentation/
│   └── streamlit_app.py
├── artifacts/              # .gitignore，模型文件
├── migrations/
├── tests/
└── reports/
```

---

## 5. 数据与样本逻辑

### 5.1 样本单位

- 主键：`stay_id`  
- 时间轴：ICU 入科 = 0，**小时**重采样  
- 划分：按 `stay_id` 7:1:2，**同一 stay 不得跨集合**

### 5.2 标签（定义在 `labels.yaml`）

| 字段 | 说明 |
|------|------|
| `horizon_h` | 4 / 8 / 12 / 24 |
| `event` | 死亡或恶化代理（如 SOFA 升高、升压药升级） |
| `label` | 未来 horizon 内是否发生 |

MVP 可先上 **12h + 死亡** 单任务，再扩展多 horizon。

### 5.3 特征三类

| 类型 | 示例 | 表 |
|------|------|-----|
| 基线 | 年龄、性别、ICD | `feat.baseline` |
| 动态聚合 | 6h/12h/24h HR/MAP/乳酸 mean/min/max | `feat.timeseries_agg` |
| 干预 | 通气、升压药 | `feat.interventions` |

宽表合并 → `feat.sample_matrix`（一行 = stay_id × 时间点 或 stay 级快照，在 `features.yaml` 定稿）。

### 5.4 缺失值（MVP）

1. 临床不可能值 → NULL  
2. 缺失率 > 20% 列 → 丢弃  
3. 其余：组内前向填充 → 中位数  

逻辑在 `domain/features/impute.py`，与存储分离。

---

## 6. 模型与输出逻辑

### 6.1 训练

```
feat.sample_matrix + label.outcomes
    │  domain/models/train.py
    ▼
artifacts/lgbm_{horizon}.pkl
artifacts/isotonic_{horizon}.pkl
model.registry 表（版本、指标、路径）
```

**触发**：`application/train.py --horizon 12`

### 6.2 推理与校准

```
raw_score = lgbm.predict_proba(x)
cal_score = isotonic.transform(raw_score)
risk_0_100 = round(cal_score * 100)
```

### 6.3 建议规则（`domain/recommend/rules.py`）

| P_calibrated | 建议 |
|--------------|------|
| < 0.2 | 观察 |
| 0.2 – 0.4 | 复查 |
| 0.4 – 0.7 | 关注 |
| ≥ 0.7 | 评估 |

### 6.4 SHAP

- 训练集背景样本 → `explainer` 序列化或运行时采样  
- 单患者：`shap_values` → Top3–5 特征名 + 贡献方向  
- 由 L4 `predict_patient` 一并返回 dict，L5 只负责展示  

---

## 7. 配置自洽

| 文件 | 消费者 |
|------|--------|
| `database.yaml` | L1，`database=icu_decision` |
| `data.yaml` | L4 ETL，`source` / `mimic_dsn` |
| `features.yaml` | L3 features |
| `labels.yaml` | L3 labels |

---

## 8. 模块依赖顺序（实施时按此顺序打通）

```
1. infra + migrations
2. data_access + mock 表
3. domain/etl + features + labels  → feat + label 表有数据
4. domain/models/train + evaluate   → 基线 AUC 可出
5. domain/models/predict + calibrate
6. domain/explain + recommend
7. application 编排（CLI 全流程）
8. presentation/streamlit
9. ETL 切换 mimic 源，重训
```

---

## 9. 评估逻辑（`application/evaluate.py`）

| 维度 | 实现 |
|------|------|
| 算法 | AUC、F1、Recall@阈值 |
| 预警 | 各 horizon 提前量、覆盖率 |
| 可解释 | Top 特征与文献对照（报告） |
| 性能 | 单 stay 推理耗时 |

结果写入 `reports/` 与 `app.experiment_runs` 表。

---

## 10. 扩展 backlog

| 项 | 接入点 |
|----|--------|
| GRU-D / TFT | `domain/models/temporal/`，新 feat 序列输入 |
| SAITS 补全 | `domain/features/impute_deep.py`，ETL 前处理 |
| FastAPI | 新建 `presentation/api/`，L4 复用 |
| MLflow | 替换 `model.registry` 或双写 |
| Waveform/ECG | 新 `raw.waveform` + 特征模块 |

---

