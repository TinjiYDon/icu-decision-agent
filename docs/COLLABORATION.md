# 三人协作手册 · icu-decision-agent

> 独立仓库，与 icu-scheduling-agent **无代码依赖**。本文面向全体成员。

## 1. 角色与目录

| 成员 | 角色 | 拥有目录 | 入口命令 |
|------|------|----------|----------|
| **A** | 数据/基础设施 | `domain/etl/` `migrations/` `scripts/` `infra/` | `run_data_pipeline.ps1` |
| **B** | 算法 | `domain/features/` `domain/labels/` `domain/models/` | `application/train.py` |
| **C** | 应用/集成负责人 | `application/` `presentation/` `data_access/` | `application/run_p0.py` |

各目录详见同级 `OWNER.md`。

## 2. 集成负责人（成员 C）职责

- 合并 PR 前确认 `pytest` 与分层规则
- 维护 [`STATUS.md`](STATUS.md)（全仓进度唯一真相）
- 裁定 L4 接口变更（如 `predict_patient(stay_id)` 返回值）

## 3. 垂直切片（按 sprint 交付）

不要「A 只写 ETL、B 只写模型、C 只写 UI」三头并进无联调——按**切片**顺序推进：

| Sprint | 切片 | 主责 | 协作 | 验收 |
|--------|------|------|------|------|
| S0 ✓ | 数据检查点 | A | — | `run_data_pipeline.ps1` 绿 |
| S1 | 训练闭环 | B | A 保 feat/label 表 | `train` 产出 artifacts + 指标 |
| S2 | 单患者预测 | C | B 提供 predict 接口 | L4 可调、pytest |
| S3 | Streamlit 演示 | C | B 解释字段 | 选 stay_id 见分数 |
| S4 | MCP（可选） | C | B 定义 schema | `predict_risk` JSON |

任务清单：[`BACKLOG.md`](BACKLOG.md)

## 4. 对接契约

| 类型 | 位置 | 规则 |
|------|------|------|
| 表结构 | `migrations/*.sql` | 仅 A 提交；B/C 提需求开 issue |
| 训练输出 | `artifacts/` + `model.registry` | B 定义；C 读取推理 |
| L4 API | `application/*.py` docstring | C 维护；B 不直接改 Streamlit |
| 配置 | `configs/*.example` | 改 example + PROJECT_GUIDE 一行 |
| 数据版本 | `dumps/DATA_VERSION_*.json` | A 导出时自动生成 |

## 5. 周会（15 分钟）

1. 各成员报：完成了哪条切片、`STATUS` 是否更新
2. C 报：有无 blocked / 接口变更
3. 定下周 1–2 条切片，写入 issue

## 6. 调试分工

| 现象 | 先找 |
|------|------|
| ETL / staging 行数不对 | A |
| 训练指标异常 | B → A 查 feat |
| Streamlit 报错 | C → 查 L4 是否调通 |
| restore / dump 失败 | A |

## 7. 相关文档

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- [adr/001-layer-boundaries.md](adr/001-layer-boundaries.md)
