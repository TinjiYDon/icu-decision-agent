# ICU 临床恶化预警与人机协同决策智能体（icu-decision-agent）

独立开源研究与工程项目 · 3 人协作 · 仓库 [icu-decision-agent](https://github.com/TinjiYDon/icu-decision-agent)

- **V3 优化规范与公式（先读 · SSOT）**：[`docs/OPTIMIZATION_SPEC.md`](docs/OPTIMIZATION_SPEC.md)
- **阶段 0–5 全流程行动清单**：[`docs/ACTION_CHECKLIST.md`](docs/ACTION_CHECKLIST.md)
- **唯一实验指标台账**：[`docs/METRICS_LEDGER.md`](docs/METRICS_LEDGER.md)
- **阶段门禁与文档总索引**：[`docs/STAGE.md`](docs/STAGE.md) · [`docs/README.md`](docs/README.md) · [`docs/COLLABORATION.md`](docs/COLLABORATION.md)

## 当前技术主线（2026-11-21 中期后 V3 架构）

```text
MIMIC-IV v3.1（storetime <= t · 在院存活过滤 · subject_id 患者级切分 · 精确 deathtime）
  → LightGBM 事件级目标消融（M0 强基线 → M1 事件/提前量加权 → M2a 事件级 LSE+BCE 自定义 fobj）+ 概率校准
  → 状态依赖动态复核队列（预算 K · D0 静态直选 / D1 静默排重 / D2 跃升再报 / D2-Dual 三区动作）
  → 同源时间粗化（LARE 1h/6h/12h/日级）+ 多自由度 ε-近优名单稳定性与归因
  → Streamlit 软著原型（SHAP→RAG→LLM 解释 · 带时间/来源的事实纠正与 VoI 留痕）
```

> 历史淘汰文档（旧 H4/H5/H6 CUGEW 及 7–10 月旧计划）已归档至 [`docs/archive/`](docs/archive/)。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
copy configs\database.yaml.example configs\database.yaml
copy configs\data.yaml.example configs\data.yaml
.\scripts\apply_migrations.ps1
$env:PYTHONPATH = (Get-Location)
# 运行无 DB 核心单测
.\.venv\Scripts\python.exe -m pytest tests/test_lare.py tests/test_lar_policy.py tests/test_lare_runner.py tests/test_policy_set.py tests/test_time_coarsening.py tests/test_selective_review.py tests/test_pac_safety_net.py tests/test_features_leak.py tests/test_smoke.py tests/test_dca.py -q
```

## 文档导航

| 文档 | 说明 |
| :--- | :--- |
| [docs/OPTIMIZATION_SPEC.md](docs/OPTIMIZATION_SPEC.md) | **V3 优化说明、痛点诊断与三大支柱算法公式** |
| [docs/ACTION_CHECKLIST.md](docs/ACTION_CHECKLIST.md) | **阶段 0–5 行动清单、分工矩阵与 Gate 门控** |
| [docs/METRICS_LEDGER.md](docs/METRICS_LEDGER.md) | **唯一实验指标台账（区分作废口径与 V3 正式口径）** |
| [docs/STAGE.md](docs/STAGE.md) | 阶段门禁与演进记录 |
| [docs/LARE.md](docs/LARE.md) / [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) | LARE / LAR-Policy 技术细节与模块索引 |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) / [docs/TUNING_LOCAL.md](docs/TUNING_LOCAL.md) | Streamlit 原型演示与启动 |
| [docs/DUMP_READY.md](docs/DUMP_READY.md) / [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | 数据恢复、工程结构与命令 |
| [docs/README.md](docs/README.md) | 完整文档索引与归档说明 |

## 控制台与 Docker

```powershell
docker compose up -d   # PostgreSQL 端口 5433（可选）
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
.\scripts\run_console.ps1
```

线下 dump 与 `artifacts/` **不入 Git**，遵守 [`docs/adr/001-layer-boundaries.md`](docs/adr/001-layer-boundaries.md) 分层边界。

## Independent experiment archive

See [experiments/README.md](experiments/README.md) for the repository-local algorithms, comparison results, reproduction steps and historical limitations.
