# 贡献指南

## 团队（3 人）

| 角色 | 职责 | 主要目录 |
|------|------|----------|
| **成员 A · 数据/基础设施** | ETL、migration、dump、环境脚本 | `domain/etl/` `migrations/` `scripts/` `infra/` |
| **成员 B · 算法** | 特征、标签、模型、训练指标 | `domain/features/` `domain/labels/` `domain/models/` |
| **成员 C · 应用/集成** | L4 用例、Streamlit、联调、**集成负责人** | `application/` `presentation/` `data_access/` |

详细分工与任务切片：[`docs/COLLABORATION.md`](docs/COLLABORATION.md)

## 分层规则（必须遵守）

```
L5 presentation  →  L4 application  →  L3 domain  →  L2 data_access  →  L1 infra
```

- **禁止** Streamlit 直接 `import domain.*` 或写 SQL
- **禁止** `domain/` 内直接创建 DB 连接（经 L2 Repository）
- 改 L4 函数签名前：开 issue，@集成负责人（成员 C）

## 合并门槛

```powershell
$env:PYTHONPATH = (Get-Location)
pytest tests/test_smoke.py tests/test_db.py tests/test_etl.py -q
# 若改算法：pytest tests/test_train.py -q（存在时）
# 若改数据管道：.\scripts\run_data_pipeline.ps1 -SkipEtl  # 可选
```

## Pull Request

使用 [PR 模板](.github/pull_request_template.md)，必填：变更层、验证命令、影响范围。

## Issue 标签

`data` · `algo` · `app` · `ui` · `infra` · `bug` · `blocked`

任务 backlog：[`docs/BACKLOG.md`](docs/BACKLOG.md)
