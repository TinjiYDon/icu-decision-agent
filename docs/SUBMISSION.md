# 双创提交版说明（decision）

## 版本

- 分支：`release/shuangchuang-2026-11`
- 基线：`origin/main`（V3 阶段 0，`4ea13c2`）
- 预计标签：`v2026.11-submit`

## 本次包含

- LARE 评估协议：`domain/models/lare.py`
- LAR-Policy / Hoeffding：`domain/models/lar_policy.py`
- ε-最优策略集：`domain/models/policy_set.py`
- 受控时间粗化：`domain/labels/time_coarsening.py`
- MIMIC LARE 运行器：`scripts/lare_mimic_run.py`
- V3 阶段 0 文档：`docs/OPTIMIZATION_SPEC.md`、`docs/ACTION_CHECKLIST.md`、`docs/METRICS_LEDGER.md`
- 离线依赖快照：`requirements.lock.txt`

## 验证结果（本机，2026-11-23）

```text
核心无 DB 单测：87 passed（10 个文件）
preflight：15/15 passed（使用项目 .venv）
```

## 实验边界（必须随材料一起说明）

1. 当前仓库的实验数字来自 2026-11-13 报告（V2a/V2b 历史口径），**不是 V3 正式封版数字**；`docs/METRICS_LEDGER.md` 的 V3 表仍为空。
2. 报告使用的 B0–B7 全量体征脚本、`lare_controlled_validation.py` **尚未合入本仓库**；原始实验代码与结果在仓外 `experiments/`。
3. V3 的 `storetime` 因果过滤、在院存活过滤、`subject_id` 60/20/20 切分仍处于阶段 1，未完成。
4. 因此本版本可提交“中期代码 + 实验报告”，不能宣称“V3 最终实验已完成”。

## 复现

```powershell
$env:PYTHONPATH=(Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_lare.py tests/test_lar_policy.py tests/test_lare_runner.py tests/test_policy_set.py tests/test_time_coarsening.py tests/test_selective_review.py tests/test_pac_safety_net.py tests/test_features_leak.py tests/test_smoke.py tests/test_dca.py -q
```

实验报告与结果：`../experiments/report/`、`../experiments/decision/results/`。
