# Dump 交付说明（队友 restore / 训练）

> 更新：2026-07-27 · **dump 不入 GitHub** · 线下分发

## 人读摘要

| 仓 | 文件 | schemas_only | 可用于 |
|----|------|--------------|--------|
| decision | `icu_decision_P0-full_mimic_94458stays_20260726.dump` | **false** | ✅ **LightGBM 监督训练**（含 feat/label） |
| scheduling | `icu_scheduling_P0-full_mimic_94458stays_20260727.dump` | **false** | ✅ **SOFA/CP-SAT 仿真**（含 sofa/staging） |
| Layer0 `mimic.labevents` | 不在 Layer1 dump 内 | — | 本机已有 1.58e8 行；队友若无 Layer0 仍可用 scheduling dump 内已算好的 SOFA |
| online PPO 轨迹 | — | — | ❌ 不在 dump 内；见 Draft PR #3 / `PPO_SMOKE.md` |

## Agent 上下文

```text
decision restore: .\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_P0-full_mimic_94458stays_20260726.dump
然后: PYTHONPATH=. python -m application.train --from-existing
scheduling restore: 同上指向 20260727 dump
禁止: 把 schemas_only / 20260708 旧 dump 当全量训练底座
禁止: full dump 恢复后在无 Layer0 时运行默认 application.train（会重建 feat/label）
禁止: 宣称 Layer1 dump 含 labevents 或 PPO 轨迹
验收 DB: feat/label 或 sofa 行数 ≈ 94458
```

## 恢复命令

```powershell
# Decision
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_P0-full_mimic_94458stays_20260726.dump

# Scheduling
cd d:\project\icu-scheduling-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_scheduling_P0-full_mimic_94458stays_20260727.dump
```
