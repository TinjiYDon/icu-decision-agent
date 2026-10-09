# Progress · icu-decision-agent

> 更新：2026-10-09（文档 C5 + HITL 演示种子；dump 仍缺）

## 里程碑

| 里程碑 | 状态 |
|--------|------|
| PR #9 解释链 | ✅ 合入 |
| deathtime 标签路径 | ✅ 代码 + **已在 Layer0 重算并重训** |
| S2 指标（v2） | ✅ test PR-AUC≈0.092 · Brier≈0.151 · 阳性 5638/472290 |
| GRU-D 真序列训练 | 🟡 smoke 有；chart/labs 6h 窗口稀疏，完整 ETL 未完成 |
| 新 dump 导出 | ⬜ 本机 `dumps/` 无 v2 `.dump`；restore / D3 真跑未做 |
| Demo DCA 口述 | ✅ `docs/DEMO_SCRIPT.md` |
| HITL 飞轮 KPI | ✅ STATUS 已写演示种子（非临床周报） |

## 下一刀

1. 把 `icu_decision_v2-full_mimic_94458stays_20260922.dump` 放到 `dumps/` 后 `restore_layer1.ps1`（Docker 默认 **5433**）  
2. pytest 整合白名单 + 可选 `d3_grud_minimal_compare` → 把 PR-AUC/Brier 写入 STATUS（C3）  
3. 优化 lab/chart 序列抽取后再训 PyTorch GRU-D；校准/工作点针对更低阳性率重定
