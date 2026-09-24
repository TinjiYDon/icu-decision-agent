# 本仓顶尖视角下一步（decision）

> 2026-09-24 · **独立项目** · 已采入 DX-1/2/3  
> 不与 scheduling 硬耦合。

## 总原则

1. **决策价值 > 排行榜 AUC**：主报 PR-AUC、Brier、DCA/净受益；ROC 仅对照。
2. **时间因果硬约束**：特征 `charttime < t`；标签窗口 `[t, t+H]`。
3. **人机闭环可审计**：分数 → 解释 → 采纳/驳回 → care_plan。
4. **本仓独立交付**：不调用床位优化 API。

## vNext（对齐已采入波次）

| 优先级 | 项 | 建议落地 |
|--------|----|----------|
| P0 | D3 主表 PR-AUC/Brier | `d3_grud_minimal_compare` + STATUS |
| P0 | dump + pytest 整合绿 | [INTEGRATION_PREP.md](INTEGRATION_PREP.md) |
| P1 | H3 飞轮 KPI | `summarize_hitl` |
| P1 | Demo 挂 DCA | [DEMO_SCRIPT.md](DEMO_SCRIPT.md) |
| P2 | care_plan 打磨 | 非床位 |
| P2 | ADR-001 #12 | 可选 |

## 非目标

- 接 scheduling `optimize_beds`
- 用 GRU-D 替换 TreeSHAP→RAG→LLM 主链
- 宣称临床生产上线
- 以 ROC 作为 D3 唯一卖点
