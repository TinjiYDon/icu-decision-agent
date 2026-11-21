# 本仓顶尖视角下一步（decision）

> 2026-09-29 · **独立项目** · 已采入 DX-1/2/3 · TFT-lite 消融 · labs-only D3 通路

## 总原则

1. **决策价值 > 排行榜 AUC**：主报 PR-AUC、Brier、DCA/净受益；ROC 仅对照。
2. **时间因果硬约束**：特征 `charttime < t`；标签窗口 `[t, t+H]`。
3. **人机闭环可审计**：分数 → 解释 → 采纳/驳回 → care_plan。
4. **本仓独立交付**：不调用床位优化 API。

## vNext

| 优先级 | 项 | 状态 |
|--------|----|------|
| P0 | D3 主表 PR-AUC/Brier | ✅ 协议；`--labs-only` 真跑（缺 vitals dump） |
| P0 | 化验时序三方对照 | ✅ 10 labs · LGBM/GRU-D/TFT-lite · 分层 · `lab_triple_compare` |
| P0 | dump + pytest 整合 | 持续 |
| P1 | H3 飞轮 KPI | ⚠️ audit 仍 empty（需演示台提交反馈） |
| P1 | Demo 挂 DCA +「创新」页 | ✅ DEMO_SCRIPT |
| P2 | care_plan 打磨 | 可选 |
| P2 | 导入 chartevents 后恢复「要求 vital」门控 | 可选加深 |

## 非目标

- 接 scheduling `optimize_beds`
- 用 GRU-D / TFT-lite 替换默认 LightGBM 床旁推理
- 宣称临床生产上线 / 外部 SOTA
