# Merge 门槛 · 对齐最终目标

> 2026-09-24 · Owner：`TinjiYDon`  
> **最终目标**：有创新点 · 优化实际产出 · 算法与架构先进性

## 合入判决规则

| 判据 | 过线标准 |
|------|----------|
| **创新点** | 绑定 H1–H4 / DX-1..3 / S2-MOO，或可写进答辩「相对谁更好」 |
| **实际产出** | 可跑通 demo / 报告 / 可解释分配；指标进 STATUS |
| **算法先进性** | 公平对照、DCA、多目标机理、时序双轨等 |
| **架构先进性** | ADR 分层、L4 门面、禁 UI 直连 SQL/domain |

**不合**：仅删文档/清文件、无度量、与创新假设无关的 nightly chore。

## 已合入（创新主线）

| PR | 仓 | 贡献 |
|----|-----|------|
| [#18](https://github.com/TinjiYDon/icu-decision-agent/pull/18) | decision | DX-1/2/3 解释·DCA·公平双轨 |
| 后续 `8e7b482` | decision | PR-AUC/Brier 主验收纠偏 |
| [#10](https://github.com/TinjiYDon/icu-scheduling-agent/pull/10) | scheduling | S2-MOO（H4） |

## 本批建议合入

| PR/分支 | 判据 | 动作 |
|---------|------|------|
| **ADR-001**（`fix/adr-001-rebase`） | **架构先进性**：UI→L4 `ui_queries`，禁 Streamlit 直连 DB | **合入** |
| decision #15/#16/#17 · scheduling #9 | 清理无创新产出 | **关闭**（不占主线） |

## 合入后仍不做宣称

- 无外部验证 → 不写泛化 SOTA  
- 无 S2-TRAJ → 不写 online MIMIC-PPO  
- 不跨仓硬耦合  

## 下一拍（产出闭合）

见各仓 [INTEGRATION_PREP.md](INTEGRATION_PREP.md)：dump 真跑 → STATUS 填 PR-AUC/Brier 与 MOO 表 → HITL KPI / S2-TRAJ。
