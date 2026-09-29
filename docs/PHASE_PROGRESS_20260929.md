# 阶段性进展总结 · icu-decision-agent · 2026-09-29

## 定位

独立 ICU **早期恶化预警 + AIGC 人机协同 CDSS**；预测时刻 `t=intime+h`；**不**做床位调度、不耦合 scheduling。

## 本阶段已交付

| 波次 | 内容 | 证据 |
|------|------|------|
| 标签 v2 | deathtime 精确死亡标签重训 | STATUS · PR-AUC test **0.092** |
| DX-1 | 解释安全（引用/过承/RAG） | PR#18 · 50 stays 评测 |
| DX-2 | DCA 决策曲线 | `domain/models/dca.py` |
| DX-3 | GRU-D 时序双轨对照 | D3 报告 · 主验收 PR-AUC/Brier |
| ADR-001 | UI→L4 `ui_queries` 架构分层 | PR#19 已合 |

## 关键数字

| 指标 | test |
|------|------|
| ROC-AUC（对照） | 0.776 |
| **PR-AUC（主）** | **0.092** |
| **Brier（主）** | **0.151** |

## 合入状态

open PR = **0**；#12/#15–17 chore 已关；无需再 merge。

## 下一拍

HITL KPI 写入 STATUS；可选 dump 同步；演示挂 DCA 口述。
