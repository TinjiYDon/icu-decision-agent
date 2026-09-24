# 方向同步（组员必读）· 2026-09-24

> Owner：`TinjiYDon` · 已合入：PR [#18](https://github.com/TinjiYDon/icu-decision-agent/pull/18) → `main`  
> 请 `git pull origin main` 后阅读；有异议在 Issue / 群内回复。

## 人读摘要

| 项 | 内容 |
|----|------|
| 定位 | **独立** ICU 早期恶化预警 + AIGC 人机协同 CDSS |
| 禁止 | 调用 / 依赖 `icu-scheduling-agent` 床位 API |
| 本周合入 | 组员自选包 **D1 解释工程化 + D2 DCA + D3 GRU-D 公平对照**（PR #18 · Liu Jiawei） |
| 请做 | `git pull`；跑下方验收；勿再基于旧 `liujiawei` 分支开 PR |

## 命名对照（避免和旧 ROADMAP 波次混淆）

| 组员交付编号 | 实际内容 | 对应旧 ROADMAP / 假设 |
|--------------|----------|------------------------|
| **D1**（本次） | 引用分段容差 · RAG 18 主题 · Prompt 详细解释 · 质量指标 | ≈ H3 解释质量 / AIGC 可信 |
| **D2**（本次） | 决策曲线 DCA + Bootstrap CI + 验收页 | ≈ 临床效用 / 工作点净受益延伸 |
| **D3**（本次） | GRU-D vs LGBM 同 h=6、同 split、stay 对齐 | ≈ 旧 **D1** 双轨对照 |
| 旧 ROADMAP「D2 HITL / D3 care_plan」 | 已在更早 PR#14 落地骨架 | 继续补飞轮 KPI，非本次重点 |

## 已合入改动（请 Review）

- `domain/explain/*` 引用校验 / RAG 路由 / Prompt / quality_eval  
- `domain/models/dca.py` · `scripts/d2_dca_full.py` · 验收页 CI-DCA  
- `scripts/d3_grud_minimal_compare.py` · GRU-D `gamma_h` / padding / paired_compare  
- 测试：`test_dca` · `test_d3_grud_compare` · `test_explain_audit` · `test_reference_validator` 等  
- 文档：`D1_OPTIMIZATION_SUMMARY.md` · `D3_PLAN.md` · `DUMP_READY.md` 更新  

## 验收命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_dca.py tests/test_d3_grud_compare.py tests/test_reference_validator.py tests/test_explain_audit.py -q
# 可选（需 dump / 模型）：
# .\.venv\Scripts\python.exe scripts\d3_grud_minimal_compare.py
# .\.venv\Scripts\python.exe scripts\d2_dca_full.py
```

## 下一阶段（Owner 建议）

| 优先级 | 项 | Owner 建议 |
|--------|----|------------|
| P0 | 本机 restore 新 dump + 全量 pytest 绿 | A + C |
| P0 | D3 对照结果写回 `STATUS.md` 正式表（PR-AUC/Brier，勿只报 ROC） | B |
| P1 | HITL 飞轮周 KPI 填表（`summarize_hitl`） | C |
| P1 | 解释质量真库复测（非 mock）写入报告 | B |
| P2 | ADR-001 UI 分层 Draft #12（有空再合） | C |
| 禁 | 跨仓接 scheduling；无 LIT 宣称 SOTA | 全员 |

## Agent 上下文

```text
SSOT: docs/STATUS.md · docs/TEAM_DIRECTION.md
merged: PR#18 D1/D2/D3 teammate package
禁区: dumps/ artifacts/ · 禁止接 scheduling optimize_beds
```
