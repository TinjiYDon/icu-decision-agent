# 代码整合与完善准备清单 · decision

> 2026-09-24 · Owner：`TinjiYDon` · 前置：创新点已采入 ROADMAP/SOTA

## 已就绪（勿重复造）

| 模块 | 路径提示 |
|------|----------|
| 解释工程 DX-1 | `domain/explain/reference_validator.py` · `rag_retriever.py` · `prompts.py` · `quality_eval.py` |
| DCA DX-2 | `domain/models/dca.py` · `scripts/d2_dca_full.py` · UI `accept.py` / `charts.py` |
| 双轨 DX-3 | `scripts/d3_grud_minimal_compare.py` · `tests/test_d3_grud_compare.py` |

## 整合顺序（建议）

1. `git pull origin main`  
2. restore 符合 `DUMP_READY.md` 的 dump（含多 hour 若跑 D3）  
3. `pytest tests/test_dca.py tests/test_d3_grud_compare.py tests/test_reference_validator.py tests/test_explain_audit.py -q`  
4. 可选真跑：`python scripts/d3_grud_minimal_compare.py` → 检查报告 **PR-AUC/Brier 主表**  
5. 可选：`python scripts/d2_dca_full.py`；`python -m application.summarize_hitl`  

## 完善缺口（代码侧）

| ID | 项 | 状态 |
|----|----|------|
| C1 | D3 `paired_compare` 主验收改 PR-AUC/Brier | 本批落地 |
| C2 | 测试读 JSON `encoding=utf-8` | 本批落地 |
| C3 | STATUS 正式对照表填 PR-AUC/Brier（需真跑数字） | 待本机 dump |
| C4 | HITL KPI 写入 STATUS | 待跑飞轮 |
| C5 | Demo 脚本挂 DCA 口述 | 文档小改可后置 |

## 禁区

- dumps/ artifacts/ 不入 Git  
- 不接 scheduling 床位 API  
