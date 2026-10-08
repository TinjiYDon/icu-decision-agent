# 给老师的创新口径 · icu-decision-agent

> 2026-09-29 · 与桌面汇报 PPT 对齐 · 不宣称外部 SOTA

## 一句话

**骨干仍是 LightGBM；加深的是标签、评价协议、时序公平对照、可追溯解释。**

## 不要这样讲

- 「我们用了 LightGBM / GRU，所以模型新。」
- 只报 ROC，不报 PR-AUC / Brier。
- 把 GRU-D 说成已经替换默认床旁模型。

## 要这样讲

| 假设 | 相对谁 | 改了什么 | 证据文件 |
|------|--------|----------|----------|
| H1 | 粗 `dod` + 单报 AUC | deathtime v2 + DCA | `STATUS.md` · `domain/models/dca.py` |
| H2 | 两个模型各训各测 | 同 split 双轨；缺 vitals 时 `--labs-only` | `scripts/d3_grud_minimal_compare.py` |
| H2 深化 | 仅 3 项化验 / 无注意力消融 | 10 项化验序列 · LGBM / GRU-D / TFT-lite 同分层 split · GRU PR-AUC +0.031 vs LGBM（混合抽样，非自然患病率） | `scripts/d3_lab_triple_compare.py` · `artifacts/d3/lab_triple_compare.md` |
| H3 | 只有 SHAP 条 | RAG 引用校验 + HITL | DX-1 · Streamlit「解释」 |

## 底座 vs 深化（答辩一页）

| 层 | 是什么 | 对老师怎么说 |
|------|--------|----------------|
| 底座 | MIMIC + 无泄漏特征 + LightGBM | 可复现骨干，不是卖点 |
| H2 深化 | 10 项化验不规则序列 · GRU-D + TFT-lite 同 split | Layer0 无 chartevents；主看 PR-AUC；负结果也报 |
| TFT-lite | 时间注意力消融 | 不是 Lim 全文 TFT，不是默认模型 |

## 演示台

Streamlit：**创新**页 + 验收页 DCA。TFT-lite 烟测：`python scripts/d3_tft_lite_smoke.py`。  
化验三方：`python scripts/d3_lab_triple_compare.py --limit 600`。
