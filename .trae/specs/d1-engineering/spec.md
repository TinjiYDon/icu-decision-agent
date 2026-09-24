# D1 工程化落地计划 · Spec

> change-id: `d1-engineering`  
> 对齐原始参考方案，适配当前项目状态与 AI-only 评估限制

---

## Why

D1 解释链安全对照已完成五阶基线（50 stays，7 项指标），证明了 RAG 链路在特征覆盖和事实 grounding 上优于纯 LLM，但**引用有效率仅 44.9%**（核心瓶颈），且缺少面向"有证据、可追溯、可复核"的系统性自动化评估。本计划将 D1.2 检索优化与 D1.3 可解释性验证工程化落地，形成可重复、可度量的质量提升闭环。

## What Changes

### D1.2 检索与提示词优化

- **[新增] 引用校验层重构**：`reference_validator.py` 分段容差策略 + 改用 parent_content 做校验；引用无效时的原因分类（无匹配/单位不匹配/截断丢失）
- **[新增] RAG 路由扩展**：`rag_retriever.py` 新增 10+ 主题路由（BUN/肌酐/WBC/PLT/MAP/SpO2/FiO2/INR/胆红素/白蛋白/血小板），统一指向 threshold 类别
- **[新增] Few-shot 示例注入**：`prompts.py` 的 `USER_PROMPT_TEMPLATE` 加入 2 条高质量示例（从 D 模式高引用有效率案例选取），展示期望的输出格式和内容深度
- **[修改] 因果词后处理精细化**：`prompts.py` 的 `_CAUSAL_PATTERNS` 拆分为"强制替换"（导致/引起/引发/造成/诱发/致）和"白名单保留"（提示/关联/多见于/常见于/伴随/相关），避免医学自然表述被机械替换
- **[修改] 提示词约束松动**：去除"必须以前缀'模型统计观察'开头"的强制要求，改为"建议以'模型统计观察'作为每个因子的起始标记"，让 LLM 有更好的表达自由度

### D1.3 可解释性自动化验证

- **[新增] Human-aligned 指标模块**：`audit.py` 新增三个指标
  - `risk_consistency`：SHAP方向与解释文本中风险表述的一致性（高风险因子不应说"降低风险"）
  - `clinical_term_density`：因子解释中使用医学术语的比例（来自 feature_meta.yaml 标准名）
  - `readability_score`：因子解释长度是否落在合理区间 [80, 250] 字符
- **[新增] LLM-as-Judge 质量评估**：`domain/explain/quality_eval.py`（新文件），用 Agnes LLM 对 D 模式输出做以下评判：
  - `evidence_support_score`（0-5分）：解释是否有充分文献支撑
  - `hallucination_check`（bool）：是否存在与 RAG context 明显矛盾的内容
  - `clinical_coherence_score`（0-5分）：解释整体临床逻辑是否自洽
- **[新增] 解释质量报告生成**：`scripts/generate_explain_report.py`（新文件），汇总所有指标生成一份完整的 D1.3 报告（Markdown），包含当前质量基线、趋势分析、改进建议
- **[新增] 知识分块策略验证**：`tests/test_chunk_strategy.py`（新文件），对比固定长度分块 vs 语义分块（按临床语义边界切分）对检索命中率的影响，使用已有 30 文档基准

### 文档与记录

- **[新增] docs/D1_ENGINEERING_PLAN.md**：完整工程化落地计划（本 spec 的执行版）
- **[修改] docs/STATUS.md**：D1 部分更新当前指标基线和各阶段目标

## Impact

- **Affected specs**: 对齐 D1.2 + D1.3 两部分需求，补充原始参考方案中已实现部分的现状记录
- **Affected code**:
  - `domain/explain/reference_validator.py` — 重写
  - `domain/explain/rag_retriever.py` — 扩展路由
  - `domain/explain/prompts.py` — 添加 few-shot + 调整约束
  - `domain/explain/audit.py` — 新增 3 个指标 + LLM-as-Judge 集成
  - `domain/explain/quality_eval.py` — **新增**
  - `scripts/generate_explain_report.py` — **新增**
  - `tests/test_explain_audit.py` — 扩展测试
  - `tests/test_chunk_strategy.py` — **新增**
  - `knowledge_base/chunks/` — 重建索引（如分块策略变化）
- **No breaking changes** to L4 interface (`predict_patient_with_explanation`)
- **No breaking changes** to existing D1 audit modes (A/B/C/D)

## MODIFIED Requirements

### Requirement: 引用校验
**当前**：绝对容差 ±0.05，使用 snippet[:200] 做校验  
**改为**：分段容差（比值 ±0.5 / 浓度 ±15% 相对 / 百分比 ±2）+ 使用 parent_content 做校验，保留 snippet 用于 UI 展示

### Requirement: RAG 查询路由
**当前**：6 个主题路由（sofa/ards/shock_index/lactate/gcs/sepsis）  
**改为**：16+ 个主题路由，覆盖 MIMIC 常用 ICU 特征（BUN/肌酐/WBC/PLT/MAP/SpO2/FiO2/INR/胆红素/白蛋白/乳酸清除率等）

### Requirement: 因果词处理
**当前**：全局替换 6 个因果词为"与...相关"  
**改为**：仅替换强制列表中的词；白名单中的医学惯用词（提示/关联/多见于等）保留

### Requirement: D1 评估指标
**当前**：7 项自动指标（SHAP对齐/引用有效/过度承诺/特征覆盖/事实grounding/证据具体度/降级率）  
**改为**：10 项（+风险一致性/临床术语密度/可读性分数），新增 LLM-as-Judge 3 项评分

## REMOVED Requirements

无移除项。

---

## 与原始参考方案的对齐说明

| 参考方案项 | 我们的实现方式 | 理由 |
|-----------|--------------|------|
| PubMed API + Pyserini BM25 | 保持 FAISS + m3e-base，扩展路由 | 知识库固定（30文档），BM25 收益边际；路由扩展是更实际的召回优化 |
| BioBERT/MedCPT embedding | 暂时保持 m3e-base | 切换 embedding 需要重建整个索引；可后续评估 |
| Cross-encoder 重排 | 暂不实现 | 当前 top-k=3 已足够；重排在大规模检索下才有意义 |
| 句级相关性过滤 + LLM 摘要压缩 | 暂不实现 | 当前 parent_content 策略已保证上下文完整；压缩会损失可追溯性 |
| NLI 句级引用支持验证（DeBERTa） | **LLM-as-Judge 替代** | 无 GPU 环境，LLM 可完成同等任务且无需额外模型部署 |
| 实体归因验证 | `risk_consistency` 指标 + LLM hallucination_check | 用 SHAP方向一致性检查替代，避免引用与患者背景不匹配 |
| 临床医生双盲评估 | **AI-only 替代**：LLM-as-Judge + 多维自动指标 | 无医生资源；采用标准化 Prompt 评判，可复现 |
| 反事实测试（证据消融/替换/SHAP扰动） | **E1-E6 消融实验**（工程化实现） | 通过脚本批量跑，替代人工反事实操作 |

---

## 验收标准（D1.3 目标）

| 指标 | 当前基线（50 stays, h=1） | Phase1 后目标 | Phase2 后目标 |
|------|------------------------|-------------|-------------|
| 引用有效率 | 44.9% | **≥ 70%** | **≥ 75%** |
| 特征覆盖密度 | 18.5% | 18.5%（不变） | **≥ 22%** |
| 事实 grounding 率 | 82.6% | 82.6%（不变） | **≥ 85%** |
| 风险一致性 | 未测量 | — | **≥ 90%** |
| 可读性分数 | 未测量 | — | **≥ 0.80** |
| LLM evidence_support | 未测量 | — | **≥ 4.0 / 5** |
| LLM hallucination_rate | 未测量 | — | **≤ 5%** |
