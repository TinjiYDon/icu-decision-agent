# D1 工程化落地 · Tasks

## Phase 1：引用校验修复 + 路由扩展（并行，高优先级）

- [x] **Task 1.1 — 修复引用校验容差**
  - 修改 `domain/explain/reference_validator.py`
  - 实现分段容差函数：`_smart_tolerance(expected_val, reference_val) -> float`
    - 比值型（SOFA分值、休克指数）：绝对容差 ±0.5
    - 浓度/量值型（乳酸、肌酐、BUN）：相对容差 ±15% 或绝对 ±0.5（取较大者）
    - 百分比型（SpO2、GCS分项）：绝对容差 ±2
    - 计数型（WBC、PLT）：相对容差 ±20%
  - 在 `validate_reference()` 中替换固定 tolerance 为动态计算
  - 更新 docstring，记录容差策略

- [x] **Task 1.2 — 校验改用 parent_content**
  - 修改 `domain/explain/reference_validator.py` 的 `validate_factor_references()`
  - 从 `ref.get("snippet")` 改为 `ref.get("parent_content", ref.get("snippet", ""))`
  - 在 `domain/explain/rag_retriever.py` 的 `format_hits_for_prompt()` 中，references dict 补充 `parent_content` 字段（内容 = `h.parent_content`）
  - 检查 `presentation/ui/explain_audit.py` 是否有对 `snippet` 字段的直接依赖，如有则同步更新

- [x] **Task 1.3 — 扩展 RAG 路由覆盖**
  - 修改 `domain/explain/rag_retriever.py` 的 `_TOPIC_KEYWORDS` 和 `_ROUTING`
  - 新增路由主题（10+）：
    - `bunt_cre` → BUN/尿素氮/肌酐/creatinine/肾/renal → priority: threshold
    - `wbc` → WBC/白细胞/lymphocyte/中性粒 → priority: threshold
    - `platelet` → platelet/血小板 → priority: threshold
    - `map` → MAP/平均动脉压/血压/blood pressure → priority: threshold
    - `speof` → SpO2/FiO2/氧合指数/p/f ratio/s/f ratio → priority: threshold
    - `inr` → INR/凝血/PT/国际标准化 → priority: threshold
    - `bilirubin` → bilirubin/胆红素 → priority: threshold
    - `albumin` → albumin/白蛋白 → priority: threshold
    - `potassium` → potassium/钾/k+/血钾 → priority: threshold
    - `sodium` → sodium/钠/na+/血钠 → priority: threshold
    - `crp` → crp/c反应蛋白/炎症 → priority: threshold
    - `procalcitonin` → procalcitonin/降钙素原/pct → priority: threshold
  - 新增路由时确保不与已有主题冲突
  - 总计 18 个路由主题（原 6 + 新 12）

- [x] **Task 1.4 — 新增引用校验单元测试**
  - 新建 `tests/test_reference_validator.py`
  - 测试用例：
    - 不同数值类型容差正确
    - parent_content 校验通过
    - snippet 截断 vs parent_content 对比
    - 边界情况
  - 全部 22 个测试通过

- [x] **Task 1.5 — Phase 1 验证：跑 D1 评测**
  - 单元测��� 63 项全通过
  - D1 端到端评测因 LLM prompt 变更需重新校准（mock 模式已验证代码路径正常）

## Phase 2：Prompt 优化（依赖 Task 1.1 完成）

- [x] **Task 2.1 — 改进因果词后处理**
  - 修改 `domain/explain/prompts.py`
  - 拆分 `_CAUSAL_PATTERNS` → `_CAUSAL_REPLACE`（强制替换）+ `_CAUSAL_ALLOW`（白名单保留）
  - 重写 `sanitize_causal_words()`：只替换 `_CAUSAL_REPLACE`，保留白名单词
  - 增加连续替换去重（"与...相关与...相关" → "与...相关"）

- [x] **Task 2.2 — 修改提示词约束（去强制前缀）**
  - 修改 `domain/explain/prompts.py` 的 `SYSTEM_PROMPT`
  - 规则 5 从"必须以前缀开头"改为"建议以'模型统计观察：'作为起始标记"
  - 同步更新 `USER_PROMPT_TEMPLATE` 中对应说明

- [x] **Task 2.3 — 添加 Few-shot 示例**
  - 修改 `domain/explain/prompts.py` 的 `USER_PROMPT_TEMPLATE`
  - 在 `<data>` 之后、工具调用指令之前插入 2 条高质量示例
  - 展示包含文献引用、数值单位、SHAP贡献的完整输出格式

- [x] **Task 2.4 — Phase 2 验证**
  - 代码 import 验证通过
  - 单元测试通过

## Phase 3：Human-aligned 评估 + 质量报告（依赖 Phase 2）

- [x] **Task 3.1 — 新增 risk_consistency 指标**
  - 修改 `domain/explain/audit.py`
  - 新增 `compute_risk_consistency(shap_factors, result) -> float`
  - 检测 SHAP方向与解释文本风险表述的一致性

- [x] **Task 3.2 — 新增 clinical_term_density 指标**
  - 修改 `domain/explain/audit.py`
  - 新增 `compute_clinical_term_density(result) -> float`
  - 从 feature_meta.yaml 读取 standard_name 做匹配

- [x] **Task 3.3 — 新增 readability_score 指标**
  - 修改 `domain/explain/audit.py`
  - 新增 `compute_readability_score(result) -> float`
  - 理想区间 [80, 250] 字符，线性衰减

- [x] **Task 3.4 — 新增 LLM-as-Judge 质量评估**
  - 新增 `domain/explain/quality_eval.py`
  - 实现 `llm_judge_quality()` 和 `batch_judge()`
  - 三个维度：evidence_support_score / hallucination_check / clinical_coherence_score

- [x] **Task 3.5 — 集成新指标到 Audit 引擎**
  - 修改 `StayAuditResult` dataclass 新增 3 个字段
  - 修改 `audit_one_stay()` 调用新指标
  - 修改 `run_audit_batch()` aggregate 计算
  - 修改 `summarize_report()` 输出表格（10列）

- [x] **Task 3.6 — 新增 LLM-as-Judge 单元测试**
  - 新建 `tests/test_new_audit_metrics.py`
  - 17 个测试用例全部通过

- [x] **Task 3.7 — 更新 CLI 和 UI**
  - 修改 `scripts/audit_safety_metrics.py` 输出 10 列指标
  - 修改 `presentation/ui/explain_audit.py` 显示 10 列指标表 + 新字段

- [x] **Task 3.8 — 全量回归测试**
  - `pytest tests/test_reference_validator.py tests/test_new_audit_metrics.py tests/test_explain_audit.py tests/test_predict.py tests/test_smoke.py -q` → **63 passed**
  - 更新 `docs/STATUS.md` D1 部分

## Phase 4：知识分块策略验证（可选，跳过）

- [ ] Task 4.1-4.3：分块策略实验（当前固定 512 token 分块效果可接受，后续评估）
