# D1 工程化落地 · Checklist

## Phase 1：引用校验修复 + 路由扩展

- [ ] 引用校验容差函数 `_smart_tolerance()` 实现正确（4 种数值类型分段逻辑）
- [ ] `validate_reference()` 使用动态容差而非固定 ±0.05
- [ ] `validate_factor_references()` 改用 `parent_content` 做校验
- [ ] `format_hits_for_prompt()` 在 references dict 中补充 `parent_content` 字段
- [ ] UI 侧（`explain_audit.py`）未因 parent_content 字段变更而报错
- [ ] RAG 路由新增 10+ 主题，不与已有主题冲突
- [ ] `retrieve_for_feature()` 对数值型特征附加单位信息
- [ ] 引用校验单元测试全部通过（test_reference_validator.py）
- [ ] Phase 1 后 D1 评测：引用有效率 ≥ 70%
- [ ] STATUS.md D1 部分更新为 Phase 1 基线

## Phase 2：Prompt 优化

- [ ] `_CAUSAL_REPLACE` 和 `_CAUSAL_ALLOW` 正确拆分
- [ ] `sanitize_causal_words()` 只替换强制列表，保留白名单词
- [ ] Fluency check 逻辑不产生死循环或性能问题
- [ ] SYSTEM_PROMPT 中"必须前缀"改为"建议前缀"
- [ ] USER_PROMPT_TEMPLATE 中包含 2 条高质量 few-shot 示例
- [ ] Phase 2 后 D1 评测：过度承诺率不回升（≤ 10%），引用有效率稳定 ≥ 70%
- [ ] 抽样 5 条输出人工抽查因果词替换合理性

## Phase 3：Human-aligned 评估

- [ ] `compute_risk_consistency()` 正确检测 SHAP 方向与解释文本的一致性
- [ ] `compute_clinical_term_density()` 正确从 feature_meta.yaml 读取标准名
- [ ] `compute_readability_score()` 区间 [80, 250] 逻辑正确
- [ ] `quality_eval.py` 中 LLM-as-Judge 3 个维度均可正常调用
- [ ] LLM-as-Judge 单次调用 timeout=30s，不阻塞主流程
- [ ] Audit StayAuditResult 新增 3 个字段，aggregate 计算正确
- [ ] summarize_report() 输出表格包含新 3 列
- [ ] test_quality_eval.py 全部通过
- [ ] generate_explain_report.py 可生成完整 Markdown 报告
- [ ] Phase 3 完整评测：risk_consistency ≥ 90%，readability ≥ 0.80，hallucination_rate ≤ 5%

## Phase 4：分块策略验证（可选）

- [ ] semantic_chunk() 函数实现正确
- [ ] 新旧索引可独立存在，互不影响
- [ ] 检索命中率对比实验完成并记录

## 全量回归

- [ ] `.\.venv\Scripts\python.exe -m pytest tests/test_explain_audit.py tests/test_reference_validator.py tests/test_quality_eval.py tests/test_chunk_strategy.py -q` 全通过
- [ ] `.\.venv\Scripts\python.exe -m pytest tests/test_predict.py tests/test_smoke.py -q` 不受影响
- [ ] Streamlit 解释页面可正常渲染新字段
- [ ] MCP explain_risk 工具不受影响
