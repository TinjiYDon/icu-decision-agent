# LLM 解释质量提升计划

> 基于代码审计 + D1 50 stays 评测结果（引用有效率 44.9%、特征覆盖 18.5%、事实 grounding 82.6%）

---

## 一、现状诊断（来自代码审计）

### 1. 引用校验层（`reference_validator.py`）

| 问题 | 位置 | 影响 |
|------|------|------|
| 容差为绝对值 ±0.05 | 第70行 `abs(ev - rv) <= tolerance` | 对 BUN=45、Cr=1.2、乳酸=2.3 等临床数值过于严格，正常范围描述性数值（如"2.5-7.1"）无法匹配 |
| 校验用 snippet[:200] | `rag_retriever.py` 第254行 `snippet: h.content[:200]` | snippet 是子chunk截断，丢失父chunk上下文；而 `parent_content` 完整内容才是有信息量的 |
| 没有单位感知 | `extract_numbers` 只提取数字 | "2.3 mmol/L" 和 "2.3 mg/dL" 被当作相同数值匹配，导致假阳性 |

### 2. RAG 检索层（`rag_retriever.py`）

| 问题 | 位置 | 影响 |
|------|------|------|
| 路由仅覆盖6个主题 | `_TOPIC_KEYWORDS` 第44-51行 | BUN、肌酐、WBC、PLT、MAP、SpO2/FiO2、INR、胆红素等高频特征无路由，走通用检索 → 相关性低 |
| 路由权重设计不对称 | `_ROUTING` 第55-62行 | sofa→threshold，但 creatinine→无路由；lactate→threshold，但 WBC→无路由 |

### 3. Prompt 层（`prompts.py`）

| 问题 | 位置 | 影响 |
|------|------|------|
| 无 few-shot 示例 | `USER_PROMPT_TEMPLATE` 第31-57行 | LLM 输出格式不稳定，有时偏离"模型统计观察"前缀要求 |
| 约束过死板 | SYSTEM_PROMPT 第5条"必须以前缀开头" | 部分因子解释僵硬，LLM被迫每句都加前缀，可读性下降 |

### 4. 后处理层（`prompts.py`）

| 问题 | 位置 | 影响 |
|------|------|------|
| 因果词全局替换 | `sanitize_causal_words` 第99-104行 | "导致"→"与...相关"，"引起"→"与...相关"，全部统一替换，丢失医学表述的自然性 |
| 无白名单例外 | 无 | "提示""关联""多见于"等医学惯用词也被替换 |

### 5. 评估层（`audit.py`）

| 问题 | 位置 | 影响 |
|------|------|------|
| 缺少可读性指标 | 无 | 无法判断解释"是否好读" |
| 缺少临床术语覆盖率 | 无 | 无法判断解释是否使用了足够医学术语 |
| 缺少风险一致性检查 | 无 | 无法发现"高风险因子却写降低风险"的矛盾 |

---

## 二、改进方案（分三阶段）

### Phase 1：快速修复（预计改动约60行，效果立竿见影）

#### 1.1 修复引用校验容差（`reference_validator.py`）

**改动：** 将绝对容差改为**分段容差+相对容差混合策略**

```
当前：abs(ev - rv) <= 0.05  （对所有数值统一容差）
改为：
  - 比值型（SOFA分值、休克指数）：绝对容差 ±0.5
  - 浓度/量值型（乳酸、肌酐、BUN）：相对容差 ±15% 或 绝对容差 ±0.5（取较大者）
  - 百分比型（SpO2、GCS分项）：绝对容差 ±2
  - 计数型（WBC、PLT）：相对容差 ±20%
```

**新增函数：** `_smart_tolerance(expected_val: float, reference_val: float) -> float`

#### 1.2 校验改用 parent_content（`reference_validator.py`）

**改动：** `validate_factor_references` 中从 `ref.get("snippet")` 改为 `ref.get("parent_content", ref.get("snippet", ""))`

同时修改 `rag_retriever.py` 的 `format_hits_for_prompt`，在 references dict 中补充 `parent_content` 字段（UI仍展示snippet）。

**风险：** 引用列表数据结构扩展，需检查 UI 侧是否有依赖 `snippet` 字段的地方。

#### 1.3 扩展 RAG 路由覆盖（`rag_retriever.py`）

**改动：** 扩充 `_TOPIC_KEYWORDS` 和 `_ROUTING`

新增覆盖主题（按特征频率排序）：
| 主题 | 关键词 | 优先 category |
|------|--------|--------------|
| bunt_cre | BUN/尿素氮/肌酐/creatinine/肾 | threshold |
| wbc | WBC/白细胞/lymphocyte | threshold |
| platelet | platelet/血小板 | threshold |
| map | MAP/平均动脉压/血压 | threshold |
| speof | SpO2/FiO2/氧合指数/p/f ratio | threshold |
| inr | INR/凝血/PT | threshold |
| bilirubin | bilirubin/胆红素 | threshold |
| albumin | albumin/白蛋白 | threshold |

**风险：** 与已有主题（lactate、sofa、ards）不冲突；threshold 类别在 v3 知识库中有 `feature_thresholds.md` 提供统一阈值来源。

---

### Phase 2：质量优化（预计改动约80行，效果显著）

#### 2.1 Prompt 增加 few-shot 示例（`prompts.py`）

**改动：** 在 `USER_PROMPT_TEMPLATE` 的 `<data>` 之后、工具调用指令之前，插入2个高质量示例。

示例来源：从现有D模式评测结果中选取2个**引用有效率=1.0、过度承诺率=0.0**的高质量解释，作为示范。

示例结构：
```
### 示例 1（高乳酸血症）
<example>
summary: "入ICU后6小时预测未来12小时死亡风险为 68.2%（高风险）。"
factor_analysis:
  - feature: lab_lactate
    clinical_interpretation: "模型统计观察：乳酸实测值 4.2 mmol/L（高于危急值 4.0 mmol/L），与近期研究[REF-02]中报道的脓毒症休克患者乳酸升高模式一致，SHAP贡献 0.034（升高风险）。"
    reference_id: "REF-02"
  - feature: vital_sofa_resp
    clinical_interpretation: "模型统计观察：SOFA呼吸分项为3分，SHAP贡献 0.021（升高风险）。根据序贯器官衰竭评分标准[REF-01]，该分值提示中度呼吸功能异常。"
    reference_id: "REF-01"
</example>
```

**风险：** 示例内容需定期更新，否则可能与当前知识库脱节。建议在 `knowledge_base/raw/` 下维护一个 `fewshot_examples.md`，由脚本从高质量解释中定期生成。

#### 2.2 改进因果词后处理（`prompts.py`）

**改动：** 区分**危险因果词**和**允许关联词**

```
当前替换规则：
  导致/引起/诱发/造成/引发/致 → 与...相关   （一刀切）

改为：
  危险词（仍需替换）：导致、引起、诱发、造成、引发、致、使得
  允许词（保留）：提示、关联、多见于、常见于、伴随、相关于、见于

  规则：
  1. 危险词统一替换为"与...相关"
  2. 允许词保持不动
  3. 若替换后语句不通顺，保留原词（避免破坏表达）
```

**实现方式：**
- `_CAUSAL_PATTERNS` 拆分为 `_CAUSAL_REPLACE`（危险词）和 `_CAUSAL_ALLOW`（白名单）
- `sanitize_causal_words` 增加白名单跳过逻辑
- 新增 `_check_fluency_after_replace(text)` 函数，检测替换后是否产生明显不通顺的表述

---

### Phase 3：评估增强（预计改动约100行，长期价值）

#### 3.1 新增可读性指标（`audit.py`）

**指标：** `readability_score`
- 定义：因子平均解释长度（字符数），落在 [80, 200] 范围内得1分，之外线性衰减
- 原理：太短（<80字）= 信息不足；太长（>200字）= 信息密度低

**指标：** `clinical_term_density`
- 定义：因子解释中出现医学术语的比例（术语表来自 `feature_meta.yaml` 的 standard_name）
- 原理：临床医生更倾向于使用标准术语而非口语表述

#### 3.2 新增风险一致性检查（`audit.py`）

**指标：** `risk_consistency`
- 逻辑：
  1. 对每个因子，判断其 SHAP方向（升高风险/降低风险）
  2. 检查 factor_analysis 中的 clinical_interpretation 是否与该方向一致
  3. 不一致则扣分
- 示例：SHAP方向为正（升高风险）但解释写"该指标降低风险"→ 不一致

#### 3.3 整合到 UI 和报告

- `explain_audit.py` 新增两列指标
- `summarize_report` 新增结论段落

---

## 三、实施顺序与依赖

```
Phase 1.1 (容差修复) ──┐
Phase 1.2 (parent_content) ─┼──→  Phase 1 完成 → 跑 audit 验证引用有效率提升
Phase 1.3 (路由扩展)   ──┘

Phase 2.1 (few-shot) ──┐
Phase 2.2 (因果词优化) ─┴──→  Phase 2 完成 → 跑 audit 验证 readability + risk_consistency

Phase 3 (评估增强) ──────────────────────────→  Phase 3 完成 → 完整报告更新
```

**Phase 1 各子项互相独立，可并行实施。**

---

## 四、预期效果

| 指标 | 当前(D模式) | Phase1后 | Phase2后 |
|------|------------|---------|---------|
| 引用有效率 | 44.9% | 70-75% | 75-80% |
| 特征覆盖密度 | 18.5% | 18.5%（不变） | 22-25% |
| 事实grounding率 | 82.6% | 82.6%（不变） | 85-88% |
| 证据具体度 | — | — | 显著提升 |
| 可读性分数 | 未测量 | 未测量 | ~0.85 |
| 风险一致性 | 未测量 | 未测量 | 预计>95% |

---

## 五、验证方式

每次 Phase 完成后，运行：
```powershell
.\.venv\Scripts\python.exe scripts\audit_safety_metrics.py --limit 50
```
对比 Phase 前后的指标变化，确认有改善后再进入下一阶段。

---

## 六、风险与回退

| 风险 | 缓解措施 | 回退方式 |
|------|---------|---------|
| parent_content 导致引用过长 | 限制 parent_content 最大长度（如2000字符） | 回退到 snippet |
| 路由扩展引入噪声 | 先只加高置信度路由，低置信度逐步放开 | 注释掉新增路由 |
| few-shot 示例过时 | 示例存为独立文件，可定期更新 | 删除示例块 |
| 因果词白名单放行不当表述 | 保留强制替换兜底逻辑 | 恢复原全局替换 |
