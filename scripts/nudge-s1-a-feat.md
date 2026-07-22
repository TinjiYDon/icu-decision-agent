## 催促 · 成员 A · #S1-2 feat/label 对齐

### 人读摘要

| 项 | 内容 |
|----|------|
| Owner | A 数据 |
| 任务 | 确认 `feat.sample_matrix` / `label.mortality_12h` 与 B 训练对齐 |
| 额外 | 提供**含数据**的 Layer1 dump（当前 schemas_only 不足） |

### Agent 上下文

```text
验证：pytest tests/test_etl.py -q（若有）
文档：docs/PARAM_STORY.md
注意：hospital_expire_flag 泄漏风险，与 B 讨论是否移出特征
验收：STATUS 注明 dump 是否含表数据
```
