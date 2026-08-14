# 知识库构建与索引管理

> 用途：SHAP+LLM 解释层 RAG 知识库基础设施
> 版本：v0.1
> 更新日期：2026-08-06

---

## 1. 知识库文件结构

```
knowledge_base/
├── raw/                          # 原始文档（不进 Git，加入 .gitignore）
│   ├── terms_medical.md          # 医学术语定义与临床阈值
│   ├── clinical_guidelines.md    # 临床指南与评分标准
│   ├── feature_thresholds.md     # 特征阈值对照表（由 feature_meta.yaml 生成）
│   └── mimic_literature.md       # MIMIC-IV 相关发表论文摘要
├── chunks/                       # 切分后的 chunk 元数据
│   └── chunk_manifest.json       # {version, build_time, doc_count, chunk_count}
├── faiss_index/                  # FAISS 索引文件
│   └── medical_knowledge.faiss   # 向量索引
│   └── index_version.txt         # 索引版本信息
└── README.md                     # 知识库使用说明
```

---

## 2. Chunk 切分策略

```python
# 配置参数
CHUNK_SIZE = 512          # 默认 chunk 大小（字符数）
CHUNK_OVERLAP = 64        # chunk 间重叠字符数
MAX_TABLE_CHUNK = 2048    # 表格/评分标准最大字符数（可突破 512）
MIN_CHUNK_SIZE = 128      # 最小 chunk，小于此则丢弃

# 切分规则
1. 优先按段落/句子边界切分（使用 SentenceSplitter）
2. 表格、评分标准整体保留为一个 chunk，不受 512 字符限制，上限 2048
3. 超过 2048 字符的表格按行拆分，每行不超过 512 字符
4. 父子索引：子 chunk 用于向量检索；命中后返回完整父 chunk 给 LLM
```

---

## 3. 索引构建流程

```bash
# 构建索引
python scripts/build_knowledge_index.py

# 流程：
# 1. 读取 knowledge_base/raw/ 下所有 .md 文件
# 2. 按上述策略切分为 chunk
# 3. 使用 m3e-base 模型为每个 chunk 生成 embedding
# 4. 构建 FAISS 索引并保存到 knowledge_base/faiss_index/
# 5. 写入 index_version.txt（包含构建时间、chunk 总数、embedding 模型版本）
# 6. 写入 chunks/chunk_manifest.json（元数据）
```

---

## 4. 版本管理

```
索引版本文件格式（index_version.txt）：
---------------------------------------
version: 20260806-v1
build_time: 2026-08-06T15:30:00
embedding_model: moka-ai/m3e-base
chunk_count: 156
doc_count: 3
source_docs:
  - terms_medical.md (v1)
  - clinical_guidelines.md (v1)
  - feature_thresholds.md (v1)
---------------------------------------
```

每次解释输出中携带版本号，实现全链路可追溯：
```json
{
  "explanation_version": "20260806-v1",
  "rag_version": "20260806-v1",
  "model_version": "S2-full_full_94458stays_20260806"
}
```

---

## 5. 知识库更新

当以下内容变更时，需重建索引：
- 新增/修改 `knowledge_base/raw/` 下的文档
- 更新 `configs/feature_meta.yaml`（特征元数据变化）
- 更新 embedding 模型版本

更新流程：
```bash
# 1. 修改原始文档
# 2. 重建索引
python scripts/build_knowledge_index.py
# 3. 验证索引（可选）
python scripts/verify_index.py
# 4. 备份旧索引（可选）
mv knowledge_base/faiss_index/medical_knowledge.faiss \
   knowledge_base/faiss_index/medical_knowledge_faiss_backup_20260806.faiss
```

---

## 6. 检索元数据过滤

FAISS 本身不支持元数据过滤，需在检索层实现：

```python
def retrieve_with_filter(query_embedding, top_k=3, category=None):
    """
    带类别过滤的 RAG 检索。

    Args:
        query_embedding: 查询向量
        top_k: 返回数量
        category: 文档类别过滤（term|guideline|threshold|all）

    Returns:
        list of {"id": str, "content": str, "source": str, "score": float}
    """
    # 1. 如果指定了 category，先过滤 chunk IDs
    if category and category != "all":
        candidate_ids = get_ids_by_category(category)
    else:
        candidate_ids = None

    # 2. 在候选集合内做向量检索
    if candidate_ids:
        scores, indices = index.search(
            query_embedding.reshape(1, -1),
            k=top_k * 2  # 多取一些，过滤后再取 top_k
        )
        # 3. 过滤不在候选集合内的结果
        filtered = [
            {"id": ids[i], "score": float(scores[0][i])}
            for i in range(len(indices[0]))
            if ids[i] in candidate_ids
        ][:top_k]
    else:
        scores, indices = index.search(query_embedding.reshape(1, -1), k=top_k)
        filtered = [
            {"id": ids[i], "score": float(scores[0][i])}
            for i in range(len(indices[0]))
        ]

    # 4. 组装完整结果
    return [get_chunk_detail(r["id"]) for r in filtered]
```

---

## 7. 引用一致性校验

```python
def validate_reference(explanation_text, reference_chunk, min_similarity=0.5):
    """
    校验 LLM 解释是否与引用片段一致。

    方法 1：关键数值核对
    - 从 explanation_text 中提取所有数值（正则：[\d.]+）
    - 检查这些数值是否出现在 reference_chunk 中

    方法 2：语义相似度辅助校验
    - 计算 explanation 与 reference 的 embedding 相似度
    - 低于阈值则标记 reference_valid=false

    Returns:
        {"valid": bool, "matched_values": list, "similarity": float}
    """
```

---

## 8. 知识库内容来源说明

| 文档 | 来源 | 版权状态 |
|------|------|---------|
| terms_medical.md | PubMed 开放获取论文 + MIMIC-IV 特征词典 | 公开可用 |
| clinical_guidelines.md | NICE 指南（CC BY）+ SSC 指南 + ARDS 新定义 | 公开可用 |
| feature_thresholds.md | 项目自建（基于 feature_meta.yaml） | 项目自有 |
| mimic_literature.md | MIMIC-IV 相关发表论文摘要 | 公开可用 |

> 注：SNOMED CT、UMLS 等商业术语库未纳入本知识库，避免版权风险。

---

*本文档为 SHAP+LLM 解释层基础设施文档。*
