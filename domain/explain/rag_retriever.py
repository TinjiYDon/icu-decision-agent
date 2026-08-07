"""RAG 检索：FAISS 向量检索 + 元数据过滤 + 查询路由。

特性：
    - 延迟加载 embedding 模型与 FAISS 索引（首次调用才加载）
    - 元数据过滤：按 category（term/guideline/threshold/literature）过滤
    - 查询路由（PR-16）：根据查询意图决定优先召回的文档类别
    - 父子 chunk：检索子 chunk，返回父 chunk 完整内容
    - 低置信度过滤：min_score 以下丢弃
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from domain.explain.config import get_config
from domain.explain.knowledge_builder import Chunk


@dataclass
class RetrievalHit:
    """单条检索结果。"""
    chunk_id: str
    parent_id: str
    source: str
    category: str
    scope: str
    title: str
    content: str           # 子 chunk 内容（命中片段）
    parent_content: str    # 父 chunk 完整内容（给 LLM）
    score: float
    is_table: bool
    evidence_level: str


# ---------- 查询意图分类（PR-16 路由） ----------

# 粗粒度主题映射：关键词 → 优先 category
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "sofa": ["sofa", "序贯器官", "器官衰竭", "器官功能"],
    "ards": ["ards", "氧合", "p/f", "s/f", "spo2/fio2", "急性呼吸窘迫"],
    "shock_index": ["休克指数", "shock index", "si"],
    "lactate": ["乳酸", "lactate"],
    "gcs": ["gcs", "glasgow", "昏迷", "意识"],
    "sepsis": ["脓毒症", "sepsis", "septic"],
}

# 路由规则：主题 → (优先 category, 降权 categories)
# 设计意图：阈值类查询优先召回 feature_thresholds.md（单一事实源，PR-03）
_ROUTING: dict[str, tuple[str, list[str]]] = {
    "sofa": ("threshold", ["term", "guideline"]),
    "ards": ("guideline", ["term", "threshold"]),
    "shock_index": ("threshold", ["term", "guideline"]),
    "lactate": ("threshold", ["guideline", "term"]),
    "gcs": ("term", ["threshold", "guideline"]),
    "sepsis": ("guideline", ["term", "threshold"]),
}

# 文献类默认降权（PR-08）
_LITERATURE_DOWNGRADE = 0.5


def classify_query(query: str) -> str | None:
    """识别查询主题，返回主题 key（如 'sofa'），未命中返回 None。"""
    q = query.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return topic
    return None


def _get_routing_weights(topic: str | None) -> dict[str, float]:
    """根据主题返回 category 权重。"""
    if topic is None or topic not in _ROUTING:
        return {"term": 1.0, "guideline": 1.0, "threshold": 1.0, "literature": _LITERATURE_DOWNGRADE}
    primary, downgrade = _ROUTING[topic]
    weights = {primary: 1.5}
    for cat in downgrade:
        weights[cat] = 0.7
    weights["literature"] = _LITERATURE_DOWNGRADE
    # 兜底：未列出的 category 不降权
    for cat in ("term", "guideline", "threshold"):
        weights.setdefault(cat, 1.0)
    return weights


# ---------- 索引与模型加载（延迟） ----------

@dataclass
class _LoadedIndex:
    index: Any              # faiss.Index
    chunks: list[Chunk]
    model: Any              # SentenceTransformer
    dim: int


@lru_cache(maxsize=1)
def _load_index() -> _LoadedIndex | None:
    """延迟加载 FAISS 索引 + chunk 元数据 + embedding 模型。"""
    cfg = get_config()
    if not cfg.rag.index_path.exists():
        return None
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            f"缺少依赖：{e.name}。请运行 `pip install sentence-transformers faiss-cpu`"
        ) from e

    # Windows 中文路径下 faiss.read_index 会失败，改用 Python 读取字节再反序列化
    with open(cfg.rag.index_path, "rb") as f:
        index_bytes = f.read()
    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
    chunks_json_path = cfg.rag.chunks_dir / "chunks.json"
    if not chunks_json_path.exists():
        return None
    with chunks_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Chunk(**c) for c in data["chunks"]]
    model = SentenceTransformer(cfg.rag.embedding_model)
    return _LoadedIndex(index=index, chunks=chunks, model=model, dim=index.d)


def is_available() -> bool:
    """知识库是否可用（索引文件存在且可加载）。"""
    cfg = get_config()
    return cfg.rag.index_path.exists() and (cfg.rag.chunks_dir / "chunks.json").exists()


# ---------- 检索 ----------

def retrieve_context(
    query: str,
    top_k: int | None = None,
    category_filter: str | None = None,
    min_score: float | None = None,
) -> list[RetrievalHit]:
    """从 FAISS 检索与查询最相关的医学知识片段。

    Args:
        query: 查询文本（通常是特征名 + 异常值描述）
        top_k: 返回数量，默认用 config
        category_filter: 可选，按 category 过滤（term/guideline/threshold/literature）
        min_score: 最低相似度阈值，默认用 config

    Returns:
        list[RetrievalHit]，按分数降序
    """
    cfg = get_config()
    top_k = top_k or cfg.rag.top_k
    min_score = min_score if min_score is not None else cfg.rag.min_score

    loaded = _load_index()
    if loaded is None:
        return []

    # 1. 查询 embedding
    q_emb = loaded.model.encode([query], normalize_embeddings=True)
    q_arr = np.asarray(q_emb, dtype=np.float32)

    # 2. 向量检索（多取一些用于过滤后仍够 top_k）
    fetch_k = min(top_k * 4, len(loaded.chunks))
    scores, indices = loaded.index.search(q_arr, fetch_k)

    # 3. 路由权重
    topic = classify_query(query)
    weights = _get_routing_weights(topic)

    # 4. 组装 + 过滤 + 加权
    hits: list[RetrievalHit] = []
    seen_parents: set[str] = set()
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(loaded.chunks):
            continue
        c = loaded.chunks[idx]
        raw_score = float(scores[0][i])
        if raw_score < min_score:
            continue
        if category_filter and c.category != category_filter:
            continue
        # 父 chunk 去重：同一父 chunk 只保留得分最高的子 chunk
        if c.parent_id in seen_parents:
            continue
        seen_parents.add(c.parent_id)
        # 路由加权
        weighted = raw_score * weights.get(c.category, 1.0)
        hits.append(RetrievalHit(
            chunk_id=c.chunk_id,
            parent_id=c.parent_id,
            source=c.source,
            category=c.category,
            scope=c.scope,
            title=c.title,
            content=c.content,
            parent_content=c.parent_content,
            score=weighted,
            is_table=c.is_table,
            evidence_level=c.evidence_level,
        ))
        if len(hits) >= top_k:
            break

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def retrieve_for_feature(feature_name: str, feature_value: Any, standard_name: str) -> list[RetrievalHit]:
    """针对单个特征的检索：用特征标准名 + 实测值构造查询。

    Args:
        feature_name: 模型特征键名（如 lab_lactate）
        feature_value: 实测值
        standard_name: 标准医学名称（来自 feature_meta.yaml，如 Lactate（乳酸））

    Returns:
        list[RetrievalHit]
    """
    parts = [standard_name, feature_name]
    if feature_value is not None:
        parts.append(f"实测值 {feature_value}")
    query = " ".join(parts)
    return retrieve_context(query, top_k=2)


def format_hits_for_prompt(hits: list[RetrievalHit]) -> tuple[str, list[dict[str, Any]]]:
    """将检索结果格式化为 Prompt 片段 + 引用列表。

    Returns:
        (context_text, references)
        context_text: 用于注入 Prompt 的医学参考片段
        references: 结构化引用列表
    """
    if not hits:
        return "", []
    blocks: list[str] = []
    refs: list[dict[str, Any]] = []
    for i, h in enumerate(hits, 1):
        ref_id = f"REF-{i:02d}"
        blocks.append(
            f"[{ref_id}] 来源：{h.source} | 类别：{h.category} | 标题：{h.title}\n"
            f"{h.parent_content}"
        )
        refs.append({
            "id": ref_id,
            "chunk_id": h.chunk_id,
            "source": h.source,
            "title": h.title,
            "snippet": h.content[:200],
            "score": round(h.score, 4),
            "category": h.category,
            "evidence_level": h.evidence_level,
        })
    return "\n\n".join(blocks), refs
