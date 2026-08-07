"""知识库构建：解析 Markdown → 父子 chunk → m3e-base embedding → FAISS 索引。

父子 Chunk 设计（PR-05/PR-11 落地）：
    - 父 chunk：一个 Markdown 段落（### 子标题下的完整内容，含表格）
    - 子 chunk：父 chunk 内按字符切分的片段（用于精准检索）
    - 检索时：用子 chunk 向量命中，返回完整父 chunk 给 LLM（语义完整）

表格保护（PR-05）：
    - Markdown 表格（|...| 行）整体作为一个子 chunk，不拆分
    - 上限 MAX_TABLE_CHUNK=2048 字符；超过按行拆分

知识点 ID（PR-04）：
    - 每个父 chunk 分配 ID：{TYPE}-{KEY}-{NUM}（如 TERM-LACTATE-001）
    - 每个 chunk 附带元数据：category, source, scope, evidence_level
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.explain.config import KB_DIR, get_config


@dataclass
class Chunk:
    """单个 chunk（子或父）。"""
    chunk_id: str          # 全局唯一 ID，如 "TERM-LACTATE-001-sub-2"
    parent_id: str         # 父 chunk ID
    source: str            # 来源文件名（如 terms_medical.md）
    category: str          # term | guideline | threshold | literature
    scope: str             # 适用人群，如 "通用" / "脓毒症患者"
    title: str             # 父 chunk 的标题（如 "lab_lactate（乳酸）"）
    content: str           # 子 chunk 内容（用于 embedding）
    parent_content: str    # 父 chunk 完整内容（命中后给 LLM）
    is_table: bool = False
    evidence_level: str = "B"  # A/B/C
    seq: int = 0           # 在父 chunk 内的序号


@dataclass
class BuildResult:
    """索引构建结果。"""
    index_path: Path
    chunks_path: Path
    version_path: Path
    doc_count: int
    chunk_count: int
    parent_count: int
    build_time: str
    embedding_model: str


# ---------- Markdown 解析 ----------

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_HEADING3 = re.compile(r"^###\s+(.+)$")
_HEADING2 = re.compile(r"^##\s+(.+)$")


def _classify_source(filename: str) -> str:
    """根据文件名推断 category。"""
    name = filename.lower()
    if "threshold" in name:
        return "threshold"
    if "guideline" in name:
        return "guideline"
    if "literature" in name:
        return "literature"
    return "term"


def _extract_scope(text: str) -> str:
    """从文本中提取适用范围标记。"""
    m = re.search(r"【适用[：:]\s*([^】]+)】", text)
    if m:
        return m.group(1).strip()
    return "通用"


def _split_paragraphs(md: str) -> list[dict[str, str]]:
    """按 ### 子标题切分父 chunk；每个父 chunk 含 title + content。

    返回：[{"title": str, "content": str}, ...]
    一个父 chunk 是一个 ### 段落（含其下所有内容，直到下一个 ### 或 ##）。
    """
    lines = md.splitlines()
    paragraphs: list[dict[str, str]] = []
    cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_lines and any(ln.strip() for ln in cur_lines):
            content = "\n".join(cur_lines).strip()
            if content:
                paragraphs.append({"title": cur_title, "content": content})

    for ln in lines:
        h3 = _HEADING3.match(ln)
        h2 = _HEADING2.match(ln)
        if h3 or h2:
            flush()
            cur_title = (h3 or h2).group(1).strip()
            cur_lines = []
        else:
            cur_lines.append(ln)
    flush()
    return paragraphs


def _extract_tables_and_text(content: str) -> list[tuple[str, bool]]:
    """将父 chunk 内容拆分为 [(片段, 是否表格), ...]。

    表格行连续块整体保留为一个子 chunk；其余按段落切分。
    """
    pieces: list[tuple[str, bool]] = []
    lines = content.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if _TABLE_ROW.match(ln) or _TABLE_SEP.match(ln):
            # 收集连续表格行
            tbl: list[str] = []
            while i < n and (_TABLE_ROW.match(lines[i]) or _TABLE_SEP.match(lines[i])):
                tbl.append(lines[i])
                i += 1
            table_text = "\n".join(tbl).strip()
            if table_text:
                pieces.append((table_text, True))
        else:
            # 收集非表格行直到下一个表格
            txt: list[str] = []
            while i < n and not (_TABLE_ROW.match(lines[i]) or _TABLE_SEP.match(lines[i])):
                txt.append(lines[i])
                i += 1
            text_block = "\n".join(txt).strip()
            if text_block:
                pieces.append((text_block, False))
    return pieces


def _split_text_piece(text: str, chunk_size: int, overlap: int, min_size: int) -> list[str]:
    """按字符切分文本片段（保留句子边界近似）。"""
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_size else []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if len(piece) >= min_size:
            pieces.append(piece)
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return pieces


def _make_chunk_id(category: str, title: str, seq: int) -> str:
    """生成 chunk ID：{TYPE}-{KEY}-{NUM}。"""
    type_code = {"term": "TERM", "guideline": "GUIDE", "threshold": "THRESH", "literature": "LIT"}.get(category, "MISC")
    key = re.sub(r"[^A-Za-z0-9]", "-", title).strip("-").upper()[:24] or "GENERIC"
    return f"{type_code}-{key}-{seq:03d}"


def build_chunks_from_doc(md_path: Path, cfg) -> list[Chunk]:
    """从单个 Markdown 文件构建 chunk 列表。"""
    category = _classify_source(md_path.name)
    md = md_path.read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(md)
    chunks: list[Chunk] = []

    for para in paragraphs:
        title = para["title"]
        content = para["content"]
        scope = _extract_scope(content)
        # 子 chunk 切分
        pieces = _extract_tables_and_text(content)
        sub_seq = 0
        parent_id = ""
        for text, is_table in pieces:
            if is_table:
                # 表格整体保留，若超过 MAX_TABLE_CHUNK 按行拆
                if len(text) <= cfg.rag.max_table_chunk:
                    sub_pieces = [text]
                else:
                    # 按行拆，每行不超过 chunk_size
                    tbl_lines = text.splitlines()
                    sub_pieces = []
                    cur: list[str] = []
                    cur_len = 0
                    for tl in tbl_lines:
                        if cur_len + len(tl) + 1 > cfg.rag.chunk_size and cur:
                            sub_pieces.append("\n".join(cur))
                            cur = [tl]
                            cur_len = len(tl)
                        else:
                            cur.append(tl)
                            cur_len += len(tl) + 1
                    if cur:
                        sub_pieces.append("\n".join(cur))
            else:
                sub_pieces = _split_text_piece(text, cfg.rag.chunk_size, cfg.rag.chunk_overlap, cfg.rag.min_chunk_size)

            for sp in sub_pieces:
                if not sp.strip():
                    continue
                if not parent_id:
                    parent_id = _make_chunk_id(category, title, len(chunks))
                cid = f"{parent_id}-sub-{sub_seq}"
                chunks.append(Chunk(
                    chunk_id=cid,
                    parent_id=parent_id,
                    source=md_path.name,
                    category=category,
                    scope=scope,
                    title=title,
                    content=sp,
                    parent_content=content,
                    is_table=is_table,
                    evidence_level="A" if category in ("threshold", "guideline") else "B",
                    seq=sub_seq,
                ))
                sub_seq += 1
    return chunks


def build_index(verbose: bool = True) -> BuildResult:
    """构建 FAISS 索引。从 knowledge_base/raw/ 读取所有 .md 文件。"""
    from sentence_transformers import SentenceTransformer
    import faiss

    cfg = get_config()
    raw_dir = cfg.rag.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"知识库原始文档目录不存在：{raw_dir}")

    md_files = sorted(raw_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"知识库 raw 目录下没有 .md 文件：{raw_dir}")

    # 1. 解析所有文档为 chunk
    all_chunks: list[Chunk] = []
    for md_path in md_files:
        chunks = build_chunks_from_doc(md_path, cfg)
        all_chunks.extend(chunks)
        if verbose:
            print(f"  [解析] {md_path.name}: {len(chunks)} chunks")

    if not all_chunks:
        raise ValueError("解析后无可用 chunk，请检查 raw/ 文档内容")

    # 2. 加载 embedding 模型
    if verbose:
        print(f"  [加载] embedding 模型：{cfg.rag.embedding_model}")
    model = SentenceTransformer(cfg.rag.embedding_model)

    # 3. 生成 embedding
    texts = [c.content for c in all_chunks]
    if verbose:
        print(f"  [embedding] {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=verbose)
    emb_arr = np.asarray(embeddings, dtype=np.float32)

    # 4. 构建 FAISS 索引（Inner Product，配合 normalize 后等价于 cosine）
    dim = emb_arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_arr)

    # 5. 持久化
    cfg.rag.index_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.rag.chunks_dir.mkdir(parents=True, exist_ok=True)
    # 注意：faiss.write_index 在 Windows + 中文路径下会失败（C++ FileIOWriter 限制），
    # 改用 faiss.serialize_index 得到字节，再用 Python 写文件（Python 对中文路径支持良好）。
    serialized = faiss.serialize_index(index)
    with open(cfg.rag.index_path, "wb") as f:
        f.write(serialized)

    # 6. chunk 元数据（不存 parent_content 全文到 manifest，但单独存 chunks.json）
    chunks_json_path = cfg.rag.chunks_dir / "chunks.json"
    chunks_data = {
        "chunks": [
            {
                **{k: v for k, v in asdict(c).items()},
            }
            for c in all_chunks
        ],
    }
    with chunks_json_path.open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    # 7. 索引版本文件
    version_path = cfg.rag.chunks_dir / "index_version.txt"
    parent_count = len({c.parent_id for c in all_chunks})
    build_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    version_text = (
        f"version: {time.strftime('%Y%m%d')}-v1\n"
        f"build_time: {build_time}\n"
        f"embedding_model: {cfg.rag.embedding_model}\n"
        f"chunk_count: {len(all_chunks)}\n"
        f"parent_count: {parent_count}\n"
        f"doc_count: {len(md_files)}\n"
        f"source_docs:\n"
    )
    for md in md_files:
        version_text += f"  - {md.name}\n"
    version_path.write_text(version_text, encoding="utf-8")

    # 8. chunk_manifest（精简版，供快速校验）
    manifest_path = cfg.rag.chunks_dir / "chunk_manifest.json"
    manifest = {
        "version": f"{time.strftime('%Y%m%d')}-v1",
        "build_time": build_time,
        "doc_count": len(md_files),
        "chunk_count": len(all_chunks),
        "parent_count": parent_count,
        "embedding_model": cfg.rag.embedding_model,
        "embedding_dim": dim,
        "sources": [md.name for md in md_files],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  [完成] 索引写入：{cfg.rag.index_path}")
        print(f"  [完成] chunks 写入：{chunks_json_path}")
        print(f"  [完成] 版本文件：{version_path}")

    return BuildResult(
        index_path=cfg.rag.index_path,
        chunks_path=chunks_json_path,
        version_path=version_path,
        doc_count=len(md_files),
        chunk_count=len(all_chunks),
        parent_count=parent_count,
        build_time=build_time,
        embedding_model=cfg.rag.embedding_model,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("构建 RAG 知识库索引")
    print("=" * 60)
    result = build_index()
    print(f"\n构建完成：{result.chunk_count} chunks / {result.parent_count} parents / {result.doc_count} docs")
