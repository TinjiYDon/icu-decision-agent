"""构建 RAG 知识库索引的命令行脚本。

用法：
    python -m scripts.build_knowledge_index
    或
    python scripts/build_knowledge_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python scripts/build_knowledge_index.py 运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.knowledge_builder import build_index


def main() -> int:
    print("=" * 60)
    print("构建 RAG 知识库索引（m3e-base + FAISS）")
    print("=" * 60)
    try:
        result = build_index(verbose=True)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到：{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[错误] 构建失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 60)
    print("构建完成")
    print("=" * 60)
    print(f"  文档数:    {result.doc_count}")
    print(f"  父chunk数: {result.parent_count}")
    print(f"  子chunk数: {result.chunk_count}")
    print(f"  模型:      {result.embedding_model}")
    print(f"  构建时间:  {result.build_time}")
    print(f"  索引文件:  {result.index_path}")
    print(f"  chunks:    {result.chunks_path}")
    print(f"  版本文件:  {result.version_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
