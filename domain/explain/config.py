"""解释层配置加载：configs/explain.yaml + configs/feature_meta.yaml + 环境变量。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
KB_DIR = ROOT / "knowledge_base"


def _load_dotenv() -> None:
    """加载 .env 文件到环境变量（若尚未加载）。

    优先使用 python-dotenv；不可用时手动解析。
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    # 若已有 AGNES_API_KEY 等环境变量则不覆盖
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # 手动解析 .env（简单 key=value 格式）
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# 模块加载时立即加载 .env
_load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "agnes"
    # 默认使用国内镜像端点（国际端点 apihub.agnes-ai.com 在国内被 DNS 污染不可达）
    model: str = "agnes-2.5-flash"
    base_url: str = "https://apihub.agnes-ai.cn/v1"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 60


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = "moka-ai/m3e-base"
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_table_chunk: int = 2048
    min_chunk_size: int = 128
    top_k: int = 3
    min_score: float = 0.3
    index_path: Path = KB_DIR / "faiss_index" / "medical_knowledge.faiss"
    reference_validation: bool = True
    raw_dir: Path = KB_DIR / "raw"
    chunks_dir: Path = KB_DIR / "chunks"


@dataclass(frozen=True)
class OutputConfig:
    include_disclaimers: bool = True
    max_factor_analysis: int = 4
    language: str = "zh-CN"


@dataclass(frozen=True)
class ExplainConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _load_explain_yaml() -> dict[str, Any]:
    path = CONFIG_DIR / "explain.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_config() -> ExplainConfig:
    """加载 explain.yaml 并合并环境变量（API Key 等）。"""
    raw = _load_explain_yaml()
    llm_raw = raw.get("llm", {})
    rag_raw = raw.get("rag", {})
    out_raw = raw.get("output", {})

    api_key = os.getenv("AGNES_API_KEY", "") or llm_raw.get("api_key", "")
    model = os.getenv("AGNES_MODEL", "") or llm_raw.get("model", "agnes-2.0-flash")
    base_url = os.getenv("AGNES_BASE_URL", "") or llm_raw.get("base_url", "https://apihub.agnes-ai.com/v1")

    index_path_str = rag_raw.get("index_path", "")
    index_path = Path(index_path_str) if index_path_str else (KB_DIR / "faiss_index" / "medical_knowledge.faiss")
    if not index_path.is_absolute():
        index_path = ROOT / index_path

    return ExplainConfig(
        llm=LLMConfig(
            provider=llm_raw.get("provider", "agnes"),
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=float(llm_raw.get("temperature", 0.1)),
            max_tokens=int(llm_raw.get("max_tokens", 2048)),
            timeout_seconds=int(llm_raw.get("timeout_seconds", 30)),
        ),
        rag=RAGConfig(
            embedding_model=rag_raw.get("embedding_model", "moka-ai/m3e-base"),
            chunk_size=int(rag_raw.get("chunk_size", 512)),
            chunk_overlap=int(rag_raw.get("chunk_overlap", 64)),
            max_table_chunk=int(rag_raw.get("max_table_chunk", 2048)),
            min_chunk_size=int(rag_raw.get("min_chunk_size", 128)),
            top_k=int(os.getenv("KB_TOP_K", "") or rag_raw.get("top_k", 3)),
            min_score=float(rag_raw.get("min_score", 0.3)),
            index_path=index_path,
            reference_validation=bool(rag_raw.get("reference_validation", True)),
            raw_dir=KB_DIR / "raw",
            chunks_dir=KB_DIR / "chunks",
        ),
        output=OutputConfig(
            include_disclaimers=bool(out_raw.get("include_disclaimers", True)),
            max_factor_analysis=int(out_raw.get("max_factor_analysis", 4)),
            language=out_raw.get("language", "zh-CN"),
        ),
    )


@lru_cache
def get_feature_meta() -> dict[str, Any]:
    """加载特征元数据（白名单 + 单位 + 正常范围 + 危急值）。"""
    path = CONFIG_DIR / "feature_meta.yaml"
    if not path.exists():
        return {"features": {}, "denied": []}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {"features": {}, "denied": []}
