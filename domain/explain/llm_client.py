"""Agnes AI LLM 客户端封装（OpenAI 兼容接口 + Function Calling 强约束）。

特性：
    - OpenAI SDK 兼容（base_url 指向 Agnes）
    - 使用 Function Calling 强制 JSON 输出（PR-4.3）
    - 超时与异常捕获
    - 不向 API 发送任何 PII（由上层保证）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from domain.explain.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 调用结果。"""
    ok: bool
    content: str = ""               # 原始文本输出
    parsed: dict[str, Any] | None = None  # Function Calling 解析后的 JSON
    elapsed_ms: int = 0
    error: str = ""
    usage: dict[str, int] | None = None


# Function Calling 输出 Schema（强约束 JSON 结构）
EXPLAIN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_explanation",
        "description": "提交结构化的患者风险评估解释报告",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "1-2句话总结当前风险水平",
                },
                "factor_analysis": {
                    "type": "array",
                    "description": "按SHAP绝对值降序排列的关键驱动因素分析",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string", "description": "特征键名"},
                            "clinical_interpretation": {
                                "type": "string",
                                "description": "以「模型统计观察：」开头的临床解读",
                            },
                            "reference_id": {
                                "type": "string",
                                "description": "引用的参考片段编号，如 REF-01；无引用填空字符串",
                            },
                        },
                        "required": ["feature", "clinical_interpretation", "reference_id"],
                    },
                },
                "coverage_note": {
                    "type": "string",
                    "description": "已解释特征累计SHAP贡献占比说明",
                },
            },
            "required": ["summary", "factor_analysis", "coverage_note"],
        },
    },
}


class LLMClient:
    """Agnes LLM 客户端（OpenAI 兼容）。"""

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("缺少 openai 依赖，请运行 `pip install openai`") from e
        cfg = get_config()
        if not cfg.llm.api_key:
            raise RuntimeError("AGNES_API_KEY 未配置，请检查 .env 文件")
        self._client = OpenAI(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout_seconds,
        )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        use_function_calling: bool = True,
    ) -> LLMResponse:
        """调用 LLM 生成解释。

        Args:
            system_prompt: System Prompt
            user_prompt: User Prompt
            use_function_calling: 是否使用 Function Calling 强制结构化输出

        Returns:
            LLMResponse
        """
        import time
        cfg = get_config()
        t0 = time.time()
        try:
            client = self._ensure_client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            kwargs: dict[str, Any] = {
                "model": cfg.llm.model,
                "messages": messages,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
            }
            if use_function_calling:
                kwargs["tools"] = [EXPLAIN_TOOL_SCHEMA]
                kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit_explanation"}}

            resp = client.chat.completions.create(**kwargs)
            elapsed_ms = int((time.time() - t0) * 1000)
            choice = resp.choices[0]
            content = choice.message.content or ""

            parsed = None
            if use_function_calling and choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                args_str = tc.function.arguments or "{}"
                try:
                    parsed = json.loads(args_str)
                except json.JSONDecodeError as e:
                    logger.warning("Function Calling 返回非法 JSON: %s", e)
                    parsed = None

            # fallback: Function Calling 未触发时，尝试从 content 解析 JSON
            if parsed is None and content:
                content_stripped = content.strip()
                if content_stripped.startswith("{"):
                    try:
                        parsed = json.loads(content_stripped)
                        logger.info("LLM 以 JSON content 返回，已解析")
                    except json.JSONDecodeError as e:
                        logger.warning("content 非合法 JSON: %s", e)

            usage = None
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }

            return LLMResponse(
                ok=True,
                content=content,
                parsed=parsed,
                elapsed_ms=elapsed_ms,
                usage=usage,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            err_msg = f"{type(e).__name__}: {e}"
            logger.exception("LLM 调用失败")
            return LLMResponse(ok=False, elapsed_ms=elapsed_ms, error=err_msg)
