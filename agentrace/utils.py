"""Internal helpers for serialization and truncation."""

from __future__ import annotations

import json
from typing import Any

_MAX_ATTR_LENGTH = 32_000  # OTel backends typically have attribute size limits


def safe_serialize(obj: Any, max_length: int = _MAX_ATTR_LENGTH) -> str:
    """Serialize an object to JSON string, truncating if needed."""
    try:
        if isinstance(obj, str):
            text = obj
        elif isinstance(obj, (dict, list, tuple)):
            text = json.dumps(obj, default=str, ensure_ascii=False)
        elif hasattr(obj, "model_dump"):
            text = json.dumps(obj.model_dump(), default=str, ensure_ascii=False)
        elif hasattr(obj, "__dict__"):
            text = json.dumps(obj.__dict__, default=str, ensure_ascii=False)
        else:
            text = str(obj)
    except Exception:
        text = str(obj)

    if len(text) > max_length:
        return text[:max_length] + "...[truncated]"
    return text


def extract_openai_response(response: Any) -> dict[str, Any]:
    """Extract token usage and metadata from an OpenAI-style response."""
    info: dict[str, Any] = {}
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        if hasattr(usage, "prompt_tokens"):
            info["input_tokens"] = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            info["output_tokens"] = usage.completion_tokens
    if hasattr(response, "model"):
        info["response_model"] = response.model
    if hasattr(response, "choices") and response.choices:
        reasons = [
            c.finish_reason for c in response.choices if hasattr(c, "finish_reason")
        ]
        if reasons:
            info["finish_reasons"] = reasons
        # Extract completion text
        first = response.choices[0]
        if hasattr(first, "message") and hasattr(first.message, "content"):
            info["completion"] = first.message.content
    return info


def extract_anthropic_response(response: Any) -> dict[str, Any]:
    """Extract token usage and metadata from an Anthropic-style response."""
    info: dict[str, Any] = {}
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        if hasattr(usage, "input_tokens"):
            info["input_tokens"] = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            info["output_tokens"] = usage.output_tokens
    if hasattr(response, "model"):
        info["response_model"] = response.model
    if hasattr(response, "stop_reason"):
        info["finish_reasons"] = [response.stop_reason]
    if hasattr(response, "content") and response.content:
        texts = [
            block.text
            for block in response.content
            if hasattr(block, "text")
        ]
        if texts:
            info["completion"] = "\n".join(texts)
    return info
