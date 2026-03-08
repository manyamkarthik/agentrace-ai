"""Anthropic SDK integration for automatic tracing."""

from __future__ import annotations

import functools
from typing import Any

from opentelemetry.trace import StatusCode, use_span

from agentrace import attributes as attrs
from agentrace.config import get_config
from agentrace.tracer import get_tracer
from agentrace.utils import safe_serialize, extract_anthropic_response
from agentrace.metrics import calculate_cost


_original_create = None
_original_async_create = None


def instrument() -> None:
    """Monkey-patch the Anthropic SDK to auto-trace all message creation calls.

    Usage:
        from agentrace.integrations import anthropic_patch
        anthropic_patch.instrument()

        client = anthropic.Anthropic()
        response = client.messages.create(...)
    """
    global _original_create, _original_async_create

    try:
        import anthropic
        from anthropic.resources.messages import Messages, AsyncMessages
    except ImportError:
        raise ImportError("Anthropic SDK not installed. Run: pip install agentrace[anthropic]")

    if _original_create is not None:
        return

    _original_create = Messages.create
    _original_async_create = AsyncMessages.create

    @functools.wraps(_original_create)
    def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        tracer = get_tracer()
        config = get_config()
        span = tracer.start_llm_span(f"anthropic.messages.{model}", model=model)
        span.set_attribute(attrs.GEN_AI_SYSTEM, "anthropic")

        if config.capture_prompts and "messages" in kwargs:
            span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(kwargs["messages"]))

        with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
            try:
                result = _original_create(self, *args, **kwargs)
                _process_anthropic_response(span, result, model, config.capture_prompts)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    @functools.wraps(_original_async_create)
    async def traced_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        tracer = get_tracer()
        config = get_config()
        span = tracer.start_llm_span(f"anthropic.messages.{model}", model=model)
        span.set_attribute(attrs.GEN_AI_SYSTEM, "anthropic")

        if config.capture_prompts and "messages" in kwargs:
            span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(kwargs["messages"]))

        with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
            try:
                result = await _original_async_create(self, *args, **kwargs)
                _process_anthropic_response(span, result, model, config.capture_prompts)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    Messages.create = traced_create
    AsyncMessages.create = traced_async_create


def _process_anthropic_response(span, result, model, should_capture):
    info = extract_anthropic_response(result)
    if info.get("response_model"):
        span.set_attribute(attrs.GEN_AI_RESPONSE_MODEL, info["response_model"])
        model = info["response_model"]
    if info.get("input_tokens") is not None:
        span.set_attribute(attrs.GEN_AI_USAGE_INPUT_TOKENS, info["input_tokens"])
    if info.get("output_tokens") is not None:
        span.set_attribute(attrs.GEN_AI_USAGE_OUTPUT_TOKENS, info["output_tokens"])
    if info.get("finish_reasons"):
        span.set_attribute(attrs.GEN_AI_RESPONSE_FINISH_REASON, str(info["finish_reasons"]))
    if should_capture and info.get("completion"):
        span.set_attribute(attrs.AGENTRACE_COMPLETION, safe_serialize(info["completion"]))
    if model and info.get("input_tokens") is not None and info.get("output_tokens") is not None:
        cost = calculate_cost(model, info["input_tokens"], info["output_tokens"])
        if cost is not None:
            span.set_attribute(attrs.AGENTRACE_COST_USD, cost)


def uninstrument() -> None:
    """Remove the monkey-patch and restore original Anthropic methods."""
    global _original_create, _original_async_create

    if _original_create is None:
        return

    try:
        from anthropic.resources.messages import Messages, AsyncMessages
        Messages.create = _original_create
        AsyncMessages.create = _original_async_create
    except ImportError:
        pass

    _original_create = None
    _original_async_create = None
