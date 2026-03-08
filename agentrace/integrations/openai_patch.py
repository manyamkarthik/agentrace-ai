"""OpenAI SDK integration for automatic tracing."""

from __future__ import annotations

import functools
from typing import Any

from opentelemetry.trace import StatusCode, use_span

from agentrace import attributes as attrs
from agentrace.config import get_config
from agentrace.decorators import _process_llm_response
from agentrace.tracer import get_tracer
from agentrace.utils import safe_serialize


_original_create = None
_original_async_create = None


def instrument() -> None:
    """Monkey-patch the OpenAI SDK to auto-trace all chat completion calls.

    Usage:
        from agentrace.integrations import openai_patch
        openai_patch.instrument()

        # All subsequent OpenAI calls are traced automatically
        client = OpenAI()
        response = client.chat.completions.create(...)
    """
    global _original_create, _original_async_create

    try:
        import openai
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except ImportError:
        raise ImportError("OpenAI SDK not installed. Run: pip install agentrace[openai]")

    if _original_create is not None:
        return  # Already instrumented

    _original_create = Completions.create
    _original_async_create = AsyncCompletions.create

    @functools.wraps(_original_create)
    def traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        tracer = get_tracer()
        config = get_config()
        span = tracer.start_llm_span(f"openai.chat.{model}", model=model)
        span.set_attribute(attrs.GEN_AI_SYSTEM, "openai")

        if config.capture_prompts and "messages" in kwargs:
            span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(kwargs["messages"]))

        with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
            try:
                result = _original_create(self, *args, **kwargs)
                _process_llm_response(span, result, model, config.capture_prompts)
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
        span = tracer.start_llm_span(f"openai.chat.{model}", model=model)
        span.set_attribute(attrs.GEN_AI_SYSTEM, "openai")

        if config.capture_prompts and "messages" in kwargs:
            span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(kwargs["messages"]))

        with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
            try:
                result = await _original_async_create(self, *args, **kwargs)
                _process_llm_response(span, result, model, config.capture_prompts)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    Completions.create = traced_create
    AsyncCompletions.create = traced_async_create


def uninstrument() -> None:
    """Remove the monkey-patch and restore original OpenAI methods."""
    global _original_create, _original_async_create

    if _original_create is None:
        return

    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
        Completions.create = _original_create
        AsyncCompletions.create = _original_async_create
    except ImportError:
        pass

    _original_create = None
    _original_async_create = None
