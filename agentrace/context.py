"""Context manager API for manual span control."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry.trace import Span, StatusCode, use_span

from agentrace import attributes as attrs
from agentrace.config import get_config
from agentrace.metrics import calculate_cost
from agentrace.tracer import get_tracer
from agentrace.utils import (
    safe_serialize,
    extract_openai_response,
    extract_anthropic_response,
)


@contextmanager
def trace_span(
    name: str, kind: str = "span", **span_attrs: Any
) -> Generator[Span, None, None]:
    """Generic context manager for creating a traced span.

    Usage:
        with trace_span("preprocessing", kind="chain") as span:
            span.set_attribute("custom.key", "value")
            result = do_work()
    """
    tracer = get_tracer()
    span = tracer.start_span(name, kind=kind)
    for key, val in span_attrs.items():
        span.set_attribute(key, val)

    with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


class LLMSpanContext:
    """Wrapper around a span with LLM-specific helper methods."""

    def __init__(self, span: Span, model: str | None = None) -> None:
        self.span = span
        self.model = model

    def record_response(self, response: Any) -> None:
        """Auto-extract token usage, cost, and completion from an LLM response object."""
        config = get_config()
        info = extract_openai_response(response)
        if not info:
            info = extract_anthropic_response(response)

        if info.get("response_model"):
            self.span.set_attribute(attrs.GEN_AI_RESPONSE_MODEL, info["response_model"])
            self.model = info["response_model"]
        if info.get("input_tokens") is not None:
            self.span.set_attribute(attrs.GEN_AI_USAGE_INPUT_TOKENS, info["input_tokens"])
        if info.get("output_tokens") is not None:
            self.span.set_attribute(attrs.GEN_AI_USAGE_OUTPUT_TOKENS, info["output_tokens"])
        if info.get("finish_reasons"):
            self.span.set_attribute(attrs.GEN_AI_RESPONSE_FINISH_REASON, str(info["finish_reasons"]))
        if config.capture_prompts and info.get("completion"):
            self.span.set_attribute(attrs.AGENTRACE_COMPLETION, safe_serialize(info["completion"]))

        if self.model and info.get("input_tokens") is not None and info.get("output_tokens") is not None:
            cost = calculate_cost(self.model, info["input_tokens"], info["output_tokens"])
            if cost is not None:
                self.span.set_attribute(attrs.AGENTRACE_COST_USD, cost)

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        """Manually record token usage and cost."""
        self.span.set_attribute(attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        self.span.set_attribute(attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        if cost_usd is not None:
            self.span.set_attribute(attrs.AGENTRACE_COST_USD, cost_usd)
        elif self.model:
            cost = calculate_cost(self.model, input_tokens, output_tokens)
            if cost is not None:
                self.span.set_attribute(attrs.AGENTRACE_COST_USD, cost)

    def record_messages(self, messages: list[dict]) -> None:
        """Record prompt messages."""
        self.span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(messages))

    def set_attribute(self, key: str, value: Any) -> None:
        """Pass-through to the underlying span."""
        self.span.set_attribute(key, value)


@contextmanager
def trace_llm_call(
    name: str, model: str | None = None
) -> Generator[LLMSpanContext, None, None]:
    """Context manager for tracing an LLM call with helper methods.

    Usage:
        with trace_llm_call("gpt-4o-call", model="gpt-4o") as llm:
            response = client.chat.completions.create(model="gpt-4o", messages=msgs)
            llm.record_response(response)
    """
    tracer = get_tracer()
    span = tracer.start_llm_span(name, model=model)
    ctx = LLMSpanContext(span, model=model)

    with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
        try:
            yield ctx
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
