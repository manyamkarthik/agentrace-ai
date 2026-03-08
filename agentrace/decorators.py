"""Decorator-based tracing API for LLM, tool, agent, chain, and retrieval spans."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar, overload

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode, use_span

from agentrace import attributes as attrs
from agentrace.config import get_config
from agentrace.metrics import calculate_cost
from agentrace.tracer import get_tracer
from agentrace.utils import (
    safe_serialize,
    extract_openai_response,
    extract_anthropic_response,
)

F = TypeVar("F", bound=Callable[..., Any])


def observe(
    name: str | None = None,
    kind: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Callable[[F], F]:
    """General-purpose tracing decorator.

    Args:
        name: Span name. Defaults to function name.
        kind: Span kind - "llm", "tool", "agent", "chain", "retrieval", or "span".
        capture_input: Record function arguments as span attributes.
        capture_output: Record function return value as span attributes.
        session_id: Override session ID for this span.
        user_id: Override user ID for this span.
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span = tracer.start_span(span_name, kind=kind)
            if session_id:
                span.set_attribute(attrs.AGENTRACE_SESSION_ID, session_id)
            if user_id:
                span.set_attribute(attrs.AGENTRACE_USER_ID, user_id)

            with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
                if capture_input:
                    _record_input(span, fn, args, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                    if capture_output:
                        span.set_attribute(
                            attrs.AGENTRACE_OUTPUT, safe_serialize(result)
                        )
                    return result
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span = tracer.start_span(span_name, kind=kind)
            if session_id:
                span.set_attribute(attrs.AGENTRACE_SESSION_ID, session_id)
            if user_id:
                span.set_attribute(attrs.AGENTRACE_USER_ID, user_id)

            with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
                if capture_input:
                    _record_input(span, fn, args, kwargs)
                try:
                    result = fn(*args, **kwargs)
                    if capture_output:
                        span.set_attribute(
                            attrs.AGENTRACE_OUTPUT, safe_serialize(result)
                        )
                    return result
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def trace_llm(
    model: str | None = None,
    name: str | None = None,
    capture_prompts: bool | None = None,
) -> Callable[[F], F]:
    """Decorator for LLM calls. Auto-extracts token usage and cost from responses.

    Args:
        model: Model name (e.g. "gpt-4o"). Also extracted from response if available.
        name: Span name. Defaults to function name.
        capture_prompts: Record prompt/completion text. Defaults to global config.
    """

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _run_llm_span(fn, span_name, model, capture_prompts, args, kwargs, is_async=True)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_llm_span(fn, span_name, model, capture_prompts, args, kwargs, is_async=False)

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _run_llm_span(
    fn: Callable,
    span_name: str,
    model: str | None,
    capture_prompts: bool | None,
    args: tuple,
    kwargs: dict,
    is_async: bool,
) -> Any:
    """Shared logic for sync/async LLM span execution."""
    tracer = get_tracer()
    config = get_config()
    should_capture = capture_prompts if capture_prompts is not None else config.capture_prompts

    span = tracer.start_llm_span(span_name, model=model)

    # Try to extract model from kwargs
    kwarg_model = kwargs.get("model", model)
    if kwarg_model:
        span.set_attribute(attrs.GEN_AI_REQUEST_MODEL, kwarg_model)

    # Record prompt messages if available
    if should_capture:
        messages = kwargs.get("messages")
        if messages:
            span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(messages))

    if is_async:
        return _run_llm_span_async(fn, span, kwarg_model or model, should_capture, args, kwargs)
    return _run_llm_span_sync(fn, span, kwarg_model or model, should_capture, args, kwargs)


async def _run_llm_span_async(fn, span, model, should_capture, args, kwargs):
    with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
        try:
            result = await fn(*args, **kwargs)
            _process_llm_response(span, result, model, should_capture)
            return result
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def _run_llm_span_sync(fn, span, model, should_capture, args, kwargs):
    with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
        try:
            result = fn(*args, **kwargs)
            _process_llm_response(span, result, model, should_capture)
            return result
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def _process_llm_response(span, result, model, should_capture):
    """Extract token usage, cost, and completion from an LLM response."""
    # Try OpenAI format first, then Anthropic
    info = extract_openai_response(result)
    if not info:
        info = extract_anthropic_response(result)

    if info.get("response_model"):
        span.set_attribute(attrs.GEN_AI_RESPONSE_MODEL, info["response_model"])
        model = info["response_model"]
    if info.get("input_tokens") is not None:
        span.set_attribute(attrs.GEN_AI_USAGE_INPUT_TOKENS, info["input_tokens"])
    if info.get("output_tokens") is not None:
        span.set_attribute(attrs.GEN_AI_USAGE_OUTPUT_TOKENS, info["output_tokens"])
    if info.get("finish_reasons"):
        span.set_attribute(
            attrs.GEN_AI_RESPONSE_FINISH_REASON, str(info["finish_reasons"])
        )
    if should_capture and info.get("completion"):
        span.set_attribute(attrs.AGENTRACE_COMPLETION, safe_serialize(info["completion"]))

    # Calculate cost
    if model and info.get("input_tokens") is not None and info.get("output_tokens") is not None:
        cost = calculate_cost(model, info["input_tokens"], info["output_tokens"])
        if cost is not None:
            span.set_attribute(attrs.AGENTRACE_COST_USD, cost)


def trace_tool(
    tool_name: str | None = None,
    name: str | None = None,
    capture_io: bool | None = None,
) -> Callable[[F], F]:
    """Decorator for tool/function calls in an agent workflow."""

    def decorator(fn: F) -> F:
        span_name = name or fn.__name__
        resolved_tool = tool_name or fn.__name__
        config = get_config()
        should_capture = capture_io if capture_io is not None else config.capture_tool_io

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span = tracer.start_tool_span(span_name, tool_name=resolved_tool)
            with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
                if should_capture:
                    _record_input(span, fn, args, kwargs, attr_key=attrs.AGENTRACE_TOOL_INPUT)
                try:
                    result = await fn(*args, **kwargs)
                    if should_capture:
                        span.set_attribute(attrs.AGENTRACE_TOOL_OUTPUT, safe_serialize(result))
                    return result
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span = tracer.start_tool_span(span_name, tool_name=resolved_tool)
            with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
                if should_capture:
                    _record_input(span, fn, args, kwargs, attr_key=attrs.AGENTRACE_TOOL_INPUT)
                try:
                    result = fn(*args, **kwargs)
                    if should_capture:
                        span.set_attribute(attrs.AGENTRACE_TOOL_OUTPUT, safe_serialize(result))
                    return result
                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def trace_agent(
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for agent entry points. Creates a parent span for the agent run."""
    return observe(name=name, kind="agent")


def trace_chain(
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for chain/pipeline steps."""
    return observe(name=name, kind="chain")


def trace_retrieval(
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for retrieval operations (vector search, document fetch, etc.)."""
    return observe(name=name, kind="retrieval")


def _record_input(
    span, fn: Callable, args: tuple, kwargs: dict, attr_key: str = attrs.AGENTRACE_INPUT
) -> None:
    """Record function input as a span attribute."""
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        input_data: dict[str, Any] = {}
        for i, arg in enumerate(args):
            key = params[i] if i < len(params) else f"arg_{i}"
            input_data[key] = arg
        input_data.update(kwargs)
        span.set_attribute(attr_key, safe_serialize(input_data))
    except Exception:
        pass
