"""LangChain callback handler integration for automatic tracing."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from opentelemetry.trace import Span, StatusCode, use_span

from agentrace import attributes as attrs
from agentrace.tracer import get_tracer
from agentrace.utils import safe_serialize
from agentrace.metrics import calculate_cost

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # type: ignore[misc, assignment]


class AgentraceCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback handler that creates OTel spans for each step.

    Usage:
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler
        handler = AgentraceCallbackHandler()
        chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """

    def __init__(self) -> None:
        self._spans: dict[UUID, Span] = {}

    def _start_span(self, run_id: UUID, name: str, kind: str) -> Span:
        tracer = get_tracer()
        span = tracer.start_span(name, kind=kind)
        self._spans[run_id] = span
        return span

    def _end_span(self, run_id: UUID) -> None:
        span = self._spans.pop(run_id, None)
        if span:
            span.end()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "llm"
        span = self._start_span(run_id, f"langchain.llm.{name}", "llm")
        model = kwargs.get("invocation_params", {}).get("model_name", "")
        if model:
            span.set_attribute(attrs.GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(attrs.AGENTRACE_PROMPT, safe_serialize(prompts))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span and hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                span.set_attribute(attrs.GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens", 0))
                span.set_attribute(attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("completion_tokens", 0))
        self._end_span(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        self._end_span(run_id)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "chain"
        span = self._start_span(run_id, f"langchain.chain.{name}", "chain")
        span.set_attribute(attrs.AGENTRACE_INPUT, safe_serialize(inputs))

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_attribute(attrs.AGENTRACE_OUTPUT, safe_serialize(outputs))
        self._end_span(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        self._end_span(run_id)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool")
        span = self._start_span(run_id, f"langchain.tool.{name}", "tool")
        span.set_attribute(attrs.AGENTRACE_TOOL_NAME, name)
        span.set_attribute(attrs.AGENTRACE_TOOL_INPUT, safe_serialize(input_str))

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_attribute(attrs.AGENTRACE_TOOL_OUTPUT, safe_serialize(output))
        self._end_span(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        self._end_span(run_id)

    def on_retriever_start(
        self, serialized: dict[str, Any], query: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        span = self._start_span(run_id, "langchain.retriever", "retrieval")
        span.set_attribute(attrs.AGENTRACE_RETRIEVAL_QUERY, query)

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span and documents:
            span.set_attribute(attrs.AGENTRACE_RETRIEVAL_NUM_DOCS, len(documents))
        self._end_span(run_id)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
        self._end_span(run_id)
