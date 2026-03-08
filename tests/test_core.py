"""Tests for agentrace core functionality."""

import pytest
from typing import Sequence
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry import trace


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory exporter for testing."""

    def __init__(self):
        self._spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._spans)

    def clear(self):
        self._spans.clear()

    def shutdown(self):
        self.clear()

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True

import agentrace
from agentrace import (
    observe,
    trace_llm,
    trace_tool,
    trace_agent,
    trace_chain,
    trace_retrieval,
    set_session,
    set_user,
    calculate_cost,
    register_model_pricing,
)
from agentrace.context import trace_span, trace_llm_call
from agentrace.session import session_context
from agentrace import attributes as attrs
from agentrace.utils import safe_serialize, extract_openai_response


_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def setup_tracer():
    """Clear spans between tests and reset tracer cache."""
    _exporter.clear()
    from agentrace.tracer import _tracer
    _tracer._tracer = None
    yield _exporter


class TestObserveDecorator:
    def test_basic_observe(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def my_func(x, y):
            return x + y

        result = my_func(1, 2)
        assert result == 3

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "my_func"
        assert spans[0].attributes[attrs.AGENTRACE_SPAN_KIND] == "span"

    def test_observe_with_name(self, setup_tracer):
        exporter = setup_tracer

        @observe(name="custom-name", kind="chain")
        def my_func():
            return "ok"

        my_func()
        spans = exporter.get_finished_spans()
        assert spans[0].name == "custom-name"
        assert spans[0].attributes[attrs.AGENTRACE_SPAN_KIND] == "chain"

    def test_observe_captures_input_output(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def add(a, b):
            return a + b

        add(3, 4)
        span = exporter.get_finished_spans()[0]
        assert "3" in span.attributes[attrs.AGENTRACE_INPUT]
        assert "7" in span.attributes[attrs.AGENTRACE_OUTPUT]

    def test_observe_records_exception(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            fail()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code.name == "ERROR"

    @pytest.mark.asyncio
    async def test_observe_async(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        async def async_func(x):
            return x * 2

        result = await async_func(5)
        assert result == 10

        spans = exporter.get_finished_spans()
        assert len(spans) == 1


class TestTraceLLM:
    def test_trace_llm_basic(self, setup_tracer):
        exporter = setup_tracer

        @trace_llm(model="gpt-4o")
        def call_llm(messages):
            return {"content": "hello"}

        call_llm([{"role": "user", "content": "hi"}])
        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.AGENTRACE_SPAN_KIND] == "llm"
        assert span.attributes[attrs.GEN_AI_REQUEST_MODEL] == "gpt-4o"

    def test_trace_llm_extracts_openai_response(self, setup_tracer):
        exporter = setup_tracer

        # Mock OpenAI-style response
        response = MagicMock()
        response.model = "gpt-4o"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = MagicMock()
        choice.message.content = "Hello world"
        response.choices = [choice]

        @trace_llm(model="gpt-4o")
        def call_llm():
            return response

        call_llm()
        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert span.attributes[attrs.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert attrs.AGENTRACE_COST_USD in span.attributes


class TestTraceTool:
    def test_trace_tool(self, setup_tracer):
        exporter = setup_tracer

        @trace_tool(tool_name="web_search")
        def search(query):
            return ["result1"]

        result = search("test")
        assert result == ["result1"]

        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.AGENTRACE_SPAN_KIND] == "tool"
        assert span.attributes[attrs.AGENTRACE_TOOL_NAME] == "web_search"


class TestTraceAgent:
    def test_trace_agent(self, setup_tracer):
        exporter = setup_tracer

        @trace_agent(name="my-agent")
        def agent():
            return "done"

        agent()
        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.AGENTRACE_SPAN_KIND] == "agent"


class TestContextManagers:
    def test_trace_span(self, setup_tracer):
        exporter = setup_tracer

        with trace_span("my-span", kind="chain") as span:
            span.set_attribute("custom.key", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["custom.key"] == "value"

    def test_trace_llm_call(self, setup_tracer):
        exporter = setup_tracer

        with trace_llm_call("test-llm", model="gpt-4o") as llm:
            llm.record_usage(input_tokens=100, output_tokens=50)

        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert span.attributes[attrs.GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs.AGENTRACE_COST_USD in span.attributes


class TestSession:
    def test_set_session(self, setup_tracer):
        exporter = setup_tracer
        set_session("s-123")

        @observe()
        def my_func():
            return "ok"

        my_func()
        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.AGENTRACE_SESSION_ID] == "s-123"

    def test_session_context(self, setup_tracer):
        exporter = setup_tracer

        with session_context(session_id="ctx-session", user_id="ctx-user"):
            @observe()
            def my_func():
                return "ok"
            my_func()

        span = exporter.get_finished_spans()[0]
        assert span.attributes[attrs.AGENTRACE_SESSION_ID] == "ctx-session"
        assert span.attributes[attrs.AGENTRACE_USER_ID] == "ctx-user"


class TestMetrics:
    def test_calculate_cost(self):
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost > 0

    def test_register_model_pricing(self):
        register_model_pricing("my-model", 0.001, 0.002)
        cost = calculate_cost("my-model", input_tokens=1000, output_tokens=1000)
        assert cost == pytest.approx(0.003)

    def test_unknown_model(self):
        cost = calculate_cost("unknown-model-xyz", 100, 100)
        assert cost is None


class TestUtils:
    def test_safe_serialize_dict(self):
        result = safe_serialize({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_safe_serialize_truncation(self):
        long_str = "x" * 50_000
        result = safe_serialize(long_str)
        assert result.endswith("...[truncated]")

    def test_extract_openai_response(self):
        response = MagicMock()
        response.model = "gpt-4o"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = MagicMock()
        choice.message.content = "Hello"
        response.choices = [choice]

        info = extract_openai_response(response)
        assert info["input_tokens"] == 10
        assert info["output_tokens"] == 20
        assert info["response_model"] == "gpt-4o"
        assert info["completion"] == "Hello"


class TestNestedSpans:
    def test_nested_decorators(self, setup_tracer):
        exporter = setup_tracer

        @trace_tool(tool_name="search")
        def search(q):
            return ["r1"]

        @trace_agent(name="agent")
        def agent():
            return search("test")

        agent()
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        kinds = {s.attributes[attrs.AGENTRACE_SPAN_KIND] for s in spans}
        assert kinds == {"agent", "tool"}
