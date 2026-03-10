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
    """Clear spans between tests and reset tracer cache + config."""
    _exporter.clear()
    from agentrace.tracer import _tracer
    _tracer._tracer = None
    # Reset config so init() guard doesn't block tests
    from agentrace.config import _Config
    import agentrace.config as _cfg
    _cfg._config = _Config()
    # Reset session/user contextvars
    from agentrace.session import _session_id_var, _user_id_var
    _session_id_var.set(None)
    _user_id_var.set(None)
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


class TestInitConfig:
    def test_init_with_custom_exporter_uses_simple_processor(self):
        """Bug 1 fix: custom SpanExporter instances should use SimpleSpanProcessor by default."""
        from agentrace.config import _resolve_exporter
        exporter = InMemorySpanExporter()
        processor = _resolve_exporter(exporter, None, None, batch=False)
        assert isinstance(processor, SimpleSpanProcessor)

    def test_init_with_batch_true_uses_batch_processor(self):
        from agentrace.config import _resolve_exporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        exporter = InMemorySpanExporter()
        processor = _resolve_exporter(exporter, None, None, batch=True)
        assert isinstance(processor, BatchSpanProcessor)
        processor.shutdown()

    def test_init_with_provider(self, setup_tracer):
        """Bug 3 fix: accept a pre-built TracerProvider."""
        from agentrace.config import get_config
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        agentrace.init(
            service_name="test-provider",
            provider=provider,
            capture_prompts=False,
        )

        config = get_config()
        assert config.service_name == "test-provider"
        assert config.capture_prompts is False
        assert config.initialized is True

        provider.shutdown()


class TestCallbackHandlerNoneSerialzied:
    """Bug: on_chain_start crashes when LangGraph passes serialized=None."""

    def test_on_chain_start_with_none_serialized(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        # Should NOT raise — LangGraph passes None for internal nodes
        handler.on_chain_start(None, {"input": "test"}, run_id=run_id)
        handler.on_chain_end({"output": "done"}, run_id=run_id)

    def test_on_chain_start_with_empty_serialized(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        # Empty dict — should also not crash
        handler.on_chain_start({}, {"input": "test"}, run_id=run_id)
        handler.on_chain_end({"output": "done"}, run_id=run_id)

    def test_on_llm_start_with_none_serialized(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(None, ["hello"], run_id=run_id)
        handler.on_llm_end(MagicMock(llm_output=None), run_id=run_id)

    def test_on_tool_start_with_none_serialized(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        handler.on_tool_start(None, "input", run_id=run_id)
        handler.on_tool_end("output", run_id=run_id)


class TestSetUserInsideDecorator:
    """Bug: set_user() called inside @trace_agent body doesn't appear on span."""

    def test_set_user_inside_observe(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def my_func():
            set_user("late-user-123")
            return "ok"

        my_func()
        span = exporter.get_finished_spans()[0]
        # User set inside the function body MUST appear on the span
        assert span.attributes.get(attrs.AGENTRACE_USER_ID) == "late-user-123"

    def test_set_session_inside_observe(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def my_func():
            set_session("late-session-456")
            return "ok"

        my_func()
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get(attrs.AGENTRACE_SESSION_ID) == "late-session-456"

    @pytest.mark.asyncio
    async def test_set_user_inside_async_observe(self, setup_tracer):
        exporter = setup_tracer

        @observe(kind="agent")
        async def agent_fn():
            set_user("async-user-789")
            return "done"

        await agent_fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get(attrs.AGENTRACE_USER_ID) == "async-user-789"


# ── v0.3.0 Bug Tests ──────────────────────────────────────────────────────


class TestBug1_LLMCompletionNeverCaptured:
    """on_llm_end should capture completion text from response.generations."""

    def test_on_llm_end_captures_completion_text(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        exporter = setup_tracer
        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(None, ["hello"], run_id=run_id)

        # Mock LangChain LLMResult with generations
        response = MagicMock()
        gen = MagicMock()
        gen.text = "This is the LLM response"
        response.generations = [[gen]]
        response.llm_output = {}

        handler.on_llm_end(response, run_id=run_id)

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1
        span = spans[-1]
        assert "This is the LLM response" in span.attributes.get(attrs.AGENTRACE_COMPLETION, "")


class TestBug2_GeminiTokenCounts:
    """on_llm_end should capture Gemini-style usage_metadata tokens."""

    def test_on_llm_end_gemini_usage_metadata(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        exporter = setup_tracer
        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(None, ["hello"], run_id=run_id)

        response = MagicMock()
        gen = MagicMock()
        gen.text = "response"
        response.generations = [[gen]]
        # Gemini-style: usage_metadata instead of token_usage
        response.llm_output = {
            "usage_metadata": {
                "input_tokens": 42,
                "output_tokens": 17,
            }
        }

        handler.on_llm_end(response, run_id=run_id)

        span = exporter.get_finished_spans()[-1]
        assert span.attributes.get(attrs.GEN_AI_USAGE_INPUT_TOKENS) == 42
        assert span.attributes.get(attrs.GEN_AI_USAGE_OUTPUT_TOKENS) == 17


class TestBug3_GeminiModelName:
    """on_llm_start should read 'model' key (Gemini), not just 'model_name' (OpenAI)."""

    def test_on_llm_start_gemini_model_key(self, setup_tracer):
        from uuid import uuid4
        from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

        exporter = setup_tracer
        handler = AgentraceCallbackHandler()
        run_id = uuid4()

        # Gemini passes model under "model" not "model_name"
        handler.on_llm_start(
            None, ["hello"], run_id=run_id,
            invocation_params={"model": "gemini-2.0-flash"},
        )
        handler.on_llm_end(MagicMock(llm_output=None, generations=[]), run_id=run_id)

        span = exporter.get_finished_spans()[-1]
        assert span.attributes.get(attrs.GEN_AI_REQUEST_MODEL) == "gemini-2.0-flash"


class TestBug5_DoubleExceptionEvents:
    """Error spans should have exactly 1 exception event, not 2."""

    def test_observe_error_single_exception_event(self, setup_tracer):
        exporter = setup_tracer

        @observe()
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail()

        span = exporter.get_finished_spans()[0]
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1, f"Expected 1 exception event, got {len(exception_events)}"

    def test_trace_tool_error_single_exception_event(self, setup_tracer):
        exporter = setup_tracer

        @trace_tool(tool_name="bad_tool")
        def fail():
            raise RuntimeError("tool error")

        with pytest.raises(RuntimeError):
            fail()

        span = exporter.get_finished_spans()[0]
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1

    def test_trace_llm_error_single_exception_event(self, setup_tracer):
        exporter = setup_tracer

        @trace_llm(model="gpt-4o")
        def fail():
            raise RuntimeError("llm error")

        with pytest.raises(RuntimeError):
            fail()

        span = exporter.get_finished_spans()[0]
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1


class TestBug7_DoubleInit:
    """init() called twice should not silently create split-brain state."""

    def test_init_twice_raises_or_warns(self, setup_tracer):
        from agentrace.config import _config, init, get_config

        # First init
        exporter1 = InMemorySpanExporter()
        p1 = TracerProvider()
        p1.add_span_processor(SimpleSpanProcessor(exporter1))
        init(service_name="first", provider=p1)
        assert get_config().initialized is True

        # Second init should raise RuntimeError
        exporter2 = InMemorySpanExporter()
        p2 = TracerProvider()
        p2.add_span_processor(SimpleSpanProcessor(exporter2))
        with pytest.raises(RuntimeError):
            init(service_name="second", provider=p2)

        p1.shutdown()
        p2.shutdown()


class TestBug8_TraceToolMissingReattachContext:
    """set_user() inside @trace_tool should appear on tool span."""

    def test_set_user_inside_trace_tool(self, setup_tracer):
        exporter = setup_tracer

        @trace_tool(tool_name="my_tool")
        def tool_fn():
            set_user("tool-user-999")
            return "done"

        tool_fn()
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get(attrs.AGENTRACE_USER_ID) == "tool-user-999"


class TestBug9_ChainRetrievalNameAlias:
    """trace_chain and trace_retrieval should accept name aliases like trace_tool."""

    def test_trace_chain_accepts_chain_name_kwarg(self, setup_tracer):
        # Should not raise TypeError
        @trace_chain(name="my-chain")
        def chain_fn():
            return "ok"
        chain_fn()

    def test_trace_retrieval_accepts_retrieval_name_kwarg(self, setup_tracer):
        @trace_retrieval(name="my-retrieval")
        def retrieval_fn():
            return ["doc1"]
        retrieval_fn()


class TestBug10_RetrievalSpecificAttributes:
    """@trace_retrieval should set agentrace.retrieval.query and num_documents."""

    def test_trace_retrieval_sets_query_attr(self, setup_tracer):
        exporter = setup_tracer

        @trace_retrieval(name="search")
        def search(query, top_k=5):
            return ["doc1", "doc2", "doc3"]

        search("what is otel?")
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get(attrs.AGENTRACE_RETRIEVAL_QUERY) == "what is otel?"

    def test_trace_retrieval_sets_num_docs(self, setup_tracer):
        exporter = setup_tracer

        @trace_retrieval(name="search")
        def search(query):
            return ["doc1", "doc2"]

        search("test")
        span = exporter.get_finished_spans()[0]
        assert span.attributes.get(attrs.AGENTRACE_RETRIEVAL_NUM_DOCS) == 2


class TestBug11_SessionContextReset:
    """session_context should correctly reset both session and user on exit."""

    def test_session_context_resets_both_vars(self, setup_tracer):
        from agentrace.session import get_session, get_user

        # Set initial values
        set_session("outer-session")
        set_user("outer-user")

        with session_context(session_id="inner-session", user_id="inner-user"):
            assert get_session() == "inner-session"
            assert get_user() == "inner-user"

        # After exit, previous values should be restored
        assert get_session() == "outer-session"
        assert get_user() == "outer-user"

    def test_session_context_resets_session_only(self, setup_tracer):
        from agentrace.session import get_session, get_user

        set_session("original")
        set_user("keep-this")

        with session_context(session_id="temp"):
            assert get_session() == "temp"

        assert get_session() == "original"
        assert get_user() == "keep-this"

    def test_session_context_resets_user_only(self, setup_tracer):
        from agentrace.session import get_session, get_user

        set_session("keep-this")
        set_user("original")

        with session_context(user_id="temp"):
            assert get_user() == "temp"

        assert get_user() == "original"
        assert get_session() == "keep-this"
