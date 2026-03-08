"""Core tracer wrapper around OpenTelemetry."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from agentrace._version import __version__
from agentrace import attributes as attrs
from agentrace.session import get_session, get_user
from agentrace.config import get_config


class AgentTracer:
    """Internal tracer wrapper. Not part of the public API."""

    def __init__(self) -> None:
        self._tracer: Tracer | None = None

    @property
    def tracer(self) -> Tracer:
        if self._tracer is None:
            self._tracer = trace.get_tracer("agentrace", __version__)
        return self._tracer

    def _attach_common(self, span: Span) -> None:
        """Attach session/user IDs if set."""
        session_id = get_session() or get_config().default_session_id
        user_id = get_user() or get_config().default_user_id
        if session_id:
            span.set_attribute(attrs.AGENTRACE_SESSION_ID, session_id)
        if user_id:
            span.set_attribute(attrs.AGENTRACE_USER_ID, user_id)

    def start_span(self, name: str, kind: str = "span") -> Span:
        """Start a generic span."""
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, kind)
        self._attach_common(span)
        return span

    def start_llm_span(self, name: str, model: str | None = None) -> Span:
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, "llm")
        if model:
            span.set_attribute(attrs.GEN_AI_REQUEST_MODEL, model)
        self._attach_common(span)
        return span

    def start_tool_span(self, name: str, tool_name: str | None = None) -> Span:
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, "tool")
        if tool_name:
            span.set_attribute(attrs.AGENTRACE_TOOL_NAME, tool_name)
        self._attach_common(span)
        return span

    def start_agent_span(self, name: str) -> Span:
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, "agent")
        span.set_attribute(attrs.AGENTRACE_AGENT_NAME, name)
        self._attach_common(span)
        return span

    def start_chain_span(self, name: str) -> Span:
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, "chain")
        self._attach_common(span)
        return span

    def start_retrieval_span(self, name: str, query: str | None = None) -> Span:
        span = self.tracer.start_span(name)
        span.set_attribute(attrs.AGENTRACE_SPAN_KIND, "retrieval")
        if query:
            span.set_attribute(attrs.AGENTRACE_RETRIEVAL_QUERY, query)
        self._attach_common(span)
        return span


# Module-level singleton
_tracer = AgentTracer()


def get_tracer() -> AgentTracer:
    return _tracer
