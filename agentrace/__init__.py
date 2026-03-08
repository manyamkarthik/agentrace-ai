"""agentrace - Lightweight OpenTelemetry-native tracing for LLM and AI agent applications."""

from agentrace._version import __version__
from agentrace.config import init, shutdown
from agentrace.decorators import (
    observe,
    trace_llm,
    trace_tool,
    trace_agent,
    trace_chain,
    trace_retrieval,
)
from agentrace.context import trace_span, trace_llm_call
from agentrace.session import set_session, set_user, session_context
from agentrace.metrics import calculate_cost, register_model_pricing

__all__ = [
    "__version__",
    "init",
    "shutdown",
    "observe",
    "trace_llm",
    "trace_tool",
    "trace_agent",
    "trace_chain",
    "trace_retrieval",
    "trace_span",
    "trace_llm_call",
    "set_session",
    "set_user",
    "session_context",
    "calculate_cost",
    "register_model_pricing",
]
