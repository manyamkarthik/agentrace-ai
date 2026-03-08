"""Configuration and initialization for agentrace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.resources import Resource


@dataclass
class _Config:
    """Global agentrace configuration. Set by init()."""
    service_name: str = "agentrace"
    capture_prompts: bool = True
    capture_tool_io: bool = True
    default_session_id: str | None = None
    default_user_id: str | None = None
    initialized: bool = False


_config = _Config()


def get_config() -> _Config:
    return _config


def init(
    service_name: str = "agentrace",
    exporters: list[str | SpanExporter] | None = None,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
    capture_prompts: bool = True,
    capture_tool_io: bool = True,
    default_session_id: str | None = None,
    default_user_id: str | None = None,
    resource_attributes: dict[str, str] | None = None,
) -> None:
    """Initialize agentrace. Call once at application startup.

    Args:
        service_name: Name of your service/agent.
        exporters: List of exporter names ("console", "otlp") or SpanExporter instances.
                   Defaults to ["console"].
        otlp_endpoint: OTLP collector endpoint (for "otlp" exporter).
        otlp_headers: Headers for OTLP exporter (e.g. auth tokens).
        capture_prompts: Whether to record prompt/completion text on LLM spans.
        capture_tool_io: Whether to record tool input/output.
        default_session_id: Default session ID for all spans.
        default_user_id: Default user ID for all spans.
        resource_attributes: Additional OTel resource attributes.
    """
    global _config

    if exporters is None:
        exporters = ["console"]

    # Build resource
    attrs: dict[str, Any] = {"service.name": service_name}
    if resource_attributes:
        attrs.update(resource_attributes)
    resource = Resource.create(attrs)

    # Create provider
    provider = TracerProvider(resource=resource)

    # Add exporters
    for exp in exporters:
        processor = _resolve_exporter(exp, otlp_endpoint, otlp_headers)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    # Store config
    _config = _Config(
        service_name=service_name,
        capture_prompts=capture_prompts,
        capture_tool_io=capture_tool_io,
        default_session_id=default_session_id,
        default_user_id=default_user_id,
        initialized=True,
    )

    # Set defaults for session/user
    if default_session_id:
        from agentrace.session import set_session
        set_session(default_session_id)
    if default_user_id:
        from agentrace.session import set_user
        set_user(default_user_id)


def _resolve_exporter(
    exporter: str | SpanExporter,
    otlp_endpoint: str | None,
    otlp_headers: dict[str, str] | None,
) -> SimpleSpanProcessor | BatchSpanProcessor:
    """Resolve an exporter string or instance into a span processor."""
    if isinstance(exporter, SpanExporter):
        return BatchSpanProcessor(exporter)

    if exporter == "console":
        from agentrace.exporters.console import AgentTraceConsoleExporter
        return SimpleSpanProcessor(AgentTraceConsoleExporter())

    if exporter == "otlp":
        from agentrace.exporters.otlp import create_otlp_exporter
        exp = create_otlp_exporter(endpoint=otlp_endpoint, headers=otlp_headers)
        return BatchSpanProcessor(exp)

    raise ValueError(
        f"Unknown exporter: {exporter!r}. Use 'console', 'otlp', or a SpanExporter instance."
    )


def shutdown() -> None:
    """Flush and shut down the tracer provider."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
