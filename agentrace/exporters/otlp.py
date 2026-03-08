"""OTLP exporter setup wrapper."""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SpanExporter


def create_otlp_exporter(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    protocol: str = "grpc",
) -> SpanExporter:
    """Create an OTLP span exporter.

    Requires: pip install agentrace[otlp]

    Args:
        endpoint: Collector endpoint. Defaults to localhost:4317 (gRPC) or :4318 (HTTP).
        headers: Auth headers.
        protocol: "grpc" or "http".
    """
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise ImportError(
                "OTLP gRPC exporter not installed. Run: pip install agentrace[otlp]"
            )
        kwargs = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        if headers:
            kwargs["headers"] = list(headers.items())
        return OTLPSpanExporter(**kwargs)
    elif protocol == "http":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise ImportError(
                "OTLP HTTP exporter not installed. Run: pip install opentelemetry-exporter-otlp-proto-http"
            )
        kwargs = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        if headers:
            kwargs["headers"] = headers
        return OTLPSpanExporter(**kwargs)
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}. Use 'grpc' or 'http'.")
