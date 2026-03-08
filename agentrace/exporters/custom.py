"""Base class for user-defined custom exporters."""

from __future__ import annotations

from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class AgentraceExporter(SpanExporter):
    """Base class for custom agentrace exporters.

    Subclass this to send spans to custom destinations (databases, webhooks, etc.).

    Example:
        class MyWebhookExporter(AgentraceExporter):
            def export(self, spans):
                for span in spans:
                    requests.post(self.url, json=self.span_to_dict(span))
                return SpanExportResult.SUCCESS
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise NotImplementedError("Subclass must implement export()")

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True

    @staticmethod
    def span_to_dict(span: ReadableSpan) -> dict:
        """Helper to convert a span to a plain dict for serialization."""
        duration_ns = (span.end_time or 0) - (span.start_time or 0)
        return {
            "name": span.name,
            "trace_id": format(span.context.trace_id, "032x") if span.context else None,
            "span_id": format(span.context.span_id, "016x") if span.context else None,
            "parent_span_id": (
                format(span.parent.span_id, "016x") if span.parent else None
            ),
            "start_time_ns": span.start_time,
            "end_time_ns": span.end_time,
            "duration_ms": duration_ns / 1e6,
            "status": span.status.status_code.name if span.status else None,
            "attributes": dict(span.attributes) if span.attributes else {},
        }
