"""Pretty-printed console exporter for development and debugging."""

from __future__ import annotations

import json
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentrace import attributes as attrs

# ANSI color codes
_COLORS = {
    "llm": "\033[94m",       # Blue
    "tool": "\033[92m",      # Green
    "agent": "\033[93m",     # Yellow
    "chain": "\033[96m",     # Cyan
    "retrieval": "\033[95m", # Magenta
    "span": "\033[37m",      # White/Gray
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


class AgentTraceConsoleExporter(SpanExporter):
    """Exports spans to console with color-coded, human-readable formatting."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self._print_span(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True

    def _print_span(self, span: ReadableSpan) -> None:
        span_attrs = dict(span.attributes) if span.attributes else {}
        kind = span_attrs.get(attrs.AGENTRACE_SPAN_KIND, "span")
        color = _COLORS.get(kind, _COLORS["span"])

        # Duration
        duration_ns = (span.end_time or 0) - (span.start_time or 0)
        duration_s = duration_ns / 1e9

        # Build output
        parts = [f"{color}{_BOLD}[{kind.upper()}]{_RESET} {color}{span.name}{_RESET}"]
        parts.append(f" {_DIM}{duration_s:.2f}s{_RESET}")

        # Token info for LLM spans
        input_tokens = span_attrs.get(attrs.GEN_AI_USAGE_INPUT_TOKENS)
        output_tokens = span_attrs.get(attrs.GEN_AI_USAGE_OUTPUT_TOKENS)
        if input_tokens is not None or output_tokens is not None:
            tok_str = f"  tokens: {input_tokens or '?'}\u2192{output_tokens or '?'}"
            parts.append(f" {_DIM}{tok_str}{_RESET}")

        # Cost
        cost = span_attrs.get(attrs.AGENTRACE_COST_USD)
        if cost is not None:
            parts.append(f" {_DIM}${cost:.4f}{_RESET}")

        # Model
        model = span_attrs.get(attrs.GEN_AI_REQUEST_MODEL) or span_attrs.get(attrs.GEN_AI_RESPONSE_MODEL)
        if model:
            parts.append(f" {_DIM}({model}){_RESET}")

        # Tool name
        tool_name = span_attrs.get(attrs.AGENTRACE_TOOL_NAME)
        if tool_name:
            parts.append(f" {_DIM}[{tool_name}]{_RESET}")

        # Session/User
        session = span_attrs.get(attrs.AGENTRACE_SESSION_ID)
        user = span_attrs.get(attrs.AGENTRACE_USER_ID)
        if session or user:
            ctx_parts = []
            if session:
                ctx_parts.append(f"session={session}")
            if user:
                ctx_parts.append(f"user={user}")
            parts.append(f" {_DIM}{{{', '.join(ctx_parts)}}}{_RESET}")

        # Trace ID
        trace_id = format(span.context.trace_id, "032x") if span.context else "?"
        parts.append(f" {_DIM}trace={trace_id[:8]}...{_RESET}")

        # Error status
        if span.status and span.status.status_code.name == "ERROR":
            parts.append(f" \033[91m[ERROR: {span.status.description}]{_RESET}")

        print("".join(parts))
