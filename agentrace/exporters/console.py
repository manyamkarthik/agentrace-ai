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
_RED = "\033[91m"

_MAX_TEXT = 200


def _parse_json(raw) -> dict | str:
    """Try to parse a JSON string, return dict or original string."""
    if not raw:
        return ""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return str(raw)


def _truncate(text: str, max_len: int = _MAX_TEXT) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _clean_input(data: dict) -> str:
    """Format input dict, dropping internal fields like user_id."""
    clean = {k: v for k, v in data.items() if k not in ("user_id",)}
    return json.dumps(clean, ensure_ascii=False, default=str)


class AgentTraceConsoleExporter(SpanExporter):
    """Exports spans to console with color-coded, human-readable formatting.

    Shows different detail levels per span kind:
    - AGENT: separator line, agent name, user, input message, output (truncated)
    - TOOL: tool name, cleaned input, key output fields
    - LLM: model, tokens, cost
    - CHAIN/RETRIEVAL: name, duration
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self._print_span(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True

    def _print_span(self, span: ReadableSpan) -> None:
        a = dict(span.attributes) if span.attributes else {}
        kind = a.get(attrs.AGENTRACE_SPAN_KIND, "span")
        color = _COLORS.get(kind, _COLORS["span"])
        duration_s = ((span.end_time or 0) - (span.start_time or 0)) / 1e9
        is_error = span.status and span.status.status_code.name == "ERROR"
        status = f"{_RED}ERROR{_RESET}" if is_error else "ok"

        if kind == "agent":
            self._print_agent(span, a, color, duration_s, is_error)
        elif kind == "tool":
            self._print_tool(span, a, color, duration_s, is_error)
        elif kind == "llm":
            self._print_llm(span, a, color, duration_s, is_error)
        else:
            self._print_generic(span, a, color, kind, duration_s, is_error)

    def _print_agent(self, span, a, color, dur, is_error):
        user = a.get(attrs.AGENTRACE_USER_ID) or a.get(attrs.AGENTRACE_SESSION_ID) or ""
        status = f"{_RED}ERROR{_RESET}" if is_error else "ok"

        print(f"\n{_DIM}{'─' * 60}{_RESET}")
        user_str = f"  user={user}" if user else ""
        print(f"{color}{_BOLD}[AGENT]{_RESET} {color}{span.name}{_RESET}  {_DIM}{dur:.2f}s{_RESET}{_DIM}{user_str}{_RESET}  {status}")

        # Show input message
        inp = _parse_json(a.get(attrs.AGENTRACE_INPUT, ""))
        if isinstance(inp, dict):
            text = inp.get("text") or inp.get("message") or inp.get("question") or ""
            if not text:
                # Try first string value
                for v in inp.values():
                    if isinstance(v, str) and v:
                        text = v
                        break
        else:
            text = str(inp) if inp else ""
        if text:
            print(f"  {_DIM}input:{_RESET}  {_truncate(text)}")

        # Show output (truncated)
        out = a.get(attrs.AGENTRACE_OUTPUT, "")
        if out:
            print(f"  {_DIM}output:{_RESET} {_truncate(str(out))}")

        if is_error:
            print(f"  {_RED}error:  {span.status.description}{_RESET}")

    def _print_tool(self, span, a, color, dur, is_error):
        tool_name = a.get(attrs.AGENTRACE_TOOL_NAME, span.name)
        status = f"{_RED}FAIL{_RESET}" if is_error else "ok"

        print(f"  {color}{_BOLD}[TOOL]{_RESET} {color}{tool_name}{_RESET}  {_DIM}{dur:.3f}s{_RESET}  {status}")

        # Show input (cleaned)
        inp = _parse_json(a.get(attrs.AGENTRACE_TOOL_INPUT, ""))
        if isinstance(inp, dict) and inp:
            print(f"    {_DIM}in:{_RESET}  {_truncate(_clean_input(inp))}")
        elif inp:
            print(f"    {_DIM}in:{_RESET}  {_truncate(str(inp))}")

        # Show output
        out = _parse_json(a.get(attrs.AGENTRACE_TOOL_OUTPUT, ""))
        # Tool output is often double-serialized (json.dumps of a dict)
        if isinstance(out, str) and out:
            out = _parse_json(out)
        if isinstance(out, dict) and out:
            print(f"    {_DIM}out:{_RESET} {_truncate(json.dumps(out, ensure_ascii=False, default=str))}")
        elif out:
            print(f"    {_DIM}out:{_RESET} {_truncate(str(out))}")

        if is_error:
            print(f"    {_RED}err: {span.status.description}{_RESET}")

    def _print_llm(self, span, a, color, dur, is_error):
        model = a.get(attrs.GEN_AI_REQUEST_MODEL) or a.get(attrs.GEN_AI_RESPONSE_MODEL) or ""
        system = a.get("gen_ai.system", "")
        input_tokens = a.get(attrs.GEN_AI_USAGE_INPUT_TOKENS)
        output_tokens = a.get(attrs.GEN_AI_USAGE_OUTPUT_TOKENS)
        cost = a.get(attrs.AGENTRACE_COST_USD)

        parts = [f"  {color}{_BOLD}[LLM]{_RESET} {color}{span.name}{_RESET}  {_DIM}{dur:.2f}s{_RESET}"]
        if model:
            parts.append(f"  {_DIM}({model}){_RESET}")
        if input_tokens is not None or output_tokens is not None:
            parts.append(f"  {_DIM}tokens: {input_tokens or '?'}\u2192{output_tokens or '?'}{_RESET}")
        if cost is not None:
            parts.append(f"  {_DIM}${cost:.4f}{_RESET}")

        print("".join(parts))

        # Show completion text if available
        completion = a.get(attrs.AGENTRACE_COMPLETION, "")
        if completion:
            print(f"    {_DIM}response:{_RESET} {_truncate(str(completion))}")

        if is_error:
            print(f"    {_RED}err: {span.status.description}{_RESET}")

    def _print_generic(self, span, a, color, kind, dur, is_error):
        parts = [f"  {color}{_BOLD}[{kind.upper()}]{_RESET} {color}{span.name}{_RESET}  {_DIM}{dur:.2f}s{_RESET}"]

        # Show any input/output if present
        inp = a.get(attrs.AGENTRACE_INPUT, "")
        out = a.get(attrs.AGENTRACE_OUTPUT, "")

        print("".join(parts))
        if inp:
            print(f"    {_DIM}in:{_RESET}  {_truncate(str(inp))}")
        if out:
            print(f"    {_DIM}out:{_RESET} {_truncate(str(out))}")
        if is_error:
            print(f"    {_RED}err: {span.status.description}{_RESET}")
