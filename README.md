# agentrace-ai

[![PyPI version](https://badge.fury.io/py/agentrace-ai.svg)](https://pypi.org/project/agentrace-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Lightweight, OpenTelemetry-native tracing library for LLM and AI agent applications. Think of it as a drop-in observability layer for your agents — like Langfuse or Arize Phoenix, but as a simple `pip install` with zero infrastructure.

```bash
pip install agentrace-ai
```

---

## Why agentrace-ai?

| | agentrace-ai | Full platforms (Langfuse, Phoenix) |
|---|---|---|
| **Install** | `pip install agentrace-ai` | Docker + DB + UI server |
| **Setup** | 2 lines of code | Config files, env vars, accounts |
| **Dependencies** | 2 (otel-api + otel-sdk) | 100+ packages |
| **How it works** | Import as a Python module | Deploy as a separate service |
| **Lock-in** | Zero — pure OpenTelemetry spans | Vendor-specific schemas |
| **Export** | Console, OTLP, file, or custom | Platform-specific |

agentrace-ai gives you **tracing you own** — every span is a standard OpenTelemetry span that works with Jaeger, Grafana Tempo, Datadog, or any OTLP-compatible backend.

---

## Quick Start

```python
import agentrace
from agentrace import trace_llm, trace_tool, trace_agent

# Initialize (console output by default)
agentrace.init(service_name="my-agent")

@trace_tool(tool_name="web_search")
def search(query):
    return ["result 1", "result 2"]

@trace_llm(model="gpt-4o")
def call_llm(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

@trace_agent(name="research-agent")
def run(question):
    results = search(question)
    return call_llm([{"role": "user", "content": str(results)}])

run("What is OpenTelemetry?")
agentrace.shutdown()
```

**Output:**
```
[TOOL]  web_search       0.10s  [web_search]   trace=a37c24f1...
[LLM]   call_llm         1.20s  tokens: 150->320  $0.0044  (gpt-4o)  trace=a37c24f1...
[AGENT] research-agent   1.35s  trace=a37c24f1...
```

---

## Installation

```bash
# Core (console tracing)
pip install agentrace-ai

# With integrations
pip install agentrace-ai[openai]       # Auto-trace OpenAI calls
pip install agentrace-ai[anthropic]    # Auto-trace Anthropic calls
pip install agentrace-ai[langchain]    # LangChain/LangGraph callback handler
pip install agentrace-ai[otlp]        # Export to Jaeger, Grafana Tempo, etc.
pip install agentrace-ai[all]         # Everything
```

---

## Core API

### Decorators

The primary way to add tracing. Works with both sync and async functions.

```python
from agentrace import observe, trace_llm, trace_tool, trace_agent, trace_chain, trace_retrieval

# General purpose — trace any function
@observe()
def process(data):
    return transform(data)

# LLM calls — auto-extracts tokens, cost, model from response
@trace_llm(model="gpt-4o")
def call_openai(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

# Tool/function calls in agent workflows
@trace_tool(tool_name="calculator")
def calculate(expression):
    return eval(expression)

# Agent entry points — creates parent span for the run
@trace_agent(name="my-agent")
async def run_agent(task):
    ...

# Pipeline/chain steps
@trace_chain(name="rag-pipeline")
def rag(query):
    ...

# Retrieval operations (vector search, document fetch)
@trace_retrieval(name="vector-search")
def retrieve(query, top_k=5):
    ...
```

### Context Managers

For tracing code blocks where decorators don't fit.

```python
from agentrace.context import trace_span, trace_llm_call

# Generic span
with trace_span("preprocessing", kind="chain") as span:
    span.set_attribute("custom.key", "value")
    result = do_work()

# LLM span with helper methods
with trace_llm_call("summarize", model="gpt-4o") as llm:
    response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    llm.record_response(response)   # Auto-extracts tokens, cost, completion
    # Or manually:
    # llm.record_usage(input_tokens=100, output_tokens=50)
    # llm.record_messages(msgs)
```

### Session & User Tracking

Track which user and session each trace belongs to.

```python
from agentrace import set_session, set_user
from agentrace.session import session_context

# Set globally — all subsequent spans inherit these
set_session("session-abc-123")
set_user("user-42")

# Also works inside decorated functions — session/user is captured before span ends
@trace_agent(name="my-agent")
async def run(user_id, task):
    set_user(user_id)  # This is correctly attached to the agent span
    return await do_work(task)

# Or scoped to a block
with session_context(session_id="s-123", user_id="u-42"):
    run_agent(...)  # All spans inside get session/user attributes
```

### Cost Tracking

Automatic cost calculation for OpenAI and Anthropic models.

```python
from agentrace import calculate_cost, register_model_pricing

# Built-in pricing for: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo,
# o1, o1-mini, claude-opus-4, claude-sonnet-4, claude-haiku-4, and more.

cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
# Returns: 0.0075 (USD)

# Register custom model pricing
register_model_pricing("my-fine-tuned-model", input_cost_per_1k=0.005, output_cost_per_1k=0.015)
```

---

## Integrations

### OpenAI — Auto-Instrumentation

Automatically trace every `chat.completions.create` call with zero code changes.

```python
import agentrace
from agentrace.integrations import openai_patch

agentrace.init(service_name="my-app")
openai_patch.instrument()   # Patches OpenAI SDK globally

from openai import OpenAI
client = OpenAI()

# This call is now automatically traced with:
# - Model name, token usage, cost
# - Prompt messages and completion text
# - Latency and error tracking
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# To remove instrumentation:
# openai_patch.uninstrument()
```

### Anthropic — Auto-Instrumentation

```python
from agentrace.integrations import anthropic_patch

anthropic_patch.instrument()

import anthropic
client = anthropic.Anthropic()

# Automatically traced
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### LangChain / LangGraph

```python
from agentrace.integrations.langchain_cb import AgentraceCallbackHandler

handler = AgentraceCallbackHandler()

# Works with any LangChain chain, agent, or LangGraph graph
result = chain.invoke(
    {"input": "What is AI?"},
    config={"callbacks": [handler]},
)
```

The callback handler automatically traces:
- LLM calls (`on_llm_start/end`) with token usage
- Chain steps (`on_chain_start/end`) with inputs/outputs
- Tool calls (`on_tool_start/end`) with tool name, input, output
- Retriever calls (`on_retriever_start/end`) with query and document count

---

## Exporters

### Console (default)

Color-coded, human-readable output for development.

```python
agentrace.init(exporters=["console"])
```

### OTLP (Jaeger, Grafana Tempo, Datadog, etc.)

Export to any OpenTelemetry-compatible backend.

```python
agentrace.init(
    service_name="my-agent",
    exporters=["otlp"],
    otlp_endpoint="http://localhost:4317",          # gRPC
    otlp_headers={"Authorization": "Bearer xxx"},   # Optional auth
)
```

### Console + OTLP together

```python
agentrace.init(exporters=["console", "otlp"], otlp_endpoint="http://localhost:4317")
```

### Custom Exporter

Build your own — write spans to a database, webhook, file, or anywhere.

```python
from agentrace.exporters.custom import AgentraceExporter
from opentelemetry.sdk.trace.export import SpanExportResult

class WebhookExporter(AgentraceExporter):
    def __init__(self, url):
        self.url = url

    def export(self, spans):
        for span in spans:
            data = self.span_to_dict(span)  # Built-in helper
            requests.post(self.url, json=data)
        return SpanExportResult.SUCCESS

# Use it
agentrace.init(exporters=["console", WebhookExporter("https://my-webhook.com/traces")])
```

`span_to_dict()` returns:
```json
{
  "name": "call_llm",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "parent_span_id": "ghi789...",
  "duration_ms": 1200.5,
  "status": "OK",
  "attributes": {
    "agentrace.span.kind": "llm",
    "gen_ai.request.model": "gpt-4o",
    "gen_ai.usage.input_tokens": 150,
    "gen_ai.usage.output_tokens": 320,
    "agentrace.cost.usd": 0.0044
  }
}
```

---

## Configuration Reference

```python
agentrace.init(
    service_name="my-agent",              # Service name in traces
    exporters=["console"],                # "console", "otlp", or SpanExporter instances
    otlp_endpoint=None,                   # OTLP collector URL
    otlp_headers=None,                    # Auth headers for OTLP
    capture_prompts=True,                 # Record prompt/completion text on LLM spans
    capture_tool_io=True,                 # Record tool input/output
    default_session_id=None,              # Default session ID for all spans
    default_user_id=None,                 # Default user ID for all spans
    resource_attributes=None,             # Extra OTel resource attributes
    batch=False,                          # True = BatchSpanProcessor, False = SimpleSpanProcessor
    provider=None,                        # Bring your own pre-built TracerProvider
)
```

---

## Span Attributes

agentrace records these attributes on spans, following [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| Attribute | Description | Example |
|-----------|-------------|---------|
| `agentrace.span.kind` | Span type | `"llm"`, `"tool"`, `"agent"`, `"chain"`, `"retrieval"` |
| `gen_ai.system` | LLM provider | `"openai"`, `"anthropic"` |
| `gen_ai.request.model` | Requested model | `"gpt-4o"` |
| `gen_ai.response.model` | Response model | `"gpt-4o-2024-08-06"` |
| `gen_ai.usage.input_tokens` | Input token count | `150` |
| `gen_ai.usage.output_tokens` | Output token count | `320` |
| `agentrace.cost.usd` | Calculated cost | `0.0044` |
| `agentrace.prompt` | Input messages (JSON) | `[{"role": "user", ...}]` |
| `agentrace.completion` | Output text | `"Hello! How can I help?"` |
| `agentrace.tool.name` | Tool name | `"web_search"` |
| `agentrace.tool.input` | Tool input | `{"query": "..."}` |
| `agentrace.tool.output` | Tool output | `["result1", "result2"]` |
| `agentrace.session.id` | Session identifier | `"session-abc-123"` |
| `agentrace.user.id` | User identifier | `"user-42"` |
| `agentrace.input` | Function arguments | `{"x": 1, "y": 2}` |
| `agentrace.output` | Function return value | `3` |

---

## Real-World Example: Health Agent (Arogya)

Here's how agentrace was integrated into [Arogya](https://github.com/padkri/arogya), a WhatsApp-based health tracking agent using LangGraph + Gemini:

```python
# src/tracing.py
from agentrace import observe, trace_tool, set_session
from agentrace.context import trace_span

def init_tracing(log_file="traces.log"):
    agentrace.init(service_name="arogya", exporters=["console"])

    # Trace all tools
    from src.tools import tools
    for name in ["log_weight", "log_meal", "log_water", "log_workout"]:
        original = getattr(tools, name)
        setattr(tools, name, trace_tool(tool_name=name)(original))

    # Trace agent runs
    from src.agent import graph
    original_run = graph.run_agent

    async def traced_run(user_id, text, media_url=None):
        set_session(user_id)
        with trace_span("agent.tracking", kind="agent") as span:
            span.set_attribute("agentrace.input", text)
            return await original_run(user_id, text, media_url)

    graph.run_agent = traced_run
```

**Result: 35 spans captured** across onboarding + tracking:
```
[AGENT] agent.onboarding  0.74s  {session=user-123}  trace=be3b5f85...
[TOOL]  register_user     0.03s  [register_user]     trace=bbf62a95...
[AGENT] agent.tracking    2.15s  {session=user-123}  trace=74f8a5a1...
[TOOL]  log_weight        0.01s  [log_weight]        trace=74f8a5a1...
[TOOL]  log_meal          0.01s  [log_meal]           trace=66c328aa...
[TOOL]  get_body_metrics  0.00s  [get_body_metrics]  trace=9c3226f4...
```

---

## API Reference

### Top-Level Functions

| Function | Description |
|----------|-------------|
| `agentrace.init(...)` | Initialize tracing. Call once at startup. |
| `agentrace.shutdown()` | Flush and shut down the tracer. |

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@observe(name, kind, capture_input, capture_output)` | General-purpose tracing |
| `@trace_llm(model, name, capture_prompts)` | LLM calls with auto token/cost extraction |
| `@trace_tool(tool_name, name, capture_io)` | Tool/function calls |
| `@trace_agent(name)` | Agent entry points |
| `@trace_chain(name)` | Chain/pipeline steps |
| `@trace_retrieval(name)` | Retrieval operations |

### Context Managers

| Context Manager | Description |
|----------------|-------------|
| `trace_span(name, kind, **attrs)` | Generic span block |
| `trace_llm_call(name, model)` | LLM span with `record_response()`, `record_usage()` |

### Session/User

| Function | Description |
|----------|-------------|
| `set_session(session_id)` | Set session ID for subsequent spans |
| `set_user(user_id)` | Set user ID for subsequent spans |
| `session_context(session_id, user_id)` | Scoped session/user context manager |

### Cost

| Function | Description |
|----------|-------------|
| `calculate_cost(model, input_tokens, output_tokens)` | Calculate USD cost |
| `register_model_pricing(model, input_cost, output_cost)` | Add custom model pricing |

---

## License

MIT

---

Built with OpenTelemetry. Works with any OTLP-compatible backend.
