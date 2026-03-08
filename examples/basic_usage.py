"""Basic agentrace usage example - no external LLM SDKs needed."""

import time
import agentrace
from agentrace import observe, trace_llm, trace_tool, trace_agent, set_session, set_user
from agentrace.context import trace_span, trace_llm_call

# 1. Initialize (console output by default)
agentrace.init(service_name="demo-agent")

# 2. Set session/user context
set_session("session-001")
set_user("user-42")


# 3. Decorate your functions
@trace_tool(tool_name="web_search")
def search_web(query: str) -> list[str]:
    """Simulate a web search."""
    time.sleep(0.1)  # Simulate latency
    return [f"Result for '{query}': page 1", f"Result for '{query}': page 2"]


@trace_tool(tool_name="calculator")
def calculate(expression: str) -> float:
    """Simulate a calculator tool."""
    time.sleep(0.05)
    return eval(expression)  # Don't do this in production!


@observe(kind="chain", name="summarize-chain")
def summarize(text: str) -> str:
    """Simulate a summarization chain."""
    time.sleep(0.05)
    return f"Summary of: {text[:50]}..."


@trace_agent(name="research-agent")
def run_agent(question: str) -> str:
    """Run a simple research agent."""
    # Step 1: Search
    results = search_web(question)

    # Step 2: Summarize results
    context = "\n".join(results)
    summary = summarize(context)

    # Step 3: Calculate something
    answer = calculate("2 + 2")

    # Step 4: Manual span with context manager
    with trace_span("format-output", kind="chain") as span:
        span.set_attribute("custom.result_count", len(results))
        time.sleep(0.02)
        final = f"Answer to '{question}': {summary} (calculated: {answer})"

    return final


# 4. Run it!
if __name__ == "__main__":
    print("=== Running agentrace demo ===\n")
    result = run_agent("What is OpenTelemetry?")
    print(f"\nFinal result: {result}")

    # Flush traces
    agentrace.shutdown()
