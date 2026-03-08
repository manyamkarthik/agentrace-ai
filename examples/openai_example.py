"""Example: Tracing OpenAI calls with agentrace.

Requires: pip install agentrace[openai]
Set OPENAI_API_KEY environment variable before running.
"""

import agentrace
from agentrace import trace_llm, trace_tool, trace_agent, set_session

# Initialize
agentrace.init(service_name="openai-agent")
set_session("demo-session")


# Option 1: Auto-instrument all OpenAI calls
from agentrace.integrations import openai_patch
openai_patch.instrument()

from openai import OpenAI
client = OpenAI()


# Option 2: Or use decorators for specific functions
@trace_tool(tool_name="get_weather")
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72F, sunny"


@trace_agent(name="weather-agent")
def weather_agent(question: str) -> str:
    # This OpenAI call is auto-traced by the patch
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": question},
        ],
    )

    # Check if we need to call a tool
    answer = response.choices[0].message.content
    if "weather" in question.lower():
        weather = get_weather("San Francisco")
        return f"{answer}\n\nActual weather: {weather}"

    return answer


if __name__ == "__main__":
    result = weather_agent("What's the weather like in San Francisco?")
    print(f"\nResult: {result}")
    agentrace.shutdown()
