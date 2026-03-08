"""Semantic attribute constants for LLM/agent tracing.

Follows OpenTelemetry GenAI semantic conventions where they exist,
extends with agentrace.* keys where they don't.
"""

# --- OpenTelemetry GenAI semantic conventions ---
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"

# --- agentrace extensions ---
AGENTRACE_SPAN_KIND = "agentrace.span.kind"
AGENTRACE_PROMPT = "agentrace.prompt"
AGENTRACE_COMPLETION = "agentrace.completion"
AGENTRACE_COST_USD = "agentrace.cost.usd"
AGENTRACE_TOOL_NAME = "agentrace.tool.name"
AGENTRACE_TOOL_INPUT = "agentrace.tool.input"
AGENTRACE_TOOL_OUTPUT = "agentrace.tool.output"
AGENTRACE_AGENT_NAME = "agentrace.agent.name"
AGENTRACE_AGENT_STEP = "agentrace.agent.step"
AGENTRACE_SESSION_ID = "agentrace.session.id"
AGENTRACE_USER_ID = "agentrace.user.id"
AGENTRACE_RETRIEVAL_QUERY = "agentrace.retrieval.query"
AGENTRACE_RETRIEVAL_NUM_DOCS = "agentrace.retrieval.num_documents"
AGENTRACE_INPUT = "agentrace.input"
AGENTRACE_OUTPUT = "agentrace.output"
