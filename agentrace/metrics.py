"""Token counting and cost calculation utilities."""

from __future__ import annotations

_COST_PER_1K: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    # Anthropic
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-20250414": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
}


def register_model_pricing(
    model: str, input_cost_per_1k: float, output_cost_per_1k: float
) -> None:
    """Register or update pricing for a model."""
    _COST_PER_1K[model] = {"input": input_cost_per_1k, "output": output_cost_per_1k}


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Calculate USD cost for a model call. Returns None if model pricing unknown."""
    pricing = _COST_PER_1K.get(model)
    if pricing is None:
        # Try prefix matching for versioned model names
        for key, val in _COST_PER_1K.items():
            if model.startswith(key):
                pricing = val
                break
    if pricing is None:
        return None
    return (input_tokens / 1000 * pricing["input"]) + (
        output_tokens / 1000 * pricing["output"]
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate using word-based heuristic (~0.75 words per token)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) / 0.75))
