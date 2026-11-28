"""Claude list prices (USD per 1M tokens, Anthropic first-party API) and cost math.

Prices change; update this table when Anthropic's pricing page does. Unknown
models return None so callers show "unknown" instead of a wrong number.
"""

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens; ours are ~10K
}

# Models that accept server-side refusal fallbacks (`fallbacks="default"`).
# Claude Haiku 5.5 does not: sending the parameter to it is rejected.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
