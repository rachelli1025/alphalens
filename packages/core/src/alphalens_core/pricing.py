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


# Prompt caching multipliers on the input price (5-minute cache).
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    """`input_tokens` excludes cached tokens, as in the API's usage block."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    billed_input = (
        input_tokens
        + cache_write_tokens * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * CACHE_READ_MULTIPLIER
    )
    return (billed_input * price_in + output_tokens * price_out) / 1_000_000
