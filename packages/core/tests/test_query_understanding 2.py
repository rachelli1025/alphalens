"""resolve_tickers with a fake Anthropic client (no network)."""

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from alphalens_core.query_understanding import resolve_tickers

COMPANIES = {"AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms, Inc."}


class FakeClient:
    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []

        def create(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            if error:
                raise error
            text = reply if isinstance(reply, str) else json.dumps(reply)
            return SimpleNamespace(
                content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=600, output_tokens=15),
                model="claude-haiku-5-5",
            )

        self.messages = SimpleNamespace(create=create)


def test_returns_known_tickers_and_cost() -> None:
    client = FakeClient({"tickers": ["GOOGL", "META", "GOOGL"]})
    r = resolve_tickers(client, "claude-haiku-5-5", "Google vs Facebook?", COMPANIES)  # type: ignore[arg-type]
    assert r.tickers == ["GOOGL", "META"]  # deduped, order kept
    assert r.cost_usd == pytest.approx((600 * 0.10 + 15 * 0.50) / 1_000_000)


def test_schema_restricts_to_known_tickers_and_prompt_lists_names() -> None:
    client = FakeClient({"tickers": []})
    resolve_tickers(client, "claude-haiku-5-5", "q", COMPANIES)  # type: ignore[arg-type]
    call = client.calls[0]
    schema = call["output_config"]["format"]["schema"]
    assert schema["properties"]["tickers"]["items"]["enum"] == ["AAPL", "GOOGL", "META"]
    assert "GOOGL: Alphabet Inc." in call["messages"][0]["content"]


def test_unknown_tickers_are_dropped() -> None:
    client = FakeClient({"tickers": ["AAPL", "ZZZZ"]})
    r = resolve_tickers(client, "claude-haiku-5-5", "q", COMPANIES)  # type: ignore[arg-type]
    assert r.tickers == ["AAPL"]


@pytest.mark.parametrize(
    "client",
    [
        FakeClient("not json"),
        FakeClient({"wrong_key": []}),
        FakeClient(
            error=anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )
        ),
    ],
)
def test_failures_fall_back_to_no_filter(client: FakeClient) -> None:
    r = resolve_tickers(client, "claude-haiku-5-5", "q", COMPANIES)  # type: ignore[arg-type]
    assert r.tickers == []


def test_no_companies_means_no_call() -> None:
    client = FakeClient({"tickers": []})
    assert resolve_tickers(client, "m", "q", {}).tickers == []  # type: ignore[arg-type]
    assert client.calls == []
