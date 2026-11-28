"""Query understanding: which companies is a question about? (DESIGN.md §8, first step)

Users write "Google", "Facebook" or "JP Morgan", not tickers, so a small model maps
the question onto the companies we have filings for. Structured outputs constrain the
answer to an enum of known tickers, so the model cannot invent one.

Failure is non-fatal: on any API error we return no tickers and retrieval searches
every company, which is worse but still works.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import anthropic
from sqlalchemy import select
from sqlalchemy.orm import Session

from alphalens_core.db.models import Company
from alphalens_core.pricing import cost_usd

log = logging.getLogger(__name__)

PROMPT = """\
Which of these companies is the question about?

Companies (ticker: name):
{companies}

Question: {question}

Rules:
- Return the tickers of companies the question asks about, including when it uses a \
common name, brand or former name (for example Google means Alphabet, Facebook means Meta).
- Return several tickers if the question compares or lists specific companies.
- Return an empty list if the question names no specific company from the list."""


@dataclass(frozen=True)
class TickerResolution:
    tickers: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def cost_usd(self) -> float | None:
        if not self.model:
            return 0.0
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


def list_companies(session: Session) -> dict[str, str]:
    """ticker -> company name, for every company with ingested filings."""
    rows = session.execute(select(Company.ticker, Company.name).order_by(Company.ticker)).all()
    return {ticker: name for ticker, name in rows}


def _schema(tickers: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tickers": {"type": "array", "items": {"type": "string", "enum": tickers}},
        },
        "required": ["tickers"],
        "additionalProperties": False,
    }


def resolve_tickers(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    companies: dict[str, str],
) -> TickerResolution:
    if not companies:
        return TickerResolution(tickers=[])

    listing = "\n".join(f"{t}: {name}" for t, name in companies.items())
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _schema(list(companies))},
            },
            messages=[
                {"role": "user", "content": PROMPT.format(companies=listing, question=question)}
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        raw = json.loads(text)["tickers"]
    except (anthropic.APIError, StopIteration, json.JSONDecodeError, KeyError, TypeError):
        log.exception("ticker resolution failed; searching all companies")
        return TickerResolution(tickers=[])

    # The enum already restricts values; dedupe and keep only known tickers anyway.
    tickers = list(dict.fromkeys(t for t in raw if t in companies))
    return TickerResolution(
        tickers=tickers,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=response.model,
    )
