"""POST /api/ask with retrieval and Claude replaced by fakes (no DB, no network)."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from alphalens_core.config import Settings, get_settings
from alphalens_core.embeddings import EmbeddingError
from alphalens_core.query_understanding import TickerResolution
from alphalens_core.retrieval import RetrievedChunk
from app.main import app
from app.routers.ask import get_claude, get_retriever, get_ticker_resolver


def _chunk(i: int, item: str = "1A") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=10 + i,
        content=f"Passage {i}.",
        context_header=f"[AAPL | 10-K | FY2026 | Item {item}]",
        ticker="AAPL",
        form_type="10-K",
        fiscal_year=2026,
        item_code=item,
        section_title="Risk Factors",
        accession_no="0000320193-26-000026",
        url="https://www.sec.gov/x.htm",
        score=0.03,
        vec_rank=i + 1,
        kw_rank=None,
    )


CHUNKS = [_chunk(0), _chunk(1), _chunk(2, "7")]


def _text(text: str, *cites: tuple[int, str]) -> SimpleNamespace:
    return SimpleNamespace(
        type="text",
        text=text,
        citations=[
            SimpleNamespace(type="char_location", document_index=d, cited_text=t) for d, t in cites
        ]
        or None,
    )


class FakeClaude:
    def __init__(self, content: list[Any]) -> None:
        self.calls = 0
        resp = SimpleNamespace(
            content=content,
            stop_reason="end_turn",
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=900, output_tokens=60),
        )

        def create(**_: Any) -> SimpleNamespace:
            self.calls += 1
            return resp

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


@pytest.fixture
def seen() -> dict[str, Any]:
    return {}


@pytest.fixture
def client(seen: dict[str, Any]) -> Iterator[TestClient]:
    def fake_retriever() -> Any:
        def retrieve(question: str, tickers: list[str] | None, k: int) -> list[RetrievedChunk]:
            seen.update(question=question, tickers=tickers, k=k)
            return CHUNKS[:k]

        return retrieve

    claude = FakeClaude(
        [
            _text("Summary. "),
            _text("Asia assembly", (2, "Passage 2."), (0, "Passage 0.")),
            _text(" and single sources.", (0, "Passage 0.")),
        ]
    )
    seen["claude"] = claude

    def fake_resolver() -> Any:
        def resolve(question: str) -> TickerResolution:
            seen["resolved_question"] = question
            return TickerResolution(
                tickers=["AAPL"], input_tokens=600, output_tokens=15, model="claude-haiku-5-5"
            )

        return resolve

    app.dependency_overrides[get_ticker_resolver] = fake_resolver
    app.dependency_overrides[get_retriever] = fake_retriever
    app.dependency_overrides[get_claude] = lambda: claude
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_returns_parts_and_deduplicated_sources(
    client: TestClient, seen: dict[str, Any]
) -> None:
    resp = client.post("/api/ask", json={"question": "What are the risks?", "tickers": ["AAPL"]})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == "Summary. Asia assembly and single sources."
    # Sources are the cited chunks only, numbered in first-cited order: chunk 2, then chunk 0.
    assert [s["chunk_id"] for s in body["sources"]] == [12, 10]
    assert body["sources"][0]["item_code"] == "7"
    assert body["parts"][0]["citations"] == []
    assert [c["source"] for c in body["parts"][1]["citations"]] == [0, 1]
    assert body["parts"][2]["citations"] == [{"source": 1, "cited_text": "Passage 0."}]
    assert (body["input_tokens"], body["output_tokens"]) == (900, 60)
    assert body["cost_usd"] == pytest.approx((900 * 4 + 60 * 20) / 1_000_000)
    assert seen["tickers"] == ["AAPL"]
    assert body["tickers"] == ["AAPL"]
    assert body["tickers_resolved"] is False  # caller gave tickers: no resolution call
    assert "resolved_question" not in seen
    assert seen["k"] == 8


@pytest.mark.parametrize(
    "payload",
    [
        {},  # question missing
        {"question": "hi"},  # too short
        {"question": "x" * 1001},  # too long
        {"question": "valid question", "k": 0},
        {"question": "valid question", "k": 21},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/api/ask", json=payload).status_code == 422


def test_embedding_failure_is_502(client: TestClient, seen: dict[str, Any]) -> None:
    def broken() -> Any:
        def retrieve(*_: Any) -> list[RetrievedChunk]:
            raise EmbeddingError("Voyage HTTP 429")

        return retrieve

    app.dependency_overrides[get_retriever] = broken
    resp = client.post("/api/ask", json={"question": "What are the risks?"})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "unavailable" in detail
    assert "429" not in detail  # upstream details stay in server logs
    assert seen["claude"].calls == 0  # never reached Claude


def _with_key(key: str) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(access_key=key)


@pytest.mark.parametrize("header", [None, "wrong-key", ""])
def test_access_key_required_when_configured(client: TestClient, header: str | None) -> None:
    _with_key("s3cret")
    headers = {} if header is None else {"X-Access-Key": header}
    resp = client.post("/api/ask", json={"question": "What are the risks?"}, headers=headers)
    assert resp.status_code == 401


def test_correct_access_key_is_accepted(client: TestClient) -> None:
    _with_key("s3cret")
    resp = client.post(
        "/api/ask", json={"question": "What are the risks?"}, headers={"X-Access-Key": "s3cret"}
    )
    assert resp.status_code == 200


def test_health_endpoints_stay_open(client: TestClient) -> None:
    _with_key("s3cret")
    assert client.get("/healthz").status_code == 200


def test_tickers_are_resolved_when_not_given(client: TestClient, seen: dict[str, Any]) -> None:
    resp = client.post("/api/ask", json={"question": "How many people does Apple employ?"})
    body = resp.json()
    assert resp.status_code == 200
    assert seen["resolved_question"] == "How many people does Apple employ?"
    assert seen["tickers"] == ["AAPL"]  # resolved tickers drive retrieval
    assert body["tickers"] == ["AAPL"]
    assert body["tickers_resolved"] is True
    # Cost includes the Haiku resolution call: 600 in / 15 out at $0.10 / $0.50 per 1M.
    answer_cost = (900 * 4 + 60 * 20) / 1_000_000
    assert body["cost_usd"] == pytest.approx(answer_cost + (600 * 0.10 + 15 * 0.50) / 1_000_000)
