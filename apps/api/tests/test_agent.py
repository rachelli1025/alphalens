"""POST /api/agent with the agent replaced by scripted events (no DB, no network)."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from alphalens_core.agent.loop import AgentResult, Done, TextDelta, ToolCall, ToolEnd, ToolStart
from alphalens_core.answer import AnswerPart, Citation
from alphalens_core.config import Settings, get_settings
from alphalens_core.retrieval import RetrievedChunk
from app.main import app
from app.routers.agent import get_agent_runner

CHUNK = RetrievedChunk(
    chunk_id=7,
    content="Export controls limit sales to China.",
    context_header="[NVDA | 10-K | FY2026 | Item 1A]",
    ticker="NVDA",
    form_type="10-K",
    fiscal_year=2026,
    item_code="1A",
    section_title="Risk Factors",
    accession_no="x",
    url="https://www.sec.gov/nvda.htm",
    score=0.0,
    vec_rank=1,
    kw_rank=None,
)


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def scripted(question: str) -> Iterator[Any]:
    call = ToolCall(
        "query_financials", {"sql": "SELECT 1"}, "SQL: 1 rows", False, "SELECT 1 LIMIT 500"
    )
    yield TextDelta("Let me check.")
    yield ToolStart("t1", "query_financials", {"sql": "SELECT 1"})
    yield ToolEnd("t1", call)
    yield TextDelta("Risk: ")
    yield TextDelta("exports.")
    parts = [
        AnswerPart("Risk: "),
        AnswerPart("exports.", [Citation(CHUNK, "Export controls limit sales to China.")]),
    ]
    yield Done(
        AgentResult(
            parts=parts,
            tool_calls=[call],
            sql_queries=["SELECT 1 LIMIT 500"],
            model="claude-haiku-5-5",
            input_tokens=1000,
            output_tokens=100,
        )
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_agent_runner] = lambda: scripted
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_agent_streams_tool_status_text_and_done(client: TestClient) -> None:
    resp = client.post("/api/agent", json={"question": "Nvidia risks?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    assert [e for e, _ in events] == ["text", "tool_start", "tool_end", "text", "text", "done"]
    assert events[1][1] == {"id": "t1", "name": "query_financials", "input": {"sql": "SELECT 1"}}
    assert events[2][1]["sql"] == "SELECT 1 LIMIT 500"

    done = events[-1][1]
    assert done["answer"] == "Risk: exports."
    assert [s["chunk_id"] for s in done["sources"]] == [7]
    assert done["parts"][1]["citations"] == [
        {"source": 0, "cited_text": "Export controls limit sales to China."}
    ]
    assert done["tool_calls"][0]["summary"] == "SQL: 1 rows"
    assert done["cost_usd"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


def test_failures_mid_stream_become_an_error_event(client: TestClient) -> None:
    def failing(question: str) -> Iterator[Any]:
        yield ToolStart("t1", "search_filings", {"query": "x"})
        raise RuntimeError("Voyage said: account 1234 over quota")

    app.dependency_overrides[get_agent_runner] = lambda: failing
    events = parse_sse(client.post("/api/agent", json={"question": "Nvidia risks?"}).text)
    assert [e for e, _ in events] == ["tool_start", "error"]
    assert "1234" not in events[-1][1]["message"]  # provider details stay in the logs


def test_agent_requires_the_access_key_when_set(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(access_key="secret")
    assert client.post("/api/agent", json={"question": "Nvidia risks?"}).status_code == 401
    ok = client.post(
        "/api/agent", json={"question": "Nvidia risks?"}, headers={"X-Access-Key": "secret"}
    )
    assert ok.status_code == 200


def test_short_questions_are_rejected(client: TestClient) -> None:
    assert client.post("/api/agent", json={"question": "hi"}).status_code == 422
