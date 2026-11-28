"""run_agent with a scripted fake Claude and fake tools (no network, no database)."""

from types import SimpleNamespace
from typing import Any

import pytest

from alphalens_core.agent import loop as loop_module
from alphalens_core.agent.loop import run_agent
from alphalens_core.agent.tools import ToolContext, ToolOutput, chunk_source
from alphalens_core.retrieval import RetrievedChunk

CHUNK = RetrievedChunk(
    chunk_id=7,
    content="Microsoft Cloud gross margin percentage decreased to 66%.",
    context_header="[MSFT | 10-K | FY2026 | Item 7]",
    ticker="MSFT",
    form_type="10-K",
    fiscal_year=2026,
    item_code="7",
    section_title=None,
    accession_no="x",
    url="https://www.sec.gov/msft.htm",
    score=0.0,
    vec_rank=1,
    kw_rank=None,
)


def _usage(i: int = 100, o: int = 10, write: int = 0, read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=i,
        output_tokens=o,
        cache_creation_input_tokens=write,
        cache_read_input_tokens=read,
    )


def _tool_use(id_: str, name: str, inp: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _resp(content: list[Any], stop: str, usage: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, stop_reason=stop, model="claude-haiku-5-5", usage=usage or _usage()
    )


class ScriptedClaude:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot: the loop keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    context = ToolContext(session=None, embedder=None, reranker=None)  # type: ignore[arg-type]

    def fake_run_tool(c: ToolContext, name: str, tool_input: dict[str, Any]) -> ToolOutput:
        if name == "search_filings":
            c.chunks[CHUNK.chunk_id] = CHUNK
            return ToolOutput(
                [{"type": "search_result", "source": chunk_source(CHUNK)}], summary="1 passage"
            )
        if tool_input.get("sql") == "bad":
            return ToolOutput("Query rejected: only SELECT", is_error=True, summary="SQL rejected")
        return ToolOutput('{"rows": [[0.36]]}', summary="SQL: 1 rows")

    monkeypatch.setattr(loop_module, "run_tool", fake_run_tool)
    return context


def test_tools_then_answer_with_citation(ctx: ToolContext) -> None:
    citation = SimpleNamespace(
        type="search_result_location",
        source=chunk_source(CHUNK),
        cited_text="Microsoft Cloud gross margin percentage decreased to 66%.",
    )
    claude = ScriptedClaude(
        [
            _resp(
                [
                    SimpleNamespace(type="thinking"),
                    _tool_use("t1", "query_financials", {"sql": "SELECT 1"}),
                    _tool_use("t2", "search_filings", {"query": "cloud margin"}),
                ],
                "tool_use",
                _usage(write=2000),
            ),
            _resp(
                [
                    SimpleNamespace(type="text", text="Margin fell", citations=[citation]),
                    SimpleNamespace(type="text", text=".", citations=None),
                ],
                "end_turn",
                _usage(read=2000),
            ),
        ]
    )
    result = run_agent(claude, "claude-haiku-5-5", "why?", ctx, {"MSFT": "Microsoft"})  # type: ignore[arg-type]

    # Both tool results return together in one user message, after the assistant turn.
    second = claude.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert [r["tool_use_id"] for r in second[2]["content"]] == ["t1", "t2"]
    assert second[1]["content"][0].type == "thinking"  # assistant turn sent back unchanged

    assert result.text == "Margin fell."
    assert result.parts[0].citations[0].chunk.chunk_id == 7
    assert [c.name for c in result.tool_calls] == ["query_financials", "search_filings"]
    assert "MSFT: Microsoft" in claude.calls[0]["system"]
    # 200 in + 2000 cache write (1.25x) + 2000 cache read (0.1x) at $0.10/M, 20 out at $0.50/M
    assert result.cost_usd == pytest.approx(((200 + 2500 + 200) * 0.10 + 20 * 0.50) / 1e6)


def test_sql_errors_go_back_as_error_results(ctx: ToolContext) -> None:
    claude = ScriptedClaude(
        [
            _resp([_tool_use("t1", "query_financials", {"sql": "bad"})], "tool_use"),
            _resp([SimpleNamespace(type="text", text="ok", citations=None)], "end_turn"),
        ]
    )
    result = run_agent(claude, "m", "q", ctx, {})  # type: ignore[arg-type]
    tool_result = claude.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "only SELECT" in tool_result["content"]
    assert result.tool_calls[0].is_error


def test_tool_limit_turns_tools_off(ctx: ToolContext) -> None:
    looping = [
        _resp([_tool_use(f"t{i}", "query_financials", {"sql": "SELECT 1"})], "tool_use")
        for i in range(3)
    ]
    final = _resp([SimpleNamespace(type="text", text="done", citations=None)], "end_turn")
    claude = ScriptedClaude([*looping, final])
    result = run_agent(claude, "m", "q", ctx, {}, max_tool_calls=2)  # type: ignore[arg-type]

    assert result.hit_tool_limit
    assert [c.is_error for c in result.tool_calls] == [False, False, True]
    assert claude.calls[-1]["tool_choice"] == {"type": "none"}
    assert result.text == "done"


def test_refusal_without_text(ctx: ToolContext) -> None:
    claude = ScriptedClaude([_resp([], "refusal")])
    result = run_agent(claude, "m", "q", ctx, {})  # type: ignore[arg-type]
    assert "declined" in result.text
