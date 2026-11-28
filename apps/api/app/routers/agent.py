"""POST /api/agent: the research agent, streamed as Server-Sent Events.

The response is a text/event-stream; each event is `event: <name>` plus one JSON `data:`
line:

    tool_start  {id, name, input}                 the agent started a tool call
    tool_end    {id, name, summary, is_error, sql} it finished (sql: executed statement)
    text        {text}                            a piece of the model's text
    done        AgentDone                         final answer with citations and cost
    error       {message}                         something failed; the stream ends

Text that arrives before a tool_start is the model's preamble for that turn; clients
should drop it at tool_start. The answer is the text after the last tool call, and
`done` repeats it in full with citations.

Stateless for now: one question per request. Conversations with history come later.
"""

import json
import logging
from collections.abc import Callable, Iterator
from typing import Annotated, Any

import anthropic
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from alphalens_core.agent.loop import AgentEvent, Done, TextDelta, ToolEnd, ToolStart, stream_agent
from alphalens_core.agent.tools import ToolContext
from alphalens_core.config import get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import VoyageEmbedder
from alphalens_core.query_understanding import list_companies
from alphalens_core.reranking import VoyageReranker
from app.routers.ask import (
    PartOut,
    SourceOut,
    get_claude,
    get_embedder,
    get_reranker,
    require_access_key,
    serialize_parts,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"], dependencies=[Depends(require_access_key)])


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class ToolCallOut(BaseModel):
    name: str
    input: dict[str, Any]
    summary: str
    is_error: bool
    sql: str | None


class AgentDone(BaseModel):
    answer: str
    parts: list[PartOut]
    sources: list[SourceOut]
    tool_calls: list[ToolCallOut]
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None
    hit_tool_limit: bool


AgentRunner = Callable[[str], Iterator[AgentEvent]]


def get_agent_runner(
    claude: Annotated[anthropic.Anthropic, Depends(get_claude)],
    embedder: Annotated[VoyageEmbedder, Depends(get_embedder)],
    reranker: Annotated[VoyageReranker, Depends(get_reranker)],
) -> AgentRunner:
    def run(question: str) -> Iterator[AgentEvent]:
        # The session lives inside the generator: it must stay open while the response
        # streams, which is after the endpoint function has returned.
        settings = get_settings()
        with get_sync_sessionmaker()() as session:
            ctx = ToolContext(session=session, embedder=embedder, reranker=reranker)
            yield from stream_agent(
                claude,
                settings.agent_model,
                question,
                ctx,
                list_companies(session),
                effort=settings.agent_effort,
            )

    return run


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _done(event: Done) -> AgentDone:
    r = event.result
    parts, sources = serialize_parts(r.parts)
    return AgentDone(
        answer=r.text,
        parts=parts,
        sources=sources,
        tool_calls=[ToolCallOut(**vars(c)) for c in r.tool_calls],
        model=r.model,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cache_read_tokens=r.cache_read_tokens,
        cache_write_tokens=r.cache_write_tokens,
        cost_usd=r.cost_usd,
        hit_tool_limit=r.hit_tool_limit,
    )


def event_stream(events: Iterator[AgentEvent]) -> Iterator[str]:
    try:
        for event in events:
            if isinstance(event, TextDelta):
                yield _sse("text", {"text": event.text})
            elif isinstance(event, ToolStart):
                yield _sse("tool_start", {"id": event.id, "name": event.name, "input": event.input})
            elif isinstance(event, ToolEnd):
                c = event.call
                yield _sse(
                    "tool_end",
                    {
                        "id": event.id,
                        "name": c.name,
                        "summary": c.summary,
                        "is_error": c.is_error,
                        "sql": c.sql,
                    },
                )
            elif isinstance(event, Done):
                yield _sse("done", _done(event).model_dump())
    except Exception:
        # The 200 status is already sent, so errors become an event. Details go to the
        # log only: provider errors include request IDs and account details.
        log.exception("agent run failed")
        yield _sse("error", {"message": "the research agent failed; please try again"})


# Plain `def`: the agent loop blocks, so Starlette iterates the stream in a thread.
@router.post("/agent")
def agent(
    req: AgentRequest, run: Annotated[AgentRunner, Depends(get_agent_runner)]
) -> StreamingResponse:
    return StreamingResponse(
        event_stream(run(req.question)),
        media_type="text/event-stream",
        # no-cache: proxies must not store it; X-Accel-Buffering: forward each event now.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
