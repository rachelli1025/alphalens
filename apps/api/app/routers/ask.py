"""POST /api/ask: one question in, one cited answer out.

W1 simplification of DESIGN.md §11: stateless and non-streaming. W3 replaces it
with /api/chat/conversations/{id}/messages (history + SSE streaming).
"""

from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from alphalens_core.answer import Answer, answer_question
from alphalens_core.config import get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import EmbeddingError, VoyageEmbedder
from alphalens_core.retrieval import RetrievedChunk, hybrid_search

router = APIRouter(prefix="/api", tags=["ask"])


# ---------- request / response schemas ----------


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    tickers: list[str] | None = Field(default=None, max_length=10)
    k: int = Field(default=8, ge=1, le=20)


class CitationOut(BaseModel):
    source: int  # index into AskResponse.sources
    cited_text: str


class PartOut(BaseModel):
    text: str
    citations: list[CitationOut]


class SourceOut(BaseModel):
    chunk_id: int
    ticker: str
    form_type: str
    fiscal_year: int | None
    item_code: str
    section_title: str | None
    context_header: str
    url: str


class AskResponse(BaseModel):
    answer: str
    parts: list[PartOut]
    sources: list[SourceOut]  # only chunks the answer actually cites, in first-cited order
    model: str
    input_tokens: int
    output_tokens: int
    refused: bool


# ---------- dependencies (overridden in tests) ----------

Retriever = Callable[[str, list[str] | None, int], list[RetrievedChunk]]


@lru_cache
def get_embedder() -> VoyageEmbedder:
    s = get_settings()
    return VoyageEmbedder(s.voyage_api_key, s.embedding_model)


@lru_cache
def get_claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None)


def get_session() -> Iterator[Session]:
    with get_sync_sessionmaker()() as session:
        yield session


def get_retriever(
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[VoyageEmbedder, Depends(get_embedder)],
) -> Retriever:
    def retrieve(question: str, tickers: list[str] | None, k: int) -> list[RetrievedChunk]:
        return hybrid_search(session, embedder, question, tickers=tickers, k=k)

    return retrieve


# ---------- endpoint ----------


def to_response(answer: Answer) -> AskResponse:
    sources: list[SourceOut] = []
    index: dict[int, int] = {}  # chunk_id -> position in sources
    parts = []
    for part in answer.parts:
        citations = []
        for c in part.citations:
            if c.chunk.chunk_id not in index:
                index[c.chunk.chunk_id] = len(sources)
                sources.append(
                    SourceOut(
                        chunk_id=c.chunk.chunk_id,
                        ticker=c.chunk.ticker,
                        form_type=c.chunk.form_type,
                        fiscal_year=c.chunk.fiscal_year,
                        item_code=c.chunk.item_code,
                        section_title=c.chunk.section_title,
                        context_header=c.chunk.context_header,
                        url=c.chunk.url,
                    )
                )
            citations.append(CitationOut(source=index[c.chunk.chunk_id], cited_text=c.cited_text))
        parts.append(PartOut(text=part.text, citations=citations))
    return AskResponse(
        answer=answer.text,
        parts=parts,
        sources=sources,
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        refused=answer.refused,
    )


# Plain `def`, not `async def`: retrieval and the SDK calls are blocking, so
# FastAPI runs this in its thread pool instead of blocking the event loop.
@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    retrieve: Annotated[Retriever, Depends(get_retriever)],
    claude: Annotated[anthropic.Anthropic, Depends(get_claude)],
) -> AskResponse:
    try:
        chunks = retrieve(req.question, req.tickers, req.k)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=f"embedding service error: {exc}") from exc

    try:
        answer = answer_question(claude, get_settings().agent_model, req.question, chunks)
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="could not reach the Claude API") from exc

    return to_response(answer)
