"""POST /api/ask: one question in, one cited answer out.

W1 simplification of DESIGN.md §11: stateless and non-streaming. W3 replaces it
with /api/chat/conversations/{id}/messages (history + SSE streaming).
"""

import hmac
import logging
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Annotated

import anthropic
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from alphalens_core.answer import Answer, AnswerPart, answer_question
from alphalens_core.config import Settings, get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import EmbeddingError, VoyageEmbedder
from alphalens_core.query_understanding import TickerResolution, list_companies, resolve_tickers
from alphalens_core.reranking import RerankError, VoyageReranker
from alphalens_core.retrieval import RetrievedChunk, hybrid_search, rerank_search

log = logging.getLogger(__name__)


def require_access_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_access_key: Annotated[str | None, Header()] = None,
) -> None:
    """Every /api/ask call spends real money, so deployed instances require a shared key.

    compare_digest takes the same time whether the first or last character is wrong,
    so the key cannot be guessed one character at a time from response timings.
    """
    if not settings.access_key:
        return
    if x_access_key is None or not hmac.compare_digest(x_access_key, settings.access_key):
        raise HTTPException(status_code=401, detail="missing or invalid access key")


router = APIRouter(prefix="/api", tags=["ask"], dependencies=[Depends(require_access_key)])


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
    cost_usd: float | None  # list-price estimate (answer + ticker resolution); None if unknown
    refused: bool
    tickers: list[str]  # companies searched; empty = all companies
    tickers_resolved: bool  # True if inferred from the question rather than given by the caller


# ---------- dependencies (overridden in tests) ----------

Retriever = Callable[[str, list[str] | None, int], list[RetrievedChunk]]
TickerResolver = Callable[[str], TickerResolution]


@lru_cache
def get_embedder() -> VoyageEmbedder:
    s = get_settings()
    return VoyageEmbedder(s.voyage_api_key, s.embedding_model)


@lru_cache
def get_claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None)


@lru_cache
def get_reranker() -> VoyageReranker:
    s = get_settings()
    return VoyageReranker(s.voyage_api_key, s.rerank_model)


def get_session() -> Iterator[Session]:
    with get_sync_sessionmaker()() as session:
        yield session


def get_retriever(
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[VoyageEmbedder, Depends(get_embedder)],
    reranker: Annotated[VoyageReranker, Depends(get_reranker)],
) -> Retriever:
    def retrieve(question: str, tickers: list[str] | None, k: int) -> list[RetrievedChunk]:
        if get_settings().rerank_enabled:
            return rerank_search(session, embedder, reranker, question, tickers=tickers, k=k)
        return hybrid_search(session, embedder, question, tickers=tickers, k=k)

    return retrieve


def get_ticker_resolver(
    session: Annotated[Session, Depends(get_session)],
    claude: Annotated[anthropic.Anthropic, Depends(get_claude)],
) -> TickerResolver:
    def resolve(question: str) -> TickerResolution:
        return resolve_tickers(
            claude, get_settings().query_model, question, list_companies(session)
        )

    return resolve


# ---------- endpoint ----------


def serialize_parts(answer_parts: list[AnswerPart]) -> tuple[list[PartOut], list[SourceOut]]:
    """Answer parts with citations as indexes into a list of the cited chunks."""
    sources: list[SourceOut] = []
    index: dict[int, int] = {}  # chunk_id -> position in sources
    parts = []
    for part in answer_parts:
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
    return parts, sources


def to_response(answer: Answer, resolution: TickerResolution, resolved: bool) -> AskResponse:
    parts, sources = serialize_parts(answer.parts)
    return AskResponse(
        answer=answer.text,
        parts=parts,
        sources=sources,
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=(
            None
            if answer.cost_usd is None or resolution.cost_usd is None
            else answer.cost_usd + resolution.cost_usd
        ),
        refused=answer.refused,
        tickers=resolution.tickers,
        tickers_resolved=resolved,
    )


# Plain `def`, not `async def`: retrieval and the SDK calls are blocking, so
# FastAPI runs this in its thread pool instead of blocking the event loop.
@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    retrieve: Annotated[Retriever, Depends(get_retriever)],
    claude: Annotated[anthropic.Anthropic, Depends(get_claude)],
    resolve: Annotated[TickerResolver, Depends(get_ticker_resolver)],
) -> AskResponse:
    # Caller-given tickers win; otherwise infer them from the question (never raises).
    if req.tickers:
        resolution, resolved = TickerResolution(tickers=[t.upper() for t in req.tickers]), False
    else:
        resolution, resolved = resolve(req.question), True

    # Upstream errors are logged in full but returned to callers as generic messages:
    # raw provider errors include request IDs and account details.
    try:
        chunks = retrieve(req.question, resolution.tickers or None, req.k)
    except (EmbeddingError, RerankError) as exc:
        log.exception("retrieval failed")
        raise HTTPException(status_code=502, detail="search is temporarily unavailable") from exc

    try:
        settings = get_settings()
        answer = answer_question(
            claude, settings.answer_model, req.question, chunks, effort=settings.answer_effort
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        log.exception("Claude call failed")
        raise HTTPException(
            status_code=502, detail="the answer service is temporarily unavailable"
        ) from exc

    return to_response(answer, resolution, resolved)
