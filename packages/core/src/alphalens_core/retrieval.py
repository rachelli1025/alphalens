"""Hybrid retrieval over `chunks` (DESIGN.md §8).

One SQL statement runs two searches and fuses them:
- vector: nearest chunks by cosine distance on `embedding` (HNSW index);
- keyword: full-text match on `tsv` (GIN index), ranked by ts_rank_cd;
- Reciprocal Rank Fusion: score = sum over lists of 1 / (rrf_k + rank).

RRF only uses ranks, so the two very different score scales never need to be
normalized against each other.
"""

from dataclasses import dataclass
from typing import Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from alphalens_core.db.models import EMBEDDING_DIM, Company
from alphalens_core.embeddings import Embedder

RRF_K = 60  # standard constant from the RRF paper; damps the weight of top ranks
CANDIDATES = 50  # how many results each search contributes before fusion

# "vector" / "keyword" keep only one list in the fusion step; used for eval ablations.
SearchMode = Literal["hybrid", "vector", "keyword"]
_SOURCES: dict[str, list[str]] = {
    "hybrid": ["vec", "kw"],
    "vector": ["vec"],
    "keyword": ["kw"],
}

# Keyword side uses OR between terms: plainto_tsquery ANDs every word, which for a
# natural-language question ("what are apple's main supply chain risks") usually
# matches nothing. ts_rank_cd still ranks chunks with more matching terms higher.
_SQL = """
WITH q AS (
    SELECT to_tsquery('english', replace(plainto_tsquery('english', :qtext)::text, '&', '|'))
        AS query
),
vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> :qvec) AS r
    FROM chunks
    WHERE embedding IS NOT NULL {filter}
    ORDER BY embedding <=> :qvec
    LIMIT :candidates
),
kw AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q.query) DESC) AS r
    FROM chunks, q
    WHERE tsv @@ q.query {filter}
    ORDER BY ts_rank_cd(tsv, q.query) DESC
    LIMIT :candidates
),
fused AS (
    SELECT id, SUM(1.0 / (:rrf_k + r)) AS score,
           MIN(r) FILTER (WHERE src = 'vec') AS vec_rank,
           MIN(r) FILTER (WHERE src = 'kw') AS kw_rank
    FROM (
        SELECT id, r, 'vec' AS src FROM vec
        UNION ALL
        SELECT id, r, 'kw' AS src FROM kw
    ) both_lists
    WHERE src = ANY(:sources)
    GROUP BY id
)
SELECT c.id, c.content, c.context_header, s.item_code, s.title AS section_title,
       f.accession_no, f.form_type, f.fiscal_year, f.url, co.ticker,
       fused.score, fused.vec_rank, fused.kw_rank
FROM fused
JOIN chunks c ON c.id = fused.id
JOIN filing_sections s ON s.id = c.section_id
JOIN filings f ON f.id = s.filing_id
JOIN companies co ON co.id = c.company_id
ORDER BY fused.score DESC
LIMIT :k
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    content: str
    context_header: str
    ticker: str
    form_type: str
    fiscal_year: int | None
    item_code: str
    section_title: str | None
    accession_no: str
    url: str
    score: float
    vec_rank: int | None  # rank in the vector list, None if it only came from keywords
    kw_rank: int | None  # rank in the keyword list, None if it only came from vectors


def hybrid_search(
    session: Session,
    embedder: Embedder,
    question: str,
    *,
    tickers: list[str] | None = None,
    k: int = 8,
    mode: SearchMode = "hybrid",
) -> list[RetrievedChunk]:
    """Top-k chunks for a question, optionally restricted to some companies."""
    params: dict[str, object] = {
        "sources": _SOURCES[mode],
        "qtext": question,
        "qvec": embedder.embed([question], "query")[0],
        "candidates": CANDIDATES,
        "rrf_k": RRF_K,
        "k": k,
    }
    filter_sql = ""
    if tickers:
        ids = session.scalars(
            select(Company.id).where(Company.ticker.in_([t.upper() for t in tickers]))
        ).all()
        if not ids:
            return []
        filter_sql = "AND company_id = ANY(:company_ids)"
        params["company_ids"] = list(ids)

    stmt = text(_SQL.format(filter=filter_sql)).bindparams(
        bindparam("qvec", type_=Vector(EMBEDDING_DIM))
    )
    rows = session.execute(stmt, params).mappings().all()
    return [
        RetrievedChunk(
            chunk_id=row["id"],
            content=row["content"],
            context_header=row["context_header"],
            ticker=row["ticker"],
            form_type=row["form_type"],
            fiscal_year=row["fiscal_year"],
            item_code=row["item_code"],
            section_title=row["section_title"],
            accession_no=row["accession_no"],
            url=row["url"],
            score=float(row["score"]),
            vec_rank=row["vec_rank"],
            kw_rank=row["kw_rank"],
        )
        for row in rows
    ]
