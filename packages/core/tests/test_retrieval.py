"""Hybrid retrieval against a real Postgres (CI service container); skipped without one.

Headers avoid words used in the queries: `tsv` indexes header + content, so a
header like "Item 1A Risk Factors" would make every chunk match "risk".
Chunks get one-hot embeddings so vector distances are exact: a chunk whose
embedding equals the query vector has distance 0, every other chunk distance 1.
"""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from alphalens_core.db.models import EMBEDDING_DIM, Chunk, Company, Filing, FilingSection
from alphalens_core.embeddings import InputType
from alphalens_core.retrieval import hybrid_search

CIKS = {"TSTA": "0009999998", "TSTB": "0009999997"}


def one_hot(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


class QueryEmbedder:
    """Returns one-hot(0) for every query, i.e. 'closest to chunks embedded at index 0'."""

    def __init__(self) -> None:
        self.input_types: list[InputType] = []

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        self.input_types.append(input_type)
        return [one_hot(0) for _ in texts]


# (ticker, content, one-hot index)
CHUNKS = [
    ("TSTA", "Supply chain concentration in Asia is a key risk.", 0),  # vector + keyword
    ("TSTA", "The board approved a dividend increase and buybacks.", 1),  # neither
    ("TSTA", "Our supply chain relies on single-source components.", 2),  # keyword only
    ("TSTB", "TSTB faces supply chain risk from tariffs.", 0),  # vector + keyword, company B
]


def _cleanup(s: Session) -> None:
    for i, cik in enumerate(CIKS.values()):
        s.execute(delete(Filing).where(Filing.accession_no == f"{cik}-25-00000{i}"))
        s.execute(delete(Company).where(Company.cik == cik))
    s.commit()


@pytest.fixture
def session(pg_engine: Engine) -> Iterator[Session]:
    engine = pg_engine

    with Session(engine) as s:
        _cleanup(s)
        sections: dict[str, FilingSection] = {}
        for i, (ticker, cik) in enumerate(CIKS.items()):
            company = Company(ticker=ticker, cik=cik, name=f"{ticker} Inc.")
            filing = Filing(
                company=company,
                accession_no=f"{cik}-25-00000{i}",
                form_type="10-K",
                filed_at=date(2025, 10, 31),
                fiscal_year=2025,
                url=f"https://example.com/{ticker}.htm",
            )
            sections[ticker] = FilingSection(filing=filing, item_code="7", title="MD&A", text="...")
            s.add(sections[ticker])
        s.flush()
        for idx, (ticker, content, vec) in enumerate(CHUNKS):
            section = sections[ticker]
            s.add(
                Chunk(
                    section_id=section.id,
                    company_id=section.filing.company_id,
                    chunk_idx=idx,
                    content=content,
                    context_header=f"[{ticker} | 10-K | FY2025 | Item 7 MD&A]",
                    embedding=one_hot(vec),
                    form_type="10-K",
                    fiscal_year=2025,
                    filed_at=date(2025, 10, 31),
                )
            )
        s.commit()
        yield s
        s.rollback()
        _cleanup(s)


def _contents(results: list) -> list[str]:  # type: ignore[type-arg]
    return [r.content for r in results]


def test_question_is_embedded_as_query(session: Session) -> None:
    embedder = QueryEmbedder()
    hybrid_search(session, embedder, "supply chain risk", tickers=["TSTA"])
    assert embedder.input_types == ["query"]


def test_chunk_in_both_lists_ranks_first(session: Session) -> None:
    results = hybrid_search(session, QueryEmbedder(), "supply chain risk", tickers=["TSTA"])
    top = results[0]
    assert top.content.startswith("Supply chain concentration")
    assert top.vec_rank == 1
    assert top.kw_rank is not None
    assert top.ticker == "TSTA"
    assert top.item_code == "7"


def test_keyword_only_match_beats_no_match(session: Session) -> None:
    results = hybrid_search(session, QueryEmbedder(), "supply chain risk", tickers=["TSTA"])
    contents = _contents(results)
    keyword_only = contents.index("Our supply chain relies on single-source components.")
    no_match = contents.index("The board approved a dividend increase and buybacks.")
    assert keyword_only < no_match
    assert results[no_match].kw_rank is None


def test_or_semantics_matches_partial_questions(session: Session) -> None:
    # "tariffs" and "supply" never appear together with "geopolitical"; AND would match nothing.
    results = hybrid_search(
        session, QueryEmbedder(), "geopolitical tariffs and supply", tickers=["TSTB"]
    )
    assert results[0].kw_rank == 1


def test_ticker_filter_excludes_other_companies(session: Session) -> None:
    results = hybrid_search(session, QueryEmbedder(), "supply chain risk", tickers=["tsta"])
    assert {r.ticker for r in results} == {"TSTA"}


def test_without_filter_searches_all_companies(session: Session) -> None:
    results = hybrid_search(session, QueryEmbedder(), "supply chain risk", k=50)
    tickers = {r.ticker for r in results}
    assert {"TSTA", "TSTB"} <= tickers


def test_unknown_ticker_returns_nothing(session: Session) -> None:
    assert hybrid_search(session, QueryEmbedder(), "anything", tickers=["NOPE"]) == []


def test_k_limits_results(session: Session) -> None:
    assert len(hybrid_search(session, QueryEmbedder(), "supply chain", tickers=["TSTA"], k=2)) == 2


def test_vector_mode_ignores_keyword_matches(session: Session) -> None:
    results = hybrid_search(
        session, QueryEmbedder(), "supply chain risk", tickers=["TSTA"], mode="vector"
    )
    assert all(r.kw_rank is None for r in results)
    assert results[0].vec_rank == 1


def test_keyword_mode_returns_only_keyword_matches(session: Session) -> None:
    results = hybrid_search(
        session, QueryEmbedder(), "supply chain risk", tickers=["TSTA"], mode="keyword"
    )
    assert results
    assert all(r.kw_rank is not None and r.vec_rank is None for r in results)
    assert "The board approved a dividend increase and buybacks." not in _contents(results)
