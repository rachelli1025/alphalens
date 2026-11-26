"""End-to-end ingestion against a real Postgres (CI service container).

Skipped when no database is reachable. EDGAR and Voyage are faked.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from alphalens_core.config import get_settings
from alphalens_core.db.models import Chunk, Company, Filing, FilingSection
from alphalens_core.embeddings import InputType
from alphalens_core.ingest import ingest_latest_filing
from alphalens_core.ingest.filings import IngestError
from alphalens_core.sec.client import CompanyInfo, FilingRef

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"
TEST_CIK = "0009999999"
ACCESSION = "0009999999-25-000001"


class FakeSource:
    def __init__(self, html: str) -> None:
        self.html = html

    def lookup_company(self, ticker: str) -> CompanyInfo:
        return CompanyInfo(cik=TEST_CIK, ticker=ticker, name="Acme Corp.", industry="Widgets")

    def latest_filing(self, cik: str, form_type: str = "10-K") -> FilingRef:
        return FilingRef(
            accession_no=ACCESSION,
            form_type=form_type,
            filed_at=date(2025, 10, 31),
            period_of_report=date(2025, 9, 27),
            primary_document="acme-20250927.htm",
            url="https://www.sec.gov/Archives/edgar/data/9999999/x/acme-20250927.htm",
        )

    def download(self, url: str) -> str:
        return self.html


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("voyage down")
        return [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine(get_settings().database_url, connect_args={"connect_timeout": 2})
    try:
        with engine.connect() as conn:
            if conn.scalar(text("SELECT to_regclass('chunks')")) is None:
                pytest.skip("database not migrated")
    except OperationalError:
        pytest.skip("no database reachable")

    def cleanup(s: Session) -> None:
        s.execute(delete(Filing).where(Filing.accession_no == ACCESSION))  # cascades
        s.execute(delete(Company).where(Company.cik == TEST_CIK))
        s.commit()

    with Session(engine) as s:
        cleanup(s)
        yield s
        s.rollback()
        cleanup(s)
    engine.dispose()


def _counts(s: Session) -> tuple[int, int]:
    filing_id = s.scalar(select(Filing.id).where(Filing.accession_no == ACCESSION))
    sections = s.scalar(
        select(func.count()).select_from(FilingSection).where(FilingSection.filing_id == filing_id)
    )
    chunks = s.scalar(
        select(func.count())
        .select_from(Chunk)
        .join(FilingSection)
        .where(FilingSection.filing_id == filing_id)
    )
    return int(sections or 0), int(chunks or 0)


def test_ingest_writes_sections_and_embedded_chunks(session: Session) -> None:
    result = ingest_latest_filing(session, FakeSource(FIXTURE.read_text()), FakeEmbedder(), "acme")

    assert not result.skipped
    assert result.sections == 5
    assert _counts(session) == (result.sections, result.chunks)

    filing = session.scalar(select(Filing).where(Filing.accession_no == ACCESSION))
    assert filing is not None
    assert filing.status == "embedded"
    assert filing.fiscal_year == 2025

    chunk = session.scalar(
        select(Chunk)
        .join(FilingSection)
        .where(FilingSection.filing_id == filing.id, FilingSection.item_code == "1A")
        .order_by(Chunk.chunk_idx)
    )
    assert chunk is not None
    assert chunk.context_header == "[ACME | 10-K | FY2025 | Item 1A Risk Factors]"
    assert chunk.embedding is not None
    assert chunk.form_type == "10-K"
    # Postgres keeps the full-text column up to date on its own.
    hit = session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.id == chunk.id, Chunk.tsv.op("@@")(func.plainto_tsquery("english", "supply")))
    )
    assert hit == 1


def test_rerun_is_skipped_and_force_does_not_duplicate(session: Session) -> None:
    source = FakeSource(FIXTURE.read_text())
    first = ingest_latest_filing(session, source, FakeEmbedder(), "ACME")

    embedder = FakeEmbedder()
    again = ingest_latest_filing(session, source, embedder, "ACME")
    assert again.skipped
    assert embedder.calls == 0

    forced = ingest_latest_filing(session, source, FakeEmbedder(), "ACME", force=True)
    assert not forced.skipped
    assert forced.filing_id == first.filing_id
    assert _counts(session) == (first.sections, first.chunks)


def test_failure_marks_filing_failed_and_writes_nothing(session: Session) -> None:
    with pytest.raises(RuntimeError, match="voyage down"):
        ingest_latest_filing(
            session, FakeSource(FIXTURE.read_text()), FakeEmbedder(fail=True), "ACME"
        )

    status = session.scalar(select(Filing.status).where(Filing.accession_no == ACCESSION))
    assert status == "failed"
    assert _counts(session) == (0, 0)


def test_document_without_items_raises(session: Session) -> None:
    with pytest.raises(IngestError, match="no Item sections"):
        ingest_latest_filing(
            session, FakeSource("<html><p>hello</p></html>"), FakeEmbedder(), "ACME"
        )
