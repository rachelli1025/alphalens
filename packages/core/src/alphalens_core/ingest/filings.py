"""Filing ingestion: EDGAR -> filings -> filing_sections -> chunks (with embeddings).

Idempotent on natural keys: re-running for the same filing replaces its sections
and chunks instead of duplicating them, and an already-embedded filing is skipped
unless `force=True`.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from alphalens_core.chunking import chunk_text
from alphalens_core.db.models import Chunk, Company, Filing, FilingSection
from alphalens_core.embeddings import Embedder
from alphalens_core.sec.client import CompanyInfo, FilingRef
from alphalens_core.sec.parse import Section, html_to_text, split_10k_items

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


class FilingSource(Protocol):
    """What ingestion needs from EDGAR (EdgarClient implements it; tests use a fake)."""

    def lookup_company(self, ticker: str) -> CompanyInfo: ...
    def latest_filing(self, cik: str, form_type: str = "10-K") -> FilingRef: ...
    def download(self, url: str) -> str: ...


@dataclass(frozen=True)
class IngestResult:
    ticker: str
    accession_no: str
    filing_id: int
    skipped: bool
    sections: int = 0
    chunks: int = 0


def context_header(ticker: str, form_type: str, fiscal_year: int | None, s: Section) -> str:
    """Prepended to each chunk before embedding so the vector knows who/when/where."""
    fy = f"FY{fiscal_year}" if fiscal_year else "FY?"
    item = f"Item {s.item_code}" + (f" {s.title}" if s.title else "")
    return f"[{ticker} | {form_type} | {fy} | {item}]"


def upsert_company(session: Session, info: CompanyInfo) -> int:
    stmt = (
        insert(Company)
        .values(ticker=info.ticker, cik=info.cik, name=info.name, industry=info.industry)
        .on_conflict_do_update(
            index_elements=[Company.cik],
            set_={"ticker": info.ticker, "name": info.name, "industry": info.industry},
        )
        .returning(Company.id)
    )
    return session.execute(stmt).scalar_one()


def upsert_filing(session: Session, company_id: int, ref: FilingRef) -> Filing:
    # Fiscal year approximated from the period end date; W2 switches to the XBRL
    # DocumentFiscalYearFocus, which handles January/February year ends correctly.
    fiscal_year = ref.period_of_report.year if ref.period_of_report else ref.filed_at.year
    values = {
        "company_id": company_id,
        "accession_no": ref.accession_no,
        "form_type": ref.form_type,
        "filed_at": ref.filed_at,
        "period_of_report": ref.period_of_report,
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY" if ref.form_type == "10-K" else None,
        "url": ref.url,
    }
    stmt = (
        insert(Filing)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[Filing.accession_no],
            set_={k: v for k, v in values.items() if k != "accession_no"},
        )
        .returning(Filing.id)
    )
    filing_id = session.execute(stmt).scalar_one()
    filing = session.get(Filing, filing_id, populate_existing=True)
    assert filing is not None
    return filing


def ingest_latest_filing(
    session: Session,
    source: FilingSource,
    embedder: Embedder,
    ticker: str,
    *,
    form_type: str = "10-K",
    force: bool = False,
) -> IngestResult:
    ticker = ticker.upper()
    info = source.lookup_company(ticker)
    company_id = upsert_company(session, info)
    ref = source.latest_filing(info.cik, form_type)
    filing = upsert_filing(session, company_id, ref)
    filing_id = filing.id
    session.commit()
    log.info("%s %s %s (filed %s)", ticker, form_type, ref.accession_no, ref.filed_at)

    if filing.status == "embedded" and not force:
        log.info("already embedded, skipping (use force=True to rebuild)")
        return IngestResult(ticker, ref.accession_no, filing.id, skipped=True)

    try:
        html = source.download(ref.url)
        sections = [s for s in split_10k_items(html_to_text(html)) if s.text]
        if not sections:
            raise IngestError(f"no Item sections found in {ref.url}")
        log.info("parsed %d sections: %s", len(sections), [s.item_code for s in sections])

        # Replace previous output for this filing (chunks go via ON DELETE CASCADE).
        session.execute(delete(FilingSection).where(FilingSection.filing_id == filing.id))

        rows: list[Chunk] = []
        for s in sections:
            section = FilingSection(
                filing_id=filing.id,
                item_code=s.item_code,
                title=s.title,
                text=s.text,
                token_count=len(s.text) // 4,
            )
            session.add(section)
            session.flush()
            header = context_header(ticker, form_type, filing.fiscal_year, s)
            for c in chunk_text(s.text):
                rows.append(
                    Chunk(
                        section_id=section.id,
                        company_id=company_id,
                        chunk_idx=c.idx,
                        content=c.content,
                        context_header=header,
                        token_count=c.token_count,
                        form_type=form_type,
                        fiscal_year=filing.fiscal_year,
                        filed_at=filing.filed_at,
                    )
                )

        log.info("embedding %d chunks", len(rows))
        vectors = embedder.embed([f"{r.context_header}\n\n{r.content}" for r in rows], "document")
        for row, vec in zip(rows, vectors, strict=True):
            row.embedding = vec
        session.add_all(rows)
        filing.status = "embedded"
        session.commit()
    except Exception:
        session.rollback()
        session.execute(update(Filing).where(Filing.id == filing_id).values(status="failed"))
        session.commit()
        raise

    return IngestResult(
        ticker, ref.accession_no, filing.id, skipped=False, sections=len(sections), chunks=len(rows)
    )
