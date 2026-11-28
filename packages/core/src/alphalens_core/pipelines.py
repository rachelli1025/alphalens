"""Task bodies for the Airflow DAGs (airflow/dags/). Kept in core so they run and are tested
without Airflow (ADR 0004); the DAG files only decide when, in what order and with what
retries these run.

Each per-ticker function is idempotent: re-running after a success skips or upserts, so
Airflow retries and manual re-runs are safe. Results are small dicts because they travel
through XCom, which should carry identifiers and counts, not data (DESIGN.md §6.3).
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alphalens_core.config import get_settings
from alphalens_core.db.models import Company
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import VoyageEmbedder
from alphalens_core.ingest import ingest_latest_filing
from alphalens_core.ingest.financials import ingest_financials
from alphalens_core.sec import EdgarClient

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


def universe_tickers(session: Session) -> list[str]:
    """Companies the daily pipelines maintain (companies.in_universe)."""
    rows = session.scalars(
        select(Company.ticker).where(Company.in_universe.is_(True)).order_by(Company.ticker)
    )
    return list(rows)


def load_universe() -> list[str]:
    with get_sync_sessionmaker()() as session:
        return universe_tickers(session)


def refresh_filing(ticker: str, form_type: str = "10-K") -> dict[str, Any]:
    """Ingest the latest filing of one company; a no-op if it is already embedded."""
    s = get_settings()
    with (
        EdgarClient(s.sec_user_agent) as edgar,
        VoyageEmbedder(s.voyage_api_key, s.embedding_model) as embedder,
        get_sync_sessionmaker()() as session,
    ):
        r = ingest_latest_filing(session, edgar, embedder, ticker, form_type=form_type)
        return {
            "ticker": r.ticker,
            "accession_no": r.accession_no,
            "new": not r.skipped,
            "chunks": r.chunks,
            "voyage_tokens": embedder.total_tokens,
        }


def refresh_financials(ticker: str) -> dict[str, Any]:
    """Upsert all XBRL facts of one company."""
    s = get_settings()
    with EdgarClient(s.sec_user_agent) as edgar, get_sync_sessionmaker()() as session:
        r = ingest_financials(session, edgar, ticker)
        return {"ticker": r.ticker, "facts": r.facts, "labelled": r.labelled}


def summarize(expected: list[str], results: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Validation step: fail the run if any ticker produced no result.

    Mapped tasks that failed after all retries hand back nothing, so comparing the
    tickers we got results for with the tickers we expected finds them.
    """
    done = {r["ticker"] for r in results if r}
    missing = sorted(set(expected) - done)
    summary = {
        "expected": len(expected),
        "succeeded": len(done),
        "missing": missing,
        "new_filings": sorted(r["ticker"] for r in results if r and r.get("new")),
    }
    log.info("pipeline summary: %s", summary)
    if missing:
        raise PipelineError(f"no result for {missing}")
    return summary
