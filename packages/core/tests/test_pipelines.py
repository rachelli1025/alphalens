"""pipelines.summarize (pure) and universe_tickers (real Postgres in CI)."""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from alphalens_core.db.models import Company
from alphalens_core.pipelines import PipelineError, summarize, universe_tickers


def test_summary_lists_new_filings() -> None:
    s = summarize(
        ["AAPL", "MSFT"],
        [{"ticker": "AAPL", "new": True}, {"ticker": "MSFT", "new": False}],
    )
    assert s == {"expected": 2, "succeeded": 2, "missing": [], "new_filings": ["AAPL"]}


def test_failed_tickers_fail_the_run() -> None:
    # A mapped task that failed after its retries hands back nothing (None or absent).
    with pytest.raises(PipelineError, match=r"\['MSFT', 'NVDA'\]"):
        summarize(["AAPL", "MSFT", "NVDA"], [{"ticker": "AAPL"}, None])


@pytest.fixture
def session(pg_engine: Engine) -> Iterator[Session]:
    ciks = ["0009999994", "0009999993"]
    with Session(pg_engine) as s:
        s.execute(delete(Company).where(Company.cik.in_(ciks)))
        s.add_all(
            [
                Company(ticker="UNIA", cik=ciks[0], name="In Universe", in_universe=True),
                Company(ticker="UNIB", cik=ciks[1], name="Out of Universe", in_universe=False),
            ]
        )
        s.commit()
        yield s
        s.rollback()
        s.execute(delete(Company).where(Company.cik.in_(ciks)))
        s.commit()


def test_universe_only_includes_flagged_companies(session: Session) -> None:
    tickers = universe_tickers(session)
    assert "UNIA" in tickers
    assert "UNIB" not in tickers
