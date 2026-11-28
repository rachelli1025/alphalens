"""ingest_financials + v_financials against a real Postgres (CI); skipped without one."""

import copy
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.orm import Session

from alphalens_core.db.models import Company, FinancialFact
from alphalens_core.ingest.financials import ingest_financials
from alphalens_core.sec.client import CompanyInfo

CIK = "0009999996"


def _e(start: str | None, end: str, val: float, accn: str, fy: int, filed: str) -> dict[str, Any]:
    e: dict[str, Any] = {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
    }
    if start:
        e["start"] = start
    return e


BASE: dict[str, Any] = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        _e("2024-01-01", "2024-12-31", 100.0, "a-24", 2024, "2025-02-01"),
                        _e("2025-01-01", "2025-12-31", 120.0, "a-25", 2025, "2026-02-01"),
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {"USD": [_e("2025-01-01", "2025-12-31", 12.0, "a-25", 2025, "2026-02-01")]}
            },
            "Assets": {
                "units": {"USD": [_e(None, "2025-12-31", 900.0, "a-25", 2025, "2026-02-01")]}
            },
        }
    }
}


class FakeSource:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def lookup_company(self, ticker: str) -> CompanyInfo:
        return CompanyInfo(cik=CIK, ticker=ticker, name="Facts Corp.", industry=None)

    def company_facts(self, cik: str) -> dict[str, Any]:
        return self.payload


@pytest.fixture
def session(pg_engine: Engine) -> Iterator[Session]:
    def cleanup(s: Session) -> None:
        ids = select(Company.id).where(Company.cik == CIK).scalar_subquery()
        s.execute(delete(FinancialFact).where(FinancialFact.company_id == ids))
        s.execute(delete(Company).where(Company.cik == CIK))
        s.commit()

    with Session(pg_engine) as s:
        cleanup(s)
        yield s
        s.rollback()
        cleanup(s)


def _view(s: Session, fy: int) -> dict[str, Any]:
    row = (
        s.execute(
            text("SELECT * FROM v_financials WHERE ticker = 'FCTS' AND fiscal_year = :fy"),
            {"fy": fy},
        )
        .mappings()
        .one()
    )
    return dict(row)


def _count(s: Session) -> int:
    ids = select(Company.id).where(Company.cik == CIK).scalar_subquery()
    return int(
        s.scalar(
            select(func.count()).select_from(FinancialFact).where(FinancialFact.company_id == ids)
        )
        or 0
    )


def test_ingest_fills_the_wide_view(session: Session) -> None:
    result = ingest_financials(session, FakeSource(BASE), "fcts")
    assert result.facts == 4
    row = _view(session, 2025)
    assert row["fiscal_period"] == "FY"
    assert (float(row["revenue"]), float(row["net_income"]), float(row["total_assets"])) == (
        120.0,
        12.0,
        900.0,
    )
    assert row["gross_profit"] is None  # not reported -> NULL, not 0


def test_rerun_is_idempotent(session: Session) -> None:
    ingest_financials(session, FakeSource(BASE), "FCTS")
    ingest_financials(session, FakeSource(BASE), "FCTS")
    assert _count(session) == 4


def test_restatement_in_a_later_filing_wins_in_the_view(session: Session) -> None:
    restated = copy.deepcopy(BASE)
    restated["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        # FY2024 restated to 105 as a comparative in the FY2025 10-K
        _e("2024-01-01", "2024-12-31", 105.0, "a-25", 2025, "2026-02-01")
    )
    ingest_financials(session, FakeSource(restated), "FCTS")
    assert float(_view(session, 2024)["revenue"]) == 105.0
    assert _count(session) == 5  # the original 100 is kept for history
