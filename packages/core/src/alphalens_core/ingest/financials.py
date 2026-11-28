"""XBRL company facts -> financial_facts. Idempotent: re-running upserts on the natural key."""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from alphalens_core.db.models import FinancialFact
from alphalens_core.ingest.filings import upsert_company
from alphalens_core.sec.client import CompanyInfo
from alphalens_core.xbrl import extract_facts

log = logging.getLogger(__name__)

BATCH = 1000


class FactsSource(Protocol):
    def lookup_company(self, ticker: str) -> CompanyInfo: ...
    def company_facts(self, cik: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FinancialsResult:
    ticker: str
    facts: int
    labelled: int  # facts with fiscal_year / fiscal_period, i.e. visible in v_financials


def ingest_financials(session: Session, source: FactsSource, ticker: str) -> FinancialsResult:
    ticker = ticker.upper()
    info = source.lookup_company(ticker)
    company_id = upsert_company(session, info)
    facts = extract_facts(source.company_facts(info.cik))

    rows = [{"company_id": company_id, **asdict(f)} for f in facts]
    update_cols = [
        "source_tag",
        "value",
        "unit",
        "fiscal_year",
        "fiscal_period",
        "form_type",
        "filed_at",
    ]
    for i in range(0, len(rows), BATCH):
        stmt = insert(FinancialFact).values(rows[i : i + BATCH])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_financial_facts_natural_key",
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        session.execute(stmt)
    session.commit()

    labelled = sum(f.fiscal_year is not None for f in facts)
    log.info("%s: %d facts (%d with fiscal labels)", ticker, len(facts), labelled)
    return FinancialsResult(ticker=ticker, facts=len(facts), labelled=labelled)
