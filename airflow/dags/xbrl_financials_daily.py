"""Daily XBRL financial facts refresh for every company in the universe (DESIGN.md §6.2).

get_universe -> ingest_financials (one mapped task per ticker) -> validate

Upserts are keyed on (company, metric, period, filing), so re-running only touches rows
whose values changed. v_financials is a plain view and needs no refresh step. No LLM or
embedding calls: the whole run costs only SEC requests.
"""

from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import Asset, dag, task

FINANCIAL_FACTS = Asset("alphalens://postgres/financial_facts")

DEFAULT_ARGS = {
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="xbrl_financials_daily",
    schedule="30 18 * * 1-5",  # weekdays 18:30 New York, after sec_filings_daily starts
    start_date=pendulum.datetime(2026, 11, 1, tz="America/New_York"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["sec", "xbrl", "ingest"],
    doc_md=__doc__,
)
def xbrl_financials_daily() -> None:
    @task
    def get_universe() -> list[str]:
        from alphalens_core.pipelines import load_universe

        return load_universe()

    @task(max_active_tis_per_dagrun=2)
    def ingest_financials(ticker: str) -> dict[str, Any]:
        from alphalens_core.pipelines import refresh_financials

        return refresh_financials(ticker)

    @task(trigger_rule="all_done", outlets=[FINANCIAL_FACTS])
    def validate(expected: list[str], results: list[dict[str, Any] | None]) -> dict[str, Any]:
        from alphalens_core.pipelines import summarize

        return summarize(expected, list(results))

    tickers = get_universe()
    validate(tickers, ingest_financials.expand(ticker=tickers))


xbrl_financials_daily()
