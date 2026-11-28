"""Daily 10-K refresh for every company in the universe (DESIGN.md §6.2).

get_universe -> ingest_filing (one mapped task per ticker) -> validate

ingest_filing is idempotent: when a company's latest 10-K is already embedded it returns
immediately without calling Voyage, so a quiet day costs a few SEC requests and nothing else.
validate fails the run if any ticker failed after its retries, and only a successful run
updates the `chunks` asset that downstream DAGs can be scheduled on.

Task bodies live in alphalens_core.pipelines (testable without Airflow); imports happen
inside tasks so parsing this file needs neither the database nor API keys.
"""

from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import Asset, dag, task

CHUNKS = Asset("alphalens://postgres/chunks")

DEFAULT_ARGS = {
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="sec_filings_daily",
    schedule="0 18 * * 1-5",  # weekdays 18:00 New York, after most filings are published
    start_date=pendulum.datetime(2026, 11, 1, tz="America/New_York"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["sec", "ingest"],
    doc_md=__doc__,
)
def sec_filings_daily() -> None:
    @task
    def get_universe() -> list[str]:
        from alphalens_core.pipelines import load_universe

        return load_universe()

    # At most two tickers at a time: keeps SEC traffic far below its 10 requests/s limit.
    @task(max_active_tis_per_dagrun=2)
    def ingest_filing(ticker: str) -> dict[str, Any]:
        from alphalens_core.pipelines import refresh_filing

        return refresh_filing(ticker)

    # all_done: run even if some tickers failed, so the failure is reported in one place.
    @task(trigger_rule="all_done", outlets=[CHUNKS])
    def validate(expected: list[str], results: list[dict[str, Any] | None]) -> dict[str, Any]:
        from alphalens_core.pipelines import summarize

        return summarize(expected, list(results))

    tickers = get_universe()
    validate(tickers, ingest_filing.expand(ticker=tickers))


sec_filings_daily()
