# ADR-0002: Use Airflow 3 for pipeline orchestration and arq for on-demand jobs

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
The daily pipelines form a dependency graph: signals need both prices and extracted news, and evals depend on fresh chunks. The system needs backfills, partial re-runs, run history, and rate limiting against external APIs. Separately, users trigger backtests that need responses in seconds.

## Decision
- **Airflow 3** runs all scheduled and batch workflows. It provides Asset-based scheduling, dynamic task mapping for per-filing work, pools for rate limiting, and deferrable operators for Batch API waits.
- **arq** (Redis) runs user-triggered, latency-sensitive jobs.
- DAGs stay thin and call `alphalens_core`.

## Consequences
- Native backfill, retries, lineage through Assets, and a full run UI.
- Heavier local setup (api-server, scheduler, dag-processor, triggerer). Managed MWAA is available in stage 2.
- Two async mechanisms to understand, with a clear split: scheduled work goes to Airflow, interactive work goes to arq.

## Alternatives considered
- **Celery Beat:** weak dependency handling and no backfill.
- **Dagster:** a strong asset model, but Airflow has wider industry adoption, which fits the project's goals.
