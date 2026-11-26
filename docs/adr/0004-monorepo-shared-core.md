# ADR-0004: Monorepo with a shared `alphalens_core` package

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
The same logic runs in several places: chunking and embedding run in Airflow, retrieval runs in the API and in evals, and signal code runs in Airflow and arq.

## Decision
Use one repository. Business logic lives in `packages/core` (`alphalens_core`), which the API, the Airflow DAGs, the arq tasks, and the evals all import. DAG files contain orchestration only.

## Consequences
- One implementation for each piece of logic, unit-tested without Airflow.
- Images for the API, Airflow, and workers all install the core package. Version skew is avoided by building them from the same commit.
