# AlphaLens

AI research copilot for US equities: cited Q&A over SEC filings, financials and news, plus a news-sentiment signal evaluated with event studies and backtests.

- Design: [docs/DESIGN.md](docs/DESIGN.md)
- Decisions: [docs/adr/](docs/adr/)

> Not investment advice.

## Layout

```
apps/web        Next.js 15 + React + TypeScript
apps/api        FastAPI + Alembic migrations
packages/core   alphalens_core: shared business logic (imported by API, Airflow, workers)
airflow/        DAGs, plugins, DAG tests
infra/          docker-compose, Postgres init
```

## Prerequisites

- Docker Desktop
- [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched automatically)
- Node 22+

## Run the full stack

```bash
cp .env.example .env          # then fill in values
docker compose --env-file .env -f infra/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| Web | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |

## Develop without Docker for app code

```bash
uv sync --all-packages        # Python deps for core + api
uv run pytest                 # tests
uv run ruff check . && uv run ruff format --check .

cd apps/web && npm install && npm run dev
```

Migrations (needs Postgres running):

```bash
cd apps/api && uv run alembic upgrade head
```
