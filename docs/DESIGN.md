# AlphaLens — System Design

| | |
|---|---|
| **Status** | Draft v1.1 |
| **Last updated** | 2026-11-25 |
| **Scope** | MVP (v1) |
| **Decision records** | [docs/adr/](adr/) |

---

## 1. Overview

AlphaLens is an AI research platform for US equities. It combines two capabilities:

1. **Research Copilot**: users ask questions in natural language. The system retrieves from SEC filings, structured financials, and news events, and answers with citations that link back to the source passage.
2. **Sentiment Signal**: news is continuously converted into structured sentiment events. These events are aggregated into a daily signal and evaluated with event studies, information coefficient (IC), and a cost-aware long-short backtest.

### 1.1 Goals
- Grounded answers: every factual claim links to a source chunk or event.
- Production engineering: idempotent pipelines, observability, evals in CI, cost controls, auth, and multi-tenancy.
- Rigorous signal methodology: point-in-time data, transaction costs, and significance testing.

### 1.2 Non-goals (v1)
- Trade execution or brokerage integration.
- Intraday or tick-level data.
- Earnings call transcripts (paid data; planned for v2).
- Investment advice. The UI states "Not investment advice".

### 1.3 Success metrics
| Area | Metric | v1 target |
|---|---|---|
| Retrieval | Recall@8 on golden set | ≥ 0.80 |
| Answers | Faithfulness (LLM judge) | ≥ 0.90 |
| Text-to-SQL | Execution accuracy | ≥ 0.85 |
| Extraction | `event_type` macro-F1 | ≥ 0.75 |
| Latency | Chat time-to-first-token p50 | < 2.5 s |
| Pipelines | Daily DAG success rate | ≥ 99% |

Targets are initial guesses and will be re-set after the first baseline eval.

---

## 2. Scope

### 2.1 MVP features
| ID | Feature | Description |
|---|---|---|
| F1 | Research Chat | Streaming Q&A with citations; a side panel shows the cited passage |
| F2 | Company Page | XBRL financial charts, filing timeline, candlestick chart with event markers, sentiment series |
| F3 | Event Feed | Live news events with ticker, type, sentiment, and confidence; duplicate stories are clustered |
| F4 | Signal Lab | Event-study CAR curves, IC time series, long-short backtest PnL, with adjustable parameters |
| F5 | Watchlist | Followed companies; the home page shows their latest events |
| F6 | Auth & tenancy | Per-user conversations and watchlists |
| F7 | Admin | Eval trends, LLM cost, pipeline status |

### 2.2 Universe
v1 covers the S&P 100 and five years of 10-K, 10-Q, and 8-K filings. Expansion to the S&P 500 comes after the pipelines are stable.

### 2.3 Future (v2+)
Daily email brief, alerts (sentiment shock or new 8-K), year-over-year 10-K diff, earnings call transcripts.

---

## 3. Architecture

```
                    ┌────────────────────────────────────┐
                    │  Next.js 15 (React + TypeScript)   │
                    │ Research · Company · Feed · Signals│
                    └─────────────────┬──────────────────┘
                                      │ REST + SSE
                    ┌─────────────────▼──────────────────┐
                    │           FastAPI (API)            │
                    │   auth · rate limit · quotas       │
                    │  ┌──────────────────────────────┐  │
                    │  │ Research Agent (Claude tools)│  │
                    │  └──────────────────────────────┘  │
                    └───┬─────────────────────┬──────────┘
                        │                     │ enqueue
          ┌─────────────▼─────────┐    ┌──────▼───────┐    ┌──────────────┐
          │ PostgreSQL 16         │    │   Redis 7    │◄───│ arq workers  │
          │  app DB:              │    │ cache · queue│    │ on-demand    │
          │   relational          │    │ rate limits  │    │ backtests    │
          │   + pgvector (HNSW)   │    └──────────────┘    └──────────────┘
          │   + tsvector (BM25)   │
          │  airflow DB (metadata)│
          └─────────────▲─────────┘
                        │
     ┌──────────────────┴────────────────────────────────────────┐
     │                        Airflow 3                          │
     │ api-server · scheduler · dag-processor · triggerer        │
     └──────────────────────────┬────────────────────────────────┘
                                │ imports
                  ┌─────────────▼──────────────┐
                  │  alphalens_core (package)  │  ◄── also imported by FastAPI
                  │ ingestion · rag · extraction│
                  │ signals · sql_guard · llm   │
                  └────────────────────────────┘

  Observability: Langfuse (LLM traces, cost) · OpenTelemetry → Grafana · Sentry
```

### 3.1 Principles
1. **One Postgres instance** holds relational data, vectors (pgvector), and full-text search (tsvector). A single query can combine metadata filters, vector search, and BM25. See [ADR-0001](adr/0001-pgvector-as-vector-store.md).
2. **Separate workloads**: the API serves low-latency requests, Airflow runs scheduled and batch DAGs, and arq runs user-triggered async jobs. See [ADR-0002](adr/0002-airflow-for-orchestration.md).
3. **Thin DAGs and a shared core**: business logic lives in `alphalens_core`, which FastAPI and Airflow both import. DAGs only orchestrate. See [ADR-0004](adr/0004-monorepo-shared-core.md).
4. **The agent orchestrates tools**: the model decides which tools to call; the pipeline is not hard-coded. See [ADR-0003](adr/0003-self-hosted-agent-loop.md).
5. **Everything is idempotent**: every write is an upsert on a natural key, so any job can be re-run safely.

---

## 4. Tech stack

### 4.1 Frontend
| Concern | Choice |
|---|---|
| Framework | Next.js 15 (App Router), React 19, TypeScript |
| UI | Tailwind CSS, shadcn/ui |
| Market charts | TradingView Lightweight Charts |
| Other charts | Recharts |
| Data fetching | TanStack Query; SSE for chat streaming |
| API types | `openapi-typescript`, generated from the FastAPI OpenAPI schema |
| E2E tests | Playwright |

### 4.2 Backend
| Concern | Choice |
|---|---|
| Runtime | Python 3.12, FastAPI, Pydantic v2 |
| ORM / migrations | SQLAlchemy 2.0 (async), Alembic |
| On-demand jobs | arq (Redis-backed, asyncio) |
| SQL guard | `sqlglot` AST allow-list and a read-only DB role |
| Parsing | `sec-parser`, BeautifulSoup, custom Item splitter |
| Numerics | pandas, numpy, scipy, statsmodels |
| Tests | pytest, pytest-asyncio, testcontainers |
| Quality | ruff, mypy, pre-commit |

### 4.3 AI
| Concern | Choice |
|---|---|
| LLM SDK | Anthropic Python SDK |
| Agent / chat model | `claude-opus-5-5` with adaptive thinking and streaming |
| Extraction model | `claude-haiku-5-5`, chosen for cost because extraction is the highest-volume LLM workload. The extraction eval must meet the §1.3 target; if it falls short, escalate low-confidence cases to `claude-opus-5-5` |
| Structured extraction | Structured Outputs (strict JSON schema) |
| Historical backfill | Message Batches API |
| Cost | Prompt caching on the system prompt and tool definitions |
| Safety | Handle `refusal` stop reasons; server-side fallback enabled |
| Embeddings | Voyage `voyage-finance-2` (1024-d) |
| Reranker | Voyage rerank (swappable for `bge-reranker`) |

### 4.4 Data and orchestration
| Concern | Choice |
|---|---|
| OLTP + vectors + FTS | PostgreSQL 16, pgvector (HNSW), tsvector (GIN) |
| Cache / queue / rate limits | Redis 7 |
| Orchestration | Apache Airflow 3 (TaskFlow, Assets, dynamic task mapping, deferrable operators) |

### 4.5 Infrastructure and operations
| Concern | Choice |
|---|---|
| Local | Docker Compose |
| Deploy, stage 1 | Vercel (web), Railway/Fly.io (API, arq, Airflow on LocalExecutor), Neon (Postgres + pgvector), Upstash Redis |
| Deploy, stage 2 | AWS: ECS Fargate, RDS Postgres, ElastiCache, MWAA; Terraform |
| CI/CD | GitHub Actions |
| Auth | Clerk |
| LLM observability | Langfuse |
| General observability | OpenTelemetry, Prometheus, Grafana, Sentry |
| Secrets | `.env` locally; AWS Secrets Manager in production (also used as the Airflow secrets backend) |

---

## 5. Data sources

| Data | Source | Notes |
|---|---|---|
| 10-K / 10-Q / 8-K | SEC EDGAR (submissions API, Archives) | Free; 10 req/s limit; a descriptive User-Agent is required |
| Financials | SEC XBRL `companyfacts` API | Free |
| News | Finnhub company news | Free tier; Polygon/Massive news can be added later |
| Material events | 8-K filings | Free; the highest-quality event source |
| Social (v2) | Reddit API | Deferred |
| Prices | Polygon/Massive EOD or Tiingo | yfinance is used only in development |
| Benchmark | SPY daily | Used for abnormal returns |

---

## 6. Pipelines (Airflow)

### 6.1 DAG dependencies
```
 sec_filings_daily ──────────────► Asset(chunks) ──────────────┐
 xbrl_financials_daily ──────────► Asset(financial_facts) ─────┤
 prices_eod ─────────────────────► Asset(prices_daily) ──┐     │
 news_ingest (every 5 min) ──────► Asset(news_events) ───┤     │
 news_backfill (manual) ─────────► Asset(news_events)    │     │
                                                         ▼     │
                     sentiment_signals (triggered by prices_daily)
                                     │                         │
                                     ▼                         ▼
                              Asset(signal_results)      nightly_eval
```

### 6.2 DAG catalogue
| DAG | Schedule (America/New_York) | Tasks |
|---|---|---|
| `sec_filings_daily` | Weekdays 18:00 | `fetch_submissions` → `download_filing` (mapped per filing) → `parse_sections` → `chunk_and_embed` → `validate` |
| `xbrl_financials_daily` | Weekdays 18:30 | `fetch_companyfacts` → `upsert_facts` → `refresh_views` → `validate` |
| `prices_eod` | Weekdays 17:30 | `fetch_ohlcv` → `validate` → `upsert` |
| `news_ingest` | Every 5 min; `max_active_runs=1`; `catchup=False` | `fetch_news` → `dedup_cluster` → `llm_extract` |
| `sentiment_signals` | Asset-scheduled on `prices_daily` | `aggregate_sentiment` → [`event_study`, `compute_ic`, `backtest`] |
| `nightly_eval` | Daily 02:00 | [`retrieval_eval`, `sql_eval`, `extraction_eval`] → `write_eval_run` → `alert_on_regression` |
| `news_backfill` | Manual; params `start`, `end` | `fetch_history` → `submit_batch` → deferrable wait → `ingest_results` |

### 6.3 Conventions
- **Idempotency**: tasks process their `data_interval` and upsert on natural keys.
- **Backfill**: historical loads reuse the daily DAGs through `airflow backfill`.
- **Pools**: `edgar_pool` keeps SEC traffic under its rate limit; `llm_pool` caps concurrent Claude calls.
- **Deferrable waits**: Batch API polling runs on the triggerer and does not hold a worker slot.
- **XCom carries IDs only**. Bulk data moves through Postgres.
- **Retries**: `retries=3` with exponential backoff.
- **Alerting**: an `on_failure_callback` posts to Slack or email.
- **Data quality**: each DAG ends with a `validate` task (row counts, nulls, date continuity). A failure blocks downstream assets.
- **Tests**: CI runs a DagBag import test and DAG structure tests.

---

## 7. Processing logic

### 7.1 Filings to chunks
1. Split each filing by Item (1, 1A, 7, 7A, 8, …). Chunks never cross a section boundary.
2. Inside a section, use a recursive splitter: about 800 tokens per chunk, with 15% overlap.
3. Tables become standalone markdown chunks with `is_table = true`.
4. Before embedding, add a contextual header to each chunk, for example `[NVDA | 10-K | FY2025 | Item 7 MD&A | Data Center]`.
5. Natural key: `(accession_no, item_code, chunk_idx)`.

### 7.2 XBRL to financials
Map about 30 core concepts (Revenue, NetIncome, EPS diluted, GrossProfit, OperatingCashFlow, …) into the long table `financial_facts`. A wide view, `v_financials_quarterly`, is exposed to text-to-SQL.

### 7.3 News to events
```
fetch → url_hash dedup → embed(title + summary)
  → cosine vs same-ticker articles from the last 72h
       ≥ 0.90 → attach to the existing event_cluster
       else   → create a new cluster
  → structured extraction (one call per new cluster) → news_events
```

Extraction schema:
```json
{
  "tickers": [{"symbol": "NVDA", "relevance": 0.95}],
  "event_type": "earnings_beat | earnings_miss | guidance_raise | guidance_cut | m_and_a | lawsuit | regulatory | product_launch | exec_change | analyst_rating | macro | other",
  "sentiment": -1.0,
  "confidence": 0.0,
  "magnitude": "low | medium | high",
  "is_new_information": true,
  "summary": "string",
  "rationale": "string"
}
```
Article text is untrusted input. The prompt treats it as data, and the strict output schema limits what an injected instruction can change.

### 7.4 Signal computation
1. **Point-in-time alignment**: an event published after 16:00 ET maps to the next trading day (`effective_trade_date`).
2. **Daily score**: `Σ sentiment × confidence × relevance × w(magnitude)`, counted once per cluster. The event count is stored alongside the score.
3. **Event study**: abnormal returns from a market model against SPY (estimation window [-250, -30]). Report CAR[-1,+1], CAR[0,+5], and CAR[0,+20], grouped by event type and sentiment sign, with t-stats.
4. **IC**: daily cross-sectional Spearman correlation of the score with N-day forward returns. Report mean IC, IC IR, and t-stat.
5. **Backtest**: quintile long-short with 1-day and 5-day rebalancing and 10 bps per side. Report Sharpe, max drawdown, and turnover.

> Expectation: a signal built from free news may be weak after costs. The aim is a methodology that is correct and honestly reported, not an impressive equity curve.

---

## 8. Retrieval (RAG)

```
query
 → query understanding (LLM): tickers, period, form types, sections
                               → metadata filter + 1–3 rewritten queries
 → hybrid retrieval (single SQL statement)
      pgvector top-50 (filtered)  +  tsvector BM25 top-50 (filtered)
      → Reciprocal Rank Fusion (k = 60)
 → rerank → top-8
 → chunks passed to Claude as document blocks with citations enabled
 → answer + cited_text → UI highlights the source span
```

Reference query:
```sql
WITH vec AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> :qvec) AS r
  FROM chunks
  WHERE company_id = ANY(:companies) AND fiscal_year >= :from_year
  ORDER BY embedding <=> :qvec LIMIT 50
),
bm25 AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, q) DESC) AS r
  FROM chunks, plainto_tsquery('english', :qtext) q
  WHERE company_id = ANY(:companies) AND fiscal_year >= :from_year AND tsv @@ q
  ORDER BY ts_rank(tsv, q) DESC LIMIT 50
)
SELECT id, SUM(1.0 / (60 + r)) AS score
FROM (SELECT * FROM vec UNION ALL SELECT * FROM bm25) t
GROUP BY id ORDER BY score DESC LIMIT 50;
```

**Why hybrid:** filings are dense with exact tokens (product names, Item numbers, figures) that pure vector search tends to miss. BM25 covers them.

---

## 9. Research agent

### 9.1 Tools
| Tool | Purpose |
|---|---|
| `search_filings(query, tickers?, form_types?, date_range?, sections?)` | Qualitative questions, through hybrid RAG |
| `query_financials(question)` | Quantitative questions, through guarded text-to-SQL |
| `search_news_events(tickers, date_range, event_types?, min_abs_sentiment?)` | News and events |
| `get_sentiment_series(ticker, date_range)` | Sentiment trend |
| `get_price_series(ticker, date_range)` | Prices |
| `get_signal_stats(ticker?, event_type?)` | Historical reaction to similar events |

### 9.2 Text-to-SQL guard
1. The model sees only the schemas of allow-listed views.
2. `sqlglot` parses the generated SQL. Only a single `SELECT` that references allow-listed views is accepted.
3. `LIMIT 500` is injected when it is missing.
4. The query runs under a read-only role with `statement_timeout = 3s`.
5. On error, the message is returned to the model, with at most 2 repair attempts.

### 9.3 Loop
- A self-hosted tool-use loop on the Messages API inside FastAPI ([ADR-0003](adr/0003-self-hosted-agent-loop.md)).
- Tool status events and tokens stream to the client over SSE.
- At most 8 tool calls per turn.
- Prompt caching on the system prompt and tool definitions.
- Every call is traced in Langfuse and recorded in `llm_calls`.
- `refusal` stop reasons are handled, with server-side fallback enabled.

---

## 10. Data model

```sql
-- reference
companies(id, ticker UNIQUE, cik UNIQUE, name, sector, industry, in_universe)

-- filings
filings(id, company_id, accession_no UNIQUE, form_type, filed_at,
        period_of_report, fiscal_year, fiscal_period, url, status)
filing_sections(id, filing_id, item_code, title, text, token_count)
chunks(id, section_id, company_id, chunk_idx, content, context_header,
       is_table, token_count,
       embedding vector(1024),          -- HNSW (vector_cosine_ops)
       tsv tsvector,                    -- GIN
       form_type, fiscal_year, filed_at)  -- denormalized for filtering

-- financials
financial_facts(id, company_id, concept, value, unit, period_start, period_end,
                fiscal_year, fiscal_period, form_type, filed_at, accession_no,
                UNIQUE(company_id, concept, period_end, fiscal_period, accession_no))
VIEW v_financials_quarterly

-- prices
prices_daily(company_id, date, open, high, low, close, adj_close, volume,
             PRIMARY KEY (company_id, date))

-- news and signals
news_articles(id, source, url, url_hash UNIQUE, title, summary, body,
              published_at, fetched_at, embedding vector(1024), cluster_id)
event_clusters(id, first_seen_at, canonical_article_id, article_count)
news_events(id, cluster_id, company_id, event_type, sentiment, confidence,
            magnitude, is_new_information, relevance, summary, rationale,
            model, prompt_version, extracted_at, effective_trade_date)
sentiment_daily(company_id, trade_date, score, event_count,
                PRIMARY KEY (company_id, trade_date))
signal_results(id, run_id, signal_name, params jsonb, metric, value, computed_at)
backtest_jobs(id, user_id, params jsonb, status, result jsonb, created_at)

-- users and chat
users(id, auth_provider_id UNIQUE, email, created_at)
watchlists(user_id, company_id, added_at, PRIMARY KEY (user_id, company_id))
conversations(id, user_id, title, created_at)
messages(id, conversation_id, role, content jsonb, created_at)
message_citations(id, message_id, chunk_id NULL, news_event_id NULL,
                  cited_text, position)

-- operations
llm_calls(id, purpose, model, input_tokens, output_tokens, cache_read_tokens,
          latency_ms, cost_usd, trace_id, created_at)
eval_runs(id, suite, git_sha, metrics jsonb, created_at)
```
Pipeline run history comes from the Airflow metadata DB and is not duplicated in the app DB.

---

## 11. API

```
POST   /api/chat/conversations
POST   /api/chat/conversations/{id}/messages          → text/event-stream
GET    /api/chat/conversations/{id}

GET    /api/companies?q=
GET    /api/companies/{ticker}
GET    /api/companies/{ticker}/financials
GET    /api/companies/{ticker}/filings
GET    /api/companies/{ticker}/prices?from&to
GET    /api/companies/{ticker}/sentiment?from&to

GET    /api/events?tickers&types&from&to&min_sentiment&cursor
GET    /api/signals/event-study?event_type&horizon
GET    /api/signals/ic?horizon
POST   /api/signals/backtest                         → 202 { job_id }
GET    /api/signals/backtest/{job_id}

GET    /api/watchlist     POST /api/watchlist     DELETE /api/watchlist/{ticker}

GET    /api/admin/evals   /api/admin/costs   /api/admin/pipelines
GET    /healthz   /readyz   /metrics
```

---

## 12. Frontend

| Route | Content |
|---|---|
| `/research` | Conversation list · streaming chat with tool status · citation panel |
| `/company/[ticker]` | Candlestick chart with event markers · sentiment series · financials · filings · news |
| `/feed` | Live events, filterable by watchlist, type, and sentiment |
| `/signals` | CAR curves · IC series · backtest PnL with parameters |
| `/admin` | Eval trends · LLM cost · DAG status |

---

## 13. Evaluation

| Suite | Dataset | Metrics |
|---|---|---|
| Retrieval | ~100 hand-labelled (question → relevant chunks) | Recall@8, MRR; ablation: vector, BM25, hybrid, hybrid + rerank |
| Answer | Same set, with reference answers | Faithfulness (LLM judge), citation precision |
| Text-to-SQL | ~50 (question → expected result) | Execution accuracy |
| Extraction | ~200 hand-labelled articles | `event_type` macro-F1; sentiment correlation with labels |
| Signal | — | IC, t-stat, net Sharpe (reported, not gated) |

- **CI gate**: a small eval subset runs on every PR. A retrieval Recall@8 drop of more than 5 points fails the build.
- **Nightly**: the full suites run in `nightly_eval`. Results go to `eval_runs` and are charted in Admin.

---

## 14. Non-functional requirements

| Area | Approach |
|---|---|
| Reliability | Idempotent upserts; Airflow retries and backfill; validation gates on assets |
| Rate limiting | Airflow pools for external APIs; Redis token bucket per user |
| Cost | Prompt caching; Batch API for backfills; per-user daily token quota; `llm_calls` dashboard |
| Security | SQL guard; JWT verification; per-user row scoping; secrets in a manager; CORS allow-list |
| Prompt injection | Retrieved and ingested text is treated as data; strict output schemas |
| Reproducibility | `model` and `prompt_version` stored with every extracted event |
| Observability | One trace per request across API → agent → tools → DB; Langfuse for LLM calls; Sentry for errors |
| Compliance | "Not investment advice" disclaimer; SEC fair-access User-Agent |

---

## 15. Repository layout

```
alphalens/
├── apps/
│   ├── web/                    # Next.js + React + TypeScript
│   └── api/                    # FastAPI
│       ├── app/{routers,agent,auth}/
│       └── alembic/
├── packages/
│   └── core/                   # alphalens_core
│       ├── ingestion/          # sec, xbrl, news, prices
│       ├── rag/                # chunking, retrieval, rerank
│       ├── extraction/
│       ├── signals/
│       ├── sql_guard/
│       ├── db/                 # SQLAlchemy models
│       └── llm/                # Claude client wrapper, cost accounting
├── airflow/{dags,plugins,tests}/
├── tasks/                      # arq jobs
├── evals/{datasets,suites}/
├── infra/
│   ├── docker-compose.yml
│   └── terraform/
├── .github/workflows/
└── docs/{DESIGN.md,adr/}
```

---

## 16. Delivery plan

The approach is a walking skeleton first: one thin end-to-end slice, deployed in week 1, followed by one vertical slice per week.

| Week | Deliverable |
|---|---|
| W1 | **Walking skeleton, deployed.** Monorepo, docker-compose (with Airflow), CI/CD, first migration. One company and one 10-K go through ingest → chunk → embed → retrieve → Claude answer → UI. |
| W2 | Retrieval quality: section-aware chunking, hybrid search, rerank, retrieval eval baseline. S&P 100 filings and XBRL ingestion. |
| W3 | Agent: tools, guarded text-to-SQL, SSE streaming, citations panel. |
| W4 | News: `news_ingest`, `prices_eod`, `news_backfill`; extraction eval. |
| W5 | Signals: `sentiment_signals`, arq on-demand backtests; Company and Signals pages. |
| W6 | Auth, watchlist, feed, admin; `nightly_eval`; CI eval gate. |
| W7 | Hardening: observability, quotas, load test, Terraform (stage 2), README and demo. |

---

## 17. Open decisions

Defaults are in bold and will be used unless changed.

| # | Decision | Options | Default |
|---|---|---|---|
| D1 | Extraction model | Opus 5.5 ($4 / $20 per MTok) · Haiku 5.5 ($0.10 / $0.50 per MTok) | **Decided: Haiku 5.5** (cost). Agent/chat stays on Opus 5.5 |
| D2 | Agent hosting | Self-hosted loop · Managed Agents (beta; billed per session runtime on top of tokens) | **Self-hosted loop** |
| D3 | Deployment | Stage 1 PaaS · Stage 2 AWS | **Stage 1 first, then Terraform in W7** |
| D4 | News sources | Finnhub · Polygon/Massive · Reddit | **Finnhub + 8-K; Reddit in v2** |
| D5 | Auth | Clerk · NextAuth | **Clerk** |
| D6 | Embeddings | Voyage (paid) · bge (local, free) | **Voyage `voyage-finance-2`** |

---

## 18. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| 10-K HTML is inconsistent across filers | Bad section splits hurt retrieval | Parser fallback to heading heuristics; validation task flags filings with missing Items |
| XBRL concept drift (companies use different tags) | Missing or wrong financials | Concept alias map; coverage report per company |
| Free news coverage is thin | Weak signal | Use 8-K as the primary event source; report coverage alongside results |
| Look-ahead bias | Inflated backtest | `effective_trade_date`; point-in-time tests in CI |
| LLM cost overrun during backfill | Budget | Batch API; per-DAG token budget; `llm_calls` alerts |
| Prompt injection via news | Corrupted events | Data-only prompting; strict schema; extraction eval includes adversarial samples |
