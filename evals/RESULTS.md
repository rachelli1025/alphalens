# Retrieval eval results

Run with `uv run alphalens eval-retrieval [options]`. k = 8 throughout. Until W6 stores
runs in `eval_runs`, record each run that changes a default here.

## Golden set v1: 15 Apple cases

| Date | Corpus | Search scope | hybrid | vector | keyword |
|---|---|---|---|---|---|
| 2026-11-28 | AAPL only (97 chunks) | AAPL | 1.00 / 0.84 | 1.00 / 0.87 | 0.87 / 0.58 |
| 2026-11-28 | 11 companies (1,914 chunks) | AAPL | 1.00 / 0.84 | 1.00 / 0.87 | 0.87 / 0.58 |
| 2026-11-28 | 11 companies | all companies (`--ignore-tickers`) | 0.87 / 0.63 | 1.00 / 0.87 | 0.40 / 0.24 |
| 2026-11-28 | 11 companies | resolved by Haiku (`--auto-tickers`) | 1.00 / 0.84 | 1.00 / 0.87 | 0.80 / 0.52 |

Cells are Recall@8 / MRR.

Takeaways: on one filing every mode saturates. Without a company filter the keyword
side cannot tell companies apart (filings say "the Company", and `ts_rank_cd` has no
IDF), which drags hybrid below vector; resolving tickers from the question fixes it.

## Golden set v2: 35 cases, 11 companies (`--auto-tickers`)

| Date | Change | hybrid | vector | keyword | rerank |
|---|---|---|---|---|---|
| 2026-11-28 | baseline (nvda-china labels completed) | 0.89 / 0.66 | 0.91 / 0.71 | 0.77 / 0.43 | |
| 2026-11-28 | + `rerank-2.5` over 50 hybrid candidates | 0.89 / 0.66 | 0.91 / 0.71 | 0.77 / 0.43 | **1.00 / 0.90** |

Takeaways: the right passage was usually among the 50 candidates but ranked low;
the cross-encoder fixes the order (all 4 hybrid misses recovered, 29/35 at rank 1).
Every relevant chunk is in the top 5 with rerank, so k = 5 is worth testing against
the answer eval. Rerank is weaker on cross-company questions (cross-china-trade at 5):
query decomposition per company is the next retrieval change.
