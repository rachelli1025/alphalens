"""Research agent tools (DESIGN.md §9.1): definitions sent to Claude and their implementations.

search_filings returns `search_result` blocks with citations enabled, so the final answer
cites exact passages just like /api/ask. query_financials runs model-written SQL through
the guard; rejections and database errors come back as error results the model can fix.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from alphalens_core.agent.sql_guard import SqlRejected, run_query
from alphalens_core.embeddings import Embedder
from alphalens_core.reranking import Reranker
from alphalens_core.retrieval import RetrievedChunk, rerank_search

MAX_RESULT_ROWS_SHOWN = 100  # rows sent back to the model; more would only cost tokens

V_FINANCIALS_SCHEMA = """\
View v_financials: one row per company and fiscal period.
  ticker TEXT, fiscal_year INT, fiscal_period TEXT ('FY', 'Q1', 'Q2', 'Q3'; Q4 is not \
available), period_end DATE,
  revenue, cost_of_revenue, gross_profit, rd_expense, sga_expense, operating_income,
  interest_expense, income_tax, net_income, operating_cash_flow, capex,
  depreciation_amortization, share_repurchases, dividends_paid   -- USD over the period
  cash, current_assets, total_assets, current_liabilities, total_liabilities,
  long_term_debt, inventory, stockholders_equity                 -- USD at period_end
  eps_basic, eps_diluted                                         -- USD per share
  diluted_shares                                                 -- shares
Amounts are in whole US dollars (divide by 1e9 for billions). NULL means not reported \
(banks have no gross_profit or inventory).
fiscal_year is each company's own fiscal year: NVDA and WMT end in January, MSFT in \
June, AAPL in September, most others in December. Compare companies by period_end when \
calendar alignment matters."""

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_filings",
        "description": (
            "Search the text of SEC 10-K filings (risk factors, business description, "
            "MD&A, legal proceedings, notes). Use for qualitative questions: what a company "
            "says, why something changed, risks, strategy. Returns passages you can cite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, phrased as a search query.",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these companies. Omit to search all.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "query_financials",
        "description": (
            "Run one read-only PostgreSQL SELECT over reported financial statement data "
            "(from SEC XBRL). Use for numbers: revenue, margins, growth, comparisons, "
            "rankings. Only the view below exists; results are capped at 500 rows.\n\n"
            + V_FINANCIALS_SCHEMA
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement."},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class ToolContext:
    """Per-run state: database access plus every chunk returned so far, so citations in
    the final answer can be mapped back to chunks."""

    session: Session
    embedder: Embedder
    reranker: Reranker
    chunks: dict[int, RetrievedChunk] = field(default_factory=dict)
    sql_queries: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolOutput:
    content: list[dict[str, Any]] | str
    is_error: bool = False
    summary: str = ""  # one line for logs and the UI's tool status
    sql: str | None = None  # query_financials: the statement actually executed


def chunk_source(chunk: RetrievedChunk) -> str:
    return f"{chunk.url}#chunk-{chunk.chunk_id}"


def chunk_id_from_source(source: str) -> int | None:
    m = re.search(r"#chunk-(\d+)$", source)
    return int(m.group(1)) if m else None


def _search_filings(ctx: ToolContext, query: str, tickers: list[str] | None = None) -> ToolOutput:
    chunks = rerank_search(ctx.session, ctx.embedder, ctx.reranker, query, tickers=tickers, k=8)
    if not chunks:
        return ToolOutput("No matching passages.", summary=f"search '{query}': 0 passages")
    for c in chunks:
        ctx.chunks[c.chunk_id] = c
    blocks = [
        {
            "type": "search_result",
            "source": chunk_source(c),
            "title": c.context_header,
            "content": [{"type": "text", "text": c.content}],
            "citations": {"enabled": True},
        }
        for c in chunks
    ]
    companies = sorted({c.ticker for c in chunks})
    return ToolOutput(blocks, summary=f"search '{query}': {len(chunks)} passages from {companies}")


def _query_financials(ctx: ToolContext, sql: str) -> ToolOutput:
    try:
        result = run_query(ctx.session, sql)
    except SqlRejected as exc:
        return ToolOutput(f"Query rejected: {exc}", is_error=True, summary="SQL rejected")
    except DBAPIError as exc:
        message = str(exc.orig).splitlines()[0] if exc.orig else str(exc)
        return ToolOutput(f"Database error: {message}", is_error=True, summary="SQL error")
    ctx.sql_queries.append(result.sql)
    shown = result.rows[:MAX_RESULT_ROWS_SHOWN]
    payload = {
        "sql": result.sql,
        "columns": result.columns,
        "rows": shown,
        "row_count": len(result.rows),
        "note": "rows truncated" if len(result.rows) > len(shown) or result.truncated else "",
    }
    return ToolOutput(
        json.dumps(payload, default=str),
        summary=f"SQL: {len(result.rows)} rows",
        sql=result.sql,
    )


def run_tool(ctx: ToolContext, name: str, tool_input: dict[str, Any]) -> ToolOutput:
    if name == "search_filings":
        return _search_filings(ctx, tool_input["query"], tool_input.get("tickers") or None)
    if name == "query_financials":
        return _query_financials(ctx, tool_input["sql"])
    return ToolOutput(f"Unknown tool {name!r}", is_error=True, summary=f"unknown tool {name}")
