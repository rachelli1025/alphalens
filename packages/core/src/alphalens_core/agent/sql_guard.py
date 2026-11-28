"""Guarded text-to-SQL for the research agent (DESIGN.md §9.2).

The model writes SQL against allow-listed views; nothing it writes runs unchecked:

1. Parse with sqlglot. Exactly one statement, and it must be a read query (SELECT,
   optionally with CTEs or UNION). DML, DDL, SELECT INTO and row locks are rejected.
2. Every table reference must be an allow-listed view or a CTE defined in the query.
3. Functions that touch the server (pg_sleep, pg_read_file, dblink, set_config, ...)
   are rejected.
4. An outer LIMIT is added when missing and capped at MAX_ROWS.
5. It runs in a READ ONLY transaction with a short statement_timeout.

Defence in depth: step 5 alone would stop writes, but the earlier steps give the model
a clear error to repair instead of a database exception, and keep it off other tables.
"""

import datetime as dt
import decimal
import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlglot import exp

ALLOWED_VIEWS = frozenset({"v_financials"})
MAX_ROWS = 500
STATEMENT_TIMEOUT_MS = 3000

BLOCKED_FUNCTION_PREFIXES = ("pg_", "dblink", "lo_", "current_setting", "set_config", "query_to")


class SqlRejected(ValueError):
    """The query failed validation; the message is meant to be shown to the model."""


@dataclass(frozen=True)
class QueryResult:
    sql: str  # the statement actually executed (with LIMIT)
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool  # True when MAX_ROWS rows came back and more may exist


def validate(sql: str) -> str:
    """Return the safe, LIMITed SQL to run, or raise SqlRejected with a reason."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as exc:
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))  # drop terminal colour codes
        raise SqlRejected(f"could not parse SQL: {message}") from exc
    if len(statements) != 1:
        raise SqlRejected("send exactly one SQL statement")
    tree = statements[0]

    if not isinstance(tree, exp.Select | exp.Union):
        raise SqlRejected("only SELECT queries are allowed")
    for node in tree.walk():
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Create | exp.Drop):
            raise SqlRejected("only SELECT queries are allowed")
        if isinstance(node, exp.Into):
            raise SqlRejected("SELECT INTO is not allowed")
        if isinstance(node, exp.Lock):
            raise SqlRejected("row locks (FOR UPDATE / FOR SHARE) are not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name:  # FROM generate_series(...) and other table functions
            raise SqlRejected("table functions are not allowed in FROM")
        if table.db and name not in cte_names:
            raise SqlRejected(f"use unqualified view names, not {table.sql()}")
        if name not in ALLOWED_VIEWS and name not in cte_names:
            allowed = ", ".join(sorted(ALLOWED_VIEWS))
            raise SqlRejected(f"table {table.name!r} is not allowed; query only: {allowed}")

    for func in tree.find_all(exp.Func):
        # Unknown functions parse as Anonymous and keep their written name.
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name.startswith(BLOCKED_FUNCTION_PREFIXES):
            raise SqlRejected(f"function {name}() is not allowed")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(MAX_ROWS)
    else:
        value = limit.expression
        if not isinstance(value, exp.Literal) or not value.is_int or int(value.this) > MAX_ROWS:
            tree = tree.limit(MAX_ROWS)
    return tree.sql(dialect="postgres")


def _jsonable(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    return value


def run_query(session: Session, sql: str) -> QueryResult:
    """Validate, then execute read-only with a timeout. Raises SqlRejected or a DB error."""
    safe_sql = validate(sql)
    try:
        session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
        # exec_driver_sql sends the SQL as is: text() would read "::float" casts in
        # model-written SQL as ":float" bind parameters.
        result = session.connection().exec_driver_sql(safe_sql)
        columns = list(result.keys())
        rows = [[_jsonable(v) for v in row] for row in result.fetchall()]
    finally:
        session.rollback()  # ends the read-only transaction either way; nothing to commit
    return QueryResult(sql=safe_sql, columns=columns, rows=rows, truncated=len(rows) >= MAX_ROWS)
