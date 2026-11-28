from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from alphalens_core.agent.sql_guard import MAX_ROWS, SqlRejected, run_query, validate


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ticker, revenue FROM v_financials WHERE fiscal_year = 2025",
        "SELECT ticker, revenue::float / 1e9 AS rev FROM v_financials",
        "WITH t AS (SELECT * FROM v_financials) SELECT * FROM t",
        "SELECT 1 FROM v_financials UNION SELECT 2 FROM v_financials",
        "select ticker from V_FINANCIALS",  # case-insensitive names
    ],
)
def test_read_queries_on_allowed_view_pass_and_get_a_limit(sql: str) -> None:
    assert validate(sql).endswith(f"LIMIT {MAX_ROWS}")


def test_small_limits_are_kept_and_large_ones_capped() -> None:
    assert validate("SELECT * FROM v_financials LIMIT 5").endswith("LIMIT 5")
    assert validate("SELECT * FROM v_financials LIMIT 100000").endswith(f"LIMIT {MAX_ROWS}")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM v_financials", "only SELECT"),
        ("DROP TABLE companies", "only SELECT"),
        ("UPDATE v_financials SET revenue = 0", "only SELECT"),
        ("SELECT * FROM companies", "not allowed"),
        ("SELECT * FROM v_financials WHERE ticker IN (SELECT ticker FROM chunks)", "not allowed"),
        ("SELECT * FROM public.v_financials", "unqualified"),
        ("SELECT pg_sleep(10) FROM v_financials", "pg_sleep"),
        ("SELECT current_setting('server_version') FROM v_financials", "current_setting"),
        ("SELECT * INTO evil FROM v_financials", "SELECT INTO"),
        ("SELECT * FROM v_financials FOR UPDATE", "locks"),
        ("SELECT * FROM generate_series(1, 10) AS g", "table functions"),
        ("SELECT * FROM v_financials; DELETE FROM chunks", "exactly one"),
        ("not sql at all", "could not parse"),
    ],
)
def test_unsafe_queries_are_rejected_with_a_reason(sql: str, reason: str) -> None:
    with pytest.raises(SqlRejected, match=reason):
        validate(sql)


def test_parse_errors_have_no_terminal_colour_codes() -> None:
    with pytest.raises(SqlRejected) as info:
        validate("not sql at all")
    assert "\x1b" not in str(info.value)


# ---- against a real Postgres (CI); the view exists after migrations ----


@pytest.fixture
def session(pg_engine: Engine) -> Iterator[Session]:
    with Session(pg_engine) as s:
        yield s


def test_run_query_returns_columns_and_rows(session: Session) -> None:
    result = run_query(
        session, "SELECT 1::numeric AS one, DATE '2025-01-31' AS d FROM v_financials LIMIT 1"
    )
    assert result.columns == ["one", "d"]
    assert result.rows in ([], [[1.0, "2025-01-31"]])  # empty view in a fresh CI database
    assert result.sql.endswith("LIMIT 1")


def test_slow_queries_hit_the_statement_timeout(session: Session) -> None:
    # Independent of the view's contents (empty in a fresh CI database): counts to 1e9.
    slow = (
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 1000000000) "
        "SELECT count(*) FROM t"
    )
    with pytest.raises(DBAPIError, match="statement timeout"):
        run_query(session, slow)
