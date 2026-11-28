"""financial_facts (XBRL) and v_financials view

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copy of alphalens_core.xbrl.METRICS names at this revision. A migration must not
# import app code: later edits to METRICS would silently change what this file creates.
METRIC_COLUMNS = [
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "rd_expense",
    "sga_expense",
    "operating_income",
    "interest_expense",
    "income_tax",
    "net_income",
    "eps_basic",
    "eps_diluted",
    "diluted_shares",
    "operating_cash_flow",
    "capex",
    "depreciation_amortization",
    "share_repurchases",
    "dividends_paid",
    "cash",
    "current_assets",
    "total_assets",
    "current_liabilities",
    "total_liabilities",
    "long_term_debt",
    "inventory",
    "stockholders_equity",
]

# One row per company and fiscal period (FY, Q1, Q2, Q3), one column per metric.
# Q4 is not reported on its own in XBRL (it is FY minus Q1-Q3) and is not derived yet.
# For each value, the latest filed fact wins, so restatements replace earlier numbers.
_columns = ",\n    ".join(
    f"max(l.value) FILTER (WHERE l.concept = '{m}') AS {m}" for m in METRIC_COLUMNS
)
VIEW_SQL = f"""
CREATE VIEW v_financials AS
WITH latest AS (
    SELECT DISTINCT ON (f.company_id, f.concept, f.fiscal_year, f.fiscal_period)
           f.company_id, f.concept, f.fiscal_year, f.fiscal_period, f.period_end, f.value
    FROM financial_facts f
    WHERE f.fiscal_year IS NOT NULL
    ORDER BY f.company_id, f.concept, f.fiscal_year, f.fiscal_period, f.filed_at DESC
)
SELECT c.ticker,
    l.fiscal_year,
    l.fiscal_period,
    max(l.period_end) AS period_end,
    {_columns}
FROM latest l
JOIN companies c ON c.id = l.company_id
GROUP BY c.ticker, l.fiscal_year, l.fiscal_period
"""


def upgrade() -> None:
    op.create_table(
        "financial_facts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("concept", sa.String(64), nullable=False),
        sa.Column("source_tag", sa.String(128), nullable=False),
        sa.Column("value", sa.Numeric(24, 4), nullable=False),
        sa.Column("unit", sa.String(16), nullable=False),
        sa.Column("period_start", sa.Date()),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("fiscal_year", sa.Integer()),
        sa.Column("fiscal_period", sa.String(4)),
        sa.Column("form_type", sa.String(16), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column("accession_no", sa.String(25), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_financial_facts"),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_financial_facts_company_id_companies",
        ),
        sa.UniqueConstraint(
            "company_id",
            "concept",
            "period_start",
            "period_end",
            "accession_no",
            name="uq_financial_facts_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(
        "ix_financial_facts_lookup",
        "financial_facts",
        ["company_id", "concept", "fiscal_year"],
    )
    op.execute(VIEW_SQL)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_financials")
    op.drop_index("ix_financial_facts_lookup", table_name="financial_facts")
    op.drop_table("financial_facts")
