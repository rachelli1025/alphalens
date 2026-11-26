"""initial filings schema: companies, filings, filing_sections, chunks

Revision ID: 0001
Revises:
Create Date: 2026-11-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("cik", sa.String(10), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sector", sa.String(64)),
        sa.Column("industry", sa.String(128)),
        sa.Column("in_universe", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_companies"),
        sa.UniqueConstraint("ticker", name="uq_companies_ticker"),
        sa.UniqueConstraint("cik", name="uq_companies_cik"),
    )

    op.create_table(
        "filings",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("accession_no", sa.String(25), nullable=False),
        sa.Column("form_type", sa.String(16), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column("period_of_report", sa.Date()),
        sa.Column("fiscal_year", sa.Integer()),
        sa.Column("fiscal_period", sa.String(4)),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_filings"),
        sa.ForeignKeyConstraint(
            ["company_id"], ["companies.id"], name="fk_filings_company_id_companies"
        ),
        sa.UniqueConstraint("accession_no", name="uq_filings_accession_no"),
    )
    op.create_index("ix_filings_company_filed", "filings", ["company_id", "filed_at"])

    op.create_table(
        "filing_sections",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("filing_id", sa.BigInteger(), nullable=False),
        sa.Column("item_code", sa.String(8), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer()),
        sa.PrimaryKeyConstraint("id", name="pk_filing_sections"),
        sa.ForeignKeyConstraint(
            ["filing_id"],
            ["filings.id"],
            ondelete="CASCADE",
            name="fk_filing_sections_filing_id_filings",
        ),
        sa.UniqueConstraint("filing_id", "item_code", name="uq_filing_sections_filing_id"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("section_id", sa.BigInteger(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("chunk_idx", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_header", sa.Text(), nullable=False),
        sa.Column("is_table", sa.Boolean(), nullable=False),
        sa.Column("token_count", sa.Integer()),
        sa.Column("embedding", Vector(1024)),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', context_header || ' ' || content)", persisted=True),
        ),
        sa.Column("form_type", sa.String(16), nullable=False),
        sa.Column("fiscal_year", sa.Integer()),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["filing_sections.id"],
            ondelete="CASCADE",
            name="fk_chunks_section_id_filing_sections",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["companies.id"], name="fk_chunks_company_id_companies"
        ),
        sa.UniqueConstraint("section_id", "chunk_idx", name="uq_chunks_section_id"),
    )
    op.create_index("ix_chunks_filter", "chunks", ["company_id", "fiscal_year", "form_type"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("filing_sections")
    op.drop_index("ix_filings_company_filed", table_name="filings")
    op.drop_table("filings")
    op.drop_table("companies")
