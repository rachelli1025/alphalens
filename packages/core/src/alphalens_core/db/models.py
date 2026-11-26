"""Core tables for W1: companies, filings, sections, chunks. See docs/DESIGN.md §10."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from alphalens_core.db.base import Base, TimestampMixin

EMBEDDING_DIM = 1024  # voyage-finance-2


class Company(TimestampMixin, Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    cik: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)  # zero-padded
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(64))
    industry: Mapped[str | None] = mapped_column(String(128))
    in_universe: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    filings: Mapped[list["Filing"]] = relationship(back_populates="company")


class Filing(TimestampMixin, Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    accession_no: Mapped[str] = mapped_column(String(25), unique=True, nullable=False)
    form_type: Mapped[str] = mapped_column(String(16), nullable=False)
    filed_at: Mapped[date] = mapped_column(Date, nullable=False)
    period_of_report: Mapped[date | None] = mapped_column(Date)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_period: Mapped[str | None] = mapped_column(String(4))  # FY, Q1..Q4
    url: Mapped[str] = mapped_column(Text, nullable=False)
    # pending -> downloaded -> parsed -> embedded | failed
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)

    company: Mapped[Company] = relationship(back_populates="filings")
    sections: Mapped[list["FilingSection"]] = relationship(back_populates="filing")

    __table_args__ = (Index("ix_filings_company_filed", "company_id", "filed_at"),)


class FilingSection(Base):
    __tablename__ = "filing_sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"), nullable=False
    )
    item_code: Mapped[str] = mapped_column(String(8), nullable=False)  # "1", "1A", "7"
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)

    filing: Mapped[Filing] = relationship(back_populates="sections")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="section")

    __table_args__ = (UniqueConstraint("filing_id", "item_code"),)


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("filing_sections.id", ondelete="CASCADE"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    chunk_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_header: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_table: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # BM25 side of hybrid search; maintained by Postgres.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', context_header || ' ' || content)", persisted=True),
    )

    # Denormalized for fast filtering before vector search (DESIGN.md §8).
    form_type: Mapped[str] = mapped_column(String(16), nullable=False)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    filed_at: Mapped[date] = mapped_column(Date, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    section: Mapped[FilingSection] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("section_id", "chunk_idx"),
        Index("ix_chunks_filter", "company_id", "fiscal_year", "form_type"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )
