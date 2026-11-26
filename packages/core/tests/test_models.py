from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from alphalens_core.db import Base, Chunk


def test_expected_tables_registered() -> None:
    assert {"companies", "filings", "filing_sections", "chunks"} <= set(Base.metadata.tables)


def test_chunks_ddl_has_vector_and_generated_tsvector() -> None:
    ddl = str(CreateTable(Chunk.__table__).compile(dialect=postgresql.dialect()))
    assert "embedding VECTOR(1024)" in ddl
    assert "GENERATED ALWAYS AS (to_tsvector('english'" in ddl


def test_chunks_has_hnsw_cosine_index() -> None:
    idx = next(i for i in Chunk.__table__.indexes if i.name == "ix_chunks_embedding_hnsw")
    ddl = str(CreateIndex(idx).compile(dialect=postgresql.dialect()))
    assert "USING hnsw" in ddl
    assert "vector_cosine_ops" in ddl
