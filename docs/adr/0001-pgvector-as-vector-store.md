# ADR-0001: Use pgvector in Postgres as the vector store

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
The system needs vector search over filing chunks (about 1M rows at S&P 100 scale) and news articles. It also needs metadata filters (ticker, fiscal year, form type), BM25 keyword search, and joins to relational tables.

## Decision
Store embeddings in Postgres with pgvector (HNSW index). Use tsvector with a GIN index for BM25. Run hybrid retrieval as a single SQL statement fused with Reciprocal Rank Fusion. Access the store through a `VectorStore` interface in `alphalens_core.rag` (deferred, see Amendment).

## Consequences
- One database to run, back up, and keep consistent. No dual-write synchronization.
- Filters, joins, and transactions work natively.
- At much larger scale (tens of millions of vectors, high QPS), a dedicated engine such as Qdrant may be needed. The `VectorStore` interface keeps that migration contained.

## Alternatives considered
- **Qdrant / Weaviate / Pinecone:** stronger at large scale, but they add a service and a sync path, and hybrid fusion moves into application code.

## Amendment (2026-11-26): no `VectorStore` interface yet
W1 implements retrieval as a plain function, `alphalens_core.retrieval.hybrid_search`, which runs the hybrid SQL directly. There is only one store, so an interface would add indirection with no second implementation to justify it.

The migration path in Consequences still holds: callers depend only on `hybrid_search(session, embedder, question, tickers=..., k=...)` returning `RetrievedChunk`s. If a second engine is ever needed, extract a `VectorStore` protocol with that signature at that point.
