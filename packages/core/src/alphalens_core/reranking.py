"""Voyage rerank over plain HTTP (https://docs.voyageai.com/reference/reranker-api).

A reranker is a cross-encoder: it reads the query and one document together and
scores how well the document answers the query. That is more precise than comparing
separately computed embeddings, but too slow to run over the whole corpus, so it
re-orders the ~50 candidates hybrid search returns (DESIGN.md §8).
"""

import time
from types import TracebackType
from typing import Protocol, Self

import httpx

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """(index into documents, relevance score), best first, at most top_k items."""
        ...


class VoyageReranker:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise RerankError("VOYAGE_API_KEY is not set")
        self.model = model
        self.max_retries = max_retries
        self.total_tokens = 0
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = {
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_k": min(top_k, len(documents)),
        }
        for attempt in range(self.max_retries + 1):
            resp = self._http.post(VOYAGE_RERANK_URL, json=payload)
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                break
            time.sleep(2**attempt)
        if resp.status_code != 200:
            raise RerankError(f"Voyage rerank HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        self.total_tokens += body.get("usage", {}).get("total_tokens", 0)
        rows = sorted(body["data"], key=lambda d: d["relevance_score"], reverse=True)
        return [(row["index"], row["relevance_score"]) for row in rows]
