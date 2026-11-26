"""Voyage AI embeddings over plain HTTP (https://docs.voyageai.com/reference/embeddings-api)."""

import time
from types import TracebackType
from typing import Literal, Protocol, Self

import httpx

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
RETRY_STATUSES = {429, 500, 502, 503, 504}

InputType = Literal["document", "query"]


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class VoyageEmbedder:
    """Batches requests; documents and queries use different `input_type`s."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        batch_size: int = 64,  # 64 x ~800 tokens stays under Voyage's per-request token cap
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("VOYAGE_API_KEY is not set")
        self.model = model
        self.batch_size = batch_size
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

    def _post(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        payload = {"input": batch, "model": self.model, "input_type": input_type}
        for attempt in range(self.max_retries + 1):
            resp = self._http.post(VOYAGE_URL, json=payload)
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                break
            time.sleep(2**attempt)
        if resp.status_code != 200:
            raise EmbeddingError(f"Voyage HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        self.total_tokens += body.get("usage", {}).get("total_tokens", 0)
        rows = sorted(body["data"], key=lambda d: d["index"])
        if len(rows) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(rows)}")
        return [row["embedding"] for row in rows]

    def embed(self, texts: list[str], input_type: InputType = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i : i + self.batch_size], input_type))
        return out
