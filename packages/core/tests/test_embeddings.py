import json

import httpx
import pytest

from alphalens_core.embeddings import EmbeddingError, VoyageEmbedder


def _embedder(requests: list[dict[str, object]], batch_size: int = 2) -> VoyageEmbedder:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        requests.append(body)
        n = len(body["input"])
        # Return rows out of order: the client must sort by index.
        data = [{"index": i, "embedding": [float(len(body["input"][i]))]} for i in range(n)]
        return httpx.Response(
            200, json={"data": list(reversed(data)), "usage": {"total_tokens": 10 * n}}
        )

    return VoyageEmbedder(
        "key", "voyage-finance-2", batch_size=batch_size, transport=httpx.MockTransport(handler)
    )


def test_requires_api_key() -> None:
    with pytest.raises(EmbeddingError):
        VoyageEmbedder("", "voyage-finance-2")


def test_batches_and_preserves_order() -> None:
    requests: list[dict[str, object]] = []
    emb = _embedder(requests, batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = emb.embed(texts, "document")
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [len(r["input"]) for r in requests] == [2, 2, 1]  # type: ignore[arg-type]
    assert all(r["model"] == "voyage-finance-2" for r in requests)
    assert all(r["input_type"] == "document" for r in requests)
    assert emb.total_tokens == 50


def test_query_input_type_is_sent() -> None:
    requests: list[dict[str, object]] = []
    _embedder(requests).embed(["what are the risks?"], "query")
    assert requests[0]["input_type"] == "query"


def test_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alphalens_core.embeddings.time.sleep", lambda _s: None)
    emb = VoyageEmbedder(
        "key",
        "voyage-finance-2",
        max_retries=1,
        transport=httpx.MockTransport(lambda _r: httpx.Response(401, text="bad key")),
    )
    with pytest.raises(EmbeddingError, match="401"):
        emb.embed(["x"])
