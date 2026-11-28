import json

import httpx
import pytest

from alphalens_core.reranking import RerankError, VoyageReranker


def _reranker(requests: list[dict[str, object]], status: int = 200) -> VoyageReranker:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        requests.append(body)
        if status != 200:
            return httpx.Response(status, text="nope")
        # Score by document length so the expected order is obvious; return unsorted.
        docs = body["documents"]
        data = [{"index": i, "relevance_score": len(d) / 100} for i, d in enumerate(docs)]
        return httpx.Response(
            200, json={"data": data[: body["top_k"]], "usage": {"total_tokens": 42}}
        )

    return VoyageReranker("key", "rerank-2.5", transport=httpx.MockTransport(handler))


def test_requires_api_key() -> None:
    with pytest.raises(RerankError):
        VoyageReranker("", "rerank-2.5")


def test_returns_best_first_and_sends_model_and_top_k() -> None:
    requests: list[dict[str, object]] = []
    r = _reranker(requests)
    ranked = r.rerank("q", ["a", "ccc", "bb"], top_k=3)
    assert [i for i, _ in ranked] == [1, 2, 0]
    assert requests[0]["model"] == "rerank-2.5"
    assert requests[0]["top_k"] == 3
    assert r.total_tokens == 42


def test_top_k_is_capped_at_document_count() -> None:
    requests: list[dict[str, object]] = []
    _reranker(requests).rerank("q", ["a", "b"], top_k=8)
    assert requests[0]["top_k"] == 2


def test_no_documents_means_no_call() -> None:
    requests: list[dict[str, object]] = []
    assert _reranker(requests).rerank("q", [], top_k=8) == []
    assert requests == []


def test_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alphalens_core.reranking.time.sleep", lambda _s: None)
    with pytest.raises(RerankError, match="401"):
        _reranker([], status=401).rerank("q", ["a"], top_k=1)
