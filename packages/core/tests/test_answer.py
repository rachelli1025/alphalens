"""answer.py without network: the Anthropic client is replaced by a recording fake."""

from types import SimpleNamespace
from typing import Any

from alphalens_core.answer import answer_question, build_documents, parse_content
from alphalens_core.retrieval import RetrievedChunk


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + i,
        content=f"Passage {i} text.",
        context_header=f"[AAPL | 10-K | FY2026 | Item 1A Risk Factors] #{i}",
        ticker="AAPL",
        form_type="10-K",
        fiscal_year=2026,
        item_code="1A",
        section_title="Risk Factors",
        accession_no="0000320193-26-000026",
        url="https://www.sec.gov/x.htm",
        score=0.03,
        vec_rank=i + 1,
        kw_rank=None,
    )


CHUNKS = [_chunk(0), _chunk(1)]


def _text(text: str, citations: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=citations)


def _cite(doc: int, cited: str) -> SimpleNamespace:
    return SimpleNamespace(type="char_location", document_index=doc, cited_text=cited)


class FakeClient:
    """Mimics client.beta.messages.create and records the request."""

    def __init__(self, content: list[Any], stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        response = SimpleNamespace(
            content=content,
            stop_reason=stop_reason,
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
        )

        def create(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            return response

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_build_documents_one_per_chunk_with_citations() -> None:
    docs = build_documents(CHUNKS)
    assert len(docs) == 2
    assert docs[0]["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": "Passage 0 text.",
    }
    assert docs[1]["title"].endswith("#1")
    assert all(d["citations"] == {"enabled": True} for d in docs)


def test_parse_maps_citations_to_chunks_and_skips_other_blocks() -> None:
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        _text("Apple depends on Asian suppliers", [_cite(1, "Passage 1 text.")]),
        _text(" and more.", None),
        _text(" Bad index", [_cite(7, "ignored")]),
    ]
    parts = parse_content(content, CHUNKS)
    assert [p.text for p in parts] == [
        "Apple depends on Asian suppliers",
        " and more.",
        " Bad index",
    ]
    assert parts[0].citations[0].chunk.chunk_id == 101
    assert parts[0].citations[0].cited_text == "Passage 1 text."
    assert parts[1].citations == []
    assert parts[2].citations == []


def test_answer_question_sends_documents_then_question() -> None:
    client = FakeClient([_text("Answer.", [_cite(0, "Passage 0 text.")])])
    answer = answer_question(client, "claude-opus-5-5", "What are the risks?", CHUNKS)  # type: ignore[arg-type]

    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    blocks = call["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["document", "document", "text"]
    assert blocks[-1]["text"] == "What are the risks?"
    assert call["fallbacks"] == "default"
    assert answer.text == "Answer."
    assert answer.parts[0].citations[0].chunk.chunk_id == 100
    assert (answer.input_tokens, answer.output_tokens) == (1200, 80)
    assert not answer.refused


def test_no_chunks_means_no_api_call() -> None:
    client = FakeClient([])
    answer = answer_question(client, "claude-opus-5-5", "anything", [])  # type: ignore[arg-type]
    assert client.calls == []
    assert "No matching" in answer.text


def test_refusal_is_reported() -> None:
    client = FakeClient([], stop_reason="refusal")
    answer = answer_question(client, "claude-opus-5-5", "q", CHUNKS)  # type: ignore[arg-type]
    assert answer.refused
    assert "declined" in answer.text
