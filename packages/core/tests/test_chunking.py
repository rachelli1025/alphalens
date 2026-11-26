import re

from alphalens_core.chunking import CHARS_PER_TOKEN, chunk_text


def _paragraphs(n: int, sentences: int = 8) -> str:
    return "\n".join(
        " ".join(f"Paragraph {p} sentence {s} talks about revenue." for s in range(sentences))
        for p in range(n)
    )


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("One short paragraph.")
    assert len(chunks) == 1
    assert chunks[0].content == "One short paragraph."
    assert chunks[0].idx == 0


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("\n \n") == []


def test_chunks_respect_max_size() -> None:
    target = 100
    chunks = chunk_text(_paragraphs(30), target_tokens=target)
    assert len(chunks) > 3
    assert all(len(c.content) <= target * CHARS_PER_TOKEN for c in chunks)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_text(_paragraphs(30), target_tokens=100, overlap_ratio=0.15)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_sentence = re.split(r"(?<=\.)\s", nxt.content)[0]
        assert prev.content.endswith(first_sentence) or f"{first_sentence}\n" in prev.content


def test_all_text_is_covered() -> None:
    text = _paragraphs(20)
    joined = " ".join(c.content for c in chunk_text(text, target_tokens=100))
    for p in range(20):
        assert f"Paragraph {p} sentence 7" in joined


def test_paragraph_breaks_are_kept() -> None:
    chunks = chunk_text("First para.\nSecond para.")
    assert chunks[0].content == "First para.\nSecond para."


def test_single_huge_sentence_is_split_by_words() -> None:
    text = "word " * 2000  # no sentence boundary at all
    chunks = chunk_text(text, target_tokens=100)
    assert len(chunks) > 1
    assert all(len(c.content) <= 100 * CHARS_PER_TOKEN for c in chunks)
