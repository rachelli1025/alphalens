"""Recursive text splitter: about 800 tokens per chunk with 15% overlap (DESIGN.md §7.1).

Text is broken into sentence-sized units (words, if a sentence is too long),
keeping paragraph breaks. Units are packed greedily into chunks; each new chunk
starts with the trailing sentences of the previous one as overlap.

Token counts are estimated as chars / 4 (close for English filings). Exact
tokenization is not needed for chunk sizing; Voyage reports real usage.
"""

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class TextChunk:
    idx: int
    content: str
    token_count: int


def _split_words(sentence: str, max_chars: int) -> list[str]:
    pieces, buf = [], ""
    for word in sentence.split(" "):
        if buf and len(buf) + 1 + len(word) > max_chars:
            pieces.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}" if buf else word
    if buf:
        pieces.append(buf)
    return pieces


def _units(text: str, max_chars: int) -> list[str]:
    """Sentence-sized pieces, each carrying its trailing separator (' ' or '\\n')."""
    units: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        pieces: list[str] = []
        for sentence in SENTENCE_RE.split(para):
            pieces.extend(
                [sentence] if len(sentence) < max_chars else _split_words(sentence, max_chars - 1)
            )
        units.extend(p + " " for p in pieces[:-1])
        units.append(pieces[-1] + "\n")
    return units


def chunk_text(
    text: str,
    *,
    target_tokens: int = 800,
    overlap_ratio: float = 0.15,
) -> list[TextChunk]:
    max_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = int(max_chars * overlap_ratio)

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for unit in _units(text, max_chars):
        if current and size + len(unit) > max_chars:
            chunks.append("".join(current).strip())
            carry: list[str] = []
            carry_size = 0
            for prev in reversed(current):
                if (
                    carry_size + len(prev) > overlap_chars
                    or carry_size + len(prev) + len(unit) > max_chars
                ):
                    break
                carry.insert(0, prev)
                carry_size += len(prev)
            current, size = carry, carry_size
        current.append(unit)
        size += len(unit)

    if current:  # always holds at least the last unit, so never pure overlap
        chunks.append("".join(current).strip())

    return [TextChunk(i, c, estimate_tokens(c)) for i, c in enumerate(chunks)]
