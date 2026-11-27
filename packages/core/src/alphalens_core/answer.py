"""Grounded answers with Claude citations (DESIGN.md §8, last step).

Each retrieved chunk is sent as its own `document` block with citations enabled.
Claude's reply comes back as text blocks; every block that relies on a source
carries `citations` with the exact `cited_text` and the index of the document it
came from, so the UI can link each claim to its passage.
"""

from dataclasses import dataclass, field
from typing import Any

import anthropic

from alphalens_core.retrieval import RetrievedChunk

SYSTEM_PROMPT = """\
You are AlphaLens, a research assistant for US equities. Answer the user's \
question using only the SEC filing excerpts provided as documents.

- Base every factual statement on the documents; do not use outside knowledge.
- If the documents do not contain the answer, say so plainly instead of guessing.
- Be concise: a short direct answer first, then the supporting points.
- Do not give investment advice or recommendations to buy or sell."""


@dataclass(frozen=True)
class Citation:
    chunk: RetrievedChunk
    cited_text: str


@dataclass(frozen=True)
class AnswerPart:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    parts: list[AnswerPart]
    model: str
    input_tokens: int
    output_tokens: int
    refused: bool = False

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)


def build_documents(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """One document block per chunk; the order defines citation document_index."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c.content},
            "title": c.context_header,
            "context": f"Source: {c.url}",
            "citations": {"enabled": True},
        }
        for c in chunks
    ]


def parse_content(content: list[Any], chunks: list[RetrievedChunk]) -> list[AnswerPart]:
    """Keep text blocks (skip thinking/fallback blocks) and map citations to chunks."""
    parts = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        citations = [
            Citation(chunk=chunks[c.document_index], cited_text=c.cited_text)
            for c in (block.citations or [])
            if getattr(c, "type", None) == "char_location" and c.document_index < len(chunks)
        ]
        parts.append(AnswerPart(text=block.text, citations=citations))
    return parts


def answer_question(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    chunks: list[RetrievedChunk],
) -> Answer:
    if not chunks:
        return Answer(
            parts=[AnswerPart("No matching filing passages were found for this question.")],
            model=model,
            input_tokens=0,
            output_tokens=0,
        )

    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API retries on a fallback model in
        # the same call instead of returning an empty refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {
                "role": "user",
                "content": [*build_documents(chunks), {"type": "text", "text": question}],
            }
        ],
    )

    refused = response.stop_reason == "refusal"
    parts = parse_content(response.content, chunks)
    if refused and not parts:
        parts = [AnswerPart("The model declined to answer this question.")]
    return Answer(
        parts=parts,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        refused=refused,
    )
