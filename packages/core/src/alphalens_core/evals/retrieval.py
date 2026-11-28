"""Retrieval eval: Recall@k and MRR over a hand-labelled golden set (DESIGN.md §13).

Labels name a section and a phrase from the filing ("item 1A containing 'final
assembly of substantially all'") instead of chunk ids, so they survive re-chunking
and re-ingestion. A retrieved chunk is relevant if it matches any label of the case.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from alphalens_core.retrieval import RetrievedChunk

# hybrid / vector / keyword are single-stage SQL searches; rerank = hybrid + reranker.
EvalMode = Literal["hybrid", "vector", "keyword", "rerank"]


@dataclass(frozen=True)
class RelevantLabel:
    item: str
    contains: str


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant: list[RelevantLabel]
    tickers: list[str] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    first_relevant_rank: int | None  # 1-based rank of the first relevant chunk, None if missed


@dataclass(frozen=True)
class ModeReport:
    mode: EvalMode
    k: int
    results: list[CaseResult]

    @property
    def recall_at_k(self) -> float:
        hits = sum(r.first_relevant_rank is not None for r in self.results)
        return hits / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        total = sum(1 / r.first_relevant_rank for r in self.results if r.first_relevant_rank)
        return total / len(self.results) if self.results else 0.0


# search(question, tickers, k, mode) -> ranked chunks; injected so tests need no database.
Search = Callable[[str, list[str] | None, int, EvalMode], list[RetrievedChunk]]


def normalize(text: str) -> str:
    """Case-, whitespace- and quote-insensitive form, so labels need not match typography."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: RetrievedChunk, labels: Sequence[RelevantLabel]) -> bool:
    content = normalize(chunk.content)
    return any(
        chunk.item_code.upper() == label.item.upper() and normalize(label.contains) in content
        for label in labels
    )


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text())
    cases = []
    for entry in raw["cases"]:
        relevant = [
            RelevantLabel(item=str(r["item"]), contains=r["contains"]) for r in entry["relevant"]
        ]
        if not relevant:
            raise ValueError(f"case {entry['id']!r} has no relevant labels")
        cases.append(
            EvalCase(
                id=entry["id"],
                question=entry["question"],
                relevant=relevant,
                tickers=entry.get("tickers"),
                tags=entry.get("tags", []),
            )
        )
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def evaluate(
    cases: Sequence[EvalCase], search: Search, *, mode: EvalMode, k: int = 8
) -> ModeReport:
    results = []
    for case in cases:
        chunks = search(case.question, case.tickers, k, mode)
        rank = next((i for i, c in enumerate(chunks, 1) if is_relevant(c, case.relevant)), None)
        results.append(CaseResult(case_id=case.id, first_relevant_rank=rank))
    return ModeReport(mode=mode, k=k, results=results)
