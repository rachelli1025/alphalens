from pathlib import Path

import pytest

from alphalens_core.evals.retrieval import (
    EvalCase,
    RelevantLabel,
    evaluate,
    is_relevant,
    load_cases,
    normalize,
)
from alphalens_core.retrieval import RetrievedChunk

REPO_DATASET = Path(__file__).parents[3] / "evals" / "datasets" / "retrieval.yaml"


def _chunk(content: str, item: str = "1A") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hash(content) % 10_000,
        content=content,
        context_header="",
        ticker="AAPL",
        form_type="10-K",
        fiscal_year=2026,
        item_code=item,
        section_title=None,
        accession_no="x",
        url="x",
        score=0.0,
        vec_rank=None,
        kw_rank=None,
    )


def test_normalize_ignores_case_whitespace_and_curly_quotes() -> None:
    assert normalize("The  Company’s\nHEADQUARTERS") == "the company's headquarters"


def test_relevant_needs_matching_item_and_phrase() -> None:
    labels = [RelevantLabel(item="1a", contains="final assembly")]
    assert is_relevant(_chunk("... Final  Assembly of products", item="1A"), labels)
    assert not is_relevant(_chunk("final assembly", item="7"), labels)  # wrong item
    assert not is_relevant(_chunk("assembly", item="1A"), labels)  # phrase missing


def test_metrics_from_ranks() -> None:
    cases = [
        EvalCase(id="a", question="qa", relevant=[RelevantLabel("1A", "target")]),
        EvalCase(id="b", question="qb", relevant=[RelevantLabel("1A", "target")]),
        EvalCase(id="c", question="qc", relevant=[RelevantLabel("1A", "target")]),
    ]
    ranked = {
        "qa": [_chunk("target")],  # rank 1
        "qb": [_chunk("noise 1"), _chunk("noise 2"), _chunk("target")],  # rank 3
        "qc": [_chunk("noise 3")],  # miss
    }
    report = evaluate(cases, lambda q, _t, _k, _m: ranked[q], mode="hybrid", k=8)
    assert [r.first_relevant_rank for r in report.results] == [1, 3, None]
    assert report.recall_at_k == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_search_receives_case_inputs() -> None:
    seen = []
    case = EvalCase(id="a", question="q", relevant=[RelevantLabel("1", "x")], tickers=["AAPL"])

    def search(q: str, t: list[str] | None, k: int, m: str) -> list[RetrievedChunk]:
        seen.append((q, t, k, m))
        return []

    evaluate([case], search, mode="keyword", k=5)  # type: ignore[arg-type]
    assert seen == [("q", ["AAPL"], 5, "keyword")]


def test_load_rejects_duplicates_and_empty_labels(tmp_path: Path) -> None:
    dup = tmp_path / "dup.yaml"
    dup.write_text(
        "cases:\n"
        "  - {id: a, question: q, relevant: [{item: '1', contains: x}]}\n"
        "  - {id: a, question: q, relevant: [{item: '1', contains: x}]}\n"
    )
    with pytest.raises(ValueError, match="unique"):
        load_cases(dup)

    empty = tmp_path / "empty.yaml"
    empty.write_text("cases:\n  - {id: a, question: q, relevant: []}\n")
    with pytest.raises(ValueError, match="no relevant"):
        load_cases(empty)


def test_repo_dataset_loads() -> None:
    cases = load_cases(REPO_DATASET)
    assert len(cases) >= 12
    assert all(c.relevant for c in cases)
