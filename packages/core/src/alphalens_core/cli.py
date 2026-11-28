"""Command line entry point.

uv run alphalens ingest AAPL            # latest 10-K
uv run alphalens ingest AAPL --force    # rebuild sections and chunks
uv run alphalens ask "What are Apple's main supply chain risks?" --ticker AAPL
"""

import argparse
import logging
import sys

import anthropic

from alphalens_core.answer import Answer, answer_question
from alphalens_core.config import get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import VoyageEmbedder
from alphalens_core.ingest import ingest_latest_filing
from alphalens_core.retrieval import hybrid_search
from alphalens_core.sec import EdgarClient


def _ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    with (
        EdgarClient(settings.sec_user_agent) as edgar,
        VoyageEmbedder(settings.voyage_api_key, settings.embedding_model) as embedder,
        get_sync_sessionmaker()() as session,
    ):
        for ticker in args.tickers:
            result = ingest_latest_filing(
                session, edgar, embedder, ticker, form_type=args.form, force=args.force
            )
            if result.skipped:
                print(f"{result.ticker}: {result.accession_no} already ingested (skipped)")
            else:
                print(
                    f"{result.ticker}: {result.accession_no} -> "
                    f"{result.sections} sections, {result.chunks} chunks"
                )
        print(f"Voyage tokens used: {embedder.total_tokens}")
    return 0


def _print_answer(answer: Answer) -> None:
    """Answer text with [n] markers, then the numbered sources they point to."""
    sources: dict[int, int] = {}  # chunk_id -> source number
    out = []
    for part in answer.parts:
        out.append(part.text)
        marks = []
        for c in part.citations:
            n = sources.setdefault(c.chunk.chunk_id, len(sources) + 1)
            if f"[{n}]" not in marks:
                marks.append(f"[{n}]")
        out.append("".join(marks))
    print("".join(out).strip())

    if sources:
        print("\nSources:")
        by_id = {c.chunk.chunk_id: c.chunk for p in answer.parts for c in p.citations}
        for chunk_id, n in sources.items():
            chunk = by_id[chunk_id]
            print(f"  [{n}] {chunk.context_header}  {chunk.url}")
    cost = f"${answer.cost_usd:.4f}" if answer.cost_usd is not None else "cost unknown"
    print(
        f"\n({answer.model}, {answer.input_tokens} in / {answer.output_tokens} out tokens, {cost})"
    )


def _ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    with (
        VoyageEmbedder(settings.voyage_api_key, settings.embedding_model) as embedder,
        get_sync_sessionmaker()() as session,
    ):
        chunks = hybrid_search(session, embedder, args.question, tickers=args.ticker, k=args.k)

    if args.show_chunks:
        for i, c in enumerate(chunks, 1):
            print(
                f"{i}. score={c.score:.4f} vec_rank={c.vec_rank} kw_rank={c.kw_rank} "
                f"{c.context_header}\n   {c.content[:150]!r}"
            )
        print()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    model = args.model or settings.answer_model
    effort = args.effort or settings.answer_effort
    _print_answer(answer_question(client, model, args.question, chunks, effort=effort))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphalens")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest the latest filing for one or more tickers")
    ingest.add_argument("tickers", nargs="+")
    ingest.add_argument("--form", default="10-K")
    ingest.add_argument("--force", action="store_true", help="rebuild even if already embedded")
    ingest.set_defaults(func=_ingest)

    ask = sub.add_parser("ask", help="answer a question from ingested filings, with citations")
    ask.add_argument("question")
    ask.add_argument("--ticker", action="append", help="restrict to a company (repeatable)")
    ask.add_argument("-k", type=int, default=8, help="number of passages given to Claude")
    ask.add_argument("--show-chunks", action="store_true", help="print the retrieved passages")
    ask.add_argument("--model", help="override ANSWER_MODEL, e.g. claude-opus-5-5")
    ask.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ask.set_defaults(func=_ask)

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
