"""Command line entry point.

uv run alphalens ingest AAPL            # latest 10-K
uv run alphalens ingest AAPL --force    # rebuild sections and chunks
uv run alphalens ask "What are Apple's main supply chain risks?" --ticker AAPL
uv run alphalens ingest-financials AAPL MSFT    # XBRL facts -> financial_facts / v_financials
uv run alphalens eval-retrieval                 # Recall@8 / MRR for hybrid, vector, keyword
uv run alphalens eval-retrieval --check-labels  # verify golden-set phrases exist (no API calls)
"""

import argparse
import logging
import sys
from pathlib import Path

import anthropic

from alphalens_core.answer import Answer, answer_question
from alphalens_core.config import REPO_ROOT, get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import VoyageEmbedder
from alphalens_core.ingest import ingest_latest_filing
from alphalens_core.query_understanding import list_companies, resolve_tickers
from alphalens_core.reranking import VoyageReranker
from alphalens_core.retrieval import RetrievedChunk, hybrid_search, rerank_search
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


def _ingest_financials(args: argparse.Namespace) -> int:
    from alphalens_core.ingest.financials import ingest_financials

    settings = get_settings()
    with EdgarClient(settings.sec_user_agent) as edgar, get_sync_sessionmaker()() as session:
        for ticker in args.tickers:
            r = ingest_financials(session, edgar, ticker)
            print(f"{r.ticker}: {r.facts} facts ({r.labelled} with fiscal year/period)")
    return 0


def _agent(args: argparse.Namespace) -> int:
    from alphalens_core.agent.loop import run_agent
    from alphalens_core.agent.tools import ToolContext

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    with (
        VoyageEmbedder(settings.voyage_api_key, settings.embedding_model) as embedder,
        VoyageReranker(settings.voyage_api_key, settings.rerank_model) as reranker,
        get_sync_sessionmaker()() as session,
    ):
        ctx = ToolContext(session=session, embedder=embedder, reranker=reranker)
        result = run_agent(
            client,
            args.model or settings.agent_model,
            args.question,
            ctx,
            list_companies(session),
            effort=args.effort or settings.agent_effort,
        )

    print("Tool calls:")
    for i, call in enumerate(result.tool_calls, 1):
        flag = " (error)" if call.is_error else ""
        print(f"  {i}. {call.name}{flag}: {call.summary}")
    for sql in result.sql_queries:
        print(f"     SQL: {sql}")
    print()
    from alphalens_core.answer import Answer

    _print_answer(
        Answer(
            parts=result.parts,
            model=result.model,
            input_tokens=result.input_tokens + result.cache_write_tokens + result.cache_read_tokens,
            output_tokens=result.output_tokens,
        )
    )
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    print(f"Agent cost with caching: {cost} (cache read {result.cache_read_tokens} tokens)")
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
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    with (
        VoyageEmbedder(settings.voyage_api_key, settings.embedding_model) as embedder,
        get_sync_sessionmaker()() as session,
    ):
        tickers = args.ticker
        if not tickers:
            resolution = resolve_tickers(
                client, settings.query_model, args.question, list_companies(session)
            )
            tickers = resolution.tickers
            print(f"Companies detected: {', '.join(tickers) or 'none (searching all)'}\n")
        if settings.rerank_enabled:
            with VoyageReranker(settings.voyage_api_key, settings.rerank_model) as reranker:
                chunks = rerank_search(
                    session, embedder, reranker, args.question, tickers=tickers or None, k=args.k
                )
        else:
            chunks = hybrid_search(
                session, embedder, args.question, tickers=tickers or None, k=args.k
            )

    if args.show_chunks:
        for i, c in enumerate(chunks, 1):
            print(
                f"{i}. rerank={c.rerank_score} rrf={c.score:.4f} "
                f"vec_rank={c.vec_rank} kw_rank={c.kw_rank} "
                f"{c.context_header}\n   {c.content[:150]!r}"
            )
        print()

    model = args.model or settings.answer_model
    effort = args.effort or settings.answer_effort
    _print_answer(answer_question(client, model, args.question, chunks, effort=effort))
    return 0


DEFAULT_DATASET = REPO_ROOT / "evals" / "datasets" / "retrieval.yaml"


def _check_labels(cases: list, session) -> int:  # type: ignore[type-arg, no-untyped-def]
    """Count chunks matching each label; 0 means a typo or a phrase not in the filing."""
    from sqlalchemy import text

    from alphalens_core.evals.retrieval import normalize

    rows = session.execute(
        text(
            "SELECT s.item_code, c.content FROM chunks c "
            "JOIN filing_sections s ON s.id = c.section_id"
        )
    ).all()
    bad = 0
    for case in cases:
        for label in case.relevant:
            n = sum(
                item.upper() == label.item.upper()
                and normalize(label.contains) in normalize(content)
                for item, content in rows
            )
            flag = "OK " if n else "MISSING"
            bad += n == 0
            print(f"{flag} {case.id:28} item {label.item:3} {n:2} chunk(s)  {label.contains!r}")
    print(f"\n{bad} label(s) match no chunk." if bad else "\nAll labels match at least one chunk.")
    return 1 if bad else 0


def _eval_retrieval(args: argparse.Namespace) -> int:
    from alphalens_core.evals.retrieval import EvalMode, evaluate, load_cases

    cases = load_cases(Path(args.dataset))
    settings = get_settings()
    resolved: dict[str, list[str]] = {}  # question -> tickers from query understanding
    with get_sync_sessionmaker()() as session:
        if args.check_labels:
            return _check_labels(cases, session)

        if args.auto_tickers:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
            companies = list_companies(session)
            for case in cases:
                resolved[case.question] = resolve_tickers(
                    client, settings.query_model, case.question, companies
                ).tickers

        with (
            VoyageEmbedder(settings.voyage_api_key, settings.embedding_model) as embedder,
            VoyageReranker(settings.voyage_api_key, settings.rerank_model) as reranker,
        ):

            def search(
                question: str, tickers: list[str] | None, k: int, mode: EvalMode
            ) -> list[RetrievedChunk]:
                if args.ignore_tickers:
                    tickers = None  # search every company, as if the user named none
                elif args.auto_tickers:
                    tickers = resolved[question] or None  # what production would search
                if mode == "rerank":
                    return rerank_search(
                        session, embedder, reranker, question, tickers=tickers, k=k
                    )
                return hybrid_search(session, embedder, question, tickers=tickers, k=k, mode=mode)

            reports = [evaluate(cases, search, mode=m, k=args.k) for m in args.modes]

    scope = (
        "all companies"
        if args.ignore_tickers
        else "tickers resolved from the question"
        if args.auto_tickers
        else "case tickers"
    )
    print(f"{len(cases)} cases, k={args.k}, search scope: {scope}\n")
    if args.auto_tickers:
        right = sum(sorted(resolved[c.question]) == sorted(c.tickers or []) for c in cases)
        print(f"Ticker resolution: {right}/{len(cases)} match the labelled tickers")
        for c in cases:
            if sorted(resolved[c.question]) != sorted(c.tickers or []):
                print(f"  {c.id}: resolved {resolved[c.question]}, labelled {c.tickers}")
        print()
    print(f"{'mode':10} {'Recall@' + str(args.k):>10} {'MRR':>7}")
    for r in reports:
        print(f"{r.mode:10} {r.recall_at_k:>10.2f} {r.mrr:>7.2f}")

    print(f"\nRank of first relevant chunk (- = not in top {args.k}):")
    print(f"{'case':28} {'tags':12} " + " ".join(f"{r.mode:>8}" for r in reports))
    for i, case in enumerate(cases):
        ranks = [r.results[i].first_relevant_rank for r in reports]
        cells = " ".join(f"{(str(x) if x else '-'):>8}" for x in ranks)
        print(f"{case.id:28} {','.join(case.tags):12} {cells}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphalens")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest the latest filing for one or more tickers")
    ingest.add_argument("tickers", nargs="+")
    ingest.add_argument("--form", default="10-K")
    ingest.add_argument("--force", action="store_true", help="rebuild even if already embedded")
    ingest.set_defaults(func=_ingest)

    ag = sub.add_parser("agent", help="research agent: searches filings and queries financials")
    ag.add_argument("question")
    ag.add_argument("--model", help="override AGENT_MODEL")
    ag.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ag.set_defaults(func=_agent)

    fin = sub.add_parser("ingest-financials", help="load XBRL financial facts for tickers")
    fin.add_argument("tickers", nargs="+")
    fin.set_defaults(func=_ingest_financials)

    ask = sub.add_parser("ask", help="answer a question from ingested filings, with citations")
    ask.add_argument("question")
    ask.add_argument("--ticker", action="append", help="restrict to a company (repeatable)")
    ask.add_argument("-k", type=int, default=8, help="number of passages given to Claude")
    ask.add_argument("--show-chunks", action="store_true", help="print the retrieved passages")
    ask.add_argument("--model", help="override ANSWER_MODEL, e.g. claude-opus-5-5")
    ask.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ask.set_defaults(func=_ask)

    ev = sub.add_parser("eval-retrieval", help="Recall@k and MRR on the retrieval golden set")
    ev.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ev.add_argument("-k", type=int, default=8)
    ev.add_argument(
        "--modes",
        nargs="+",
        choices=["hybrid", "vector", "keyword", "rerank"],
        default=["hybrid", "vector", "keyword", "rerank"],
    )
    ev.add_argument(
        "--check-labels", action="store_true", help="verify label phrases; no API calls"
    )
    ev.add_argument(
        "--auto-tickers",
        action="store_true",
        help="resolve tickers from each question with the query model, as production does",
    )
    ev.add_argument(
        "--ignore-tickers",
        action="store_true",
        help="search all companies instead of each case's tickers (no query understanding)",
    )
    ev.set_defaults(func=_eval_retrieval)

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
