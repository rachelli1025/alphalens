"""Command line entry point.

uv run alphalens ingest AAPL            # latest 10-K
uv run alphalens ingest AAPL --force    # rebuild sections and chunks
"""

import argparse
import logging
import sys

from alphalens_core.config import get_settings
from alphalens_core.db.session import get_sync_sessionmaker
from alphalens_core.embeddings import VoyageEmbedder
from alphalens_core.ingest import ingest_latest_filing
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphalens")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest the latest filing for one or more tickers")
    ingest.add_argument("tickers", nargs="+")
    ingest.add_argument("--form", default="10-K")
    ingest.add_argument("--force", action="store_true", help="rebuild even if already embedded")
    ingest.set_defaults(func=_ingest)

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
