"""Minimal SEC EDGAR client: ticker lookup, filing index, document download.

EDGAR rules (https://www.sec.gov/os/accessing-edgar-data):
- every request must carry a descriptive User-Agent ("<name> <email>");
- at most 10 requests per second.
"""

import threading
import time
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Any, Self

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{document}"

RETRY_STATUSES = {429, 500, 502, 503, 504}


class EdgarError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompanyInfo:
    cik: str  # zero-padded to 10 digits
    ticker: str
    name: str
    industry: str | None


@dataclass(frozen=True)
class FilingRef:
    accession_no: str  # 0000320193-25-000079
    form_type: str
    filed_at: date
    period_of_report: date | None
    primary_document: str
    url: str


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        min_interval_s: float = 0.12,  # ~8 req/s, under the 10 req/s limit
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise EdgarError("SEC_USER_AGENT must be set to '<name> <email>'")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._min_interval = min_interval_s
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._last_request = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            resp = self._http.get(url)
            if resp.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                break
            time.sleep(2**attempt)
        if resp.status_code != 200:
            raise EdgarError(f"GET {url} -> HTTP {resp.status_code}")
        return resp

    def _get_json(self, url: str) -> Any:
        return self._get(url).json()

    def lookup_company(self, ticker: str) -> CompanyInfo:
        """Resolve a ticker to CIK, name and industry."""
        ticker = ticker.upper()
        rows = self._get_json(TICKERS_URL).values()
        match = next((r for r in rows if r["ticker"].upper() == ticker), None)
        if match is None:
            raise EdgarError(f"ticker not found on EDGAR: {ticker}")
        cik = f"{int(match['cik_str']):010d}"
        sub = self._get_json(SUBMISSIONS_URL.format(cik=cik))
        return CompanyInfo(
            cik=cik,
            ticker=ticker,
            name=sub.get("name") or match["title"],
            industry=sub.get("sicDescription") or None,
        )

    def recent_filings(self, cik: str, form_type: str) -> list[FilingRef]:
        """Filings of one form type from the submissions API, newest first.

        Only the `recent` block is read (about the last 1000 filings), which covers
        several years of 10-Ks. Amendments (e.g. 10-K/A) are excluded.
        """
        recent = self._get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
        refs = []
        for i, form in enumerate(recent["form"]):
            if form != form_type:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            refs.append(
                FilingRef(
                    accession_no=acc,
                    form_type=form,
                    filed_at=date.fromisoformat(recent["filingDate"][i]),
                    period_of_report=_parse_date(recent["reportDate"][i]),
                    primary_document=doc,
                    url=ARCHIVES_URL.format(
                        cik_int=int(cik), acc_nodash=acc.replace("-", ""), document=doc
                    ),
                )
            )
        refs.sort(key=lambda r: r.filed_at, reverse=True)
        return refs

    def latest_filing(self, cik: str, form_type: str = "10-K") -> FilingRef:
        refs = self.recent_filings(cik, form_type)
        if not refs:
            raise EdgarError(f"no {form_type} found for CIK {cik}")
        return refs[0]

    def download(self, url: str) -> str:
        return self._get(url).text
