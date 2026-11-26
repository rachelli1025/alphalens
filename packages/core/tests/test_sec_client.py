from datetime import date

import httpx
import pytest

from alphalens_core.sec.client import EdgarClient, EdgarError

UA = "test-agent test@example.com"

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
SUBMISSIONS = {
    "name": "Apple Inc.",
    "sicDescription": "Electronic Computers",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-25-000100",
                "0000320193-25-000079",
                "0000320193-24-000123",
                "0000320193-25-000050",
            ],
            "form": ["10-Q", "10-K", "10-K", "10-K/A"],
            "filingDate": ["2026-01-30", "2025-10-31", "2024-11-01", "2025-12-01"],
            "reportDate": ["2025-12-27", "2025-09-27", "2024-09-28", "2025-09-27"],
            "primaryDocument": ["q1.htm", "aapl-20250927.htm", "aapl-20240928.htm", "a.htm"],
        }
    },
}


def _client(seen: list[httpx.Request]) -> EdgarClient:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json=TICKERS)
        if req.url.path == "/submissions/CIK0000320193.json":
            return httpx.Response(200, json=SUBMISSIONS)
        if req.url.path.endswith(".htm"):
            return httpx.Response(200, text="<html>10-K</html>")
        return httpx.Response(404)

    return EdgarClient(UA, min_interval_s=0, transport=httpx.MockTransport(handler))


def test_requires_user_agent_with_email() -> None:
    with pytest.raises(EdgarError):
        EdgarClient("")
    with pytest.raises(EdgarError):
        EdgarClient("no-email-here")


def test_lookup_company_pads_cik_and_sends_user_agent() -> None:
    seen: list[httpx.Request] = []
    info = _client(seen).lookup_company("aapl")
    assert info.cik == "0000320193"
    assert info.ticker == "AAPL"
    assert info.name == "Apple Inc."
    assert info.industry == "Electronic Computers"
    assert all(r.headers["User-Agent"] == UA for r in seen)


def test_unknown_ticker_raises() -> None:
    with pytest.raises(EdgarError, match="not found"):
        _client([]).lookup_company("ZZZZ")


def test_latest_10k_ignores_10q_and_amendments() -> None:
    ref = _client([]).latest_filing("0000320193", "10-K")
    assert ref.accession_no == "0000320193-25-000079"
    assert ref.filed_at == date(2025, 10, 31)
    assert ref.period_of_report == date(2025, 9, 27)
    assert ref.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )


def test_download_returns_text() -> None:
    client = _client([])
    ref = client.latest_filing("0000320193")
    assert client.download(ref.url) == "<html>10-K</html>"


def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alphalens_core.sec.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] < 3 else httpx.Response(200, json=TICKERS)

    client = EdgarClient(UA, min_interval_s=0, transport=httpx.MockTransport(handler))
    assert client._get_json("https://www.sec.gov/files/company_tickers.json") == TICKERS
    assert calls["n"] == 3


def test_non_retryable_error_raises() -> None:
    client = EdgarClient(
        UA, min_interval_s=0, transport=httpx.MockTransport(lambda _r: httpx.Response(403))
    )
    with pytest.raises(EdgarError, match="403"):
        client.download("https://www.sec.gov/x.htm")
