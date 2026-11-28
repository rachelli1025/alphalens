"""extract_facts on a hand-built companyfacts payload that contains the real-data traps."""

from datetime import date
from typing import Any

from alphalens_core.xbrl import extract_facts


def _e(
    start: str | None, end: str, val: float, accn: str, fy: int, fp: str, form: str, filed: str
) -> dict[str, Any]:
    e = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        e["start"] = start
    return e


K24, K25, Q25 = "0000000001-24-000001", "0000000001-25-000001", "0000000001-25-000002"

PAYLOAD: dict[str, Any] = {
    "facts": {
        "us-gaap": {
            # Trap 2: the concept changes; FY2024 uses Revenues, FY2025 the newer tag.
            "Revenues": {
                "units": {
                    "USD": [
                        _e(
                            "2023-10-01", "2024-09-28", 100.0, K24, 2024, "FY", "10-K", "2024-11-01"
                        ),
                    ]
                }
            },
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        # Trap 1: FY2024 repeated in the FY2025 10-K, tagged fy=2025.
                        _e(
                            "2023-10-01", "2024-09-28", 100.0, K25, 2025, "FY", "10-K", "2025-11-01"
                        ),
                        _e(
                            "2024-09-29", "2025-09-27", 120.0, K25, 2025, "FY", "10-K", "2025-11-01"
                        ),
                        # Trap 3: the same fact twice in one filing.
                        _e(
                            "2024-09-29", "2025-09-27", 120.0, K25, 2025, "FY", "10-K", "2025-11-01"
                        ),
                        # A Q2 10-Q: the 3-month amount and the 6-month year-to-date amount.
                        _e("2024-12-29", "2025-03-29", 30.0, Q25, 2025, "Q2", "10-Q", "2025-05-01"),
                        _e("2024-09-29", "2025-03-29", 58.0, Q25, 2025, "Q2", "10-Q", "2025-05-01"),
                        # Not a periodic report: ignored.
                        _e("2024-09-29", "2025-09-27", 999.0, "x", 2025, "FY", "8-K", "2025-10-30"),
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        _e(None, "2024-09-28", 500.0, K24, 2024, "FY", "10-K", "2024-11-01"),
                        _e(None, "2024-09-28", 500.0, K25, 2025, "FY", "10-K", "2025-11-01"),
                        _e(None, "2025-09-27", 600.0, K25, 2025, "FY", "10-K", "2025-11-01"),
                    ]
                }
            },
        }
    }
}


def _facts(concept: str) -> list[Any]:
    return [f for f in extract_facts(PAYLOAD) if f.concept == concept]


def test_comparatives_get_the_fiscal_year_of_their_own_period() -> None:
    fy24 = [f for f in _facts("revenue") if f.period_end == date(2024, 9, 28)]
    # Reported in both 10-Ks; both rows say FY2024 although the FY2025 filing tags fy=2025.
    assert {f.accession_no for f in fy24} == {K24, K25}
    assert {(f.fiscal_year, f.fiscal_period) for f in fy24} == {(2024, "FY")}


def test_each_filing_uses_the_first_concept_it_reports() -> None:
    by_end = {
        (f.period_end, f.accession_no): f.source_tag
        for f in _facts("revenue")
        if f.fiscal_period == "FY"
    }
    new_tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert by_end[(date(2024, 9, 28), K24)] == "us-gaap:Revenues"
    # The FY2025 filing re-reports FY2024 under the new tag; that copy is kept too.
    assert by_end[(date(2024, 9, 28), K25)] == new_tag
    assert by_end[(date(2025, 9, 27), K25)] == new_tag


def test_duplicates_in_one_filing_are_dropped_and_other_forms_ignored() -> None:
    fy25 = [f for f in _facts("revenue") if f.period_end == date(2025, 9, 27)]
    assert [(f.value, f.accession_no) for f in fy25] == [(120.0, K25)]


def test_quarter_labelled_but_year_to_date_not() -> None:
    q = {
        (f.period_start, f.value): f.fiscal_period
        for f in _facts("revenue")
        if f.accession_no == Q25
    }
    assert q[(date(2024, 12, 29), 30.0)] == "Q2"  # 3-month amount
    assert q[(date(2024, 9, 29), 58.0)] is None  # 6-month year to date: kept, unlabelled


def test_balance_sheet_values_are_point_in_time() -> None:
    assets = {(f.period_end, f.accession_no): f for f in _facts("total_assets")}
    latest = assets[(date(2025, 9, 27), K25)]
    assert latest.period_start is None
    assert (latest.fiscal_year, latest.fiscal_period, latest.value) == (2025, "FY", 600.0)
    assert assets[(date(2024, 9, 28), K25)].fiscal_year == 2024  # comparative, relabelled
