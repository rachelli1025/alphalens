"""SEC XBRL company facts -> standardized financial facts (DESIGN.md §7.2).

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json, every XBRL fact a
company has filed, grouped by taxonomy concept. Three things make it unusable as is:

1. Companies tag the same metric with different concepts, and switch over time (Apple
   used `Revenues` until 2018, then `RevenueFromContractWithCustomerExcludingAssessedTax`).
   METRICS lists candidate concepts per metric, best first; for each period and filing
   the first candidate that filing reports wins.
2. `fy` / `fp` on a fact describe the FILING that reported it, not the fact's period:
   FY2024 revenue appears again in the FY2026 10-K tagged fy=2026. We label a period
   from the filing in which it is the current period (its end = that filing's period end).
3. The same number is repeated in later filings and sometimes restated. Every
   (period, filing) is kept; the v_financials view picks the latest filed.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

PeriodType = Literal["duration", "instant"]
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

ANNUAL_DAYS = range(350, 381)  # 52/53-week fiscal years
QUARTER_DAYS = range(80, 101)


@dataclass(frozen=True)
class Metric:
    name: str
    period_type: PeriodType
    unit: str
    tags: tuple[str, ...]  # us-gaap concepts, best first


METRICS: tuple[Metric, ...] = (
    # Income statement (amounts over a period)
    Metric(
        "revenue",
        "duration",
        "USD",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
    ),
    Metric("cost_of_revenue", "duration", "USD", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    Metric("gross_profit", "duration", "USD", ("GrossProfit",)),
    Metric("rd_expense", "duration", "USD", ("ResearchAndDevelopmentExpense",)),
    Metric("sga_expense", "duration", "USD", ("SellingGeneralAndAdministrativeExpense",)),
    Metric("operating_income", "duration", "USD", ("OperatingIncomeLoss",)),
    Metric(
        "interest_expense", "duration", "USD", ("InterestExpense", "InterestExpenseNonoperating")
    ),
    Metric("income_tax", "duration", "USD", ("IncomeTaxExpenseBenefit",)),
    Metric("net_income", "duration", "USD", ("NetIncomeLoss",)),
    Metric("eps_basic", "duration", "USD/shares", ("EarningsPerShareBasic",)),
    Metric("eps_diluted", "duration", "USD/shares", ("EarningsPerShareDiluted",)),
    Metric(
        "diluted_shares", "duration", "shares", ("WeightedAverageNumberOfDilutedSharesOutstanding",)
    ),
    # Cash flow statement
    Metric(
        "operating_cash_flow", "duration", "USD", ("NetCashProvidedByUsedInOperatingActivities",)
    ),
    Metric("capex", "duration", "USD", ("PaymentsToAcquirePropertyPlantAndEquipment",)),
    Metric(
        "depreciation_amortization",
        "duration",
        "USD",
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization",
        ),
    ),
    Metric("share_repurchases", "duration", "USD", ("PaymentsForRepurchaseOfCommonStock",)),
    Metric(
        "dividends_paid",
        "duration",
        "USD",
        ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    ),
    # Balance sheet (values at a point in time)
    Metric("cash", "instant", "USD", ("CashAndCashEquivalentsAtCarryingValue",)),
    Metric("current_assets", "instant", "USD", ("AssetsCurrent",)),
    Metric("total_assets", "instant", "USD", ("Assets",)),
    Metric("current_liabilities", "instant", "USD", ("LiabilitiesCurrent",)),
    Metric("total_liabilities", "instant", "USD", ("Liabilities",)),
    Metric("long_term_debt", "instant", "USD", ("LongTermDebtNoncurrent", "LongTermDebt")),
    Metric("inventory", "instant", "USD", ("InventoryNet",)),
    Metric(
        "stockholders_equity",
        "instant",
        "USD",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
)


@dataclass(frozen=True)
class Fact:
    concept: str  # standardized metric name
    source_tag: str
    value: float
    unit: str
    period_start: date | None
    period_end: date
    fiscal_year: int | None
    fiscal_period: str | None
    form_type: str
    filed_at: date
    accession_no: str


@dataclass(frozen=True)
class _Raw:
    tag: str
    start: date | None
    end: date
    val: float
    fy: int
    fp: str
    form: str
    filed: date
    accn: str

    @property
    def days(self) -> int | None:
        return (self.end - self.start).days if self.start else None


def _raw_entries(companyfacts: dict[str, Any]) -> list[_Raw]:
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    wanted = {(t, m.unit) for m in METRICS for t in m.tags}
    out = []
    for tag, unit in wanted:
        for e in gaap.get(tag, {}).get("units", {}).get(unit, []):
            if e.get("form") not in FORMS or e.get("fy") is None or not e.get("fp"):
                continue
            out.append(
                _Raw(
                    tag=tag,
                    start=date.fromisoformat(e["start"]) if e.get("start") else None,
                    end=date.fromisoformat(e["end"]),
                    val=float(e["val"]),
                    fy=int(e["fy"]),
                    fp=str(e["fp"]),
                    form=e["form"],
                    filed=date.fromisoformat(e["filed"]),
                    accn=e["accn"],
                )
            )
    return out


def _period_labels(raws: list[_Raw]) -> dict[tuple[date | None, date], tuple[int, str]]:
    """(start, end) -> (fiscal_year, fiscal_period) for periods that some filing reports as
    its own current period. A filing's period end = the latest end among its quarterly or
    annual amounts; only full years (FY) and single quarters are labelled, not 6- or
    9-month year-to-date amounts."""
    current_end: dict[str, date] = {}
    for r in raws:
        if r.days is not None and r.days >= QUARTER_DAYS.start:
            current_end[r.accn] = max(current_end.get(r.accn, r.end), r.end)

    labels: dict[tuple[date | None, date], tuple[int, str]] = {}
    for r in sorted(raws, key=lambda r: r.filed):  # earliest filing wins on conflicts
        if current_end.get(r.accn) != r.end:
            continue
        if r.start is None:
            key: tuple[date | None, date] = (None, r.end)
        elif (r.fp == "FY" and r.days in ANNUAL_DAYS) or (
            r.fp in {"Q1", "Q2", "Q3"} and r.days in QUARTER_DAYS
        ):
            key = (r.start, r.end)
        else:
            continue
        labels.setdefault(key, (r.fy, r.fp))
    return labels


def extract_facts(companyfacts: dict[str, Any]) -> list[Fact]:
    raws = _raw_entries(companyfacts)
    labels = _period_labels(raws)
    by_tag_period: dict[tuple[str, date | None, date], list[_Raw]] = defaultdict(list)
    for r in raws:
        by_tag_period[(r.tag, r.start, r.end)].append(r)

    facts: list[Fact] = []
    for metric in METRICS:
        rank = {tag: i for i, tag in enumerate(metric.tags)}
        # Per (period, filing), keep the highest-priority tag that filing reported. Choosing
        # per filing (not per period) keeps a later filing's copy, e.g. a restatement, even
        # when the company switched concepts between filings.
        best: dict[tuple[date | None, date, str], _Raw] = {}
        for (tag, start, end), entries in by_tag_period.items():
            if tag not in rank or (start is None) != (metric.period_type == "instant"):
                continue
            for r in entries:
                key = (start, end, r.accn)
                if key not in best or rank[tag] < rank[best[key].tag]:
                    best[key] = r  # same filing repeating a fact collapses here too
        for (start, end, _accn), r in sorted(best.items(), key=lambda kv: kv[1].filed):
            fy, fp = labels.get((start, end), (None, None))
            facts.append(
                Fact(
                    concept=metric.name,
                    source_tag=f"us-gaap:{r.tag}",
                    value=r.val,
                    unit=metric.unit,
                    period_start=start,
                    period_end=end,
                    fiscal_year=fy,
                    fiscal_period=fp,
                    form_type=r.form,
                    filed_at=r.filed,
                    accession_no=r.accn,
                )
            )
    return facts
