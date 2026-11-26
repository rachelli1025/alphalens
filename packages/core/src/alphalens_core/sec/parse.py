"""10-K HTML -> plain text -> sections by Item.

Primary documents are inline-XBRL HTML. We flatten them to one block per line,
then find the real Item headings. The table of contents lists every Item too, so
we pick the chain of headings that produces substantial sections (see
`split_10k_items`).
"""

import re
import warnings
from collections import Counter
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# Item codes of Form 10-K in document order.
ITEM_ORDER = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A",
    "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
]  # fmt: skip
ITEM_RANK = {code: i for i, code in enumerate(ITEM_ORDER)}

BLOCK_TAGS = [
    "p", "div", "br", "tr", "li", "table", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6",
]  # fmt: skip

HEADING_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\b\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"(?:\|\s*)?\d{1,3}(?:\s*\|)?$")
MAX_HEADING_TITLE = 120  # longer lines are sentences that start with "Item ...", not headings
MIN_SECTION_CHARS = 500  # what counts as a "substantial" section when choosing headings


@dataclass(frozen=True)
class Section:
    item_code: str
    title: str | None
    text: str


def _is_hidden(tag: Tag) -> bool:
    style = str(tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def html_to_text(html: str) -> str:
    """Flatten filing HTML to text with one block element per line.

    Inline elements (including iXBRL number tags) stay inline, so sentences
    such as "net sales of $391.0 billion" are not broken apart.
    """
    # SEC iXBRL files are XHTML with an XML declaration; we deliberately parse them
    # as HTML (hidden-element handling, plain-HTML filings), so the warning is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # iXBRL metadata lives in a hidden <ix:header>; drop every hidden element.
    for tag in soup.find_all(_is_hidden):
        tag.decompose()
    for cell in soup.find_all(["td", "th"]):
        cell.append(NavigableString(" | "))
    for block in soup.find_all(BLOCK_TAGS):
        block.insert_before(NavigableString("\n"))
        block.append(NavigableString("\n"))

    text = soup.get_text().replace("\xa0", " ").replace("​", "")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(r"(\s*\|\s*)+$", "", line)  # trailing empty table cells
        line = re.sub(r"^(\|\s*)+", "", line)
        if line:
            lines.append(line)
    return "\n".join(_drop_page_furniture(lines))


PART_RE = re.compile(r"^part\s+[iv]+\.?$", re.IGNORECASE)
TRAILING_PAGE_RE = re.compile(r"[\s|]*\d{1,3}$")


def _drop_page_furniture(lines: list[str]) -> list[str]:
    """Remove page numbers, "PART II" dividers and per-page headers/footers.

    A footer such as "Acme Corp. | 2025 Form 10-K | 12" differs on every page only
    by its page number, so lines are compared with any trailing number stripped.
    """

    def key(line: str) -> str:
        return TRAILING_PAGE_RE.sub("", line)

    counts = Counter(key(line) for line in lines if len(line) < 80)
    # len >= 10 keeps short repeated table rows ("Total | 5") from being dropped.
    repeated = {k for k, n in counts.items() if n >= 5 and len(k) >= 10 and not HEADING_RE.match(k)}
    return [
        ln
        for ln in lines
        if not ln.isdigit() and not PART_RE.match(ln) and not (len(ln) < 80 and key(ln) in repeated)
    ]


@dataclass(frozen=True)
class _Heading:
    line_no: int
    code: str
    title: str | None
    title_on_next_line: bool


def _find_headings(lines: list[str]) -> list[_Heading]:
    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        code = m.group(1).upper()
        if code not in ITEM_RANK:
            continue
        rest = m.group(2).strip(" |")
        if len(rest) > MAX_HEADING_TITLE:
            continue
        # Table-of-contents rows end with a page number ("Risk Factors | 5"). Strip it;
        # chain selection in split_10k_items decides whether the line is a heading.
        rest = PAGE_NUMBER_RE.sub("", rest).strip(" |")
        title_on_next = False
        if not rest and i + 1 < len(lines) and len(lines[i + 1]) <= MAX_HEADING_TITLE:
            nxt = lines[i + 1]
            if not HEADING_RE.match(nxt):
                rest, title_on_next = nxt, True
        headings.append(_Heading(i, code, rest or None, title_on_next))
    return headings


def _chain_from(start: int, headings: list[_Heading]) -> list[_Heading]:
    """Greedy chain of headings with strictly increasing Item order, starting at `start`."""
    chain = [headings[start]]
    for h in headings[start + 1 :]:
        if ITEM_RANK[h.code] > ITEM_RANK[chain[-1].code]:
            chain.append(h)
    return chain


def _sections_from_chain(lines: list[str], chain: list[_Heading]) -> list[Section]:
    sections = []
    for k, h in enumerate(chain):
        body_start = h.line_no + (2 if h.title_on_next_line else 1)
        body_end = chain[k + 1].line_no if k + 1 < len(chain) else len(lines)
        text = "\n".join(lines[body_start:body_end]).strip()
        sections.append(Section(item_code=h.code, title=h.title, text=text))
    return sections


def split_10k_items(text: str) -> list[Section]:
    """Split flattened 10-K text into Item sections.

    Every chain of increasing Item headings is a candidate; the table of contents
    yields a chain of tiny sections, the real body yields long ones. We keep the
    chain with the most substantial sections (ties: more sections, then later start).
    """
    lines = text.split("\n")
    headings = _find_headings(lines)
    if not headings:
        return []

    best: tuple[tuple[int, int, int], list[Section]] | None = None
    for start in range(len(headings)):
        sections = _sections_from_chain(lines, _chain_from(start, headings))
        substantial = sum(len(s.text) >= MIN_SECTION_CHARS for s in sections)
        score = (substantial, len(sections), start)
        if best is None or score > best[0]:
            best = (score, sections)
    assert best is not None
    return best[1]
