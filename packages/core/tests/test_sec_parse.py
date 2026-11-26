from pathlib import Path

import pytest

from alphalens_core.sec.parse import html_to_text, split_10k_items

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


@pytest.fixture(scope="module")
def text() -> str:
    return html_to_text(FIXTURE.read_text())


def test_hidden_ixbrl_header_removed(text: str) -> None:
    assert "EntityCentralIndexKey" not in text


def test_inline_ixbrl_numbers_stay_in_sentence(text: str) -> None:
    assert "Total net sales were $391.0 billion in 2025." in text


def test_page_furniture_removed(text: str) -> None:
    assert "2025 Form 10-K" not in text
    assert "\n1\n" not in text


def test_table_rows_become_pipe_separated_lines(text: str) -> None:
    assert "Net sales | $ | 391,035" in text


def test_split_skips_table_of_contents(text: str) -> None:
    sections = split_10k_items(text)
    assert [s.item_code for s in sections] == ["1", "1A", "1B", "7", "8"]
    by_code = {s.item_code: s for s in sections}
    # Body text, not the TOC row, ended up in each section.
    assert by_code["1"].text.startswith("Business sentence 0")
    assert "supply chain concentration" in by_code["1A"].text
    assert by_code["1B"].text == "None."
    assert "gross margin" in by_code["7"].text
    # TOC lines never leak into a section.
    assert "TABLE OF CONTENTS" not in "".join(s.text for s in sections[1:])


def test_titles_from_same_line_and_next_line(text: str) -> None:
    by_code = {s.item_code: s for s in split_10k_items(text)}
    assert by_code["1"].title == "Business"
    assert by_code["1A"].title == "Risk Factors"  # title was on its own line
    assert by_code["1A"].text.startswith("Risk sentence number 0")
    assert by_code["7"].title is not None
    assert by_code["7"].title.startswith("Management’s Discussion")


def test_part_headers_do_not_start_sections(text: str) -> None:
    by_code = {s.item_code: s for s in split_10k_items(text)}
    assert not by_code["1"].text.startswith("PART")


def test_sentence_starting_with_item_is_not_a_heading() -> None:
    long_ref = "Item 7 of this report discusses " + "results in detail " * 10 + "for the year."
    text = "\n".join(
        [
            "Item 1. Business",
            "x " * 400,
            long_ref,
            "Item 7. MD&A",
            "y " * 400,
        ]
    )
    sections = split_10k_items(text)
    assert [s.item_code for s in sections] == ["1", "7"]
    assert long_ref in sections[0].text


def test_no_headings_returns_empty() -> None:
    assert split_10k_items("just some text\nwith no items") == []
