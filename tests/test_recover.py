"""Structure recovery for EPUBs converted from PDF.

The failure this guards against is silent: a converted book parses without
error into thousands of line-sized fragments with no headings, no code and no
figures, and every downstream pass then produces plausible nonsense from it.
"""

from __future__ import annotations

import pytest

from vectorlearn.fixtures import build_flat_epub, build_sample_epub
from vectorlearn.parse import parse_epub
from vectorlearn.parse.recover import (
    is_full_line, line_width, looks_flat, looks_like_code, normalise, rejoin,
    useful_alt,
)
from vectorlearn.parse.toc import read_toc
from vectorlearn.passes.pipeline import chapter_names, select_chapter


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    return parse_epub(build_flat_epub(tmp_path_factory.mktemp("flat") / "conv.epub"))


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return parse_epub(build_sample_epub(tmp_path_factory.mktemp("clean") / "ok.epub"))


# --- detection --------------------------------------------------------------

def test_a_converted_book_is_recognised_as_flat(flat):
    assert flat.recovered_docs == 1
    assert flat.toc_entries == 3


def test_a_structured_book_is_left_alone(clean):
    """Recovery must not fire on markup that is already good."""
    assert clean.recovered_docs == 0


def test_flatness_needs_evidence():
    assert not looks_flat([], [])
    # A title page is not a chapter: too few blocks to judge either signal.
    assert not looks_flat(["prose"] * 5, [3] * 5)
    # Real headings and real code: not flat, however short the blocks.
    assert not looks_flat(["heading", "code", "prose"] * 20, [5] * 60)
    # Hundreds of line-sized blocks with no structure: flat.
    assert looks_flat(["prose"] * 400, [9] * 400)


# --- headings from the table of contents ------------------------------------

def test_headings_come_from_the_toc(flat):
    headings = [s for s in flat.spans if s.kind == "heading"]
    assert {h.text for h in headings} == {
        "Sorting",
        "Quicksort partitioning",
        "A heading long enough that the typesetter wrapped it",
    }


def test_toc_depth_becomes_heading_level(flat):
    """Nesting in the NCX is the only source of hierarchy in a flat book."""
    top = [s for s in flat.spans if s.chapter == "Sorting"]
    assert top, "the depth-1 entry should become the chapter"
    assert all(s.chapter == "Sorting" for s in flat.spans if s.kind != "heading"
               or s.text != "Sorting")


def test_a_wrapped_heading_is_not_left_duplicated_in_the_body(flat):
    """Regression: a title long enough to wrap arrives as two fragments and
    only becomes recognisable once rejoined, so the echo must be dropped
    after merging rather than before."""
    title = "A heading long enough that the typesetter wrapped it"
    kinds = [s.kind for s in flat.spans if s.text == title]
    assert kinds == ["heading"], f"title also survives as {kinds}"


def test_no_heading_is_echoed_as_prose(flat):
    titles = {s.text for s in flat.spans if s.kind == "heading"}
    assert not [s for s in flat.spans if s.kind != "heading" and s.text in titles]


# --- content recovery -------------------------------------------------------

def test_code_is_recovered_without_a_pre_tag(flat):
    code = [s for s in flat.spans if s.kind == "code"]
    assert code, "the listing should not stay classified as prose"
    joined = "\n".join(s.text for s in code)
    assert "def partition" in joined and "pivot = a[hi]" in joined


def test_figures_are_recovered_and_junk_alt_discarded(flat):
    figs = [s for s in flat.spans if s.kind == "figure"]
    assert len(figs) == 1
    assert "Image 42" not in figs[0].text, "placeholder alt text carries no information"


def test_line_fragments_are_rejoined_into_paragraphs(flat):
    prose = [s for s in flat.spans if s.kind == "prose"]
    assert prose
    opening = next(s for s in prose if s.text.startswith("Sorting an array"))
    # All three source lines, reassembled.
    assert "compare" in opening.text and "exchange them." in opening.text
    assert max(s.word_count for s in prose) > 20


def test_cross_references_survive_recovery(flat):
    from vectorlearn.parse import mine_xrefs

    xrefs = mine_xrefs(flat.spans)
    assert any(x.target == "2.1" and x.backward for x in xrefs), \
        "the author's own prerequisite signal must survive reassembly"


# --- the heuristics themselves ----------------------------------------------

def test_full_line_detection_drives_rejoining():
    width = line_width([80] * 30 + [20] * 5)
    assert width == 80
    assert is_full_line("x" * 76, width)
    assert not is_full_line("x" * 30, width)
    # A full line continues even when it happens to end on a full stop.
    assert rejoin("x" * 78 + ".", "Next line continues here", width)
    # A short line ends the paragraph.
    assert not rejoin("Short line.", "Next paragraph starts", width)


def test_hyphenated_line_break_is_stitched():
    assert rejoin("an ever-evolv-", "ing platform", 0)


def test_list_markers_start_something_new():
    assert not rejoin("The options are", "- first option", 200)


def test_nonbreaking_spaces_are_collapsed():
    assert normalise("No\xa0\xa0matter\xa0 which") == "No matter which"


@pytest.mark.parametrize("text", [
    "import numpy as np", "def f(x):", "    y = x + 1", "[[ 1.5 2.5]", ">>> print(x)",
])
def test_code_shapes(text):
    assert looks_like_code(text)


@pytest.mark.parametrize("text", [
    "The centroids produced confirm the model.",
    "Sorting an array means rearranging its elements.",
])
def test_prose_is_not_mistaken_for_code(text):
    assert not looks_like_code(text)


def test_placeholder_alt_text_is_rejected():
    assert useful_alt("Image 106") is None
    assert useful_alt("Figure 3") is None
    assert useful_alt("A red-black tree rotation") == "A red-black tree rotation"


# --- chapter selection ------------------------------------------------------

def test_chapters_are_listed_in_reading_order(flat):
    assert chapter_names(flat) == ["Sorting"]


def test_chapter_selection_by_title(flat):
    picked = select_chapter(flat, "sorting")
    assert picked and len(picked) <= len(flat.spans)
    assert all(s.chapter == "Sorting" for s in picked)


def test_chapter_selection_by_index(flat):
    """A book with no section numbering still needs a way to say 'chapter 1'."""
    assert select_chapter(flat, "1") == select_chapter(flat, "Sorting")


def test_numbered_sections_still_win_over_index(clean):
    """Where a book does number its sections, --chapter 7 means section 7."""
    picked = select_chapter(clean, "7")
    assert picked and all(
        s.section is None or s.section.startswith("7") for s in picked
    )
