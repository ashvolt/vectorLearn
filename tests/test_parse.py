from pathlib import Path

import pytest

from vectorlearn.ir import Course, Edge, Node, Step
from vectorlearn.parse import mine_xrefs, parse_epub
from vectorlearn.passes.pipeline import select_chapter

from .fixture import build


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    book = build(tmp_path_factory.mktemp("books") / "fods.epub")
    return parse_epub(book)


def test_metadata(doc):
    assert doc.title == "Fundamentals of Data Structures"
    assert doc.source_hash.startswith("sha256:")
    assert doc.book_id == "fundamentals-of-data-structures"


def test_spans_are_ordered_and_unique(doc):
    ids = [s.span_id for s in doc.spans]
    assert len(ids) == len(set(ids))
    assert [s.ordinal for s in doc.spans] == sorted(s.ordinal for s in doc.spans)


def test_structure_survives_parsing(doc):
    kinds = {s.kind for s in doc.spans}
    assert {"heading", "prose", "code"} <= kinds
    code = [s for s in doc.spans if s.kind == "code"]
    assert any("def partition" in s.text for s in code)
    # `<=` inside a listing must survive entity decoding
    assert any("a[j] <= pivot" in s.text for s in code)


def test_sections_are_tracked(doc):
    sections = {s.section for s in doc.spans if s.section}
    assert {"2.1", "2.2", "7.1", "7.2", "7.3"} <= sections


def test_exercises_are_classified(doc):
    ex = [s for s in doc.spans if s.kind == "exercise"]
    assert ex, "exercise blocks should be detected from their heading"
    assert any("quicksort" in s.text.lower() for s in ex)


def test_figure_alt_text_is_kept(doc):
    figs = [s for s in doc.spans if s.kind == "figure"]
    assert any("Lomuto partition scanning" in s.text for s in figs)


def test_chapter_selection(doc):
    ch7 = select_chapter(doc, "7")
    assert ch7 and len(ch7) < len(doc.spans)
    assert all(s.section is None or s.section.startswith("7") for s in ch7)


def test_parse_is_deterministic(tmp_path):
    book = build(tmp_path / "a.epub")
    a, b = parse_epub(book), parse_epub(book)
    assert [s.span_id for s in a.spans] == [s.span_id for s in b.spans]
    assert a.source_hash == b.source_hash
