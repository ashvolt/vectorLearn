import pytest
from pydantic import ValidationError

from vectorlearn.evals.coverage import measure_coverage
from vectorlearn.ir import Check, Course, Edge, Environment, Node, Step
from vectorlearn.parse import mine_xrefs, parse_epub
from vectorlearn.passes.edges import break_cycles, edges_from_xrefs

from vectorlearn.fixtures import build_sample_epub as build


def _step(**kw):
    base = dict(step_id="s1", type="concept", title="t", body="b",
                source_spans=["ch07:0001"], est_seconds=60)
    return Step(**{**base, **kw})


# --- the step type contract ------------------------------------------------

def test_ungrounded_step_is_rejected():
    with pytest.raises(ValidationError):
        _step(source_spans=[])


def test_worked_example_requires_reveal():
    with pytest.raises(ValidationError):
        _step(type="worked_example")
    assert _step(type="worked_example", reveal=["a", "b"]).reveal


def test_predict_requires_prompt_and_answer():
    with pytest.raises(ValidationError):
        _step(type="predict", prompt="what?")
    assert _step(type="predict", prompt="what?", answer="this").answer


def test_code_write_requires_tests():
    with pytest.raises(ValidationError):
        _step(type="code_write", starter_code="def f(): ...")
    assert _step(type="code_write", test_code="assert f() == 1").test_code


# --- the graph -------------------------------------------------------------

def _course(nodes, edges):
    return Course(course_id="c", title="T", source_hash="sha256:x",
                  nodes=nodes, edges=edges)


def _node(nid, spans=("ch07:0001",)):
    return Node(node_id=nid, title=nid, objective=f"do {nid}", zone="z",
                source_spans=list(spans))


def test_levels_fall_out_of_the_graph():
    c = _course(
        [_node("a"), _node("b"), _node("c")],
        [Edge(src="a", dst="b", origin="xref"), Edge(src="b", dst="c", origin="xref")],
    )
    assert c.topological_levels() == [["a"], ["b"], ["c"]]


def test_independent_nodes_share_a_level():
    c = _course(
        [_node("a"), _node("b"), _node("c")],
        [Edge(src="a", dst="c", origin="xref"), Edge(src="b", dst="c", origin="xref")],
    )
    levels = c.topological_levels()
    assert levels[0] == ["a", "b"] and levels[1] == ["c"]


def test_cycles_are_broken_by_dropping_the_weakest_inferred_edge():
    edges = [
        Edge(src="a", dst="b", origin="xref", confidence=1.0),
        Edge(src="b", dst="c", origin="inferred", confidence=0.9),
        Edge(src="c", dst="a", origin="inferred", confidence=0.3),
    ]
    kept, warns = break_cycles(edges)
    assert len(kept) == 2 and warns
    assert all(e.origin == "xref" or e.confidence > 0.3 for e in kept)
    # an author-declared edge survives a cycle with inferred ones
    assert any(e.origin == "xref" for e in kept)


def test_acyclic_graph_is_left_alone():
    edges = [Edge(src="a", dst="b", origin="xref")]
    kept, warns = break_cycles(edges)
    assert kept == edges and not warns


# --- xref edges, end to end on the fixture ---------------------------------

def test_author_cross_references_become_prerequisite_edges(tmp_path):
    doc = parse_epub(build(tmp_path / "c.epub"))
    xrefs = mine_xrefs(doc.spans)
    by_id = doc.span_index()

    def spans_in(section):
        return [s.span_id for s in doc.spans if s.section == section]

    nodes = [
        _node("array-indexing", spans_in("2.1")),
        _node("swap", spans_in("2.2")),
        _node("divide-step", spans_in("7.1")),
    ]
    edges = edges_from_xrefs(nodes, by_id, xrefs)
    pairs = {(e.src, e.dst) for e in edges}

    # "Recall from Section 2.1" in 7.1 is an author-declared prerequisite
    assert ("array-indexing", "divide-step") in pairs
    assert ("swap", "divide-step") in pairs
    assert all(e.origin == "xref" and e.evidence for e in edges)


# --- coverage --------------------------------------------------------------

def test_coverage_counts_only_teachable_spans(tmp_path):
    doc = parse_epub(build(tmp_path / "d.epub"))
    spans = [
        s.model_copy(update={"teachable": s.kind != "heading"}) for s in doc.spans
    ]
    cited = [s.span_id for s in spans if s.teachable][:3]
    course = _course([_node("n", cited)], [])

    cov = measure_coverage(course, spans)
    assert cov.teachable_total == sum(1 for s in spans if s.teachable)
    assert cov.covered == 3
    assert 0.0 < cov.span_coverage < 1.0
    assert cov.uncovered_spans
    assert not cov.fabricated_citations


def test_fabricated_citations_are_surfaced(tmp_path):
    doc = parse_epub(build(tmp_path / "e.epub"))
    course = _course([_node("n", ["ch07:9999"])], [])
    cov = measure_coverage(course, doc.spans)
    assert cov.fabricated_citations == ["ch07:9999"]


# --- edge direction ---------------------------------------------------------
# A small local model asserted softmax -> McCulloch-Pitts neuron on a chapter
# that teaches the neuron first and describes softmax as normalising its
# outputs. The direction was backwards in its own words, so document order is
# now treated as evidence rather than ignored.

def _positioned(**pos):
    return dict(pos)


def test_an_edge_against_book_order_is_rejected():
    from vectorlearn.passes.edges import drop_backward_edges

    edges = [Edge(src="softmax", dst="neuron", origin="inferred", confidence=1.0)]
    positions = {"neuron": 696.0, "softmax": 762.0}  # the book teaches neuron first
    kept, warns = drop_backward_edges(edges, positions)

    assert kept == []
    assert warns and "teaches neuron first" in warns[0]


def test_an_edge_following_book_order_is_kept():
    from vectorlearn.passes.edges import drop_backward_edges

    edges = [Edge(src="neuron", dst="softmax", origin="inferred", confidence=0.8)]
    kept, warns = drop_backward_edges(edges, {"neuron": 696.0, "softmax": 762.0})
    assert len(kept) == 1 and not warns


def test_author_declared_edges_are_exempt_from_the_order_check():
    """A cross-reference states its own direction; the prior does not overrule
    the author."""
    from vectorlearn.passes.edges import drop_backward_edges

    edges = [Edge(src="later", dst="earlier", origin="xref", confidence=1.0)]
    kept, _ = drop_backward_edges(edges, {"earlier": 10.0, "later": 90.0})
    assert len(kept) == 1


def test_positions_are_the_median_span_ordinal(tmp_path):
    from vectorlearn.parse import parse_epub
    from vectorlearn.passes.edges import node_positions

    doc = parse_epub(build(tmp_path / "pos.epub"))
    by_id = doc.span_index()
    early = [s.span_id for s in doc.spans[3:6]]
    late = [s.span_id for s in doc.spans[-4:-1]]
    nodes = [_node("early", early), _node("late", late)]

    pos = node_positions(nodes, by_id)
    assert pos["early"] < pos["late"]


def test_edges_without_positions_are_left_alone():
    """Missing evidence is not evidence of a problem."""
    from vectorlearn.passes.edges import drop_backward_edges

    edges = [Edge(src="a", dst="b", origin="inferred", confidence=0.5)]
    kept, warns = drop_backward_edges(edges, {})
    assert kept == edges and not warns


# --- decomposition size -----------------------------------------------------

def test_node_target_scales_with_the_material(tmp_path):
    from vectorlearn.parse import parse_epub
    from vectorlearn.passes.nodes import MAX_NODES, MIN_NODES, target_node_count

    doc = parse_epub(build(tmp_path / "size.epub"))
    small = target_node_count(doc.spans[:6])
    large = target_node_count(doc.spans * 12)

    assert small == MIN_NODES, "a handful of spans cannot support more"
    assert large > small
    assert large <= MAX_NODES
