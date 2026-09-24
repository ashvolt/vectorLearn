"""Exercises the generation passes with a stub provider.

These tests cover *our* plumbing - citation validation, warning generation,
node assembly, the scorecard gate - not the model's judgement. That split
matters: the model's output quality is what Phase 0 measures by hand, but
the code that catches a fabricated citation has to be right regardless of
which model is connected.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from vectorlearn.evals.coverage import measure_coverage
from vectorlearn.evals.report import render_report
from vectorlearn.ir import (
    Check, CheckBundle, Course, EdgeDraft, EdgePlan, Node, NodeDraft,
    NodePlan, SpanClassification, SpanVerdict, Step, StepBundle,
)
from vectorlearn.llm import Task
from vectorlearn.parse import parse_epub
from vectorlearn.passes import lessons, nodes as nodes_pass, spans as spans_pass

from vectorlearn.fixtures import build_sample_epub as build


class StubProvider:
    """Returns canned outputs keyed by task name; records what it was asked."""

    name = "stub"

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.seen: list[Task] = []
        self.calls = 0
        self.cache_hits = 0

    def model_for(self, tier):
        return f"stub-{tier}"

    def run(self, task: Task) -> BaseModel:
        self.seen.append(task)
        self.calls += 1
        r = self.responses[task.name]
        return r(task) if callable(r) else r


@pytest.fixture
def doc(tmp_path):
    return parse_epub(build(tmp_path / "p.epub"))


# --- pass A2 ---------------------------------------------------------------

def test_classification_marks_scaffolding(doc):
    def classify(task):
        ids = [s.span_id for s in doc.spans if s.span_id in task.user]
        return SpanClassification(
            verdicts=[
                SpanVerdict(span_id=i, teachable="front" not in i, reason="fixture")
                for i in ids
            ]
        )

    out = spans_pass.classify_spans(StubProvider({"classify_spans": classify}), doc.spans)
    assert out and len(out) == len(doc.spans)
    assert all(not s.teachable for s in out if s.doc_id == "front")
    assert any(s.teachable for s in out)


def test_classification_batches_large_inputs(doc):
    p = StubProvider({"classify_spans": SpanClassification(verdicts=[])})
    spans_pass.classify_spans(p, doc.spans * 4)
    assert p.calls > 1, "should batch rather than send every span in one call"


# --- pass B ----------------------------------------------------------------

def _plan(*drafts):
    return NodePlan(zone="Quicksort", nodes=list(drafts))


def test_fabricated_span_ids_are_dropped(doc):
    real = doc.spans[5].span_id
    plan = _plan(
        NodeDraft(node_id="partition", title="Partition", objective="Do it",
                  source_spans=[real, "ch99:0000"], environment_kind="python")
    )
    nodes, warns = nodes_pass.plan_nodes(StubProvider({"plan_nodes": plan}), doc.spans)

    assert len(nodes) == 1
    assert nodes[0].source_spans == [real]
    assert any("fabricated" in w for w in warns)


def test_wholly_ungrounded_node_is_skipped(doc):
    plan = _plan(
        NodeDraft(node_id="ghost", title="Ghost", objective="?",
                  source_spans=["ch99:0000"]),
        NodeDraft(node_id="real", title="Real", objective="!",
                  source_spans=[doc.spans[5].span_id]),
    )
    nodes, warns = nodes_pass.plan_nodes(StubProvider({"plan_nodes": plan}), doc.spans)
    assert [n.node_id for n in nodes] == ["real"]
    assert any("no valid source spans" in w for w in warns)


def test_duplicate_node_ids_are_rejected(doc):
    sid = doc.spans[5].span_id
    plan = _plan(
        NodeDraft(node_id="dup", title="A", objective="a", source_spans=[sid]),
        NodeDraft(node_id="dup", title="B", objective="b", source_spans=[sid]),
    )
    nodes, warns = nodes_pass.plan_nodes(StubProvider({"plan_nodes": plan}), doc.spans)
    assert len(nodes) == 1
    assert any("duplicate" in w for w in warns)


def test_only_teachable_spans_reach_the_prompt(doc):
    spans = [s.model_copy(update={"teachable": s.kind != "heading"}) for s in doc.spans]
    heading = next(s for s in spans if not s.teachable)
    p = StubProvider({"plan_nodes": _plan(
        NodeDraft(node_id="n", title="N", objective="o",
                  source_spans=[spans[5].span_id])
    )})
    nodes_pass.plan_nodes(p, spans)
    assert heading.span_id not in p.seen[0].user


# --- passes D and E --------------------------------------------------------

def _node(doc, n=4):
    return Node(node_id="partition", title="Partition", objective="Partition an array",
                zone="Quicksort", source_spans=[s.span_id for s in doc.spans[5:5 + n]],
                environment={"kind": "python"})


def _steps(*types):
    out = []
    for i, t in enumerate(types):
        kw = dict(step_id=f"x{i}", type=t, title=t, body="body",
                  source_spans=["PLACEHOLDER"], est_seconds=60)
        if t == "worked_example":
            kw["reveal"] = ["a", "b"]
        if t == "predict":
            kw.update(prompt="q", answer="a")
        if t == "code_write":
            kw.update(test_code="assert True")
        out.append(Step(**kw))
    return StepBundle(steps=out)


def test_missing_required_step_types_are_warned(doc):
    node = _node(doc)
    bundle = _steps("concept", "concept")
    for s in bundle.steps:
        s.source_spans = [node.source_spans[0]]

    steps, warns = lessons.generate_steps(
        StubProvider({"generate_steps": bundle}), node, doc.span_index()
    )
    assert len(steps) == 2
    assert any("no worked_example" in w for w in warns)
    assert any("no predict step" in w for w in warns)
    assert any("code_write" in w for w in warns), "python node without a coding task"


def test_step_ids_are_renumbered_against_the_node(doc):
    node = _node(doc)
    bundle = _steps("concept", "worked_example", "predict", "code_write")
    for s in bundle.steps:
        s.source_spans = [node.source_spans[0]]

    steps, _ = lessons.generate_steps(
        StubProvider({"generate_steps": bundle}), node, doc.span_index()
    )
    assert [s.step_id for s in steps] == [
        "partition-s01", "partition-s02", "partition-s03", "partition-s04"
    ]


def test_a_node_with_no_checks_cannot_gate(doc):
    node = _node(doc)
    checks, warns = lessons.generate_checks(
        StubProvider({"generate_checks": CheckBundle(checks=[])}), node, doc.span_index()
    )
    assert not checks
    assert any("NO CHECKS" in w for w in warns)


def test_step_prompt_only_carries_the_nodes_own_spans(doc):
    node = _node(doc, n=2)
    bundle = _steps("concept")
    bundle.steps[0].source_spans = [node.source_spans[0]]
    p = StubProvider({"generate_steps": bundle})
    lessons.generate_steps(p, node, doc.span_index())

    prompt = p.seen[0].user
    assert all(sid in prompt for sid in node.source_spans)
    outside = next(s.span_id for s in doc.spans if s.span_id not in node.source_spans)
    assert outside not in prompt, "a node must not be taught from spans it does not own"


# --- the scorecard ---------------------------------------------------------

def _built_course(doc):
    sid = doc.spans[5].span_id
    step = Step(step_id="n-s01", type="concept", title="t", body="b",
                source_spans=[sid], est_seconds=120)
    check = Check(check_id="n-c01", kind="recall", prompt="p", answer="a",
                  grading_criteria=["says the thing"], source_spans=[sid])
    node = Node(node_id="n", title="N", objective="o", zone="Z",
                source_spans=[sid], steps=[step], checks=[check])
    return Course(course_id="c", title="T", source_hash=doc.source_hash, nodes=[node])


def test_report_fails_the_gate_on_thin_coverage(doc):
    course = _built_course(doc)
    text = render_report(course, doc, measure_coverage(course, doc.spans), None, [])
    assert "GATE FAILED" in text
    assert "word coverage" in text


def test_report_flags_a_node_that_cannot_gate(doc):
    course = _built_course(doc)
    course.nodes[0].checks = []
    text = render_report(course, doc, measure_coverage(course, doc.spans), None, [])
    assert "[FAIL] every node can gate" in text


# --- the nodes listing ------------------------------------------------------

def test_nodes_listing_groups_by_graph_depth(doc, tmp_path, capsys, monkeypatch):
    """`nodes` is the view pass B is judged on, so it has to show the
    objectives and the dependency layering, not just ids."""
    from vectorlearn.cli import main
    from vectorlearn.ir import Course, Edge

    a = _node(doc)
    b = Node(node_id="bellman", title="B", objective="Derive the Bellman equation",
             zone="Z", source_spans=a.source_spans, prereqs=["partition"])
    course = Course(course_id="c", title="T", source_hash=doc.source_hash,
                    nodes=[a, b],
                    edges=[Edge(src="partition", dst="bellman", origin="xref")])

    out = tmp_path / "out"
    out.mkdir()
    (out / "course.json").write_text(course.model_dump_json(), encoding="utf-8")
    (out / "source.json").write_text(doc.model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["nodes"]) == 0
    text = capsys.readouterr().out
    assert "Derive the Bellman equation" in text
    assert "L0  partition" in text and "L1  bellman" in text
    assert "after partition" in text
    assert "author" in text, "an author-declared edge should be marked as such"


def test_show_with_no_node_id_lists_them(doc, tmp_path, capsys, monkeypatch):
    from vectorlearn.cli import main
    from vectorlearn.ir import Course

    course = Course(course_id="c", title="T", source_hash=doc.source_hash,
                    nodes=[_node(doc)])
    out = tmp_path / "out"
    out.mkdir()
    (out / "course.json").write_text(course.model_dump_json(), encoding="utf-8")
    (out / "source.json").write_text(doc.model_dump_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["show"]) == 0
    assert "partition" in capsys.readouterr().out
