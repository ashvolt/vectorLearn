"""Every check here was written from a specific defect in the first real
lesson the pipeline produced, so each test names the thing it caught."""

from __future__ import annotations

import pytest

from vectorlearn.evals.teachability import measure_teachability
from vectorlearn.ir import Check, Course, Node, Step


def _course(*steps: Step, checks=()):
    node = Node(node_id="n", title="t", objective="o", zone="z",
                source_spans=["a"], steps=list(steps), checks=list(checks))
    return Course(course_id="c", title="T", source_hash="sha256:x", nodes=[node])


def _kinds(*steps, checks=()):
    return {f.kind for f in measure_teachability(_course(*steps, checks=checks)).findings}


def _step(**kw):
    base = dict(step_id="s", type="concept", title="t", body="b" * 40,
                source_spans=["a"], est_seconds=120)
    return Step(**{**base, **kw})


# --- duplicate steps --------------------------------------------------------

def test_a_definition_restating_the_concept_is_flagged():
    """Two steps citing one span and saying the same thing is wasted time."""
    a = _step(step_id="s1", type="concept", body=(
        "The McCulloch-Pitts neuron is a simple model of a neuron. It takes "
        "inputs, multiplies them by weights, sums them, and applies an "
        "activation function to produce an output."))
    b = _step(step_id="s2", type="definition", body=(
        "The McCulloch-Pitts neuron is a mathematical model that contains "
        "inputs, weights, and an activation function. It sums the weighted "
        "inputs and applies a threshold to produce an output."))
    assert "duplicate_step" in _kinds(a, b)


def test_steps_making_different_points_are_not_flagged():
    a = _step(step_id="s1", body="A placeholder defines the shape of the graph "
                                 "without holding any data of its own.")
    b = _step(step_id="s2", body="Availability is one minus the load, so a "
                                 "heavily used location scores near zero.")
    assert "duplicate_step" not in _kinds(a, b)


# --- worked examples --------------------------------------------------------

def test_a_code_listing_split_across_lines_is_not_a_worked_example():
    """The exact failure: `config = tf.ConfigProto(` … `)` revealed one line
    at a time. There is nothing to predict between those fragments."""
    step = _step(type="worked_example", reveal=[
        "config = tf.ConfigProto(",
        "inter_op_parallelism_threads=4,",
        "intra_op_parallelism_threads=4",
        ")",
    ])
    assert "chopped_reveal" in _kinds(step)


def test_a_traced_computation_passes():
    step = _step(type="worked_example", reveal=[
        "x · w = 10(.1) + 2(.7) + 1(.75) = 3.15",
        "add bias: 3.15 + 1 = 4.15",
        "sigmoid(4.15) = 0.984",
        "availability = 1 - 0.984 = 0.016",
    ])
    assert not _kinds(step) & {"chopped_reveal", "valueless_reveal"}


def test_a_reveal_with_no_values_has_nothing_to_predict():
    step = _step(type="worked_example", reveal=[
        "first we take the inputs", "then we weight them", "then we sum",
    ])
    assert "valueless_reveal" in _kinds(step)


# --- predict ----------------------------------------------------------------

def test_recalling_magic_numbers_is_not_prediction():
    step = _step(type="predict", prompt="What code runs the session?",
                 answer="w_t = [[.1, .7, .75, .60, .20]]\n"
                        "x_1 = [[10, 2, 1., 6., 2.]]\nb_1 = [1]")
    assert "unpredictable_predict" in _kinds(step)


def test_a_reasoned_prediction_passes():
    step = _step(type="predict", prompt="The neuron outputs 0.99. What is availability?",
                 answer="0.01, because availability is 1 minus the load.")
    assert "unpredictable_predict" not in _kinds(step)


# --- tests that test nothing ------------------------------------------------

def test_asserting_on_a_framework_handles_shape_is_hollow():
    """`assert y.shape == (1,1)` on a TensorFlow tensor passes whether or not
    the computation is right."""
    step = _step(type="code_write", starter_code="y = tf.matmul(x, w) + b",
                 test_code="assert y.shape == (1, 1)\nassert s.shape == (1, 1)")
    assert "hollow_test" in _kinds(step)


def test_checking_a_computed_value_passes():
    step = _step(type="code_write", starter_code="def partition(a, lo, hi): ...",
                 test_code="assert partition([3, 1, 2], 0, 2) == 1")
    assert "hollow_test" not in _kinds(step)


def test_a_test_ignoring_the_starter_is_flagged():
    step = _step(type="code_write", starter_code="def partition(a, lo, hi): ...",
                 test_code="assert True")
    assert "hollow_test" in _kinds(step)


# --- truncation -------------------------------------------------------------

def test_a_reference_answer_cut_off_mid_statement_is_caught():
    """The real one ended `w = tf.place`."""
    check = Check(check_id="c1", kind="code", prompt="p",
                  answer="import tensorflow as tf\nx = tf.placeholder(tf.float32)\nw = tf.place",
                  grading_criteria=["x"], source_spans=["a"])
    assert "truncated" in _kinds(_step(), checks=[check])


def test_an_unclosed_code_fence_is_caught():
    assert "truncated" in _kinds(_step(body="Here is the code:\n```python\nx = 1"))


# --- timing -----------------------------------------------------------------

def test_time_below_the_reading_time_is_flagged():
    """30 seconds to meet the McCulloch-Pitts neuron for the first time."""
    step = _step(est_seconds=30, body=" ".join(["word"] * 200))
    assert "optimistic_timing" in _kinds(step)


def test_an_honest_estimate_passes():
    step = _step(est_seconds=120, body=" ".join(["word"] * 200))
    assert "optimistic_timing" not in _kinds(step)


# --- reporting --------------------------------------------------------------

def test_a_clean_lesson_reports_no_findings():
    step = _step(type="worked_example", est_seconds=120,
                 reveal=["x · w = 3.15", "+ bias = 4.15", "sigmoid = 0.984"])
    result = measure_teachability(_course(step))
    assert result.clean and result.steps_checked == 1


def test_truncated_content_fails_the_gate():
    from vectorlearn.evals.coverage import measure_coverage
    from vectorlearn.evals.report import render_report
    from vectorlearn.ir import SourceDoc, Span

    span = Span(span_id="a", doc_id="d", ordinal=0, kind="prose", text="text here")
    doc = SourceDoc(book_id="b", title="T", source_hash="sha256:x", spans=[span])
    course = _course(_step(body="```python\nx = 1", est_seconds=120))

    text = render_report(course, doc, measure_coverage(course, [span]), None, [])
    assert "[FAIL] no truncated content" in text
    assert "TEACHABILITY" in text
