"""Passes D and E: the taught lesson and its gate.

These are the expensive passes, and in production they run lazily - when a
learner unlocks a node, not when the book is uploaded. Phase 0 runs them
eagerly over one chapter because the point of Phase 0 is to read the output
and judge it.

The step mix is prescribed rather than left open. Handing the model a form
to fill in is the single biggest quality lever available: an open request
for "a lesson" reliably returns prose, and prose is a book with extra clicks.
"""

from __future__ import annotations

from ..ir import CheckBundle, Node, Span, Step, StepBundle
from ..llm import Provider, Task
from .common import GROUNDING_RULE, render_spans, validate_citations

STEP_SYSTEM = f"""\
You write one micro-lesson for a learner who will never open the source book.

Produce 5 to 9 steps that teach the objective, in this shape:

  concept        WHY this exists and what problem it solves. Intuition
                 first. Not a restatement of the definition.
  definition     WHAT a term precisely means. If a `concept` step already
                 says the same thing in looser words, the two are one step,
                 not two — drop the weaker one. Two steps citing the same
                 span and saying the same thing is wasted time.
  worked_example **a computation traced with concrete values.** Take real
                 numbers from the source, push them through the procedure,
                 and let `reveal` show one *state change* at a time — the
                 intermediate result after each move, so the learner can
                 predict the next. Splitting a code listing across lines is
                 NOT a worked example: there is nothing to predict in
                 `config = tf.ConfigProto(` followed by `)`. If the spans
                 contain no values to trace, use a different step type.
  predict        `prompt` asks something the learner could reason out from
                 what came before. `answer` gives it, `explanation` says
                 why. Arbitrary constants are not predictable: asking
                 someone to recall `w = [[.1,.7,.75,.60,.20]]` tests memory
                 for magic numbers, not understanding. Ask what a value
                 will be, what breaks, or which of two paths is taken.
  code_write     a task for the playground. `starter_code` is a stub,
                 `test_code` holds assertions that **run and check
                 behaviour** — compare computed values, not the types or
                 shapes of framework objects. `assert y.shape == (1,1)` on
                 a framework handle tests nothing.
  pitfall        a mistake the SOURCE SPANS themselves identify, and why it
                 is wrong. Only include this step if the spans name the
                 error. Warnings you know from elsewhere about the library
                 or language are exactly the drift this whole task forbids —
                 omit the step rather than supply one from memory.

Required mix:
  - at least one `worked_example`
  - at least one `predict`
  - at least one `code_write` IF this node has a code environment
  - `pitfall` ONLY where the source material names a common error. No
    pitfall step is far better than one you supplied yourself.
  - do not open with a preamble step about what the lesson will cover
  - no two steps may make the same point; each must move the learner on

TEACH THE HARD PART

Somewhere in these spans is the one move a reader can follow mechanically
without understanding — a quantity reinterpreted as something else, a
formula whose shape is not obvious, a leap the author makes in a sentence.
Find it and spend a step on WHY, not just what. A lesson that reproduces
every line of code and skips that move leaves the learner able to type the
program and unable to say what it computes.

WHEN THE SOURCE CONTRADICTS ITSELF

Extracted text sometimes disagrees with itself — an array declared with six
slots and filled with five values, a number that does not match the figure.
Say so plainly in the step rather than passing the inconsistency on as
though it worked. Reproducing broken code silently is worse than noting it.

TIMING

`est_seconds` is how long a learner meeting this material for the first time
needs: reading it, thinking, and doing whatever the step asks. Reading alone
runs about 200 words a minute, so a 120-word explanation is at least 40
seconds before any thought. A `code_write` step is minutes, not seconds.
Daily sessions are planned by summing these, so a flattering number becomes
a promise the product breaks.

`est_seconds` is your honest estimate of how long that step takes a learner
who is meeting the material for the first time. A daily session is built by
summing these, so a flattering number becomes a broken promise later.

Write to be read on a phone: one idea per step, no step longer than a few
short paragraphs.

{GROUNDING_RULE}
"""

CHECK_SYSTEM = f"""\
You write the gate for one micro-lesson.

The node turns green when the learner passes these, so they must test
recall and application - not recognition, and never whether the page was
scrolled.

Produce 2 to 4 checks:
  recall   state a definition or property from memory
  predict  given an input, say what happens
  code     write code; `answer` holds a reference solution — the essential
           lines, complete and self-contained. Do not paste the whole
           program, and never leave it cut off mid-statement.
  explain  explain a mechanism in the learner's own words

`grading_criteria` is what a grader checks for: concrete, independently
checkable points, not "demonstrates understanding". For an `explain` check
these criteria are the entire basis of grading, so they carry the weight.

Parameterise where you can: prefer "sort this array: [given]" over a
question whose single answer can be memorised on a retry.

{GROUNDING_RULE}
"""


def _span_subset(node: Node, by_id: dict[str, Span]) -> list[Span]:
    return [by_id[s] for s in node.source_spans if s in by_id]


def generate_steps(
    provider: Provider, node: Node, spans_by_id: dict[str, Span]
) -> tuple[list[Step], list[str]]:
    spans = _span_subset(node, spans_by_id)
    allowed = {s.span_id for s in spans}
    env = node.environment.kind

    task = Task(
        name="generate_steps",
        system=STEP_SYSTEM,
        user=(
            f"OBJECTIVE: {node.objective}\n"
            f"TITLE: {node.title}\n"
            f"CODE ENVIRONMENT: {env}\n\n"
            "SOURCE SPANS — the only material you may teach from:\n\n"
            + render_spans(spans)
        ),
        output_model=StepBundle,
        tier="reason",
        meta={"node_id": node.node_id},
    )
    bundle = provider.run(task)

    steps: list[Step] = []
    warnings: list[str] = []
    for i, step in enumerate(bundle.steps):  # type: ignore[attr-defined]
        real, fake = validate_citations(step.source_spans, allowed, where=step.step_id)
        if fake:
            warnings.append(
                f"{node.node_id}/{step.step_id}: fabricated span id(s) {fake[:3]}"
            )
        if not real:
            warnings.append(f"{node.node_id}/{step.step_id}: ungrounded step, dropped")
            continue
        steps.append(
            step.model_copy(
                update={"source_spans": real, "step_id": f"{node.node_id}-s{i + 1:02d}"}
            )
        )

    kinds = {s.type for s in steps}
    if "worked_example" not in kinds:
        warnings.append(f"{node.node_id}: no worked_example")
    if "predict" not in kinds:
        warnings.append(f"{node.node_id}: no predict step")
    if env != "none" and "code_write" not in kinds:
        warnings.append(f"{node.node_id}: code environment but no code_write step")

    return steps, warnings


def generate_checks(
    provider: Provider, node: Node, spans_by_id: dict[str, Span]
) -> tuple[list, list[str]]:
    spans = _span_subset(node, spans_by_id)
    allowed = {s.span_id for s in spans}

    task = Task(
        name="generate_checks",
        system=CHECK_SYSTEM,
        user=(
            f"OBJECTIVE: {node.objective}\n"
            f"CODE ENVIRONMENT: {node.environment.kind}\n\n"
            "The lesson taught these steps:\n"
            + "\n".join(f"- [{s.type}] {s.title}" for s in node.steps)
            + "\n\nSOURCE SPANS:\n\n"
            + render_spans(spans)
        ),
        output_model=CheckBundle,
        tier="reason",
        meta={"node_id": node.node_id},
    )
    bundle = provider.run(task)

    checks, warnings = [], []
    for i, check in enumerate(bundle.checks):  # type: ignore[attr-defined]
        real, fake = validate_citations(check.source_spans, allowed, where=check.check_id)
        if fake:
            warnings.append(f"{node.node_id}/{check.check_id}: fabricated span id(s)")
        if not real:
            warnings.append(f"{node.node_id}/{check.check_id}: ungrounded check, dropped")
            continue
        checks.append(
            check.model_copy(
                update={"source_spans": real, "check_id": f"{node.node_id}-c{i + 1:02d}"}
            )
        )

    if not checks:
        warnings.append(f"{node.node_id}: NO CHECKS — node cannot gate")
    return checks, warnings
