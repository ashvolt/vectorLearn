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

  concept        the explanation. Intuition first, then precise. Short
                 paragraphs. Markdown.
  definition     a formal statement of a term the book defines.
  worked_example an instance traced through. `reveal` holds the fragments,
                 shown one at a time, so the learner can predict each move
                 before seeing it. Never dump the whole trace at once.
  predict        `prompt` asks a question, `answer` gives the answer,
                 `explanation` says why. The learner commits before seeing
                 the answer - this is where the learning happens.
  code_write     a task for the playground. `starter_code` is a stub,
                 `test_code` holds runnable assertions that decide pass/fail.
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
  code     write code; `answer` holds a reference solution
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
