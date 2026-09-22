"""Pass A2: separate teachable content from scaffolding.

Textbooks carry less padding than trade non-fiction, but they still contain
front matter, acknowledgements, chapter recaps and navigational prose that
should not become lessons. Marking those keeps them out of the denominator
when coverage is measured - otherwise coverage punishes the pipeline for
correctly ignoring the copyright page.
"""

from __future__ import annotations

from ..ir import Span, SpanClassification
from ..llm import Provider, Task
from .common import render_spans

SYSTEM = """\
You classify spans from a technical textbook as teachable or not.

TEACHABLE - carries content a learner must understand or be able to do:
explanations, definitions, algorithms, code listings, worked examples,
figures that convey a mechanism, exercises, theorems and their proofs.

NOT TEACHABLE - scaffolding around the content: title and copyright pages,
dedications, acknowledgements, prefaces about the book itself, tables of
contents, navigation ("in this chapter we will…"), pure chapter recaps that
restate what was already taught, bibliographies, indexes, and author bios.

A span that restates content taught elsewhere with no new substance is NOT
teachable. When genuinely uncertain, mark it teachable - a false positive
costs a little generation budget; a false negative silently drops material
the learner is told they have covered.

Return a verdict for every span id you were given, and no others.
"""

BATCH = 40


def classify_spans(provider: Provider, spans: list[Span]) -> list[Span]:
    """Return spans with `teachable` set. Mutates nothing in place."""
    by_id = {s.span_id: s for s in spans}
    verdicts: dict[str, bool] = {}

    for i in range(0, len(spans), BATCH):
        batch = spans[i : i + BATCH]
        task = Task(
            name="classify_spans",
            system=SYSTEM,
            user=(
                "Classify each span.\n\n"
                + render_spans(batch, max_chars=600)
            ),
            output_model=SpanClassification,
            tier="bulk",
            meta={"batch": i // BATCH},
        )
        result = provider.run(task)
        for v in result.verdicts:  # type: ignore[attr-defined]
            if v.span_id in by_id:
                verdicts[v.span_id] = v.teachable

    return [
        s.model_copy(update={"teachable": verdicts.get(s.span_id, True)}) for s in spans
    ]
