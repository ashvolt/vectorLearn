"""Fidelity: is every taught claim actually in the book?

This is the eval that matters most, because of the bargain the product
makes. The learner never opens the source, so they cannot tell when a
lesson is wrong - a hallucinated definition, an inverted causal claim, a
dropped qualifier are all invisible to the only person who would care.

The judge stays on the reasoning tier. A cheap judge produces a flattering
number, and a flattering number on the one metric that gates the whole
project is worse than having no metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..ir import Course, Span
from ..llm import Provider, Task
from ..passes.common import render_spans

Verdict = Literal["supported", "unsupported", "contradicted"]


class AssertionVerdict(BaseModel):
    step_id: str
    assertion: str = Field(description="One factual claim, quoted or closely paraphrased")
    verdict: Verdict
    reasoning: str


class NodeFidelity(BaseModel):
    verdicts: list[AssertionVerdict]


SYSTEM = """\
You audit generated course material against the book excerpts it cites.

For each step you are given, break its body into individual factual
assertions - definitions, causal claims, complexity statements, procedural
instructions, numeric values. Then judge each one against the SOURCE SPANS
alone:

  supported     the spans state or directly entail this assertion
  unsupported   plausible, possibly true, but not present in the spans
  contradicted  the spans say something different

You are NOT judging whether an assertion is true in general. An assertion
that is correct computer science but absent from the spans is `unsupported`
- that is the whole point of the audit. The learner is being taught from
this book; material sourced from anywhere else is drift, however accurate.

Ordinary connective prose ("let's look at an example") is not an assertion.
Skip it. Judge substance.

Be specific in `reasoning`, and name the span id when one supports the claim.
"""


@dataclass
class FidelityResult:
    total: int = 0
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    findings: list[AssertionVerdict] = field(default_factory=list)
    nodes_audited: int = 0

    @property
    def fidelity(self) -> float:
        return self.supported / self.total if self.total else 0.0


def measure_fidelity(
    provider: Provider,
    course: Course,
    spans_by_id: dict[str, Span],
    *,
    max_nodes: int | None = None,
) -> FidelityResult:
    result = FidelityResult()
    targets = [n for n in course.nodes if n.steps]
    if max_nodes:
        targets = targets[:max_nodes]

    for node in targets:
        cited = sorted({s for step in node.steps for s in step.source_spans})
        spans = [spans_by_id[s] for s in cited if s in spans_by_id]
        if not spans:
            continue

        rendered = "\n\n".join(
            f"<step id={s.step_id} type={s.type}>\n{s.title}\n{s.body}\n"
            + (f"reveal: {s.reveal}\n" if s.reveal else "")
            + (f"answer: {s.answer}\n" if s.answer else "")
            + "</step>"
            for s in node.steps
        )

        task = Task(
            name="judge_fidelity",
            system=SYSTEM,
            user=(
                f"SOURCE SPANS\n\n{render_spans(spans)}\n\n"
                f"GENERATED STEPS TO AUDIT\n\n{rendered}"
            ),
            output_model=NodeFidelity,
            tier="reason",
            meta={"node_id": node.node_id},
        )
        audit = provider.run(task)
        result.nodes_audited += 1

        for v in audit.verdicts:  # type: ignore[attr-defined]
            result.total += 1
            if v.verdict == "supported":
                result.supported += 1
            else:
                result.findings.append(v)
                if v.verdict == "contradicted":
                    result.contradicted += 1
                else:
                    result.unsupported += 1

    return result
