"""Pass B: spans -> learning objectives (nodes).

A node is a topic, not a section. The whole point of compiling a book into
a graph is that chapter order is an authoring artifact and dependency order
is what a learner needs; this pass is where we stop mirroring the table of
contents and start naming what someone has to be able to *do*.
"""

from __future__ import annotations

import re

from ..ir import Node, NodePlan, Span
from ..llm import Provider, Task
from .common import GROUNDING_RULE, render_spans, validate_citations

SYSTEM = f"""\
You decompose a chapter of a technical textbook into learning objectives.

An objective is something a learner can DO, stated verb-first, and checkable:
  good: "Implement Lomuto partition and state its loop invariant"
  good: "Explain why a hash table degrades to O(n) under adversarial keys"
  bad:  "Chapter 7"                (a location, not a capability)
  bad:  "Understand sorting"       (not checkable)
  bad:  "Learn about heaps"        (not checkable)

Rules for the node set you produce:
  - Produce close to the TARGET COUNT you are given. It is derived from how
    much material there is. Coming in far under it means each node is too
    big to finish in one sitting and most of the chapter goes untaught;
    coming in far over means you are splitting on paragraphs, not on ideas.
  - Every substantive heading in the supplied spans should be represented.
    The headings are the author's own decomposition — departing from it
    needs a reason, and silently dropping three of them is not one.
  - Every node owns the spans that teach it, via `source_spans`. A span may
    be owned by more than one node, but material nobody owns is material the
    learner will never see - so between them the nodes should account for the
    substantive spans supplied.
  - Order nodes so that a node's dependencies appear before it where the
    chapter allows. Edges are inferred later; this is just a sane default.
  - `node_id` is a stable kebab-case slug of the concept, not of the section
    number: "quicksort-partition", not "section-7-2". It becomes the key the
    learner's progress is stored against forever, so it must describe the
    idea rather than its location in this edition.
  - Set `environment_kind` to the runtime a learner would need to practise
    this node ("python" for a node teaching an implementable algorithm,
    "none" for a purely conceptual one).

{GROUNDING_RULE}
"""


SPANS_PER_NODE = 20
MIN_NODES, MAX_NODES = 4, 14

# Chapter furniture. These are headings, but none of them is a topic a
# learner is meant to come away able to do, so counting them inflates the
# target and makes a sound decomposition look like a failure.
BOILERPLATE_HEADING = re.compile(
    r"^\s*(summary|questions?|further reading|references|bibliography|"
    r"technical requirements|introduction|conclusion|what (this|we) "
    r"(book )?covers?|in this chapter|exercises?|answers?|index|"
    r"acknowledge?ments?)\b",
    re.I,
)


def teachable_headings(spans: list[Span]) -> list[str]:
    """Section titles that name something a learner should be able to do."""
    chapter = next((s.chapter for s in spans if s.chapter), None)
    return [
        s.text for s in spans
        if s.kind == "heading"
        and s.text != chapter                    # the chapter's own title
        and not BOILERPLATE_HEADING.match(s.text)
    ]


def target_node_count(spans: list[Span]) -> int:
    """How many objectives this much material should yield.

    A fixed range invites a model to produce the minimum regardless of how
    much it was given — which is how a 197-span chapter came back as four
    nodes covering half of it. The heading count is a second opinion rather
    than the answer, and is discounted because sub-headings are often folded
    into their parent quite legitimately.
    """
    substantive = [s for s in spans if s.kind != "heading" and s.word_count > 3]
    headings = teachable_headings(spans)
    estimate = max(len(substantive) // SPANS_PER_NODE, round(len(headings) * 0.8))
    return max(MIN_NODES, min(MAX_NODES, estimate))


def plan_nodes(
    provider: Provider, spans: list[Span], *, zone_hint: str | None = None
) -> tuple[list[Node], list[str]]:
    """Return (nodes, warnings)."""
    teachable = [s for s in spans if s.teachable]
    allowed = {s.span_id for s in teachable}
    target = target_node_count(teachable)
    headings = teachable_headings(teachable)

    task = Task(
        name="plan_nodes",
        system=SYSTEM,
        user=(
            f"Chapter material{f' (zone: {zone_hint})' if zone_hint else ''}.\n"
            f"TARGET COUNT: about {target} objectives.\n"
            + (f"\nThe author's own headings, all of which should be covered:\n"
               + "\n".join(f"  - {h}" for h in headings) + "\n" if headings else "")
            + "\nDecompose it into learning objectives.\n\n"
            + render_spans(teachable)
        ),
        output_model=NodePlan,
        tier="reason",
    )
    plan = provider.run(task)

    nodes: list[Node] = []
    warnings: list[str] = []
    seen: set[str] = set()
    claimed: set[str] = set()

    for draft in plan.nodes:  # type: ignore[attr-defined]
        real, fake = validate_citations(draft.source_spans, allowed, where=draft.node_id)
        if fake:
            warnings.append(
                f"node {draft.node_id}: dropped {len(fake)} fabricated span id(s): {fake[:3]}"
            )
        if not real:
            warnings.append(f"node {draft.node_id}: no valid source spans, skipped")
            continue

        node_id = draft.node_id
        if node_id in seen:
            warnings.append(f"duplicate node_id {node_id}, skipped")
            continue
        seen.add(node_id)

        claimed.update(real)
        nodes.append(
            Node(
                node_id=node_id,
                title=draft.title,
                objective=draft.objective,
                zone=plan.zone,  # type: ignore[attr-defined]
                source_spans=real,
                environment={"kind": draft.environment_kind},  # type: ignore[arg-type]
            )
        )

    if nodes and len(nodes) < target * 0.55:
        warnings.append(
            f"produced {len(nodes)} nodes against a target of {target}: much of "
            "the chapter will have no lesson"
        )
    unclaimed = [s for s in teachable if s.span_id not in claimed and s.word_count > 8]
    if unclaimed:
        share = len(unclaimed) / max(1, len(teachable))
        warnings.append(
            f"{len(unclaimed)} substantive spans ({share:.0%}) owned by no node"
        )

    return nodes, warnings
