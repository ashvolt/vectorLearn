"""Pass B: spans -> learning objectives (nodes).

A node is a topic, not a section. The whole point of compiling a book into
a graph is that chapter order is an authoring artifact and dependency order
is what a learner needs; this pass is where we stop mirroring the table of
contents and start naming what someone has to be able to *do*.
"""

from __future__ import annotations

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
  - 4 to 12 nodes per chapter. Fewer means each node is too big to finish in
    one sitting; more means you are splitting on paragraphs, not on ideas.
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


def plan_nodes(
    provider: Provider, spans: list[Span], *, zone_hint: str | None = None
) -> tuple[list[Node], list[str]]:
    """Return (nodes, warnings)."""
    teachable = [s for s in spans if s.teachable]
    allowed = {s.span_id for s in teachable}

    task = Task(
        name="plan_nodes",
        system=SYSTEM,
        user=(
            f"Chapter material{f' (zone: {zone_hint})' if zone_hint else ''}.\n"
            "Decompose it into learning objectives.\n\n"
            + render_spans(teachable)
        ),
        output_model=NodePlan,
        tier="reason",
    )
    plan = provider.run(task)

    nodes: list[Node] = []
    warnings: list[str] = []
    seen: set[str] = set()

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

    return nodes, warnings
