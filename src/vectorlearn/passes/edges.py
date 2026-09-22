"""Pass C: the dependency graph.

Two signals, in priority order:

  1. Author-declared cross-references, mined deterministically. Free, and
     the closest thing to ground truth a book offers.
  2. Model inference, asked only to fill what (1) did not cover.

Keeping them separate is what makes edge quality measurable: xref edges are
a precision baseline, and inferred edges can be scored against it.
"""

from __future__ import annotations

from ..ir import Edge, EdgePlan, Node, Span, XRef
from ..llm import Provider, Task
from .common import GROUNDING_RULE

SYSTEM = f"""\
You infer prerequisite edges between learning objectives from one textbook.

An edge A -> B means: a learner who has not met A cannot follow B. It is a
claim about comprehension, not about the order the book happens to use.

  - Add an edge only when B genuinely depends on A. A wrong edge either
    locks a learner out of material they are ready for, or lets them into
    material they are not - both are worse than a missing edge.
  - Do not add an edge just because A precedes B in the chapter.
  - Prefer the minimal set: if A -> B and B -> C, do not also assert A -> C.
  - The graph must stay acyclic.
  - Some edges are already known from the author's own cross-references and
    are listed as KNOWN below. Do not repeat them; find what they missed.
  - `confidence` is your own estimate, 0.0-1.0. Be honest: a 0.5 edge that
    is marked 0.5 is useful, a 0.5 edge marked 1.0 is damage.

{GROUNDING_RULE}
"""


def _section_owner(nodes: list[Node], spans_by_id: dict[str, Span]) -> dict[str, str]:
    """Map a section number to the node that owns most of its spans."""
    tally: dict[str, dict[str, int]] = {}
    for node in nodes:
        for sid in node.source_spans:
            span = spans_by_id.get(sid)
            if span and span.section:
                tally.setdefault(span.section, {}).setdefault(node.node_id, 0)
                tally[span.section][node.node_id] += 1
    return {
        section: max(counts, key=lambda n: counts[n]) for section, counts in tally.items()
    }


def edges_from_xrefs(
    nodes: list[Node], spans_by_id: dict[str, Span], xrefs: list[XRef]
) -> list[Edge]:
    """Turn author-declared backward references into prerequisite edges."""
    owner = _section_owner(nodes, spans_by_id)
    span_owner: dict[str, str] = {
        sid: node.node_id for node in nodes for sid in node.source_spans
    }

    seen: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    for x in xrefs:
        if not x.backward or x.target_kind not in ("section", "chapter"):
            continue
        src = owner.get(x.target)
        dst = span_owner.get(x.from_span)
        if not src or not dst or src == dst or (src, dst) in seen:
            continue
        seen.add((src, dst))
        edges.append(
            Edge(src=src, dst=dst, origin="xref", confidence=1.0, evidence=x.phrase[:200])
        )
    return edges


def infer_edges(
    provider: Provider, nodes: list[Node], known: list[Edge]
) -> tuple[list[Edge], list[str]]:
    """Ask the model only for edges the cross-reference miner did not find."""
    if len(nodes) < 2:
        return [], []

    roster = "\n".join(f"- {n.node_id}: {n.objective}" for n in nodes)
    known_txt = (
        "\n".join(f"- {e.src} -> {e.dst}" for e in known) or "(none)"
    )
    task = Task(
        name="infer_edges",
        system=SYSTEM,
        user=(
            f"OBJECTIVES\n{roster}\n\nKNOWN EDGES (from the author's own "
            f"cross-references — do not repeat)\n{known_txt}\n\n"
            "Return the additional prerequisite edges these objectives require."
        ),
        output_model=EdgePlan,
        tier="reason",
    )
    plan = provider.run(task)

    valid = {n.node_id for n in nodes}
    existing = {(e.src, e.dst) for e in known}
    out: list[Edge] = []
    warnings: list[str] = []

    for d in plan.edges:  # type: ignore[attr-defined]
        if d.src not in valid or d.dst not in valid:
            warnings.append(f"inferred edge references unknown node: {d.src} -> {d.dst}")
            continue
        if d.src == d.dst or (d.src, d.dst) in existing:
            continue
        existing.add((d.src, d.dst))
        out.append(
            Edge(src=d.src, dst=d.dst, origin="inferred",
                 confidence=d.confidence, evidence=d.evidence[:200])
        )

    return out, warnings


def break_cycles(edges: list[Edge]) -> tuple[list[Edge], list[str]]:
    """Drop the lowest-confidence edge on any cycle until the graph is a DAG."""
    warnings: list[str] = []
    edges = list(edges)

    while True:
        adj: dict[str, list[Edge]] = {}
        for e in edges:
            adj.setdefault(e.src, []).append(e)

        state: dict[str, int] = {}
        cycle: list[Edge] = []

        def walk(node: str, path: list[Edge]) -> bool:
            state[node] = 1
            for e in adj.get(node, []):
                s = state.get(e.dst, 0)
                if s == 1:
                    cycle.extend(path + [e])
                    return True
                if s == 0 and walk(e.dst, path + [e]):
                    return True
            state[node] = 2
            return False

        found = any(walk(n, []) for n in list(adj) if state.get(n, 0) == 0)
        if not found:
            return edges, warnings

        # xref edges are author-declared; drop an inferred one first.
        victim = min(cycle, key=lambda e: (e.origin == "xref", e.confidence))
        warnings.append(
            f"cycle broken by dropping {victim.origin} edge {victim.src} -> {victim.dst}"
        )
        edges.remove(victim)
