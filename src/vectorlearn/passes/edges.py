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

An edge A -> B means: **A must be understood before B makes sense.** A is the
foundation, B builds on it. Getting this direction backwards is the single
most damaging mistake available to you — it tells a learner to study the
advanced topic first — so state the direction to yourself before emitting
each edge.

  A useful check: if B's objective mentions A's concept as something it
  operates on, extends, or normalises, then A -> B, never the reverse. An
  objective that reads "apply X to a Y" depends on Y.

  - The objectives are listed **in the order the book teaches them**. That
    order is strong evidence: an author who teaches A before B is asserting
    that B does not depend on A. Edges that run against it are rejected
    unless the objectives themselves plainly demand it.
  - Add an edge only when B genuinely depends on A. A wrong edge either
    locks a learner out of material they are ready for, or lets them into
    material they are not - both are worse than a missing edge.
  - Do not add an edge merely because A precedes B; adjacency is not
    dependency.
  - Prefer the minimal set: if A -> B and B -> C, do not also assert A -> C.
  - The graph must stay acyclic.
  - Some edges are already known from the author's own cross-references and
    are listed as KNOWN below. Do not repeat them; find what they missed.
  - `confidence` is your own estimate, 0.0-1.0. Be honest: a 0.5 edge that
    is marked 0.5 is useful, a 0.5 edge marked 1.0 is damage.

{GROUNDING_RULE}
"""


def node_positions(
    nodes: list[Node], spans_by_id: dict[str, Span]
) -> dict[str, float]:
    """Where each node sits in the book, as the median ordinal of its spans.

    Textbooks are written so that prerequisites come first. That makes
    document order a strong prior on edge direction — strong enough to
    overrule a model that has asserted the opposite, which is a mistake
    small models make often and confidently.
    """
    out: dict[str, float] = {}
    for node in nodes:
        ordinals = sorted(
            spans_by_id[s].ordinal for s in node.source_spans if s in spans_by_id
        )
        if ordinals:
            out[node.node_id] = ordinals[len(ordinals) // 2]
    return out


def drop_backward_edges(
    edges: list[Edge], positions: dict[str, float]
) -> tuple[list[Edge], list[str]]:
    """Remove inferred edges that contradict the order the book teaches in.

    Author-declared edges are exempt: a cross-reference is explicit about its
    direction, and forward references ("as we will see") are already filtered
    out upstream.
    """
    kept: list[Edge] = []
    warnings: list[str] = []
    for e in edges:
        src, dst = positions.get(e.src), positions.get(e.dst)
        if e.origin != "xref" and src is not None and dst is not None and src > dst:
            warnings.append(
                f"dropped {e.src} -> {e.dst}: the book teaches {e.dst} first, "
                f"so the dependency cannot run this way"
            )
            continue
        kept.append(e)
    return kept, warnings


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
    provider: Provider,
    nodes: list[Node],
    known: list[Edge],
    positions: dict[str, float] | None = None,
) -> tuple[list[Edge], list[str]]:
    """Ask the model only for edges the cross-reference miner did not find."""
    if len(nodes) < 2:
        return [], []

    positions = positions or {}
    ordered = sorted(nodes, key=lambda n: positions.get(n.node_id, 0))
    roster = "\n".join(
        f"{i}. {n.node_id}: {n.objective}" for i, n in enumerate(ordered, 1)
    )
    known_txt = (
        "\n".join(f"- {e.src} -> {e.dst}" for e in known) or "(none)"
    )
    task = Task(
        name="infer_edges",
        system=SYSTEM,
        user=(
            f"OBJECTIVES, in the order the book teaches them\n{roster}\n\n"
            f"KNOWN EDGES (from the author's own cross-references — do not "
            f"repeat)\n{known_txt}\n\n"
            "Return the additional prerequisite edges these objectives require. "
            "Remember that A -> B means A must be understood first."
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
