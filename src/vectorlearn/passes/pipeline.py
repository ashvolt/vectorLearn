"""Pipeline orchestration: A -> B -> C -> D -> E.

Split into passes rather than one "convert the book" call for three reasons:
each pass is separately evaluable, separately cacheable, and separately
fixable. A monolithic call is none of those, and when its output is wrong
there is nothing to bisect.

In production, A-C run eagerly (the map appears while the learner is still
looking at the upload screen) and D-E run lazily on node unlock, so nobody
pays to generate the 180 nodes they never reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..ir import Course, Node, SourceDoc, slug
from ..llm import Provider, estimate_tokens
from ..parse import mine_xrefs, parse_epub
from . import edges as edges_pass
from . import lessons, nodes as nodes_pass, spans as spans_pass

GENERATOR_VERSION = "gen-v0.1"


@dataclass
class PipelineOptions:
    chapter: str | None = None       # doc_id substring or section prefix
    max_nodes: int | None = None     # cap D/E spend while iterating
    skip_lessons: bool = False       # A-C only: the map, without the lessons
    classify: bool = True            # run the teachable/scaffolding pass


@dataclass
class BuildReport:
    warnings: list[str] = field(default_factory=list)
    spans_total: int = 0
    spans_teachable: int = 0
    spans_selected: int = 0
    xrefs: dict[str, int] = field(default_factory=dict)
    edges_xref: int = 0
    edges_inferred: int = 0
    provider_calls: int = 0
    cache_hits: int = 0

    def log(self, msg: str) -> None:
        self.warnings.append(msg)


def chapter_names(doc: SourceDoc) -> list[str]:
    """Top-level chapter titles, in reading order."""
    return list(dict.fromkeys(s.chapter for s in doc.spans if s.chapter))


def select_chapter(doc: SourceDoc, chapter: str | None):
    """Narrow to one chapter. Phase 0 is deliberately one chapter wide.

    Books number their sections ("7.2") or they title them ("Getting Your
    Neurons to Work"); PDF conversions usually only have titles, recovered
    from the table of contents. All three are accepted, plus a 1-based index
    into the chapter list for when the title is long or awkward to type.
    """
    if not chapter:
        return doc.spans

    names = chapter_names(doc)

    # A bare number means the Nth chapter whenever the book has that many.
    # Section numbers are checked only when the index is out of range, so
    # "--chapter 7" reads as section 7 in a three-chapter excerpt and as the
    # seventh chapter in a full book — which is what someone typing it means.
    if chapter.isdigit() and 1 <= int(chapter) <= len(names):
        return [s for s in doc.spans if s.chapter == names[int(chapter) - 1]]

    needle = chapter.casefold()
    hits = [
        s for s in doc.spans
        if (s.chapter and needle in s.chapter.casefold())
        or (s.section and (s.section == chapter or s.section.startswith(chapter + ".")))
        or needle in s.doc_id.casefold()
    ]
    return hits or doc.spans


def build_course(
    epub_path: str | Path,
    provider: Provider,
    options: PipelineOptions | None = None,
) -> tuple[Course, SourceDoc, BuildReport]:
    opts = options or PipelineOptions()
    report = BuildReport()

    # --- Pass A: parse (deterministic) ------------------------------------
    doc = parse_epub(epub_path)
    doc = doc.model_copy(update={"xrefs": mine_xrefs(doc.spans)})
    report.spans_total = len(doc.spans)

    from ..parse.xrefs import xref_stats
    report.xrefs = xref_stats(doc.xrefs)

    selected = select_chapter(doc, opts.chapter)
    report.spans_selected = len(selected)

    # --- Pass A2: teachable vs scaffolding --------------------------------
    if opts.classify:
        selected = spans_pass.classify_spans(provider, selected)
        by_id = {s.span_id: s for s in selected}
        doc = doc.model_copy(
            update={"spans": [by_id.get(s.span_id, s) for s in doc.spans]}
        )
    report.spans_teachable = sum(1 for s in selected if s.teachable)

    spans_by_id = {s.span_id: s for s in selected}

    # --- Pass B: objectives ------------------------------------------------
    zone_hint = next((s.text for s in selected if s.kind == "heading"), None)
    course_nodes, warns = nodes_pass.plan_nodes(provider, selected, zone_hint=zone_hint)
    report.warnings.extend(warns)

    if not course_nodes:
        raise RuntimeError("pass B produced no nodes; nothing downstream can run")

    # --- Pass C: edges -----------------------------------------------------
    xref_edges = edges_pass.edges_from_xrefs(course_nodes, spans_by_id, doc.xrefs)
    report.edges_xref = len(xref_edges)

    positions = edges_pass.node_positions(course_nodes, spans_by_id)
    inferred, warns = edges_pass.infer_edges(
        provider, course_nodes, xref_edges, positions
    )
    report.warnings.extend(warns)

    inferred, warns = edges_pass.drop_backward_edges(inferred, positions)
    report.warnings.extend(warns)
    report.edges_inferred = len(inferred)

    all_edges, warns = edges_pass.break_cycles(xref_edges + inferred)
    report.warnings.extend(warns)

    prereqs: dict[str, list[str]] = {}
    for e in all_edges:
        prereqs.setdefault(e.dst, []).append(e.src)
    course_nodes = [
        n.model_copy(update={"prereqs": prereqs.get(n.node_id, [])}) for n in course_nodes
    ]

    # --- Passes D and E: lessons and gates ---------------------------------
    if not opts.skip_lessons:
        targets = course_nodes[: opts.max_nodes] if opts.max_nodes else course_nodes
        built: list[Node] = []
        for node in course_nodes:
            if node not in targets:
                built.append(node)
                continue
            steps, warns = lessons.generate_steps(provider, node, spans_by_id)
            report.warnings.extend(warns)
            node = node.model_copy(
                update={"steps": steps, "generated_by": GENERATOR_VERSION}
            )
            checks, warns = lessons.generate_checks(provider, node, spans_by_id)
            report.warnings.extend(warns)
            built.append(node.model_copy(update={"checks": checks}))
        course_nodes = built

    course = Course(
        course_id=slug(doc.title),
        title=doc.title,
        source_hash=doc.source_hash,
        zones=sorted({n.zone for n in course_nodes}),
        nodes=course_nodes,
        edges=all_edges,
    )

    report.provider_calls = getattr(provider, "calls", 0)
    report.cache_hits = getattr(provider, "cache_hits", 0)
    return course, doc, report


def plan_cost(epub_path: str | Path, options: PipelineOptions | None = None) -> dict:
    """Estimate the shape of a run without spending anything on it."""
    opts = options or PipelineOptions()
    doc = parse_epub(epub_path)
    selected = select_chapter(doc, opts.chapter)
    chars = sum(len(s.text) for s in selected)
    n_batches = (len(selected) + spans_pass.BATCH - 1) // spans_pass.BATCH

    # B + C are one call each; D + E are two per node. Node count is unknown
    # before pass B, so assume the middle of the prescribed 4-12 range.
    assumed_nodes = opts.max_nodes or 8
    return {
        "spans_selected": len(selected),
        "source_tokens_est": estimate_tokens_for(chars),
        "calls_classify": n_batches if opts.classify else 0,
        "calls_plan": 2,
        "calls_lessons": 0 if opts.skip_lessons else assumed_nodes * 2,
        "assumed_nodes": assumed_nodes,
    }


def estimate_tokens_for(chars: int) -> int:
    return estimate_tokens("x" * chars)
