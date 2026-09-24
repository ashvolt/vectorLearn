"""The Phase 0 scorecard.

Phase 0 has one job: decide whether the content layer is good enough to
build anything on top of. The thresholds below are the gate. Missing them
is not a prompt-tuning task to defer - it is the signal that no amount of
UI, gamification or streak design will save the project in its current form.
"""

from __future__ import annotations

from ..ir import Course, SourceDoc
from .coverage import CoverageResult
from .fidelity import FidelityResult

# Gate thresholds. Deliberately strict: the learner cannot audit this
# material, so the pipeline has to.
MIN_FIDELITY = 0.95
MAX_CONTRADICTED = 0
MIN_WORD_COVERAGE = 0.80

# A rate measured on a handful of assertions carries an interval wide enough
# to contain both pass and fail. Reporting PASS or FAIL from it states a
# confidence the measurement does not have, so below this the row abstains.
MIN_ASSERTIONS_FOR_GATE = 60


def _bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def render_report(
    course: Course,
    doc: SourceDoc,
    coverage: CoverageResult,
    fidelity: FidelityResult | None,
    build_warnings: list[str],
) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 72)
    add(f"  PHASE 0 SCORECARD — {course.title}")
    add(f"  source {course.source_hash}")
    add("=" * 72)

    # -- structure ---------------------------------------------------------
    levels = course.topological_levels()
    steps = [s for n in course.nodes for s in n.steps]
    checks = [c for n in course.nodes for c in n.checks]
    mins = course.est_seconds() // 60

    add("")
    add("STRUCTURE")
    add(f"  spans parsed        {len(doc.spans)}")
    add(f"  nodes               {len(course.nodes)}")
    add(f"  steps               {len(steps)}")
    add(f"  checks              {len(checks)}")
    add(f"  graph depth         {len(levels)} levels, widest {max((len(l) for l in levels), default=0)}")
    add(f"  estimated study     {mins} min  (~{mins / 20:.1f} days at 20 min/day)")

    by_type: dict[str, int] = {}
    for s in steps:
        by_type[s.type] = by_type.get(s.type, 0) + 1
    if by_type:
        mix = "  ".join(f"{k}:{v}" for k, v in sorted(by_type.items()))
        add(f"  step mix            {mix}")

    xref_edges = sum(1 for e in course.edges if e.origin == "xref")
    inferred = sum(1 for e in course.edges if e.origin == "inferred")
    total_edges = len(course.edges)
    share = xref_edges / total_edges if total_edges else 0.0
    add(f"  edges               {total_edges}  ({xref_edges} author-declared, {inferred} inferred)")
    add(f"  author-declared     {_bar(share)} {share:>6.1%}")

    # -- coverage ----------------------------------------------------------
    add("")
    add("COVERAGE — did the course account for the book?")
    add(f"  spans               {_bar(coverage.span_coverage)} {coverage.span_coverage:>6.1%}"
        f"  ({coverage.covered}/{coverage.teachable_total})")
    add(f"  words (weighted)    {_bar(coverage.word_coverage)} {coverage.word_coverage:>6.1%}")
    if coverage.fabricated_citations:
        add(f"  !! fabricated span ids: {len(coverage.fabricated_citations)}")
    if coverage.uncovered_spans:
        add("")
        add("  largest uncovered spans (read these — they are what the learner never sees):")
        for s in coverage.uncovered_spans[:8]:
            head = s.text[:78].replace("\n", " ")
            add(f"    [{s.span_id}] {s.word_count:>4}w {s.kind:<8} {head}")

    # -- fidelity ----------------------------------------------------------
    if fidelity:
        add("")
        add("FIDELITY — is every taught claim actually in the book?")
        add(f"  assertions          {fidelity.total} across {fidelity.nodes_audited} nodes")
        add(f"  supported           {_bar(fidelity.fidelity)} {fidelity.fidelity:>6.1%}")
        add(f"  unsupported         {fidelity.unsupported}")
        add(f"  contradicted        {fidelity.contradicted}")
        if fidelity.findings:
            add("")
            add("  findings:")
            for f in fidelity.findings[:12]:
                mark = "XX" if f.verdict == "contradicted" else "!!"
                add(f"    {mark} [{f.step_id}] {f.assertion[:88]}")
                add(f"       {f.reasoning[:88]}")

    # -- build warnings ----------------------------------------------------
    if build_warnings:
        add("")
        add(f"BUILD WARNINGS ({len(build_warnings)})")
        for w in build_warnings[:20]:
            add(f"  - {w}")
        if len(build_warnings) > 20:
            add(f"  … and {len(build_warnings) - 20} more")

    # -- the gate ----------------------------------------------------------
    add("")
    add("-" * 72)
    add("GATE")
    gates: list[tuple[str, bool | None, str]] = []
    if fidelity and fidelity.total:
        enough = fidelity.total >= MIN_ASSERTIONS_FOR_GATE
        gates.append((
            f"fidelity >= {MIN_FIDELITY:.0%}",
            (fidelity.fidelity >= MIN_FIDELITY) if enough else None,
            f"{fidelity.fidelity:.1%}" + (
                "" if enough
                else f"  (only {fidelity.total} assertions; "
                     f"{MIN_ASSERTIONS_FOR_GATE} needed to judge)"
            ),
        ))
        gates.append((
            "no contradicted claims",
            fidelity.contradicted <= MAX_CONTRADICTED,
            str(fidelity.contradicted),
        ))
    gates.append((
        f"word coverage >= {MIN_WORD_COVERAGE:.0%}",
        coverage.word_coverage >= MIN_WORD_COVERAGE,
        f"{coverage.word_coverage:.1%}",
    ))
    gates.append((
        "every node can gate (has checks)",
        all(n.checks for n in course.nodes if n.steps),
        f"{sum(1 for n in course.nodes if n.steps and not n.checks)} without",
    ))
    gates.append((
        "no fabricated citations",
        not coverage.fabricated_citations,
        str(len(coverage.fabricated_citations)),
    ))

    for label, ok, detail in gates:
        mark = "PASS" if ok else ("FAIL" if ok is False else " ?? ")
        add(f"  [{mark}] {label:<34} {detail}")

    decided = [ok for _, ok, _ in gates if ok is not None]
    undecided = any(ok is None for _, ok, _ in gates)
    add("")
    if not all(decided):
        add("  GATE FAILED — fix generation before building any UI on top of this.")
    elif undecided:
        add("  GATE UNDECIDED — everything measurable passes, but the sample is")
        add("  too small to rule on. Build more nodes and score them before")
        add("  concluding anything either way.")
    else:
        add("  GATE PASSED — the content layer holds. Phase 1 is justified.")
    add("")
    add("  Numbers are necessary, not sufficient. Read the generated nodes")
    add("  yourself before trusting this scorecard: `vectorlearn show <node_id>`.")
    add("-" * 72)

    return "\n".join(L)
