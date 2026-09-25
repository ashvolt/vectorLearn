"""Model qualification: can this local model actually drive the pipeline?

Runs the real passes against the built-in sample chapter, so it needs no book
and no network. The point is to answer, before any real work starts, which of
the models on *this machine* is worth spending a Phase 0 run on.

Four things are measured, in descending order of how much they matter:

  1. **Schema compliance** — can the model emit valid IR at all, and on the
     first try? A model that needs two repairs per call will not finish a book.
  2. **Citation discipline** — does it cite only span ids it was given?
     Fabricated ids are hallucinations with a paper trail.
  3. **Grounding** — does it stay on the source? Measured deterministically:
     the sample chapter teaches Lomuto partition and nothing else, so a lesson
     that mentions Hoare, median-of-three or introsort is reciting training
     data under the author's name. No judge required.
  4. **Instruction adherence** — node counts in range, the required step mix
     present, checks that can actually gate.

Throughput is recorded too, because a model that passes but runs at four
tokens a second is not a model you will use.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .fixtures import build_sample_epub, find_drift
from .ir import Node, Span
from .llm.ollama import CallStat, OllamaProvider
from .parse import parse_epub
from .passes import lessons, nodes as nodes_pass
from .passes.pipeline import select_chapter

Grade = Literal["gold", "silver", "bronze", "unqualified"]

GRADE_MEANING: dict[Grade, str] = {
    "gold": "full pipeline — all step types, synthesis checks, nuanced grading",
    "silver": "simplified IR — fewer step types, templated checks, no boss fights",
    "bronze": "structure only — usable for parsing and graph work, not for lessons",
    "unqualified": "cannot hold the schema; not usable for this pipeline",
}


@dataclass
class Qualification:
    model: str
    grade: Grade = "unqualified"
    reasons: list[str] = field(default_factory=list)

    tasks_run: int = 0
    tasks_ok: int = 0
    first_try: int = 0
    total_attempts: int = 0

    nodes: int = 0
    steps: int = 0
    checks: int = 0
    fabricated: int = 0
    drift: list[str] = field(default_factory=list)
    missing_types: list[str] = field(default_factory=list)

    seconds: float = 0.0
    output_tokens: int = 0
    error: str = ""

    @property
    def first_try_rate(self) -> float:
        return self.first_try / self.tasks_run if self.tasks_run else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0


def _sample_spans() -> list[Span]:
    with tempfile.TemporaryDirectory() as tmp:
        doc = parse_epub(build_sample_epub(Path(tmp) / "sample.epub"))
    return select_chapter(doc, "7")


def qualify_model(
    model: str, *, host: str | None = None, cache: bool = False
) -> Qualification:
    """Run one model through the pipeline's reasoning tasks."""
    q = Qualification(model=model)
    spans = _sample_spans()
    allowed = {s.span_id for s in spans}
    by_id = {s.span_id: s for s in spans}

    provider = OllamaProvider(
        models={"reason": model, "bulk": model},
        host=host or "http://localhost:11434",
        # A cached run would measure nothing, so qualification never reuses one.
        cache_dir=".vlcache/qualify" if cache else None,
        record=True,
    )

    def tally(warnings: list[str]) -> None:
        q.fabricated += sum(1 for w in warnings if "fabricated" in w)

    # --- Pass B: can it decompose a chapter into objectives? --------------
    try:
        q.tasks_run += 1
        built, warns = nodes_pass.plan_nodes(provider, spans, zone_hint="Quicksort")
        q.tasks_ok += 1
        q.nodes = len(built)
        tally(warns)
        if not 4 <= q.nodes <= 12:
            q.reasons.append(f"produced {q.nodes} nodes; the prompt asks for 4–12")
    except Exception as exc:  # noqa: BLE001 - any failure is a qualification result
        q.error = f"pass B: {exc}"
        return _grade(q, provider.stats)

    if not built:
        q.error = "pass B produced no usable nodes"
        return _grade(q, provider.stats)

    # --- Passes D and E on the richest node -------------------------------
    target: Node = max(built, key=lambda n: len(n.source_spans))
    try:
        q.tasks_run += 1
        steps, warns = lessons.generate_steps(provider, target, by_id)
        q.tasks_ok += 1
        q.steps = len(steps)
        tally(warns)
        q.missing_types = [
            t for t in ("worked_example", "predict")
            if t not in {s.type for s in steps}
        ]
        if not 5 <= q.steps <= 9:
            q.reasons.append(f"produced {q.steps} steps; the prompt asks for 5–9")

        body = "\n".join(
            " ".join(filter(None, [s.title, s.body, s.answer, s.explanation]))
            + " ".join(s.reveal or [])
            for s in steps
        )
        q.drift = find_drift(body)

        target = target.model_copy(update={"steps": steps})
    except Exception as exc:  # noqa: BLE001
        q.error = f"pass D: {exc}"
        return _grade(q, provider.stats)

    try:
        q.tasks_run += 1
        checks, warns = lessons.generate_checks(provider, target, by_id)
        q.tasks_ok += 1
        q.checks = len(checks)
        tally(warns)
        if not checks:
            q.reasons.append("generated no checks — nodes could not gate")
    except Exception as exc:  # noqa: BLE001
        q.error = f"pass E: {exc}"

    return _grade(q, provider.stats)


def _grade(q: Qualification, stats: list[CallStat]) -> Qualification:
    q.total_attempts = sum(s.attempts for s in stats)
    q.first_try = sum(1 for s in stats if s.ok and s.attempts == 1)
    q.seconds = sum(s.seconds for s in stats)
    q.output_tokens = sum(s.output_tokens for s in stats)

    if q.tasks_ok < q.tasks_run or q.error:
        q.grade = "unqualified"
        if q.error:
            q.reasons.insert(0, q.error)
        else:
            q.reasons.insert(0, "could not produce schema-valid output")
        return q

    # Drift and fabrication are grounding failures: the model is generating
    # from its own priors rather than from the book. That caps it at bronze —
    # useful for structure, never for teaching.
    if q.drift:
        q.reasons.append(f"taught material absent from the source: {', '.join(q.drift)}")
    if q.fabricated:
        q.reasons.append(f"cited {q.fabricated} span id(s) it was never given")
    if q.drift or q.fabricated:
        q.grade = "bronze"
        return q

    if q.missing_types:
        q.reasons.append(f"omitted required step types: {', '.join(q.missing_types)}")
    if q.first_try_rate < 1.0:
        q.reasons.append(
            f"needed {q.total_attempts - q.tasks_run} schema repair(s) across {q.tasks_run} tasks"
        )

    q.grade = "silver" if q.reasons else "gold"
    return q


# --- reporting --------------------------------------------------------------

_BADGE = {"gold": "GOLD", "silver": "SILVER", "bronze": "BRONZE", "unqualified": "UNQUAL"}

GRADE_ORDER: dict[str, int] = {"gold": 0, "silver": 1, "bronze": 2, "unqualified": 3}


def record(results: list[Qualification], path: Path) -> None:
    """Save grades so `build` can prefer a model that was actually tested."""
    import json

    rows = [
        {"model": q.model, "grade": q.grade,
         "tokens_per_second": round(q.tokens_per_second, 2)}
        for q in results
    ]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def best_qualified(path: Path) -> tuple[str, str] | None:
    """The best-graded model from a previous qualification, if any passed."""
    import json

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    usable = [r for r in rows if r.get("grade") in ("gold", "silver")]
    if not usable:
        return None
    best = min(usable, key=lambda r: (GRADE_ORDER[r["grade"]],
                                      -r.get("tokens_per_second", 0)))
    return best["model"], best["grade"]


def render_qualification(results: list[Qualification]) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 74)
    add("  MODEL QUALIFICATION — sample chapter, no book or network required")
    add("=" * 74)
    add("")
    add(f"  {'MODEL':<30} {'GRADE':<8} {'1ST TRY':>8} {'TOK/S':>7} {'TIME':>7}")
    add(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 7}")
    for q in results:
        add(
            f"  {q.model[:30]:<30} {_BADGE[q.grade]:<8} "
            f"{q.first_try_rate:>7.0%} {q.tokens_per_second:>7.1f} {q.seconds:>6.0f}s"
        )

    for q in results:
        add("")
        add("-" * 74)
        add(f"  {q.model}   →   {_BADGE[q.grade]}")
        add(f"  {GRADE_MEANING[q.grade]}")
        add("")
        add(f"    nodes {q.nodes}   steps {q.steps}   checks {q.checks}"
            f"   fabricated citations {q.fabricated}   drift terms {len(q.drift)}")
        if q.reasons:
            add("")
            for r in q.reasons:
                add(f"    - {r}")

    best = next((q for q in results if q.grade == "gold"), None) \
        or next((q for q in results if q.grade == "silver"), None)

    add("")
    add("=" * 74)
    if best:
        add(f"  USE: {best.model}")
        add("")
        add(f"    vectorlearn build book.epub --chapter 7 --model {best.model} --eval")
        if best.grade == "silver":
            add("")
            add("  Silver is workable for Phase 0 — read the reasons above and expect")
            add("  to hand-fix the gaps it leaves.")
    else:
        add("  No model on this machine reached silver.")
        add("")
        add("  Before concluding the pipeline is wrong, try a larger model: schema")
        add("  adherence and staying on-source both improve sharply with parameter")
        add("  count. If the largest model your hardware runs still grades bronze,")
        add("  that is a finding about local models, not about the design.")
    add("=" * 74)
    return "\n".join(L)
