"""Coverage: did the course actually account for the book?

Without this the pipeline can silently drop a third of a chapter and nobody
notices - least of all the learner, who is being told they are 60% done.

Coverage is also what makes that percentage honest. "60% of this book" can
mean 60% of the source material, demonstrated by recall, rather than a
scroll position. Nothing else in the design can make that claim, and it is
only possible because generated content is bound to spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Course, Span


@dataclass
class CoverageResult:
    teachable_total: int
    covered: int
    uncovered_spans: list[Span] = field(default_factory=list)
    fabricated_citations: list[str] = field(default_factory=list)
    words_total: int = 0
    words_covered: int = 0

    @property
    def span_coverage(self) -> float:
        return self.covered / self.teachable_total if self.teachable_total else 0.0

    @property
    def word_coverage(self) -> float:
        """Weighted by length - dropping one long section matters more than
        dropping five one-line spans."""
        return self.words_covered / self.words_total if self.words_total else 0.0


def measure_coverage(course: Course, spans: list[Span]) -> CoverageResult:
    teachable = [s for s in spans if s.teachable]
    cited = course.cited_spans()
    known = {s.span_id for s in spans}

    covered = [s for s in teachable if s.span_id in cited]
    uncovered = [s for s in teachable if s.span_id not in cited]

    return CoverageResult(
        teachable_total=len(teachable),
        covered=len(covered),
        uncovered_spans=sorted(uncovered, key=lambda s: -s.word_count)[:40],
        fabricated_citations=sorted(cited - known),
        words_total=sum(s.word_count for s in teachable),
        words_covered=sum(s.word_count for s in covered),
    )
