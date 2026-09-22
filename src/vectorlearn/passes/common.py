"""Shared prompt plumbing for the generation passes."""

from __future__ import annotations

from ..ir import Span

MAX_SPAN_CHARS = 1600

GROUNDING_RULE = """\
GROUNDING (non-negotiable)

You are generating course material from ONE book, for a learner who will
never open that book. You are the author's proxy, so every claim you make
must be traceable to the excerpts you were given.

  - Use ONLY the supplied spans as source material. Do not add facts,
    history, alternative algorithms, library names or performance numbers
    from your own knowledge, however correct they may be.
  - Cite the span ids you actually used in `source_spans`. Never cite a
    span id that was not supplied.
  - If the supplied spans are too thin to teach the objective, say so by
    producing less, not by inventing more.
  - Keep the author's terminology and notation. If the book calls it a
    "sift-down", you call it a sift-down.
"""


def render_spans(spans: list[Span], *, max_chars: int = MAX_SPAN_CHARS) -> str:
    """Format spans for a prompt, with ids the model must cite back."""
    lines = []
    for s in spans:
        text = s.text if len(s.text) <= max_chars else s.text[:max_chars] + " […truncated]"
        section = f" §{s.section}" if s.section else ""
        lines.append(f"<span id={s.span_id} kind={s.kind}{section}>\n{text}\n</span>")
    return "\n\n".join(lines)


def validate_citations(
    cited: list[str], allowed: set[str], *, where: str
) -> tuple[list[str], list[str]]:
    """Split citations into (real, fabricated).

    A fabricated citation is a hallucination with a paper trail, and it is
    the cheapest hallucination in the pipeline to catch - the span either
    exists or it does not.
    """
    real = [c for c in cited if c in allowed]
    fake = [c for c in cited if c not in allowed]
    return real, fake
