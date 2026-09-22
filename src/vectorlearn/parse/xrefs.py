"""Cross-reference mining: the author's own prerequisite edges.

A textbook that says "recall from Section 2.1" has *stated* a dependency.
Mining those costs nothing, has very high precision, and turns the hardest
step in the pipeline (edge inference) from "the model guesses the whole
graph" into "the model fills documented gaps" - with a precision baseline
to measure against.
"""

from __future__ import annotations

import re

from ..ir import Span, XRef

_TARGET = r"(\d+(?:\.\d+)*)"

# Cues that mark a reference as pointing *backward* - the strongest
# prerequisite signal a book gives us.
_BACKWARD_CUE = re.compile(
    r"\b(recall|as we (?:saw|discussed|noted|showed)|we (?:saw|discussed|introduced|defined|proved)"
    r"|introduced in|defined in|described in|discussed in|covered in|from|earlier in|above in)\b",
    re.I,
)
_FORWARD_CUE = re.compile(
    r"\b(we (?:will|'ll) (?:see|discuss|cover|prove|return)|as we(?:'ll| will) see"
    r"|later in|deferred to|coming up in|returns? in)\b",
    re.I,
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\bSections?\s+{_TARGET}", re.I), "section"),
    (re.compile(rf"§\s*{_TARGET}"), "section"),
    (re.compile(rf"\bChapters?\s+{_TARGET}", re.I), "chapter"),
    (re.compile(rf"\bFigures?\s+{_TARGET}", re.I), "figure"),
    (re.compile(rf"\b(?:Equations?|Eqs?\.)\s+{_TARGET}", re.I), "equation"),
]

_WINDOW = 70  # characters of context scanned for a direction cue


def _as_tuple(ref: str | None) -> tuple[int, ...]:
    if not ref:
        return ()
    try:
        return tuple(int(p) for p in ref.split("."))
    except ValueError:
        return ()


def _is_backward(phrase: str, source: str | None, target: str) -> bool:
    """Decide whether a reference points at earlier material.

    Numeric ordering is authoritative when both sides are numbered; the
    surrounding language is the fallback, and a bare "see Section 2.1"
    with neither signal is treated as backward, which is the common case.
    """
    src_t, tgt_t = _as_tuple(source), _as_tuple(target)
    if src_t and tgt_t and src_t != tgt_t:
        return tgt_t < src_t
    if _FORWARD_CUE.search(phrase):
        return False
    return True


def mine_xrefs(spans: list[Span]) -> list[XRef]:
    """Extract author-declared references from prose and exercise spans."""
    out: list[XRef] = []
    for span in spans:
        if span.kind not in ("prose", "exercise", "list"):
            continue
        for pattern, target_kind in _PATTERNS:
            for m in pattern.finditer(span.text):
                target = m.group(1)
                if target == span.section:
                    continue  # a section citing itself is not an edge
                lo = max(0, m.start() - _WINDOW)
                phrase = span.text[lo : m.end() + 20]
                out.append(
                    XRef(
                        from_span=span.span_id,
                        from_section=span.section,
                        target=target,
                        target_kind=target_kind,  # type: ignore[arg-type]
                        phrase=phrase.strip(),
                        backward=_is_backward(phrase, span.section, target),
                    )
                )
    return out


def xref_stats(xrefs: list[XRef]) -> dict[str, int]:
    return {
        "total": len(xrefs),
        "backward": sum(1 for x in xrefs if x.backward),
        "forward": sum(1 for x in xrefs if not x.backward),
        "explicit_cue": sum(1 for x in xrefs if _BACKWARD_CUE.search(x.phrase)),
    }
