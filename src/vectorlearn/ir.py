"""The course IR: the format everything else is a renderer over.

Three layers live here:

  1. Source layer  (Span, XRef)   - produced deterministically from the book.
  2. Course layer  (Node, Step)   - produced by the generation passes.
  3. Provenance    (source_spans) - the link between them.

`source_spans` is the load-bearing field. Coverage, citations, fidelity
checking and selective regeneration all key off it. A Step or Check with
an empty `source_spans` is ungrounded and is rejected by validation.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# 1. Source layer - deterministic, no model involved
# --------------------------------------------------------------------------

SpanKind = Literal[
    "heading", "prose", "code", "figure", "table", "list", "math", "exercise"
]


class Span(BaseModel):
    """An atomic, addressable piece of the source book.

    Span ids are derived from document order, so they are stable across
    re-parses of the same file and can be cited by generated content.
    """

    span_id: str
    doc_id: str
    ordinal: int
    kind: SpanKind
    text: str
    section: str | None = None  # nearest enclosing numbered section, e.g. "7.2"
    chapter: str | None = None  # top-level TOC entry this span falls under
    heading_path: list[str] = Field(default_factory=list)
    word_count: int = 0
    teachable: bool = True  # set by the span classifier; see passes/a_parse

    @model_validator(mode="after")
    def _count(self) -> Span:
        if not self.word_count:
            object.__setattr__(self, "word_count", len(self.text.split()))
        return self


class XRef(BaseModel):
    """An author-declared cross-reference.

    These are the highest-precision prerequisite signal in the whole
    pipeline: when a textbook says "recall from Section 2.1", the author
    has stated a dependency edge for us.
    """

    from_span: str
    from_section: str | None
    target: str  # "2.1", "7", "3.4"
    target_kind: Literal["section", "chapter", "figure", "equation"]
    phrase: str  # the matched text, for auditing
    backward: bool  # target precedes source => prerequisite candidate


class SourceDoc(BaseModel):
    """Pass A output for one book."""

    book_id: str
    title: str
    source_hash: str
    spans: list[Span]
    xrefs: list[XRef] = Field(default_factory=list)

    # Parse diagnostics. `recovered_docs` counts content documents that
    # arrived without usable markup and had their structure reconstructed —
    # a non-zero value is the signal that this book came from a PDF.
    recovered_docs: int = 0
    toc_entries: int = 0

    def span_index(self) -> dict[str, Span]:
        return {s.span_id: s for s in self.spans}

    def teachable_spans(self) -> list[Span]:
        return [s for s in self.spans if s.teachable]


# --------------------------------------------------------------------------
# 2. Course layer - generated, schema-constrained
# --------------------------------------------------------------------------

# Deliberately small. Every type here is a renderer to build and a validation
# rule to maintain; add one only when a real book demands it.
StepType = Literal[
    "concept",        # the explanation
    "definition",     # formal statement of a term
    "worked_example", # traced instance, revealed progressively
    "predict",        # commit to an answer before the reveal
    "code_write",     # task for the playground
    "pitfall",        # the common misconception
]


class Step(BaseModel):
    """One teaching beat. The unit of a daily session."""

    step_id: str
    type: StepType
    title: str
    body: str  # markdown
    source_spans: list[str] = Field(min_length=1)
    est_seconds: int = Field(ge=15, le=900)

    # worked_example: ordered fragments revealed one at a time
    reveal: list[str] | None = None
    # predict: the question, the answer, and why
    prompt: str | None = None
    answer: str | None = None
    explanation: str | None = None
    # code_write: starter and tests for the playground
    starter_code: str | None = None
    test_code: str | None = None

    @model_validator(mode="after")
    def _type_contract(self) -> Step:
        if self.type == "worked_example" and not self.reveal:
            raise ValueError("worked_example requires `reveal`")
        if self.type == "predict" and not (self.prompt and self.answer):
            raise ValueError("predict requires `prompt` and `answer`")
        if self.type == "code_write" and not self.test_code:
            raise ValueError("code_write requires `test_code`")
        return self


CheckKind = Literal["recall", "predict", "code", "explain"]


class Check(BaseModel):
    """A gate item. A node turns green when these pass, not when it is read."""

    check_id: str
    kind: CheckKind
    prompt: str
    answer: str
    grading_criteria: list[str] = Field(min_length=1)
    source_spans: list[str] = Field(min_length=1)


class Environment(BaseModel):
    """Which surface the workspace mounts for this node."""

    kind: Literal["python", "javascript", "sql", "shell", "none"] = "none"
    setup: str | None = None


class Node(BaseModel):
    """A topic: one box on the map, one gate, N steps."""

    node_id: str  # stable slug - progress keys off this, never off content
    title: str
    objective: str  # one sentence, verb-first
    zone: str
    prereqs: list[str] = Field(default_factory=list)
    source_spans: list[str] = Field(min_length=1)
    environment: Environment = Field(default_factory=Environment)
    steps: list[Step] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    generated_by: str | None = None  # generator version, for selective regen

    @property
    def est_seconds(self) -> int:
        return sum(s.est_seconds for s in self.steps)


class Edge(BaseModel):
    src: str
    dst: str
    # Where the edge came from. `xref` edges are author-declared and are
    # trusted over `inferred` ones when the two disagree.
    origin: Literal["xref", "inferred", "manual"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: str | None = None


class Course(BaseModel):
    schema_version: int = SCHEMA_VERSION
    course_id: str
    title: str
    source_hash: str
    zones: list[str] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def node_index(self) -> dict[str, Node]:
        return {n.node_id: n for n in self.nodes}

    def est_seconds(self) -> int:
        return sum(n.est_seconds for n in self.nodes)

    def cited_spans(self) -> set[str]:
        """Every span id claimed by any node, step or check."""
        out: set[str] = set()
        for n in self.nodes:
            out.update(n.source_spans)
            for s in n.steps:
                out.update(s.source_spans)
            for c in n.checks:
                out.update(c.source_spans)
        return out

    def topological_levels(self) -> list[list[str]]:
        """Layer the DAG. Levels fall out of the graph; we never assign them."""
        incoming: dict[str, set[str]] = {n.node_id: set() for n in self.nodes}
        for e in self.edges:
            if e.dst in incoming and e.src in incoming:
                incoming[e.dst].add(e.src)

        levels, placed = [], set()
        while len(placed) < len(incoming):
            layer = sorted(
                nid for nid, deps in incoming.items()
                if nid not in placed and deps <= placed
            )
            if not layer:  # cycle - emit the remainder so the caller can see it
                levels.append(sorted(set(incoming) - placed))
                break
            levels.append(layer)
            placed.update(layer)
        return levels


# --------------------------------------------------------------------------
# 3. Pass output envelopes - what the model is asked to emit
# --------------------------------------------------------------------------

class NodePlan(BaseModel):
    """Pass B output for one chapter."""

    zone: str
    nodes: list[NodeDraft]


class NodeDraft(BaseModel):
    node_id: str
    title: str
    objective: str
    source_spans: list[str] = Field(min_length=1)
    environment_kind: Literal["python", "javascript", "sql", "shell", "none"] = "none"


class EdgeDraft(BaseModel):
    src: str
    dst: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str


class EdgePlan(BaseModel):
    """Pass C output: only the edges the cross-reference miner missed."""

    edges: list[EdgeDraft]


class StepBundle(BaseModel):
    """Pass D output for one node."""

    steps: list[Step]


class CheckBundle(BaseModel):
    """Pass E output for one node."""

    checks: list[Check]


class SpanVerdict(BaseModel):
    span_id: str
    teachable: bool
    reason: str


class SpanClassification(BaseModel):
    verdicts: list[SpanVerdict]


NodePlan.model_rebuild()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slug(text: str, max_len: int = 48) -> str:
    s = _SLUG_STRIP.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "untitled"


def content_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()[:32]
