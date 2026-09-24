"""Teachability: is this a lesson, or annotated source code?

Fidelity asks whether the content is true to the book. Coverage asks whether
it accounts for the book. Neither asks whether it teaches — and the first
real lesson this pipeline produced was faithful, complete, and close to
useless: two steps saying the same thing, a "worked example" that split a
config block across four lines, the chapter's one conceptual leap reduced to
a bare line of code, and tests that could not run.

Every check here is deterministic. That is a deliberate limit: a model asked
to rate teaching rates fluency, and a score that tracks polish would be worse
than no score, because it would be trusted. These instead ask narrow
falsifiable questions — is this step a copy of that one, do these reveal
fragments trace anything, would this assertion execute — which admit a right
answer and need no judgement.

Findings are advisory. They flag work for a human to look at, not a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ir import Course, Node, Step

READING_WORDS_PER_MINUTE = 200
DUPLICATE_OVERLAP = 0.6
MIN_WORDS_TO_COMPARE = 8


@dataclass
class Finding:
    kind: str
    where: str
    detail: str


@dataclass
class TeachabilityResult:
    findings: list[Finding] = field(default_factory=list)
    steps_checked: int = 0

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out


_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d+\.?\d*")
_OPENERS, _CLOSERS = "([{", ")]}"


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if len(wa) < MIN_WORDS_TO_COMPARE or len(wb) < MIN_WORDS_TO_COMPARE:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _unbalanced(text: str) -> bool:
    """True when brackets do not close within this fragment."""
    depth = 0
    for ch in text:
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


# --- individual checks ------------------------------------------------------

def _check_duplicates(node: Node) -> list[Finding]:
    out = []
    for i, a in enumerate(node.steps):
        for b in node.steps[i + 1:]:
            if _overlap(a.body, b.body) >= DUPLICATE_OVERLAP:
                out.append(Finding(
                    "duplicate_step", f"{a.step_id}+{b.step_id}",
                    f"[{a.type}] and [{b.type}] make the same point "
                    f"({_overlap(a.body, b.body):.0%} shared wording)",
                ))
    return out


def _check_reveal(step: Step) -> list[Finding]:
    """A trace shows state changing. Chopped-up code shows nothing.

    The signature of the failure is syntactic: fragments that do not close
    their own brackets are pieces of one statement, not successive states.
    """
    if step.type != "worked_example" or not step.reveal:
        return []

    fragments = [f for f in step.reveal if f.strip()]
    if not fragments:
        return []

    chopped = sum(1 for f in fragments if _unbalanced(f))
    if chopped >= max(1, len(fragments) // 2):
        return [Finding(
            "chopped_reveal", step.step_id,
            f"{chopped}/{len(fragments)} reveal fragments are pieces of one "
            "statement — a code listing split up, not a computation traced",
        )]

    # A trace carries values that change. No numbers anywhere is a smell.
    if not any(_NUMBER.search(f) for f in fragments):
        return [Finding(
            "valueless_reveal", step.step_id,
            "no values anywhere in the reveal — nothing for the learner to "
            "predict between steps",
        )]
    return []


def _check_predict(step: Step) -> list[Finding]:
    """An answer full of constants the prompt never mentions is a memory test."""
    if step.type != "predict" or not (step.prompt and step.answer):
        return []

    asked = set(_NUMBER.findall(step.prompt) or [])
    answered = set(_NUMBER.findall(step.answer))
    unguessable = answered - asked
    if len(unguessable) >= 4:
        return [Finding(
            "unpredictable_predict", step.step_id,
            f"answer introduces {len(unguessable)} constants the question "
            "never gives — recall of magic numbers, not reasoning",
        )]
    if step.answer.count("\n") >= 8:
        return [Finding(
            "unpredictable_predict", step.step_id,
            f"answer is {step.answer.count(chr(10)) + 1} lines long; nobody "
            "predicts that much",
        )]
    return []


def _check_test(step: Step) -> list[Finding]:
    """An assertion on a framework handle's shape or type tests nothing."""
    if step.type != "code_write" or not step.test_code:
        return []

    test = step.test_code
    if not re.search(r"\bassert\b|\bunittest\b|\bpytest\b", test):
        return [Finding("hollow_test", step.step_id, "test code asserts nothing")]

    checks = re.findall(r"assert\s+([^\n#]+)", test)
    if checks and all(
        re.search(r"\.(shape|dtype|ndim|type)\b|isinstance\(", c) for c in checks
    ):
        return [Finding(
            "hollow_test", step.step_id,
            "every assertion inspects a shape or type rather than a computed "
            "value — this passes without the code being correct",
        )]

    if step.starter_code:
        defined = set(re.findall(r"^\s*(\w+)\s*=", step.starter_code, re.M))
        defined |= set(re.findall(r"\bdef\s+(\w+)", step.starter_code))
        if defined and not (defined & set(_WORD.findall(test))):
            return [Finding(
                "hollow_test", step.step_id,
                "test references nothing the starter defines",
            )]
    return []


def _check_truncation(node: Node) -> list[Finding]:
    out = []
    fields = [(s.step_id, "body", s.body) for s in node.steps]
    fields += [(s.step_id, "answer", s.answer) for s in node.steps if s.answer]
    fields += [(c.check_id, "answer", c.answer) for c in node.checks]

    for where, name, text in fields:
        if not text:
            continue
        if text.count("```") % 2:
            out.append(Finding("truncated", where, f"{name} has an unclosed code fence"))
        elif re.search(r"[\w.]\.\w{0,6}$", text.strip()) and not text.strip().endswith("."):
            out.append(Finding("truncated", where, f"{name} stops mid-statement"))
    return out


def _check_timing(node: Node) -> list[Finding]:
    out = []
    for step in node.steps:
        words = len(step.body.split()) + len(" ".join(step.reveal or []).split())
        reading = words / READING_WORDS_PER_MINUTE * 60
        if step.est_seconds < reading * 0.75:
            out.append(Finding(
                "optimistic_timing", step.step_id,
                f"{step.est_seconds}s allotted for ~{words} words, which take "
                f"{reading:.0f}s just to read",
            ))
    return out


# --- entry point ------------------------------------------------------------

def measure_teachability(course: Course) -> TeachabilityResult:
    result = TeachabilityResult()
    for node in course.nodes:
        if not node.steps:
            continue
        result.steps_checked += len(node.steps)
        result.findings += _check_duplicates(node)
        result.findings += _check_truncation(node)
        result.findings += _check_timing(node)
        for step in node.steps:
            result.findings += _check_reveal(step)
            result.findings += _check_predict(step)
            result.findings += _check_test(step)
    return result
