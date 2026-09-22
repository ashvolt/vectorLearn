# vectorLearn

A **course compiler**. A technical book goes in; a dependency-graph course
comes out — objectives, a prerequisite DAG, taught lessons bound to the
source, and gates that a learner has to actually pass.

The premise: chapter order is an authoring artifact, dependency order is
what a learner needs. Compiling a book into a graph is what lets the course
answer *"which six of these forty sections do I actually need, and in what
order?"* — which is work nobody does by hand.

> **Status: Phase 0.** No UI, no server, no accounts. Phase 0 exists to
> answer one question — *is the generated content good enough to build on?* —
> and to stop the project if the answer is no. See [Phase 0](#phase-0) below.

## The format is the product

Everything else is a renderer over `src/vectorlearn/ir.py`. The map, the
reader, the playground, the notes panel — all of it reads this one schema.
Get it right and the UI can be rebuilt three times without losing a single
learner's progress.

```
Course              the compiled book
 └─ Node            a topic: one box on the map, one gate       (stable node_id)
     ├─ Step        one teaching beat — the daily unit          (typed)
     └─ Check       the gate: green means passed, not read
```

Two fields carry most of the weight:

- **`source_spans`** — every step and check names the source passages it was
  generated from. Coverage, citations, fidelity auditing and selective
  regeneration all key off this. A step with no spans is ungrounded and is
  rejected by schema validation, not by convention.
- **`node_id`** — stable, slugged from the *concept* rather than the section
  number. Progress is stored against it, so regenerating a lesson with a
  better prompt must never cost a learner their map.

Step types are deliberately few — six. Each one is a renderer to build and a
validation rule to maintain, and a model handed a form to fill in produces
far better lessons than one handed a blank page. `worked_example` reveals
progressively; `predict` makes the learner commit before the answer.

## Pipeline

Passes, not one conversion — each is separately evaluable, cacheable and
fixable, and when output is wrong there is something to bisect.

| Pass | What | Model | When (in production) |
|------|------|-------|----------------------|
| **A** | EPUB → spans, cross-reference mining | none | on upload |
| **A2** | teachable content vs. scaffolding | bulk | on upload |
| **B** | spans → learning objectives | reasoning | on upload |
| **C** | prerequisite edges | mined + reasoning | on upload |
| **D** | objectives → taught steps | reasoning | **lazily, on node unlock** |
| **E** | steps → gate checks | reasoning | lazily, with D |

A–C are fast and cheap, so the map appears while the learner is still
looking at the upload screen. D–E are the expensive passes and run
just-in-time, so nobody pays to generate the 180 nodes they never reach.

**Pass C is worth a closer look.** When a textbook says *"recall from
Section 2.1"*, the author has stated a dependency edge. Mining those costs
nothing and has very high precision, which turns the hardest step in the
pipeline from *"the model guesses the whole graph"* into *"the model fills
documented gaps"* — with a precision baseline to measure the guesses
against. `parse` reports how many it found.

EPUB only, on purpose: it is structured HTML, so code listings, figures and
headings survive. PDF is a layout format that has forgotten it ever had
structure.

## The model seam

Every model call is a **typed task** — stable system prompt, volatile
payload, Pydantic output model the response must validate against — behind
a provider-agnostic interface, with schema-repair retries and a
content-addressed cache.

Nothing in Phase 0 needs bring-your-own-model, but building the seam now
means BYOK later is an implementation of `Provider`, not a refactor. The
model-qualification suite that grades a connected model is this same task
set run against a fixture.

Tier routing (`bulk` / `reason`) is a default, not a law — `--model X`
runs every tier on one model, which is what a BYOK user with a single
endpoint will do. The fidelity judge stays on the reasoning tier
deliberately: a cheap judge produces a flattering number on the one metric
that gates the project.

## Usage

```bash
pip install -e ".[dev]"

# Deterministic — no API key needed
vectorlearn parse  book.epub --chapter 7     # spans, sections, xrefs found
vectorlearn plan   book.epub --chapter 7     # what a run would cost

export ANTHROPIC_API_KEY=sk-ant-...
vectorlearn build  book.epub --chapter 7 --eval
vectorlearn show   quicksort-partition --sources
```

`build` writes `out/course.json`, `out/source.json` and
`out/build_warnings.txt`. `--eval` appends the scorecard.

## Phase 0

Phase 0 is one chapter of one book, hand-inspected. Two measured gates and
one that cannot be automated:

| Gate | Threshold | Why |
|------|-----------|-----|
| **Fidelity** | ≥ 95% supported, **0 contradicted** | The learner never opens the book, so they cannot tell when a lesson is wrong. An assertion that is correct computer science but absent from the cited spans counts as *unsupported* — that is drift, not a pass. |
| **Coverage** | ≥ 80% of teachable words owned by some node | Without it the pipeline can silently drop a third of a chapter while telling the learner they are 60% done. Coverage is also what makes that percentage honest. |
| **Reading it yourself** | no threshold | The numbers are necessary, not sufficient. `vectorlearn show` prints a node as a learner sees it; `--sources` prints the cited passages beside it. |

`vectorlearn eval` exits non-zero when a gate fails.

**If the gates fail, that is the finding.** No amount of map, streak or
playground work rescues a course that teaches things the book never said.

## Tests

```bash
pytest          # 35 tests, no API key required
```

The suite covers the deterministic half end-to-end (parsing, cross-reference
mining, graph layering, cycle-breaking, coverage) against a synthetic EPUB
fixture, and the generation passes' plumbing — fabricated-citation
rejection, ungrounded-step rejection, step-mix warnings, the scorecard gate
— against a stub provider.

What it does **not** cover is the model's judgement. That is what Phase 0
measures by hand, and it is the only part that decides whether this works.
