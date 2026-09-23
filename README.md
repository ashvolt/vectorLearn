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

**Runs entirely on local models.** No API key, no account, nothing leaves the
machine. Ollama is the default provider; the book never goes anywhere.

Every model call is a **typed task** — stable system prompt, volatile payload,
Pydantic output model the response must validate against — behind a
provider-agnostic interface, with schema-repair retries and a content-addressed
cache. Ollama constrains generation to a JSON Schema via its `format` field, so
the same Pydantic model that validates a response also shapes it on the way out.

Three local-specific hazards are handled, each of which fails *silently*:
`num_ctx` is sized from the actual prompt (Ollama's default truncates a chapter
without saying so), Pydantic `$ref`/`$defs` are inlined before being sent, and
the schema is also described in the prompt — grammar constraints guarantee
well-formed JSON, not sensible field contents.

Tier routing (`bulk` / `reason`) lets a small model do span classification while
a larger one writes lessons; `--model X` alone runs everything on one.

## Usage

```bash
pip install -e ".[dev]"

# 1. Which models does this machine have, and which can do the job?
vectorlearn models
vectorlearn qualify --all

# 2. Deterministic passes — no model at all
vectorlearn parse book.epub --chapter 7      # spans, sections, xrefs found
vectorlearn plan  book.epub --chapter 7      # the shape of a run

# 3. The real thing
vectorlearn build book.epub --chapter 7 --model qwen2.5:32b --eval
vectorlearn show  quicksort-partition --sources
```

Omit `--model` and the largest installed model is used. `build` writes
`out/course.json`, `out/source.json` and `out/build_warnings.txt`; `--eval`
appends the scorecard.

## Model qualification

`vectorlearn qualify` answers the first question — *which local model is worth a
Phase 0 run?* — before you have a book. It runs the real passes against a sample
chapter shipped inside the package, so it needs no network and no input.

| Grade | Meaning |
|-------|---------|
| **gold** | full pipeline — all step types, synthesis checks, nuanced grading |
| **silver** | simplified IR — fewer step types, templated checks, no boss fights |
| **bronze** | structure only — fine for parsing and graph work, not for lessons |
| **unqualified** | cannot hold the schema |

Four things are measured, and the third is the interesting one:

1. **Schema compliance** — valid IR, and on the first try. A model needing two
   repairs per call will not finish a book.
2. **Citation discipline** — does it cite only span ids it was given?
3. **Grounding** — the sample chapter teaches Lomuto partition and nothing else,
   so a lesson mentioning Hoare, median-of-three or introsort is reciting
   training data under the author's name. Deterministic, offline, no judge
   required — and it catches a response that is otherwise schema-perfect.
4. **Instruction adherence** — node counts in range, required step mix present,
   checks that can actually gate.

Throughput is recorded too: a model that passes at four tokens a second is not
a model you will use.

## Phase 0

Phase 0 is one chapter of one book, hand-inspected. Two measured gates and
one that cannot be automated:

| Gate | Threshold | Why |
|------|-----------|-----|
| **Fidelity** | ≥ 95% supported, **0 contradicted** | The learner never opens the book, so they cannot tell when a lesson is wrong. An assertion that is correct computer science but absent from the cited spans counts as *unsupported* — that is drift, not a pass. |
| **Coverage** | ≥ 80% of teachable words owned by some node | Without it the pipeline can silently drop a third of a chapter while telling the learner they are 60% done. Coverage is also what makes that percentage honest. |
| **Reading it yourself** | no threshold | The numbers are necessary, not sufficient. `vectorlearn show` prints a node as a learner sees it; `--sources` prints the cited passages beside it. |

Run `qualify` first. A bronze model will fail these gates, and you want to know
that it was the model rather than the pipeline.

`vectorlearn eval` exits non-zero when a gate fails.

**If the gates fail, that is the finding.** No amount of map, streak or
playground work rescues a course that teaches things the book never said.

## Tests

```bash
pytest          # 60 tests, no Ollama or API key required
```

The suite covers the deterministic half end-to-end (parsing, cross-reference
mining, graph layering, cycle-breaking, coverage) against a synthetic EPUB
fixture; the generation passes' plumbing — fabricated-citation rejection,
ungrounded-step rejection, step-mix warnings, the scorecard gate — against a
stub provider; and the local provider itself — schema flattening, the repair
loop, context sizing, tier routing, drift detection, grading — against a stub
HTTP server standing in for Ollama.

What it does **not** cover is the model's judgement. That is what Phase 0
measures by hand, and it is the only part that decides whether this works.
