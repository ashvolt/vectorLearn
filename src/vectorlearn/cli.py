"""Phase 0 command line.

    vectorlearn parse  book.epub                  # pass A only — no API key needed
    vectorlearn plan   book.epub --chapter 7      # what a run would cost
    vectorlearn build  book.epub --chapter 7      # passes A–E -> out/
    vectorlearn eval   out/course.json            # the scorecard
    vectorlearn show   quicksort-partition        # read a node as a learner would
    vectorlearn schema                            # dump the IR JSON Schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evals import measure_coverage, measure_fidelity, render_report
from .ir import Course, SourceDoc
from .parse import mine_xrefs, parse_epub
from .parse.xrefs import xref_stats
from .passes import PipelineOptions, build_course
from .passes.pipeline import plan_cost, select_chapter

OUT = Path("out")


CREDENTIAL_HELP = """\
No Anthropic credentials found.

Passes A and C's cross-reference mining are deterministic and need no key:

    vectorlearn parse book.epub --chapter 7
    vectorlearn plan  book.epub --chapter 7

Passes B, D and E, and the fidelity judge, call a model. Set one of:

    export ANTHROPIC_API_KEY=sk-ant-...
    ant auth login                     # stores a profile the SDK reads
"""


class CredentialError(RuntimeError):
    pass


def _provider(args):
    from .llm import AnthropicProvider

    models = {}
    if getattr(args, "model", None):
        models = {"bulk": args.model, "reason": args.model}
    try:
        return AnthropicProvider(
            cache_dir=None if getattr(args, "no_cache", False) else ".vlcache",
            models=models or None,
        )
    except (TypeError, ValueError) as exc:  # SDK raises TypeError on unresolved auth
        raise CredentialError(str(exc)) from exc


def cmd_parse(args) -> int:
    doc = parse_epub(args.book)
    doc = doc.model_copy(update={"xrefs": mine_xrefs(doc.spans)})
    spans = select_chapter(doc, args.chapter)

    kinds: dict[str, int] = {}
    for s in spans:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1

    print(f"title       {doc.title}")
    print(f"hash        {doc.source_hash}")
    print(f"spans       {len(doc.spans)} total, {len(spans)} selected")
    print(f"words       {sum(s.word_count for s in spans):,}")
    print("kinds       " + "  ".join(f"{k}:{v}" for k, v in sorted(kinds.items())))
    print("xrefs       " + "  ".join(f"{k}:{v}" for k, v in xref_stats(doc.xrefs).items()))

    sections = sorted({s.section for s in spans if s.section}, key=_sortkey)
    print(f"sections    {', '.join(sections[:14])}{' …' if len(sections) > 14 else ''}")

    if args.dump:
        OUT.mkdir(exist_ok=True)
        (OUT / "source.json").write_text(doc.model_dump_json(indent=2))
        print(f"\nwrote {OUT / 'source.json'}")
    return 0


def _sortkey(s: str) -> tuple:
    try:
        return tuple(int(p) for p in s.split("."))
    except ValueError:
        return (9999,)


def cmd_plan(args) -> int:
    est = plan_cost(
        args.book,
        PipelineOptions(chapter=args.chapter, max_nodes=args.max_nodes,
                        skip_lessons=args.skip_lessons),
    )
    calls = est["calls_classify"] + est["calls_plan"] + est["calls_lessons"]
    print("A run over this selection would make roughly:\n")
    print(f"  spans selected          {est['spans_selected']}")
    print(f"  source tokens (est)     {est['source_tokens_est']:,}")
    print(f"  classify calls          {est['calls_classify']}   (bulk tier)")
    print(f"  structure calls         {est['calls_plan']}   (reasoning tier)")
    print(f"  lesson calls            {est['calls_lessons']}   (reasoning tier, "
          f"assuming {est['assumed_nodes']} nodes)")
    print(f"  total model calls       {calls}")
    print("\nToken estimates are crude and for planning only. Caching means a")
    print("re-run after an unrelated edit costs close to nothing.")
    return 0


def cmd_build(args) -> int:
    provider = _provider(args)
    opts = PipelineOptions(
        chapter=args.chapter,
        max_nodes=args.max_nodes,
        skip_lessons=args.skip_lessons,
        classify=not args.no_classify,
    )

    course, doc, report = build_course(args.book, provider, opts)

    OUT.mkdir(exist_ok=True)
    (OUT / "course.json").write_text(course.model_dump_json(indent=2))
    (OUT / "source.json").write_text(doc.model_dump_json(indent=2))
    (OUT / "build_warnings.txt").write_text("\n".join(report.warnings))

    print(f"nodes {len(course.nodes)}  steps {sum(len(n.steps) for n in course.nodes)}  "
          f"edges {len(course.edges)}  calls {report.provider_calls}  "
          f"cache hits {report.cache_hits}")
    print(f"wrote {OUT / 'course.json'} and {OUT / 'source.json'}")

    if report.warnings:
        print(f"\n{len(report.warnings)} build warnings -> {OUT / 'build_warnings.txt'}")

    if args.eval:
        return _run_eval(course, doc, provider, args)
    print("\nNext: vectorlearn eval out/course.json")
    return 0


def _load(path: Path) -> tuple[Course, SourceDoc]:
    course = Course.model_validate_json(path.read_text())
    src = path.parent / "source.json"
    if not src.exists():
        print(f"missing {src} — run `build` or `parse --dump` first", file=sys.stderr)
        raise SystemExit(2)
    return course, SourceDoc.model_validate_json(src.read_text())


def _run_eval(course: Course, doc: SourceDoc, provider, args) -> int:
    cited_docs = {sid.split(":")[0] for sid in course.cited_spans()}
    scope = [s for s in doc.spans if s.doc_id in cited_docs] or doc.spans

    coverage = measure_coverage(course, scope)
    fidelity = None
    if not args.no_fidelity:
        fidelity = measure_fidelity(
            provider, course, doc.span_index(), max_nodes=args.max_nodes
        )

    warnings = []
    wpath = OUT / "build_warnings.txt"
    if wpath.exists():
        warnings = [w for w in wpath.read_text().splitlines() if w.strip()]

    report = render_report(course, doc, coverage, fidelity, warnings)
    print(report)
    OUT.mkdir(exist_ok=True)
    (OUT / "scorecard.txt").write_text(report)
    return 0 if "GATE PASSED" in report else 1


def cmd_eval(args) -> int:
    course, doc = _load(Path(args.course))
    provider = None if args.no_fidelity else _provider(args)
    return _run_eval(course, doc, provider, args)


def cmd_show(args) -> int:
    course, doc = _load(Path(args.course))
    node = course.node_index().get(args.node_id)
    if not node:
        print("known nodes:", ", ".join(n.node_id for n in course.nodes), file=sys.stderr)
        return 2

    spans = doc.span_index()
    print(f"# {node.title}\n")
    print(f"objective : {node.objective}")
    print(f"zone      : {node.zone}")
    print(f"prereqs   : {', '.join(node.prereqs) or '(none — entry point)'}")
    print(f"env       : {node.environment.kind}")
    print(f"time      : ~{node.est_seconds // 60} min across {len(node.steps)} steps\n")

    for i, s in enumerate(node.steps, 1):
        print(f"{'─' * 68}\nSTEP {i}/{len(node.steps)}  [{s.type}]  {s.title}"
              f"   ({s.est_seconds}s)")
        print(f"cites: {', '.join(s.source_spans)}\n")
        print(s.body)
        if s.reveal:
            print("\nreveal:")
            for j, r in enumerate(s.reveal, 1):
                print(f"  {j}. {r}")
        if s.prompt:
            print(f"\nQ: {s.prompt}\nA: {s.answer}")
            if s.explanation:
                print(f"   why: {s.explanation}")
        if s.starter_code:
            print(f"\nstarter:\n{s.starter_code}")
        if s.test_code:
            print(f"\ntests:\n{s.test_code}")
        if args.sources:
            print("\n--- cited source ---")
            for sid in s.source_spans:
                if sid in spans:
                    print(f"[{sid}] {spans[sid].text[:400]}")

    print(f"\n{'═' * 68}\nGATE — {len(node.checks)} checks")
    for c in node.checks:
        print(f"\n[{c.kind}] {c.prompt}")
        print(f"  answer   : {c.answer[:200]}")
        print(f"  criteria : {'; '.join(c.grading_criteria)}")
    return 0


def cmd_schema(args) -> int:
    print(json.dumps(Course.model_json_schema(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vectorlearn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--chapter", help="doc id substring or section prefix, e.g. 7")
        sp.add_argument("--max-nodes", type=int, help="cap nodes taken through D/E")
        sp.add_argument("--skip-lessons", action="store_true", help="passes A–C only")

    sp = sub.add_parser("parse", help="pass A only (no model calls)")
    sp.add_argument("book")
    sp.add_argument("--chapter")
    sp.add_argument("--dump", action="store_true", help="write out/source.json")
    sp.set_defaults(fn=cmd_parse)

    sp = sub.add_parser("plan", help="estimate a run without spending on it")
    sp.add_argument("book")
    common(sp)
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("build", help="run passes A–E")
    sp.add_argument("book")
    common(sp)
    sp.add_argument("--eval", action="store_true", help="run the scorecard afterwards")
    sp.add_argument("--no-classify", action="store_true")
    sp.add_argument("--no-fidelity", action="store_true")
    sp.add_argument("--no-cache", action="store_true")
    sp.add_argument("--model", help="run every tier on one model (BYOK rehearsal)")
    sp.set_defaults(fn=cmd_build)

    sp = sub.add_parser("eval", help="score a built course")
    sp.add_argument("course", nargs="?", default="out/course.json")
    sp.add_argument("--max-nodes", type=int)
    sp.add_argument("--no-fidelity", action="store_true")
    sp.add_argument("--no-cache", action="store_true")
    sp.add_argument("--model")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("show", help="read one node the way a learner would")
    sp.add_argument("node_id")
    sp.add_argument("--course", default="out/course.json")
    sp.add_argument("--sources", action="store_true", help="print the cited spans too")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("schema", help="dump the course JSON Schema")
    sp.set_defaults(fn=cmd_schema)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except CredentialError:
        print(CREDENTIAL_HELP, file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - the CLI is the last line of defence
        cause = str(exc)
        if "authentication" in cause.lower() or "api_key" in cause:
            print(CREDENTIAL_HELP, file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
