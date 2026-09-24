"""Phase 0 command line. Runs entirely on local models via Ollama.

    vectorlearn models                            # what this machine has pulled
    vectorlearn qualify --all                     # which of them can drive the pipeline
    vectorlearn parse   book.epub                 # pass A only — no model at all
    vectorlearn plan    book.epub --chapter 7     # the shape of a run
    vectorlearn build   book.epub --chapter 7     # passes A–E -> out/
    vectorlearn eval    out/course.json           # the scorecard
    vectorlearn show    quicksort-partition       # read a node as a learner would
    vectorlearn schema                            # dump the IR JSON Schema

`qualify` needs no book and no network: it runs the real passes against a
sample chapter that ships with the package, so the first question — which
local model is worth a Phase 0 run — can be answered before anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evals import measure_coverage, measure_fidelity, render_report
from .llm import DEFAULT_HOST, DEFAULT_TIMEOUT
from .ir import Course, SourceDoc
from .parse import mine_xrefs, parse_epub
from .parse.xrefs import xref_stats
from .passes import PipelineOptions, build_course
from .passes.pipeline import chapter_names, plan_cost, select_chapter

OUT = Path("out")


OLLAMA_HELP = """\
Cannot reach Ollama.

    ollama serve                 # start it
    ollama pull qwen2.5:14b      # or whatever your machine runs
    vectorlearn models           # confirm what is installed

Pass A needs no model at all:

    vectorlearn parse book.epub --chapter 7
"""


class ProviderError(RuntimeError):
    pass


def _resolve_model(args) -> str:
    """Use the requested model, else the largest one this machine has."""
    from .llm import installed_models

    if getattr(args, "model", None):
        return args.model

    found = installed_models(getattr(args, "host", None) or DEFAULT_HOST)
    if not found:
        raise ProviderError("no models installed — try `ollama pull qwen2.5:14b`")
    chosen = found[0]["name"]
    print(f"[no --model given; using largest installed: {chosen} "
          f"({chosen and found[0]['param_size']}, {found[0]['gib']} GiB)]\n",
          file=sys.stderr)
    return chosen


def _provider(args):
    """Build the provider. Local by default; Anthropic only on request."""
    from .llm import AnthropicProvider, OllamaProvider, OllamaUnavailable

    cache_dir = None if getattr(args, "no_cache", False) else ".vlcache"

    if getattr(args, "provider", "ollama") == "anthropic":
        try:
            m = getattr(args, "model", None)
            return AnthropicProvider(
                cache_dir=cache_dir,
                models={"bulk": m, "reason": m} if m else None,
            )
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"anthropic: {exc}") from exc

    try:
        reason = _resolve_model(args)
        return OllamaProvider(
            models={"reason": reason, "bulk": getattr(args, "bulk_model", None) or reason},
            host=getattr(args, "host", None) or DEFAULT_HOST,
            cache_dir=cache_dir,
            timeout=getattr(args, "timeout", None) or DEFAULT_TIMEOUT,
        )
    except OllamaUnavailable as exc:
        raise ProviderError(str(exc)) from exc


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
    if sections:
        print(f"sections    {', '.join(sections[:14])}{' …' if len(sections) > 14 else ''}")

    chapters = chapter_names(doc)
    print(f"chapters    {len(chapters)}   (vectorlearn chapters {args.book})")
    if doc.recovered_docs:
        print(f"recovered   {doc.recovered_docs} document(s) had no usable markup; "
              f"structure rebuilt from {doc.toc_entries} TOC entries")

    if args.dump:
        OUT.mkdir(exist_ok=True)
        (OUT / "source.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
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
    (OUT / "course.json").write_text(course.model_dump_json(indent=2), encoding="utf-8")
    (OUT / "source.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    (OUT / "build_warnings.txt").write_text("\n".join(report.warnings), encoding="utf-8")

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
    course = Course.model_validate_json(path.read_text(encoding="utf-8"))
    src = path.parent / "source.json"
    if not src.exists():
        print(f"missing {src} — run `build` or `parse --dump` first", file=sys.stderr)
        raise SystemExit(2)
    return course, SourceDoc.model_validate_json(src.read_text(encoding="utf-8"))


def _run_eval(course: Course, doc: SourceDoc, provider, args) -> int:
    if course.scope_spans:
        wanted = set(course.scope_spans)
        scope = [s for s in doc.spans if s.span_id in wanted]
    else:  # courses built before the scope was recorded
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
        warnings = [w for w in wpath.read_text(encoding="utf-8").splitlines() if w.strip()]

    report = render_report(course, doc, coverage, fidelity, warnings)
    print(report)
    OUT.mkdir(exist_ok=True)
    (OUT / "scorecard.txt").write_text(report, encoding="utf-8")
    return 0 if "GATE PASSED" in report else 1


def cmd_eval(args) -> int:
    course, doc = _load(Path(args.course))
    provider = None if args.no_fidelity else _provider(args)
    return _run_eval(course, doc, provider, args)


def cmd_nodes(args) -> int:
    """The objectives, which is what pass B is judged on."""
    course, _ = _load(Path(args.course))
    levels = course.topological_levels()
    depth_of = {nid: i for i, layer in enumerate(levels) for nid in layer}

    print(f"{course.title} — {len(course.nodes)} nodes, {len(course.edges)} edges, "
          f"{len(levels)} levels\n")
    for node in sorted(course.nodes, key=lambda n: (depth_of.get(n.node_id, 0), n.node_id)):
        lvl = depth_of.get(node.node_id, 0)
        print(f"  L{lvl}  {node.node_id}")
        print(f"      {node.objective}")
        bits = [f"{len(node.source_spans)} spans"]
        if node.steps:
            bits.append(f"{len(node.steps)} steps")
            bits.append(f"{len(node.checks)} checks")
            bits.append(f"~{node.est_seconds // 60} min")
        else:
            bits.append("no lesson yet")
        if node.environment.kind != "none":
            bits.append(node.environment.kind)
        if node.prereqs:
            bits.append("after " + ", ".join(node.prereqs))
        print(f"      {' · '.join(bits)}\n")

    if course.edges:
        print("edges:")
        for e in course.edges:
            mark = "author" if e.origin == "xref" else f"inferred {e.confidence:.0%}"
            print(f"  {e.src} -> {e.dst}   ({mark})")
    else:
        print("no edges — every node is an entry point, which is rarely true")
    return 0


def cmd_show(args) -> int:
    course, doc = _load(Path(args.course))
    if not args.node_id:
        return cmd_nodes(args)
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


def cmd_chapters(args) -> int:
    doc = parse_epub(args.book)
    names = chapter_names(doc)
    if not names:
        print("no chapters detected — this EPUB has neither headings nor a usable TOC",
              file=sys.stderr)
        return 1

    counts: dict[str, dict[str, int]] = {}
    for s in doc.spans:
        if s.chapter:
            c = counts.setdefault(s.chapter, {})
            c[s.kind] = c.get(s.kind, 0) + 1
            c["words"] = c.get("words", 0) + s.word_count

    print(f"{'#':>3}  {'WORDS':>7} {'CODE':>5} {'FIG':>4} {'EX':>4}  TITLE")
    print(f"{'-' * 3}  {'-' * 7} {'-' * 5} {'-' * 4} {'-' * 4}  {'-' * 44}")
    for i, name in enumerate(names, 1):
        c = counts.get(name, {})
        print(f"{i:>3}  {c.get('words', 0):>7,} {c.get('code', 0):>5} "
              f"{c.get('figure', 0):>4} {c.get('exercise', 0):>4}  {name[:52]}")
    print("\nSelect one with --chapter <number> or --chapter <part of the title>.")
    return 0


def cmd_models(args) -> int:
    from .llm import OllamaUnavailable, installed_models

    try:
        found = installed_models(args.host or DEFAULT_HOST)
    except OllamaUnavailable as exc:
        raise ProviderError(str(exc)) from exc

    if not found:
        print("no models installed — try `ollama pull qwen2.5:14b`")
        return 1

    print(f"{'MODEL':<34} {'PARAMS':>8} {'QUANT':>10} {'SIZE':>8}")
    print(f"{'-' * 34} {'-' * 8} {'-' * 10} {'-' * 8}")
    for m in found:
        print(f"{m['name'][:34]:<34} {m['param_size']:>8} {m['quant']:>10} {m['gib']:>6} GiB")
    print("\nLargest first. Qualify them with:  vectorlearn qualify --all")
    return 0


def cmd_qualify(args) -> int:
    from .llm import OllamaUnavailable, installed_models
    from .qualify import qualify_model, render_qualification

    host = args.host or DEFAULT_HOST
    try:
        if args.all:
            targets = [m["name"] for m in installed_models(host)]
            if args.limit:
                targets = targets[: args.limit]
        elif args.model:
            targets = [args.model]
        else:
            targets = [_resolve_model(args)]
    except OllamaUnavailable as exc:
        raise ProviderError(str(exc)) from exc

    if not targets:
        print("no models to qualify — try `ollama pull qwen2.5:14b`", file=sys.stderr)
        return 1

    results = []
    for name in targets:
        print(f"qualifying {name} …", file=sys.stderr, flush=True)
        try:
            results.append(qualify_model(name, host=host, cache=args.cache))
        except OllamaUnavailable as exc:
            raise ProviderError(str(exc)) from exc

    order = {"gold": 0, "silver": 1, "bronze": 2, "unqualified": 3}
    results.sort(key=lambda q: (order[q.grade], -q.tokens_per_second))

    report = render_qualification(results)
    print(report)
    OUT.mkdir(exist_ok=True)
    (OUT / "qualification.txt").write_text(report, encoding="utf-8")
    return 0 if results and results[0].grade in ("gold", "silver") else 1


def cmd_schema(args) -> int:
    print(json.dumps(Course.model_json_schema(), indent=2))
    return 0


def _force_utf8_console() -> None:
    """Windows consoles still default to a legacy code page (cp1252), which
    cannot encode the arrows and bars the reports use. Reconfiguring is
    harmless everywhere else."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass  # already utf-8, or not a real stream (piped, captured)


def main(argv: list[str] | None = None) -> int:
    from .llm import OllamaTimeout, OllamaUnavailable

    _force_utf8_console()
    p = argparse.ArgumentParser(prog="vectorlearn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--chapter", help="doc id substring or section prefix, e.g. 7")
        sp.add_argument("--max-nodes", type=int, help="cap nodes taken through D/E")
        sp.add_argument("--skip-lessons", action="store_true", help="passes A–C only")

    def model_opts(sp):
        sp.add_argument("--model", help="ollama model; defaults to the largest installed")
        sp.add_argument("--bulk-model", help="cheaper model for span classification")
        sp.add_argument("--host", help=f"ollama host (default {DEFAULT_HOST})")
        sp.add_argument("--timeout", type=float, metavar="SECONDS",
                        help=f"per-call limit (default {DEFAULT_TIMEOUT:.0f}s); "
                             "raise it on a slow CPU-only machine")
        sp.add_argument("--provider", choices=["ollama", "anthropic"], default="ollama")
        sp.add_argument("--no-cache", action="store_true")

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
    model_opts(sp)
    sp.set_defaults(fn=cmd_build)

    sp = sub.add_parser("eval", help="score a built course")
    sp.add_argument("course", nargs="?", default="out/course.json")
    sp.add_argument("--max-nodes", type=int)
    sp.add_argument("--no-fidelity", action="store_true")
    model_opts(sp)
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("nodes", help="list the objectives and the graph")
    sp.add_argument("--course", default="out/course.json")
    sp.set_defaults(fn=cmd_nodes)

    sp = sub.add_parser("show", help="read one node the way a learner would")
    sp.add_argument("node_id", nargs="?", help="omit to list them")
    sp.add_argument("--course", default="out/course.json")
    sp.add_argument("--sources", action="store_true", help="print the cited spans too")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("chapters", help="list the book's chapters and their contents")
    sp.add_argument("book")
    sp.set_defaults(fn=cmd_chapters)

    sp = sub.add_parser("models", help="list models installed on this machine")
    sp.add_argument("--host")
    sp.set_defaults(fn=cmd_models)

    sp = sub.add_parser("qualify", help="benchmark local models against the pipeline")
    sp.add_argument("--model", help="one model to test")
    sp.add_argument("--all", action="store_true", help="test every installed model")
    sp.add_argument("--limit", type=int, help="with --all, test only the N largest")
    sp.add_argument("--host")
    sp.add_argument("--cache", action="store_true", help="reuse cached runs (skews timings)")
    sp.set_defaults(fn=cmd_qualify)

    sp = sub.add_parser("schema", help="dump the course JSON Schema")
    sp.set_defaults(fn=cmd_schema)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ProviderError as exc:
        print(f"{exc}\n", file=sys.stderr)
        if "ollama" in str(exc).lower() or "model" in str(exc).lower():
            print(OLLAMA_HELP, file=sys.stderr)
        return 3
    except OllamaTimeout as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 4
    except OllamaUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print(OLLAMA_HELP, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
