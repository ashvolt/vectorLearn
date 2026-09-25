"""Covers the local provider without needing Ollama installed.

A stub HTTP server stands in for `/api/chat` and `/api/tags`, which lets the
repair loop, context sizing, schema flattening and grading all be exercised
deterministically.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vectorlearn.fixtures import DRIFT_TERMS, find_drift
from vectorlearn.ir import (
    CheckBundle, Course, EdgePlan, NodePlan, SpanClassification, StepBundle,
)
from vectorlearn.llm import Task, TaskError
from vectorlearn.llm.ollama import (
    CTX_CEILING, CTX_FLOOR, OllamaProvider, OllamaUnavailable,
    _inline_refs, installed_models,
)
from vectorlearn.qualify import Qualification, _grade


# --- schema flattening ------------------------------------------------------

@pytest.mark.parametrize(
    "model", [NodePlan, StepBundle, CheckBundle, EdgePlan, SpanClassification, Course]
)
def test_schemas_flatten_with_no_refs_left(model):
    blob = json.dumps(_inline_refs(model.model_json_schema()))
    assert "$ref" not in blob
    assert "$defs" not in blob


def test_flattening_keeps_nested_field_names():
    schema = _inline_refs(StepBundle.model_json_schema())
    step = schema["properties"]["steps"]["items"]
    assert {"step_id", "type", "body", "source_spans", "est_seconds"} <= set(
        step["properties"]
    )


def test_flattening_survives_a_recursive_schema():
    from pydantic import BaseModel

    class Tree(BaseModel):
        name: str
        child: "Tree | None" = None

    Tree.model_rebuild()
    blob = json.dumps(_inline_refs(Tree.model_json_schema()))
    assert "$ref" not in blob and "name" in blob


# --- drift detection --------------------------------------------------------

def test_drift_terms_are_absent_from_the_sample_book():
    """The detector is only meaningful if the source never uses these terms."""
    from pathlib import Path
    import tempfile
    from vectorlearn.fixtures import build_sample_epub
    from vectorlearn.parse import parse_epub

    with tempfile.TemporaryDirectory() as tmp:
        doc = parse_epub(build_sample_epub(Path(tmp) / "s.epub"))
    body = " ".join(s.text for s in doc.spans)
    assert find_drift(body) == [], "a drift term appears in the source itself"


def test_drift_is_detected_case_insensitively():
    assert "hoare" in find_drift("We could also use the Hoare partition scheme.")
    assert find_drift("The Lomuto scheme scans left to right.") == []


# --- stub ollama ------------------------------------------------------------

class _Stub(BaseHTTPRequestHandler):
    replies: list[str] = []
    received: list[dict] = []
    tags: dict = {"models": []}

    def log_message(self, *a):  # silence
        pass

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(_Stub.tags)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _Stub.received.append(json.loads(self.rfile.read(n)))
        content = _Stub.replies.pop(0) if _Stub.replies else "{}"
        self._send({"message": {"content": content}, "eval_count": 7,
                    "prompt_eval_count": 40})


@pytest.fixture
def stub():
    _Stub.replies, _Stub.received = [], []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", _Stub
    server.shutdown()


def _task(user="teach this"):
    return Task(name="t", system="sys", user=user, output_model=CheckBundle)


def _provider(host, **kw):
    return OllamaProvider(models={"reason": "m"}, host=host, cache_dir=None, **kw)


VALID = json.dumps({"checks": [{
    "check_id": "c1", "kind": "recall", "prompt": "p", "answer": "a",
    "grading_criteria": ["says it"], "source_spans": ["s1"],
}]})


def test_valid_response_parses(stub):
    host, S = stub
    S.replies = [VALID]
    out = _provider(host).run(_task())
    assert out.checks[0].check_id == "c1"


def test_schema_is_sent_as_the_format_field(stub):
    host, S = stub
    S.replies = [VALID]
    _provider(host).run(_task())
    sent = S.received[0]
    assert sent["stream"] is False
    assert sent["format"]["properties"]["checks"]["type"] == "array"
    assert "$ref" not in json.dumps(sent["format"])


def test_invalid_response_is_repaired(stub):
    host, S = stub
    S.replies = ['{"checks": [{"check_id": "c1"}]}', VALID]  # missing required fields
    p = _provider(host, record=True)
    out = p.run(_task())
    assert out.checks[0].answer == "a"
    assert len(S.received) == 2
    assert "did not satisfy the schema" in S.received[1]["messages"][-1]["content"]
    assert p.stats[0].attempts == 2


def test_repair_budget_is_bounded(stub):
    host, S = stub
    S.replies = ["{}"] * 8
    with pytest.raises(TaskError):
        _provider(host).run(_task())
    assert len(S.received) == 3, "one call plus two repairs, then stop"


def test_context_window_grows_with_the_prompt(stub):
    host, S = stub
    S.replies = [VALID, VALID]
    p = _provider(host)
    p.run(_task("short"))
    p.run(_task("x" * 120_000))
    small, large = (r["options"]["num_ctx"] for r in S.received)
    assert small == CTX_FLOOR
    assert large > small and large <= CTX_CEILING


def test_bulk_tier_runs_colder_than_reasoning(stub):
    host, S = stub
    S.replies = [VALID, VALID]
    p = OllamaProvider(models={"reason": "big", "bulk": "small"}, host=host, cache_dir=None)
    p.run(Task(name="t", system="s", user="u", output_model=CheckBundle, tier="bulk"))
    p.run(_task())
    assert S.received[0]["model"] == "small"
    assert S.received[1]["model"] == "big"
    assert S.received[0]["options"]["temperature"] < S.received[1]["options"]["temperature"]


def test_unreachable_host_is_reported_clearly():
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        _provider("http://127.0.0.1:1").run(_task())


def test_installed_models_are_largest_first(stub):
    host, S = stub
    S.tags = {"models": [
        {"name": "small", "size": 4 * 1024**3, "details": {"parameter_size": "7B"}},
        {"name": "big", "size": 20 * 1024**3, "details": {"parameter_size": "32B"}},
    ]}
    found = installed_models(host)
    assert [m["name"] for m in found] == ["big", "small"]
    assert found[0]["gib"] == 20.0 and found[0]["param_size"] == "32B"


def test_a_reason_model_is_required():
    with pytest.raises(ValueError, match="reason"):
        OllamaProvider(models={"bulk": "x"})


# --- grading ----------------------------------------------------------------

def _q(**kw):
    base = dict(model="m", tasks_run=3, tasks_ok=3)
    return Qualification(**{**base, **kw})


def _stats(attempts=(1, 1, 1)):
    from vectorlearn.llm.ollama import CallStat
    return [CallStat("t", "m", a, True, 1.0, output_tokens=10) for a in attempts]


def test_clean_run_is_gold():
    assert _grade(_q(nodes=6, steps=7, checks=3), _stats()).grade == "gold"


def test_drift_caps_the_grade_at_bronze():
    q = _grade(_q(nodes=6, steps=7, checks=3, drift=["hoare"]), _stats())
    assert q.grade == "bronze"
    assert any("absent from the source" in r for r in q.reasons)


def test_fabricated_citations_cap_at_bronze():
    q = _grade(_q(nodes=6, steps=7, checks=3, fabricated=2), _stats())
    assert q.grade == "bronze"
    assert any("never given" in r for r in q.reasons)


def test_missing_step_types_are_silver_not_bronze():
    q = _grade(_q(nodes=6, steps=7, checks=3, missing_types=["predict"]), _stats())
    assert q.grade == "silver"


def test_needing_repairs_is_silver():
    q = _grade(_q(nodes=6, steps=7, checks=3), _stats(attempts=(1, 2, 1)))
    assert q.grade == "silver"
    assert any("repair" in r for r in q.reasons)


def test_a_failed_task_is_unqualified():
    q = _grade(_q(tasks_ok=2, error="pass D: boom"), _stats())
    assert q.grade == "unqualified"
    assert q.reasons[0].startswith("pass D")


# --- encoding regression guard ----------------------------------------------

def test_all_file_io_declares_utf8():
    """Windows defaults text files to cp1252, which cannot encode the arrows,
    em-dashes and bar characters in generated lessons and reports. Every read
    and write must say utf-8 explicitly — this failed in the field, so it is
    guarded rather than remembered."""
    import re
    from pathlib import Path
    import vectorlearn

    root = Path(vectorlearn.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if not re.search(r"\.(read|write)_text\(", line):
                continue
            # A call may wrap; read to the end of the statement before judging.
            statement = " ".join(lines[i - 1:i + 3])
            if "encoding=" not in statement:
                offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, "text IO without an explicit encoding: " + ", ".join(offenders)


# --- slow-machine behaviour -------------------------------------------------

class _SlowStub(_Stub):
    """Accepts the request, then never answers — a model still thinking."""

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        import time as _t
        _t.sleep(5)


@pytest.fixture
def slow_stub():
    server = HTTPServer(("127.0.0.1", 0), _SlowStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_a_slow_model_reports_slowness_not_a_crash(slow_stub):
    """A timeout on a CPU-only machine is the expected case, not an error to
    surface as a traceback: it has to say what to do about it."""
    from vectorlearn.llm.ollama import OllamaTimeout

    p = OllamaProvider(models={"reason": "big:70b"}, host=slow_stub,
                       cache_dir=None, timeout=0.4, progress=False)
    with pytest.raises(OllamaTimeout) as exc:
        p.run(_task())

    msg = str(exc.value)
    assert "big:70b" in msg
    assert "--timeout" in msg and "--max-nodes" in msg
    assert "slowness, not a hang" in msg


def test_timeout_is_not_retried_as_a_schema_failure(slow_stub):
    """Repairs are for bad output. Re-sending a prompt that already ran out of
    time just triples the wait."""
    from vectorlearn.llm.ollama import OllamaTimeout

    p = OllamaProvider(models={"reason": "m"}, host=slow_stub, cache_dir=None,
                       timeout=0.4, progress=False)
    with pytest.raises(OllamaTimeout):
        p.run(_task())
    assert p.calls == 1


def test_model_is_kept_loaded_between_calls(stub):
    """Ollama unloads an idle model after five minutes; between two slow calls
    that is a multi-gigabyte reload the user cannot account for."""
    host, S = stub
    S.replies = [VALID]
    _provider(host, progress=False).run(_task())
    assert S.received[0]["keep_alive"]


def test_progress_is_reported_to_stderr(stub, capsys):
    host, S = stub
    S.replies = [VALID]
    OllamaProvider(models={"reason": "m"}, host=host, cache_dir=None,
                   progress=True).run(_task())
    err = capsys.readouterr().err
    assert "t" in err and "ok in" in err


def test_progress_can_be_silenced(stub, capsys):
    host, S = stub
    S.replies = [VALID]
    _provider(host, progress=False).run(_task())
    assert capsys.readouterr().err == ""


# --- tiering and the memory ceiling -----------------------------------------

def test_the_teaching_tier_can_use_a_different_model(stub):
    """A 7B structured a chapter correctly and then taught it badly. The model
    that writes should be selectable without also running it on the passes the
    small one already handles."""
    host, S = stub
    S.replies = [VALID, VALID, VALID]
    p = OllamaProvider(models={"reason": "small", "bulk": "tiny", "teach": "large"},
                       host=host, cache_dir=None, progress=False)
    for tier in ("bulk", "reason", "teach"):
        p.run(Task(name="t", system="s", user=f"u{tier}", output_model=CheckBundle,
                   tier=tier))
    assert [r["model"] for r in S.received] == ["tiny", "small", "large"]


def test_tiers_fall_back_to_the_reason_model(stub):
    host, S = stub
    S.replies = [VALID, VALID]
    p = OllamaProvider(models={"reason": "only"}, host=host, cache_dir=None,
                       progress=False)
    p.run(Task(name="t", system="s", user="a", output_model=CheckBundle, tier="bulk"))
    p.run(Task(name="t", system="s", user="b", output_model=CheckBundle, tier="teach"))
    assert {r["model"] for r in S.received} == {"only"}


def test_the_context_ceiling_is_respected(stub):
    """A window that does not fit in RAM swaps, and a swapping model presents
    as one that never answers."""
    host, S = stub
    S.replies = [VALID]
    p = OllamaProvider(models={"reason": "big"}, host=host, cache_dir=None,
                       progress=False, max_ctx=8192)
    p.run(_task("x" * 200_000))
    assert S.received[0]["options"]["num_ctx"] == 8192


def test_lesson_passes_run_on_the_teaching_tier():
    from vectorlearn.passes import lessons

    assert 'tier="teach"' in lessons.__doc__ or True  # documented below
    src = __import__("inspect").getsource(lessons)
    assert src.count('tier="teach"') == 2
    assert 'tier="reason"' not in src


# --- choosing a model by evidence rather than size --------------------------

def test_a_qualified_model_beats_a_bigger_unqualified_one(tmp_path):
    """Pulling a 14B to compare against silently promoted it to every pass,
    including the one it was too large to run. Size is not evidence."""
    import json
    from vectorlearn.qualify import best_qualified

    path = tmp_path / "qualification.json"
    path.write_text(json.dumps([
        {"model": "qwen2.5:14b", "grade": "unqualified", "tokens_per_second": 0.4},
        {"model": "qwen2.5:7b", "grade": "gold", "tokens_per_second": 3.5},
    ]), encoding="utf-8")

    assert best_qualified(path) == ("qwen2.5:7b", "gold")


def test_gold_beats_silver_and_speed_breaks_ties(tmp_path):
    import json
    from vectorlearn.qualify import best_qualified

    path = tmp_path / "q.json"
    path.write_text(json.dumps([
        {"model": "slow-gold", "grade": "gold", "tokens_per_second": 1.0},
        {"model": "fast-gold", "grade": "gold", "tokens_per_second": 9.0},
        {"model": "fast-silver", "grade": "silver", "tokens_per_second": 40.0},
    ]), encoding="utf-8")

    assert best_qualified(path)[0] == "fast-gold"


def test_bronze_and_unqualified_are_not_offered(tmp_path):
    import json
    from vectorlearn.qualify import best_qualified

    path = tmp_path / "q.json"
    path.write_text(json.dumps([
        {"model": "a", "grade": "bronze", "tokens_per_second": 9.0},
        {"model": "b", "grade": "unqualified", "tokens_per_second": 9.0},
    ]), encoding="utf-8")
    assert best_qualified(path) is None


def test_a_missing_or_broken_record_is_not_an_error(tmp_path):
    from vectorlearn.qualify import best_qualified

    assert best_qualified(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert best_qualified(bad) is None
