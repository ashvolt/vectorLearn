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
