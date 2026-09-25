"""Local model provider, speaking Ollama's native chat API.

Ollama constrains generation to a JSON Schema via the `format` field, which is
what makes the whole typed-task design work locally: the same Pydantic model
that validates a response also shapes it on the way out.

Three things are handled here that a naive client gets wrong, and each of them
fails silently rather than loudly:

  - **Context window.** Ollama defaults `num_ctx` to a few thousand tokens and
    quietly truncates anything longer. A chapter's spans overflow that easily,
    and the symptom is a model that "ignores half the source" rather than an
    error. We size the window from the actual prompt.
  - **Schema `$ref`s.** Pydantic emits `$defs`/`$ref` for nested models. The
    grammar compiler behind `format` handles those unevenly across versions, so
    we inline them first.
  - **Schema in the prompt.** Grammar constraints guarantee well-formed JSON,
    not sensible field *contents*. Smaller models fill fields far better when
    the shape is also described in the text.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .client import MAX_REPAIRS, Task, TaskError, Tier, _Cache

DEFAULT_HOST = "http://localhost:11434"

# Generous on purpose. A CPU-only machine spends minutes on prompt evaluation
# before the first token appears, and a chapter-sized prompt at a few tokens a
# second can run past any timeout picked for a GPU.
DEFAULT_TIMEOUT = 3600.0

# Ollama unloads an idle model after five minutes by default. Between two slow
# calls that means reloading several gigabytes from disk, which shows up as a
# pause nobody can account for.
KEEP_ALIVE = "30m"

# Structure passes want near-determinism; lesson prose is allowed a little room.
TEMPERATURE: dict[Tier, float] = {"bulk": 0.0, "reason": 0.3, "teach": 0.4}

# Ollama silently truncates past num_ctx, so we ask for headroom over the
# prompt rather than trusting the model's default.
CTX_FLOOR = 8192
CTX_CEILING = 32768
CTX_HEADROOM_TOKENS = 3000


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable, or the model is not pulled."""


class OllamaTimeout(OllamaUnavailable):
    """The model did not finish within the allotted time."""


def _timeout_help(payload: dict[str, Any], timeout: float) -> str:
    ctx = (payload.get("options") or {}).get("num_ctx", "?")
    return (
        f"{payload.get('model')} did not respond within {timeout / 60:.0f} minutes "
        f"(context {ctx}).\n\n"
        "This is usually slowness, not a hang: a CPU-only machine spends several "
        "minutes evaluating a chapter-sized prompt before the first token appears.\n\n"
        "  --timeout 7200        allow two hours per call\n"
        "  --max-nodes 2         generate fewer lessons\n"
        "  --chapter <smaller>   pick a shorter chapter\n\n"
        "If `ollama ps` shows 100% CPU and you have a GPU, fixing that is worth "
        "more than any of the above."
    )


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` into a single self-contained schema."""
    defs = schema.get("$defs", {})

    def walk(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [walk(n, seen) for n in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in seen:  # recursive model - leave it open rather than loop
                return {"type": "object"}
            target = defs.get(name, {})
            merged = walk(target, seen | {name})
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**merged, **extra} if extra else merged
        return {k: walk(v, seen) for k, v in node.items() if k != "$defs"}

    return walk({k: v for k, v in schema.items() if k != "$defs"}, frozenset())


def _describe(schema: dict[str, Any]) -> str:
    """A compact field guide appended to the prompt."""
    return json.dumps(schema, indent=1, ensure_ascii=False)[:2600]


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        if exc.code == 404:
            raise OllamaUnavailable(
                f"model not found — run `ollama pull {payload.get('model')}`"
            ) from exc
        raise OllamaUnavailable(f"ollama returned {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise OllamaTimeout(_timeout_help(payload, timeout)) from exc
        raise OllamaUnavailable(
            f"cannot reach ollama at {url} — is `ollama serve` running? ({exc.reason})"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OllamaTimeout(_timeout_help(payload, timeout)) from exc


@dataclass
class CallStat:
    task: str
    model: str
    attempts: int
    ok: bool
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0


class OllamaProvider:
    """Runs the whole pipeline against locally served models."""

    name = "ollama"

    def __init__(
        self,
        *,
        models: dict[Tier, str] | None = None,
        host: str = DEFAULT_HOST,
        cache_dir: str | Path | None = ".vlcache",
        timeout: float = DEFAULT_TIMEOUT,
        record: bool = False,
        progress: bool = True,
        max_ctx: int | None = None,
    ) -> None:
        if not models or "reason" not in models:
            raise ValueError("OllamaProvider needs at least a 'reason' model")
        self.models: dict[Tier, str] = {
            "bulk": models.get("bulk") or models["reason"],
            "reason": models["reason"],
            "teach": models.get("teach") or models["reason"],
        }
        self.max_ctx = max_ctx or CTX_CEILING
        self.host = host.rstrip("/")
        self.cache = _Cache(Path(cache_dir) if cache_dir else None)
        self.timeout = timeout
        self.calls = 0
        self.cache_hits = 0
        self.stats: list[CallStat] = []
        self._record = record
        self._progress = progress

    # -- Provider protocol --------------------------------------------------

    def model_for(self, tier: Tier) -> str:
        return self.models[tier]

    def run(self, task: Task) -> BaseModel:
        model = self.model_for(task.tier)
        key = task.fingerprint(self.name, model)

        cached = self.cache.get(key, task.output_model)
        if cached is not None:
            self.cache_hits += 1
            return cached

        result = self._call_with_repair(task, model)
        self.cache.put(key, result)
        return result

    # -- internals ----------------------------------------------------------

    def _call_with_repair(self, task: Task, model: str) -> BaseModel:
        raw_schema = task.output_model.model_json_schema()
        schema = _inline_refs(raw_schema)
        user = (
            f"{task.user}\n\n"
            "Return JSON matching exactly this schema. Every required field must "
            "be present and meaningfully filled — an empty list or a placeholder "
            f"string is a failed response.\n\n{_describe(schema)}"
        )

        messages = [
            {"role": "system", "content": task.system},
            {"role": "user", "content": user},
        ]
        num_ctx = self._ctx_for(task.system + user)
        started = time.monotonic()
        last_error = ""
        stat = CallStat(task=task.name, model=model, attempts=0, ok=False, seconds=0.0)

        for attempt in range(MAX_REPAIRS + 1):
            self.calls += 1
            stat.attempts = attempt + 1
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": TEMPERATURE[task.tier],
                    "num_ctx": num_ctx,
                },
            }
            self._announce(task, model, num_ctx, attempt)
            data = _post(f"{self.host}/api/chat", payload, self.timeout)
            stat.prompt_tokens = data.get("prompt_eval_count", stat.prompt_tokens)
            stat.output_tokens += data.get("eval_count", 0)
            content = (data.get("message") or {}).get("content", "")

            try:
                parsed = task.output_model.model_validate_json(content)
                stat.ok = True
                stat.seconds = time.monotonic() - started
                self._log(stat)
                return parsed
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:900]

            if attempt < MAX_REPAIRS:
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "That response did not satisfy the schema.\n"
                            f"Validation error:\n{last_error}\n\n"
                            "Produce the same content again, corrected."
                        ),
                    },
                ]

        stat.seconds = time.monotonic() - started
        stat.error = last_error
        self._log(stat)
        raise TaskError(
            f"task {task.name!r} on {model!r} failed after {MAX_REPAIRS} repairs: {last_error}"
        )

    def _ctx_for(self, prompt: str) -> int:
        """Size the window to the prompt, within the machine's ceiling.

        The KV cache grows with the window, and on a memory-bound machine a
        window that does not fit turns into swapping — which presents as a
        model that never answers rather than one that refuses. `--max-ctx`
        exists so a larger model can be used at all on such a machine.
        """
        need = len(prompt) // 4 + CTX_HEADROOM_TOKENS
        want = max(CTX_FLOOR, min(CTX_CEILING, 1 << (need - 1).bit_length()))
        return min(want, self.max_ctx)

    def _announce(self, task: Task, model: str, num_ctx: int, attempt: int) -> None:
        """Say what is starting. On a slow machine a silent minute is
        indistinguishable from a hang, and these calls run for many."""
        if not self._progress:
            return
        what = task.meta.get("node_id") or task.meta.get("batch")
        label = f"{task.name}" + (f" [{what}]" if what is not None else "")
        retry = f" (repair {attempt})" if attempt else ""
        print(f"  · {label}{retry} — {model}, ctx {num_ctx} …",
              file=sys.stderr, flush=True)

    def _log(self, stat: CallStat) -> None:
        self.stats.append(stat)
        if self._progress and stat.seconds:
            rate = f"{stat.tokens_per_second:.1f} tok/s" if stat.output_tokens else ""
            status = "ok" if stat.ok else "FAILED"
            print(f"    {status} in {stat.seconds:.0f}s  {stat.output_tokens} tok  {rate}",
                  file=sys.stderr, flush=True)


# -- discovery ---------------------------------------------------------------

def installed_models(host: str = DEFAULT_HOST) -> list[dict[str, Any]]:
    """List models this machine has pulled, largest first.

    Used by `vectorlearn qualify --all` so a benchmark run needs no arguments:
    the machine's own capability decides what gets tested.
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise OllamaUnavailable(
            f"cannot reach ollama at {host} — is `ollama serve` running? ({exc.reason})"
        ) from exc

    models = data.get("models", [])
    for m in models:
        details = m.get("details") or {}
        m["param_size"] = details.get("parameter_size", "?")
        m["quant"] = details.get("quantization_level", "?")
        m["gib"] = round(m.get("size", 0) / 1024**3, 1)
    return sorted(models, key=lambda m: m.get("size", 0), reverse=True)
