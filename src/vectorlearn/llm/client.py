"""The model seam.

Every model call in this pipeline is a *typed task*: a stable system prompt,
a volatile user payload, and a Pydantic output model that the response must
validate against. Nothing here is a free-form prompt returning prose.

That constraint is what buys three things later:

  - **Bring-your-own-model.** Swapping providers is implementing `Provider`,
    not rewriting the pipeline. The qualification suite that grades a
    connected model is just this task set run against a fixture.
  - **Caching.** Keyed on (provider, model, task, inputs, schema), so
    re-running a pass after an unrelated edit costs nothing.
  - **Repair.** A schema violation is recoverable: hand the model its own
    error and ask again, bounded.

Tiering exists because the passes have genuinely different difficulty. Bulk
classification over thousands of spans is not the same problem as writing a
lesson, and paying reasoning-tier prices for it is waste.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Tier = Literal["bulk", "reason"]

MAX_REPAIRS = 2
MAX_TOKENS = 16_000


class TaskError(RuntimeError):
    """A task failed after exhausting its repair budget."""


@dataclass(frozen=True)
class Task:
    """One schema-constrained model call."""

    name: str
    system: str          # stable across calls in a pass -> cacheable prefix
    user: str            # volatile payload
    output_model: type[BaseModel]
    tier: Tier = "reason"
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def fingerprint(self, provider: str, model: str) -> str:
        schema = json.dumps(
            self.output_model.model_json_schema(), sort_keys=True, separators=(",", ":")
        )
        blob = "\x00".join([provider, model, self.name, self.system, self.user, schema])
        return hashlib.sha256(blob.encode()).hexdigest()


class Provider(Protocol):
    name: str

    def model_for(self, tier: Tier) -> str: ...

    def run(self, task: Task) -> BaseModel: ...


class _Cache:
    """Content-addressed task cache on disk."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        if root:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, model: type[T]) -> T | None:
        if not self.root:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            path.unlink(missing_ok=True)  # schema moved on; drop the stale entry
            return None

    def put(self, key: str, value: BaseModel) -> None:
        if self.root:
            (self.root / f"{key}.json").write_text(value.model_dump_json(indent=2), encoding="utf-8")


class AnthropicProvider:
    """Reference implementation of the seam.

    Tier routing is a default, not a law - override `models` to run the whole
    pipeline on one model (which is what a BYOK user with a single endpoint
    will do).
    """

    name = "anthropic"

    DEFAULT_MODELS: dict[Tier, str] = {
        # Bulk span classification: thousands of short, easy judgements.
        "bulk": "claude-haiku-4-5",
        # Node planning, edge inference, lesson and check generation, and the
        # fidelity judge. The judge stays on the reasoning tier on purpose:
        # a weak judge produces a flattering Phase 0 number, which is worse
        # than no number at all.
        "reason": "claude-opus-5",
    }

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = ".vlcache",
        models: dict[Tier, str] | None = None,
        api_key: str | None = None,
    ) -> None:
        import anthropic  # imported lazily so `parse` works with no SDK config

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.models = {**self.DEFAULT_MODELS, **(models or {})}
        self.cache = _Cache(Path(cache_dir) if cache_dir else None)
        self.calls = 0
        self.cache_hits = 0

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

    def _call_with_repair(self, task: Task, model: str) -> BaseModel:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.user}]
        last_error = ""

        for attempt in range(MAX_REPAIRS + 1):
            params: dict[str, Any] = {
                "model": model,
                "max_tokens": MAX_TOKENS,
                # The system prompt is the stable prefix; the book excerpt in
                # `user` is what varies. Caching here is what makes running a
                # pass over 200 nodes affordable.
                "system": [
                    {
                        "type": "text",
                        "text": task.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": messages,
                "output_format": task.output_model,
            }
            if task.tier == "reason":
                params["thinking"] = {"type": "adaptive"}

            self.calls += 1
            try:
                response = self.client.messages.parse(**params)
                parsed = response.parsed_output
                if parsed is None:
                    raise ValueError("model returned no parseable output")
                return parsed
            except ValidationError as exc:
                last_error = str(exc)
            except ValueError as exc:
                last_error = str(exc)

            if attempt < MAX_REPAIRS:
                messages = [
                    {"role": "user", "content": task.user},
                    {
                        "role": "assistant",
                        "content": "(previous attempt failed schema validation)",
                    },
                    {
                        "role": "user",
                        "content": (
                            "That output did not validate against the required schema.\n"
                            f"Validation error:\n{last_error}\n\n"
                            "Emit the same content again, corrected to satisfy the schema exactly."
                        ),
                    },
                ]

        raise TaskError(f"task {task.name!r} failed after {MAX_REPAIRS} repairs: {last_error}")


def estimate_tokens(text: str) -> int:
    """Rough planning estimate only.

    Deliberately crude: it exists so `build --plan` can quote a ballpark
    before spending money. Anything that needs a real number should call
    the token counting endpoint.
    """
    return max(1, len(text) // 4)


def default_provider(**kwargs: Any) -> Provider:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # Not fatal: the SDK also resolves `ant auth login` profiles.
        pass
    return AnthropicProvider(**kwargs)
