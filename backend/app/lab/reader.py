# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the reader: ONE short-answer prompt for every arm.

Every arm's packed context is read by the same model, with the same system
prompt, the same user template and the same output cap, so a difference in
EM/F1 is a difference in evidence and nothing else. The prompt asks for the
answer alone, in as few words as possible (EM/F1 punish verbosity), from the
evidence only — and, when the evidence is EMPTY, from the model's own
knowledge. That last clause is what makes N0 a closed-book baseline rather
than a column of "I don't know".

Requests are OpenAI ``chat.completions`` bodies:

  • reasoning models (``^gpt-5|^o[1-9]``) get no temperature (they reject
    one), ``reasoning_effort`` from ``settings.openai_reasoning_effort`` and
    ``max_completion_tokens = 32 + reasoning allowance`` — the cap covers the
    hidden reasoning tokens too, which is what makes the estimator's upper
    bound a real bound;
  • every other model gets ``temperature 0``, ``max_tokens 32`` and a ``seed``.

The same body goes to the realtime path (:func:`answer_realtime`, below) and
to the Batch API (``batch.py``), so the two modes read identical prompts.

NEVER call this against the real API from a test: :func:`default_client`
refuses to build a real client while pytest is running.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.lab.tokens import count_tokens
from app.services.llm_provider import is_reasoning_model, reasoning_effort_for

SYSTEM_PROMPT = (
    "You answer questions from the evidence provided, and from that evidence only. "
    "Reply with the answer alone, in as few words as possible (a name, a date, a number, "
    "or yes/no), with no explanation and no full sentence. If the evidence is empty, "
    "answer from your own knowledge."
)
USER_TEMPLATE = "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
EMPTY_EVIDENCE = "(none)"

#: Visible answer tokens. HotpotQA answers are a few words; 32 leaves room.
MAX_ANSWER_TOKENS = 32
#: Hidden reasoning tokens allowed on top of the answer for a reasoning model —
#: the same default as ``benchmarks/procedural`` (sized for effort "minimal").
#: An ASSUMPTION until a pilot calibrates it; lowering it lowers the up-front
#: bound AND the request's own cap, so the bound stays true.
REASONING_ALLOWANCE = 512
DEFAULT_SEED = 20260924

#: Per-message framing tokens of the chat format (OpenAI cookbook rule of thumb).
TOKENS_PER_MESSAGE = 3
TOKENS_PER_REPLY_PRIMING = 3

PROMPT_VERSION = hashlib.sha256(
    f"{SYSTEM_PROMPT}\x1f{USER_TEMPLATE}\x1f{EMPTY_EVIDENCE}".encode()
).hexdigest()[:12]


def render_user(question: str, packed_text: str) -> str:
    evidence = (packed_text or "").strip() or EMPTY_EVIDENCE
    return USER_TEMPLATE.format(evidence=evidence, question=(question or "").strip())


def max_output_tokens(model: str, reasoning_allowance: int | None = None) -> int:
    """The request's output cap: 32, plus the reasoning allowance for a reasoning model."""
    if not is_reasoning_model(model):
        return MAX_ANSWER_TOKENS
    allowance = REASONING_ALLOWANCE if reasoning_allowance is None else reasoning_allowance
    return MAX_ANSWER_TOKENS + max(0, int(allowance))


def build_request(
    question: str,
    packed_text: str,
    model: str,
    *,
    seed: int = DEFAULT_SEED,
    reasoning_allowance: int | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """The ``chat.completions`` request body for one (question, context)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user(question, packed_text)},
        ],
    }
    if is_reasoning_model(model):
        effort = reasoning_effort or get_settings().openai_reasoning_effort
        body["reasoning_effort"] = reasoning_effort_for(model, effort)
        body["max_completion_tokens"] = max_output_tokens(model, reasoning_allowance)
    else:
        body["temperature"] = 0
        body["max_tokens"] = MAX_ANSWER_TOKENS
        body["seed"] = int(seed)
    return body


def request_prompt_tokens(body: dict, model: str | None = None) -> tuple[int, bool]:
    """``(prompt tokens, estimated)`` of a request body, framing included."""
    model = model or str(body.get("model") or "")
    total = TOKENS_PER_REPLY_PRIMING
    estimated = False
    for message in body.get("messages") or []:
        n, est = count_tokens(str(message.get("content") or ""), model)
        total += n + TOKENS_PER_MESSAGE
        estimated = estimated or est
    return total, estimated


def request_output_cap(body: dict) -> int:
    """The completion-token cap a body carries (what the bill can reach at most)."""
    for key in ("max_completion_tokens", "max_tokens"):
        if body.get(key) is not None:
            return int(body[key])
    return MAX_ANSWER_TOKENS


def request_hash(body: dict) -> str:
    """Content hash of a request body — identical bodies are read once."""
    import json

    blob = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Responses ────────────────────────────────────────────────────────────────
@dataclass
class ReaderResult:
    """One answer and the usage the provider reported for it."""

    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    #: True when the request was never sent (the spend cap refused it).
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cached_tokens": self.cached_tokens,
            },
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReaderResult:
        usage = data.get("usage") or {}
        return cls(
            answer=str(data.get("answer") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
            cached_tokens=int(usage.get("cached_tokens") or 0),
            model=str(data.get("model") or ""),
            system_fingerprint=data.get("system_fingerprint"),
            finish_reason=data.get("finish_reason"),
            error=data.get("error"),
            skipped=bool(data.get("skipped")),
        )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or key access — SDK objects and plain JSON dicts alike."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_completion(response: Any) -> ReaderResult:
    """A ``chat.completion`` (SDK object or JSON dict, e.g. a Batch output body)."""
    choices = _get(response, "choices") or []
    first = choices[0] if choices else None
    message = _get(first, "message")
    usage = _get(response, "usage")
    completion_details = _get(usage, "completion_tokens_details")
    prompt_details = _get(usage, "prompt_tokens_details")
    return ReaderResult(
        answer=str(_get(message, "content") or "").strip(),
        prompt_tokens=int(_get(usage, "prompt_tokens") or 0),
        completion_tokens=int(_get(usage, "completion_tokens") or 0),
        reasoning_tokens=int(_get(completion_details, "reasoning_tokens") or 0),
        cached_tokens=int(_get(prompt_details, "cached_tokens") or 0),
        model=str(_get(response, "model") or ""),
        system_fingerprint=_get(response, "system_fingerprint"),
        finish_reason=_get(first, "finish_reason"),
    )


# ── The metered cap ──────────────────────────────────────────────────────────
class SpendGuard:
    """Reserve a request's WORST-CASE cost before it is sent; settle the real one after.

    With ``max_concurrency`` requests in flight, each in-flight request holds
    its upper-bound reservation, so ``spent + reserved`` never passes
    ``max_usd`` — the run stops cleanly *before* the cap, never after it. A
    request that errors keeps its reservation as spent (we cannot know whether
    it was billed), which errs toward stopping early.
    """

    def __init__(
        self,
        max_usd: float | None,
        *,
        upper_usd: Callable[[dict], float | None],
        actual_usd: Callable[[ReaderResult], float | None],
        spent: float = 0.0,
    ) -> None:
        self.max_usd = max_usd
        self.upper_usd = upper_usd
        self.actual_usd = actual_usd
        self.spent = float(spent)
        self.reserved = 0.0
        self.refused = 0

    def try_reserve(self, body: dict) -> float | None:
        """The amount reserved, or ``None`` when the cap refuses this request."""
        amount = self.upper_usd(body)
        if self.max_usd is None:
            return amount or 0.0
        if amount is None:  # unpriced model: a cap cannot be honoured, so refuse
            self.refused += 1
            return None
        if self.spent + self.reserved + amount > self.max_usd:
            self.refused += 1
            return None
        self.reserved += amount
        return amount

    def settle(self, reserved: float, result: ReaderResult) -> None:
        self.reserved = max(0.0, self.reserved - reserved)
        if result.error is not None:
            self.spent += reserved
            return
        actual = self.actual_usd(result)
        self.spent += reserved if actual is None else actual


def default_client() -> Any:
    """The official ``AsyncOpenAI`` client on ``settings.openai_api_key``.

    Refuses under pytest: the key in ``.env`` is real, and a hermetic suite must
    never be one missing fake away from spending it.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError(
            "refusing to create a real OpenAI client under pytest — pass a fake client"
        )
    from openai import AsyncOpenAI

    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; the Lab reader needs it")
    return AsyncOpenAI(api_key=key)


async def answer_realtime(
    requests: Sequence[dict],
    client: Any = None,
    max_concurrency: int = 4,
    *,
    guard: SpendGuard | None = None,
    on_result: Callable[[int, ReaderResult], Awaitable[None] | None] | None = None,
) -> list[ReaderResult]:
    """Send every request body; return one :class:`ReaderResult` per body, in order.

    ``guard`` meters spend: once it refuses a request, nothing further is sent
    and every unsent request comes back ``skipped``. ``on_result(i, result)``
    is called as each request completes (the runner appends it to disk there,
    which is what makes a crashed run resumable).
    """
    if not requests:
        return []
    client = client if client is not None else default_client()
    results: list[ReaderResult | None] = [None] * len(requests)
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    stopped = False

    async def emit(index: int, result: ReaderResult) -> None:
        results[index] = result
        if on_result is not None:
            maybe = on_result(index, result)
            if asyncio.iscoroutine(maybe):
                await maybe

    async def one(index: int, body: dict) -> None:
        nonlocal stopped
        async with semaphore:
            if stopped:
                await emit(index, ReaderResult(error="spend cap reached", skipped=True))
                return
            reserved = 0.0
            if guard is not None:
                amount = guard.try_reserve(body)
                if amount is None:
                    stopped = True
                    await emit(index, ReaderResult(error="spend cap reached", skipped=True))
                    return
                reserved = amount
            try:
                response = await client.chat.completions.create(**body)
                result = parse_completion(response)
            except Exception as e:  # noqa: BLE001 - one failed call must not sink the run
                result = ReaderResult(error=f"{type(e).__name__}: {e}"[:500])
            if not result.model:
                result.model = str(body.get("model") or "")
            if guard is not None:
                guard.settle(reserved, result)
            await emit(index, result)

    await asyncio.gather(*(one(i, body) for i, body in enumerate(requests)))
    return [r if r is not None else ReaderResult(error="not run", skipped=True) for r in results]
