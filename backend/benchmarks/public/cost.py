# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — token & cost accounting for the public benchmark

The internal benchmark (``benchmarks/run_benchmark.py``) is free: it ships a
pre-computed extraction and never calls a model. The HotpotQA harness cannot be,
because the whole point is to build the graph the way a *user's* graph is built —
through the real LLM extraction path. That spends the author's own money, so
every run has to be able to say what it cost, and a run that has not happened yet
has to be able to say what it *would* cost.

This module is that accounting, and it is deliberately small and boring.

WHAT IT COUNTS
    Whatever the provider actually reported. LangChain puts token usage on the
    response — ``AIMessage.usage_metadata`` (``input_tokens`` /
    ``output_tokens``) on modern versions, ``response_metadata["token_usage"]``
    (``prompt_tokens`` / ``completion_tokens``) on older ones and on several
    providers. :func:`usage_from_response` reads both.

    When neither is present the tokens are **estimated** at
    ``len(text) / CHARS_PER_TOKEN`` and the resulting :class:`Usage` carries
    ``estimated_calls > 0`` for the rest of its life. Estimated and measured
    usage add up into the same ledger, but the ledger never forgets that some of
    it was estimated and every report line says so. A cost report that quietly
    mixes a measurement with a guess is worse than no cost report.

WHAT IT CANNOT DO
    Prices are a hard-coded table (:data:`PRICES_USD_PER_1M_TOKENS`), recorded on
    :data:`PRICES_CHECKED_ON`. **They are not fetched, they are not live, and
    they go stale.** Every USD figure this module produces is therefore an
    ESTIMATE, is labelled as one, and must be checked against the invoice before
    anybody quotes it. Re-verify the table at :data:`PRICING_URL` and move the
    date when you do. A model with no entry in the table is not guessed at: its
    tokens are still reported and its cost comes back as ``None``.

USAGE

    ledger = CostLedger("gpt-4o-mini", label="ingestion")   # accumulator …
    ledger.record_response(response, prompt_text=prompt)

    with CostLedger("gpt-4o-mini", label="ingestion") as ledger:   # … or CM
        chain = prompt | metered(llm, ledger)
        await chain.ainvoke(...)
    print("\\n".join(ledger.lines()))   # tokens, calls, elapsed, estimated USD

Nothing here imports LangChain at module scope — :func:`metered` imports it
lazily — so the pure accounting maths stays importable (and unit-testable)
without any provider SDK installed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

# ── Prices ───────────────────────────────────────────────────────────────────
#: USD per 1,000,000 tokens, per model.
#:
#: ⚠️  MAINTENANCE: these were recorded by hand on :data:`PRICES_CHECKED_ON` from
#: the public price list. They are NOT fetched at run time and they WILL go
#: stale — providers change prices, and a pinned model name can be re-priced
#: without being renamed. Re-verify at :data:`PRICING_URL`, update the numbers
#: *and* the date, in one edit. Everything downstream calls its output an
#: estimate precisely because of this table.
PRICES_CHECKED_ON = "2026-07-21"
PRICING_URL = "https://openai.com/api/pricing/"


@dataclass(frozen=True)
class Price:
    """Per-1M-token prices for one model. ``output`` is 0.0 for embeddings."""

    input_usd_per_1m: float
    output_usd_per_1m: float


PRICES_USD_PER_1M_TOKENS: dict[str, Price] = {
    # Chat models.
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1-nano": Price(0.10, 0.40),
    # Embedding models — priced on input only.
    "text-embedding-3-small": Price(0.02, 0.0),
    "text-embedding-3-large": Price(0.13, 0.0),
}

#: Fallback token estimator. Four characters per token is the usual English
#: rule of thumb for BPE tokenizers; it is not a tokenizer and is never claimed
#: to be one. Anything counted this way is flagged ``estimated``.
CHARS_PER_TOKEN = 4


def resolve_price(model: str, prices: dict[str, Price] | None = None) -> Price | None:
    """Price for ``model``, tolerating a dated/pinned suffix.

    Providers serve ``gpt-4o-mini`` as ``gpt-4o-mini-2024-07-18``; the table
    holds the family. Exact match first, then the longest table key that the
    model name starts with, so ``gpt-4.1-mini-x`` cannot be priced as
    ``gpt-4.1``. Returns ``None`` when nothing matches — the caller must then
    report tokens without a cost rather than invent one.
    """
    table = PRICES_USD_PER_1M_TOKENS if prices is None else prices
    name = (model or "").strip()
    if name in table:
        return table[name]
    candidates = [key for key in table if name.startswith(key)]
    if not candidates:
        return None
    return table[max(candidates, key=len)]


def tokens_from_chars(chars: int) -> int:
    """``ceil(chars / CHARS_PER_TOKEN)``, never negative.

    Rounds up so an estimate is never *below* the truth for short strings, and
    returns 0 for zero characters.
    """
    n = max(0, int(chars))
    return math.ceil(n / CHARS_PER_TOKEN) if n else 0


def estimate_tokens(text: str) -> int:
    """Rough token count of ``text`` — see :func:`tokens_from_chars`."""
    return tokens_from_chars(len(text or ""))


# ── Usage ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Usage:
    """Tokens attributable to one or more model calls.

    ``estimated_calls`` is the number of those calls whose tokens the provider
    did **not** report, so they were derived from character counts. It is
    carried, never dropped: ``measured + estimated`` is still an estimate.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    estimated_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated(self) -> bool:
        """True when any part of this usage was estimated rather than reported."""
        return self.estimated_calls > 0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
            estimated_calls=self.estimated_calls + other.estimated_calls,
        )

    def scaled(self, factor: int) -> Usage:
        """This usage repeated ``factor`` times — the dry-run's whole arithmetic."""
        n = max(0, int(factor))
        return Usage(
            prompt_tokens=self.prompt_tokens * n,
            completion_tokens=self.completion_tokens * n,
            calls=self.calls * n,
            estimated_calls=self.estimated_calls * n,
        )


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _response_text(response: object) -> str:
    """Best-effort text of a LangChain response, for the fallback estimate."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multi-part content blocks (``[{"type": "text", "text": ...}, …]``).
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def usage_from_response(response: object, *, prompt_text: str = "") -> Usage:
    """Tokens for one model call, measured if the provider said so, else estimated.

    Three sources, in order of trust:

      1. ``response.usage_metadata`` — LangChain's normalized shape
         (``input_tokens`` / ``output_tokens``).
      2. ``response.response_metadata["token_usage"]`` — the raw provider block
         (``prompt_tokens`` / ``completion_tokens``), still a *measurement*.
      3. ``len(text) / CHARS_PER_TOKEN`` over ``prompt_text`` and the response
         body — an ESTIMATE, and marked as one.

    A metadata block that reports zero for both counts is treated as absent:
    some providers attach the key and fill it with nothing, and silently
    recording 0 tokens would under-report the bill.
    """
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, dict):
        prompt = _int(metadata.get("input_tokens"))
        completion = _int(metadata.get("output_tokens"))
        if prompt or completion:
            return Usage(prompt_tokens=prompt, completion_tokens=completion, calls=1)

    raw = getattr(response, "response_metadata", None)
    if isinstance(raw, dict):
        block = raw.get("token_usage") or raw.get("usage") or {}
        if isinstance(block, dict):
            prompt = _int(block.get("prompt_tokens", block.get("input_tokens")))
            completion = _int(block.get("completion_tokens", block.get("output_tokens")))
            if prompt or completion:
                return Usage(prompt_tokens=prompt, completion_tokens=completion, calls=1)

    return Usage(
        prompt_tokens=estimate_tokens(prompt_text),
        completion_tokens=estimate_tokens(_response_text(response)),
        calls=1,
        estimated_calls=1,
    )


# ── Money ────────────────────────────────────────────────────────────────────
def usd(usage: Usage, price: Price | None) -> float | None:
    """Cost of ``usage`` at ``price``, or ``None`` when the model has no price.

    Unrounded: rounding happens once, in :func:`format_usd`, so a sum of many
    small calls is not rounded twice.
    """
    if price is None:
        return None
    return (
        usage.prompt_tokens * price.input_usd_per_1m
        + usage.completion_tokens * price.output_usd_per_1m
    ) / 1_000_000


def format_usd(amount: float | None) -> str:
    """USD to 4 decimal places — sub-cent runs are the normal case here."""
    if amount is None:
        return "n/a"
    return f"${amount:.4f}"


# ── Ledger ───────────────────────────────────────────────────────────────────
class CostLedger:
    """Accumulates :class:`Usage` for one model and prices it.

    Usable two ways, both exercised by the harness:

    * as a plain accumulator — ``ledger.record(...)`` / ``record_response(...)``
      from anywhere, for as long as you like;
    * as a context manager — ``with CostLedger(model) as ledger:`` — which
      additionally times the block, so the report can say how long the money
      took to spend.

    It never prints and never raises on an unknown model: :meth:`usd` returns
    ``None`` and :meth:`lines` says the price is missing.
    """

    def __init__(
        self,
        model: str,
        *,
        label: str = "",
        prices: dict[str, Price] | None = None,
    ) -> None:
        self.model = model
        self.label = label or model
        self.prices = PRICES_USD_PER_1M_TOKENS if prices is None else prices
        self.usage = Usage()
        self.elapsed: float = 0.0
        self._started: float | None = None

    # -- accumulation -------------------------------------------------------
    def record(self, usage: Usage) -> Usage:
        """Add ``usage`` to the ledger and return the running total."""
        self.usage = self.usage + usage
        return self.usage

    def record_response(self, response: object, prompt_text: str = "") -> Usage:
        """Meter one model response. Returns the usage attributed to *that call*."""
        one = usage_from_response(response, prompt_text=prompt_text)
        self.record(one)
        return one

    def merge(self, other: CostLedger | Usage) -> Usage:
        """Fold another ledger's (or usage's) tokens into this one.

        Only meaningful for ledgers on the same model — pricing a merged ledger
        uses *this* ledger's model — so a mismatch is rejected loudly rather than
        silently mispriced.
        """
        if isinstance(other, CostLedger):
            if other.model != self.model:
                raise ValueError(
                    f"refusing to merge a {other.model!r} ledger into a {self.model!r} "
                    "one: the total would be priced at the wrong rate"
                )
            self.elapsed += other.elapsed
            return self.record(other.usage)
        return self.record(other)

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> CostLedger:
        self._started = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._started is not None:
            self.elapsed += time.monotonic() - self._started
            self._started = None
        return False  # never swallow an exception — a failed run still cost money

    # -- reporting ----------------------------------------------------------
    @property
    def price(self) -> Price | None:
        return resolve_price(self.model, self.prices)

    def usd(self) -> float | None:
        return usd(self.usage, self.price)

    def summary(self) -> dict:
        """Everything a report needs, as plain data (so it can be asserted on)."""
        return {
            "label": self.label,
            "model": self.model,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "calls": self.usage.calls,
            "estimated_calls": self.usage.estimated_calls,
            "estimated": self.usage.estimated,
            "usd": self.usd(),
            "priced": self.price is not None,
            "elapsed_s": self.elapsed,
            "prices_checked_on": PRICES_CHECKED_ON,
        }

    def lines(self) -> list[str]:
        """Human-readable cost report. Always says the USD figure is an estimate."""
        u = self.usage
        head = (
            f"{self.label}: {u.calls} LLM call{'' if u.calls == 1 else 's'} · "
            f"{u.prompt_tokens:,} prompt + {u.completion_tokens:,} completion = "
            f"{u.total_tokens:,} tokens"
        )
        if self.elapsed:
            head += f" · {self.elapsed:,.1f}s"
        lines = [head]
        amount = self.usd()
        if amount is None:
            lines.append(
                f"  Cost: unknown — no price on file for {self.model!r}. Add it to "
                f"cost.PRICES_USD_PER_1M_TOKENS (checked {PRICES_CHECKED_ON}, "
                f"{PRICING_URL})."
            )
        else:
            price = self.price
            lines.append(
                f"  Cost: {format_usd(amount)} ESTIMATED at {self.model} "
                f"${price.input_usd_per_1m:.2f}/1M in + "
                f"${price.output_usd_per_1m:.2f}/1M out, prices hand-recorded on "
                f"{PRICES_CHECKED_ON} and NOT fetched live — verify against the "
                f"invoice before quoting ({PRICING_URL})."
            )
        if u.estimated_calls:
            lines.append(
                f"  ⚠️  {u.estimated_calls} of {u.calls} calls reported no usage "
                f"metadata; their tokens are a len/{CHARS_PER_TOKEN} character "
                "estimate, so the token counts above are themselves partly estimated."
            )
        return lines


# ── Metering a LangChain runnable ────────────────────────────────────────────
def metered(runnable: object, ledger: CostLedger) -> object:
    """Wrap a chat model so every response it returns is recorded in ``ledger``.

    Returns a ``Runnable``, so it drops into ``prompt | llm`` exactly where the
    real model went — which is what lets the harness meter
    ``graph_builder.build_knowledge_graph`` without editing a line of the
    production ingestion path.

    LangChain is imported here rather than at module scope so the accounting
    maths above stays importable with no provider SDK installed.
    """
    from langchain_core.runnables import RunnableLambda

    def _prompt_text(value: object) -> str:
        """Characters we sent — only used if the provider reports no usage."""
        messages = getattr(value, "messages", None)
        if messages is None and isinstance(value, list):
            messages = value
        if messages is None:
            return str(value)
        return "\n".join(str(getattr(m, "content", m) or "") for m in messages)

    def _invoke(value, config=None):
        response = runnable.invoke(value, config)  # type: ignore[attr-defined]
        ledger.record_response(response, _prompt_text(value))
        return response

    async def _ainvoke(value, config=None):
        response = await runnable.ainvoke(value, config)  # type: ignore[attr-defined]
        ledger.record_response(response, _prompt_text(value))
        return response

    return RunnableLambda(_invoke, afunc=_ainvoke, name="metered")


# ── Estimating a run that has not happened yet (the --dry-run path) ──────────
def estimate_call(prompt_chars: int, completion_tokens: int) -> Usage:
    """One *hypothetical* call, priced from characters. Always flagged estimated."""
    return Usage(
        prompt_tokens=tokens_from_chars(prompt_chars),
        completion_tokens=max(0, int(completion_tokens)),
        calls=1,
        estimated_calls=1,
    )


def estimate_batch(
    prompt_chars: list[int] | tuple[int, ...],
    *,
    overhead_chars: int = 0,
    completion_tokens: int = 0,
) -> Usage:
    """Estimated usage for one call per element of ``prompt_chars``.

    ``overhead_chars`` is the fixed prompt template wrapped around every item —
    the harness measures it from the real prompt rather than guessing, because on
    short documents the template is most of the input.
    """
    total = Usage()
    for chars in prompt_chars:
        total = total + estimate_call(int(chars) + max(0, int(overhead_chars)), completion_tokens)
    return total


def with_estimated_calls(usage: Usage, estimated_calls: int) -> Usage:
    """Copy of ``usage`` with its estimated-call count overridden (test helper)."""
    return replace(usage, estimated_calls=max(0, int(estimated_calls)))
