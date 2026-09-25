# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the dry-run estimator. Free: no model, no Neo4j, no network.

Before any spend, a Lab run is priced per PHASE and per (arm × budget) CELL:

  • ingest        — LLM extraction of the corpus (one call per paragraph).
                    Tokens per paragraph are the MEASURED means from the
                    HotpotQA run (434 prompt / 464 completion on gpt-4o-mini,
                    ``benchmarks/public/results_hotpotqa.md``) unless a
                    calibration file supplies better ones.
  • retrieval-LLM — calls an arm makes while retrieving (0 for every arm today).
  • reader        — one short-answer call per (arm, budget, question), with
                    identical requests counted once (N0 reads the same empty
                    context at every budget, so it is charged once).

Two numbers per figure:

  • a POINT estimate — prompts tokenised for real (``tokens.py``) on the
    rendered reader prompt, context at the budget (or the measured context
    when the free retrieve phase already ran), output at the assumed answer
    length (+ an assumed reasoning spend for a reasoning model);
  • an UPPER BOUND — output at the request's own cap (32 + the reasoning
    allowance), 0% cache hits, and a +10% tokenizer margin on every prompt.

``refuse`` is set when the upper bound exceeds ``max_usd`` — or cannot be
computed at all (a model with no price on file): a cap that cannot be checked
is not a cap. ``mode="retrieve"`` (the $0 tier) never reads and never ingests,
so it is always $0.

Budget ``None`` (an arm's default, uncapped context) has NO bound before the
retrieve phase has measured it: the estimator assumes
:data:`DEFAULT_CONTEXT_TOKENS` (point) and :data:`DEFAULT_CONTEXT_UPPER_TOKENS`
(upper) and says so. The runner re-estimates on the MEASURED contexts after
retrieval and refuses there if the real contexts break the cap — two gates,
the second one exact on input.

Prices come from ``benchmarks/public/cost.py`` (hand-recorded and dated); Batch
mode applies ``cost.BATCH_PRICE_MULTIPLIER`` (×0.5) to reader and ingest.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.lab import reader
from app.lab.tokens import tokenizer_label
from app.services.llm_provider import is_reasoning_model
from benchmarks.public import cost

MODES: tuple[str, ...] = ("realtime", "batch", "retrieve")

#: +10% on every prompt count in the upper bound (tokenizer / framing drift).
TOKENIZER_MARGIN = 0.10
#: Visible answer tokens assumed by the POINT estimate (the cap is 32).
POINT_ANSWER_TOKENS = 8
#: Hidden reasoning tokens assumed per call by the POINT estimate for a
#: reasoning model at effort "minimal" — an ASSUMPTION; calibrate from a pilot.
POINT_REASONING_TOKENS = 64
#: Context assumed for an uncapped (budget None) cell before it is measured.
DEFAULT_CONTEXT_TOKENS = 4_000
DEFAULT_CONTEXT_UPPER_TOKENS = 16_000
#: Assumed size of one retrieval-time LLM call (no arm makes one today).
RETRIEVAL_CALL_PROMPT_TOKENS = 600
RETRIEVAL_CALL_COMPLETION_TOKENS = 100

#: Measured on 500 HotpotQA paragraphs with gpt-4o-mini: 217,001 prompt +
#: 231,982 completion tokens (``benchmarks/public/results_hotpotqa.md``).
INGEST_PROMPT_TOKENS_PER_PARAGRAPH = 434.0
INGEST_COMPLETION_TOKENS_PER_PARAGRAPH = 464.0
INGEST_MODEL = "gpt-4o-mini"
INGEST_SOURCE = "measured: 500 HotpotQA paragraphs, gpt-4o-mini (results_hotpotqa.md)"
#: Extraction has no output cap, so its upper bound is the measured mean +25%.
INGEST_UPPER_COMPLETION_MARGIN = 0.25
#: Tokens per community summary call, when summaries are planned (off by default).
COMMUNITY_SUMMARY_PROMPT_TOKENS = 600
COMMUNITY_SUMMARY_COMPLETION_TOKENS = 120

#: Default calibration file (written by a measured run; optional).
CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "lab_runs" / "calibration.json"


@dataclass
class IngestPlan:
    """What ingesting the corpus would cost: one extraction call per paragraph."""

    paragraphs: int
    model: str = INGEST_MODEL
    batch: bool = True
    prompt_tokens_per_paragraph: float = INGEST_PROMPT_TOKENS_PER_PARAGRAPH
    completion_tokens_per_paragraph: float = INGEST_COMPLETION_TOKENS_PER_PARAGRAPH
    source: str = INGEST_SOURCE
    #: Community-summary calls (0 = summaries off, the Lab default).
    community_summaries: int = 0


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    """The calibration file's contents, or ``{}`` when there is none (never raises).

    Shape::

        {"ingest": {"model": "gpt-4o-mini", "prompt_tokens_per_paragraph": 430.2,
                    "completion_tokens_per_paragraph": 455.9, "measured_on": "..."},
         "reader": {"gpt-5-nano": {"answer_tokens": 6.1, "reasoning_tokens": 12.4}}}
    """
    target = path or CALIBRATION_PATH
    try:
        data = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def calibrated_ingest(plan: IngestPlan, calibration: Mapping[str, Any]) -> IngestPlan:
    """``plan`` with measured per-paragraph tokens from ``calibration`` when it has them."""
    ingest = calibration.get("ingest") if calibration else None
    if not isinstance(ingest, Mapping):
        return plan
    if ingest.get("model") and str(ingest["model"]) != plan.model:
        return plan  # a calibration for another model says nothing about this one
    try:
        prompt = float(ingest["prompt_tokens_per_paragraph"])
        completion = float(ingest["completion_tokens_per_paragraph"])
    except (KeyError, TypeError, ValueError):
        return plan
    return IngestPlan(
        paragraphs=plan.paragraphs,
        model=plan.model,
        batch=plan.batch,
        prompt_tokens_per_paragraph=prompt,
        completion_tokens_per_paragraph=completion,
        source=f"calibration file ({ingest.get('measured_on') or 'undated'})",
        community_summaries=plan.community_summaries,
    )


@dataclass
class PhaseEstimate:
    phase: str
    model: str
    batch: bool
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    upper_prompt_tokens: int = 0
    upper_completion_tokens: int = 0
    point_usd: float | None = 0.0
    upper_usd: float | None = 0.0
    price_checked_on: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CellEstimate:
    arm: str
    budget: int | None
    calls: int
    prompt_tokens: int
    completion_tokens: int
    upper_prompt_tokens: int
    upper_completion_tokens: int
    point_usd: float | None
    upper_usd: float | None
    #: Mean context tokens assumed (point) — measured when ``measured`` is True.
    context_tokens: float
    measured: bool = False
    note: str = ""


@dataclass
class Estimate:
    mode: str
    reader_model: str
    batch: bool
    n_questions: int
    phases: list[PhaseEstimate]
    cells: list[CellEstimate]
    total_point_usd: float | None
    total_upper_usd: float | None
    max_usd: float | None
    refuse: bool
    refuse_reason: str | None
    tokenizer: str
    reasoning_allowance: int
    max_output_tokens: int
    price_dates: dict[str, str]
    assumptions: list[str] = field(default_factory=list)

    def phase(self, name: str) -> PhaseEstimate | None:
        return next((p for p in self.phases if p.phase == name), None)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pricing_url"] = cost.PRICING_URL
        data["batch_multiplier"] = cost.BATCH_PRICE_MULTIPLIER
        return data


def _money(prompt: float, completion: float, model: str, batch: bool) -> float | None:
    usage = cost.Usage(
        prompt_tokens=int(math.ceil(prompt)), completion_tokens=int(math.ceil(completion))
    )
    return cost.usd(usage, cost.resolve_price(model), batch=batch)


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _prompt_overheads(questions: Sequence[str], model: str) -> list[int]:
    """Real-tokenizer prompt tokens per question with an EMPTY context, framing included."""
    return [
        reader.request_prompt_tokens(reader.build_request(q, "", model), model)[0]
        for q in questions
    ]


def _reader_point_output(model: str, calibration: Mapping[str, Any]) -> tuple[float, float, str]:
    """``(answer tokens, reasoning tokens, provenance)`` assumed per call."""
    measured = (calibration.get("reader") or {}).get(model) if calibration else None
    answer, reasoning, source = float(POINT_ANSWER_TOKENS), 0.0, "assumed"
    if is_reasoning_model(model):
        reasoning = float(POINT_REASONING_TOKENS)
    if isinstance(measured, Mapping):
        try:
            answer = float(measured.get("answer_tokens", answer))
            reasoning = float(measured.get("reasoning_tokens", reasoning))
            source = "calibration file"
        except (TypeError, ValueError):
            pass
    return answer, reasoning, source


def estimate_ingest(plan: IngestPlan) -> PhaseEstimate:
    """The ingest phase for ``plan`` (point = measured means; upper = +10% / +25%)."""
    n = max(0, int(plan.paragraphs))
    prompt = n * plan.prompt_tokens_per_paragraph
    completion = n * plan.completion_tokens_per_paragraph
    upper_prompt = prompt * (1 + TOKENIZER_MARGIN)
    upper_completion = completion * (1 + INGEST_UPPER_COMPLETION_MARGIN)
    calls = n
    notes = [
        f"{n} extraction calls, one per paragraph; tokens/paragraph from {plan.source}",
        f"upper bound: prompt +{TOKENIZER_MARGIN:.0%}, completion "
        f"+{INGEST_UPPER_COMPLETION_MARGIN:.0%} (extraction has no output cap)",
    ]
    if plan.community_summaries:
        s = int(plan.community_summaries)
        calls += s
        prompt += s * COMMUNITY_SUMMARY_PROMPT_TOKENS
        completion += s * COMMUNITY_SUMMARY_COMPLETION_TOKENS
        upper_prompt += s * COMMUNITY_SUMMARY_PROMPT_TOKENS * (1 + TOKENIZER_MARGIN)
        upper_completion += s * COMMUNITY_SUMMARY_COMPLETION_TOKENS * (
            1 + INGEST_UPPER_COMPLETION_MARGIN
        )
        notes.append(f"{s} community-summary calls (assumed sizes)")
    else:
        notes.append("community summaries OFF (no Lab arm needs them)")
    return PhaseEstimate(
        phase="ingest",
        model=plan.model,
        batch=plan.batch,
        calls=calls,
        prompt_tokens=int(math.ceil(prompt)),
        completion_tokens=int(math.ceil(completion)),
        upper_prompt_tokens=int(math.ceil(upper_prompt)),
        upper_completion_tokens=int(math.ceil(upper_completion)),
        point_usd=_money(prompt, completion, plan.model, plan.batch),
        upper_usd=_money(upper_prompt, upper_completion, plan.model, plan.batch),
        price_checked_on=cost.price_checked_on(plan.model),
        notes=notes,
    )


def estimate_run(
    *,
    questions: Sequence[str],
    arms: Sequence[str],
    budgets: Sequence[int | None],
    reader_model: str,
    cells: Sequence[tuple[str, int | None]] | None = None,
    mode: str = "realtime",
    batch: bool | None = None,
    max_usd: float | None = None,
    ingest: IngestPlan | None = None,
    context_tokens: Mapping[tuple[str, int | None], Sequence[int]] | None = None,
    unique_requests: Mapping[tuple[str, int | None], int] | None = None,
    reasoning_allowance: int | None = None,
    calibration: Mapping[str, Any] | None = None,
    arm_llm_calls: Mapping[str, int] | None = None,
) -> Estimate:
    """Price a Lab run before it spends anything.

    ``questions`` are the question texts (their prompts are tokenised for
    real). ``context_tokens[(arm, budget)]`` are MEASURED per-question context
    tokens from a completed retrieve phase; without them the budget bounds the
    context. ``unique_requests[(arm, budget)]`` is the number of distinct
    requests a cell adds after de-duplication (measured by the runner); without
    it N0 is charged once and every other cell once per question.
    ``arm_llm_calls[arm]`` overrides the registry's retrieval-LLM call counts.
    ``cells`` replaces the ``arms × budgets`` grid with an explicit (arm, budget)
    list (the runner passes its grid plus any extra cells).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
    batch = (mode == "batch") if batch is None else bool(batch)
    calibration = dict(calibration or {})
    allowance = reader.REASONING_ALLOWANCE if reasoning_allowance is None else max(
        0, int(reasoning_allowance)
    )
    reasoning = is_reasoning_model(reader_model)
    out_cap = reader.max_output_tokens(reader_model, allowance)
    n = len(questions)
    assumptions: list[str] = []
    price_dates = {"table": cost.PRICES_CHECKED_ON}
    if batch:
        price_dates["batch_multiplier"] = cost.BATCH_CHECKED_ON

    if arm_llm_calls is None:
        from app.lab.arms import ARMS

        arm_llm_calls = {a: ARMS[a].retrieval_llm_calls for a in arms if a in ARMS}

    grid = list(dict.fromkeys(cells)) if cells is not None else [
        (arm, budget) for arm in arms for budget in budgets
    ]
    phases: list[PhaseEstimate] = []
    cells_out: list[CellEstimate] = []

    if mode == "retrieve":
        reader_phase = PhaseEstimate(
            phase="reader", model=reader_model, batch=batch,
            notes=["retrieve-only: no reader call, no LLM, $0"],
        )
        phases.append(reader_phase)
        for arm, budget in grid:
            cells_out.append(CellEstimate(arm, budget, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                                          note="retrieve-only"))
        return Estimate(
            mode=mode, reader_model=reader_model, batch=batch, n_questions=n, phases=phases,
            cells=cells_out, total_point_usd=0.0, total_upper_usd=0.0, max_usd=max_usd,
            refuse=False, refuse_reason=None, tokenizer=tokenizer_label(reader_model),
            reasoning_allowance=allowance if reasoning else 0, max_output_tokens=out_cap,
            price_dates=price_dates,
            assumptions=["retrieve-only mode never calls a model: the estimate is $0"],
        )

    # ── Ingest ──
    if ingest is not None and ingest.paragraphs > 0:
        plan = calibrated_ingest(ingest, calibration)
        phases.append(estimate_ingest(plan))
        price_dates[f"ingest:{plan.model}"] = cost.price_checked_on(plan.model)

    # ── Retrieval-time LLM calls ──
    r_calls = sum(int(arm_llm_calls.get(a, 0)) for a in dict.fromkeys(a for a, _ in grid)) * n
    r_phase = PhaseEstimate(phase="retrieval_llm", model=reader_model, batch=False,
                            calls=r_calls, price_checked_on=cost.price_checked_on(reader_model))
    if r_calls:
        p = r_calls * RETRIEVAL_CALL_PROMPT_TOKENS
        c = r_calls * RETRIEVAL_CALL_COMPLETION_TOKENS
        r_phase.prompt_tokens, r_phase.completion_tokens = p, c
        r_phase.upper_prompt_tokens = int(math.ceil(p * (1 + TOKENIZER_MARGIN)))
        r_phase.upper_completion_tokens = c
        r_phase.point_usd = _money(p, c, reader_model, False)
        r_phase.upper_usd = _money(r_phase.upper_prompt_tokens, c, reader_model, False)
        r_phase.notes.append("assumed call size; retrieval calls are realtime")
    else:
        r_phase.notes.append("every selected arm retrieves with 0 LLM calls")
    phases.append(r_phase)

    # ── Reader ──
    overheads = _prompt_overheads(questions, reader_model)
    overhead_total = sum(overheads)
    answer_pt, reasoning_pt, out_source = _reader_point_output(reader_model, calibration)
    point_out = answer_pt + (reasoning_pt if reasoning else 0.0)
    rd = PhaseEstimate(phase="reader", model=reader_model, batch=batch,
                       price_checked_on=cost.price_checked_on(reader_model))
    unmeasured_default = False
    n0_charged = False
    for arm, budget in grid:
        measured = context_tokens.get((arm, budget)) if context_tokens else None
        if arm == "null_closed_book":
            ctx_point, ctx_upper, is_measured = 0.0, 0.0, True
        elif measured is not None:
            vals = [float(v) for v in measured]
            ctx_point = sum(vals) / len(vals) if vals else 0.0
            ctx_upper, is_measured = ctx_point, True
        elif budget is None:
            ctx_point, ctx_upper, is_measured = (
                float(DEFAULT_CONTEXT_TOKENS), float(DEFAULT_CONTEXT_UPPER_TOKENS), False
            )
            unmeasured_default = True
        else:
            ctx_point, ctx_upper, is_measured = float(budget), float(budget), False
        if unique_requests is not None and (arm, budget) in unique_requests:
            calls = int(unique_requests[(arm, budget)])
            note = "measured after de-duplication"
        elif arm == "null_closed_book":
            calls = 0 if n0_charged else n
            note = "identical at every budget: charged once" if n0_charged else ""
            n0_charged = True
        else:
            calls, note = n, ""
        share = calls / n if n else 0.0
        prompt = (overhead_total + ctx_point * n) * share
        upper_prompt = (overhead_total + ctx_upper * n) * share * (1 + TOKENIZER_MARGIN)
        completion = point_out * calls
        upper_completion = float(out_cap * calls)
        cell = CellEstimate(
            arm=arm,
            budget=budget,
            calls=calls,
            prompt_tokens=int(math.ceil(prompt)),
            completion_tokens=int(math.ceil(completion)),
            upper_prompt_tokens=int(math.ceil(upper_prompt)),
            upper_completion_tokens=int(math.ceil(upper_completion)),
            point_usd=_money(prompt, completion, reader_model, batch),
            upper_usd=_money(upper_prompt, upper_completion, reader_model, batch),
            context_tokens=ctx_point,
            measured=is_measured,
            note=note,
        )
        cells_out.append(cell)
        rd.calls += calls
        rd.prompt_tokens += cell.prompt_tokens
        rd.completion_tokens += cell.completion_tokens
        rd.reasoning_tokens += int(math.ceil(reasoning_pt * calls)) if reasoning else 0
        rd.upper_prompt_tokens += cell.upper_prompt_tokens
        rd.upper_completion_tokens += cell.upper_completion_tokens
    rd.point_usd = _money(rd.prompt_tokens, rd.completion_tokens, reader_model, batch)
    rd.upper_usd = _money(rd.upper_prompt_tokens, rd.upper_completion_tokens, reader_model, batch)
    rd.notes.append(
        f"output: point {answer_pt:g} answer tokens"
        + (f" + {reasoning_pt:g} reasoning" if reasoning else "")
        + f" ({out_source}); upper = the request cap {out_cap} tokens"
    )
    phases.append(rd)
    price_dates[f"reader:{reader_model}"] = cost.price_checked_on(reader_model)

    if reasoning:
        assumptions.append(
            f"{reader_model} is a REASONING model: hidden reasoning is billed as output. The "
            f"upper bound allows {allowance} reasoning tokens per call (the request's own "
            f"max_completion_tokens = {out_cap}); the point estimate assumes {reasoning_pt:g} "
            f"({out_source}). Calibrate on a pilot."
        )
    if unmeasured_default:
        assumptions.append(
            f"uncapped (default-context) cells are not bounded before retrieval: assumed "
            f"{DEFAULT_CONTEXT_TOKENS:,} tokens (point) / {DEFAULT_CONTEXT_UPPER_TOKENS:,} "
            "(upper) per question; the runner re-checks on the measured contexts before reading"
        )
    if not context_tokens:
        assumptions.append("capped cells assume the context fills its budget (point = upper)")
    assumptions.append(
        f"upper bound: +{TOKENIZER_MARGIN:.0%} on every prompt, 0% cache hits, output at the cap"
    )
    assumptions.append(
        f"prices hand-recorded in benchmarks/public/cost.py, NOT fetched live — verify at "
        f"{cost.PRICING_URL}"
    )
    if batch:
        assumptions.append(f"Batch API pricing: ×{cost.BATCH_PRICE_MULTIPLIER} on input and output")

    total_point: float | None = 0.0
    total_upper: float | None = 0.0
    for ph in phases:
        total_point = _add(total_point, ph.point_usd)
        total_upper = _add(total_upper, ph.upper_usd)

    refuse, reason = False, None
    if max_usd is not None:
        if total_upper is None:
            refuse = True
            reason = (
                f"no price on file for {reader_model!r}"
                + (f" or {ingest.model!r}" if ingest else "")
                + " — the upper bound cannot be checked against the cap"
            )
        elif total_upper > max_usd:
            refuse = True
            reason = (
                f"upper bound {cost.format_usd(total_upper)} exceeds the cap "
                f"{cost.format_usd(max_usd)}"
            )
    return Estimate(
        mode=mode,
        reader_model=reader_model,
        batch=batch,
        n_questions=n,
        phases=phases,
        cells=cells_out,
        total_point_usd=total_point,
        total_upper_usd=total_upper,
        max_usd=max_usd,
        refuse=refuse,
        refuse_reason=reason,
        tokenizer=tokenizer_label(reader_model),
        reasoning_allowance=allowance if reasoning else 0,
        max_output_tokens=out_cap,
        price_dates=price_dates,
        assumptions=assumptions,
    )
