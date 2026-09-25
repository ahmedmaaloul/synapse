# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — leaderboard metrics: quality, what it cost, and what it beat.

Pure and offline: everything here reads stored per-question rows, so a run can
be re-scored as often as you like without Neo4j or a model.

QUALITY. EM and token F1 with HotpotQA's official normalisation
(``app.services.qa_metrics``), reported in POINTS (0–100).

COST (per arm × budget cell):
  • tokens per correct = total tokens ÷ #EM-correct, and ÷ ΣF1 (F1-weighted);
  • cost-of-pass = $ ÷ #correct (Erol et al., arXiv:2504.13359), shown as
    "$ per 100 correct";
  • amortized cost-of-pass(Q) = (C_ingest / Q + C_query) ÷ accuracy, at
    Q ∈ {100, 1,000, 10,000} queries per corpus — C_query is the per-query
    reader cost, accuracy the EM rate, and C_ingest is charged only to arms
    whose index needs the LLM-extracted graph.

FLOORS. Every gain is read against the evidence floors:
  • gain above N0 = F1 − F1(closed-book);
  • gain above N2 = F1 − F1(random context at the SAME budget);
  • graph premium at B = F1(graph arm, B) − max(F1(bm25, B), F1(dense, B)).
Each is a PAIRED bootstrap over the same questions (10,000 resamples,
seeded), and is reported only when it clears the effect floor — one question,
100/n points — AND its 95% CI excludes 0. Otherwise it is "too close to call".

PARETO. Two frontiers, F1 vs $ per query and F1 vs tokens per query: the
non-dominated (arm, budget) points, cheapest first.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.qa_metrics import exact_match, f1, normalize_answer

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260924
CONFIDENCE = 0.95
AMORTIZATION_QS: tuple[int, ...] = (100, 1_000, 10_000)

N0 = "null_closed_book"
N1 = "null_vocabulary"
N2 = "null_random"
PASSAGE_BASELINES: tuple[str, ...] = ("bm25", "dense")


# ── Per question ─────────────────────────────────────────────────────────────
def answer_scores(prediction: str | None, gold: str | Sequence[str] | None) -> tuple[float, float]:
    """``(em, f1)`` in ``[0, 1]``, best over the gold aliases."""
    return exact_match(prediction, gold), f1(prediction, gold)


def contains_answer(context: str, gold: str | Sequence[str] | None) -> bool:
    """Whether a normalised gold answer occurs, word-bounded, in the normalised context.

    The containment-style retrieval measure — deliberately the family of metric
    the vocabulary null games, which is why it is always shown beside the floors.
    ``yes``/``no`` golds are never "contained" (every context holds the word).
    """
    golds = [gold] if isinstance(gold, str) else list(gold or [])
    haystack = f" {normalize_answer(context)} "
    for g in golds:
        needle = normalize_answer(g)
        if needle and needle not in {"yes", "no"} and f" {needle} " in haystack:
            return True
    return False


def effect_floor(n: int) -> float:
    """One question, in points: ``100 / n`` (0 when there are no questions).

    At n = 1 the floor is 100 points, which no difference can exceed: a single
    question is never a reportable effect (its one-sample bootstrap CI is a
    point and would always "exclude 0").
    """
    return 100.0 / n if n >= 1 else 0.0


# ── Paired bootstrap ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Comparison:
    """``a − b`` over the same questions, in points, with a paired-bootstrap CI."""

    diff: float
    ci_low: float
    ci_high: float
    n: int
    floor: float
    reportable: bool
    against: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "diff": self.diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "floor": self.floor,
            "reportable": self.reportable,
            "against": self.against,
            "verdict": verdict(self),
        }


def verdict(c: Comparison) -> str:
    if c.n == 0:
        return "no data"
    if not c.reportable:
        return "too close to call"
    return "above" if c.diff > 0 else "below"


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), ``q`` in [0, 100]."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def bootstrap_means(
    values: Sequence[float], *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED
) -> list[float]:
    """``iterations`` resampled means of ``values`` (with replacement), seeded.

    numpy when available (vectorised, in blocks to bound memory), otherwise a
    pure-Python loop. The resample indices depend only on ``(seed, n)``, so
    every comparison over the same questions uses the same resamples.
    """
    n = len(values)
    if n == 0 or iterations <= 0:
        return []
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with fastembed
        rng = random.Random(seed)
        return [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations)]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = np.empty(iterations, dtype=float)
    block = max(1, min(iterations, 2_000_000 // max(1, n)))
    for start in range(0, iterations, block):
        m = min(block, iterations - start)
        idx = rng.integers(0, n, size=(m, n))
        means[start : start + m] = data[idx].mean(axis=1)
    return means.tolist()


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
    scale: float = 100.0,
    against: str = "",
) -> Comparison:
    """Paired bootstrap CI of ``mean(a) − mean(b)``; values in [0, 1] → points via ``scale``."""
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length ({len(a)} vs {len(b)})")
    n = len(a)
    floor = effect_floor(n)
    if n == 0:
        return Comparison(0.0, 0.0, 0.0, 0, floor, False, against)
    diffs = [(x - y) * scale for x, y in zip(a, b, strict=True)]
    diff = sum(diffs) / n
    means = sorted(bootstrap_means(diffs, iterations=iterations, seed=seed))
    tail = (1.0 - confidence) / 2.0 * 100.0
    lo, hi = _percentile(means, tail), _percentile(means, 100.0 - tail)
    excludes_zero = lo > 0 or hi < 0
    return Comparison(diff, lo, hi, n, floor, excludes_zero and abs(diff) > floor, against)


# ── Cost ─────────────────────────────────────────────────────────────────────
def amortized_cost_of_pass(
    ingest_usd: float | None, query_usd: float | None, accuracy: float, q: int
) -> float | None:
    """``(C_ingest / Q + C_query) ÷ accuracy`` — $ per correct answer at Q queries."""
    if query_usd is None or accuracy <= 0 or q <= 0:
        return None
    return ((ingest_usd or 0.0) / q + query_usd) / accuracy


def pareto_frontier(
    points: Iterable[Mapping[str, Any]], *, x: str, y: str
) -> list[dict[str, Any]]:
    """Non-dominated points: minimise ``x``, maximise ``y``. Cheapest first.

    Points with a missing ``x`` or ``y`` are ignored. Exact duplicates on both
    axes are all kept (neither dominates the other).
    """
    usable = [dict(p) for p in points if p.get(x) is not None and p.get(y) is not None]
    frontier = []
    for p in usable:
        dominated = any(
            q[x] <= p[x] and q[y] >= p[y] and (q[x] < p[x] or q[y] > p[y]) for q in usable
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda p: (p[x], -p[y], str(p.get("arm")), str(p.get("budget"))))
    return frontier


# ── Cells ────────────────────────────────────────────────────────────────────
def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pts(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def cell_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    ingest_usd: float | None = None,
    qs: Sequence[int] = AMORTIZATION_QS,
    read: bool = True,
) -> dict[str, Any]:
    """Aggregate one (arm, budget) cell's per-question rows.

    ``read=False`` (retrieve-only runs) leaves out every answer/cost metric.
    ``ingest_usd=None`` means the ingest bill is unknown: amortized figures are
    then ``None`` rather than computed as if ingest had been free.
    """
    n = len(rows)
    out: dict[str, Any] = {"n": n}

    # Retrieval-level — always available.
    ctx = [float(r.get("context_tokens") or 0) for r in rows]
    out["context_tokens_mean"] = _mean(ctx)
    out["context_tokens_max"] = max(ctx) if ctx else None
    out["truncated_rate"] = _pts(_mean([1.0 if r.get("truncated") else 0.0 for r in rows]))
    kinds: dict[str, float] = {}
    for r in rows:
        for kind, count in (r.get("by_kind") or {}).items():
            kinds[kind] = kinds.get(kind, 0.0) + float(count)
    out["units_by_kind_mean"] = {k: v / n for k, v in sorted(kinds.items())} if n else {}
    out["units_mean"] = _mean([float(r.get("units_used") or 0) for r in rows])
    contained = [r["containment"] for r in rows if r.get("containment") is not None]
    out["containment"] = _pts(_mean([1.0 if c else 0.0 for c in contained]))
    for key in ("recall_permissive", "recall_strict", "both_gold_permissive", "both_gold_strict"):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = _pts(_mean(vals))
    out["retrieval_errors"] = sum(1 for r in rows if r.get("retrieval_error"))
    if not read:
        return out

    # Answer metrics are over the questions that were actually answered; a read
    # error (a failed or cap-skipped request, or a failed retrieval that was
    # therefore never read) is counted, never scored as a 0.
    answered = [r for r in rows if not r.get("read_error")]
    m = len(answered)
    ems = [float(r.get("em") or 0.0) for r in answered]
    f1s = [float(r.get("f1") or 0.0) for r in answered]
    correct = sum(ems)
    prompt = sum(int(r.get("prompt_tokens") or 0) for r in answered)
    completion = sum(int(r.get("completion_tokens") or 0) for r in answered)
    reasoning = sum(int(r.get("reasoning_tokens") or 0) for r in answered)
    cached = sum(int(r.get("cached_tokens") or 0) for r in answered)
    total = prompt + completion
    usd_values = [r.get("usd") for r in answered]
    usd = None if any(v is None for v in usd_values) else sum(float(v) for v in usd_values)
    usd_q = usd / m if (usd is not None and m) else None
    accuracy = correct / m if m else 0.0

    out.update(
        {
            "answered": m,
            "read_errors": n - m,
            "em": _pts(_mean(ems)),
            "f1": _pts(_mean(f1s)),
            "correct": correct,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "cached_tokens": cached,
            "total_tokens": total,
            "tokens_per_query": total / m if m else None,
            "tokens_per_correct": total / correct if correct else None,
            "tokens_per_f1": total / sum(f1s) if sum(f1s) else None,
            "usd": usd,
            "usd_per_query": usd_q,
            "usd_per_100_correct": usd / correct * 100.0 if (usd is not None and correct) else None,
            "ingest_usd": ingest_usd,
            # Unknown ingest cost → no amortized figure (never a silent $0 ingest).
            "amortized_cost_of_pass": {
                str(q): (
                    None
                    if ingest_usd is None
                    else amortized_cost_of_pass(ingest_usd, usd_q, accuracy, q)
                )
                for q in qs
            },
        }
    )
    return out


def _budget_key(budget: Any) -> str:
    return "default" if budget is None else str(budget)


def _by_qid(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    """``qid -> value`` over the answered rows (read errors are not paired)."""
    return {str(r["qid"]): float(r.get(key) or 0.0) for r in rows if not r.get("read_error")}


def _paired(
    a_rows: Sequence[Mapping[str, Any]],
    b_rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    iterations: int,
    seed: int,
    against: str,
) -> Comparison:
    a, b = _by_qid(a_rows, key), _by_qid(b_rows, key)
    common = sorted(set(a) & set(b))
    return paired_bootstrap(
        [a[q] for q in common],
        [b[q] for q in common],
        iterations=iterations,
        seed=seed,
        against=against,
    )


def leaderboard(
    cells: Mapping[tuple[str, int | None], Sequence[Mapping[str, Any]]],
    *,
    arm_meta: Mapping[str, Mapping[str, Any]] | None = None,
    ingest_usd: float | None = None,
    read: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The whole leaderboard: one row per (arm, budget), comparisons, frontiers.

    ``cells`` maps ``(arm, budget)`` to that cell's per-question rows (keys
    ``qid``, ``em``, ``f1`` in [0, 1], token and ``usd`` fields, and the
    retrieval-level fields). ``arm_meta[arm]`` supplies ``family``,
    ``needs_graph``, ``title`` and ``is_null``. ``ingest_usd`` is charged to the
    arms that need the extracted graph.

    Rows are ranked by cost-normalised quality ($ per 100 correct, ascending;
    unpriced or zero-correct cells last) and every row carries ``is_null`` so a
    UI can keep the floors visually next to what they floor.
    """
    arm_meta = arm_meta or {}
    rows_out: list[dict[str, Any]] = []
    f1_mean: dict[tuple[str, int | None], float] = {}
    for (arm, budget), rows in cells.items():
        meta = arm_meta.get(arm, {})
        charged = ingest_usd if meta.get("needs_graph") else 0.0
        summary = cell_summary(rows, ingest_usd=charged, read=read)
        if read and summary.get("f1") is not None:
            f1_mean[(arm, budget)] = summary["f1"]
        rows_out.append(
            {
                "arm": arm,
                "budget": budget,
                "budget_label": _budget_key(budget),
                "family": meta.get("family"),
                "title": meta.get("title", arm),
                "is_null": bool(meta.get("is_null", arm in (N0, N1, N2))),
                **summary,
            }
        )

    n0_cells = [key for key in cells if key[0] == N0]
    if read:
        for row in rows_out:
            arm, budget = row["arm"], row["budget"]
            own = cells[(arm, budget)]
            comparisons: dict[str, Any] = {}
            if n0_cells and arm != N0:
                n0_key = (N0, budget) if (N0, budget) in cells else n0_cells[0]
                comparisons["gain_above_n0"] = _paired(
                    own, cells[n0_key], key="f1", iterations=iterations, seed=seed, against=N0
                ).to_dict()
            if (N2, budget) in cells and arm != N2:
                comparisons["gain_above_n2"] = _paired(
                    own, cells[(N2, budget)], key="f1", iterations=iterations, seed=seed,
                    against=N2,
                ).to_dict()
            if row["family"] == "graph":
                baselines = [(b, f1_mean[(b, budget)]) for b in PASSAGE_BASELINES
                             if (b, budget) in f1_mean]
                if baselines:
                    best = max(baselines, key=lambda pair: (pair[1], pair[0]))[0]
                    comparisons["graph_premium"] = _paired(
                        own, cells[(best, budget)], key="f1", iterations=iterations,
                        seed=seed, against=best,
                    ).to_dict()
            row["comparisons"] = comparisons

    def rank_key(row: dict[str, Any]) -> tuple:
        cost = row.get("usd_per_100_correct")
        return (cost is None, cost if cost is not None else 0.0, -(row.get("f1") or 0.0),
                row["arm"], row["budget_label"])

    if read:
        rows_out.sort(key=rank_key)
        for i, row in enumerate(rows_out, start=1):
            row["rank"] = i if row.get("usd_per_100_correct") is not None else None
    else:
        family_order = {"null": 0, "passage": 1, "graph": 2}
        rows_out.sort(
            key=lambda r: (
                math.inf if r["budget"] is None else r["budget"],  # default context last
                family_order.get(r.get("family") or "", 3),
                r["arm"],
            )
        )

    points = [
        {
            "arm": r["arm"],
            "budget": r["budget"],
            "f1": r.get("f1"),
            "usd_per_query": r.get("usd_per_query"),
            "tokens_per_query": r.get("tokens_per_query"),
            "is_null": r["is_null"],
        }
        for r in rows_out
    ]
    floors = {
        "n0_f1": next((r.get("f1") for r in rows_out if r["arm"] == N0), None),
        "n2_f1_by_budget": {
            r["budget_label"]: r.get("f1") for r in rows_out if r["arm"] == N2
        },
    }
    n = max((r["n"] for r in rows_out), default=0)
    return {
        "mode": "read" if read else "retrieve",
        "n_questions": n,
        "effect_floor_points": effect_floor(n),
        "bootstrap": {"iterations": iterations, "seed": seed, "confidence": CONFIDENCE},
        "rows": rows_out,
        "floors": floors,
        "frontiers": {
            "f1_vs_usd": pareto_frontier(points, x="usd_per_query", y="f1") if read else [],
            "f1_vs_tokens": pareto_frontier(points, x="tokens_per_query", y="f1") if read else [],
        },
    }
