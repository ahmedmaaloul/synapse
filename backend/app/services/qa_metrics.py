# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — QA answer metrics (Exact Match / token F1)

The procedural-graph evolution loop accepts or rejects a candidate graph on a
validation *score*, and the benchmark compares systems on the same number, so
the metric must be the one the literature reports — not a look-alike. This is
the SQuAD answer normalization and the official HotpotQA scoring rules
(``hotpot_evaluate_v1.py``), reimplemented from their published definitions:

  • normalize: lowercase → drop ASCII punctuation → drop the articles
    a / an / the → collapse whitespace;
  • EM: normalized strings are identical;
  • F1: bag-of-tokens overlap between the normalized strings;
  • HotpotQA special case: if either side normalizes to ``yes``, ``no`` or
    ``noanswer`` and the two differ, F1 is 0 — "yes" must not earn partial
    credit against "yes, in 1998" style golds, and vice versa.

Two faithful edge cases worth knowing: an empty prediction scores 0 F1 (and
EM 1 only against an empty gold), and punctuation *inside* a token is deleted,
not split on (``"1,000"`` → ``"1000"``), exactly like the reference scripts.

A gold may be a list of acceptable answers; the best score over them counts.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Literal

Metric = Literal["f1", "em"]
METRICS: tuple[str, ...] = ("f1", "em")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)
_SPECIAL = frozenset({"yes", "no", "noanswer"})

Gold = str | Sequence[str]


def normalize_answer(text: str | None) -> str:
    """SQuAD/HotpotQA normalization (lower, no punctuation/articles, single spaces)."""
    lowered = (text or "").lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def _golds(gold: Gold | None) -> list[str]:
    if gold is None:
        return [""]
    if isinstance(gold, str):
        return [gold]
    return [str(g) for g in gold] or [""]


def _exact_match_one(prediction: str | None, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def _f1_one(prediction: str | None, gold: str) -> float:
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    if (pred in _SPECIAL or truth in _SPECIAL) and pred != truth:
        return 0.0
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str | None, gold: Gold | None) -> float:
    """1.0 if the normalized prediction equals a normalized gold, else 0.0."""
    return max(_exact_match_one(prediction, g) for g in _golds(gold))


def f1(prediction: str | None, gold: Gold | None) -> float:
    """Token-overlap F1 in [0, 1] (HotpotQA rules), best over the golds."""
    return max(_f1_one(prediction, g) for g in _golds(gold))


def score(prediction: str | None, gold: Gold | None, metric: str = "f1") -> float:
    """Dispatch on ``metric`` (``"f1"`` or ``"em"``); anything else is a ``ValueError``."""
    if metric == "f1":
        return f1(prediction, gold)
    if metric == "em":
        return exact_match(prediction, gold)
    raise ValueError(f"unknown metric {metric!r} (expected one of {', '.join(METRICS)})")
