# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the ONE packer: ranked evidence units in, budgeted context out.

Every arm's evidence goes through this function and nothing else, so the arms
are compared on what they retrieve, never on how it is laid out or measured.

POLICY (recorded verbatim in every run manifest, see :data:`PACKER_POLICY`):

  • SELECTION is greedy by rank — score descending, ties by the arm's own
    order. A unit that does not fit the remaining budget is SKIPPED, and the
    next, smaller one may still fit. Half a unit is never packed: half a
    passage is exactly the kind of thing a model reads confidently and wrongly.
    The one exception is the ``name_list`` kind (the vocabulary null, one unit
    by construction): it is cut at a whole-name line boundary, which keeps an
    alphabetical PREFIX of the vocabulary, instead of vanishing — otherwise N1
    would silently collapse into N0 at every small budget.
  • The budget is measured with the reader's real tokenizer (``tokens.py``) on
    the rendered text, headers and separators included, and is NEVER exceeded:
    selection uses per-piece counts, then the exact count of the final text is
    checked and the lowest-ranked unit is dropped until it fits (BPE merges
    across a boundary can differ from the sum of the pieces by a token or two).
  • RENDERING groups units into one section per kind, each under a header line
    (``Facts:``, ``Relations:``, ``Paths:``, ``Communities:``, ``Excerpts:``,
    ``Names:``); units are separated by a blank line, and so are sections.
    Unit text is emitted verbatim.
  • ``order="score"`` (default): the best section first, best unit first.
    ``order="ascending"``: PathRAG's reliability-ascending placement (Chen et
    al., arXiv:2502.14902) — the exact mirror, least reliable first and the
    best unit LAST, nearest the question. Order changes placement only; the
    SAME units are selected either way.
  • Identical ``(kind, text)`` units are packed once (the higher-ranked copy).
  • ``budget_tokens=None`` means no cap — the arm's default context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.lab.evidence import UNIT_KINDS, Evidence, EvidenceUnit
from app.lab.tokens import count_tokens

Order = Literal["score", "ascending"]
ORDERS: tuple[str, ...] = ("score", "ascending")

SECTION_HEADERS: dict[str, str] = {
    "entity": "Facts:",
    "relation": "Relations:",
    "path": "Paths:",
    "community": "Communities:",
    "prose": "Excerpts:",
    "name_list": "Names:",
}
SEPARATOR = "\n\n"

PACKER_VERSION = "1"
PACKER_POLICY: dict[str, Any] = {
    "version": PACKER_VERSION,
    "selection": "greedy by rank (score desc, arm order on ties); a unit that does not fit "
    "is skipped and a later, smaller one may still fit; never exceeds the budget",
    "partial_units": "never, except name_list: cut at a whole-name line boundary",
    "dedupe": "identical (kind, text) units packed once",
    "rendering": "one section per kind under a header line; units and sections separated "
    "by a blank line; unit text verbatim",
    "headers": dict(SECTION_HEADERS),
    "orders": {
        "score": "best section first, best unit first",
        "ascending": "PathRAG reliability-ascending: mirror of score, best unit last",
    },
}


@dataclass(frozen=True, slots=True)
class PlacedUnit:
    """A kept unit and where its text sits in :attr:`Packed.text` (``[start, end)``)."""

    unit: EvidenceUnit
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.unit.kind,
            "source_id": self.unit.source_id,
            "score": self.unit.score,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class Packed:
    """A packed context. ``units_used`` counts kept units; ``by_kind`` splits it.

    ``truncated`` is True when any unit was skipped or cut for the budget.
    ``estimated`` is True when the token counts came from the len/4 fallback.
    ``placed`` lists the kept units in rendered order with their text spans.
    """

    text: str
    units_used: int
    tokens: int
    truncated: bool
    by_kind: dict[str, int]
    budget: int | None = None
    estimated: bool = False
    skipped: int = 0
    placed: tuple[PlacedUnit, ...] = field(default=())

    def unit_texts(self) -> list[tuple[EvidenceUnit, str]]:
        """``(unit, the text actually packed for it)`` in rendered order."""
        return [(p.unit, self.text[p.start : p.end]) for p in self.placed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "units_used": self.units_used,
            "tokens": self.tokens,
            "truncated": self.truncated,
            "by_kind": dict(self.by_kind),
            "budget": self.budget,
            "estimated": self.estimated,
            "skipped": self.skipped,
            "units": [p.to_dict() for p in self.placed],
        }


class _Counter:
    """Token counts memoised per string, remembering whether any was estimated."""

    def __init__(self, model: str, cache: dict[str, tuple[int, bool]] | None) -> None:
        self.model = model
        self.cache = cache if cache is not None else {}
        self.estimated = False

    def __call__(self, text: str) -> int:
        hit = self.cache.get(text)
        if hit is None:
            hit = count_tokens(text, self.model)
            self.cache[text] = hit
        if hit[1]:
            self.estimated = True
        return hit[0]


def _ranked(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Non-empty, deduplicated units, best first (stable on the arm's order)."""
    seen: set[tuple[str, str]] = set()
    kept: list[tuple[int, EvidenceUnit]] = []
    for index, unit in enumerate(units):
        if not (unit.text or "").strip():
            continue
        key = (unit.kind, unit.text)
        if key in seen:
            continue
        seen.add(key)
        kept.append((index, unit))
    kept.sort(key=lambda pair: (-float(pair[1].score), pair[0]))
    return [unit for _, unit in kept]


def _render(selected: list[EvidenceUnit], order: str) -> tuple[str, tuple[PlacedUnit, ...]]:
    """Lay the selected units (given best first) out as text, with spans."""
    if not selected:
        return "", ()
    rank = {id(unit): i for i, unit in enumerate(selected)}
    sections: dict[str, list[EvidenceUnit]] = {}
    for unit in selected:
        sections.setdefault(unit.kind, []).append(unit)
    kinds = sorted(sections, key=lambda k: (rank[id(sections[k][0])], UNIT_KINDS.index(k)))
    if order == "ascending":
        kinds.reverse()
        for kind in kinds:
            sections[kind].reverse()

    parts: list[str] = []
    placed: list[PlacedUnit] = []
    cursor = 0
    for s, kind in enumerate(kinds):
        if s:
            parts.append(SEPARATOR)
            cursor += len(SEPARATOR)
        header = SECTION_HEADERS[kind] + "\n"
        parts.append(header)
        cursor += len(header)
        for u, unit in enumerate(sections[kind]):
            if u:
                parts.append(SEPARATOR)
                cursor += len(SEPARATOR)
            parts.append(unit.text)
            placed.append(PlacedUnit(unit, cursor, cursor + len(unit.text)))
            cursor += len(unit.text)
    return "".join(parts), tuple(placed)


def _fit_name_list(unit: EvidenceUnit, room: int, count: _Counter) -> EvidenceUnit | None:
    """The longest whole-line prefix of a name list that fits ``room`` tokens."""
    if room <= 0:
        return None
    lines = [line for line in unit.text.split("\n") if line.strip()]
    lo, hi, best = 1, len(lines), 0
    while lo <= hi:  # largest m with tokens(prefix(m)) <= room
        mid = (lo + hi) // 2
        if count("\n".join(lines[:mid])) <= room:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if best == 0:
        return None
    return EvidenceUnit(
        text="\n".join(lines[:best]), kind=unit.kind, source_id=unit.source_id, score=unit.score
    )


def pack(
    evidence: Evidence | list[EvidenceUnit],
    budget_tokens: int | None,
    model: str,
    order: str = "score",
    *,
    token_cache: dict[str, tuple[int, bool]] | None = None,
) -> Packed:
    """Pack ranked evidence into at most ``budget_tokens`` tokens of ``model``.

    ``token_cache`` may be shared across calls packing the same evidence at
    several budgets, so each unit is tokenised once per run, not once per budget.
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r} (expected one of {', '.join(ORDERS)})")
    units = evidence.units if isinstance(evidence, Evidence) else list(evidence)
    ranked = _ranked(units)
    count = _Counter(model, token_cache)

    if budget_tokens is None:
        text, placed = _render(ranked, order)
        return _result(text, placed, count(text) if text else 0, False, None, count, 0)

    budget = max(0, int(budget_tokens))
    sep = count(SEPARATOR)
    header_cost = {k: count(h + "\n") for k, h in SECTION_HEADERS.items()}

    selected: list[EvidenceUnit] = []
    kinds_open: set[str] = set()
    used = 0
    skipped = 0
    cut = False
    for unit in ranked:
        if unit.kind in kinds_open:
            overhead = sep
        else:
            overhead = header_cost[unit.kind] + (sep if kinds_open else 0)
        cost = count(unit.text)
        if used + overhead + cost <= budget:
            selected.append(unit)
            kinds_open.add(unit.kind)
            used += overhead + cost
            continue
        if unit.kind == "name_list":
            shorter = _fit_name_list(unit, budget - used - overhead, count)
            if shorter is not None:
                selected.append(shorter)
                kinds_open.add(unit.kind)
                used += overhead + count(shorter.text)
                cut = True
                continue
        skipped += 1

    # Exact check on the rendered text: drop the lowest-ranked unit until it fits.
    text, placed = _render(selected, order)
    tokens = count(text) if text else 0
    while tokens > budget and selected:
        selected.pop()
        skipped += 1
        text, placed = _render(selected, order)
        tokens = count(text) if text else 0
    return _result(text, placed, tokens, bool(skipped) or cut, budget, count, skipped)


def _result(
    text: str,
    placed: tuple[PlacedUnit, ...],
    tokens: int,
    truncated: bool,
    budget: int | None,
    count: _Counter,
    skipped: int,
) -> Packed:
    by_kind: dict[str, int] = {}
    for p in placed:
        by_kind[p.unit.kind] = by_kind.get(p.unit.kind, 0) + 1
    return Packed(
        text=text,
        units_used=len(placed),
        tokens=tokens,
        truncated=truncated,
        by_kind=by_kind,
        budget=budget,
        estimated=count.estimated,
        skipped=skipped,
        placed=placed,
    )
