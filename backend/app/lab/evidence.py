# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the evidence shape every arm returns.

An arm never renders a prompt and never cuts to a budget. It returns a list of
*ranked units* — one fact, one relation, one path, one source excerpt — each
carrying the score it was ranked by. The shared packer (``packer.py``) is the
only thing that turns units into context text, so every arm is budgeted,
rendered and counted by exactly the same rules. That is what makes a
same-budget comparison fair: the arms differ in WHAT they retrieve, never in
how their evidence is laid out or measured.

Unit kinds:

  • ``prose``     — verbatim corpus text (a stored ``:Chunk``). ``source_id``
                    is the chunk's document name, which on HotpotQA is the
                    paragraph title, i.e. the gold label.
  • ``entity``    — a materialised entity block (name, type, description and,
                    for the shipped path, its relationship lines).
  • ``relation``  — one ``(source)-[REL]->(target)`` triple as text.
  • ``path``      — a rendered multi-hop reasoning path.
  • ``community`` — a community summary block (global search).
  • ``name_list`` — a bare list of entity names (the vocabulary null, N1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UnitKind = Literal["prose", "entity", "relation", "path", "community", "name_list"]

#: Every legal :attr:`EvidenceUnit.kind`, in the order the packer lays sections
#: out when two sections tie.
UNIT_KINDS: tuple[str, ...] = ("entity", "relation", "path", "community", "prose", "name_list")


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """One ranked piece of evidence.

    ``score`` is the arm's own ranking signal; only its ORDER matters to the
    packer (higher = better), so scales need not agree across arms.
    ``source_id`` is provenance: the document for ``prose``, the entity name for
    ``entity``, ``None`` when a unit has no single source.
    """

    text: str
    kind: UnitKind
    source_id: str | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in UNIT_KINDS:
            raise ValueError(
                f"unknown evidence kind {self.kind!r} (expected one of {', '.join(UNIT_KINDS)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "source_id": self.source_id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceUnit:
        return cls(
            text=str(data.get("text") or ""),
            kind=data.get("kind", "prose"),
            source_id=data.get("source_id"),
            score=float(data.get("score") or 0.0),
        )


@dataclass
class Evidence:
    """What one arm retrieved for one question: ranked units + free-form meta.

    ``meta`` carries anything an arm wants on the record (seeds, the route the
    shipped path took, path counts, …). It is stored with the run, never sent
    to the reader.
    """

    units: list[EvidenceUnit]
    arm: str
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, arm: str, **meta: Any) -> Evidence:
        return cls(units=[], arm=arm, meta=dict(meta))

    def __len__(self) -> int:
        return len(self.units)

    def by_kind(self) -> dict[str, int]:
        """Unit counts per kind (kinds with no unit are omitted)."""
        counts: dict[str, int] = {}
        for unit in self.units:
            counts[unit.kind] = counts.get(unit.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "units": [u.to_dict() for u in self.units],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            units=[EvidenceUnit.from_dict(u) for u in data.get("units") or []],
            arm=str(data.get("arm") or ""),
            meta=dict(data.get("meta") or {}),
        )
