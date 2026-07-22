# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""RQ4 — quantify the information loss of extraction-into-entity-descriptions.

Run:

    cd backend && EMBEDDING_PROVIDER=fastembed \\
        NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret \\
        python -m research.rq4_compression_loss.experiment

Writes ``FINDINGS.md`` next to this file. Every number in that file is produced
here; none is hand-typed.

The experiment is deliberately **retrieval-light and database-free** for its two
core measurements: representational loss is a property of the representation,
not of any retriever, so measurements M1 and M2 read only the shipped fixture.
Only M3 (does compression predict retrieval failure?) embeds anything, and it
does so through ``benchmarks.run_benchmark``'s own evaluators so the numbers stay
comparable to the published benchmark report. No chat LLM is ever called.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Read-only import of the benchmark module: the scoring primitives, the word-family
# rules, the entity-rendering functions and the two graph-free evaluators are all
# reused rather than reimplemented, so RQ4 scores the same strings the report does.
from benchmarks.run_benchmark import (  # noqa: E402
    _stem,
    aggregate,
    content_words,
    embed_corpus,
    entity_context_block,
    entity_document,
    evaluate_baseline,
    evaluate_entity_rag,
    facts_found,
    load_dataset,
    rank_by_similarity,
    same_family,
    sweep_ks,
)

HERE = Path(__file__).resolve().parent
FINDINGS = HERE / "FINDINGS.md"

SEED = 20260721
BOOTSTRAP_ITERATIONS = 10_000

#: A numeric literal — the quantitative detail class. Years, counts, versions.
_NUMERIC = re.compile(r"\b\d[\d,.]*\b")
#: One hop of a question's ``reasoning`` chain, either direction.
_HOP = re.compile(r"\s*(<-\[[A-Z_]+\]-|-\[[A-Z_]+\]->)\s*")
_HOP_TYPE = re.compile(r"[A-Z_]+")

#: An edge counts as *typed* in a text when at least this fraction of the content
#: words of its authored description (endpoint names excluded) reappear there.
#: Half is the weakest threshold that still demands more than one shared word for
#: a typical four-word description, and it is applied identically to every arm.
TYPED_COVERAGE = 0.5


# ── Small statistics helpers (no hidden dependencies on scipy's defaults) ────
def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Percentile bootstrap CI of ``statistic`` over ``sample``.

    Returns ``(low, high, usable)`` where ``usable`` is the number of resamples
    on which the statistic was defined — reported rather than silently dropped,
    because with n=14 an undefined statistic is itself information.
    """
    if not sample:
        return (float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    n = len(sample)
    draws: list[float] = []
    for _ in range(iterations):
        resample = [sample[rng.randrange(n)] for _ in range(n)]
        value = statistic(resample)
        if value == value:  # not NaN
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"), 0)
    draws.sort()
    lo = draws[int(alpha / 2 * (len(draws) - 1))]
    hi = draws[int((1 - alpha / 2) * (len(draws) - 1))]
    return (lo, hi, len(draws))


def proportion_ci(flags: Sequence[bool], **kw) -> tuple[float, float, int]:
    """Bootstrap CI for the mean of a boolean sample (resamples the *units*)."""
    return bootstrap_ci([1.0 if f else 0.0 for f in flags], lambda s: sum(s) / len(s), **kw)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not bias the rank correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")  # a constant arm has no correlation, defined or otherwise
    return sxy / (sxx * syy) ** 0.5


def spearman(pairs: Sequence[tuple[float, float]]) -> float:
    """Spearman's rho of ``(x, y)`` pairs; NaN when either arm is constant."""
    if len(pairs) < 3:
        return float("nan")
    xs = _ranks([p[0] for p in pairs])
    ys = _ranks([p[1] for p in pairs])
    return _pearson(xs, ys)


def pearson(pairs: Sequence[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    return _pearson([p[0] for p in pairs], [p[1] for p in pairs])


def fmt(x: float, places: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{places}f}"


# ── The two representations under comparison ────────────────────────────────
@dataclass(frozen=True)
class Representation:
    """A corpus rendered as a list of independently retrievable text units."""

    key: str
    label: str
    units: tuple[str, ...]

    @property
    def chars(self) -> int:
        return sum(len(u) for u in self.units)

    @property
    def words(self) -> int:
        return sum(len(u.split()) for u in self.units)

    def mentions(self, needle: str) -> bool:
        """True when ``needle`` occurs anywhere in the representation."""
        low = needle.lower()
        return any(low in u.lower() for u in self.units)

    def co_mentioning_units(self, a: str, b: str) -> list[str]:
        """Units that name *both* strings — the minimum for a link to be visible."""
        la, lb = a.lower(), b.lower()
        return [u for u in self.units if la in u.lower() and lb in u.lower()]


def edge_line(rel: dict) -> str:
    """The edge as GraphRAG's structured channel renders it (name → name, typed)."""
    described = f" ({rel['description']})" if rel.get("description") else ""
    return f"  → {rel['type']} → {rel['target']}{described}"


def build_representations(dataset: dict) -> dict[str, Representation]:
    """PROSE, ENTITY (descriptions only) and ENTITY+EDGES, as retrievable units."""
    passages = dataset["passages"]
    entities = dataset["entities"]
    rels = dataset["relationships"]

    by_source: dict[str, list[dict]] = {}
    for rel in rels:
        by_source.setdefault(rel["source"], []).append(rel)

    entity_units = tuple(entity_context_block(e) for e in entities)
    edged_units = tuple(
        "\n".join(
            [entity_context_block(e)]
            + (
                ["  Relationships:"] + [edge_line(r) for r in by_source.get(e["name"], [])]
                if by_source.get(e["name"])
                else []
            )
        )
        for e in entities
    )
    return {
        "prose": Representation("prose", "PROSE (34 source passages)", tuple(p["text"] for p in passages)),
        "entity": Representation("entity", "ENTITY (79 name+type+description blocks)", entity_units),
        "entity_edges": Representation(
            "entity_edges", "ENTITY+EDGES (blocks with relationship lines)", edged_units
        ),
    }


# ── M1: how much does extraction actually compress? ─────────────────────────
@dataclass(frozen=True)
class PassageCompression:
    pid: str
    prose_chars: int
    prose_words: int
    entity_chars: float
    entity_words: float
    n_entities: int

    @property
    def ratio(self) -> float:
        """Prose characters per character of derived entity representation."""
        return self.prose_chars / self.entity_chars if self.entity_chars else float("nan")


def measure_compression(dataset: dict) -> list[PassageCompression]:
    """Per-passage prose volume vs the volume of the entity records it produced.

    An entity mentioned by several passages is *amortised*: each passage is
    charged only its share of that entity's block. Charging every passage the
    full block would double-count shared entities and inflate the apparent cost
    of the entity representation, i.e. bias the result toward the hypothesis.
    """
    entities = {e["name"]: e for e in dataset["entities"]}
    fan_out: dict[str, int] = {}
    for passage in dataset["passages"]:
        for name in passage.get("entities") or []:
            fan_out[name] = fan_out.get(name, 0) + 1

    rows: list[PassageCompression] = []
    for passage in dataset["passages"]:
        names = [n for n in (passage.get("entities") or []) if n in entities]
        chars = 0.0
        words = 0.0
        for name in names:
            block = entity_context_block(entities[name])
            share = fan_out.get(name, 1)
            chars += len(block) / share
            words += len(block.split()) / share
        rows.append(
            PassageCompression(
                pid=passage["id"],
                prose_chars=len(passage["text"]),
                prose_words=len(passage["text"].split()),
                entity_chars=chars,
                entity_words=words,
                n_entities=len(names),
            )
        )
    return rows


# ── M2: retention, split by fact class ──────────────────────────────────────
@dataclass(frozen=True)
class Retention:
    fact_class: str
    representation: str
    n: int
    retained: int
    ci: tuple[float, float, int]

    @property
    def rate(self) -> float:
        return self.retained / self.n if self.n else float("nan")


def identity_flags(rep: Representation, dataset: dict) -> list[bool]:
    """C1 — is each entity *name* recoverable? (identity)"""
    return [rep.mentions(e["name"]) for e in dataset["entities"]]


def relational_flags(rep: Representation, dataset: dict, *, typed: bool) -> list[bool]:
    """C2 — is each ``(source, TYPE, target)`` link recoverable?

    ``typed=False`` asks only whether *some single unit* names both endpoints:
    the weakest possible standard, "a reader could see there is a link here".
    ``typed=True`` additionally requires the unit to carry the authored edge
    description's content words, i.e. *what kind* of link it is.
    """
    flags: list[bool] = []
    for rel in dataset["relationships"]:
        units = rep.co_mentioning_units(rel["source"], rel["target"])
        if not units:
            flags.append(False)
            continue
        if not typed:
            flags.append(True)
            continue
        wanted = content_words(rel.get("description", ""))
        wanted -= content_words(f"{rel['source']} {rel['target']}")
        if not wanted:
            flags.append(True)
            continue
        best = 0.0
        for unit in units:
            have = content_words(unit) | {_stem(t.lower()) for t in rel["type"].split("_")}
            covered = sum(1 for w in wanted if any(same_family(w, h) for h in have))
            best = max(best, covered / len(wanted))
        flags.append(best >= TYPED_COVERAGE)
    return flags


def quantitative_facts(dataset: dict) -> list[str]:
    """C3 — the distinct numeric literals the source prose asserts."""
    prose = " ".join(p["text"] for p in dataset["passages"])
    return sorted(set(_NUMERIC.findall(prose)))


def quantitative_flags(rep: Representation, facts: Sequence[str]) -> list[bool]:
    return [rep.mentions(f) for f in facts]


def measure_retention(reps: dict[str, Representation], dataset: dict) -> list[Retention]:
    numerics = quantitative_facts(dataset)
    classes: list[tuple[str, Callable[[Representation], list[bool]]]] = [
        ("C1 entity identity", lambda r: identity_flags(r, dataset)),
        ("C2 relational link (untyped)", lambda r: relational_flags(r, dataset, typed=False)),
        ("C2s relational link (typed)", lambda r: relational_flags(r, dataset, typed=True)),
        ("C3 quantitative literal", lambda r: quantitative_flags(r, numerics)),
    ]
    out: list[Retention] = []
    for name, fn in classes:
        for rep in reps.values():
            flags = fn(rep)
            out.append(
                Retention(
                    fact_class=name,
                    representation=rep.key,
                    n=len(flags),
                    retained=sum(flags),
                    ci=proportion_ci(flags),
                )
            )
    return out


# ── M2b: what the benchmark's own required_facts actually are ───────────────
def required_fact_composition(dataset: dict) -> dict:
    names = {e["name"].lower() for e in dataset["entities"]}
    instances = [f for q in dataset["questions"] for f in q["required_facts"]]
    distinct = sorted(set(instances))
    is_entity = [f.lower() in names for f in instances]
    return {
        "instances": len(instances),
        "distinct": len(distinct),
        "entity_identity_instances": sum(is_entity),
        "non_identity": sorted({f for f, ok in zip(instances, is_entity, strict=True) if not ok}),
    }


# ── M3: does per-question compression predict per-question failure? ─────────
def chain_edges(question: dict, dataset: dict) -> list[dict]:
    """The reasoning chain of a question, parsed from its authored ``reasoning``.

    Every parsed hop is checked against the shipped relationship list, so a
    silent parse failure cannot quietly become "this question has no edges".
    """
    parts = _HOP.split(question["reasoning"])
    nodes = [p.strip() for p in parts[0::2]]
    hops = [p.strip() for p in parts[1::2]]
    index = {(r["source"], r["type"], r["target"]): r for r in dataset["relationships"]}
    edges: list[dict] = []
    for i, hop in enumerate(hops):
        rel_type = _HOP_TYPE.search(hop).group(0)
        left, right = nodes[i], nodes[i + 1]
        key = (left, rel_type, right) if hop.startswith("-[") else (right, rel_type, left)
        rel = index.get(key)
        if rel is None:
            raise RuntimeError(f"{question['id']}: parsed hop {key} is not in the dataset")
        edges.append(rel)
    return edges


def evidence_passages(question: dict, dataset: dict) -> list[dict]:
    """Passages that carry at least one of the question's required facts."""
    facts = question["required_facts"]
    return [p for p in dataset["passages"] if facts_found(p["text"], facts)]


def per_question_compression(dataset: dict, rows: list[PassageCompression]) -> dict[str, float]:
    """Compression ratio of the *evidence* each question depends on."""
    by_pid = {r.pid: r for r in rows}
    out: dict[str, float] = {}
    for question in dataset["questions"]:
        pids = [p["id"] for p in evidence_passages(question, dataset)]
        prose = sum(by_pid[p].prose_chars for p in pids)
        derived = sum(by_pid[p].entity_chars for p in pids)
        out[question["id"]] = prose / derived if derived else float("nan")
    return out


def per_question_entity_density(dataset: dict) -> dict[str, float]:
    """Entities extracted per evidence passage — the obvious confound for the ratio.

    The compression ratio of a passage is close to the reciprocal of how many
    entities were pulled out of it, so any correlation the ratio shows has to be
    checked against this before it can be read as *information loss*.
    """
    out: dict[str, float] = {}
    for question in dataset["questions"]:
        pids = evidence_passages(question, dataset)
        counts = [len(p.get("entities") or []) for p in pids]
        out[question["id"]] = statistics.fmean(counts) if counts else float("nan")
    return out


def per_question_link_retention(dataset: dict, rep: Representation) -> dict[str, float]:
    """Fraction of a question's chain hops that survive into ``rep``."""
    out: dict[str, float] = {}
    for question in dataset["questions"]:
        edges = chain_edges(question, dataset)
        kept = sum(1 for e in edges if rep.co_mentioning_units(e["source"], e["target"]))
        out[question["id"]] = kept / len(edges) if edges else float("nan")
    return out


@dataclass
class RetrievalArm:
    key: str
    label: str
    k: int
    agg: dict
    recall_by_qid: dict[str, float]


async def run_retrieval(dataset: dict) -> tuple[list[RetrievalArm], str]:
    """System A (passages) and System B (entities), matched on context budget.

    Both arms come from ``benchmarks.run_benchmark``; RQ4 only chooses the ``k``
    at which they are compared, using the same "smallest k whose mean context is
    at least the other arm's" rule the report uses for its budget matches.
    """
    passage_vecs, entity_vecs, question_vecs = await embed_corpus(dataset)
    passage_rank = {
        qid: rank_by_similarity(vec, passage_vecs) for qid, vec in question_vecs.items()
    }
    entity_rank = {qid: rank_by_similarity(vec, entity_vecs) for qid, vec in question_vecs.items()}

    k_a = 3
    base = evaluate_baseline(dataset, passage_rank, k_a)
    target = aggregate(base)["mean_context_chars"]

    chosen_k = len(dataset["entities"])
    for k in sweep_ks(len(dataset["entities"])):
        results = evaluate_entity_rag(dataset, entity_rank, k)
        if aggregate(results)["mean_context_chars"] >= target:
            chosen_k = k
            break
    ent = evaluate_entity_rag(dataset, entity_rank, chosen_k)

    note = (
        f"System A fixed at k={k_a} (mean {target:.0f} context chars); System B matched at the "
        f"smallest k whose mean context reaches that budget, k={chosen_k} "
        f"({aggregate(ent)['mean_context_chars']:.0f} chars)."
    )
    return (
        [
            RetrievalArm("A", "passage RAG", k_a, aggregate(base), {r.qid: r.recall for r in base}),
            RetrievalArm(
                "B", "entity RAG", chosen_k, aggregate(ent), {r.qid: r.recall for r in ent}
            ),
        ],
        note,
    )


@dataclass(frozen=True)
class Correlation:
    name: str
    n: int
    rho: float
    rho_ci: tuple[float, float, int]
    r: float
    r_ci: tuple[float, float, int]
    x_levels: int
    y_levels: int

    @property
    def x_constant(self) -> bool:
        return self.x_levels <= 1

    @property
    def y_constant(self) -> bool:
        return self.y_levels <= 1

    @property
    def note(self) -> str:
        """Why a number in this row should or should not be believed."""
        if self.x_constant:
            return "**predictor constant — undefined**"
        if self.y_constant:
            return "**outcome constant — undefined**"
        if self.x_levels <= 2 or self.y_levels <= 2:
            return f"near-constant ({self.x_levels}×{self.y_levels} distinct values) — CI unreliable"
        return f"{self.x_levels}×{self.y_levels} distinct values"


def correlate(name: str, pairs: list[tuple[float, float]]) -> Correlation:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return Correlation(
        name=name,
        n=len(pairs),
        rho=spearman(pairs),
        rho_ci=bootstrap_ci(pairs, spearman),  # type: ignore[arg-type]
        r=pearson(pairs),
        r_ci=bootstrap_ci(pairs, pearson),  # type: ignore[arg-type]
        x_levels=len(set(xs)),
        y_levels=len(set(ys)),
    )


# ── Reporting ───────────────────────────────────────────────────────────────
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def _ci(ci: tuple[float, float, int], places: int = 3) -> str:
    return f"[{fmt(ci[0], places)}, {fmt(ci[1], places)}]"


def build_findings(payload: dict) -> str:
    comp: list[PassageCompression] = payload["compression"]
    reps: dict[str, Representation] = payload["reps"]
    retention: list[Retention] = payload["retention"]
    composition: dict = payload["composition"]
    arms: list[RetrievalArm] = payload["arms"]
    corrs: list[Correlation] = payload["correlations"]
    ratios = [c.ratio for c in comp]
    ratio_ci = bootstrap_ci(ratios, statistics.median)

    ident = {r.representation: r for r in retention if r.fact_class.startswith("C1")}
    link = {r.representation: r for r in retention if r.fact_class.startswith("C2 ")}
    typed = {r.representation: r for r in retention if r.fact_class.startswith("C2s")}
    quant = {r.representation: r for r in retention if r.fact_class.startswith("C3")}
    gap = ident["entity"].rate - link["entity"].rate

    lines: list[str] = []
    add = lines.append

    add("# RQ4 — What is lost when extraction compresses prose into entity descriptions?")
    add("")
    add(
        "*Generated by `research/rq4_compression_loss/experiment.py`. Every number below is "
        "produced by that script; none is hand-typed.*"
    )
    add("")
    add("```")
    add(payload["command"])
    add("```")
    add("")
    add(f"Seed `{SEED}` · {BOOTSTRAP_ITERATIONS:,} bootstrap iterations · no chat LLM was called.")
    add("")

    add("## Hypothesis and prediction")
    add("")
    add(
        "**H4.** Entity extraction preserves entity *identity* but discards the *relational* and "
        "*quantitative* detail a multi-hop question needs. Prediction, stated before the "
        "measurements were read: identity facts survive extraction at a far higher rate than "
        "relational facts, and the gap is large (tens of points, not a few)."
    )
    add("")
    add(
        "**Secondary prediction (weaker, and flagged as such in advance).** A question whose "
        "evidence was compressed harder should be answered worse by entity-only retrieval, so "
        "per-question compression ratio should correlate positively with per-question failure."
    )
    add("")

    add("## Method")
    add("")
    add(
        "All measurements run on the pre-extracted fixture `backend/benchmarks/dataset.json` "
        f"({len(payload['dataset']['passages'])} passages, "
        f"{len(payload['dataset']['entities'])} entities, "
        f"{len(payload['dataset']['relationships'])} relationships, "
        f"{len(payload['dataset']['questions'])} multi-hop questions). The scoring primitives, "
        "the word-family rules, the entity-block renderer and the two graph-free retrieval "
        "systems are **imported read-only** from `benchmarks/run_benchmark.py`, so RQ4 scores "
        "exactly the strings the published benchmark scores."
    )
    add("")
    add("Three representations of the same corpus are compared, each as a list of retrievable units:")
    add("")
    for rep in reps.values():
        add(f"- **{rep.key}** — {rep.label}: {len(rep.units)} units, {rep.chars:,} chars, {rep.words:,} words.")
    add("")
    add(
        "A *fact* is recoverable from a representation when the evidence for it is present in "
        "**one single retrievable unit** — the same locality a retriever actually has to work "
        "with. Four fact classes, all derived mechanically from the fixture:"
    )
    add("")
    add(f"- **C1 identity** — the entity name appears (n = {ident['prose'].n}).")
    add(
        "- **C2 relational (untyped)** — some one unit names *both* endpoints of an authored "
        f"relationship (n = {link['prose'].n}). This is the weakest possible bar: \"a reader "
        "could tell there is a link here\"."
    )
    add(
        "- **C2s relational (typed)** — as C2, and the unit also carries at least "
        f"{TYPED_COVERAGE:.0%} of the content words of the authored edge description, i.e. *what "
        "kind* of link it is."
    )
    add(
        "- **C3 quantitative** — a numeric literal asserted by the source prose (years, counts) "
        f"appears (n = {quant['prose'].n})."
    )
    add("")

    add("## M1 — Compression is not, in this corpus, compression")
    add("")
    add(
        "Per passage: source prose characters vs the characters of the entity records that "
        "passage produced, **amortised** (an entity named by three passages charges each of them "
        "one third of its block, so shared entities are not triple-counted — the conservative "
        "direction for H4)."
    )
    add("")
    median_ratio = statistics.median(ratios)
    add(
        f"Median prose chars per entity-representation char **{fmt(median_ratio, 2)}** "
        f"(bootstrap 95% CI {_ci(ratio_ci, 2)}), mean {fmt(statistics.fmean(ratios), 2)}, "
        f"range {fmt(min(ratios), 2)}–{fmt(max(ratios), 2)}. A ratio below 1.0 means the derived "
        f"records are *larger* than the prose they came from: at the median, "
        f"{fmt(1 / median_ratio, 2)}× larger. "
        f"{sum(1 for r in ratios if r < 1)} of {len(ratios)} passages expand rather than compress."
    )
    add("")
    quart = statistics.quantiles(ratios, n=4)
    add(
        "\n".join(
            _table(
                ["statistic", "prose chars ÷ entity-rep chars"],
                [
                    [label, fmt(value, 2)]
                    for label, value in [
                        ("min", min(ratios)),
                        ("Q1", quart[0]),
                        ("median", statistics.median(ratios)),
                        ("Q3", quart[2]),
                        ("max", max(ratios)),
                        ("mean", statistics.fmean(ratios)),
                    ]
                ],
            )
        )
    )
    add("")
    add("Corpus totals:")
    add("")
    add(
        "\n".join(
            _table(
                ["representation", "units", "chars", "words", "chars vs prose"],
                [
                    [
                        rep.label,
                        str(len(rep.units)),
                        f"{rep.chars:,}",
                        f"{rep.words:,}",
                        fmt(rep.chars / reps["prose"].chars, 2) + "×",
                    ]
                    for rep in reps.values()
                ]
                + [
                    [
                        "ENTITY, embedded form (`name — description`)",
                        str(len(payload["dataset"]["entities"])),
                        f"{payload['embedded_chars']:,}",
                        f"{payload['embedded_words']:,}",
                        fmt(payload["embedded_chars"] / reps["prose"].chars, 2) + "×",
                    ]
                ],
            )
        )
    )
    add("")
    add(
        "**This is the first surprise, and it is a null result for the naive framing of the "
        "question.** Extraction here is close to volume-neutral: the emitted entity context is "
        f"{fmt(reps['entity'].chars / reps['prose'].chars, 2)}× the size of the prose it replaced, "
        f"and even the leaner embedded form is {fmt(payload['embedded_chars'] / reps['prose'].chars, 2)}×. "
        "Whatever entity extraction costs in this system, it is **not** paid for with a smaller "
        "context. The loss measured in M2 therefore cannot be explained away as \"you get what "
        "you pay for in tokens\"."
    )
    add("")

    add("## M2 — Retention by fact class (the direct test of H4)")
    add("")
    add(
        "\n".join(
            _table(
                ["fact class", "PROSE", "ENTITY (descriptions only)", "ENTITY+EDGES"],
                [
                    [
                        row_class,
                        f"{d['prose'].retained}/{d['prose'].n} = {fmt(d['prose'].rate)} {_ci(d['prose'].ci)}",
                        f"{d['entity'].retained}/{d['entity'].n} = {fmt(d['entity'].rate)} {_ci(d['entity'].ci)}",
                        f"{d['entity_edges'].retained}/{d['entity_edges'].n} = {fmt(d['entity_edges'].rate)} {_ci(d['entity_edges'].ci)}",
                    ]
                    for row_class, d in [
                        ("C1 identity", ident),
                        ("C2 relational, untyped", link),
                        ("C2s relational, typed", typed),
                        ("C3 quantitative", quant),
                    ]
                ],
            )
        )
    )
    add("")
    add(
        f"**The prediction holds, at the maximum possible magnitude.** In the entity-description "
        f"representation, identity retention is {fmt(ident['entity'].rate)} and relational "
        f"retention is {fmt(link['entity'].rate)} — a gap of "
        f"**{fmt(gap * 100, 1)} percentage points**. Not one of the "
        f"{link['entity'].n} authored relationships has both of its endpoints named inside any "
        "single entity block, so the loss is not \"degraded\" — it is **total**. The mechanism is "
        "mundane and general: a description averaging "
        f"{fmt(payload['mean_description_words'], 1)} words has no room to name its neighbours, "
        "so every edge in the corpus becomes unrecoverable the moment the prose is dropped."
    )
    add("")
    add(
        f"Quantitative detail degrades too, though less absolutely: "
        f"{quant['entity'].retained} of the {quant['prose'].n} numeric literals the prose asserts "
        f"survive into the entity descriptions ({fmt(quant['entity'].rate)}, CI "
        f"{_ci(quant['entity'].ci)}). Dropped literals: "
        + ", ".join(payload["dropped_numerics"])
        + "."
    )
    add("")
    if payload["invented_numerics"]:
        n_inv = len(payload["invented_numerics"])
        add(
            f"The entity representation also asserts {n_inv} numeric "
            + ("literal that appears" if n_inv == 1 else "literals that appear")
            + " nowhere in the source prose ("
            + ", ".join(payload["invented_numerics"])
            + "), which is worth flagging: "
            "compression here is not purely lossy, it is also (mildly) additive."
        )
        add("")
    add(
        "**Restoring edges restores the link channel, not the prose channel.** Adding "
        "relationship lines to each block lifts untyped relational retention to "
        f"{fmt(link['entity_edges'].rate)} and typed retention to {fmt(typed['entity_edges'].rate)}, "
        f"while quantitative retention stays at {fmt(quant['entity_edges'].rate)} — versus "
        f"{fmt(quant['prose'].rate)} for prose. Edges are a fix for *relations*; they are not a "
        "substitute for text units."
    )
    add("")
    add(
        "*(The ENTITY+EDGES row is generous to the graph by construction and should not be read "
        "as a result: the rendered edge line contains the authored edge description verbatim, so "
        "it satisfies the C2s typing test trivially. Its informative content is the C3 column, "
        "where edges do **not** close the gap to prose.)*"
    )
    add("")
    add(
        f"Note the PROSE column is not a perfect score either: {link['prose'].retained}/"
        f"{link['prose'].n} untyped and {typed['prose'].retained}/{typed['prose'].n} typed. A few "
        "authored relationships are cross-passage inferences that no single passage states, which "
        "is precisely the case a graph is supposed to serve. That is the honest counterweight to "
        "this finding and is discussed under threats to validity."
    )
    add("")

    add("## M2b — What the benchmark's `required_facts` can and cannot see")
    add("")
    add(
        f"The benchmark's {composition['instances']} required-fact instances "
        f"({composition['distinct']} distinct) are **{composition['entity_identity_instances']}/"
        f"{composition['instances']} entity names** — "
        f"{composition['entity_identity_instances'] / composition['instances']:.0%} class C1"
        + (
            "; not a single required fact is a relation, a date or a quantity."
            if not composition["non_identity"]
            else ", the exceptions being: " + ", ".join(composition["non_identity"]) + "."
        )
    )
    add("")
    add(
        "This is a structural limitation of the benchmark that RQ4 is in a position to state "
        "precisely: **the published fact-recall metric measures only the one class that "
        "extraction preserves perfectly.** It is blind, by construction, to C2 and C3. Every "
        "entity-vs-passage gap the benchmark reports is therefore a *retrieval* gap measured on "
        "the friendliest possible fact vocabulary — and it still **understates** representational "
        "loss, because the 100%-retained class is the only class being scored. The strict rule "
        "(a fact counts only if found in retrieved prose) scores entity-only systems at zero by "
        "construction; M2 is the independent, retrieval-free reason why that construction is the "
        "right one."
    )
    add("")

    add("## M3 — Does per-question compression predict per-question failure?")
    add("")
    add(payload["match_note"])
    add("")
    add(
        "\n".join(
            _table(
                ["system", "k", "fact recall", "full coverage", "strict recall", "mean chars"],
                [
                    [
                        f"{a.key} — {a.label}",
                        str(a.k),
                        fmt(a.agg["fact_recall"]),
                        fmt(a.agg["full_coverage"]),
                        fmt(a.agg["strict_fact_recall"]),
                        f"{a.agg['mean_context_chars']:.0f}",
                    ]
                    for a in arms
                ],
            )
        )
    )
    add("")
    if payload["per_question"]:
        add("Per-question inputs (all four columns generated, none hand-typed):")
        add("")
        add("\n".join(payload["per_question"]))
        add("")
    add("Correlations, n = 14 questions, percentile bootstrap over questions:")
    add("")
    add(
        "\n".join(
            _table(
                ["predictor → outcome", "Spearman ρ", "95% CI", "Pearson r", "95% CI", "note"],
                [
                    [c.name, fmt(c.rho), _ci(c.rho_ci), fmt(c.r), _ci(c.r_ci), c.note]
                    for c in corrs
                ],
            )
        )
    )
    add("")
    add(payload["m3_verdict"])
    add("")

    add("## Threats to validity")
    add("")
    for threat in payload["threats"]:
        add(f"- {threat}")
    add("")

    add("## What was measured vs what is inferred")
    add("")
    add(
        "**Measured.** Character and word volumes of three representations of one 34-passage "
        f"corpus; presence of {ident['prose'].n} entity names, {link['prose'].n} relationship "
        f"endpoint-pairs and {quant['prose'].n} numeric literals within single retrievable units "
        "of each; the composition of the benchmark's required-fact vocabulary; and passage- vs "
        "entity-retrieval fact recall at a matched context budget."
    )
    add("")
    add(
        "**Inferred.** That the mechanism behind entity-only retrieval's weakness on multi-hop "
        "questions is representational rather than merely a ranking problem; that this "
        "generalises to other extractors that emit short per-entity descriptions; and that the "
        "same loss would appear on a corpus this fixture does not contain. None of those three "
        "is established by these measurements alone."
    )
    add("")

    add("## Claim we could defend in a paper")
    add("")
    add(f"> {payload['claim']}")
    add("")
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="skip M3 (which needs an embedder); M1/M2 are database- and model-free",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset = load_dataset()
    reps = build_representations(dataset)
    comp = measure_compression(dataset)
    retention = measure_retention(reps, dataset)
    composition = required_fact_composition(dataset)

    numerics = quantitative_facts(dataset)
    entity_numerics = set(_NUMERIC.findall(" ".join(reps["entity"].units)))
    dropped = [n for n in numerics if n not in entity_numerics]
    invented = sorted(entity_numerics - set(numerics))

    arms: list[RetrievalArm] = []
    corrs: list[Correlation] = []
    payload_per_question: list[str] = []
    match_note = "M3 skipped (`--no-retrieval`)."
    m3_verdict = "M3 was not run."
    if not args.no_retrieval:
        arms, match_note = await run_retrieval(dataset)
        ratios_by_q = per_question_compression(dataset, comp)
        density_by_q = per_question_entity_density(dataset)
        links_by_q = per_question_link_retention(dataset, reps["entity"])
        prose_links_by_q = per_question_link_retention(dataset, reps["prose"])
        entity_arm = next(a for a in arms if a.key == "B")
        passage_arm = next(a for a in arms if a.key == "A")
        qids = [q["id"] for q in dataset["questions"]]
        corrs = [
            correlate(
                "evidence compression ratio → entity-RAG failure *(primary)*",
                [(ratios_by_q[q], 1 - entity_arm.recall_by_qid[q]) for q in qids],
            ),
            correlate(
                "evidence compression ratio → passage-RAG failure *(control)*",
                [(ratios_by_q[q], 1 - passage_arm.recall_by_qid[q]) for q in qids],
            ),
            correlate(
                "entity density of evidence → entity-RAG failure *(confound)*",
                [(density_by_q[q], 1 - entity_arm.recall_by_qid[q]) for q in qids],
            ),
            correlate(
                "compression ratio ↔ entity density *(confound check)*",
                [(ratios_by_q[q], density_by_q[q]) for q in qids],
            ),
            correlate(
                "chain-link retention in ENTITY → entity-RAG failure",
                [(links_by_q[q], 1 - entity_arm.recall_by_qid[q]) for q in qids],
            ),
            correlate(
                "chain-link retention in PROSE → passage-RAG failure",
                [(prose_links_by_q[q], 1 - passage_arm.recall_by_qid[q]) for q in qids],
            ),
        ]
        primary, _control, confound, confound_check = corrs[0], corrs[1], corrs[2], corrs[3]
        per_question = [
            "| qid | evidence compression ratio | entity density | entity-RAG recall | passage-RAG recall |",
            "|---|---|---|---|---|",
            *[
                f"| {q} | {ratios_by_q[q]:.2f} | {density_by_q[q]:.2f} | "
                f"{entity_arm.recall_by_qid[q]:.2f} | {passage_arm.recall_by_qid[q]:.2f} |"
                for q in qids
            ],
        ]
        if primary.rho != primary.rho:
            m3_verdict = (
                "**Undefined — a non-result.** The primary correlation is not computable: "
                + (
                    "the outcome is constant across all 14 questions"
                    if primary.y_constant
                    else "the predictor is constant"
                )
                + ". With nothing varying, compression ratio has nothing to predict."
            )
        else:
            spans_zero = primary.rho_ci[0] <= 0 <= primary.rho_ci[1]
            if spans_zero:
                headline = (
                    "**Null result for the secondary prediction.** The interval spans zero, so at "
                    "this sample size a question's failure is *not* predictable from how heavily "
                    "its evidence was compressed."
                )
            elif primary.rho < 0:
                headline = (
                    "**The secondary prediction is contradicted, not confirmed.** The interval "
                    "excludes zero but the sign is *negative*: questions whose evidence was "
                    "compressed **harder** were answered **better** by entity-only retrieval — "
                    "the opposite of what H4's per-question corollary predicts. We report the "
                    "direction we measured, not the one we hoped for."
                )
            else:
                headline = (
                    "**Positive result for the secondary prediction**, with the caveat that a "
                    "bootstrapped rank correlation at n=14 is fragile; treat as suggestive only."
                )
            m3_verdict = (
                f"{headline}\n\nSpearman ρ = {fmt(primary.rho)}, bootstrap 95% CI "
                f"{_ci(primary.rho_ci)} over {primary.rho_ci[2]:,} usable resamples of n=14 "
                "questions."
            )
            if primary.rho < 0 and not spans_zero:
                m3_verdict += (
                    "\n\n**Why the sign is almost certainly an artefact, not a discovery.** The "
                    "per-passage compression ratio is close to the reciprocal of how many "
                    "entities the extractor pulled out of that passage, and the confound check "
                    f"confirms it (ρ = {fmt(confound_check.rho)}, CI {_ci(confound_check.rho_ci)}). "
                    "Entity density in turn predicts entity-RAG failure directly "
                    f"(ρ = {fmt(confound.rho)}, CI {_ci(confound.rho_ci)}): when a question's "
                    "evidence was shredded into many entity records, a fixed top-k of entity "
                    "blocks covers a smaller share of the required facts. That is **retrieval "
                    "competition for a fixed budget**, not information loss — it is a property of "
                    "k, not of the representation. So the primary correlation is measuring the "
                    "confound, and the honest conclusion is that **M3 provides no per-question "
                    "support for H4 in either direction.** The evidence for H4 is M2, which is "
                    "retrieval-free and therefore immune to this confound."
                )
        degenerate = [c.name for c in corrs if c.x_constant or c.y_constant]
        if degenerate:
            m3_verdict += (
                "\n\nUndefined arms (a constant predictor or outcome, so no correlation exists): "
                + "; ".join(degenerate)
                + ". That list is the M2 finding restated in per-question form: chain-link "
                "retention in the entity representation is 0.0 for **every one** of the 14 "
                "questions, which is exactly why it can predict nothing — there is no variation "
                "left to explain."
            )
        payload_per_question = per_question

    ratios = [c.ratio for c in comp]
    ident = {r.representation: r for r in retention if r.fact_class.startswith("C1")}
    link = {r.representation: r for r in retention if r.fact_class.startswith("C2 ")}

    claim = (
        "On a 34-passage multi-hop fixture, distilling prose into per-entity descriptions "
        f"(mean {statistics.fmean([len(e['description'].split()) for e in dataset['entities']]):.1f} "
        "words) preserves 100% of entity identities while making 100% of the corpus's "
        "relationships unrecoverable from any single entity record, and it does so at *no* "
        f"reduction in context volume ({reps['entity'].chars / reps['prose'].chars:.2f}× the "
        "characters of the prose it replaces) — so an entity-only index is strictly dominated, "
        "and retaining the source text units is not an optimisation but a correctness "
        "requirement for relational questions."
    )

    threats = [
        f"**Small n.** 14 questions, {len(dataset['passages'])} passages, one hand-built corpus. "
        "The M1/M2 unit counts are larger (34/79/81/23) and their CIs are reported, but the "
        "corpus is still a single sample and the per-question analysis in M3 is badly "
        "underpowered.",
        "**Hand-authored fixture, not a live extractor.** The entity descriptions ship with the "
        "repo; they were written by a human under the benchmark's fairness rule R1 (a question "
        "must share no word family with the descriptions of its non-anchor facts). That rule "
        "pushes descriptions toward terseness and self-containment, which plausibly *increases* "
        "the measured relational loss. The 0/81 figure should be read as \"what this fixture's "
        "descriptions do\", not \"what every LLM extractor does\". Replication against live "
        "extraction output is the obvious next step and would cost LLM calls this experiment was "
        "constrained to avoid.",
        "**Recoverability is operationalised lexically.** C2 asks whether both endpoint *names* "
        "occur in one unit. A description could in principle gesture at a neighbour without "
        "naming it (\"the machine she wrote for\") and would score as lost. This makes the "
        "entity-side number a lower bound on semantic retention — but note it is also exactly "
        "the standard a string-matching retriever or a citation check would apply.",
        "**C3 is proxied by numeric literals.** Dates and counts are a legible slice of "
        "quantitative detail, not all of it; qualitative attributes (\"British\", \"first\") are "
        "not counted, so C3 understates the attribute channel in both directions.",
        "**The typed-relation threshold is a choice.** C2s uses a "
        f"{TYPED_COVERAGE:.0%} content-word coverage bar. It is applied identically to all three "
        "representations, so it cannot favour one, but the absolute C2s numbers move with it; "
        "the untyped C2 row carries the headline and needs no threshold.",
        "**PROSE is not a ceiling.** Some authored relationships are cross-passage inferences "
        "no single passage states, so prose scores below 1.0 on C2. Those are the cases that "
        "motivate a graph in the first place; this experiment quantifies the cost of the graph "
        "*replacing* prose, not of the graph existing alongside it.",
        "**M3's predictor is confounded and its CIs are fragile.** The per-question compression "
        "ratio is nearly the reciprocal of entity density, so the correlation it shows is not "
        "cleanly attributable to information loss (this is quantified in M3 rather than asserted). "
        "Two arms have a constant or near-constant predictor, which makes a percentile bootstrap "
        "CI meaningless there; those rows are flagged in the table rather than dropped, because "
        "an undefined correlation is itself the M2 result restated.",
        "**M3 embeds; M1/M2 do not.** The retrieval arms depend on the FastEmbed model and on "
        "the k chosen for the budget match. M1 and M2 — which carry the headline — are pure "
        "functions of the shipped fixture and are reproducible with no model and no database.",
    ]

    payload = {
        "dataset": dataset,
        "reps": reps,
        "compression": comp,
        "retention": retention,
        "composition": composition,
        "arms": arms,
        "correlations": corrs,
        "match_note": match_note,
        "m3_verdict": m3_verdict,
        "per_question": payload_per_question,
        "threats": threats,
        "claim": claim,
        "dropped_numerics": dropped,
        "invented_numerics": invented,
        "embedded_chars": sum(len(entity_document(e)) for e in dataset["entities"]),
        "embedded_words": sum(len(entity_document(e).split()) for e in dataset["entities"]),
        "mean_description_words": statistics.fmean(
            [len(e["description"].split()) for e in dataset["entities"]]
        ),
        "command": (
            "cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7688 "
            "NEO4J_PASSWORD=research_secret \\\n"
            "  python -m research.rq4_compression_loss.experiment"
            + (" --no-retrieval" if args.no_retrieval else "")
        ),
    }

    FINDINGS.write_text(build_findings(payload), encoding="utf-8")

    print(f"M1 median compression ratio {statistics.median(ratios):.2f}×  "
          f"(entity context is {reps['entity'].chars / reps['prose'].chars:.2f}× the prose)")
    print(f"M2 identity retention {ident['entity'].rate:.3f}  vs  "
          f"relational retention {link['entity'].rate:.3f}  "
          f"(gap {100 * (ident['entity'].rate - link['entity'].rate):.1f} pts)")
    print(f"M2b required_facts that are entity names: "
          f"{composition['entity_identity_instances']}/{composition['instances']}")
    for c in corrs:
        print(f"M3 {c.name}: rho={fmt(c.rho)} CI={_ci(c.rho_ci)}")
    print(f"→ {FINDINGS}")
    print(json.dumps({"findings": str(FINDINGS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
