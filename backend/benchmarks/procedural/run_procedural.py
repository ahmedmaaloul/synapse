# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedural Graphs benchmark (does procedural memory help the Navigator?)

WHY THIS EXISTS. Synapse implements Procedural Graphs (Lu, Chen, Wu, Arık,
"Procedural Graphs: Self-Evolving Execution Structures for LLM Agents",
arXiv:2609.09153) on top of its own GraphRAG Navigator
(``app.services.graph_agent``). The paper reports that graph guidance helps and
that it costs tokens (Table 9, HotpotQA: 10,116 vs 4,003 tokens per question
for its best mode). Synapse adds two things of its own that the paper never
measured: *raw* local guidance, which makes zero extra LLM calls, and a
localization cascade. Both are hypotheses. This harness is how they get tested
instead of asserted: every system answers the same held-out questions, and
every number in the report comes from the run.

THE SYSTEMS (all the same Navigator, the same tools, the same model):

  no_pg           guidance "none". The paper's unguided baseline.
  pg_raw_local    the expert prior, raw local subgraph. Synapse's default:
                  zero guidance LLM calls (the missing cell of the paper's
                  Table 3).
  pg_gen_local    the expert prior, generative local guidance. The paper's
                  configuration: one guidance LLM call per step. On the prior
                  the localization cascade reduces to the paper's Match: every
                  parsed action is a tool the prior has a node for (exact),
                  and a failed parse falls back to the full graph (none).
  pg_gen_full     generative guidance over the FULL graph (forced). The
                  paper's scope ablation.
  pg_evolved_raw  with ``--evolve ROUNDS``: the prior after
  pg_evolved_gen  ``procedural_evolution.evolve`` (Algorithm 1) on the train
                  split, gated on the val split, saved under
                  ``<prior>-evolved-bench`` so the default graph is untouched.

THE METRICS, per system, on the TEST split (the paper's Table 9 columns):
EM and F1 (SQuAD/HotpotQA normalization, ``app.services.qa_metrics``), mean
solver steps, LLM calls, guidance LLM calls, input/output tokens (read from the
provider's usage metadata; when it is missing they are estimated at chars/4 and
flagged ``≈``), parse failures and latency.

EFFECT-SIZE FLOOR. With N test questions, one question is worth 100/N points.
A gap smaller than that is reported as TOO CLOSE TO CALL, never as a lead: the
same rule as the retrieval benchmarks. A gap of exactly one question IS called,
because on EM every gap is a whole number of questions. The comparison adds a
1e-9 tolerance so that floating-point subtraction (5/9 − 4/9 < 1/9) cannot
demote it. No significance is claimed anywhere; on 9 questions a one-question
gap is a direction, not a finding.

THE DATASETS.
  • ``--dataset demo`` (default): ``demo_qa.json``, 30 questions over the
    zero-key demo graph (``scripts/demo_graph.json``), split train 12 / val 9 /
    test 9. Every answer is checked against the fixture before a run starts
    (``verify_demo_qa``). The graph must already be seeded
    (``python -m scripts.seed_demo --clear``) and hold nothing else: the
    harness refuses a missing fixture entity, and also extra entities or any
    source chunk, because other documents' content would surface in the
    tools' results. ``--allow-mixed-graph`` runs anyway and flags the report.
    The file is a JSON array of ``{question, answer, split, …}``
    items, so ``synapse-graphrag evolve --train-split train --val-split val``
    reads it as-is.
  • ``--dataset hotpotqa --n N --reuse-graph``: the seeded HotpotQA sample from
    ``benchmarks/public/hotpotqa.py``, split 40/30/30 by position. This harness
    NEVER ingests. The corpus must already be in Neo4j (built by
    ``benchmarks.public.run_hotpotqa``); anything else is refused.

COST. Every system spends real LLM calls, and generative guidance doubles them.
  • ``--dry-run`` prints an UPPER-BOUND call and cost estimate and exits
    WITHOUT calling any model, embedder or database. Prompt sizes come from the
    real templates; the price table is ``benchmarks/public/cost.py``, and every
    USD figure quotes the date of the price it used (``cost.price_checked_on``).
  • Reasoning models (gpt-5, o-series: ``llm_provider.is_reasoning_model``)
    also bill hidden reasoning tokens as output. The dry run adds
    ``REASONING_TOKENS_ALLOWANCE`` per call for them, an ASSUMPTION sized for
    effort ``minimal`` that the pilot run calibrates (the report prints the
    measured mean); the metered caps count the real ones, because the
    provider's ``output_tokens`` includes them.
  • ``--max-usd`` (hard stop): the run refuses to start when the upper bound
    is over it, and every model call is metered and checked against it BEFORE
    it is made. ``--max-llm-calls`` is a second, count-based cap; it is the
    only one that still works for a model with no price on file. With
    ``--evolve``, a cap too small to reach one whole evolution round after
    the systems before it (charged at their worst case) is refused before any
    call (``EvolutionFloor``), not after those systems were paid for.
  • A real run is priced as the model it CALLS (the configured one).
    ``--model`` re-prices a ``--dry-run``; in a real run it is used only when
    the configured model id has no price on file (a deployment alias, a local
    model), and the report says so. A cheaper ``--model`` can never weaken
    the ``--max-usd`` cap of a priced model.
  • Nothing is written to the database except by ``--evolve`` (versions,
    rejections and trajectories of ``<prior>-evolved-bench``).

RUN IT:

    cd backend && python -m scripts.seed_demo --clear            # free, no LLM
    cd backend && python -m benchmarks.procedural.run_procedural --dry-run
    cd backend && python -m benchmarks.procedural.run_procedural --systems no_pg,pg_raw_local

Exit codes: 0 success · 1 dataset or arguments unusable · 2 Neo4j unreachable ·
3 the run produced a result the harness cannot honestly report · 4 the graph is
not prepared (demo not seeded or mixed with other documents, HotpotQA corpus not
ingested, or HotpotQA without ``--reuse-graph``) · 5 a cost cap (pre-flight or
mid-run) · 6 no chat model is configured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services import (
    graph_agent,
    procedural_evolution,
    procedural_guidance,
    procedural_store,
)
from app.services.llm_provider import is_reasoning_model, reasoning_effort_for
from app.services.procedural_graph import (
    START,
    ProceduralGraph,
    graph_diff,
    serialize_full,
    serialize_local,
    summarize_diff,
)
from app.services.qa_metrics import METRICS, exact_match, normalize_answer
from app.services.qa_metrics import f1 as f1_score
from benchmarks.public import cost, hotpotqa, run_hotpotqa
from benchmarks.run_benchmark import DatasetError, IntegrityError
from benchmarks.run_benchmark import _markdown_lines as markdown_lines

logger = logging.getLogger(__name__)

RunFn = Callable[..., Awaitable[Any]]

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DEMO_QA_PATH = HERE / "demo_qa.json"
DEMO_GRAPH_PATH = HERE.parent.parent / "scripts" / "demo_graph.json"
#: Gitignored: a run's numbers belong to that run, not to the repository.
RESULTS_PATH = HERE / "results_procedural.md"

# ── Cost discipline ──────────────────────────────────────────────────────────
#: Same default cap as the HotpotQA harness. On gpt-4o-mini the four base
#: systems over the 9 demo test questions are a few cents even at the upper
#: bound; the cap is there for a bigger model or a bigger ``--n``.
DEFAULT_MAX_USD = 2.00
#: Completion tokens ``--dry-run`` assumes per call. Guesses, labelled as such:
#: the real run measures them and the report prints the measured means.
DRY_RUN_SOLVER_COMPLETION_TOKENS = 120
DRY_RUN_GUIDANCE_COMPLETION_TOKENS = 350
DRY_RUN_REFINER_COMPLETION_TOKENS = 1500
#: Hidden reasoning tokens ``--dry-run`` adds to EVERY call of an OpenAI
#: reasoning model (gpt-5, o-series), which bills them as output. An
#: ASSUMPTION, not a bound: nothing caps reasoning (no max_completion_tokens is
#: sent), and 512 is sized for reasoning effort ``minimal`` on short ReAct
#: turns. Calibrate it on the pilot run — the report prints the measured mean
#: per call — before trusting a large run's estimate. The metered ``--max-usd``
#: cap counts the real reasoning tokens whatever this says.
REASONING_TOKENS_ALLOWANCE = 512
#: Characters one turn adds to the rendered trajectory besides its observation:
#: the solver's own completion (the ``Thought:`` and ``Action:`` lines, at the
#: assumed completion size), which every later prompt re-reads, plus framing.
DRY_RUN_TURN_CHARS = DRY_RUN_SOLVER_COMPLETION_TOKENS * cost.CHARS_PER_TOKEN + 40

# ── Dataset rules ────────────────────────────────────────────────────────────
SPLITS: tuple[str, ...] = ("train", "val", "test")
DEMO_SPLIT_SIZES: dict[str, int] = {"train": 12, "val": 9, "test": 9}
#: HotpotQA splits by position in the seeded sample; val and test get 30% each.
HELD_OUT_FRACTION = 0.3
QA_TYPES: tuple[str, ...] = ("single", "bridge", "comparison")
COMPARE_ASKS: tuple[str, ...] = ("earlier", "later", "first_before_second")
#: Share of demo questions that must need two or more graph facts.
MIN_MULTI_HOP_SHARE = 0.6
#: "Short exact answers": names and years, never sentences.
MAX_ANSWER_WORDS = 4
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")

#: Suffix of the graph ``--evolve`` writes. The default graph is never touched.
EVOLVED_SUFFIX = "-evolved-bench"

#: Tolerance of the effect-size comparison (see the module docstring).
FLOOR_TOLERANCE = 1e-9


class BudgetExceeded(procedural_evolution.SpendCapReached):
    """A cost cap was reached. ``partial`` holds what finished before it.

    A ``SpendCapReached`` (itself a ``RuntimeError``), so that when the cap
    refuses evolution's refiner call ``procedural_evolution.evolve`` stops
    with ``stopped="budget"`` instead of logging a refiner failure.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.partial: BenchResult | None = None


class GraphNotReady(RuntimeError):
    """Neo4j does not hold the graph this dataset must be answered from."""


# ── Dataset ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class QAItem:
    """One question. ``gold`` (answer + aliases) is what EM/F1 score against."""

    id: str
    question: str
    answer: str
    split: str
    qtype: str = "bridge"
    hops: int = 2
    aliases: tuple[str, ...] = ()

    @property
    def gold(self) -> list[str]:
        return [self.answer, *self.aliases]

    def as_evolution_item(self) -> dict:
        """The ``{question, answer}`` shape ``procedural_evolution.evolve`` takes."""
        return {"question": self.question, "answer": self.gold}


@dataclass
class Dataset:
    name: str
    items: list[QAItem]
    source: str
    #: Dataset-specific facts the readiness check and the report need.
    notes: dict = field(default_factory=dict)

    def split(self, name: str) -> list[QAItem]:
        return [item for item in self.items if item.split == name]

    def sizes(self) -> dict[str, int]:
        return {name: len(self.split(name)) for name in SPLITS}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise DatasetError(f"could not read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path} is not valid JSON: {e}") from e


def evidence_facts(item: Mapping[str, Any]) -> tuple[list[list[str]], list[list[str]]]:
    """``(relations, descriptions)`` of a demo item, as lists (empty when absent)."""
    evidence = item.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return [], []
    relations = evidence.get("relations") or []
    descriptions = evidence.get("descriptions") or []
    return list(relations), list(descriptions)


def evidence_hops(item: Mapping[str, Any]) -> int:
    """Graph facts a reader must combine: one per relation triple or description read."""
    relations, descriptions = evidence_facts(item)
    return len(relations) + len(descriptions)


def _year(description: str) -> int | None:
    years = {int(match) for match in _YEAR.findall(description or "")}
    return years.pop() if len(years) == 1 else None


def _connected(relations: Sequence[Sequence[str]], descriptions: Sequence[Sequence[str]]) -> bool:
    """True when the evidence forms one chain of shared entities (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for source, _, target in relations:
        parent[find(source)] = find(target)
    for entity, _ in descriptions:
        find(entity)
    return len({find(x) for x in list(parent)}) <= 1


def answer_forms(item: Mapping[str, Any]) -> list[str]:
    """``[canonical, *aliases]`` of an item whose ``answer`` is a string or a list."""
    raw = item.get("answer")
    forms = raw if isinstance(raw, list) else [raw]
    return [str(form).strip() for form in forms if form is not None and str(form).strip()]


def verify_demo_qa(payload: Any, fixture: Mapping[str, Any]) -> list[str]:
    """Every problem with a demo QA set, checked against the demo graph fixture.

    ``payload`` is the file's JSON array of items. ``answer`` is a string, or
    a list whose first entry is the canonical answer and the rest acceptable
    aliases (the shape ``qa_metrics``, the API and the client CLI all score).

    Empty means every answer is derivable from facts the Navigator can see.
    The Navigator's tools show relation triples (``neighbors``, ``find_path``)
    and entity descriptions (``search_entities``, ``neighbors``); they never
    print a relationship's own description, and the seeded demo graph has no
    source passages. So evidence is limited to those two kinds of fact, and:

      1. every relation triple exists verbatim (source, TYPE, target, direction)
         in the fixture, and every description snippet is a case-insensitive
         substring of that entity's fixture description;
      2. ``hops`` equals the number of evidence facts, and the ``type`` agrees
         (``single`` = 1 fact, ``bridge`` ≥ 2 facts chained through shared
         entities, ``comparison`` = 2 descriptions);
      3. the canonical answer is grounded: for ``comparison`` it is RECOMPUTED
         from the years in the two fixture descriptions; otherwise it is an
         entity named in the evidence or a substring of a description snippet,
         and it does not appear in its own question;
      4. when the answer is a relation endpoint, that final hop is unambiguous:
         no other entity stands in the same relation to the same other end;
      5. set-level: ids and questions are unique, splits are 12/9/9, every
         answer form is at most four words, and at least 60% of items need two
         facts.
    """
    if not isinstance(payload, list):
        return ["the QA file must be a JSON array of items"]
    entities = {
        str(e.get("name")): str(e.get("description") or "")
        for e in fixture.get("entities") or []
        if isinstance(e, Mapping)
    }
    triples = [
        (str(r.get("source")), str(r.get("type")), str(r.get("target")))
        for r in fixture.get("relationships") or []
        if isinstance(r, Mapping)
    ]
    triple_set = set(triples)

    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    splits: Counter[str] = Counter()
    multi_hop = 0
    items = payload
    for index, item in enumerate(items):
        label = f"item #{index}"
        if not isinstance(item, Mapping):
            problems.append(f"{label} is not an object")
            continue
        qid = str(item.get("id") or "").strip()
        label = qid or label
        question = str(item.get("question") or "").strip()
        forms = answer_forms(item)
        answer = forms[0] if forms else ""
        split = item.get("split")
        qtype = item.get("type")
        if not qid:
            problems.append(f"{label} has no id")
        elif qid in seen_ids:
            problems.append(f"duplicate id {qid!r}")
        seen_ids.add(qid)
        if not question.endswith("?"):
            problems.append(f"{label}: the question must be a question ending in '?'")
        if normalize_answer(question) in seen_questions:
            problems.append(f"{label}: duplicate question")
        seen_questions.add(normalize_answer(question))
        if not answer:
            problems.append(f"{label} has no answer")
            continue
        for form in forms:
            if len(form.split()) > MAX_ANSWER_WORDS:
                problems.append(
                    f"{label}: answer {form!r} is longer than {MAX_ANSWER_WORDS} words"
                )
        if len({normalize_answer(form) for form in forms}) != len(forms):
            problems.append(f"{label}: the answer aliases repeat each other")
        if split not in SPLITS:
            problems.append(f"{label}: split must be one of {', '.join(SPLITS)}")
        else:
            splits[split] += 1
        if qtype not in QA_TYPES:
            problems.append(f"{label}: type must be one of {', '.join(QA_TYPES)}")
            continue

        relations, descriptions = evidence_facts(item)
        malformed = [r for r in relations if not (isinstance(r, list) and len(r) == 3)]
        malformed += [d for d in descriptions if not (isinstance(d, list) and len(d) == 2)]
        if malformed or not (relations or descriptions):
            problems.append(f"{label}: evidence must be [source, TYPE, target] / [entity, text]")
            continue
        for source, rel_type, target in relations:
            if (source, rel_type, target) not in triple_set:
                problems.append(
                    f"{label}: relation {source} -[{rel_type}]-> {target} is not in the fixture"
                )
        for entity, snippet in descriptions:
            if entity not in entities:
                problems.append(f"{label}: entity {entity!r} is not in the fixture")
            elif str(snippet).lower() not in entities[entity].lower():
                problems.append(
                    f"{label}: {snippet!r} is not in the fixture description of {entity!r}"
                )

        facts = evidence_hops(item)
        if item.get("hops") != facts:
            problems.append(f"{label}: hops is {item.get('hops')!r} but the evidence has {facts}")
        if facts >= 2:
            multi_hop += 1
        if qtype == "single" and facts != 1:
            problems.append(f"{label}: a 'single' question must rest on exactly one fact")
        if qtype == "bridge":
            if facts < 2:
                problems.append(f"{label}: a 'bridge' question needs at least two facts")
            elif not _connected(relations, descriptions):
                problems.append(f"{label}: the bridge evidence is not one connected chain")

        norm_answer = normalize_answer(answer)
        if qtype == "comparison":
            problems += _check_comparison(label, item, answer, descriptions, entities)
            continue
        named = {normalize_answer(e) for r in relations for e in (r[0], r[2])}
        named |= {normalize_answer(d[0]) for d in descriptions}
        in_snippet = any(norm_answer in normalize_answer(str(d[1])) for d in descriptions)
        if norm_answer not in named and not in_snippet:
            problems.append(f"{label}: answer {answer!r} is not grounded in its evidence")
        if norm_answer in normalize_answer(question):
            problems.append(f"{label}: the answer {answer!r} appears in its own question")
        final_hops = [
            r for r in relations if norm_answer in (normalize_answer(r[0]), normalize_answer(r[2]))
        ]
        for source, rel_type, target in final_hops:
            if normalize_answer(target) == norm_answer:
                rivals = [t for t in triples if t[0] == source and t[1] == rel_type]
            else:
                rivals = [t for t in triples if t[2] == target and t[1] == rel_type]
            if len(rivals) != 1:
                problems.append(
                    f"{label}: the final hop {source} -[{rel_type}]-> {target} is ambiguous "
                    f"({len(rivals)} matching relations in the fixture)"
                )

    if splits != Counter(DEMO_SPLIT_SIZES):
        problems.append(
            "splits must be "
            + " / ".join(f"{k} {v}" for k, v in DEMO_SPLIT_SIZES.items())
            + f", got {dict(splits)}"
        )
    if items and multi_hop / len(items) < MIN_MULTI_HOP_SHARE:
        problems.append(
            f"only {multi_hop}/{len(items)} questions need two or more facts "
            f"(at least {MIN_MULTI_HOP_SHARE:.0%} required)"
        )
    return problems


def _check_comparison(
    label: str,
    item: Mapping[str, Any],
    answer: str,
    descriptions: Sequence[Sequence[str]],
    entities: Mapping[str, str],
) -> list[str]:
    """Recompute a comparison answer from the years in the fixture descriptions."""
    compare = item.get("compare") or {}
    names = list(compare.get("entities") or []) if isinstance(compare, Mapping) else []
    ask = compare.get("ask") if isinstance(compare, Mapping) else None
    if len(names) != 2 or ask not in COMPARE_ASKS:
        return [f"{label}: comparison needs compare.entities (two) and compare.ask"]
    if sorted(d[0] for d in descriptions) != sorted(names):
        return [f"{label}: comparison evidence must be the two compared entities' descriptions"]
    years = [_year(entities.get(name, "")) for name in names]
    if None in years:
        return [f"{label}: each compared description must carry exactly one year"]
    first, second = years
    if first == second:
        return [f"{label}: the compared years are equal ({first}); the question has no answer"]
    if ask == "earlier":
        expected = names[0] if first < second else names[1]
    elif ask == "later":
        expected = names[0] if first > second else names[1]
    else:
        expected = "yes" if first < second else "no"
    if normalize_answer(expected) != normalize_answer(answer):
        return [f"{label}: the years ({first}, {second}) give {expected!r}, not {answer!r}"]
    return []


def _item_from_json(raw: Mapping[str, Any]) -> QAItem:
    answer, *aliases = answer_forms(raw)
    return QAItem(
        id=str(raw["id"]),
        question=str(raw["question"]).strip(),
        answer=answer,
        split=str(raw["split"]),
        qtype=str(raw.get("type") or "bridge"),
        hops=int(raw.get("hops") or evidence_hops(raw)),
        aliases=tuple(aliases),
    )


def load_demo_dataset(
    qa_path: Path = DEMO_QA_PATH, graph_path: Path = DEMO_GRAPH_PATH
) -> Dataset:
    """The demo QA set, refused unless it verifies against the demo graph fixture."""
    payload = load_json(qa_path)
    fixture = load_json(graph_path)
    problems = verify_demo_qa(payload, fixture)
    if problems:
        listed = "\n  - ".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n  ... and {len(problems) - 20} more"
        raise DatasetError(
            f"{qa_path} does not verify against {graph_path}:\n  - {listed}{more}"
        )
    return Dataset(
        name="demo",
        items=[_item_from_json(raw) for raw in payload],
        source=str(qa_path.name),
        notes={"entities": sorted(str(e["name"]) for e in fixture["entities"])},
    )


def split_sizes(n: int) -> dict[str, int]:
    """Positional 40/30/30 split sizes; at least one question in every split."""
    if n < len(SPLITS):
        raise DatasetError(f"need at least {len(SPLITS)} questions (one per split), got {n}")
    held_out = max(1, round(n * HELD_OUT_FRACTION))
    return {"train": n - 2 * held_out, "val": held_out, "test": held_out}


def load_hotpotqa_dataset(
    n: int, *, seed: int, data_path: Path | None = None, allow_download: bool = True
) -> Dataset:
    """The seeded HotpotQA sample (``benchmarks/public/hotpotqa.py``), split by position.

    The sample is already a seeded shuffle, so a positional split is a random
    split that ``(seed, n)`` names. ``notes["titles"]`` is every paragraph the
    sample needs in Neo4j; the readiness check refuses a graph missing any.
    """
    corpus = run_hotpotqa.load_corpus(
        questions=n, seed=seed, data_path=data_path, allow_download=allow_download
    )
    sizes = split_sizes(len(corpus.questions))
    labels = ["train"] * sizes["train"] + ["val"] * sizes["val"] + ["test"] * sizes["test"]
    items = [
        QAItem(
            id=question.id,
            question=question.question,
            answer=question.answer,
            split=label,
            qtype=question.hop_type,
            hops=2,
        )
        for question, label in zip(corpus.questions, labels, strict=True)
    ]
    return Dataset(
        name="hotpotqa",
        items=items,
        source=str(corpus.notes.get("source") or "hotpotqa"),
        notes={"titles": corpus.titles, "seed": seed, "n": n},
    )


def export_splits(dataset: Dataset, directory: Path) -> list[Path]:
    """Write ``train.json`` / ``val.json`` / ``test.json`` as ``[{question, answer}]``.

    That is the input shape of ``synapse-graphrag evolve --train/--val``. The
    demo file already is (with ``--train-split`` / ``--val-split``); this is
    how a HotpotQA sample, split here by position, gets there too. An item
    with aliases keeps them: ``answer`` becomes ``[canonical, *aliases]``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name in SPLITS:
        path = directory / f"{name}.json"
        rows = [
            {"question": i.question, "answer": i.gold if i.aliases else i.answer}
            for i in dataset.split(name)
        ]
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


# ── Systems ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SystemSpec:
    key: str
    label: str
    #: "none" | "prior" | "evolved"
    graph: str
    #: "none" | "raw" | "generative"
    guidance: str
    full_graph: bool = False
    #: Where the configuration comes from, for the report.
    origin: str = ""


SYSTEMS: dict[str, SystemSpec] = {
    spec.key: spec
    for spec in (
        SystemSpec("no_pg", "Navigator, no procedural graph", "none", "none",
                   origin="baseline (the paper's unguided row)"),
        SystemSpec("pg_raw_local", "Expert prior, raw local subgraph", "prior", "raw",
                   origin="Synapse addition: zero guidance LLM calls"),
        SystemSpec("pg_gen_local", "Expert prior, generative local guidance", "prior",
                   "generative", origin="the paper's configuration"),
        SystemSpec("pg_gen_full", "Expert prior, generative full-graph guidance", "prior",
                   "generative", full_graph=True, origin="the paper's scope ablation"),
        SystemSpec("pg_evolved_raw", "Evolved graph, raw local subgraph", "evolved", "raw",
                   origin="Algorithm 1 on the train split"),
        SystemSpec("pg_evolved_gen", "Evolved graph, generative local guidance", "evolved",
                   "generative", origin="Algorithm 1 on the train split"),
    )
}
BASE_SYSTEMS: tuple[str, ...] = ("no_pg", "pg_raw_local", "pg_gen_local", "pg_gen_full")
EVOLVED_SYSTEMS: tuple[str, ...] = ("pg_evolved_raw", "pg_evolved_gen")

#: (system, baseline, what the comparison tests). Printed only when both ran.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("pg_raw_local", "no_pg", "Synapse's default vs no procedural graph"),
    ("pg_gen_local", "no_pg", "the paper's configuration vs no procedural graph"),
    ("pg_gen_full", "no_pg", "full-graph generative guidance vs no procedural graph"),
    (
        "pg_raw_local",
        "pg_gen_local",
        "raw vs generative on the same local subgraph (Synapse's hypothesis: the guidance "
        "LLM is not needed)",
    ),
    ("pg_gen_local", "pg_gen_full", "local vs full scope, both generative (the paper's ablation)"),
    ("pg_evolved_raw", "pg_raw_local", "evolved vs expert prior, raw guidance"),
    ("pg_evolved_gen", "pg_gen_local", "evolved vs expert prior, generative guidance"),
)


def resolve_systems(raw: str | None, *, evolve: bool) -> list[str]:
    """The systems to run, in canonical order. Evolved systems need ``--evolve``."""
    if not raw:
        return list(BASE_SYSTEMS) + (list(EVOLVED_SYSTEMS) if evolve else [])
    wanted = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in SYSTEMS]
    if unknown:
        raise ValueError(
            f"unknown system(s) {', '.join(unknown)}; choose from {', '.join(SYSTEMS)}"
        )
    if not evolve and any(name in EVOLVED_SYSTEMS for name in wanted):
        raise ValueError("pg_evolved_* systems need --evolve ROUNDS")
    return [name for name in SYSTEMS if name in wanted]


# ── Spend: metering and the two hard caps ────────────────────────────────────
def billing_model(requested: str, configured: str) -> tuple[str, str | None]:
    """The model a REAL run is priced and capped as, and a note when ``--model`` lost.

    The run calls ``configured``, so its price governs ``--max-usd`` whenever
    one is on file: a cheaper ``--model`` would otherwise weaken the cap (gpt-4o
    metered as gpt-4o-mini is ~17x under). ``--model`` stands in only when the
    configured id has no price (a deployment alias, a local model), and then
    the cap is only as good as that mapping.
    """
    if not requested or requested == configured:
        return configured, None
    if cost.resolve_price(configured) is not None:
        return configured, (
            f"--model {requested!r} prices --dry-run estimates only: this run calls "
            f"{configured!r} and is priced and capped as it."
        )
    return requested, (
        f"No price on file for the configured {configured!r}: pricing it as --model "
        f"{requested!r}. The --max-usd cap is only as accurate as that mapping."
    )


class SpendGuard:
    """Meters every model call into per-phase ledgers and enforces the caps.

    The cap is checked BEFORE each call, so a run can overshoot ``max_usd`` by
    at most the one call that crossed it. A refused call raises
    :class:`BudgetExceeded` inside the Navigator, which reports it as an error
    step; ``tripped`` is how the harness still sees it after the run returns.
    """

    def __init__(self, model: str, *, max_usd: float | None, max_calls: int | None) -> None:
        self.model = model
        self.max_usd = max_usd
        self.max_calls = max_calls
        self.ledgers: list[cost.CostLedger] = []
        self.tripped: str | None = None

    def ledger(self, label: str) -> cost.CostLedger:
        ledger = cost.CostLedger(self.model, label=label)
        self.ledgers.append(ledger)
        return ledger

    @property
    def calls(self) -> int:
        return sum(ledger.usage.calls for ledger in self.ledgers)

    @property
    def usage(self) -> cost.Usage:
        total = cost.Usage()
        for ledger in self.ledgers:
            total = total + ledger.usage
        return total

    def usd(self) -> float | None:
        return cost.usd(self.usage, cost.resolve_price(self.model))

    def remaining_calls(self) -> int | None:
        return None if self.max_calls is None else max(0, self.max_calls - self.calls)

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if the next call could pass a cap."""
        if self.tripped:
            raise BudgetExceeded(self.tripped)
        reason = None
        if self.max_calls is not None and self.calls >= self.max_calls:
            reason = f"LLM-call cap reached: {self.calls} of --max-llm-calls {self.max_calls}"
        spent = self.usd()
        if reason is None and self.max_usd is not None and spent is not None:
            if spent >= self.max_usd:
                reason = (
                    f"metered spend {cost.format_usd(spent)} reached the --max-usd cap of "
                    f"{cost.format_usd(self.max_usd)} (estimated from "
                    f"{cost.price_checked_on(self.model)} prices)"
                )
        if reason:
            self.tripped = reason
            raise BudgetExceeded(reason)

    def raise_if_tripped(self) -> None:
        if self.tripped:
            raise BudgetExceeded(self.tripped)

    def wrap(self, model: Any, ledger: cost.CostLedger) -> GuardedModel:
        return GuardedModel(model, self, ledger)


class GuardedModel:
    """A chat model that checks the caps before each call and meters its response.

    Metering reads the provider's own usage metadata through
    ``cost.CostLedger.record_response`` (estimated at chars/4, and flagged, when
    it is absent), the same accounting as the HotpotQA harness.
    """

    def __init__(self, model: Any, guard: SpendGuard, ledger: cost.CostLedger) -> None:
        self.model = model
        self.guard = guard
        self.ledger = ledger

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.guard.check()
        response = await self.model.ainvoke(messages, *args, **kwargs)
        self.ledger.record_response(response, _prompt_text(messages))
        return response


def _prompt_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Sequence):
        return "\n".join(str(getattr(m, "content", m) or "") for m in messages)
    return str(messages)


# ── Scoring ──────────────────────────────────────────────────────────────────
@dataclass
class Outcome:
    """One question answered by one system."""

    item: QAItem
    answer: str | None
    em: float
    f1: float
    steps: int
    stopped: str
    parse_failures: int
    llm_calls: int
    guidance_llm_calls: int
    input_tokens: int
    output_tokens: int
    estimated: bool
    context_chars: int
    latency_s: float
    localizations: Counter = field(default_factory=Counter)
    graph: dict | None = None
    error: str | None = None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _as_dict(result: Any) -> dict:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def outcome_from_result(item: QAItem, result: Any) -> Outcome:
    """Score one ``AgentResult`` (or its dict) against the item's gold answers."""
    data = _as_dict(result)
    usage = data.get("usage") or {}
    steps = list(data.get("steps") or [])
    answer = data.get("answer")
    return Outcome(
        item=item,
        answer=answer,
        em=exact_match(answer, item.gold),
        f1=f1_score(answer, item.gold),
        steps=len(steps),
        stopped=str(data.get("stopped") or ""),
        parse_failures=int(data.get("parse_failures") or 0),
        llm_calls=int(usage.get("llm_calls") or 0),
        guidance_llm_calls=int(usage.get("guidance_llm_calls") or 0),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        estimated=bool(usage.get("estimated")),
        context_chars=int(usage.get("context_chars") or 0),
        latency_s=float(data.get("latency_s") or 0.0),
        localizations=Counter(s["localization"] for s in steps if s.get("localization")),
        graph=data.get("graph"),
        error=data.get("error"),
    )


def aggregate(outcomes: Sequence[Outcome]) -> dict:
    """Per-question means (and totals) over one system's outcomes."""
    n = len(outcomes)
    localization: Counter = Counter()
    for outcome in outcomes:
        localization.update(outcome.localizations)

    def mean(attribute: str) -> float:
        return sum(getattr(o, attribute) for o in outcomes) / n if n else 0.0

    def total(attribute: str) -> int:
        return sum(getattr(o, attribute) for o in outcomes)

    return {
        "n": n,
        "em": mean("em"),
        "f1": mean("f1"),
        "steps": mean("steps"),
        "llm_calls": mean("llm_calls"),
        "llm_calls_total": total("llm_calls"),
        "guidance_llm_calls": mean("guidance_llm_calls"),
        "guidance_llm_calls_total": total("guidance_llm_calls"),
        "input_tokens": mean("input_tokens"),
        "output_tokens": mean("output_tokens"),
        "tokens": mean("tokens"),
        "tokens_total": total("tokens"),
        "estimated": any(o.estimated for o in outcomes),
        "estimated_questions": sum(1 for o in outcomes if o.estimated),
        "parse_failures": mean("parse_failures"),
        "parse_failures_total": total("parse_failures"),
        "latency_s": mean("latency_s"),
        "context_chars": mean("context_chars"),
        "answered": sum(1 for o in outcomes if o.stopped == "answer") / n if n else 0.0,
        "max_steps_hits": sum(1 for o in outcomes if o.stopped == "max_steps"),
        "errors": sum(1 for o in outcomes if o.stopped == "error"),
        "localization": dict(localization),
    }


def verdict(
    label: str,
    key: str,
    baseline: Mapping[str, Any],
    system: Mapping[str, Any],
    *,
    baseline_label: str,
    system_label: str,
) -> str:
    """One metric compared under the effect-size floor (wording of ``metric_verdict``)."""
    n = int(system.get("n", 0) or baseline.get("n", 0) or 0)
    floor = run_hotpotqa.effect_floor(n)
    delta = system[key] - baseline[key]
    if abs(delta) < FLOOR_TOLERANCE:
        return f"{label}: TIE ({system[key]:.1%} for both, N={n})."
    if n < 2:
        return (
            f"{label}: NOT SCOREABLE — {system_label} {system[key]:.1%} vs {baseline_label} "
            f"{baseline[key]:.1%} on N={n}. One question is not a measurement."
        )
    if abs(delta) < floor - FLOOR_TOLERANCE:
        leader = system_label if delta > 0 else baseline_label
        return (
            f"{label}: TOO CLOSE TO CALL — {system_label} {system[key]:.1%} vs "
            f"{baseline_label} {baseline[key]:.1%} ({delta * 100:+.1f} pp, nominally {leader}). "
            f"The gap is smaller than one question out of {n} ({floor * 100:.1f} pp). No "
            "significance is claimed."
        )
    ahead, behind = (system_label, baseline_label) if delta > 0 else (baseline_label, system_label)
    return (
        f"{label}: {ahead} ahead of {behind} — {system[key]:.1%} vs {baseline[key]:.1%} "
        f"({delta * 100:+.1f} pp for {system_label}, at least one question = "
        f"{floor * 100:.1f} pp, N={n}). A direction on a small sample, not a significance "
        "result."
    )


# ── Execution ────────────────────────────────────────────────────────────────
@dataclass
class Models:
    """The unwrapped chat models; the harness meters and caps them per phase."""

    solver: Any
    guidance: Any
    refiner: Any = None


def make_models(settings: Any, *, evolve: bool) -> Models:
    """The configured provider's models. Raises ``ProviderConfigError`` without a key."""
    from app.services import llm_provider

    return Models(
        solver=llm_provider.get_chat_llm(temperature=settings.agent_temperature),
        guidance=llm_provider.get_chat_llm(temperature=0),
        refiner=llm_provider.get_chat_llm(temperature=0, json_mode=True) if evolve else None,
    )


def navigator_tools() -> dict:
    """The Navigator's real tools (deterministic retrieval over Neo4j, no LLM)."""
    return dict(graph_agent.DEFAULT_TOOLS)


@dataclass
class SystemRun:
    spec: SystemSpec
    outcomes: list[Outcome] = field(default_factory=list)
    ledger_label: str = ""
    graph_version: int | None = None

    @property
    def agg(self) -> dict:
        return aggregate(self.outcomes)


@dataclass
class BenchResult:
    dataset: Dataset
    prior: ProceduralGraph
    max_steps: int
    metric: str = "f1"
    runs: dict[str, SystemRun] = field(default_factory=dict)
    evolution: dict | None = None
    evolved_graph: ProceduralGraph | None = None
    evolved_version: int | None = None
    guard: SpendGuard | None = None
    provenance: dict = field(default_factory=dict)
    graph_status: dict = field(default_factory=dict)
    #: The model the run CALLED, and the reasoning effort it was sent (None for
    #: a model that takes a temperature instead). Set by ``main``.
    model: str = ""
    reasoning_effort: str | None = None


def evolved_name(prior: ProceduralGraph) -> str:
    return f"{prior.name}{EVOLVED_SUFFIX}"


def episode_calls_bound(guidance: str, max_steps: int) -> int:
    """Most LLM calls one episode can make: a solver turn per step, plus a guidance
    call per step in generative mode (the rule ``procedural_evolution`` budgets by)."""
    return int(max_steps) * (2 if guidance == "generative" else 1)


def _bound_run(
    run: RunFn,
    *,
    guard: SpendGuard,
    llm: Any,
    guidance_llm: Any,
    tools: Mapping[str, Any] | None,
    full_graph: bool = False,
) -> RunFn:
    """``run`` with the metered models and tools bound, the caps checked around it."""

    async def bound(question: str, **kwargs: Any) -> Any:
        guard.check()
        result = await run(
            question,
            llm=llm,
            guidance_llm=guidance_llm,
            tools=tools,
            full_graph=full_graph,
            **kwargs,
        )
        guard.raise_if_tripped()
        return result

    return bound


async def run_system(
    spec: SystemSpec,
    items: Sequence[QAItem],
    *,
    graph: ProceduralGraph | None,
    graph_version: int | None,
    guard: SpendGuard,
    models: Models,
    tools: Mapping[str, Any] | None,
    max_steps: int,
    run: RunFn,
    echo: bool = True,
) -> SystemRun:
    """Answer every item with one system; its model calls go to its own ledger."""
    ledger = guard.ledger(f"{spec.key} ({spec.label})")
    bound = _bound_run(
        run,
        guard=guard,
        llm=guard.wrap(models.solver, ledger),
        guidance_llm=guard.wrap(models.guidance, ledger),
        tools=tools,
        full_graph=spec.full_graph,
    )
    # Each system starts cold: the guidance cache is part of the design being
    # measured, but hits carried over from another system would not be.
    procedural_guidance.clear_caches()
    system_run = SystemRun(spec=spec, ledger_label=ledger.label, graph_version=graph_version)
    use_graph = spec.guidance != "none" and graph is not None
    for position, item in enumerate(items, start=1):
        result = await bound(
            item.question,
            graph_name=graph.name if use_graph else None,
            graph=graph if use_graph else None,
            graph_version=graph_version if use_graph else None,
            guidance=spec.guidance,
            max_steps=max_steps,
        )
        outcome = outcome_from_result(item, result)
        system_run.outcomes.append(outcome)
        if echo:
            print(
                f"   {spec.key} {position}/{len(items)} · F1 {outcome.f1:.2f} · "
                f"{outcome.steps} steps · {outcome.llm_calls} calls · {outcome.stopped}"
            )
    return system_run


def _evolution_printer(echo: bool) -> Callable[[dict], None]:
    def on_progress(event: dict) -> None:
        if not echo:
            return
        stage, round_number = event.get("stage"), event.get("round")
        if "processed" in event:
            if event["processed"] == event["total"]:
                print(
                    f"   … round {round_number} {stage}: {event['total']} rollouts · "
                    f"{event.get('llm_calls', 0)} LLM calls so far"
                )
        elif stage == "baseline":
            print(f"   baseline val score {event['score']:.3f}")
        elif stage == "accepted":
            print(
                f"   ✅ round {round_number}: accepted v{event['version']} "
                f"({event['previous_score']:.3f} → {event['score']:.3f}; {event['change']})"
            )
        elif stage == "rejected":
            score = f" at {event['score']:.3f}" if event.get("score") is not None else ""
            print(f"   ✖ round {round_number}: rejected ({event['reason']}{score})")

    return on_progress


async def _evolve(
    result: BenchResult,
    *,
    rounds: int,
    batch_size: int,
    guidance: str,
    guard: SpendGuard,
    models: Models,
    tools: Mapping[str, Any] | None,
    run: RunFn,
    evolve_fn: RunFn,
    load_evolved: RunFn,
    echo: bool,
    reserve: int = 0,
) -> None:
    """Algorithm 1 on the train split, gated on val, saved under a separate name.

    ``reserve`` is the call budget the evolved systems still need after this
    phase. Evolution gets what is left of the harness cap minus that reserve,
    so a long search cannot starve the rows it exists to produce.
    """
    if models.refiner is None:
        # evolve() would fall back to get_chat_llm(): an unmetered, uncapped model.
        raise ValueError("evolution needs Models.refiner; make_models(evolve=True) builds it")
    prior = result.prior
    name = evolved_name(prior)
    train = [item.as_evolution_item() for item in result.dataset.split("train")]
    val = [item.as_evolution_item() for item in result.dataset.split("val")]
    ledger = guard.ledger("evolution (rollouts, guidance and refiner)")
    remaining = guard.remaining_calls()
    if remaining is not None:
        remaining -= max(0, int(reserve))
    procedural_guidance.clear_caches()
    if echo:
        print(
            f"🧬 Evolving {prior.name} as {name}: {rounds} round(s), batch "
            f"{min(batch_size, len(train))} of {len(train)} train, gate on {len(val)} val …"
        )
    try:
        report = await evolve_fn(
            name,
            train=train,
            val=val,
            rounds=rounds,
            batch_size=min(batch_size, len(train)),
            mode="static",
            metric=result.metric,
            guidance=guidance,
            max_llm_calls=max(
                1, remaining if remaining is not None else get_settings().evolution_max_llm_calls
            ),
            on_progress=_evolution_printer(echo),
            run=_bound_run(
                run,
                guard=guard,
                llm=guard.wrap(models.solver, ledger),
                guidance_llm=guard.wrap(models.guidance, ledger),
                tools=tools,
            ),
            refiner_llm=guard.wrap(models.refiner, ledger),
            max_steps=result.max_steps,
            initial=prior,
        )
    except procedural_evolution.EvolutionBudgetTooSmall as e:
        # main() refuses such a cap before anything is spent (evolution_call_bound);
        # this keeps a caller that skipped that check from losing the paid rows
        # to a traceback: the run stops as capped, with what finished.
        source = (
            "EVOLUTION_MAX_LLM_CALLS"
            if remaining is None
            else f"--max-llm-calls {guard.max_calls:,} ({guard.calls:,} spent, "
            f"{max(0, int(reserve)):,} reserved for the evolved rows)"
        )
        raise BudgetExceeded(
            f"evolution refused to start: it needs {e.minimum:,} LLM calls for its baseline "
            f"and one full round, and {source} left it {e.max_llm_calls:,}"
        ) from e
    result.evolution = report
    # A cap that refused the refiner call ends evolve() cleanly with
    # stopped="budget"; the run as a whole is then incomplete, and says so.
    guard.raise_if_tripped()
    if report.get("final_version") is None:
        # Nothing was accepted: the evolved graph IS the prior (under its new name).
        evolved = prior.copy()
        evolved.name = name
        result.evolved_graph, result.evolved_version = evolved, None
        return
    loaded = await load_evolved(name)
    if loaded is None:
        raise IntegrityError(
            f"evolution reported {name} v{report['final_version']} saved, but it cannot be "
            "read back from Neo4j"
        )
    result.evolved_graph, meta = loaded
    result.evolved_version = meta.get("version")


def _check_integrity(result: BenchResult) -> None:
    """Refuse a system whose every run failed: its zeros would read as a finding."""
    for key, system_run in result.runs.items():
        outcomes = system_run.outcomes
        if outcomes and all(o.stopped == "error" for o in outcomes):
            raise IntegrityError(
                f"every {key} run ended in a model error (first: {outcomes[0].error}). Its "
                "EM/F1 of 0 would be a statement about the provider, not the system."
            )


async def execute(
    dataset: Dataset,
    *,
    systems: Sequence[str],
    prior: ProceduralGraph,
    guard: SpendGuard,
    models: Models,
    max_steps: int,
    tools: Mapping[str, Any] | None = None,
    evolve_rounds: int = 0,
    evolve_batch_size: int | None = None,
    evolve_guidance: str = "raw",
    metric: str = "f1",
    run: RunFn | None = None,
    evolve_fn: RunFn | None = None,
    load_evolved: RunFn | None = None,
    echo: bool = True,
) -> BenchResult:
    """Run every system on the test split (evolving first when asked); return the result.

    ``run`` / ``evolve_fn`` / ``load_evolved`` default to the real
    ``graph_agent.run_agent``, ``procedural_evolution.evolve`` and
    ``procedural_store.load_graph_with_meta``; tests inject fakes. A cost cap
    raises :class:`BudgetExceeded` carrying the partial result.
    """
    run = run or graph_agent.run_agent
    evolve_fn = evolve_fn or procedural_evolution.evolve
    load_evolved = load_evolved or procedural_store.load_graph_with_meta
    batch_size = evolve_batch_size or get_settings().evolution_default_batch_size
    result = BenchResult(dataset=dataset, prior=prior, max_steps=max_steps, metric=metric)
    result.guard = guard
    test_items = dataset.split("test")
    if not test_items:
        raise IntegrityError("the test split is empty; there is nothing to score")

    async def run_one(key: str, graph: ProceduralGraph | None, version: int | None) -> None:
        if echo:
            print(f"🔎 {key} — {SYSTEMS[key].label} ({len(test_items)} test questions) …")
        result.runs[key] = await run_system(
            SYSTEMS[key],
            test_items,
            graph=graph,
            graph_version=version,
            guard=guard,
            models=models,
            tools=tools,
            max_steps=max_steps,
            run=run,
            echo=echo,
        )

    try:
        for key in systems:
            spec = SYSTEMS[key]
            if spec.graph != "evolved":
                await run_one(key, prior if spec.graph == "prior" else None, None)
        if evolve_rounds:
            reserve = sum(
                len(test_items) * episode_calls_bound(SYSTEMS[key].guidance, max_steps)
                for key in systems
                if SYSTEMS[key].graph == "evolved"
            )
            await _evolve(
                result,
                rounds=evolve_rounds,
                batch_size=batch_size,
                guidance=evolve_guidance,
                guard=guard,
                models=models,
                tools=tools,
                run=run,
                evolve_fn=evolve_fn,
                load_evolved=load_evolved,
                echo=echo,
                reserve=reserve,
            )
            for key in systems:
                if SYSTEMS[key].graph == "evolved":
                    await run_one(key, result.evolved_graph, result.evolved_version)
    except BudgetExceeded as e:
        e.partial = result
        raise
    _check_integrity(result)
    return result


# ── Readiness (the harness never builds the graph it is scored on) ───────────
DEMO_ENTITIES_QUERY = "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name"
GRAPH_COUNTS_QUERY = """
MATCH (e:Entity) WITH count(e) AS entities
OPTIONAL MATCH (c:Chunk)
RETURN entities, count(c) AS chunks
"""


async def check_graph_ready(dataset: Dataset, *, allow_mixed: bool = False) -> dict:
    """Verify Neo4j holds the graph ``dataset`` is answered from; raise otherwise.

    Demo: every fixture entity must be present, and nothing else may be: extra
    entities or any source chunk mean other documents share the database, and
    ``search_entities`` / ``search_passages`` would answer from them (refused
    unless ``allow_mixed``). HotpotQA: every paragraph of the sample must be an
    ingested document (the ``--reuse-graph`` check of ``run_hotpotqa``).
    Returns what the report discloses: graph size, and for the demo whether
    the graph is ``mixed``.
    """
    from app.neo4j_driver import execute_query

    status: dict[str, Any] = {}
    if dataset.name == "demo":
        names = dataset.notes.get("entities") or []
        rows = await execute_query(DEMO_ENTITIES_QUERY, {"names": names}) or []
        present = {str(row.get("name")) for row in rows}
        missing = [name for name in names if name not in present]
        if missing:
            raise GraphNotReady(
                f"{len(missing)} of {len(names)} demo entities are not in Neo4j (e.g. "
                f"{missing[:3]}). Seed the demo graph first — free, no LLM: "
                "`cd backend && python -m scripts.seed_demo --clear`."
            )
    else:
        titles = dataset.notes.get("titles") or []
        documents = await run_hotpotqa.graph_documents()
        missing = [title for title in titles if title not in documents]
        if missing:
            raise GraphNotReady(
                f"{len(missing)} of {len(titles)} paragraphs of this HotpotQA sample are not in "
                f"Neo4j (e.g. {missing[:3]}). This harness never ingests: build the graph "
                "with `python -m benchmarks.public.run_hotpotqa` (which spends money on "
                "extraction) using the same --seed and a --questions of at least --n."
            )
    try:
        rows = await execute_query(GRAPH_COUNTS_QUERY) or []
        if rows:
            status = {k: int(rows[0].get(k) or 0) for k in ("entities", "chunks")}
    except Exception as e:  # noqa: BLE001 - disclosure only; the gate below then cannot run
        logger.info("Could not size the graph for the report: %s", e)
    if dataset.name == "demo" and status:
        fixture = len(dataset.notes.get("entities") or [])
        status["mixed"] = status["entities"] > fixture or status["chunks"] > 0
        if status["mixed"] and not allow_mixed:
            raise GraphNotReady(
                f"Neo4j holds {status['entities']} entities and {status['chunks']} source "
                f"chunks; the demo fixture is {fixture} entities and no chunks. Other "
                "documents share this database, so the Navigator's search tools would "
                "return their content and the numbers would not be about the demo set. "
                "Clear it with `cd backend && python -m scripts.seed_demo --clear` (free, no "
                "LLM; it deletes those documents from the knowledge graph, procedural "
                "memory is kept), or pass --allow-mixed-graph to run anyway, flagged in the "
                "report."
            )
    return status


# ── Estimate (the --dry-run path: zero model calls) ──────────────────────────
@dataclass
class PhaseEstimate:
    label: str
    usage: cost.Usage
    note: str = ""


@dataclass
class Plan:
    phases: list[PhaseEstimate]
    #: The per-call reasoning allowance already inside every phase's
    #: completion tokens (0 for a model that does not reason).
    reasoning_tokens_per_call: int = 0

    @property
    def usage(self) -> cost.Usage:
        total = cost.Usage()
        for phase in self.phases:
            total = total + phase.usage
        return total

    @property
    def calls(self) -> int:
        return self.usage.calls

    def usd(self, model: str) -> float | None:
        return cost.usd(self.usage, cost.resolve_price(model))


def args_reasoning_override(plan) -> bool:
    """Whether ``plan`` was estimated with a calibrated (non-default) allowance."""
    return plan.reasoning_tokens_per_call not in (0, REASONING_TOKENS_ALLOWANCE)


def reasoning_allowance(*models: str, override: int | None = None) -> int:
    """Per-call reasoning tokens to assume: the allowance if ANY of ``models``
    reasons (the model called and the one it is priced as can differ), else 0.

    ``override`` replaces the default :data:`REASONING_TOKENS_ALLOWANCE` with a
    value calibrated on a measured run (``--reasoning-allowance``). It only moves
    the up-front bound: the metered cap always counts the real reasoning tokens.
    """
    per_call = REASONING_TOKENS_ALLOWANCE if override is None else max(0, int(override))
    return per_call if any(is_reasoning_model(m) for m in models) else 0


def with_reasoning(usage: cost.Usage, per_call: int) -> cost.Usage:
    """``usage`` plus ``per_call`` hidden reasoning (output) tokens on every call."""
    extra = usage.calls * max(0, int(per_call))
    return cost.Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens + extra,
        calls=usage.calls,
        estimated_calls=usage.estimated_calls,
        reasoning_tokens=usage.reasoning_tokens + extra,
    )


def step_contexts(graph: ProceduralGraph, *, full_graph: bool, hops: int) -> tuple[str, str]:
    """``(first step, any later step)``: the longest guidance context each can carry.

    The first step is always localized on ``Start`` (the trajectory is empty),
    so its context is known exactly. A later step is localized on whatever the
    last action matched, and a step that matches no node (a parse failure, or
    an evolved graph missing a tool) falls back to the FULL graph. So the
    upper bound for a later step is the longer of the full graph and the
    largest local neighbourhood, which in practice is the full graph.
    """
    full = serialize_full(graph)
    if full_graph:
        return full, full
    local = [serialize_local(graph, node_id, hops) for node_id in graph.nodes]
    first = serialize_local(graph, START, hops) if START in graph.nodes else full
    return first, max([full, *local], key=len)


def estimate_rollout(
    question: str,
    graph: ProceduralGraph | None,
    *,
    guidance: str,
    full_graph: bool,
    max_steps: int,
    settings: Any,
) -> cost.Usage:
    """Upper-bound usage of one episode: every step taken, every observation at the cap.

    Prompt sizes are measured by rendering the real solver and guidance
    templates; only the completions are assumed. See :func:`step_contexts`
    for the guidance context each step is charged.
    """
    usage = cost.Usage()
    observation = int(settings.agent_observation_max_chars)
    window = int(settings.procedural_window)
    guided = graph is not None and guidance != "none"
    first, later = (
        step_contexts(graph, full_graph=full_graph, hops=int(settings.procedural_hops))
        if guided
        else ("", "")
    )
    generated = "x" * (DRY_RUN_GUIDANCE_COMPLETION_TOKENS * cost.CHARS_PER_TOKEN)
    for step in range(max_steps):
        context = first if step == 0 else later
        if guided and guidance == "generative":
            recent = min(step, window) * (
                procedural_guidance.WINDOW_OBSERVATION_CHARS + DRY_RUN_TURN_CHARS
            )
            prompt = procedural_guidance.GUIDANCE_PROMPT.format(
                task_description=graph.description or procedural_guidance._FALLBACK_TASK,
                graph_context_desc=(
                    procedural_guidance._FULL_CONTEXT_DESC
                    if full_graph
                    else procedural_guidance._LOCAL_CONTEXT_DESC
                ),
                graph_context=context,
                query=question,
                # format_window's placeholder; the steps themselves are ``recent``.
                recent_context=procedural_guidance.format_window([], window),
                graph_source="complete Procedural Graph" if full_graph else "localized subgraph",
            )
            usage = usage + cost.estimate_call(
                len(prompt) + recent, DRY_RUN_GUIDANCE_COMPLETION_TOKENS
            )
        injected = generated if guidance == "generative" else context
        base = len(
            graph_agent.build_prompt(
                question, [], guidance=injected if guided else "", raw_guidance=guidance == "raw"
            )
        )
        trace = step * (observation + DRY_RUN_TURN_CHARS)
        usage = usage + cost.estimate_call(base + trace, DRY_RUN_SOLVER_COMPLETION_TOKENS)
    return usage


def estimate_plan(
    dataset: Dataset,
    *,
    systems: Sequence[str],
    prior: ProceduralGraph,
    max_steps: int,
    evolve_rounds: int = 0,
    evolve_batch_size: int | None = None,
    evolve_guidance: str = "raw",
    metric: str = "f1",
    settings: Any = None,
    reasoning_tokens: int = 0,
) -> Plan:
    """Upper-bound calls and tokens for the whole run. Pure: no model, no database.

    ``reasoning_tokens`` is added to every call's completion (see
    :data:`REASONING_TOKENS_ALLOWANCE` and :func:`reasoning_allowance`). Hidden
    reasoning is never returned into the conversation, so it adds nothing to
    any later prompt.
    """
    settings = settings or get_settings()
    test = dataset.split("test")
    phases = []

    def episodes(items: Sequence[Any], graph, guidance: str, full: bool = False) -> cost.Usage:
        total = cost.Usage()
        for item in items:
            question = item.question if isinstance(item, QAItem) else str(item["question"])
            total = total + estimate_rollout(
                question,
                graph,
                guidance=guidance,
                full_graph=full,
                max_steps=max_steps,
                settings=settings,
            )
        return total

    for key in systems:
        spec = SYSTEMS[key]
        graph = None if spec.graph == "none" else prior
        note = (
            "the prior's size stands in for the evolved graph, which can be larger; the "
            "metered caps still bind"
            if spec.graph == "evolved"
            else ""
        )
        phases.append(
            PhaseEstimate(
                f"{key} ({spec.label})",
                episodes(test, graph, spec.guidance, spec.full_graph),
                note,
            )
        )

    if evolve_rounds:
        train = [i.as_evolution_item() for i in dataset.split("train")]
        val = dataset.split("val")
        batch = min(evolve_batch_size or settings.evolution_default_batch_size, len(train))
        trace_chars = int(settings.evolution_trajectory_max_chars)
        refiner_base = len(
            procedural_evolution.build_refiner_prompt(
                prior, mode="static", metric=metric, rollouts=[], rejected=[], max_chars=0
            )
        )
        usage = episodes(val, prior, evolve_guidance)  # S0 on D_val
        for k in range(1, evolve_rounds + 1):
            batch_items = procedural_evolution.training_batch(train, k, batch)
            usage = usage + episodes(batch_items, prior, evolve_guidance)
            rejected = min(k - 1, procedural_evolution.MAX_REJECTIONS_SHOWN)
            refiner_chars = refiner_base + trace_chars + rejected * (
                procedural_evolution.REJECTED_EDITS_CHARS + DRY_RUN_TURN_CHARS
            )
            usage = usage + cost.estimate_call(refiner_chars, DRY_RUN_REFINER_COMPLETION_TOKENS)
            usage = usage + episodes(val, prior, evolve_guidance)
        phases.append(
            PhaseEstimate(
                f"evolution ({evolve_rounds} round(s), {evolve_guidance} guidance)",
                usage,
                f"baseline on val, then per round: {batch} train rollouts, 1 refiner call, "
                f"{len(val)} val rollouts",
            )
        )
    per_call = max(0, int(reasoning_tokens))
    if per_call:
        phases = [
            PhaseEstimate(phase.label, with_reasoning(phase.usage, per_call), phase.note)
            for phase in phases
        ]
    return Plan(phases, reasoning_tokens_per_call=per_call)


@dataclass(frozen=True, slots=True)
class EvolutionFloor:
    """The smallest ``--max-llm-calls`` an ``--evolve`` run is sure to get through.

    Evolution runs after every non-evolved system and before the evolved rows.
    It gets the cap minus what the first spent minus the reserve of the second
    (see :func:`_evolve`), and ``procedural_evolution.evolve`` refuses a budget
    below its baseline plus one whole round (``evolution``). Charging the
    systems before it at their worst case (``before``), a cap of ``total``
    always leaves evolution that much, so it can never refuse after those
    systems were paid for.
    """

    before: int
    evolution: int
    reserve: int
    max_steps: int

    @property
    def total(self) -> int:
        return self.before + self.evolution + self.reserve

    def refusal(self, max_calls: int) -> str | None:
        """Why ``max_calls`` is too small for this run, or ``None`` when it is enough."""
        if int(max_calls) >= self.total:
            return None
        return (
            f"--max-llm-calls {int(max_calls):,} is below the {self.total:,} calls this --evolve "
            f"run can need: {self.before:,} for the systems that run before evolution, "
            f"{self.evolution:,} for evolution's baseline and one full round, and "
            f"{self.reserve:,} reserved for the evolved rows (every episode at --max-steps "
            f"{self.max_steps}). Raise --max-llm-calls to at least {self.total:,}, or lower "
            "--max-steps, --evolve-batch-size or the --systems that run before evolution."
        )


def evolution_floor(
    dataset: Dataset,
    *,
    systems: Sequence[str],
    max_steps: int,
    evolve_batch_size: int | None = None,
    evolve_guidance: str = "raw",
    settings: Any = None,
) -> EvolutionFloor:
    """The :class:`EvolutionFloor` of an ``--evolve`` run. Pure: no model, no database."""
    settings = settings or get_settings()
    test = len(dataset.split("test"))
    train = len(dataset.split("train"))
    evolved = [key for key in systems if SYSTEMS[key].graph == "evolved"]

    def bound(keys: Sequence[str]) -> int:
        return sum(test * episode_calls_bound(SYSTEMS[key].guidance, max_steps) for key in keys)

    batch = evolve_batch_size or settings.evolution_default_batch_size
    return EvolutionFloor(
        before=bound([key for key in systems if key not in evolved]),
        evolution=procedural_evolution.minimum_llm_calls(
            train_size=train,
            val_size=len(dataset.split("val")),
            batch_size=min(batch, train),
            per_rollout=procedural_evolution.rollout_worst_case(max_steps, evolve_guidance),
        ),
        reserve=bound(evolved),
        max_steps=int(max_steps),
    )


def dry_run_lines(
    plan: Plan,
    dataset: Dataset,
    *,
    systems: Sequence[str],
    model: str,
    max_steps: int,
    max_usd: float,
    evolve_rounds: int = 0,
    settings: Any = None,
) -> list[str]:
    """Estimate the bill of a run that has NOT happened. Zero model calls."""
    settings = settings or get_settings()
    price = cost.resolve_price(model)
    sizes = dataset.sizes()
    lines = [
        "DRY RUN — no model was called, no token was spent, Neo4j was not touched.",
        f"  Plan: dataset {dataset.name} (test {sizes['test']} · val {sizes['val']} · train "
        f"{sizes['train']}) · systems {', '.join(systems)} · max_steps {max_steps} · "
        f"evolution {'off' if not evolve_rounds else f'{evolve_rounds} round(s)'}",
        f"  Every figure is an UPPER BOUND: each episode is assumed to use all {max_steps} "
        f"steps, every observation to hit the {settings.agent_observation_max_chars:,}-char "
        "cap, and every guided step after the first to carry the full graph (the fallback "
        "when a step matches no node). Prompt sizes are measured from the real solver, "
        "guidance and refiner templates; "
        f"completions are assumed ({DRY_RUN_SOLVER_COMPLETION_TOKENS} solver / "
        f"{DRY_RUN_GUIDANCE_COMPLETION_TOKENS} guidance / {DRY_RUN_REFINER_COMPLETION_TOKENS} "
        "refiner tokens).",
    ]
    if plan.reasoning_tokens_per_call:
        effort = reasoning_effort_for(
            model, getattr(settings, "openai_reasoning_effort", "minimal")
        )
        lines.append(
            f"  {model} is a REASONING model: its hidden reasoning tokens are billed as output, "
            f"so every call above also carries a {plan.reasoning_tokens_per_call:,}-token "
            "reasoning allowance ("
            + ("calibrated with --reasoning-allowance" if args_reasoning_override(plan) else
               "REASONING_TOKENS_ALLOWANCE")
            + "). That allowance is an ASSUMPTION, not a bound — nothing caps reasoning — "
            "to be calibrated on a measured run (its report prints the measured mean per call). "
            f"The metered caps count the real reasoning tokens. Effort sent: '{effort}'."
        )
        if effort != "minimal":
            lines.append(
                f"  ⚠️  Effort '{effort}' reasons longer than 'minimal': the allowance, and so "
                "this estimate, is probably LOW. Only the metered --max-usd cap is reliable."
            )
    for phase in plan.phases:
        amount = cost.usd(phase.usage, price)
        note = f" ({phase.note})" if phase.note else ""
        lines.append(
            f"  - {phase.label}: ≤ {phase.usage.calls:,} LLM calls, ~{phase.usage.prompt_tokens:,} "
            f"prompt + {phase.usage.completion_tokens:,} completion tokens, "
            f"{cost.format_usd(amount)}{note}"
        )
    total = plan.usage
    amount = plan.usd(model)
    # Reasoning is the one term that is assumed, not bounded: say how much of it is in.
    assumed = (
        f" (incl. {total.reasoning_tokens:,} ASSUMED reasoning tokens)"
        if total.reasoning_tokens
        else ""
    )
    if amount is None:
        lines.append(
            f"  TOTAL (upper bound): ≤ {total.calls:,} LLM calls, ~{total.total_tokens:,} tokens"
            f"{assumed}. Cost UNKNOWN — no price on file for {model!r}, so --max-usd cannot be enforced; "
            "only --max-llm-calls can stop this run."
        )
    else:
        lines.append(
            f"  TOTAL (upper bound): ≤ {total.calls:,} LLM calls, ~{total.total_tokens:,} tokens"
            f"{assumed}, {cost.format_usd(amount)} on {model} from prices hand-recorded on "
            f"{cost.price_checked_on(model)} — an ESTIMATE, not a quote ({cost.PRICING_URL})."
        )
        if amount > max_usd:
            lines.append(
                f"  ⛔ This exceeds --max-usd ({cost.format_usd(max_usd)}). The real run would "
                "refuse to start: lower --max-steps, run fewer --systems, or raise --max-usd "
                "deliberately."
            )
    lines.append(
        "  Embeddings are not included: the tools embed each search query, and semantic "
        "localization embeds unmatched steps. With EMBEDDING_PROVIDER=fastembed they run "
        "locally and cost nothing."
    )
    return lines


# ── Reporting ────────────────────────────────────────────────────────────────
def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def table_lines(result: BenchResult) -> list[str]:
    """The Table 9-style comparison, per-question means over the test split."""
    lines = [
        "| System | Configuration | EM | F1 | Steps | LLM calls | Guidance calls | "
        "Tokens in / out | Parse fail. | Latency (s) | Answered |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, system_run in result.runs.items():
        agg = system_run.agg
        mark = "≈" if agg["estimated"] else ""
        lines.append(
            f"| `{key}` | {system_run.spec.label} | {_pct(agg['em'])} | {_pct(agg['f1'])} | "
            f"{agg['steps']:.2f} | {agg['llm_calls']:.2f} | {agg['guidance_llm_calls']:.2f} | "
            f"{mark}{agg['input_tokens']:,.0f} / {mark}{agg['output_tokens']:,.0f} | "
            f"{agg['parse_failures']:.3f} | {agg['latency_s']:.2f} | "
            f"{agg['answered'] * agg['n']:.0f}/{agg['n']} |"
        )
    return lines


def verdict_lines(result: BenchResult) -> list[str]:
    """Every comparison whose two systems ran, on EM and F1, under the floor."""
    lines = []
    for system, baseline, why in COMPARISONS:
        if system not in result.runs or baseline not in result.runs:
            continue
        a, b = result.runs[system].agg, result.runs[baseline].agg
        lines.append(f"{system} vs {baseline} — {why}:")
        for label, key in (("EM", "em"), ("F1", "f1")):
            lines.append(
                "  "
                + verdict(label, key, b, a, baseline_label=baseline, system_label=system)
            )
        if b["tokens"]:
            change = (a["tokens"] - b["tokens"]) / b["tokens"]
            lines.append(
                f"  Tokens per question: {a['tokens']:,.0f} vs {b['tokens']:,.0f} "
                f"({change:+.1%}); LLM calls per question {a['llm_calls']:.2f} vs "
                f"{b['llm_calls']:.2f}. Measured on the same {a['n']} questions."
            )
    return lines


def localization_lines(result: BenchResult) -> list[str]:
    """How often each localization method fired, and the context it injected."""
    lines = []
    for key, system_run in result.runs.items():
        if system_run.spec.guidance == "none":
            continue
        agg = system_run.agg
        counts = agg["localization"]
        total = sum(counts.values())
        shown = ", ".join(f"{m} {counts[m]}" for m in sorted(counts)) or "none recorded"
        lines.append(
            f"{key}: {total} localized steps ({shown}); guidance context "
            f"{agg['context_chars']:,.0f} chars per question."
        )
    return lines


def evolution_lines(result: BenchResult) -> list[str]:
    report = result.evolution
    if not report:
        return []
    floor = report.get("effect_floor") or 0.0
    lines = [
        f"Graph `{report['graph']}` evolved from `{result.prior.name}` in {report['mode']} mode "
        f"({report['guidance']} guidance, gate metric {report['metric']}): "
        f"{report['rounds_run']} of {report['rounds_requested']} round(s) decided, "
        f"{report['accepted_rounds']} accepted, stopped: {report['stopped']}.",
        f"Validation score {_score(report.get('baseline_score'))} → "
        f"{_score(report.get('final_score'))}; final version "
        f"{report.get('final_version') or 'none (nothing accepted: the evolved systems ran the prior)'}; "
        f"{report.get('llm_calls', 0)} LLM calls.",
        f"The gate's resolution is one val question = {floor * 100:.1f} pp: an accept or "
        "reject is a step in a search, not a significance test (the paper says the same of "
        "its own gate).",
    ]
    if result.evolved_graph is not None:
        change = summarize_diff(graph_diff(result.prior, result.evolved_graph))
        lines.append(f"Evolved graph vs the expert prior: {change}.")
    return lines


def _score(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def evolution_table(result: BenchResult) -> list[str]:
    report = result.evolution
    if not report or not report.get("rounds"):
        return []
    lines = [
        "| Round | Train mean | Candidate (val) | Accepted | Reason | Change |",
        "| ---: | ---: | ---: | :---: | --- | --- |",
    ]
    for entry in report["rounds"]:
        change = summarize_diff(entry["diff"]) if entry.get("diff") else "—"
        if entry.get("diagnostics"):
            change = f"{change}; {entry['diagnostics'][0]}"
        lines.append(
            f"| {entry['round']} | {_score(entry.get('train_mean'))} | "
            f"{_score(entry.get('candidate_score'))} | {'yes' if entry.get('accepted') else 'no'} | "
            f"{entry.get('reason')} | {change.replace('|', '/')} |"
        )
    return lines


def cost_lines(result: BenchResult) -> list[str]:
    guard = result.guard
    lines = ["Cost (ESTIMATED — see benchmarks/public/cost.py; prices are hand-recorded):"]
    if guard is None or not guard.ledgers:
        return lines + ["  No model call was metered."]
    for ledger in guard.ledgers:
        lines += [f"  {line}" for line in ledger.lines()]
    amount = guard.usd()
    lines.append(
        f"  TOTAL: {guard.calls:,} LLM calls, {guard.usage.total_tokens:,} tokens, "
        + (
            f"{cost.format_usd(amount)} (estimated from {cost.price_checked_on(guard.model)} "
            "prices)."
            if amount is not None
            else "not priced (no price on file for this model)."
        )
    )
    usage = guard.usage
    if usage.reasoning_tokens:
        measured = usage.calls - usage.estimated_calls
        mean = usage.reasoning_tokens / measured if measured else 0.0
        lines.append(
            f"  Reasoning: {usage.reasoning_tokens:,} hidden reasoning tokens (billed as output, "
            f"included above), a mean of {mean:,.0f} per measured call against the dry run's "
            f"{REASONING_TOKENS_ALLOWANCE:,}-token REASONING_TOKENS_ALLOWANCE — calibrate the "
            "allowance from this before estimating a larger run."
        )
    return lines


def threats_lines(result: BenchResult) -> list[str]:
    """Threats to validity. Generated, so the numbers in it cannot drift."""
    n = len(result.dataset.split("test"))
    floor = run_hotpotqa.effect_floor(n)
    items = [
        f"SAMPLE SIZE. {n} test questions: one question is {floor * 100:.1f} pp of EM, so "
        "every gap below that is refused above, and a gap of one or two questions is a "
        "direction, not a finding. No statistical test is run.",
        "NOT A REPRODUCTION. The paper evaluates its own HotpotQA agent with Gemini models on "
        "1,000 test questions; this is Synapse's Navigator over a knowledge graph, with the "
        "configured model, on a small split. The numbers are comparable to each other, not to "
        "the paper's Table 9.",
        "LOCALIZATION BY TOOL NAME. The active node is matched from the last action's tool "
        "name (then semantically). Several procedure steps behind one tool cannot be told "
        "apart; the localization counts above show how often each method fired.",
        (
            f"NO TEMPERATURE CONTROL. `{result.model}` is a reasoning model: it accepts only "
            "its default sampling temperature, so none is sent (agent_temperature is ignored) "
            f"and it reasons at effort '{result.reasoning_effort}'. Answers can move between "
            "runs more than at temperature 0; re-run before reading a one-question gap."
            if result.reasoning_effort
            else "TEMPERATURE 0 IS NOT DETERMINISM. Provider-side nondeterminism can move a "
            "question between runs; re-run before reading a one-question gap."
        ),
        "ONE MODEL. Solver, guidance and refiner are the same configured model. Tokens come "
        "from the provider's usage metadata; a ≈ marks a system where some were estimated "
        "at chars/4.",
    ]
    if result.dataset.name == "demo":
        items.insert(
            1,
            "SELF-AUTHORED QUESTIONS. The 30 demo questions were written by this project "
            "against its own fixture. `verify_demo_qa` proves each answer is derivable from "
            "facts the tools show; it says nothing about difficulty or bias. HotpotQA "
            "(`--dataset hotpotqa`) is the external check.",
        )
        chunks = result.graph_status.get("chunks")
        entities = result.graph_status.get("entities")
        items.append(
            "NO SOURCE PASSAGES ON THE DEMO GRAPH. Seeding writes entities and relations only, "
            "so `read_sources` and `search_passages` find nothing here"
            + (f" (this database holds {chunks} chunks from other documents)" if chunks else "")
            + ". Prior edges that recommend them cost steps on this dataset and not on "
            "HotpotQA."
        )
        fixture = len(result.dataset.notes.get("entities") or [])
        if entities and fixture and entities > fixture:
            items.append(
                f"EXTRA ENTITIES. Neo4j holds {entities} entities, the demo fixture {fixture}: "
                "other documents share the database, and their entities can surface in "
                "search results."
            )
    if result.evolution:
        items.append(
            "THE EVOLUTION GATE IS NOISY. It compares two means over "
            f"{len(result.dataset.split('val'))} val questions; an accepted round can be "
            "noise, and the evolved rows are scored on the same test split as the prior."
        )
    return items


def per_question_lines(result: BenchResult) -> list[str]:
    keys = list(result.runs)
    if not keys:
        return []
    lines = [
        "| # | Type | Hops | Question | Gold | " + " | ".join(f"`{k}`" for k in keys) + " |",
        "| --- | --- | ---: | --- | --- | " + " | ".join("---:" for _ in keys) + " |",
    ]
    first = result.runs[keys[0]].outcomes
    for index, outcome in enumerate(first):
        item = outcome.item
        cells = []
        for key in keys:
            outcomes = result.runs[key].outcomes
            cells.append(f"{outcomes[index].f1:.2f}" if index < len(outcomes) else "—")
        lines.append(
            f"| {item.id} | {item.qtype} | {item.hops} | {item.question.replace('|', '/')} | "
            f"{item.answer} | " + " | ".join(cells) + " |"
        )
    return lines


def provenance_lines(provenance: Mapping[str, Any]) -> list[str]:
    return [f"- **{key}:** {value}" for key, value in provenance.items()]


def results_markdown(result: BenchResult) -> str:
    n = len(result.dataset.split("test"))
    floor = run_hotpotqa.effect_floor(n)
    lines = [
        "# Procedural Graphs on the GraphRAG Navigator",
        "",
        "Does procedural memory make Synapse's Navigator answer better, or only cost more? "
        "Every system is the same ReAct agent over the same knowledge-graph tools and the "
        "same model; only the Procedural Graph guidance differs (Lu, Chen, Wu, Arık, "
        "arXiv:2609.09153). Written by `backend/benchmarks/procedural/run_procedural.py`; "
        "never edited by hand, and gitignored because it describes one run.",
        "",
        "## Provenance",
        "",
    ]
    lines += provenance_lines(result.provenance)
    lines += [
        "",
        f"## Results — {n} test questions, per-question means",
        "",
        f"**Effect floor: one question = {floor * 100:.1f} points.** A gap below it is not "
        "reported as a difference.",
        "",
    ]
    lines += table_lines(result)
    lines += [
        "",
        "EM and F1 are percentages (SQuAD/HotpotQA normalization). Steps, calls, tokens, "
        "parse failures and latency are means per question. ≈ marks tokens partly estimated "
        "at chars/4 because the provider reported no usage.",
        "",
        "## What the numbers support",
        "",
    ]
    lines += markdown_lines(verdict_lines(result)) or ["- No comparable pair ran."]
    lines += ["", "## Localization", ""]
    lines += [f"- {line}" for line in localization_lines(result)] or ["- No guided system ran."]
    if result.evolution:
        lines += ["", "## Evolution (Algorithm 1)", ""]
        lines += [f"- {line}" for line in evolution_lines(result)]
        lines += [""] + evolution_table(result)
    lines += ["", "## Per question (F1)", ""]
    lines += per_question_lines(result)
    lines += ["", "## Cost", ""]
    lines += markdown_lines(cost_lines(result))
    lines += ["", "## Threats to validity", ""]
    lines += [f"- {item}" for item in threats_lines(result)]
    lines += [
        "",
        "---",
        "",
        "Generated by `backend/benchmarks/procedural/run_procedural.py` — Synapse, © 2026 "
        "Ahmed Maaloul, PolyForm-Noncommercial-1.0.0.",
        "",
    ]
    return "\n".join(lines)


def print_report(result: BenchResult, *, incomplete: str | None = None) -> None:
    print()
    print("=" * 78)
    if incomplete:
        print(f"INCOMPLETE — {incomplete}. Nothing is written for a partial run.")
    print(f"Procedural Graphs benchmark — {result.dataset.name}, test N={len(result.dataset.split('test'))}")
    print("=" * 78)
    for line in table_lines(result):
        print(line)
    print()
    for line in verdict_lines(result):
        print(line)
    for line in localization_lines(result):
        print(line)
    for line in evolution_lines(result):
        print(line)
    print()
    for line in cost_lines(result):
        print(line)


def git_commit() -> str:
    """Short HEAD, marked when the working tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=HERE, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head} (uncommitted changes)" if dirty else head


def reproduce_command(args: argparse.Namespace) -> str:
    parts = ["python -m benchmarks.procedural.run_procedural", f"--dataset {args.dataset}"]
    if args.dataset == "hotpotqa":
        parts += [f"--n {args.n or DEFAULT_N}", "--reuse-graph"]
    parts.append(f"--seed {args.seed}")
    if args.systems:
        parts.append(f"--systems {args.systems}")
    if getattr(args, "allow_mixed_graph", False):
        parts.append("--allow-mixed-graph")
    if args.max_steps:
        parts.append(f"--max-steps {args.max_steps}")
    if args.evolve:
        parts += [
            f"--evolve {args.evolve}",
            f"--evolve-guidance {args.evolve_guidance}",
            f"--metric {args.metric}",
        ]
        if args.evolve_batch_size:
            parts.append(f"--evolve-batch-size {args.evolve_batch_size}")
    return " ".join(parts)


def build_provenance(
    result: BenchResult,
    *,
    args: argparse.Namespace,
    settings: Any,
    model: str,
    commit: str,
    date: str,
    priced_as: str | None = None,
) -> dict:
    sizes = result.dataset.sizes()
    prior = result.prior
    status = result.graph_status
    return {
        "Date": date,
        "Commit": commit,
        "Dataset": f"{result.dataset.name} ({result.dataset.source}); splits train "
        f"{sizes['train']} / val {sizes['val']} / test {sizes['test']}; seed {args.seed}",
        "Graph in Neo4j": (
            f"{status.get('entities', '?')} entities, {status.get('chunks', '?')} chunks"
            + (
                " — MIXED: the database holds content beyond the demo fixture "
                "(run with --allow-mixed-graph)"
                if status.get("mixed")
                else ""
            )
        ),
        "Model": f"`{model}` (provider `{settings.llm_provider}`, "
        + (
            "reasoning model: no temperature sent, reasoning effort "
            f"`{reasoning_effort_for(model, settings.openai_reasoning_effort)}`"
            if is_reasoning_model(model)
            else f"temperature {settings.agent_temperature}"
        )
        + f"); embeddings `{settings.embedding_provider}`"
        + (f"; priced as `{priced_as}`" if priced_as and priced_as != model else ""),
        "Navigator": f"max_steps {result.max_steps}, observation cap "
        f"{settings.agent_observation_max_chars} chars",
        "Guidance": f"hops {settings.procedural_hops}, window {settings.procedural_window}, "
        f"semantic threshold {settings.procedural_semantic_threshold}, cache size "
        f"{settings.procedural_guidance_cache_size} (cleared between systems)",
        "Expert prior": f"`{prior.name}` from the bundled JSON ({len(prior.nodes)} nodes, "
        f"{len(prior.edges)} edges)",
        "Evolution": (
            "off"
            if not args.evolve
            else f"{args.evolve} round(s), {args.evolve_guidance} guidance, gate {args.metric}, "
            f"saved as `{evolved_name(prior)}`"
        ),
        "Prices": f"`{priced_as or model}` hand-recorded "
        f"{cost.price_checked_on(priced_as or model)} ({cost.PRICING_URL})",
        "Reproduce": f"`{reproduce_command(args)}`",
    }


# ── Entry point ──────────────────────────────────────────────────────────────
DEFAULT_SEED = hotpotqa.DEFAULT_SEED
DEFAULT_N = hotpotqa.DEFAULT_N


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.procedural.run_procedural",
        description=(
            "Procedural Graphs on Synapse's GraphRAG Navigator: no graph vs raw vs generative "
            "guidance (vs evolved). Spends real LLM calls — always --dry-run first."
        ),
    )
    parser.add_argument(
        "--dataset", choices=("demo", "hotpotqa"), default="demo",
        help="demo_qa.json over the seeded demo graph (default), or a HotpotQA sample",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help=f"HotpotQA only: sampled questions, split 40/30/30 (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--reuse-graph", action="store_true",
        help="HotpotQA only, and REQUIRED: score the corpus already ingested by run_hotpotqa. "
             "This harness never ingests.",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="HotpotQA only: path to hotpot_dev_distractor_v1.json (default: the cache)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="HotpotQA only: never reach the network; fail if the dataset is not cached",
    )
    parser.add_argument(
        "--allow-mixed-graph", action="store_true",
        help="demo only: run even when Neo4j holds more than the demo fixture (other "
             "documents' entities or chunks); the report is flagged MIXED",
    )
    parser.add_argument(
        "--qa", type=Path, default=DEMO_QA_PATH,
        help=f"demo only: the QA file (default: {DEMO_QA_PATH.name}); verified before use",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"HotpotQA sampling seed (default: {DEFAULT_SEED}); recorded for the demo set",
    )
    parser.add_argument(
        "--systems", default=None,
        help=f"comma-separated subset of {', '.join(SYSTEMS)} (default: the four base "
             "systems, plus the evolved ones with --evolve)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Navigator turns per question (default: AGENT_MAX_STEPS)",
    )
    parser.add_argument(
        "--evolve", type=int, default=0, metavar="ROUNDS",
        help="run Algorithm 1 for ROUNDS on the train split first and add pg_evolved_raw / "
             "pg_evolved_gen (saved as <prior>-evolved-bench; the default graph is untouched)",
    )
    parser.add_argument(
        "--evolve-batch-size", type=int, default=None,
        help="training questions per round (default: EVOLUTION_DEFAULT_BATCH_SIZE, capped at "
             "the train split)",
    )
    parser.add_argument(
        "--evolve-guidance", choices=("raw", "generative"), default="raw",
        help="guidance used by evolution rollouts (default: raw — half the calls)",
    )
    parser.add_argument(
        "--metric", choices=METRICS, default="f1",
        help="the evolution gate's metric (default: f1). The report shows EM and F1 either way.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="estimate calls and cost, then exit WITHOUT calling any model or database",
    )
    parser.add_argument(
        "--max-usd", type=float, default=DEFAULT_MAX_USD,
        help=f"hard stop on estimated spend, checked before every call (default: "
             f"{DEFAULT_MAX_USD})",
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=None,
        help="hard stop on LLM calls (default: the dry-run upper bound); the only cap that "
             "works for an unpriced model",
    )
    parser.add_argument(
        "--reasoning-allowance", type=int, default=None, metavar="TOKENS",
        help="per-call reasoning tokens the up-front bound assumes for a reasoning model "
             f"(default {REASONING_TOKENS_ALLOWANCE}); set it from a measured run. The metered "
             "--max-usd cap always counts the real reasoning tokens",
    )
    parser.add_argument(
        "--model", default="",
        help="price the --dry-run estimate as this model instead of the configured one. A "
             "real run is priced (and --max-usd enforced) as the configured model; --model "
             "stands in only when that one has no price on file",
    )
    parser.add_argument(
        "--export-splits", type=Path, default=None, metavar="DIR",
        help="write train/val/test.json as [{question, answer}] (the input of "
             "`synapse-graphrag evolve`) and exit; free",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_PATH,
        help=f"markdown report path (default: {RESULTS_PATH.name}, gitignored)",
    )
    parser.add_argument("--no-write", action="store_true", help="do not write the markdown report")
    return parser


def _validate_args(args: argparse.Namespace, settings: Any) -> tuple[list[str], int]:
    """Systems and max_steps, or ``ValueError`` with the reason."""
    if args.dataset == "demo" and args.n is not None:
        raise ValueError("--n applies to --dataset hotpotqa only (the demo set is fixed)")
    if args.evolve < 0 or args.evolve > 20:
        raise ValueError("--evolve must be between 0 and 20 rounds")
    max_steps = args.max_steps if args.max_steps is not None else settings.agent_max_steps
    if not 1 <= int(max_steps) <= 20:
        raise ValueError("--max-steps must be between 1 and 20")
    if args.max_llm_calls is not None and args.max_llm_calls < 1:
        raise ValueError("--max-llm-calls must be at least 1")
    return resolve_systems(args.systems, evolve=bool(args.evolve)), int(max_steps)


async def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    settings = get_settings()

    try:
        systems, max_steps = _validate_args(args, settings)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if args.dataset == "hotpotqa" and not args.reuse_graph:
        print(
            "❌ --dataset hotpotqa requires --reuse-graph. This harness never ingests: build "
            "the graph with `python -m benchmarks.public.run_hotpotqa` (which spends money on "
            "extraction), then score it here with the same --seed.",
            file=sys.stderr,
        )
        return 4

    try:
        if args.dataset == "demo":
            dataset = load_demo_dataset(args.qa)
        else:
            dataset = load_hotpotqa_dataset(
                args.n or DEFAULT_N,
                seed=args.seed,
                data_path=args.data,
                allow_download=not args.no_download,
            )
    except DatasetError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    sizes = dataset.sizes()
    print(
        f"📚 {dataset.name}: {len(dataset.items)} questions (train {sizes['train']} · val "
        f"{sizes['val']} · test {sizes['test']}), source {dataset.source}"
    )

    if args.export_splits:
        for path in export_splits(dataset, args.export_splits):
            print(f"📄 Wrote {path}")
        return 0

    try:
        # The BUNDLED prior, not the stored graph: a stored graph may have been
        # evolved or hand-edited, and a benchmark row must name a fixed input.
        prior = procedural_store.load_prior(settings.procedural_default_graph)
    except procedural_store.ProceduralGraphNotFound as e:
        print(
            f"❌ {e}: PROCEDURAL_DEFAULT_GRAPH must name a bundled expert prior "
            f"({', '.join(procedural_store.list_priors()) or 'none found'}).",
            file=sys.stderr,
        )
        return 1
    configured = run_hotpotqa.chat_model_name(settings)
    # A dry run sizes and prices the model it is asked about (--model, else the
    # configured one). A real run calls the configured model and is priced as
    # billing_model() says; the reasoning allowance applies if either reasons.
    if args.dry_run:
        model, note = args.model or configured, None
        called = model
    else:
        model, note = billing_model(args.model, configured)
        called = configured
    plan = estimate_plan(
        dataset,
        systems=systems,
        prior=prior,
        max_steps=max_steps,
        evolve_rounds=args.evolve,
        evolve_batch_size=args.evolve_batch_size,
        evolve_guidance=args.evolve_guidance,
        metric=args.metric,
        settings=settings,
        reasoning_tokens=reasoning_allowance(called, model, override=args.reasoning_allowance),
    )
    floor = (
        evolution_floor(
            dataset,
            systems=systems,
            max_steps=max_steps,
            evolve_batch_size=args.evolve_batch_size,
            evolve_guidance=args.evolve_guidance,
            settings=settings,
        )
        if args.evolve
        else None
    )

    # ── The free path. Nothing below this point may run before it returns. ──
    if args.dry_run:
        for line in dry_run_lines(
            plan,
            dataset,
            systems=systems,
            model=model,
            max_steps=max_steps,
            max_usd=args.max_usd,
            evolve_rounds=args.evolve,
            settings=settings,
        ):
            print(line)
        refusal = floor.refusal(args.max_llm_calls) if floor and args.max_llm_calls else None
        if refusal:
            print(f"  ⛔ {refusal} The real run would refuse to start.")
        print(f"\n  To run it for real:\n    cd backend && {reproduce_command(args)}")
        return 0

    if note:
        print(f"⚠️  {note}")
    projected = plan.usd(model)
    if projected is not None and projected > args.max_usd:
        print(
            f"❌ The upper-bound estimate {cost.format_usd(projected)} exceeds --max-usd "
            f"{cost.format_usd(args.max_usd)}. Lower --max-steps, run fewer --systems, or raise "
            "the cap deliberately (--dry-run shows the breakdown).",
            file=sys.stderr,
        )
        return 5
    max_calls = args.max_llm_calls or plan.calls
    # The default cap (the plan's bound) always passes; a lowered one may not, and
    # evolve() would refuse it only after the systems before it were paid for.
    refusal = floor.refusal(max_calls) if floor else None
    if refusal:
        print(f"❌ {refusal} Refusing before any call.", file=sys.stderr)
        return 5
    if projected is None:
        print(
            f"⚠️  No price on file for {model!r}: --max-usd cannot be enforced. The only cap is "
            f"--max-llm-calls ({max_calls:,})."
        )

    from app import neo4j_driver
    from app.services.graph_schema import ensure_schema
    from app.services.llm_provider import ProviderConfigError

    if not await neo4j_driver.verify_connectivity():
        print(
            f"❌ Cannot reach Neo4j at {settings.neo4j_uri}.\n"
            "   Start it first:  docker compose up -d neo4j",
            file=sys.stderr,
        )
        return 2

    guard = SpendGuard(model, max_usd=args.max_usd, max_calls=max_calls)
    try:
        await ensure_schema()
        status = await check_graph_ready(dataset, allow_mixed=args.allow_mixed_graph)
        if status.get("mixed"):
            print(
                f"⚠️  MIXED GRAPH: Neo4j holds {status['entities']} entities and "
                f"{status['chunks']} chunks beyond the demo fixture (--allow-mixed-graph); "
                "the report is flagged."
            )
        models = make_models(settings, evolve=bool(args.evolve))
        print(
            f"💸 Caps: {cost.format_usd(args.max_usd)} and {max_calls:,} LLM calls "
            f"(upper-bound estimate {cost.format_usd(projected)}, {plan.calls:,} calls"
            + (
                f", incl. an assumed {plan.reasoning_tokens_per_call:,} reasoning tokens per "
                "call"
                if plan.reasoning_tokens_per_call
                else ""
            )
            + ")."
        )
        result = await execute(
            dataset,
            systems=systems,
            prior=prior,
            guard=guard,
            models=models,
            tools=navigator_tools(),
            max_steps=max_steps,
            evolve_rounds=args.evolve,
            evolve_batch_size=args.evolve_batch_size,
            evolve_guidance=args.evolve_guidance,
            metric=args.metric,
        )
    except GraphNotReady as e:
        print(f"❌ {e}", file=sys.stderr)
        return 4
    except ProviderConfigError as e:
        print(f"❌ No chat model is configured: {e}", file=sys.stderr)
        return 6
    except BudgetExceeded as e:
        print(f"⛔ {e}", file=sys.stderr)
        if e.partial is not None and e.partial.runs:
            print_report(e.partial, incomplete=str(e))
        return 5
    except IntegrityError as e:
        print(f"❌ Refusing to report this run: {e}", file=sys.stderr)
        return 3
    finally:
        await neo4j_driver.close_driver()

    result.graph_status = status
    result.model = configured
    result.reasoning_effort = (
        reasoning_effort_for(configured, settings.openai_reasoning_effort)
        if is_reasoning_model(configured)
        else None
    )
    result.provenance = build_provenance(
        result,
        args=args,
        settings=settings,
        model=configured,
        priced_as=model,
        commit=git_commit(),
        date=datetime.now(UTC).date().isoformat(),
    )
    print_report(result)
    if not args.no_write:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(results_markdown(result), encoding="utf-8")
        except OSError as e:
            print(f"❌ Could not write {args.out}: {e}", file=sys.stderr)
            return 3
        print(f"📄 Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
