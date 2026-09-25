# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the runner: RETRIEVE → READ → SCORE, resumable and capped.

A :class:`LabRun` names a dataset, the arms, the token budgets, ``k``, a seed,
the reader model, a mode and a spend cap. Running it fills ``run_dir``:

  manifest.json    — everything needed to reproduce and audit the run: git SHA
                     + dirty flag, Synapse version, every arm's config hash,
                     the packer policy, dataset name/split/sha256/question ids/
                     seed, embedding + extraction + reader models, tokenizer,
                     price dates, the cap, timestamps, and estimate vs actual
                     per phase.
  contexts.jsonl   — one line per (arm, budget, question): the packed context
                     text and its sha256, tokens, units by kind and their spans,
                     and (HotpotQA) the gold paragraphs it credits — so the run
                     can be RE-SCORED without Neo4j and without a model.
                     Identical texts are stored once (later lines carry only
                     the sha256).
  requests.jsonl   — the reader requests, one per DISTINCT body, already in
                     the OpenAI Batch input format
                     (``custom_id = "<run>|<arm>|<budget>|<qid>"``).
  responses.jsonl  — one line per answered request: answer, usage (prompt,
                     completion, reasoning, cached tokens) and its $ estimate.
  rows.jsonl       — the scored per-question rows (for paginated views).
  leaderboard.json — ``metrics.leaderboard`` over the rows.
  report.md        — the human-readable summary.

PHASES
  1. RETRIEVE — every (arm, question) is retrieved once and packed at every
     budget. Free (no model), deterministic, read-only against Neo4j.
  2. READ     — ``mode="realtime"``: the official OpenAI client, metered
     against ``max_usd`` BEFORE each call (worst case reserved while in
     flight), so the run stops cleanly under the cap. ``mode="batch"``:
     requests are handed to the Batch backend (``app.lab.batch``) and the run
     parks in ``batch_submitted`` until :func:`resume`. Before either, the run
     is re-estimated on the MEASURED contexts and refused if the upper bound
     breaks the cap. ``mode="retrieve"`` skips this phase entirely ($0).
  3. SCORE    — ``metrics.leaderboard``: EM/F1 (or, retrieve-only, context
     tokens, units by kind, containment and HotpotQA paragraph recall).

RESUMABLE. Re-running the same ``run_dir`` skips every completed phase and
every stored (arm, budget, question) context and answered request; a crashed
append is repaired. A ``run_dir`` holding a DIFFERENT configuration is refused.

DATASETS
  • ``demo``     — ``benchmarks/procedural/demo_qa.json`` (split ``test`` by
                   default; ``all`` for every split).
  • ``qa-file``  — a user JSON / JSONL of ``{question, answer[, id]}``.
  • ``hotpotqa`` — the seeded HotpotQA dev sample of ``benchmarks/public``;
                   REFUSED unless every pooled paragraph is already ingested
                   as a document in Neo4j. Also scores strict/permissive gold
                   paragraph recall with the harness's own crediting rules.

The Lab never writes to the knowledge graph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.lab import metrics, reader
from app.lab.arms import ARMS, BaseArm, graph_fingerprint
from app.lab.estimate import IngestPlan, estimate_ingest, estimate_run
from app.lab.evidence import Evidence
from app.lab.packer import ORDERS, PACKER_POLICY, Packed, pack
from app.lab.tokens import tokenizer_label
from benchmarks.public import cost

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
LAB_RUNS_DIR = BACKEND_ROOT / "lab_runs"
DATASETS_DIR = LAB_RUNS_DIR / "datasets"
#: Names under ``LAB_RUNS_DIR`` that are not runs (compared case-insensitively —
#: macOS and Windows file systems are): the uploaded-QA-file store and the ingest
#: calibration file. A run named after either would overwrite it or pose as it.
RESERVED_RUN_IDS = frozenset({"datasets", "calibration.json"})
DEMO_QA_PATH = BACKEND_ROOT / "benchmarks" / "procedural" / "demo_qa.json"

MODES: tuple[str, ...] = ("realtime", "batch", "retrieve")
DATASETS: tuple[str, ...] = ("demo", "qa-file", "hotpotqa")
DEFAULT_K = 8
#: The most passages a passage-ranking arm retrieves to fill a capped budget when
#: ``passage_k`` is unset (see ``LabRun.passage_policy``).
PASSAGE_FILL_MAX = 256
#: First guess of a passage's size when sizing the pool that fills a budget; the
#: pool doubles when the guess was too generous (see ``_retrieve_filling``).
PASSAGE_FILL_GUESS_TOKENS = 64
DEFAULT_SEED = 20260924
DEFAULT_READER = "gpt-5-nano"
BATCH_ENDPOINT = "/v1/chat/completions"

MANIFEST = "manifest.json"
CONTEXTS = "contexts.jsonl"
REQUESTS = "requests.jsonl"
RESPONSES = "responses.jsonl"
ROWS = "rows.jsonl"
LEADERBOARD = "leaderboard.json"
REPORT = "report.md"

EventCallback = Callable[[dict], Awaitable[None] | None]


class LabError(RuntimeError):
    """A run that cannot proceed (bad config, missing graph, foreign run_dir …)."""


class GraphNotIngested(LabError):
    """The dataset's corpus is not in Neo4j, so graph arms would score another corpus."""


class DatasetChanged(LabError):
    """The dataset no longer holds the questions and answers a run started with."""


# ── Config ───────────────────────────────────────────────────────────────────
def budget_label(budget: int | None) -> str:
    return "default" if budget is None else str(int(budget))


def parse_budget(value: Any) -> int | None:
    """``None`` / ``"default"`` → uncapped; anything else a positive int."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "default"}):
        return None
    budget = int(value)
    if budget <= 0:
        raise LabError(f"a token budget must be positive, got {budget}")
    return budget


@dataclass
class DatasetSpec:
    """Which questions a run reads.

    ``split``: demo — train | val | test (default) | all; qa-file — keep only
    items whose ``split`` field matches (``None`` = all); hotpotqa — ignored
    (always the dev distractor set). ``n`` caps the item count (hotpotqa:
    the sample size, default 20). ``offset`` (hotpotqa) skips the first sampled
    questions so two runs with the same seed can be disjoint.
    """

    name: str = "demo"
    split: str | None = None
    n: int | None = None
    path: str | None = None
    seed: int | None = None
    offset: int = 0
    allow_download: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> DatasetSpec:
        if isinstance(data, str):
            return cls(name=data)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LabRun:
    """One Lab run's configuration. ``mode`` defaults to the free retrieve-only tier."""

    dataset: DatasetSpec
    arms: list[str]
    budgets: list[int | None]
    k: int = DEFAULT_K
    seed: int = DEFAULT_SEED
    reader_model: str = DEFAULT_READER
    mode: str = "retrieve"
    max_usd: float | None = None
    run_dir: Path | None = None
    run_id: str | None = None
    #: Packing placement: "score" or PathRAG's "ascending".
    order: str = "score"
    #: Passages for passage-ranking arms (bm25, dense, null_random, ppr). ``None``
    #: (default): as many as each capped budget holds — the pool grows until the
    #: largest capped budget is covered (at most ``PASSAGE_FILL_MAX``) — and ``k``
    #: at the default (uncapped) context. An int: exactly that many, every budget.
    passage_k: int | None = None
    #: Reasoning tokens allowed per reader call (``None`` = reader.REASONING_ALLOWANCE).
    reasoning_allowance: int | None = None
    max_concurrency: int = 4
    #: Measured ingest spend of the corpus (amortized cost-of-pass); ``None`` = unknown.
    ingest_usd: float | None = None
    #: Ingest plan, shown in the estimate only (the runner never ingests).
    ingest_plan: IngestPlan | None = None
    #: (arm, budget) cells beyond the ``arms × budgets`` grid — e.g. each graph
    #: arm once at its default context (``[("synapse_d", None), …]``).
    extra_cells: list[tuple[str, int | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.dataset, dict | str):
            self.dataset = DatasetSpec.from_dict(self.dataset)
        if isinstance(self.ingest_plan, dict):
            self.ingest_plan = IngestPlan(**self.ingest_plan)
        self.arms = list(dict.fromkeys(str(a) for a in self.arms))
        self.budgets = list(dict.fromkeys(parse_budget(b) for b in (self.budgets or [None])))
        extra = []
        for cell in self.extra_cells or []:
            arm, budget = (cell["arm"], cell.get("budget")) if isinstance(cell, dict) else cell
            extra.append((str(arm), parse_budget(budget)))
        self.extra_cells = list(dict.fromkeys(extra))
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)

    def cells(self) -> list[tuple[str, int | None]]:
        """Every (arm, budget) cell, in order: the grid (arm-major), then the extras."""
        grid = [(a, b) for a in self.arms for b in self.budgets]
        return list(dict.fromkeys([*grid, *self.extra_cells]))

    def budgets_for(self, arm: str) -> list[int | None]:
        return [b for a, b in self.cells() if a == arm]

    def validate(self) -> None:
        if self.mode not in MODES:
            raise LabError(f"unknown mode {self.mode!r} (expected one of {', '.join(MODES)})")
        if self.dataset.name not in DATASETS:
            raise LabError(
                f"unknown dataset {self.dataset.name!r} (expected one of {', '.join(DATASETS)})"
            )
        if not self.arms:
            raise LabError("pick at least one arm")
        unknown = [a for a in self.arms if a not in ARMS]
        if unknown:
            raise LabError(f"unknown arm(s): {', '.join(unknown)} (known: {', '.join(ARMS)})")
        if self.order not in ORDERS:
            raise LabError(f"unknown packing order {self.order!r}")
        if self.k <= 0 or (self.passage_k is not None and self.passage_k <= 0):
            raise LabError("k must be positive")
        if self.max_usd is not None and self.max_usd < 0:
            raise LabError("max_usd must be >= 0")
        if self.run_id and "|" in self.run_id:
            raise LabError("run_id must not contain '|' (it separates custom_id fields)")
        stray = sorted({a for a, _ in self.extra_cells if a not in self.arms})
        if stray:
            raise LabError(f"extra cells name arm(s) not in the run: {', '.join(stray)}")

    def k_for(self, arm: BaseArm) -> int:
        if arm.k_role == "passages" and self.passage_k:
            return int(self.passage_k)
        return int(self.k)

    def fills_budgets(self, arm: BaseArm) -> bool:
        """Whether ``arm`` retrieves passages until each capped budget is covered."""
        return arm.k_role == "passages" and not self.passage_k

    def passage_policy(self) -> dict[str, Any]:
        """How passage-ranking arms choose their passage count (recorded in the manifest).

        A budget is only a fair comparison if every arm can fill it: a passage
        baseline or the random-context floor stuck at ``k`` passages would be
        compared at a smaller context than a graph arm that fills the budget.
        """
        if self.passage_k:
            return {"mode": "fixed", "k": int(self.passage_k)}
        return {"mode": "fill", "default_k": int(self.k), "max": PASSAGE_FILL_MAX}

    def config(self) -> dict[str, Any]:
        """Everything that shapes results — hashed to detect a foreign run_dir."""
        return {
            "dataset": self.dataset.to_dict(),
            "arms": list(self.arms),
            "budgets": list(self.budgets),
            "extra_cells": [list(c) for c in self.extra_cells],
            "k": self.k,
            "passage_k": self.passage_k,
            "seed": self.seed,
            "reader_model": self.reader_model,
            "mode": self.mode,
            "order": self.order,
            "reasoning_allowance": self.reasoning_allowance,
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = self.config()
        data.update(
            {
                "max_usd": self.max_usd,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "run_id": self.run_id,
                "max_concurrency": self.max_concurrency,
                "ingest_usd": self.ingest_usd,
                "ingest_plan": asdict(self.ingest_plan) if self.ingest_plan else None,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabRun:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def new_run_id(run: LabRun) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{run.config_hash()[:6]}"


# ── Datasets ─────────────────────────────────────────────────────────────────
@dataclass
class LabItem:
    id: str
    question: str
    gold: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabDataset:
    name: str
    split: str | None
    items: list[LabItem]
    sha256: str
    source: str
    notes: dict[str, Any] = field(default_factory=dict)
    #: HotpotQA only: the pooled paragraph titles the graph must hold.
    titles: list[str] = field(default_factory=list)

    def manifest(self, seed: int | None) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "sha256": self.sha256,
            "source": self.source,
            "n": len(self.items),
            "question_ids": [i.id for i in self.items],
            "seed": seed,
            "notes": self.notes,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _golds(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list | tuple):
        return [str(a) for a in raw if str(a).strip()]
    if raw is None:
        return []
    return [str(raw)]


def parse_qa_records(records: Iterable[Any], *, source: str = "qa file") -> list[LabItem]:
    """Validate ``{question, answer[, id][, split]}`` records into items.

    ``answer`` is a string or a list (first = canonical, rest = aliases).
    Missing ids become ``q0001``…; duplicate ids are refused.
    """
    items: list[LabItem] = []
    seen: set[str] = set()
    for i, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise LabError(f"{source}: record {i} is not an object")
        question = str(raw.get("question") or "").strip()
        gold = _golds(raw.get("answer", raw.get("answers")))
        if not question or not gold:
            raise LabError(f"{source}: record {i} needs a non-empty question and answer")
        qid = str(raw.get("id") or raw.get("_id") or f"q{i:04d}").strip()
        if qid in seen:
            raise LabError(f"{source}: duplicate id {qid!r}")
        seen.add(qid)
        meta = {"split": str(raw["split"])} if raw.get("split") else {}
        if raw.get("type"):
            meta["type"] = str(raw["type"])
        items.append(LabItem(id=qid, question=question, gold=gold, meta=meta))
    if not items:
        raise LabError(f"{source}: no questions")
    return items


def read_qa_file(path: Path, *, label: str | None = None) -> list[Any]:
    """Records from a JSON array (or ``{"items": [...]}``) or a JSONL file.

    UTF-8, with or without a byte-order mark. Error messages name the file as
    ``label`` when given (an upload validates a staging copy under a random name;
    the user should read the name they uploaded).
    """
    label = label or str(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise LabError(f"could not read {label}: {e}") from e
    stripped = text.strip()
    if not stripped:
        raise LabError(f"{label} is empty")
    if stripped[0] in "[{" and not path.suffix.lower() == ".jsonl":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise LabError(f"{label} is not valid JSON: {e}") from e
        if isinstance(data, dict):
            data = data.get("items") or data.get("questions") or []
        if not isinstance(data, list):
            raise LabError(f"{label} must hold a JSON array of QA records")
        return data
    records = []
    for n, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise LabError(f"{label}:{n} is not valid JSON: {e}") from e
    return records


def _apply_split_and_n(items: list[LabItem], split: str | None, n: int | None) -> list[LabItem]:
    if split and split != "all":
        items = [i for i in items if i.meta.get("split") == split]
    if n is not None:
        items = items[: max(0, int(n))]
    return items


def load_dataset(spec: DatasetSpec) -> LabDataset:
    """Load demo / qa-file / hotpotqa questions (hotpotqa is checked later, in Neo4j)."""
    if spec.name == "demo":
        split = spec.split or "test"
        records = read_qa_file(DEMO_QA_PATH)
        items = _apply_split_and_n(parse_qa_records(records, source="demo"), split, spec.n)
        if not items:
            raise LabError(f"the demo set has no questions in split {split!r}")
        return LabDataset("demo", split, items, _sha256_file(DEMO_QA_PATH), DEMO_QA_PATH.name)
    if spec.name == "qa-file":
        if not spec.path:
            raise LabError("dataset 'qa-file' needs a path")
        path = Path(spec.path)
        if not path.is_absolute() and not path.exists():
            path = DATASETS_DIR / path
        items = parse_qa_records(read_qa_file(path, label=path.name), source=path.name)
        items = _apply_split_and_n(items, spec.split, spec.n)
        if not items:
            raise LabError(f"{path.name}: no questions in split {spec.split!r}")
        return LabDataset("qa-file", spec.split, items, _sha256_file(path), path.name)
    if spec.name == "hotpotqa":
        return _load_hotpotqa(spec)
    raise LabError(f"unknown dataset {spec.name!r}")


def _load_hotpotqa(spec: DatasetSpec) -> LabDataset:
    from benchmarks.public import hotpotqa, run_hotpotqa

    seed = hotpotqa.DEFAULT_SEED if spec.seed is None else int(spec.seed)
    n = hotpotqa.DEFAULT_N if spec.n is None else int(spec.n)
    offset = max(0, int(spec.offset or 0))
    data_path = Path(spec.path) if spec.path else None
    try:
        sample = hotpotqa.load_sample(
            offset + n, seed=seed, path=data_path, allow_download=spec.allow_download
        )
    except (OSError, ValueError, RuntimeError) as e:
        raise LabError(f"could not load HotpotQA: {e}") from e
    questions = sample.questions[offset : offset + n]
    if not questions:
        raise LabError("the HotpotQA sample is empty (check n / offset)")
    corpus = run_hotpotqa.build_corpus(questions, sample=sample)
    items = [
        LabItem(
            id=q.id,
            question=q.question,
            gold=[q.answer],
            meta={"type": q.hop_type, "gold_titles": list(run_hotpotqa.gold_titles(q))},
        )
        for q in questions
    ]
    file_path = data_path or hotpotqa.CACHE_PATH
    sha = _sha256_file(file_path) if Path(file_path).exists() else ""
    titles = corpus.titles
    return LabDataset(
        name="hotpotqa",
        split="dev-distractor",
        items=items,
        sha256=sha,
        source=str(sample.source),
        notes={
            "seed": seed,
            "offset": offset,
            "paragraphs": len(titles),
            "paragraph_titles_sha256": hashlib.sha256(
                "\x1f".join(titles).encode("utf-8")
            ).hexdigest(),
            "pool_size": sample.pool_size,
        },
        titles=titles,
    )


async def check_graph_ready(
    dataset: LabDataset, arms: Sequence[str] | None = None
) -> dict[str, int]:
    """Refuse a corpus that is not in Neo4j. Read-only. Returns the graph fingerprint.

    HotpotQA is always checked paragraph by paragraph. Any other dataset only
    needs a non-empty graph — and not even that when every arm is closed-book.
    """
    fingerprint = await graph_fingerprint()
    only_closed_book = arms is not None and all(a == "null_closed_book" for a in arms)
    if dataset.name == "hotpotqa":
        from benchmarks.public import run_hotpotqa

        present = await run_hotpotqa.graph_documents()
        missing = [t for t in dataset.titles if t not in present]
        if missing:
            raise GraphNotIngested(
                f"{len(missing)} of {len(dataset.titles)} HotpotQA paragraphs for this sample are "
                f"not ingested in Neo4j (e.g. {missing[:3]}). Ingest the sample's corpus first "
                "(the Lab ingest batch, one document per paragraph) — scoring this graph would "
                "report numbers about a different corpus."
            )
    elif not only_closed_book and fingerprint["entities"] == 0 and fingerprint["chunks"] == 0:
        raise GraphNotIngested(
            "the knowledge graph is empty — ingest documents (or seed the demo) before a Lab run"
        )
    return fingerprint


# ── HotpotQA crediting (the harness's own rules, over packed units) ─────────
_WORD = re.compile(r"\w+")


class NameMatcher:
    """``run_hotpotqa.names_present`` with a first-token pre-filter — same answers, faster.

    A word-bounded, case-insensitive match of a name implies that the name's
    first ``\\w+`` token appears as a token of the text, so only names whose
    first token does are handed to the exact matcher. Names with no word
    character at all are always checked.
    """

    def __init__(self, names: Iterable[str]) -> None:
        self.index: dict[str, list[str]] = {}
        self.always: list[str] = []
        for name in names:
            first = _WORD.search((name or "").lower())
            if first is None:
                self.always.append(name)
            else:
                self.index.setdefault(first.group(), []).append(name)

    def present(self, text: str) -> set[str]:
        from benchmarks.public.run_hotpotqa import names_present

        if not text:
            return set()
        tokens = set(_WORD.findall(text.lower()))
        candidates = list(self.always)
        for token in tokens:
            candidates.extend(self.index.get(token, ()))
        return set(names_present(text, candidates))


def credit_packed(
    packed: Packed, provenance: dict[str, list[str]], matcher: NameMatcher
) -> dict[str, str]:
    """``credit_paragraphs`` over packed units: paragraph title → best channel.

    Same precedence as the harness (prose → entity → edge):
      • PROSE  — the documents of the packed ``prose`` units;
      • ENTITY — names in entity blocks (relationship lines removed), in the
        vocabulary null's name list and in community blocks;
      • EDGE   — names in relationship lines, relation units and paths.
    """
    from benchmarks.run_benchmark import EDGE, ENTITY, PROSE, edge_segment, strip_graph_structure

    prose_docs: set[str] = set()
    entity_parts: list[str] = []
    edge_parts: list[str] = []
    for unit, text in packed.unit_texts():
        if unit.kind == "prose":
            if unit.source_id:
                prose_docs.add(unit.source_id)
        elif unit.kind == "entity":
            entity_parts.append(strip_graph_structure(text))
            edge_parts.append(edge_segment(text))
        elif unit.kind in ("name_list", "community"):
            entity_parts.append(text)
        else:  # relation, path
            edge_parts.append(text)

    by_channel: dict[str, set[str]] = {PROSE: prose_docs, ENTITY: set(), EDGE: set()}
    for channel, parts in ((ENTITY, entity_parts), (EDGE, edge_parts)):
        for name in matcher.present("\n\n".join(p for p in parts if p)):
            by_channel[channel].update(provenance.get(name) or [])
    credited: dict[str, str] = {}
    for channel in (PROSE, ENTITY, EDGE):
        for title in sorted(by_channel[channel]):
            credited.setdefault(title, channel)
    return credited


# ── Files ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RunPaths:
    """The files of one run directory."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST

    @property
    def contexts(self) -> Path:
        return self.root / CONTEXTS

    @property
    def requests(self) -> Path:
        return self.root / REQUESTS

    @property
    def responses(self) -> Path:
        return self.root / RESPONSES

    @property
    def rows(self) -> Path:
        return self.root / ROWS

    @property
    def leaderboard(self) -> Path:
        return self.root / LEADERBOARD

    @property
    def report(self) -> Path:
        return self.root / REPORT


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    """Every parseable line of a JSONL file (a torn last line is ignored)."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping a malformed line in %s", path.name)
    return records


def _repair_jsonl(path: Path) -> None:
    """Drop a torn final line left by a crash, so appends start on a clean line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    path.write_bytes(data[: cut + 1] if cut >= 0 else b"")


def _append_jsonl(path: Path, records: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_contexts(path: Path) -> list[dict]:
    """contexts.jsonl with every deduplicated ``text`` restored from its sha256."""
    texts: dict[str, str] = {}
    records = read_jsonl(path)
    for record in records:
        if record.get("text") is not None:
            texts[record["sha256"]] = record["text"]
    for record in records:
        if record.get("text") is None:
            record["text"] = texts.get(record.get("sha256") or "", "")
    return records


# ── Provenance ───────────────────────────────────────────────────────────────
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


_TRUTHY = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}


def code_provenance() -> dict[str, Any]:
    """The commit a run was made on: ``git`` in the checkout, else ``$SYNAPSE_GIT_SHA``.

    The Docker image has neither a ``.git`` directory nor a git binary, so a run
    started there cannot see its commit; the build (or ``.env``) can pass it as
    ``SYNAPSE_GIT_SHA`` (and ``SYNAPSE_GIT_DIRTY``). With neither, the SHA is
    recorded as unknown WITH the reason (``git_unavailable``), never a bare null.
    """
    sha = _git("rev-parse", "HEAD")
    dirty: bool | None = None
    source: str | None = None
    unavailable: str | None = None
    if sha:
        source = "git"
        status = _git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status) if status is not None else None
    elif os.environ.get("SYNAPSE_GIT_SHA", "").strip():
        sha = os.environ["SYNAPSE_GIT_SHA"].strip()
        source = "env:SYNAPSE_GIT_SHA"
        dirty = _TRUTHY.get(os.environ.get("SYNAPSE_GIT_DIRTY", "").strip().lower())
    else:
        sha = None
        unavailable = (
            "no git checkout or git binary here (e.g. the Docker image); "
            "set SYNAPSE_GIT_SHA to record the commit"
        )
    version = None
    try:
        import tomllib

        with open(BACKEND_ROOT / "pyproject.toml", "rb") as fh:
            version = tomllib.load(fh).get("project", {}).get("version")
    except (OSError, ValueError):
        pass
    return {"git_sha": sha, "git_dirty": dirty, "git_source": source,
            "git_unavailable": unavailable, "synapse_version": version}


_EMBEDDING_MODEL_ATTR = {
    "fastembed": "fastembed_model",
    "openai": "openai_embedding_model",
    "azure_openai": "azure_openai_embedding_deployment",
    "gemini": "gemini_embedding_model",
    "ollama": "ollama_embedding_model",
    "vertex": "vertex_embedding_model",
    "bedrock": "bedrock_embedding_model",
    "cohere": "cohere_embedding_model",
}


def model_provenance(run: LabRun) -> dict[str, Any]:
    from app.config import get_settings

    s = get_settings()
    provider = str(s.embedding_provider)
    attr = _EMBEDDING_MODEL_ATTR.get(provider)
    try:
        from benchmarks.public.run_hotpotqa import chat_model_name

        extraction = chat_model_name(s)
    except Exception:  # noqa: BLE001 - provenance must never be what fails a run
        extraction = str(s.llm_provider)
    return {
        "embedding": {
            "provider": provider,
            "model": getattr(s, attr, provider) if attr else provider,
            "dim": s.embedding_dim,
        },
        "extraction": {
            "provider": str(s.llm_provider),
            "model": extraction,
            "temperature": s.extraction_temperature,
        },
        "reader": {
            "model": run.reader_model,
            "prompt_version": reader.PROMPT_VERSION,
            "system_prompt": reader.SYSTEM_PROMPT,
            "max_output_tokens": reader.max_output_tokens(
                run.reader_model, run.reasoning_allowance
            ),
            "reasoning_effort": s.openai_reasoning_effort,
            "seed": run.seed,
        },
    }


# ── Batch backend (implemented in app.lab.batch) ─────────────────────────────
class BatchBackend(Protocol):
    """What the runner needs from the Batch API layer (``app.lab.batch``).

    ``submit`` receives the request lines exactly as written to requests.jsonl
    (OpenAI Batch input format) and returns a status dict. ``poll`` returns at
    least ``{"done": bool, "status": str}``, and ``"rejected": [...]`` when a
    batch was refused at validation (nothing ran, $0): those requests have no
    result, are not errors, and must be sent again (see :func:`collect_batch`).
    ``collect`` returns one record per finished request: either an OpenAI Batch
    OUTPUT line (``{"custom_id", "response": {"status_code", "body"}, "error"}``)
    or ``{"custom_id", **ReaderResult.to_dict()}``.

    Optionally, ``collected(run_dir)`` returns every result collected so far,
    earlier calls included, from local files only. ``collect`` hands each result
    over ONCE; when that hand-over is interrupted (a crash, a failed download of
    a later part), ``collected`` is how the runner recovers what was billed.
    Optionally too, ``rejected_requests(run_dir)`` returns the request lines a
    rejected batch still owes (local files only): what is re-planned after a
    rejection besides the requests with no response line at all. A backend
    without them, over a run dir holding ``app.lab.batch``'s ``batches.json``,
    is read through that module's local state instead, so no wrapper can hide
    billed answers or owed requests (see :func:`_backend_local_read`).
    """

    async def submit(self, run_dir: Path, requests: list[dict], *, run_id: str) -> dict: ...

    async def poll(self, run_dir: Path) -> dict: ...

    async def collect(self, run_dir: Path) -> list[dict]: ...


class ModuleBatchBackend:
    """Adapter over ``app.lab.batch``'s module-level ``submit`` / ``poll`` / ``collect``."""

    def __init__(self, client: Any = None) -> None:
        self.client = client
        try:
            from app.lab import batch as module
        except ImportError as e:
            raise LabError(f"batch mode needs app.lab.batch ({e})") from e
        missing = [n for n in ("submit", "poll", "collect") if not hasattr(module, n)]
        if missing:
            raise LabError(
                "app.lab.batch must expose async submit(run_dir, requests, *, run_id, client), "
                f"poll(run_dir, *, client) and collect(run_dir, *, client); missing: {missing}"
            )
        self.module = module

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        result = getattr(self.module, name)(*args, client=self.client, **kwargs)
        return await result if asyncio.iscoroutine(result) else result

    async def submit(self, run_dir: Path, requests: list[dict], *, run_id: str) -> dict:
        return await self._call("submit", run_dir, requests, run_id=run_id)

    async def poll(self, run_dir: Path) -> dict:
        return await self._call("poll", run_dir)

    async def collect(self, run_dir: Path) -> list[dict]:
        return await self._call("collect", run_dir)

    async def collected(self, run_dir: Path) -> list[dict]:
        """Every result of every collected part (local files only, no network)."""
        return list(self.module.latest_records(run_dir).values())

    async def rejected_requests(self, run_dir: Path) -> list[dict]:
        """The request lines rejected parts still owe (local files only, no network)."""
        return list(self.module.rejected_requests(run_dir))


async def _backend_local_read(backend: Any, name: str, run_dir: Path) -> list[dict]:
    """``backend.<name>(run_dir)`` — or, when the backend lacks it, ``app.lab.batch``'s own
    local reading of this run dir's ``batches.json`` (``collected`` → ``latest_records``,
    ``rejected_requests`` → ``rejected_requests``). Local files only; ``[]`` otherwise.

    ``app.lab.batch.submit`` never re-sends a request whose collected result
    succeeded, so the runner must always be able to see those results: a wrapper
    that forwards only submit / poll / collect would otherwise strand billed
    answers for good (the 2026-09-25 review of the iso-token harness).
    """
    method = getattr(backend, name, None)
    if callable(method):
        result = method(run_dir)
        return list((await result if asyncio.iscoroutine(result) else result) or [])
    from app.lab import batch as module

    if not module.state_path(run_dir).exists():
        return []
    if name == "collected":
        return list(module.latest_records(run_dir).values())
    return list(module.rejected_requests(run_dir))


def result_from_record(record: dict) -> reader.ReaderResult:
    """Normalise a collected batch record into a :class:`reader.ReaderResult`."""
    if "response" in record or ("error" in record and "answer" not in record):
        response = record.get("response") or {}
        status = int(response.get("status_code") or 0) if response else 0
        body = response.get("body") if response else None
        if record.get("error") or status != 200 or not body:
            err = record.get("error") or (body or {}).get("error") or f"status {status}"
            text = err if isinstance(err, str) else json.dumps(err, default=str)
            return reader.ReaderResult(error=text[:500])
        return reader.parse_completion(body)
    return reader.ReaderResult.from_dict(record)


# ── Events ───────────────────────────────────────────────────────────────────
async def _emit(on_event: EventCallback | None, event: dict) -> None:
    if on_event is None:
        return
    try:
        maybe = on_event(event)
        if asyncio.iscoroutine(maybe):
            await maybe
    except Exception:  # noqa: BLE001 - a broken listener must not break the run
        logger.exception("lab event listener failed")


# ── Manifest ─────────────────────────────────────────────────────────────────
def _init_manifest(run: LabRun, dataset: LabDataset) -> dict[str, Any]:
    arms = {
        name: {
            "config_hash": ARMS[name].config_hash(),
            "config": ARMS[name].config(),
            "family": ARMS[name].family,
            "title": ARMS[name].title,
            "source": ARMS[name].source,
            "retrieval_llm_calls": ARMS[name].retrieval_llm_calls,
            "needs_graph": ARMS[name].needs_graph,
            "k": run.k_for(ARMS[name]),
        }
        for name in run.arms
    }
    price = cost.resolve_price(run.reader_model)
    return {
        "run_id": run.run_id,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "config": run.to_dict(),
        "config_hash": run.config_hash(),
        "code": code_provenance(),
        "arms": arms,
        "packer": {**PACKER_POLICY, "order": run.order, "passages": run.passage_policy()},
        "dataset": dataset.manifest(dataset.notes.get("seed")),
        "models": model_provenance(run),
        "tokenizer": tokenizer_label(run.reader_model),
        "prices": {
            "table_checked_on": cost.PRICES_CHECKED_ON,
            "reader_checked_on": cost.price_checked_on(run.reader_model),
            "reader_usd_per_1m": (
                {"input": price.input_usd_per_1m, "output": price.output_usd_per_1m}
                if price else None
            ),
            "batch_multiplier": cost.BATCH_PRICE_MULTIPLIER if run.mode == "batch" else None,
            "cached_input": "priced at the full input rate (cost.py has no cached rate)",
            "url": cost.PRICING_URL,
        },
        "budget_cap_usd": run.max_usd,
        "phases": {
            "retrieve": {"status": "pending"},
            "read": {"status": "pending" if run.mode != "retrieve" else "skipped"},
            "score": {"status": "pending"},
        },
        "estimate": {},
        "actual": {},
    }


def _save(paths: RunPaths, manifest: dict[str, Any], **changes: Any) -> None:
    manifest.update(changes)
    manifest["updated_at"] = _now()
    _write_json(paths.manifest, manifest)


def _phase(paths: RunPaths, manifest: dict[str, Any], name: str, **changes: Any) -> None:
    manifest["phases"].setdefault(name, {}).update(changes)
    _save(paths, manifest)


# ── Phase 1: retrieve ────────────────────────────────────────────────────────
def _cell_key(arm: str, budget: int | None, qid: str) -> tuple[str, str, str]:
    return (arm, budget_label(budget), qid)


async def _retrieve_phase(
    run: LabRun,
    dataset: LabDataset,
    paths: RunPaths,
    manifest: dict[str, Any],
    on_event: EventCallback | None,
) -> None:
    _repair_jsonl(paths.contexts)
    existing = read_jsonl(paths.contexts)
    latest = {(r["arm"], budget_label(r.get("budget")), r["qid"]): r for r in existing}
    # A cell whose retrieval FAILED is retried on resume (its new line supersedes it).
    done = {key for key, r in latest.items() if not r.get("retrieval_error")}
    stored_texts = {r["sha256"] for r in existing if r.get("text") is not None}

    matcher: NameMatcher | None = None
    provenance: dict[str, list[str]] = {}
    if dataset.name == "hotpotqa":
        from benchmarks.public.run_hotpotqa import fetch_provenance

        provenance = await fetch_provenance()
        matcher = NameMatcher(provenance)

    total = len(run.arms) * len(dataset.items)
    count = 0
    errors = 0
    started = time.monotonic()
    _phase(paths, manifest, "retrieve", status="running", started_at=_now())
    await _emit(on_event, {"type": "phase", "phase": "retrieve", "status": "running",
                           "total": total})
    for arm_name in run.arms:
        arm = ARMS[arm_name]
        k = run.k_for(arm)
        for item in dataset.items:
            count += 1
            missing = [
                b for b in run.budgets_for(arm_name) if _cell_key(arm_name, b, item.id) not in done
            ]
            if not missing:
                continue
            error = None
            token_cache: dict[str, tuple[int, bool]] = {}
            # Sized on ALL of the arm's capped budgets, not just the missing ones, so
            # a resumed cell holds exactly what an uninterrupted run would have.
            capped = [b for b in run.budgets_for(arm_name) if b is not None]
            try:
                if run.fills_budgets(arm) and capped:
                    evidence = await _retrieve_filling(
                        arm, item.question, k=k, seed=run.seed, budget=max(capped),
                        model=run.reader_model, order=run.order, token_cache=token_cache,
                    )
                else:
                    evidence = await arm.retrieve(item.question, k=k, seed=run.seed)
            except Exception as e:  # noqa: BLE001 - one failed retrieval is recorded, not fatal
                logger.warning("arm %s failed on %s: %s", arm_name, item.id, e)
                evidence, error = Evidence.empty(arm_name), f"{type(e).__name__}: {e}"[:500]
                errors += 1
            records = []
            for budget in missing:
                cell_evidence = evidence
                if budget is None and run.fills_budgets(arm) and capped:
                    cell_evidence = _top_units(evidence, k)  # the default context keeps k
                packed = pack(cell_evidence, budget, run.reader_model, run.order,
                              token_cache=token_cache)
                sha = _sha(packed.text)
                gold_titles = item.meta.get("gold_titles")
                credited = None
                if matcher is not None and gold_titles is not None:
                    full = credit_packed(packed, provenance, matcher)
                    credited = {t: full[t] for t in gold_titles if t in full}
                records.append(
                    {
                        "arm": arm_name,
                        "budget": budget,
                        "qid": item.id,
                        "text": None if sha in stored_texts else packed.text,
                        "sha256": sha,
                        "tokens": packed.tokens,
                        "tokens_estimated": packed.estimated,
                        "units_used": packed.units_used,
                        "by_kind": packed.by_kind,
                        "truncated": packed.truncated,
                        "skipped": packed.skipped,
                        "units": [p.to_dict() for p in packed.placed],
                        "credited_gold": credited,
                        "containment": metrics.contains_answer(packed.text, item.gold),
                        "retrieval_error": error,
                        "meta": _small(evidence.meta),
                    }
                )
                stored_texts.add(sha)
                done.add(_cell_key(arm_name, budget, item.id))
            _append_jsonl(paths.contexts, records)
            await _emit(on_event, {"type": "progress", "phase": "retrieve", "done": count,
                                   "total": total, "arm": arm_name})
    _phase(
        paths, manifest, "retrieve", status="done", finished_at=_now(),
        cells=len(done), errors=errors, seconds=round(time.monotonic() - started, 2),
    )
    await _emit(on_event, {"type": "phase", "phase": "retrieve", "status": "done"})


async def _retrieve_filling(
    arm: BaseArm,
    question: str,
    *,
    k: int,
    seed: int,
    budget: int,
    model: str,
    order: str,
    token_cache: dict[str, tuple[int, bool]],
) -> Evidence:
    """Retrieve enough passages to cover ``budget`` (the largest capped budget).

    Starts from a pool sized for ``PASSAGE_FILL_GUESS_TOKENS``-token passages
    (never below ``k``) and doubles it while the packed context still has room,
    the arm returned a full pool (so the corpus may hold more) and the pool is
    below ``PASSAGE_FILL_MAX``. Deterministic: same data, same pools. Smaller
    budgets pack a prefix of the same ranking, exactly as the packer always does.
    """
    pool = min(PASSAGE_FILL_MAX, max(int(k), math.ceil(budget / PASSAGE_FILL_GUESS_TOKENS)))
    while True:
        evidence = await arm.retrieve(question, k=pool, seed=seed)
        full = len(evidence.units) >= pool
        if not full or pool >= PASSAGE_FILL_MAX:
            break
        if pack(evidence, budget, model, order, token_cache=token_cache).truncated:
            break  # the budget is covered
        pool = min(PASSAGE_FILL_MAX, pool * 2)
    evidence.meta = {**(evidence.meta or {}), "fill_pool": pool,
                     "fill_capped": full and pool >= PASSAGE_FILL_MAX}
    return evidence


def _top_units(evidence: Evidence, k: int) -> Evidence:
    """The ``k`` best-ranked units (score desc, the arm's order on ties) — the packer's rank."""
    ranked = sorted(evidence.units, key=lambda u: -float(u.score))[: max(0, int(k))]
    return Evidence(ranked, evidence.arm, dict(evidence.meta or {}))


def _small(meta: dict[str, Any], limit: int = 2_000) -> dict[str, Any]:
    """Evidence meta, dropped to a stub if it is unexpectedly large."""
    blob = json.dumps(meta, default=str)
    return meta if len(blob) <= limit else {"truncated_meta": blob[:limit]}


# ── Phase 2: read ────────────────────────────────────────────────────────────
def _ordered_contexts(
    run: LabRun, dataset: LabDataset, contexts: list[dict]
) -> list[dict]:
    """Contexts in canonical (arm, budget, question) order — the request-id order.

    One record per cell: when a cell was written twice (a failed retrieval
    retried on resume), the LAST record wins.
    """
    cell_i = {(a, budget_label(b)): i for i, (a, b) in enumerate(run.cells())}
    q_i = {item.id: i for i, item in enumerate(dataset.items)}
    latest: dict[tuple[str, str, str], dict] = {}
    for c in contexts:
        cell = (c["arm"], budget_label(c.get("budget")))
        if cell in cell_i and c["qid"] in q_i:
            latest[(*cell, c["qid"])] = c
    return sorted(
        latest.values(),
        key=lambda c: (cell_i[(c["arm"], budget_label(c.get("budget")))], q_i[c["qid"]]),
    )


def plan_requests(
    run: LabRun, dataset: LabDataset, contexts: list[dict]
) -> tuple[list[dict], dict[tuple[str, str, str], str]]:
    """``(request lines, cell → custom_id)``: one line per DISTINCT request body.

    Lines are in the OpenAI Batch input format. Identical bodies (N0 at every
    budget; any arm whose context is the same at two budgets) are sent once and
    shared by every cell that needs them. A cell whose retrieval FAILED is not
    read at all: its empty context would be N0's request, and it would silently
    score N0's closed-book answer as that arm's (``score_rows`` marks it).
    """
    questions = {item.id: item.question for item in dataset.items}
    by_hash: dict[str, str] = {}
    lines: list[dict] = []
    cell_to_id: dict[tuple[str, str, str], str] = {}
    for ctx in _ordered_contexts(run, dataset, contexts):
        if ctx.get("retrieval_error"):
            continue
        body = reader.build_request(
            questions[ctx["qid"]],
            ctx.get("text") or "",
            run.reader_model,
            seed=run.seed,
            reasoning_allowance=run.reasoning_allowance,
        )
        digest = reader.request_hash(body)
        custom_id = by_hash.get(digest)
        if custom_id is None:
            custom_id = f"{run.run_id}|{ctx['arm']}|{budget_label(ctx.get('budget'))}|{ctx['qid']}"
            by_hash[digest] = custom_id
            lines.append(
                {"custom_id": custom_id, "method": "POST", "url": BATCH_ENDPOINT, "body": body}
            )
        cell_to_id[(ctx["arm"], budget_label(ctx.get("budget")), ctx["qid"])] = custom_id
    return lines, cell_to_id


def parse_custom_id(custom_id: str) -> dict[str, Any]:
    """``"<run>|<arm>|<budget>|<qid>"`` → its parts (the qid may itself contain ``|``)."""
    run_id, arm, budget, qid = custom_id.split("|", 3)
    return {"run_id": run_id, "arm": arm, "budget": parse_budget(budget), "qid": qid}


def _usd(result: reader.ReaderResult, model: str, batch: bool) -> float | None:
    usage = cost.Usage(prompt_tokens=result.prompt_tokens,
                       completion_tokens=result.completion_tokens)
    return cost.usd(usage, cost.resolve_price(model), batch=batch)


def _upper_usd(body: dict, model: str, batch: bool) -> float | None:
    prompt, _est = reader.request_prompt_tokens(body, model)
    usage = cost.Usage(
        prompt_tokens=int(prompt * 1.1) + 1, completion_tokens=reader.request_output_cap(body)
    )
    return cost.usd(usage, cost.resolve_price(model), batch=batch)


def _latest_responses(paths: RunPaths) -> dict[str, dict]:
    """custom_id → its last response line (a later retry supersedes an error)."""
    latest: dict[str, dict] = {}
    for record in read_jsonl(paths.responses):
        if record.get("custom_id"):
            latest[record["custom_id"]] = record
    return latest


def _spent(paths: RunPaths) -> float:
    return sum(float(r.get("usd") or 0.0) for r in read_jsonl(paths.responses))


def _response_record(
    custom_id: str, result: reader.ReaderResult, usd: float | None, batch: bool
) -> dict:
    return {"custom_id": custom_id, **result.to_dict(), "usd": usd, "batch": batch,
            "at": _now()}


async def _read_phase(
    run: LabRun,
    dataset: LabDataset,
    paths: RunPaths,
    manifest: dict[str, Any],
    *,
    client: Any,
    batch_backend: BatchBackend | None,
    on_event: EventCallback | None,
) -> str:
    """Returns the manifest status after reading: done | refused | aborted | batch_submitted."""
    contexts = load_contexts(paths.contexts)
    lines, cell_to_id = plan_requests(run, dataset, contexts)
    _write_jsonl(paths.requests, lines)

    # Second gate: re-estimate on the MEASURED contexts and de-duplicated requests.
    measured: dict[tuple[str, int | None], list[int]] = {}
    unique: dict[tuple[str, int | None], int] = {}
    for ctx in _ordered_contexts(run, dataset, contexts):
        key = (ctx["arm"], ctx.get("budget"))
        label = budget_label(ctx.get("budget"))
        measured.setdefault(key, []).append(int(ctx.get("tokens") or 0))
        owner = f"{run.run_id}|{ctx['arm']}|{label}|{ctx['qid']}"
        # A cell adds a request only when it is the first to need that body (a
        # cell whose retrieval failed has none: it is not read).
        needs = cell_to_id.get((ctx["arm"], label, ctx["qid"]))
        unique[key] = unique.get(key, 0) + int(needs == owner)
    batch = run.mode == "batch"
    backend = (batch_backend or ModuleBatchBackend()) if batch else None
    if backend is not None:
        # Record every answer the Batch state already holds (billed) before planning:
        # a collect interrupted after its hand-over must never leave one out (and
        # the Batch layer would refuse to send it again).
        _record_results(run, paths, await _backend_local_read(backend, "collected", paths.root))
    latest = _latest_responses(paths)
    if manifest["phases"]["read"].get("only_unanswered"):
        # Re-planning after a batch was rejected at validation: send what never ran —
        # no response line at all, or owed by a rejected part (a refused --retry-failed
        # batch held requests with an old error line) — and nothing else that came
        # back with an error (that is --retry-failed).
        owed = set()
        if backend is not None:
            owed = {str(x.get("custom_id")) for x in
                    await _backend_local_read(backend, "rejected_requests", paths.root)}
        pending = [line for line in lines if line["custom_id"] not in latest
                   or (line["custom_id"] in owed and not _answered(latest[line["custom_id"]]))]
    else:
        done_ids = {cid for cid, r in latest.items()
                    if not r.get("error") and not r.get("skipped")}
        pending = [line for line in lines if line["custom_id"] not in done_ids]
    spent = _spent(paths)
    remaining_cap = None if run.max_usd is None else max(0.0, run.max_usd - spent)
    estimate = estimate_run(
        questions=[item.question for item in dataset.items],
        arms=run.arms,
        budgets=run.budgets,
        cells=run.cells(),
        reader_model=run.reader_model,
        mode=run.mode,
        max_usd=run.max_usd,
        context_tokens=measured,
        unique_requests=unique,
        reasoning_allowance=run.reasoning_allowance,
    )
    pending_upper = 0.0
    unpriced = False
    for line in pending:
        u = _upper_usd(line["body"], run.reader_model, batch)
        if u is None:
            unpriced = True
        else:
            pending_upper += u
    manifest["estimate"]["read"] = estimate.to_dict()
    manifest["estimate"]["read_pending"] = {
        "requests": len(pending), "upper_usd": None if unpriced else pending_upper,
        "spent_usd": spent,
    }
    refuse_reason = None
    if remaining_cap is not None and pending:
        if unpriced:
            refuse_reason = f"no price on file for {run.reader_model!r}; the cap cannot be checked"
        elif pending_upper > remaining_cap:
            refuse_reason = (
                f"upper bound of the pending reads {cost.format_usd(pending_upper)} exceeds the "
                f"remaining cap {cost.format_usd(remaining_cap)} (on the measured contexts)"
            )
    if refuse_reason:
        _phase(paths, manifest, "read", status="refused", reason=refuse_reason)
        _save(paths, manifest, status="refused", refuse_reason=refuse_reason)
        await _emit(on_event, {"type": "refused", "reason": refuse_reason,
                               "estimate": estimate.to_dict()})
        return "refused"

    if not pending:
        _read_ended(manifest)
        failed = sum(1 for line in lines
                     if (latest.get(line["custom_id"]) or {}).get("error")
                     or (latest.get(line["custom_id"]) or {}).get("skipped"))
        _phase(paths, manifest, "read", status="done_with_errors" if failed else "done",
               requests=len(lines), **({"failed": failed} if failed else {}))
        return "done"

    if backend is not None:
        info = await backend.submit(paths.root, pending, run_id=str(run.run_id))
        read = manifest["phases"]["read"]
        resent = bool(read.pop("only_unanswered", None))
        read.pop("rejected", None)  # outstanding no more: history stays in "rejections"
        changes: dict[str, Any] = {}
        if resent:
            parts = list((info or {}).get("parts") or [])
            message = (
                f"{read.get('message') or 'batch rejected at validation'} — re-submitted "
                f"{len(pending):,} request(s) that never ran in {len(parts)} part(s)"
                + (", sent one at a time as the gate allows"
                   if (info or {}).get("queued") else "")
            )
            changes = {"message": message, "resubmitted_at": _now()}
        _phase(paths, manifest, "read", status="batch_submitted", submitted_at=_now(),
               requests=len(lines), pending=len(pending), batch=info, **changes)
        _save(paths, manifest, status="batch_submitted",
              **({"message": changes["message"]} if resent else {}))
        await _emit(on_event, {"type": "batch_submitted", "requests": len(pending),
                               "batch": info,
                               **({"message": changes["message"]} if resent else {})})
        return "batch_submitted"

    _phase(paths, manifest, "read", status="running", started_at=_now(),
           requests=len(lines), pending=len(pending))
    await _emit(on_event, {"type": "phase", "phase": "read", "status": "running",
                           "total": len(pending)})
    guard = reader.SpendGuard(
        run.max_usd,
        upper_usd=lambda body: _upper_usd(body, run.reader_model, False),
        actual_usd=lambda result: _usd(result, run.reader_model, False),
        spent=spent,
    )
    _repair_jsonl(paths.responses)
    finished = 0

    async def on_result(index: int, result: reader.ReaderResult) -> None:
        nonlocal finished
        if result.skipped:
            return  # never sent — resumable, not recorded
        custom_id = pending[index]["custom_id"]
        usd = _usd(result, run.reader_model, False)
        if result.error is not None:
            usd = _upper_usd(pending[index]["body"], run.reader_model, False)
        _append_jsonl(paths.responses, [_response_record(custom_id, result, usd, False)])
        finished += 1
        await _emit(on_event, {"type": "progress", "phase": "read", "done": finished,
                               "total": len(pending), "spent_usd": guard.spent})

    results = await reader.answer_realtime(
        [line["body"] for line in pending],
        client=client,
        max_concurrency=run.max_concurrency,
        guard=guard,
        on_result=on_result,
    )
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.error and not r.skipped)
    if skipped:
        reason = (
            f"spend cap: {skipped} request(s) not sent — the next one could have pushed the "
            f"metered spend past {cost.format_usd(run.max_usd)}"
        )
        _phase(paths, manifest, "read", status="aborted", reason=reason, failed=failed,
               skipped=skipped, finished_at=_now())
        _save(paths, manifest, abort_reason=reason)
        await _emit(on_event, {"type": "aborted", "reason": reason})
        return "aborted"
    _read_ended(manifest)
    _phase(paths, manifest, "read", status="done_with_errors" if failed else "done",
           finished_at=_now(), failed=failed)
    return "done"


def _answered(record: dict | None) -> bool:
    return record is not None and not record.get("error") and not record.get("skipped")


def _read_ended(manifest: dict[str, Any]) -> None:
    """The read phase is over (done / done_with_errors): nothing is re-submitted any more.

    Clears the rejection message and markers, so neither the manifest nor a
    caller printing ``message`` can still claim a re-submission in progress.
    """
    read = manifest["phases"]["read"]
    for key in ("only_unanswered", "rejected"):
        read.pop(key, None)
    manifest.pop("message", None)


def _record_results(
    run: LabRun, paths: RunPaths, records: Iterable[dict]
) -> tuple[list[dict], dict[str, dict]]:
    """Append each Batch result that responses.jsonl does not hold yet.

    An answer is never recorded twice, nor replaced by an error; an error is
    replaced by a later answer. Returns ``(appended lines, latest per custom_id)``.
    """
    _repair_jsonl(paths.responses)
    latest = _latest_responses(paths)
    out = []
    for record in records:
        custom_id = record.get("custom_id")
        if not custom_id:
            continue
        result = result_from_record(record)
        previous = latest.get(custom_id)
        if previous is not None and (not previous.get("error") or result.error):
            continue  # already answered (never billed twice), or still the same failure
        if not result.model:
            result.model = run.reader_model
        line = _response_record(custom_id, result, _usd(result, run.reader_model, True), True)
        out.append(line)
        latest[custom_id] = line
    if out:
        _append_jsonl(paths.responses, out)
    return out, latest


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


async def collect_batch(
    run: LabRun, paths: RunPaths, manifest: dict[str, Any], backend: BatchBackend,
    on_event: EventCallback | None = None,
) -> bool:
    """Poll the batch; when done, append its answers to responses.jsonl. True when done.

    Idempotent: ``backend.collect`` hands each part's results over once and marks
    the part collected, so a collect interrupted after that (a crash before the
    append, a failed download of a later part) would lose billed answers. When
    the backend can re-read what it already collected (``collected``), every
    result without a response line is appended — an answer is never recorded
    twice, nor replaced by an error. The read status is then judged over ALL
    requests, not over this call's records.

    REJECTED AT VALIDATION. When the poll lists parts OpenAI refused
    (``"rejected"``: nothing ran, $0), their requests are NOT answered-with-
    errors and the run is NOT scored. Once nothing else is running, whatever
    did finish is collected, then the run goes back to ``reading`` with the
    read phase ``pending`` (``only_unanswered``) and a message saying why:
    the next :func:`resume` / :func:`run_lab` re-plans only the requests that
    never ran, re-prices them against the remaining cap and sends them through
    the Batch backend's enqueued-token gate. Returns False in that case.
    """
    status = await backend.poll(paths.root)
    _phase(paths, manifest, "read", last_poll=status, polled_at=_now())
    await _emit(on_event, {"type": "batch_status", "status": status})
    rejected = [r for r in status.get("rejected") or [] if isinstance(r, dict)]
    if not status.get("done"):
        if rejected:  # re-planned once the parts still running have finished
            message = _rejection_message(rejected, status, waiting=True)
            _phase(paths, manifest, "read", rejected=rejected, message=message)
            _save(paths, manifest, message=message)
        return False
    records = list(await backend.collect(paths.root) or [])
    records.extend(await _backend_local_read(backend, "collected", paths.root))
    out, latest = _record_results(run, paths, records)
    wanted = [str(r["custom_id"]) for r in read_jsonl(paths.requests) if r.get("custom_id")]
    if rejected:
        message = _rejection_message(rejected, status)
        read = manifest["phases"]["read"]
        history = [*(read.get("rejections") or []),
                   {"at": _now(), "parts": [r.get("part") for r in rejected],
                    "requests": sum(int(r.get("requests") or 0) for r in rejected),
                    "code": rejected[0].get("code"), "limit": rejected[0].get("limit"),
                    "gate": status.get("max_enqueued_tokens")}]
        unanswered = sum(1 for cid in wanted if cid not in latest)
        _phase(paths, manifest, "read", status="pending", rejected=rejected,
               rejected_at=_now(), rejections=history, message=message,
               only_unanswered=True, collected=len(out), unanswered=unanswered,
               resubmit="auto" if all(r.get("retryable") for r in rejected) else "manual")
        _save(paths, manifest, status="reading", message=message)
        await _emit(on_event, {"type": "batch_rejected", "message": message,
                               "unanswered": unanswered, "rejected": rejected})
        return False
    failed = sum(
        1 for cid in wanted
        if cid not in latest or latest[cid].get("error") or latest[cid].get("skipped")
    )
    _read_ended(manifest)
    _phase(paths, manifest, "read", status="done" if not failed else "done_with_errors",
           finished_at=_now(), collected=len(out), failed=failed)
    return True


def _rejection_message(rejected: list[dict], status: dict, *, waiting: bool = False) -> str:
    """The manifest message for a rejection.

    ``batch rejected at validation: <OpenAI message>; re-submitting in parts under
    <gate> enqueued tokens`` — or why it is not re-submitted automatically.
    """
    first = rejected[0]
    reason = str(first.get("message") or first.get("code") or "no reason given").strip()
    head = "batch rejected at validation" if len(rejected) == 1 else (
        f"{len(rejected)} batch parts rejected at validation"
    )
    head = f"{head}: {reason}"
    gate = status.get("max_enqueued_tokens")
    if not all(r.get("retryable") for r in rejected):
        return (f"{head}; nothing ran ($0). Not re-submitted automatically (the file itself "
                "was refused): fix the cause, then resume to re-plan the unanswered requests")
    tail = (f"re-submitting in parts under {int(gate):,} enqueued tokens" if gate
            else "re-submitting the unanswered requests")
    if waiting:
        tail = f"{tail} once the parts still running finish"
    return f"{head}; {tail}"


# ── Phase 3: score ───────────────────────────────────────────────────────────
def score_rows(
    run: LabRun, dataset: LabDataset, contexts: list[dict], responses: dict[str, dict],
    cell_to_id: dict[tuple[str, str, str], str] | None,
) -> list[dict]:
    """One scored row per (arm, budget, question). Pure: files in, rows out."""
    items = {item.id: item for item in dataset.items}
    read = run.mode != "retrieve"
    rows = []
    for ctx in _ordered_contexts(run, dataset, contexts):
        item = items[ctx["qid"]]
        row: dict[str, Any] = {
            "arm": ctx["arm"],
            "budget": ctx.get("budget"),
            "qid": ctx["qid"],
            "question": item.question,
            "gold": item.gold,
            "context_tokens": ctx.get("tokens"),
            "context_sha256": ctx.get("sha256"),
            "units_used": ctx.get("units_used"),
            "by_kind": ctx.get("by_kind") or {},
            "truncated": ctx.get("truncated"),
            "containment": ctx.get("containment"),
            "retrieval_error": ctx.get("retrieval_error"),
        }
        gold_titles = item.meta.get("gold_titles")
        credited = ctx.get("credited_gold")
        if gold_titles and credited is not None:
            strict = [t for t in gold_titles if credited.get(t) == "prose"]
            found = [t for t in gold_titles if t in credited]
            row["recall_permissive"] = len(found) / len(gold_titles)
            row["recall_strict"] = len(strict) / len(gold_titles)
            row["both_gold_permissive"] = float(len(found) == len(gold_titles))
            row["both_gold_strict"] = float(len(strict) == len(gold_titles))
        if read:
            key = (ctx["arm"], budget_label(ctx.get("budget")), ctx["qid"])
            custom_id = (cell_to_id or {}).get(key)
            response = responses.get(custom_id) if custom_id else None
            if ctx.get("retrieval_error"):
                # Never read (see plan_requests): counted, never scored or paired.
                row["read_error"] = f"retrieval failed: {ctx['retrieval_error']}"[:500]
                row.update({"answer": None, "em": 0.0, "f1": 0.0, "usd": 0.0})
            elif response is None or response.get("error") or response.get("skipped"):
                row["read_error"] = (response or {}).get("error") or "not answered"
                row.update({"answer": None, "em": 0.0, "f1": 0.0, "usd": 0.0})
            else:
                usage = response.get("usage") or {}
                em, f1 = metrics.answer_scores(response.get("answer"), item.gold)
                row.update(
                    {
                        "request_id": custom_id,
                        "answer": response.get("answer"),
                        "em": em,
                        "f1": f1,
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                        "cached_tokens": int(usage.get("cached_tokens") or 0),
                        "usd": response.get("usd"),
                        "finish_reason": response.get("finish_reason"),
                    }
                )
        rows.append(row)
    return rows


def _score_phase(
    run: LabRun, dataset: LabDataset, paths: RunPaths, manifest: dict[str, Any]
) -> dict[str, Any]:
    contexts = load_contexts(paths.contexts)
    read = run.mode != "retrieve"
    cell_to_id = None
    responses: dict[str, dict] = {}
    if read:
        _lines, cell_to_id = plan_requests(run, dataset, contexts)
        responses = _latest_responses(paths)
    rows = score_rows(run, dataset, contexts, responses, cell_to_id)
    _write_jsonl(paths.rows, rows)

    cells: dict[tuple[str, int | None], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["budget"]), []).append(row)
    arm_meta = {name: ARMS[name].catalog_entry() for name in run.arms}
    board = metrics.leaderboard(cells, arm_meta=arm_meta, ingest_usd=run.ingest_usd, read=read)
    board["run_id"] = run.run_id
    board["dataset"] = {"name": dataset.name, "split": dataset.split, "n": len(dataset.items)}
    board["reader_model"] = run.reader_model if read else None
    board["ingest_usd"] = run.ingest_usd
    if read and run.ingest_usd is None:
        board.setdefault("notes", []).append(
            "ingest cost unknown: amortized cost-of-pass for graph arms is not computed"
        )
    _write_json(paths.leaderboard, board)

    # Actuals per phase, next to the estimates.
    all_responses = read_jsonl(paths.responses)
    actual_reader = {
        "calls": sum(1 for r in all_responses if not r.get("skipped")),
        "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0)
                             for r in all_responses),
        "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                                 for r in all_responses),
        "reasoning_tokens": sum(int((r.get("usage") or {}).get("reasoning_tokens") or 0)
                                for r in all_responses),
        "cached_tokens": sum(int((r.get("usage") or {}).get("cached_tokens") or 0)
                             for r in all_responses),
        "usd": sum(float(r.get("usd") or 0.0) for r in all_responses),
        "note": "a failed request is counted at its reserved upper bound (it may have billed)",
    }
    manifest["actual"] = {
        "reader": actual_reader,
        "retrieval_llm": {"calls": 0, "usd": 0.0},
        "ingest": {"usd": run.ingest_usd},
    }
    est = (manifest.get("estimate") or {}).get("read") or {}
    ratios: dict[str, Any] = {}
    point, upper = est.get("total_point_usd"), est.get("total_upper_usd")
    if read and point:
        ratios["reader_actual_over_point"] = actual_reader["usd"] / point
    if read and upper:
        ratios["reader_actual_over_upper"] = actual_reader["usd"] / upper
    ingest_est = (manifest.get("estimate") or {}).get("ingest") or {}
    if run.ingest_usd is not None and ingest_est.get("point_usd"):
        ratios["ingest_actual_over_point"] = run.ingest_usd / ingest_est["point_usd"]
    manifest["estimate_vs_actual"] = ratios
    _phase(paths, manifest, "score", status="done", finished_at=_now(), rows=len(rows))
    paths.report.write_text(render_report(manifest, board), encoding="utf-8")
    return board


# ── Report ───────────────────────────────────────────────────────────────────
def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cmp(row: dict, key: str) -> str:
    c = (row.get("comparisons") or {}).get(key)
    if not c:
        return "—"
    mark = "" if c.get("reportable") else " (n.s.)"
    return f"{c['diff']:+.1f}{mark}"


def render_report(manifest: dict[str, Any], board: dict[str, Any]) -> str:
    """The human-readable report.md (public-safe: method, config, results)."""
    cfg = manifest.get("config") or {}
    ds = manifest.get("dataset") or {}
    code = manifest.get("code") or {}
    lines = [
        f"# Synapse Lab run `{manifest.get('run_id')}`",
        "",
        f"- Dataset: **{ds.get('name')}** (split {ds.get('split')}, n={ds.get('n')}, "
        f"sha256 `{str(ds.get('sha256'))[:12]}`)",
        f"- Mode: **{cfg.get('mode')}** · reader `{cfg.get('reader_model')}` · k={cfg.get('k')} "
        f"· seed {cfg.get('seed')} · packing order `{cfg.get('order')}`",
        f"- Arms: {', '.join(cfg.get('arms') or [])}",
        f"- Budgets: {', '.join(budget_label(b) for b in cfg.get('budgets') or [])} "
        f"(tokenizer `{manifest.get('tokenizer')}`)"
        + (
            "; extra cells: "
            + ", ".join(f"{a}@{budget_label(b)}" for a, b in cfg.get("extra_cells") or [])
            if cfg.get("extra_cells")
            else ""
        ),
        (
            f"- Code: `{code.get('git_sha')}`{' (dirty)' if code.get('git_dirty') else ''} · "
            if code.get("git_sha")
            else f"- Code: unknown ({code.get('git_unavailable') or 'git was unavailable'}) · "
        )
        + f"Synapse {code.get('synapse_version') or 'unknown'}",
        f"- Effect floor: {board.get('effect_floor_points', 0):.2f} points (one question); a "
        "difference is called only above it AND with a paired-bootstrap 95% CI excluding 0.",
        "",
    ]
    if board.get("mode") == "read":
        actual = (manifest.get("actual") or {}).get("reader") or {}
        est = (manifest.get("estimate") or {}).get("read") or {}
        lines += [
            f"Spend: {cost.format_usd(actual.get('usd'))} actual (reader) vs estimate "
            f"{cost.format_usd(est.get('total_point_usd'))} point / "
            f"{cost.format_usd(est.get('total_upper_usd'))} upper; cap "
            f"{cost.format_usd(manifest.get('budget_cap_usd'))}. Prices hand-recorded "
            f"({(manifest.get('prices') or {}).get('reader_checked_on')}), not live.",
            "",
            "| Arm | Budget | F1 | EM | tokens/correct | $ per 100 correct | vs N0 | vs N2 | "
            "graph premium |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in board.get("rows") or []:
            name = f"*{row['arm']}* (floor)" if row.get("is_null") else row["arm"]
            per100 = row.get("usd_per_100_correct")
            lines.append(
                f"| {name} | {row['budget_label']} | {_fmt(row.get('f1'))} | "
                f"{_fmt(row.get('em'))} | {_fmt(row.get('tokens_per_correct'), 0)} | "
                f"{cost.format_usd(per100) if per100 is not None else '—'} | "
                f"{_cmp(row, 'gain_above_n0')} | {_cmp(row, 'gain_above_n2')} | "
                f"{_cmp(row, 'graph_premium')} |"
            )
        lines += ["", "Pareto frontier (F1 vs $ per query): " + ", ".join(
            f"{p['arm']}@{budget_label(p['budget'])}"
            for p in (board.get("frontiers") or {}).get("f1_vs_usd") or []
        ) or "—"]
    else:
        lines += [
            "Retrieve-only run: no model was called ($0). Retrieval-level columns, floors first "
            "at each budget.",
            "",
            "| Arm | Budget | context tokens | units | containment % | recall permissive % | "
            "recall strict % |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in board.get("rows") or []:
            name = f"*{row['arm']}* (floor)" if row.get("is_null") else row["arm"]
            lines.append(
                f"| {name} | {row['budget_label']} | {_fmt(row.get('context_tokens_mean'), 0)} "
                f"| {_fmt(row.get('units_mean'))} | {_fmt(row.get('containment'))} | "
                f"{_fmt(row.get('recall_permissive'))} | {_fmt(row.get('recall_strict'))} |"
            )
        lines += [
            "",
            "Containment and permissive recall have no lower bound: the vocabulary null (N1) "
            "scores high on both while carrying no question-specific evidence. Read every row "
            "against the floors.",
        ]
    lines.append("")
    return "\n".join(lines)


# ── Entry points ─────────────────────────────────────────────────────────────
def _prepare(run: LabRun) -> RunPaths:
    run.validate()
    if run.run_dir is None:
        run.run_id = run.run_id or new_run_id(run)
        run.run_dir = run_dir_for(run.run_id)
    elif not run.run_id:
        run.run_id = Path(run.run_dir).name
    if "|" in str(run.run_id):
        raise LabError("run_id must not contain '|'")
    run.run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(Path(run.run_dir))


def pre_run_estimate(
    run: LabRun,
    dataset: LabDataset,
    context_tokens: dict[tuple[str, int | None], list[int]] | None = None,
) -> dict[str, Any]:
    """The estimate BEFORE retrieval: budgets bound the context, or pass MEASURED ones.

    Without ``context_tokens`` every capped cell is assumed to fill its budget
    and every uncapped cell to hold ``estimate.DEFAULT_CONTEXT_UPPER_TOKENS`` —
    a loose upper bound. ``run_lab`` records this estimate but does NOT refuse on
    it (retrieval is free); its binding gate is the re-estimate on the measured
    contexts just before reading. An API can refuse on it up front; to price a
    paid run tightly first, run it retrieve-only (free) and pass
    :func:`measured_context_tokens` of that run here.
    """
    est = estimate_run(
        questions=[item.question for item in dataset.items],
        arms=run.arms,
        budgets=run.budgets,
        cells=run.cells(),
        reader_model=run.reader_model,
        mode=run.mode,
        max_usd=run.max_usd,
        ingest=run.ingest_plan,
        context_tokens=context_tokens,
        reasoning_allowance=run.reasoning_allowance,
    )
    return est.to_dict()


def measured_context_tokens(run_dir: Path | str) -> dict[tuple[str, int | None], list[int]]:
    """Per-cell context token counts from a run's contexts.jsonl (e.g. a retrieve-only run).

    Last record per (arm, budget, question) wins, as everywhere else.
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for record in read_jsonl(RunPaths(Path(run_dir)).contexts):
        latest[(record["arm"], budget_label(record.get("budget")), record["qid"])] = record
    out: dict[tuple[str, int | None], list[int]] = {}
    for record in latest.values():
        out.setdefault((record["arm"], record.get("budget")), []).append(
            int(record.get("tokens") or 0)
        )
    return out


def check_dataset_unchanged(manifest: dict[str, Any], dataset: LabDataset) -> None:
    """Refuse to continue or re-score a run on a dataset that changed under it.

    A run's manifest records the dataset's content hash and question ids; its
    contexts, answers and scores belong to exactly those. A QA file replaced in
    place (``replace=true``) or an edited demo file would otherwise be scored
    silently against other golds, with questions dropped or added.
    """
    recorded = manifest.get("dataset") or {}
    problems = []
    old_sha = recorded.get("sha256") or ""
    if old_sha and dataset.sha256 != old_sha:
        problems.append(
            f"its content hash is {(dataset.sha256 or 'unknown')[:12]}…, "
            f"not {old_sha[:12]}…"
        )
    ids = [item.id for item in dataset.items]
    if recorded.get("question_ids") is not None and list(recorded["question_ids"]) != ids:
        problems.append(f"it yields {len(ids)} questions, not the {recorded.get('n')} recorded")
    if problems:
        raise DatasetChanged(
            f"dataset {dataset.source!r} changed since run {manifest.get('run_id')!r} started: "
            + "; ".join(problems)
            + ". Restore the original file, or start a new run."
        )


async def run_lab(
    run: LabRun,
    *,
    client: Any = None,
    batch_backend: BatchBackend | None = None,
    on_event: EventCallback | None = None,
    dataset: LabDataset | None = None,
) -> dict[str, Any]:
    """Run (or resume) a Lab run. Returns the final manifest.

    Statuses: ``done`` · ``refused`` (the measured upper bound breaks the cap;
    nothing was spent) · ``aborted`` (the metered cap stopped reading; resume
    with a higher cap) · ``batch_submitted`` (call :func:`resume` later) ·
    ``reading`` with a ``message`` (a batch was rejected at validation and not
    re-sent automatically; resume re-plans its unanswered requests).
    """
    paths = _prepare(run)
    dataset = dataset or load_dataset(run.dataset)
    if paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != run.config_hash():
            raise LabError(
                f"{paths.root} holds a different run (config hash "
                f"{manifest.get('config_hash')} ≠ {run.config_hash()}); pick another run_dir"
            )
        check_dataset_unchanged(manifest, dataset)
        manifest["config"].update({"max_usd": run.max_usd, "ingest_usd": run.ingest_usd})
        manifest["budget_cap_usd"] = run.max_usd
    else:
        manifest = _init_manifest(run, dataset)
        est = pre_run_estimate(run, dataset)
        manifest["estimate"]["pre_run"] = est
        if run.ingest_plan is not None:
            manifest["estimate"]["ingest"] = asdict(estimate_ingest(run.ingest_plan))
        _save(paths, manifest)

    try:
        if manifest["status"] == "batch_submitted":
            return await resume(paths.root, client=client, batch_backend=batch_backend,
                                on_event=on_event, dataset=dataset)
        if manifest["phases"]["retrieve"].get("status") != "done":
            manifest["graph"] = await check_graph_ready(dataset, run.arms)
            _save(paths, manifest, status="retrieving")
            await _retrieve_phase(run, dataset, paths, manifest, on_event)
            _save(paths, manifest, status="retrieved")

        if run.mode != "retrieve" and manifest["phases"]["read"].get("status") not in (
            "done", "done_with_errors"
        ):
            _save(paths, manifest, status="reading")
            status = await _read_phase(run, dataset, paths, manifest, client=client,
                                       batch_backend=batch_backend, on_event=on_event)
            if status in ("refused", "batch_submitted"):
                return manifest
            if status == "aborted":
                _score_phase(run, dataset, paths, manifest)
                _save(paths, manifest, status="aborted")
                await _emit(on_event, {"type": "done", "status": "aborted"})
                return manifest

        _score_phase(run, dataset, paths, manifest)
        _save(paths, manifest, status="done", finished_at=_now())
        await _emit(on_event, {"type": "done", "status": "done"})
        return manifest
    except Exception as e:
        _save(paths, manifest, status="failed", error=f"{type(e).__name__}: {e}"[:1000])
        await _emit(on_event, {"type": "error", "error": str(e)})
        raise


async def resume(
    run_dir: Path | str,
    *,
    client: Any = None,
    batch_backend: BatchBackend | None = None,
    on_event: EventCallback | None = None,
    dataset: LabDataset | None = None,
    max_usd: float | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Continue a run from its manifest: poll/collect a batch, or finish what is left.

    ``max_usd`` (optional) raises or lowers the cap for the rest of the run —
    the way to continue a run the metered cap aborted. ``retry_failed`` re-reads
    the requests that came back with an error (realtime, or a new batch for a
    batch run) — nothing that already has an answer is ever re-sent.

    A batch OpenAI rejected at validation (e.g. the organisation's enqueued-
    token limit) is never scored as errors: the run goes back to ``reading``
    and, when the rejection is a capacity one, its unanswered requests are
    re-planned right away — re-priced against the remaining cap (the rejected
    batch cost $0) and sent in parts under the enqueued-token gate the
    rejection set (see :func:`collect_batch` and ``app.lab.batch``). Requests a
    rejected batch owed are re-planned even when they carry an older error line
    (a refused ``retry_failed`` batch), with or without ``retry_failed``.
    """
    paths = RunPaths(Path(run_dir))
    if not paths.manifest.exists():
        raise LabError(f"no Lab run at {paths.root}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    run = LabRun.from_dict({**manifest["config"], "run_dir": paths.root})
    if max_usd is not None:
        run.max_usd = max_usd
    dataset = dataset or load_dataset(run.dataset)
    check_dataset_unchanged(manifest, dataset)
    if manifest.get("status") == "batch_submitted":
        backend = batch_backend or ModuleBatchBackend(client)
        try:
            finished = await collect_batch(run, paths, manifest, backend, on_event)
        except Exception as e:
            _save(paths, manifest, error=f"{type(e).__name__}: {e}"[:1000])
            raise
        if manifest.get("status") == "reading":  # rejected at validation: nothing ran
            if manifest["phases"]["read"].get("resubmit") != "auto":
                return manifest  # the file itself was refused: the owner decides
            if retry_failed:
                manifest["phases"]["read"].pop("only_unanswered", None)
                _save(paths, manifest)
            return await run_lab(run, client=client, batch_backend=backend,
                                 on_event=on_event, dataset=dataset)
        if not finished:
            return manifest
        _score_phase(run, dataset, paths, manifest)
        _save(paths, manifest, status="done", finished_at=_now())
        await _emit(on_event, {"type": "done", "status": "done"})
        return manifest
    read_status = manifest["phases"]["read"].get("status")
    if retry_failed and manifest["phases"]["read"].pop("only_unanswered", None):
        _save(paths, manifest)
    if manifest.get("status") in ("aborted", "refused") or (
        retry_failed and read_status == "done_with_errors"
    ):
        manifest["phases"]["read"]["status"] = "pending"
        _save(paths, manifest, status="reading")
    return await run_lab(run, client=client, batch_backend=batch_backend, on_event=on_event,
                         dataset=dataset)


# ── Reading runs back (for the API / CLI) ────────────────────────────────────
def list_runs(root: Path | None = None) -> list[dict[str, Any]]:
    """Every run under ``root`` (newest first): id, status, dataset, mode, created_at."""
    base = Path(root) if root else LAB_RUNS_DIR
    if not base.is_dir():
        return []
    out = []
    for child in base.iterdir():
        if child.name.startswith(".") or child.name.lower() in RESERVED_RUN_IDS:
            continue  # the QA-file store (or a stray file), never a run
        path = child / MANIFEST
        if not path.is_file():
            continue
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cfg = m.get("config") or {}
        out.append(
            {
                "run_id": m.get("run_id") or child.name,
                "status": m.get("status"),
                "created_at": m.get("created_at"),
                "updated_at": m.get("updated_at"),
                "mode": cfg.get("mode"),
                "dataset": (m.get("dataset") or {}).get("name"),
                "n": (m.get("dataset") or {}).get("n"),
                "arms": cfg.get("arms"),
                "budgets": cfg.get("budgets"),
                "reader_model": cfg.get("reader_model"),
                "run_dir": str(child),
            }
        )
    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return out


def check_run_id(run_id: str) -> str:
    """``run_id`` when it can name a directory under ``LAB_RUNS_DIR``, else ``LabError``.

    One path segment of letters, digits, ``.``, ``_`` or ``-``; not hidden (so
    never ``.`` or ``..``); not one of :data:`RESERVED_RUN_IDS`.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or "") or run_id.startswith("."):
        raise LabError(f"invalid run id {run_id!r}")
    if run_id.lower() in RESERVED_RUN_IDS:
        raise LabError(f"run id {run_id!r} is reserved (the Lab keeps its own files there)")
    return run_id


def run_dir_for(run_id: str, root: Path | None = None) -> Path:
    """The directory of ``run_id`` under ``root``; refuses path tricks and reserved names."""
    return (Path(root) if root else LAB_RUNS_DIR) / check_run_id(run_id)


def load_run(run_dir: Path | str) -> dict[str, Any]:
    """``{"manifest", "leaderboard"}`` of a run (leaderboard ``None`` until scored)."""
    paths = RunPaths(Path(run_dir))
    if not paths.manifest.exists():
        raise LabError(f"no Lab run at {paths.root}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    board = None
    if paths.leaderboard.exists():
        board = json.loads(paths.leaderboard.read_text(encoding="utf-8"))
    return {"manifest": manifest, "leaderboard": board}


def read_rows(
    run_dir: Path | str,
    *,
    offset: int = 0,
    limit: int = 100,
    arm: str | None = None,
    budget: Any = "any",
) -> dict[str, Any]:
    """A page of scored per-question rows, optionally for one arm / budget."""
    rows = read_jsonl(RunPaths(Path(run_dir)).rows)
    if arm:
        rows = [r for r in rows if r.get("arm") == arm]
    if budget != "any":
        wanted = parse_budget(budget)
        rows = [r for r in rows if r.get("budget") == wanted]
    offset, limit = max(0, int(offset)), max(0, int(limit))
    return {"total": len(rows), "offset": offset, "limit": limit,
            "rows": rows[offset : offset + limit]}


def qa_file_path(name: str, root: Path | None = None) -> Path:
    """The stored upload called ``name`` — a bare ``*.json`` / ``*.jsonl`` filename only.

    For request handlers: a user-supplied dataset must resolve INSIDE
    ``lab_runs/datasets``, never to an arbitrary path on the server.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(json|jsonl)", name or ""):
        raise LabError(f"invalid QA file name {name!r} (expected a bare .json/.jsonl filename)")
    return (Path(root) if root else DATASETS_DIR) / name


def list_qa_files(root: Path | None = None) -> list[dict[str, Any]]:
    """Uploaded QA files under ``lab_runs/datasets`` (name, size, question count or error)."""
    base = Path(root) if root else DATASETS_DIR
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.iterdir()):
        if path.suffix.lower() not in {".json", ".jsonl"} or not path.is_file():
            continue
        entry: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
        try:
            records = read_qa_file(path, label=path.name)
            entry["n"] = len(parse_qa_records(records, source=path.name))
        except LabError as e:
            entry["error"] = str(e)
        out.append(entry)
    return out
