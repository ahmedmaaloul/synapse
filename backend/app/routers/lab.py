# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Lab Router (compare retrieval approaches on your own data, FinOps-first)

The HTTP face of ``app.lab``. The client package, the CLI and the UI code
against these routes, so their names and shapes are a contract:

    GET  /api/lab/arms                  the arms (floors · passage baselines · graph arms)
    GET  /api/lab/models                reader models with their $/1M tokens and reasoning flag
    GET  /api/lab/datasets              demo, uploaded QA files, HotpotQA when its graph is present
    POST /api/lab/qa-files              upload a QA JSON / JSONL (validated, then stored)
    POST /api/lab/estimate              the free dry-run estimate: point, upper bound, refuse
    POST /api/lab/runs                  start a run (background job)
    GET  /api/lab/runs                  every run, newest first
    GET  /api/lab/runs/{id}             manifest, leaderboard, frontiers and a page of rows
    GET  /api/lab/runs/{id}/events      the run's job progress (SSE; ``done`` carries the status)
    POST /api/lab/runs/{id}/resume      poll/collect a batch, or finish an aborted/refused run

MONEY. ``mode`` is ``retrieve`` (the default and the $0 tier: no model is
called), ``realtime`` or ``batch``. A paid mode needs ``max_usd``, and a run
whose estimated UPPER BOUND exceeds it is refused before anything starts: a 422
whose ``detail.estimate`` is the estimate. The runner then re-checks on the
MEASURED contexts after the free retrieve phase, and meters realtime spend
call by call, so the cap is enforced at three points. Nothing here spends by
default, and no MCP tool can start a run.

BATCH mode goes through ``app.lab.batch`` (the OpenAI Batch layer). When that
module is missing or incomplete, batch requests are a 501 naming what is
missing, and everything else keeps working.

JOBS. A run is long, so it follows the ``job_bus`` + SSE pattern used for
ingestion and evolution: ``POST /runs`` returns ``{job_id, run_id}`` at once,
and the run proceeds in the background. This includes batch mode, whose free
retrieve phase runs before anything is submitted.
The stream relays the runner's ``phase`` / ``progress`` / ``refused`` /
``aborted`` / ``batch_submitted`` events and ends with ``done``, whose
``data.status`` is the run's status (``done``, ``batch_submitted``,
``refused`` or ``aborted``), or with ``error``. One job per run at a time in this
process (409 otherwise).

PATHS. A QA file is addressed by its bare filename under ``lab_runs/datasets``
and a run by its id under ``lab_runs``; neither can resolve anywhere else.
HotpotQA is read from the local cache only: the API never downloads it.

The Lab never writes to the knowledge graph (every arm is read-only). Retrieval
runs on this process's event loop, and the bootstrap in scoring is CPU-bound
(vectorised with numpy), so a very large run can hold the loop for a few
seconds while it scores.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.lab import reader, runner
from app.lab.arms import FAMILY_TITLES, arm_catalog
from app.lab.estimate import INGEST_MODEL, IngestPlan, estimate_run, load_calibration
from app.lab.tokens import tokenizer_label
from app.services.jobs import TERMINAL_TYPES, job_bus
from app.services.llm_provider import is_reasoning_model
from benchmarks.public import cost

logger = logging.getLogger(__name__)
router = APIRouter()

QA_FILE_PREFIX = "qa-file:"
#: An uploaded QA file is questions and answers; 20 MB is tens of thousands of them.
MAX_QA_FILE_BYTES = 20 * 1024 * 1024
MAX_BUDGET_TOKENS = 1_000_000
#: The budgets the UI offers by default (``None`` = each arm's default context).
SUGGESTED_BUDGETS: list[int | None] = [500, 1000, 2000, 4000, 8000, None]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")

#: run id → its latest job id. Kept after the job ends so that a client that
#: subscribes late still drains the events the job queued.
_jobs_by_run: dict[str, str] = {}
#: Run ids with a job in flight in this process.
_active_runs: set[str] = set()

EventCallback = Callable[[dict], Awaitable[None]]
Work = Callable[[EventCallback], Awaitable[dict[str, Any]]]


# ── Request models ───────────────────────────────────
class CellRequest(BaseModel):
    """One (arm, budget) cell beyond the ``arms × budgets`` grid."""

    arm: str = Field(..., min_length=1, max_length=64)
    budget: int | str | None = None


class IngestRequest(BaseModel):
    """An ingest plan to price next to the run (the Lab itself never ingests).

    ``paragraphs`` is filled from the dataset when it knows its corpus
    (HotpotQA); any other dataset must state it.
    """

    paragraphs: int | None = Field(None, ge=0, le=10_000_000)
    model: str = Field(INGEST_MODEL, min_length=1, max_length=128)
    batch: bool = True
    community_summaries: int = Field(0, ge=0, le=1_000_000)


class LabRequest(BaseModel):
    """What to compare, on which questions, with which reader, under which cap."""

    # "demo" | "qa-file" | "hotpotqa", or "qa-file:<name>" (the id GET /datasets lists).
    dataset: str = Field("demo", min_length=1, max_length=300)
    # qa-file: the uploaded file's name (see POST /qa-files).
    file: str | None = Field(None, max_length=255)
    split: str | None = Field(None, max_length=64)
    n: int | None = Field(None, ge=1, le=100_000)
    # hotpotqa only: skip the first ``offset`` sampled questions (disjoint samples).
    offset: int = Field(0, ge=0, le=1_000_000)
    # hotpotqa only: the sampling seed (default: the benchmark's).
    sample_seed: int | None = None
    # Omitted: every arm.
    arms: list[str] | None = Field(None, min_length=1, max_length=64)
    budgets: list[int | str | None] = Field(default_factory=lambda: [None], min_length=1, max_length=32)
    extra_cells: list[CellRequest] = Field(default_factory=list, max_length=64)
    k: int = Field(runner.DEFAULT_K, ge=1, le=100)
    # None: passage-ranking arms fill each capped budget (k at the default context);
    # an int: exactly that many passages at every budget.
    passage_k: int | None = Field(None, ge=1, le=100)
    seed: int = runner.DEFAULT_SEED
    reader_model: str = Field(runner.DEFAULT_READER, min_length=1, max_length=128)
    mode: Literal["retrieve", "realtime", "batch"] = "retrieve"
    max_usd: float | None = Field(None, ge=0, le=100_000)
    order: Literal["score", "ascending"] = "score"
    reasoning_allowance: int | None = Field(None, ge=0, le=100_000)
    max_concurrency: int = Field(4, ge=1, le=32)
    ingest: IngestRequest | None = None
    # The corpus's MEASURED ingest spend, for amortized cost-of-pass.
    ingest_usd: float | None = Field(None, ge=0)
    # Price on the MEASURED contexts of an earlier (retrieve-only) run of the same config.
    measured_run_id: str | None = Field(None, max_length=128)
    # POST /runs only: name the run (default: timestamp + config hash).
    run_id: str | None = Field(None, max_length=128)

    @field_validator("dataset")
    @classmethod
    def _known_dataset(cls, value: str) -> str:
        value = value.strip()
        if value.startswith(QA_FILE_PREFIX):
            if not value[len(QA_FILE_PREFIX) :].strip():
                raise ValueError("'qa-file:' needs the uploaded file's name after the colon")
            return value
        if value not in runner.DATASETS:
            raise ValueError(
                f"unknown dataset {value!r} (expected {', '.join(runner.DATASETS)} or "
                f"'{QA_FILE_PREFIX}<name>')"
            )
        return value

    @field_validator("budgets")
    @classmethod
    def _valid_budgets(cls, values: list[int | str | None]) -> list[int | None]:
        return [_budget(v) for v in values]

    @field_validator("run_id", "measured_run_id")
    @classmethod
    def _valid_run_id(cls, value: str | None) -> str | None:
        if value is not None and (
            not RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}
        ):
            raise ValueError("a run id is letters, digits, '.', '_' or '-' only")
        if value is not None:
            try:
                runner.check_run_id(value)
            except runner.LabError as e:
                raise ValueError(str(e)) from e
        return value


class ResumeRequest(BaseModel):
    # A new cap for the rest of the run: how an aborted or refused run continues.
    max_usd: float | None = Field(None, ge=0, le=100_000)
    # Re-read the requests that came back with an error (never one with an answer).
    retry_failed: bool = False


# ── Helpers ──────────────────────────────────────────
class _NotFound(Exception):
    """A run or QA file that does not exist (→ 404)."""


def _budget(value: Any) -> int | None:
    try:
        budget = runner.parse_budget(value)
    except (runner.LabError, TypeError, ValueError) as e:
        raise ValueError(f"a budget is a positive token count or 'default', got {value!r}") from e
    if budget is not None and budget > MAX_BUDGET_TOKENS:
        raise ValueError(f"a budget above {MAX_BUDGET_TOKENS:,} tokens is not a budget")
    return budget


def _refusal(message: str, *diagnostics: str, **extra: Any) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"message": message, "diagnostics": [d for d in diagnostics if d], **extra},
    )


@contextmanager
def _lab_errors() -> Iterator[None]:
    """Map the Lab's domain errors to the contract's status codes."""
    try:
        yield
    except _NotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (runner.GraphNotIngested, runner.DatasetChanged) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except runner.LabError as e:
        raise _refusal("Invalid Lab request", str(e)) from e
    except ValueError as e:
        raise _refusal("Invalid Lab request", str(e)) from e


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _batch_backend() -> Any:
    """The Batch layer a batch run submits to: ``app.lab.batch`` via the runner's adapter.

    Raises :class:`runner.LabError` when the module is missing or lacks the
    functions the runner calls.
    """
    return runner.ModuleBatchBackend()


def _batch_or_501() -> Any:
    try:
        return _batch_backend()
    except Exception as e:  # noqa: BLE001 - a broken optional module must not be a 500
        raise HTTPException(
            status_code=501,
            detail=f"Batch mode is not available in this build: {e}",
        ) from e


def _batch_available() -> bool:
    try:
        _batch_backend()
    except Exception:  # noqa: BLE001
        return False
    return True


def _reader_client() -> Any:
    """The client realtime reads go through (``None``: the runner builds the official one)."""
    return None


def _require_openai_key() -> None:
    if not get_settings().openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set: the Lab reader calls the OpenAI API "
            "(the retrieve mode needs no key and costs nothing).",
        )


def _manifest_path(run_id: str) -> Path:
    try:
        run_dir = runner.run_dir_for(run_id)
    except runner.LabError as e:
        raise _NotFound(f"Unknown Lab run {run_id!r}.") from e
    path = run_dir / runner.MANIFEST
    if not path.is_file():
        raise _NotFound(f"Unknown Lab run {run_id!r}.")
    return path


def _read_manifest(run_id: str) -> dict[str, Any]:
    path = _manifest_path(run_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise runner.LabError(f"run {run_id!r} has an unreadable manifest: {e}") from e


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """What a finished job reports: status, why it stopped, and what it cost."""
    status = manifest.get("status")
    reason = None
    if status == "refused":
        reason = manifest.get("refuse_reason")
    elif status == "aborted":
        reason = manifest.get("abort_reason")
    elif status == "failed":
        reason = manifest.get("error")
    estimate = manifest.get("estimate") or {}
    dataset = manifest.get("dataset") or {}
    return {
        "run_id": manifest.get("run_id"),
        "status": status,
        "mode": (manifest.get("config") or {}).get("mode"),
        "dataset": {k: dataset.get(k) for k in ("name", "split", "n")},
        "reason": reason,
        "phases": {name: (p or {}).get("status") for name, p in (manifest.get("phases") or {}).items()},
        "spent_usd": ((manifest.get("actual") or {}).get("reader") or {}).get("usd"),
        "cap_usd": manifest.get("budget_cap_usd"),
        "pending": estimate.get("read_pending"),
        "estimate_upper_usd": (estimate.get("read") or estimate.get("pre_run") or {}).get(
            "total_upper_usd"
        ),
    }


def _start_job(background_tasks: BackgroundTasks, run_id: str, work: Work) -> str:
    job_id = job_bus.create()
    _active_runs.add(run_id)
    _jobs_by_run[run_id] = job_id
    background_tasks.add_task(_run_job, job_id, run_id, work)
    return job_id


async def _run_job(job_id: str, run_id: str, work: Work) -> None:
    """Background task: run (or resume) a Lab run and stream its events on the job bus."""

    async def on_event(event: dict) -> None:
        # The runner's own done/error events would end the stream early; the job
        # publishes the ending itself, with the run's final state.
        if event.get("type") in TERMINAL_TYPES:
            return
        await job_bus.publish(job_id, {**event, "run_id": run_id})

    try:
        manifest = await work(on_event)
        summary = _summary(manifest)
        await job_bus.publish(job_id, {"type": "done", "data": summary})
        logger.info("🧪 Lab run %s: %s", run_id, summary["status"])
    except Exception as e:  # noqa: BLE001
        logger.exception("❌ Lab run %s failed", run_id)
        await job_bus.publish(job_id, {"type": "error", "data": f"{type(e).__name__}: {e}"})
    finally:
        _active_runs.discard(run_id)


# ── Datasets ─────────────────────────────────────────
def _split_counts(items: list[runner.LabItem]) -> dict[str, int]:
    return dict(sorted(Counter(i.meta.get("split") for i in items if i.meta.get("split")).items()))


def _demo_entry() -> dict[str, Any]:
    items = runner.parse_qa_records(runner.read_qa_file(runner.DEMO_QA_PATH), source="demo")
    splits = _split_counts(items)
    return {
        "id": "demo",
        "name": "demo",
        "title": "Demo QA set",
        "splits": splits,
        "default_split": "test",
        "n": splits.get("test", len(items)),
        "source": runner.DEMO_QA_PATH.name,
    }


def _qa_file_entries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = runner.DATASETS_DIR
    ok: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    if not root.is_dir():
        return ok, bad
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        entry: dict[str, Any] = {
            "id": f"{QA_FILE_PREFIX}{path.name}",
            "name": "qa-file",
            "file": path.name,
            "title": path.name,
            "bytes": path.stat().st_size,
        }
        try:
            records = runner.read_qa_file(path, label=path.name)
            items = runner.parse_qa_records(records, source=path.name)
        except runner.LabError as e:
            bad.append({**entry, "reason": str(e)})
            continue
        ok.append({**entry, "n": len(items), "splits": _split_counts(items)})
    return ok, bad


def _raw_titles(raw: Any) -> Iterator[str]:
    """Paragraph titles of one raw HotpotQA record, in either encoding (lenient)."""
    context = raw.get("context") if isinstance(raw, dict) else None
    if isinstance(context, dict):
        yield from (t for t in context.get("title") or [] if isinstance(t, str) and t)
    elif isinstance(context, list):
        for entry in context:
            if isinstance(entry, list | tuple) and entry and isinstance(entry[0], str) and entry[0]:
                yield entry[0]


#: (path, mtime_ns, size) → every paragraph title in the cached dev set.
_hotpot_titles: dict[tuple[str, int, int], frozenset[str]] = {}


def _hotpotqa_titles(path: Path) -> frozenset[str]:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hotpot_titles:
        from benchmarks.public import hotpotqa

        titles = frozenset(t for raw in hotpotqa.load_raw_records(path) for t in _raw_titles(raw))
        _hotpot_titles.clear()
        _hotpot_titles[key] = titles
    return _hotpot_titles[key]


async def _hotpotqa_entry() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(available entry, None)`` or ``(None, unavailable entry with a reason)``.

    Present = the local HotpotQA cache exists AND the graph holds at least one
    document named after a HotpotQA paragraph (how the Lab and the benchmark
    ingest it: one document per paragraph). Whether a given SAMPLE is fully
    ingested is checked by each run, which refuses otherwise.
    """
    from benchmarks.public import hotpotqa, run_hotpotqa

    base = {
        "id": "hotpotqa",
        "name": "hotpotqa",
        "title": "HotpotQA (dev, distractor)",
        "split": "dev-distractor",
        "default_n": hotpotqa.DEFAULT_N,
        "default_seed": hotpotqa.DEFAULT_SEED,
        "license": run_hotpotqa.HOTPOTQA_LICENSE,
    }
    path = hotpotqa.CACHE_PATH
    if not path.is_file():
        return None, {
            **base,
            "reason": "no local HotpotQA cache (the API never downloads it; fetch it with "
            "`python -m benchmarks.public.hotpotqa` in backend/)",
        }
    try:
        documents = await run_hotpotqa.graph_documents()
    except Exception as e:  # noqa: BLE001 - a listing must survive a Neo4j outage
        return None, {**base, "reason": f"could not read the knowledge graph: {e}"}
    if not documents:
        return None, {**base, "reason": "the knowledge graph holds no documents"}
    try:
        titles = await asyncio.to_thread(_hotpotqa_titles, path)
    except (OSError, ValueError) as e:
        return None, {**base, "reason": f"the HotpotQA cache is unreadable: {e}"}
    present = len(documents & titles)
    if not present:
        return None, {
            **base,
            "reason": "no HotpotQA paragraph is ingested in the knowledge graph",
            "graph_documents": len(documents),
        }
    return {
        **base,
        "graph_paragraphs": present,
        "graph_documents": len(documents),
        "note": "each run checks that EVERY paragraph of its own sample is ingested, "
        "and refuses otherwise",
    }, None


def _dataset_spec(request: LabRequest) -> runner.DatasetSpec:
    name, file = request.dataset, request.file
    if name.startswith(QA_FILE_PREFIX):
        name, file = "qa-file", name[len(QA_FILE_PREFIX) :].strip()
    path: str | None = None
    if name == "qa-file":
        if not file:
            raise runner.LabError(
                "dataset 'qa-file' needs 'file': the name of an uploaded QA file "
                "(see GET /api/lab/datasets)"
            )
        target = runner.qa_file_path(file)
        if not target.is_file():
            raise _NotFound(f"No uploaded QA file named {file!r} (upload it first).")
        path = str(target)
    elif file:
        raise runner.LabError("'file' applies to the qa-file dataset only")
    if name != "hotpotqa" and (request.offset or request.sample_seed is not None):
        raise runner.LabError("'offset' and 'sample_seed' apply to the hotpotqa dataset only")
    return runner.DatasetSpec(
        name=name,
        split=request.split,
        n=request.n,
        path=path,
        seed=request.sample_seed if name == "hotpotqa" else None,
        offset=request.offset if name == "hotpotqa" else 0,
        allow_download=False,
    )


def _to_run(request: LabRequest) -> runner.LabRun:
    run = runner.LabRun(
        dataset=_dataset_spec(request),
        arms=list(request.arms) if request.arms else list(runner.ARMS),
        budgets=list(request.budgets),
        k=request.k,
        seed=request.seed,
        reader_model=request.reader_model.strip(),
        mode=request.mode,
        max_usd=request.max_usd,
        run_id=request.run_id,
        order=request.order,
        passage_k=request.passage_k,
        reasoning_allowance=request.reasoning_allowance,
        max_concurrency=request.max_concurrency,
        ingest_usd=request.ingest_usd,
        extra_cells=[(c.arm, _budget(c.budget)) for c in request.extra_cells],
    )
    run.validate()
    return run


async def _load_dataset(run: runner.LabRun) -> runner.LabDataset:
    return await asyncio.to_thread(runner.load_dataset, run.dataset)


def _ingest_plan(request: LabRequest, dataset: runner.LabDataset) -> IngestPlan | None:
    if request.ingest is None:
        return None
    paragraphs = request.ingest.paragraphs
    if paragraphs is None:
        if not dataset.titles:
            raise runner.LabError(
                "ingest.paragraphs is required: only hotpotqa knows its corpus size"
            )
        paragraphs = len(dataset.titles)
    return IngestPlan(
        paragraphs=paragraphs,
        model=request.ingest.model,
        batch=request.ingest.batch,
        community_summaries=request.ingest.community_summaries,
    )


def _measured(
    request: LabRequest, run: runner.LabRun, dataset: runner.LabDataset
) -> tuple[dict[tuple[str, int | None], list[int]] | None, dict[str, Any] | None]:
    """Context tokens MEASURED by an earlier run, when it retrieved exactly what this run will."""
    if not request.measured_run_id:
        return None, None
    source = request.measured_run_id
    manifest = _read_manifest(source)
    cfg = manifest.get("config") or {}
    ds = manifest.get("dataset") or {}
    problems = []
    if ds.get("question_ids") != [item.id for item in dataset.items] or (
        ds.get("sha256") != dataset.sha256
    ):
        problems.append("it ran on other questions")
    for key in ("k", "passage_k", "order", "seed"):
        if cfg.get(key) != getattr(run, key):
            problems.append(f"its {key} is {cfg.get(key)!r}, not {getattr(run, key)!r}")
    if (manifest.get("packer") or {}).get("passages") != run.passage_policy():
        problems.append("it chose its passage counts another way (a run from an older version)")
    if tokenizer_label(cfg.get("reader_model")) != tokenizer_label(run.reader_model):
        problems.append("it counted tokens with another tokenizer")
    arms = manifest.get("arms") or {}
    changed = [
        a for a in run.arms
        if a in arms and a in runner.ARMS and arms[a].get("config_hash") != runner.ARMS[a].config_hash()
    ]
    if changed:
        problems.append(f"arm(s) changed since: {', '.join(changed)}")
    if ((manifest.get("phases") or {}).get("retrieve") or {}).get("status") != "done":
        problems.append("its retrieve phase is not done")
    if problems:
        raise runner.LabError(f"run {source!r} cannot price this one: " + "; ".join(problems))
    tokens = runner.measured_context_tokens(runner.run_dir_for(source))
    cells = [c for c in run.cells() if c in tokens]
    return tokens, {
        "run_id": source,
        "cells": [{"arm": a, "budget": b} for a, b in cells],
        "unmeasured": [{"arm": a, "budget": b} for a, b in run.cells() if (a, b) not in tokens],
    }


def _estimate(
    run: runner.LabRun,
    dataset: runner.LabDataset,
    measured: dict[tuple[str, int | None], list[int]] | None,
) -> dict[str, Any]:
    """The estimate both /estimate and /runs show: the calibration file applies when present."""
    est = estimate_run(
        questions=[item.question for item in dataset.items],
        arms=run.arms,
        budgets=run.budgets,
        cells=run.cells(),
        reader_model=run.reader_model,
        mode=run.mode,
        max_usd=run.max_usd,
        ingest=run.ingest_plan,
        context_tokens=measured,
        reasoning_allowance=run.reasoning_allowance,
        calibration=load_calibration(),
    )
    return est.to_dict()


async def _prepare(
    request: LabRequest,
) -> tuple[runner.LabRun, runner.LabDataset, dict[str, Any]]:
    """Validate, load the questions and price the run. Free: no model, no Neo4j."""
    run = _to_run(request)
    dataset = await _load_dataset(run)
    run.ingest_plan = _ingest_plan(request, dataset)
    measured, measured_from = _measured(request, run, dataset)
    est = await asyncio.to_thread(_estimate, run, dataset, measured)
    est["dataset"] = {
        "name": dataset.name,
        "split": dataset.split,
        "n": len(dataset.items),
        "source": dataset.source,
        "sha256": dataset.sha256,
    }
    est["measured_from"] = measured_from
    est["config_hash"] = run.config_hash()
    est["cells_count"] = len(run.cells())
    return run, dataset, est


# ── Catalogue ────────────────────────────────────────
@router.get("/lab/arms")
async def list_arms():
    """Every arm with its family, one-line description, source and retrieval-LLM cost.

    Also what a run can be configured with: the modes (``retrieve`` is the free
    default), the suggested budgets, and whether batch mode is available.
    """
    return {
        "arms": arm_catalog(),
        "families": [{"family": f, "title": t} for f, t in FAMILY_TITLES.items()],
        "modes": list(runner.MODES),
        "default_mode": "retrieve",
        "budgets": SUGGESTED_BUDGETS,
        "batch_available": _batch_available(),
    }


@router.get("/lab/models")
async def list_models():
    """Reader models with their hand-recorded $/1M tokens (realtime and Batch)."""
    models = []
    for name, price in cost.PRICES_USD_PER_1M_TOKENS.items():
        if price.output_usd_per_1m <= 0:  # an embedding model, not a reader
            continue
        batch = cost.batch_price(price)
        models.append(
            {
                "name": name,
                "input_usd_per_1m": price.input_usd_per_1m,
                "output_usd_per_1m": price.output_usd_per_1m,
                "batch_input_usd_per_1m": batch.input_usd_per_1m if batch else None,
                "batch_output_usd_per_1m": batch.output_usd_per_1m if batch else None,
                "reasoning": is_reasoning_model(name),
                "max_output_tokens": reader.max_output_tokens(name),
                "tokenizer": tokenizer_label(name),
                "price_checked_on": cost.price_checked_on(name),
            }
        )
    return {
        "models": models,
        "default": runner.DEFAULT_READER,
        "batch_multiplier": cost.BATCH_PRICE_MULTIPLIER,
        "pricing_url": cost.PRICING_URL,
        "note": "prices are hand-recorded and dated, not fetched live; a reasoning model "
        "bills its hidden reasoning tokens as output",
    }


@router.get("/lab/datasets")
async def list_datasets():
    """Datasets a run can use now, and the ones that are not usable yet (with the reason)."""
    with _lab_errors():
        demo = await asyncio.to_thread(_demo_entry)
    uploaded, broken = await asyncio.to_thread(_qa_file_entries)
    hotpot, hotpot_missing = await _hotpotqa_entry()
    datasets = [demo, *uploaded] + ([hotpot] if hotpot else [])
    unavailable = [*broken] + ([hotpot_missing] if hotpot_missing else [])
    return {"datasets": datasets, "unavailable": unavailable}


# ── QA files ─────────────────────────────────────────
def _safe_qa_name(raw: str | None) -> str:
    """A bare, portable ``*.json`` / ``*.jsonl`` filename from what the client sent."""
    name = Path((raw or "").replace("\\", "/")).name.strip()
    stem, dot, suffix = name.rpartition(".")
    if not dot or suffix.lower() not in {"json", "jsonl"}:
        raise runner.LabError("a QA file must be named *.json or *.jsonl")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    if not stem:
        raise runner.LabError(f"{raw!r} is not a usable file name")
    return runner.qa_file_path(f"{stem}.{suffix.lower()}").name


@router.post("/lab/qa-files")
async def upload_qa_file(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    replace: bool = Form(False),
):
    """Store a QA set of ``{question, answer[, id][, split]}`` records (JSON array or JSONL).

    Validated in full before it is stored: every record needs a question and a
    non-empty answer, and ids must be unique. Storing over an existing file with
    DIFFERENT content is a 409 unless ``replace`` is set: past runs record the
    file's sha256 and question ids, and a run whose file changed since is never
    resumed or re-scored against the new content (``runner.DatasetChanged``, 409).
    """
    with _lab_errors():
        filename = _safe_qa_name(name or file.filename)
        contents = await file.read(MAX_QA_FILE_BYTES + 1)
        if len(contents) > MAX_QA_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"QA file larger than {MAX_QA_FILE_BYTES // (1024 * 1024)} MB",
            )
        try:
            contents.decode("utf-8")
        except UnicodeDecodeError as e:
            raise runner.LabError(f"{filename} is not UTF-8 text") from e

        root = runner.DATASETS_DIR
        incoming = root / ".incoming"
        await asyncio.to_thread(incoming.mkdir, parents=True, exist_ok=True)
        # Same suffix as the target: the reader picks JSON vs JSONL by suffix.
        staged = incoming / f"{uuid.uuid4().hex}{Path(filename).suffix}"
        staged.write_bytes(contents)
        try:
            records = runner.read_qa_file(staged, label=filename)
            items = runner.parse_qa_records(records, source=filename)
            target = runner.qa_file_path(filename)
            sha = hashlib.sha256(contents).hexdigest()
            status = "created"
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() == sha:
                    status = "unchanged"
                elif not replace:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A different QA file named {filename!r} already exists; past "
                        "runs refer to it by content hash. Upload under another name, or set "
                        "replace=true.",
                    )
                else:
                    status = "replaced"
            if status != "unchanged":
                os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
    return {
        "status": status,
        "id": f"{QA_FILE_PREFIX}{filename}",
        "name": "qa-file",
        "file": filename,
        "n": len(items),
        "splits": _split_counts(items),
        "bytes": len(contents),
        "sha256": sha,
    }


# ── Estimate ─────────────────────────────────────────
@router.post("/lab/estimate")
async def lab_estimate(request: LabRequest):
    """Price a run before anything is spent. Free: no model, no Neo4j, no network.

    Per phase (ingest when planned, retrieval-LLM, reader) and per (arm, budget)
    cell: a POINT estimate and an UPPER BOUND (output at the request cap, 0%
    cache, +10% tokenizer margin). ``refuse`` is true when the upper bound
    exceeds ``max_usd``, or when a paid model has no price on file. The
    ``retrieve`` mode is always $0. With ``measured_run_id``, cells that run
    measured are priced on its real contexts rather than on the budget.
    """
    with _lab_errors():
        _run, _dataset, est = await _prepare(request)
    return est


# ── Runs ─────────────────────────────────────────────
@router.post("/lab/runs")
async def start_run(request: LabRequest, background_tasks: BackgroundTasks):
    """Start a Lab run as a background job; follow ``GET /lab/runs/{run_id}/events``.

    Refused before anything starts, in this order: an invalid request (422);
    a paid mode without ``max_usd`` (422); batch mode without its module (501);
    a paid mode without ``OPENAI_API_KEY`` (503); an estimated upper bound
    above ``max_usd`` (422, ``detail.estimate``); a ``run_id`` that holds another
    configuration or has a job in flight (409); a graph that does not hold the
    dataset's corpus (409). Nothing is spent before the free retrieve phase,
    and the runner re-checks the cap on the measured contexts before reading.
    """
    with _lab_errors():
        run = _to_run(request)
        if run.mode != "retrieve" and run.max_usd is None:
            raise _refusal(
                "A paid run needs max_usd",
                f"mode {run.mode!r} calls {run.reader_model}; set max_usd to cap its spend "
                "(the retrieve mode is free)",
            )
        backend = _batch_or_501() if run.mode == "batch" else None
        if run.mode != "retrieve":
            _require_openai_key()  # both paid modes call the OpenAI API (Batch included)
        run, dataset, est = await _prepare(request)
        if est.get("refuse"):
            raise _refusal(
                "Refused: the estimated upper bound breaks the spend cap",
                est.get("refuse_reason") or "",
                estimate=est,
            )
        run.run_id = run.run_id or runner.new_run_id(run)
        run.run_dir = runner.run_dir_for(run.run_id)
        if run.run_id in _active_runs:
            raise HTTPException(status_code=409, detail=f"Lab run {run.run_id!r} is already running")
        manifest_path = run.run_dir / runner.MANIFEST
        retrieved = False
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != run.config_hash():
                raise HTTPException(
                    status_code=409,
                    detail=f"Lab run {run.run_id!r} already exists with another configuration; "
                    "choose another run_id or omit it",
                )
            runner.check_dataset_unchanged(existing, dataset)
            retrieved = (existing.get("phases") or {}).get("retrieve", {}).get("status") == "done"
        if not retrieved:
            try:
                await runner.check_graph_ready(dataset, run.arms)
            except runner.LabError:
                raise
            except Exception as e:  # noqa: BLE001 - Neo4j down is not the caller's fault
                raise HTTPException(
                    status_code=503, detail=f"Could not read the knowledge graph (is Neo4j up?): {e}"
                ) from e

    client = _reader_client() if run.mode == "realtime" else None

    async def work(on_event: EventCallback) -> dict[str, Any]:
        return await runner.run_lab(
            run, client=client, batch_backend=backend, on_event=on_event, dataset=dataset
        )

    job_id = _start_job(background_tasks, run.run_id, work)
    return {
        "job_id": job_id,
        "run_id": run.run_id,
        "status": "running",
        "mode": run.mode,
        "estimate": est,
        "events": f"/api/lab/runs/{run.run_id}/events",
    }


@router.get("/lab/runs")
async def list_lab_runs():
    """Every run under ``lab_runs``, newest first, with ``active`` when a job is in flight."""
    runs = await asyncio.to_thread(runner.list_runs)
    for entry in runs:
        entry["active"] = entry.get("run_id") in _active_runs
    return {"runs": runs}


@router.get("/lab/runs/{run_id}")
async def get_lab_run(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=0, le=1000),
    arm: str | None = Query(None, max_length=64),
    budget: str | None = Query(None, max_length=16),
):
    """A run: manifest, leaderboard, frontiers, floors, report and a page of scored rows.

    ``rows`` pages the per-question rows (``offset`` / ``limit``), optionally for
    one ``arm`` and one ``budget`` (a token count or ``default``). The
    leaderboard is ``null`` until the run is scored.
    """
    with _lab_errors():
        run_dir = _manifest_path(run_id).parent
        loaded = await asyncio.to_thread(runner.load_run, run_dir)
        wanted: Any = "any" if budget is None else _budget(budget)
        rows = await asyncio.to_thread(
            runner.read_rows, run_dir, offset=offset, limit=limit, arm=arm, budget=wanted
        )
    board = loaded["leaderboard"]
    report_path = run_dir / runner.REPORT
    return {
        "run_id": run_id,
        "status": loaded["manifest"].get("status"),
        "active": run_id in _active_runs,
        "job_id": _jobs_by_run.get(run_id),
        "manifest": loaded["manifest"],
        "leaderboard": board,
        "frontiers": (board or {}).get("frontiers"),
        "floors": (board or {}).get("floors"),
        "rows": rows,
        "report": report_path.read_text(encoding="utf-8") if report_path.is_file() else None,
    }


@router.get("/lab/runs/{run_id}/events")
async def run_events(run_id: str):
    """SSE stream of a run's job: runner events, then ``done`` (the summary) or ``error``.

    With no job for the run in this process (finished and already drained, or
    the backend restarted), the stream is one ``done`` event with the run's
    stored state — or ``error`` for an unknown run.
    """
    job_id = _jobs_by_run.get(run_id)
    if job_id is None and job_bus.get(run_id) is not None:
        job_id = run_id  # a job id works too

    async def stream():
        if job_id is not None and job_bus.get(job_id) is not None:
            async for event in job_bus.subscribe(job_id):
                yield _sse(event)
            return
        try:
            manifest = _read_manifest(run_id)
        except (_NotFound, runner.LabError):
            yield _sse({"type": "error", "data": "Unknown Lab run."})
            return
        yield _sse({"type": "done", "data": _summary(manifest)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/lab/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    request: ResumeRequest | None = Body(None),
):
    """Continue a run as a background job, from its manifest.

    A ``batch_submitted`` run polls its batch and, once it is done, collects the
    answers and scores them (no new spend). An ``aborted`` or ``refused`` run
    reads what is left, under ``max_usd`` when given (else its stored cap), and
    this DOES spend. ``retry_failed`` re-reads the requests that came back with
    an error. A request that already has an answer is never sent again. Batch runs
    need ``app.lab.batch`` (501 otherwise); anything that talks to OpenAI (a read, or
    a batch poll) needs ``OPENAI_API_KEY`` (503 otherwise).
    """
    request = request or ResumeRequest()
    with _lab_errors():
        manifest = _read_manifest(run_id)
        if run_id in _active_runs:
            raise HTTPException(status_code=409, detail=f"Lab run {run_id!r} is already running")
        mode = (manifest.get("config") or {}).get("mode")
        status = manifest.get("status")
        # Anything but re-scoring a finished run talks to OpenAI (reads, or polls a batch).
        reads = status != "done" or request.retry_failed
        backend = _batch_or_501() if mode == "batch" and reads else None
        if mode in ("realtime", "batch") and reads:
            _require_openai_key()
        run_dir = runner.run_dir_for(run_id)
        # Resuming or re-scoring on other questions/answers than the run started
        # with would be silent and wrong: refuse up front (409), before any job.
        spec = runner.DatasetSpec.from_dict((manifest.get("config") or {}).get("dataset") or {})
        dataset = await asyncio.to_thread(runner.load_dataset, spec)
        runner.check_dataset_unchanged(manifest, dataset)

    client = _reader_client() if mode == "realtime" else None

    async def work(on_event: EventCallback) -> dict[str, Any]:
        return await runner.resume(
            run_dir,
            client=client,
            batch_backend=backend,
            on_event=on_event,
            dataset=dataset,
            max_usd=request.max_usd,
            retry_failed=request.retry_failed,
        )

    job_id = _start_job(background_tasks, run_id, work)
    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": "running",
        "previous_status": status,
        "events": f"/api/lab/runs/{run_id}/events",
    }
