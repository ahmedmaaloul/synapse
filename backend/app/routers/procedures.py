# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedures Router (procedural memory + the GraphRAG Navigator)

The HTTP face of Procedural Graphs (Lu, Chen, Wu, Arık, arXiv:2609.09153).
The client package, the MCP server, the CLI and the UI all code against these
routes, so their names and shapes are a contract:

    GET    /api/procedures                          list stored graphs
    GET    /api/procedures/{name}[?format=text]     graph JSON (+ version, score) or its text
    GET    /api/procedures/{name}/graph-data        react-force-graph shape for the UI
    PUT    /api/procedures/{name}                   validate + save as a new version
    DELETE /api/procedures/{name}                   the graph and everything recorded for it
    POST   /api/procedures/{name}/guidance          guidance for an agent's next step
    POST   /api/procedures/{name}/trajectories      record an agent run and its score
    GET    /api/procedures/{name}/versions          version history, newest first
    POST   /api/procedures/{name}/rollback          re-save an old version as a new one
    GET    /api/procedures/{name}/rejections        the evolution loop's rejected candidates
    POST   /api/procedures/{name}/evolve            start self-evolution (background job)
    GET    /api/procedures/evolve/{job_id}/events   its progress (SSE; done carries the report)
    POST   /api/agent/ask                           the Navigator answers, non-streaming

Evolution is long-running and spends real LLM calls, so it follows the same
``job_bus`` + SSE pattern as ingestion and community rebuilds. It is capped by
``max_llm_calls``, and only one run per graph at a time is allowed in this
process, so two runs cannot race to save versions of the same graph. Everything
that would make a run pointless or destructive is refused before the job
starts: a cap too small for the baseline plus one round is a 422 naming the
minimum (``minimum_max_llm_calls``), and ``mode=scratch`` on a name that already
holds a graph is a 409 unless the request sets ``replace``.

Errors: an unknown graph is a 404, an invalid graph a 422 listing EVERY
diagnostic, and a missing LLM key a 503 carrying the provider's own message.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import GuidanceMode, get_settings
from app.services import procedural_store as store
from app.services.graph_agent import run_agent
from app.services.jobs import job_bus
from app.services.llm_provider import ProviderConfigError, get_chat_llm
from app.services.procedural_evolution import (
    EvolutionBudgetTooSmall,
    ProceduralGraphExists,
    check_budget,
    evolve,
    rollout_worst_case,
)
from app.services.procedural_graph import (
    GRAPH_NAME_PATTERN,
    START,
    InvalidProceduralGraph,
    ProceduralGraph,
    serialize_full,
)
from app.services.procedural_guidance import guide
from app.services.procedural_store import ProceduralGraphNotFound

logger = logging.getLogger(__name__)
router = APIRouter()

#: Graph names with an evolution job in flight in this process.
_running_evolutions: set[str] = set()


# ── Request models ───────────────────────────────────
class TrajectoryStep(BaseModel):
    # Nullable: the Navigator's own /api/agent/ask steps carry ``action: null``
    # after a failed parse, and replaying such a trace must not be a 422. A
    # step without an action localizes to "none" (the full graph).
    action: str | None = None
    observation: str | None = None
    thought: str | None = None
    args: dict[str, Any] | None = None


class GuidanceRequest(BaseModel):
    query: str = Field(..., min_length=1)
    trajectory: list[TrajectoryStep] = Field(default_factory=list, max_length=200)
    mode: GuidanceMode | None = None
    hops: int | None = Field(None, ge=0, le=5)
    window: int | None = Field(None, ge=0, le=20)


class TrajectoryRecord(BaseModel):
    query: str = Field(..., min_length=1)
    steps: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    score: float = Field(..., ge=0.0, le=1.0)
    source: str = Field("api", min_length=1, max_length=64)


class RollbackRequest(BaseModel):
    version: int = Field(..., ge=1)


class QAItem(BaseModel):
    question: str = Field(..., min_length=1)
    # One gold answer, or several acceptable ones (the best score counts).
    answer: str | list[str]

    @field_validator("answer")
    @classmethod
    def _non_empty_gold(cls, value: str | list[str]) -> str | list[str]:
        """Refuse a blank gold before any money is spent: it can only ever score 0.

        Blank alternatives in a list are dropped (they never raise the best
        score), the same rule as the client package's ``qa_items``.
        """
        if isinstance(value, list):
            golds = [gold.strip() for gold in value if gold.strip()]
            if not golds:
                raise ValueError("needs at least one non-empty gold answer")
            return golds
        if not value.strip():
            raise ValueError("needs a non-empty gold answer")
        return value.strip()


class EvolveRequest(BaseModel):
    train: list[QAItem] = Field(..., min_length=1, max_length=5000)
    val: list[QAItem] = Field(..., min_length=1, max_length=5000)
    rounds: int | None = Field(None, ge=1, le=20)
    batch_size: int | None = Field(None, ge=1, le=100)
    mode: Literal["static", "scratch"] = "static"
    metric: Literal["f1", "em"] = "f1"
    guidance: Literal["raw", "generative"] = "raw"
    max_llm_calls: int | None = Field(None, ge=1, le=100_000)
    # Scratch mode only: allow a run on a name that already holds a graph. The
    # stored graph is then scored once on ``val`` and a candidate must match
    # that score to overwrite it. Ignored in static mode, which always builds
    # on the stored graph.
    replace: bool = False


class AgentAskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # Absent → settings.procedural_default_graph; explicit null → no procedural graph.
    graph: str | None = None
    guidance: GuidanceMode | None = None
    max_steps: int | None = Field(None, ge=1, le=20)
    record: bool = False


# ── Helpers ──────────────────────────────────────────
@contextmanager
def _http_errors() -> Iterator[None]:
    """Map the domain errors to the contract's status codes."""
    try:
        yield
    except ProceduralGraphNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except InvalidProceduralGraph as e:
        raise HTTPException(
            status_code=422, detail={"message": e.message, "diagnostics": e.diagnostics}
        ) from e
    except ProviderConfigError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


async def _load(name: str) -> tuple[ProceduralGraph, dict]:
    loaded = await store.load_graph_with_meta(name)
    if loaded is None:
        raise ProceduralGraphNotFound(name)
    return loaded


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


# ── Graph CRUD ───────────────────────────────────────
@router.get("/procedures")
async def list_procedures():
    """Every stored Procedural Graph with its version, score and size."""
    return {"graphs": await store.list_graphs()}


@router.get("/procedures/{name}")
async def get_procedure(name: str, format: Literal["json", "text"] = Query("json")):
    """The graph as JSON (editable and PUT-able as-is), or ``?format=text``.

    The text form is the full-graph serialization that a guidance model reads.
    """
    with _http_errors():
        graph, meta = await _load(name)
    if format == "text":
        return {"text": serialize_full(graph)}
    return {**graph.to_dict(), "version": meta["version"], "score": meta["score"]}


@router.get("/procedures/{name}/graph-data")
async def get_procedure_graph_data(name: str):
    """The graph in react-force-graph's ``{nodes, links}`` shape.

    ``is_start`` and ``is_terminal`` let the UI mark where localization begins
    and where the procedure can end.
    """
    with _http_errors():
        graph, meta = await _load(name)
    terminals = set(graph.terminals())
    nodes = [
        {
            "id": node.id,
            "label": node.id,
            "type": node.type,
            "description": node.description,
            "is_start": node.id == START,
            "is_terminal": node.id in terminals,
        }
        for node in graph.nodes.values()
    ]
    return {
        "nodes": nodes,
        "links": [edge.to_dict() for edge in graph.edges],
        "version": meta["version"],
        "score": meta["score"],
    }


@router.put("/procedures/{name}")
async def put_procedure(name: str, body: dict[str, Any] = Body(...)):
    """Validate a graph JSON document and save it as the next version.

    The ``name`` in the body must match the path or be absent. An invalid
    graph is a 422 whose ``detail`` lists every diagnostic at once.
    """
    body_name = body.get("name")
    if body_name not in (None, "", name):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Graph name in the body does not match the path",
                "diagnostics": [f"body name {body_name!r} != path name {name!r}"],
            },
        )
    with _http_errors():
        graph = ProceduralGraph.from_dict(body, name=name)
        version = await store.save_graph(graph, note="saved via PUT /api/procedures")
    return {"name": name, "version": version}


@router.delete("/procedures/{name}")
async def delete_procedure(name: str):
    """Delete a graph with its versions, rejections and trajectories (idempotent)."""
    deleted = await store.delete_graph(name)
    return {"status": "success", "deleted": deleted}


# ── Online guidance + trajectories ───────────────────
@router.post("/procedures/{name}/guidance")
async def procedure_guidance(name: str, request: GuidanceRequest):
    """Guidance for an agent's next step, from its last action.

    Built for agents running their own model (MCP hosts). With the default
    ``raw`` mode this makes no LLM call here: the caller gets the serialized
    local subgraph to read itself.
    """
    trajectory = [step.model_dump(exclude_none=True) for step in request.trajectory]
    with _http_errors():
        return await guide(
            name,
            query=request.query,
            trajectory=trajectory,
            mode=request.mode,
            hops=request.hops,
            window=request.window,
        )


@router.post("/procedures/{name}/trajectories")
async def record_procedure_trajectory(name: str, request: TrajectoryRecord):
    """Record one agent run against the graph's current version (a 404 if it is unknown)."""
    meta = await store.get_meta(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=str(ProceduralGraphNotFound(name)))
    await store.record_trajectory(
        name,
        version=meta["version"],
        query=request.query,
        steps=request.steps,
        score=request.score,
        source=request.source,
    )
    return {"status": "recorded"}


# ── Versions ─────────────────────────────────────────
@router.get("/procedures/{name}/versions")
async def list_procedure_versions(name: str):
    """Every saved version, newest first, each with its score, note and diff."""
    versions = await store.list_versions(name)
    if not versions and await store.get_meta(name) is None:
        raise HTTPException(status_code=404, detail=str(ProceduralGraphNotFound(name)))
    return {"versions": versions}


@router.post("/procedures/{name}/rollback")
async def rollback_procedure(name: str, request: RollbackRequest):
    """Re-save version ``version`` as a new version. History stays append-only."""
    with _http_errors():
        version = await store.rollback(name, request.version)
    return {"name": name, "version": version}


@router.get("/procedures/{name}/rejections")
async def list_procedure_rejections(name: str, limit: int = Query(20, ge=1, le=200)):
    """Candidates the evolution loop rejected, newest first (the rejection memory).

    A 404 only when nothing is stored under ``name`` at all: a scratch run that
    accepted nothing leaves rejections for a graph that was never saved.
    """
    rejections = await store.list_rejections(name, limit=limit)
    if not rejections and await store.get_meta(name) is None:
        raise HTTPException(status_code=404, detail=str(ProceduralGraphNotFound(name)))
    return {"rejections": rejections}


# ── Offline self-evolution ───────────────────────────
async def _run_evolution(job_id: str, name: str, params: dict) -> None:
    """Background task: run Algorithm 1 and stream its progress on the job bus."""
    try:

        async def on_progress(event: dict) -> None:
            await job_bus.publish(job_id, event)

        report = await evolve(name, on_progress=on_progress, **params)
        await job_bus.publish(job_id, {"type": "done", "data": report})
        logger.info(
            "🧬 Evolution of %s finished: %s → %s (%s LLM calls, stopped: %s)",
            name,
            report.get("baseline_score"),
            report.get("final_score"),
            report.get("llm_calls"),
            report.get("stopped"),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("❌ Evolution of %s failed", name)
        await job_bus.publish(job_id, {"type": "error", "data": str(e)})
    finally:
        _running_evolutions.discard(name)


@router.post("/procedures/{name}/evolve")
async def evolve_procedure(name: str, request: EvolveRequest, background_tasks: BackgroundTasks):
    """Start self-evolution in the background; returns a ``job_id`` to follow.

    Checked before any money is spent, in this order: the graph exists (static
    mode, else 404) or the name is valid (scratch mode, else 422); scratch mode
    does not overwrite a stored graph unless ``replace`` is set (else 409);
    ``max_llm_calls`` pays for the baseline plus one full round in the worst
    case (else 422 whose ``detail.minimum_max_llm_calls`` is the smallest cap
    that does, computed exactly as ``evolve`` does); an LLM provider is
    configured (else 503); and no other run is evolving this graph (else 409).
    """
    settings = get_settings()
    stored_eval = False
    if request.mode == "static":
        if await store.get_meta(name) is None:
            raise HTTPException(status_code=404, detail=str(ProceduralGraphNotFound(name)))
    else:
        if not GRAPH_NAME_PATTERN.match(name):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Invalid graph name",
                    "diagnostics": [f"invalid graph name {name!r}"],
                },
            )
        existing = await store.get_meta(name)
        if existing is not None and not request.replace:
            raise HTTPException(
                status_code=409, detail=str(ProceduralGraphExists(name, existing.get("version")))
            )
        # With replace, the stored graph costs one extra evaluation on val.
        stored_eval = existing is not None

    params = {
        "train": [item.model_dump() for item in request.train],
        "val": [item.model_dump() for item in request.val],
        "rounds": request.rounds or settings.evolution_default_rounds,
        "batch_size": request.batch_size or settings.evolution_default_batch_size,
        "mode": request.mode,
        "metric": request.metric,
        "guidance": request.guidance,
        "max_llm_calls": request.max_llm_calls or settings.evolution_max_llm_calls,
        "replace": request.replace,
    }
    try:
        check_budget(
            params["max_llm_calls"],
            train_size=len(params["train"]),
            val_size=len(params["val"]),
            batch_size=params["batch_size"],
            # evolve() runs the navigator with AGENT_MAX_STEPS, as here.
            per_rollout=rollout_worst_case(settings.agent_max_steps, request.guidance),
            stored_eval=stored_eval,
        )
    except EvolutionBudgetTooSmall as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "max_llm_calls cannot pay for the baseline and one full round",
                "diagnostics": [str(e)],
                "minimum_max_llm_calls": e.minimum,
            },
        ) from e
    with _http_errors():
        get_chat_llm(temperature=0)  # builds the client only; raises if no provider is set up
    if name in _running_evolutions:
        raise HTTPException(status_code=409, detail=f"An evolution of '{name}' is already running")

    job_id = job_bus.create()
    _running_evolutions.add(name)
    background_tasks.add_task(_run_evolution, job_id, name, params)
    return {"job_id": job_id, "status": "processing"}


@router.get("/procedures/evolve/{job_id}/events")
async def evolve_events(job_id: str):
    """SSE stream of an evolution job: progress events, then ``done`` (report) or ``error``."""

    async def stream():
        async for event in job_bus.subscribe(job_id):
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── The Navigator ────────────────────────────────────
@router.post("/agent/ask")
async def agent_ask(request: AgentAskRequest):
    """Answer with the GraphRAG Navigator: a step-by-step walk of the knowledge graph.

    Every step is one LLM call on this backend's provider (plus one more per step
    with ``generative`` guidance), so this costs more than ``/api/chat``. The
    response is the full trace: thoughts, actions, observations, usage. With
    ``record=true`` the trajectory is stored against the graph (score null).
    """
    settings = get_settings()
    if "graph" in request.model_fields_set:
        graph_name = request.graph or None
    else:
        graph_name = settings.procedural_default_graph if settings.procedural_enabled else None

    with _http_errors():
        result = await run_agent(
            request.query,
            graph_name=graph_name,
            guidance=request.guidance,
            max_steps=request.max_steps,
        )
    payload = result.to_dict()
    recorded = False
    if request.record and result.graph is not None:
        try:
            await store.record_trajectory(
                result.graph["name"],
                version=result.graph["version"],
                query=request.query,
                steps=result.steps,
                score=None,
                source="agent",
            )
            recorded = True
        except Exception as e:  # noqa: BLE001 - the answer is already paid for; return it
            logger.warning("⚠️ Could not record the navigator trajectory: %s", e)
    payload["recorded"] = recorded
    return payload
