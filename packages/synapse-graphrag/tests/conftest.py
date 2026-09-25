# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Shared fixtures — a fake Synapse backend behind ``httpx.MockTransport``.

Hermetic by construction: no network, no Neo4j, no LLM. The fake speaks the
backend's exact wire format (JSON bodies, ``data: {json}\\n\\n`` SSE frames,
FastAPI-style ``{"detail": ...}`` errors) so the client, the MCP server and
the CLI are all exercised end to end against the real request/response shapes.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

import httpx
import pytest

from synapse_graphrag.client import SynapseClient

BASE_URL = "http://synapse.test"


def sse(*events: dict[str, Any]) -> bytes:
    """Encode events exactly as the backend does."""
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def split_bytes(payload: bytes, size: int) -> list[bytes]:
    return [payload[i : i + size] for i in range(0, len(payload), size)] or [b""]


class ChunkedStream(httpx.AsyncByteStream):
    """Chunked body that records who closed it.

    ``closed_by_reader`` is True when ``aclose`` ran in the task that read the
    stream — a deterministic, consumer-driven close — and False when the event
    loop's asyncgen finaliser got to it first, which is the race the client
    guards against with ``contextlib.aclosing``.
    """

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = list(chunks)
        self.reader: asyncio.Task | None = None
        self.closed_by_reader: bool | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.reader = asyncio.current_task()
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed_by_reader = asyncio.current_task() is self.reader
        await asyncio.sleep(0)  # a real socket close suspends too


@dataclass
class Call:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    params: dict[str, str]
    raw_path: str = ""  # percent-encoded, without the query string

    @property
    def json(self) -> Any:
        return json.loads(self.body) if self.body else None


GRAPH = {
    "nodes": [
        {"id": "1", "label": "Charles Babbage", "type": "PERSON", "properties": {"description": "Mathematician"}},
        {"id": "2", "label": "Analytical Engine", "type": "TOOL", "properties": {}},
        {"id": "3", "label": "Ada Lovelace", "type": "PERSON", "properties": {"description": "First programmer"}},
        {"id": "4", "label": "Lonely Node", "type": "CONCEPT", "properties": {}},
    ],
    "links": [
        {"source": "1", "target": "2", "type": "DESIGNED", "properties": {}},
        {"source": "3", "target": "2", "type": "WROTE_PROGRAMS_FOR", "properties": {}},
        {"source": "3", "target": "1", "type": "COLLABORATED_WITH", "properties": {}},
    ],
}

RETRIEVAL = {
    "mode": "local",
    "context": "Charles Babbage designed the Analytical Engine.\nAda Lovelace wrote programs for it.",
    "citations": [
        {"name": "Charles Babbage", "type": "PERSON", "kind": "entity"},
        {"name": "Analytical Engine", "type": "TOOL", "kind": "entity"},
    ],
    "paths": [{"nodes": ["Ada Lovelace", "Analytical Engine"], "hops": 1}],
    "sources": [{"id": "c1", "document": "history.pdf", "index": 0, "text": "Babbage designed…"}],
    "usage": {
        "context_chars": 83,
        "context_tokens_est": 21,
        "truncated": False,
        "citations": 2,
        "paths": 1,
        "sources": 1,
    },
}

ABOUT = {
    "name": "Synapse",
    "version": "0.4.0",
    "llm_provider": "gemini",
    "embedding_provider": "fastembed",
    "repository": "https://github.com/ahmedmaaloul/synapse",
    "license": "PolyForm-Noncommercial-1.0.0",
}

COMMUNITIES = {
    "communities": [
        {"id": "c-1", "title": "Victorian computing", "summary": "Babbage and Lovelace.", "size": 3, "members": ["Charles Babbage", "Ada Lovelace", "Analytical Engine"]},
    ],
    "count": 1,
}

INGEST_DONE = {
    "filename": "history.pdf",
    "chunks_processed": 3,
    "nodes_created": 12,
    "relationships_created": 18,
    "entities_extracted": 14,
    "unique_entities": 12,
    "entities_merged": 2,
    "chunks_stored": 3,
    "communities": 2,
    "communities_summarized": 2,
    "modularity": 0.41,
}

# ── Procedural memory (the HTTP contract of routers/procedures.py) ──────────
PROC_NAME = "graphrag-navigator"

PROC_GRAPH = {
    "name": PROC_NAME,
    "description": "How to answer a multi-hop question by walking the knowledge graph.",
    "cycle_policy": "forbid",
    "tools": ["search_entities", "neighbors", "answer"],
    "nodes": [
        {"id": "Start", "type": "STATUS", "description": "A question has been received."},
        {"id": "search_entities", "type": "ACTION", "description": "Find entities by name."},
        {"id": "neighbors", "type": "ACTION", "description": "List an entity's relations."},
        {"id": "answer", "type": "ACTION", "description": "Submit the final short answer."},
        {"id": "End", "type": "STATUS", "description": "Episode over."},
    ],
    "edges": [
        {"source": "Start", "target": "search_entities", "relation": "LEADS_TO", "condition": None, "guidance": "Search the entity the question names.", "pitfalls": "Do not answer from memory."},
        {"source": "search_entities", "target": "neighbors", "relation": "LEADS_TO", "condition": "the entity was found", "guidance": "Read its relations to find the bridge entity.", "pitfalls": "Do not answer from the name alone."},
        {"source": "search_entities", "target": "answer", "relation": "TRIGGERS", "condition": "the description states the answer", "guidance": "Answer with the shortest span.", "pitfalls": ""},
        {"source": "neighbors", "target": "answer", "relation": "CONVERGES_TO", "condition": None, "guidance": "Answer once every hop is supported.", "pitfalls": "Do not stop at the bridge entity."},
        {"source": "answer", "target": "End", "relation": "LEADS_TO", "condition": None, "guidance": "", "pitfalls": ""},
    ],
}

PROC_TEXT = (
    "Procedural Graph: [graphrag-navigator]\nNodes:\n- [Start] (Type: STATUS)\nTransitions:\n"
    "- Transition: [Start]→[search_entities] (Relation: LEADS_TO; Condition: unconditional)"
)

PROC_SUMMARY = {
    "name": PROC_NAME,
    "version": 3,
    "score": 0.5556,
    "nodes": 5,
    "edges": 5,
    "updated_at": "2026-09-24T12:00:00Z",
    "description": PROC_GRAPH["description"],
}

PROC_GRAPH_DATA = {
    "nodes": [
        {"id": n["id"], "label": n["id"], "type": n["type"], "description": n["description"], "is_start": n["id"] == "Start", "is_terminal": n["id"] == "End"}
        for n in PROC_GRAPH["nodes"]
    ],
    "links": [dict(e) for e in PROC_GRAPH["edges"]],
    "version": 3,
    "score": 0.5556,
}

PROC_VERSIONS = {
    "versions": [
        {"version": 3, "score": 0.5556, "accepted": True, "created_at": "2026-09-24T12:00:00Z", "note": "evolution round 2", "diff": {"added_nodes": ["neighbors"], "removed_nodes": [], "added_edges": [["search_entities", "neighbors"], ["neighbors", "answer"]], "removed_edges": [], "changed_edges": [["Start", "search_entities"]]}},
        {"version": 2, "score": 0.4444, "accepted": True, "created_at": "2026-09-24T11:00:00Z", "note": "evolution round 1", "diff": {"added_nodes": [], "removed_nodes": [], "added_edges": [], "removed_edges": [], "changed_edges": []}},
        {"version": 1, "score": None, "accepted": True, "created_at": "2026-09-24T10:00:00Z", "note": "seeded from the expert prior", "diff": {}},
    ]
}

PROC_REJECTIONS = {
    "rejections": [
        {"graph": PROC_NAME, "round": 3, "reason": "score 0.333 < 0.556", "score": 0.3333, "diagnostics": [], "edits": {"add_nodes": []}, "created_at": "2026-09-24T12:30:00Z"},
    ]
}

AGENT_RESULT = {
    "answer": "Ada Lovelace",
    "steps": [
        {"thought": "Find the engine first.", "action": "search_entities", "args": {"query": "Analytical Engine"}, "observation": "Analytical Engine (TOOL): a mechanical general-purpose computer " + "designed by Charles Babbage. " * 20, "guidance_context_chars": 412, "localization": "start"},
        {"thought": "Who wrote programs for it?", "action": "neighbors", "args": {"entity": "Analytical Engine"}, "observation": "Ada Lovelace -WROTE_PROGRAMS_FOR-> Analytical Engine", "guidance_context_chars": 380, "localization": "exact"},
        {"thought": "Every hop is supported.", "action": "answer", "args": {"text": "Ada Lovelace"}, "observation": "", "guidance_context_chars": 120, "localization": "exact"},
    ],
    "stopped": "answer",
    "parse_failures": 0,
    "usage": {"llm_calls": 3, "guidance_llm_calls": 0, "input_tokens": 2412, "output_tokens": 188, "estimated": False, "context_chars": 912},
    "graph": {"name": PROC_NAME, "version": 3},
    "latency_s": 4.21,
    "recorded": False,
}

EVOLVE_REPORT = {
    "graph": PROC_NAME,
    "mode": "static",
    "rounds_run": 2,
    "stopped": "completed",
    "baseline_score": 0.4444,
    "final_score": 0.5556,
    "final_version": 4,
    "rounds": [
        {"round": 1, "train_mean": 0.5, "candidate_score": 0.5556, "accepted": True, "reason": "", "diagnostics": [], "diff": {"added_nodes": ["Verify_Answer"], "added_edges": [["neighbors", "Verify_Answer"]]}},
        {"round": 2, "train_mean": 0.45, "candidate_score": None, "accepted": False, "reason": "structural", "diagnostics": ["node 'x' cannot reach a terminal"], "diff": None},
    ],
    "llm_calls": 212,
    "effect_floor": 0.1111,
}


# ── Synapse Lab (the HTTP contract of routers/lab.py) ───────────────────────
LAB_ARMS = {
    "arms": [
        {"name": "null_closed_book", "family": "null", "family_title": "Evidence floors", "title": "N0 · Closed-book", "description": "No evidence at all.", "source": {"citation": "Synapse null controls", "url": "https://github.com/ahmedmaaloul/synapse"}, "retrieval_llm_calls": 0, "needs_graph": False, "is_null": True, "k_role": "none"},
        {"name": "bm25", "family": "passage", "family_title": "Passage baselines", "title": "BM25", "description": "Lucene BM25 over the chunk index.", "source": {"citation": "Robertson & Zaragoza 2009", "url": "https://doi.org/10.1561/1500000019"}, "retrieval_llm_calls": 0, "needs_graph": False, "is_null": False, "k_role": "passages"},
        {"name": "synapse_lean", "family": "graph", "family_title": "Graph arms", "title": "Synapse-lean", "description": "PathRAG-style flow-pruned paths with a LiteRAG-style hub penalty.", "source": {"citation": "PathRAG arXiv:2502.14902", "url": "https://arxiv.org/abs/2502.14902"}, "retrieval_llm_calls": 0, "needs_graph": True, "is_null": False, "k_role": "seeds"},
    ],
    "families": [{"family": "null", "title": "Evidence floors"}, {"family": "passage", "title": "Passage baselines"}, {"family": "graph", "title": "Graph arms"}],
    "modes": ["realtime", "batch", "retrieve"],
    "default_mode": "retrieve",
    "budgets": [500, 1000, 2000, 4000, 8000, None],
    "batch_available": True,
}

LAB_MODELS = {
    "models": [
        {"name": "gpt-4o-mini", "input_usd_per_1m": 0.15, "output_usd_per_1m": 0.6, "batch_input_usd_per_1m": 0.075, "batch_output_usd_per_1m": 0.3, "reasoning": False, "max_output_tokens": 32, "tokenizer": "tiktoken:o200k_base", "price_checked_on": "2026-07-21"},
        {"name": "gpt-5-nano", "input_usd_per_1m": 0.05, "output_usd_per_1m": 0.4, "batch_input_usd_per_1m": 0.025, "batch_output_usd_per_1m": 0.2, "reasoning": True, "max_output_tokens": 544, "tokenizer": "tiktoken:o200k_base", "price_checked_on": "2026-09-24"},
    ],
    "default": "gpt-5-nano",
    "batch_multiplier": 0.5,
    "pricing_url": "https://openai.com/api/pricing/",
    "note": "prices are hand-recorded",
}

LAB_DATASETS = {
    "datasets": [
        {"id": "demo", "name": "demo", "title": "Demo QA set", "splits": {"test": 9, "train": 12, "val": 9}, "default_split": "test", "n": 9, "source": "demo_qa.json"},
        {"id": "qa-file:mine.jsonl", "name": "qa-file", "file": "mine.jsonl", "title": "mine.jsonl", "bytes": 300, "n": 3, "splits": {"test": 2, "train": 1}},
    ],
    "unavailable": [{"id": "hotpotqa", "name": "hotpotqa", "reason": "no HotpotQA paragraph is ingested in the knowledge graph"}],
}

LAB_RUN_ID = "20260925-101500-a1b2c3"
LAB_ABORTED_ID = "aborted-run"
LAB_BATCH_ID = "batch-run"


def _cmp(diff: float, reportable: bool, against: str) -> dict[str, Any]:
    return {"diff": diff, "ci_low": diff - 5, "ci_high": diff + 5, "n": 9, "floor": 11.11, "reportable": reportable, "against": against, "verdict": "above" if reportable else "too close to call"}


LAB_BOARD_READ = {
    "mode": "read",
    "n_questions": 9,
    "effect_floor_points": 11.11,
    "bootstrap": {"iterations": 10000, "seed": 20260924, "confidence": 0.95},
    "rows": [
        {"arm": "bm25", "budget": 500, "budget_label": "500", "family": "passage", "title": "BM25", "is_null": False, "n": 9, "f1": 66.7, "em": 55.6, "tokens_per_correct": 812.4, "usd_per_100_correct": 0.0123, "comparisons": {"gain_above_n0": _cmp(22.2, True, "null_closed_book")}, "rank": 1},
        {"arm": "synapse_lean", "budget": 500, "budget_label": "500", "family": "graph", "title": "Synapse-lean", "is_null": False, "n": 9, "f1": 70.1, "em": 55.6, "tokens_per_correct": 901.0, "usd_per_100_correct": 0.0150, "comparisons": {"gain_above_n0": _cmp(25.6, True, "null_closed_book"), "graph_premium": _cmp(3.4, False, "bm25")}, "rank": 2},
        {"arm": "null_closed_book", "budget": 500, "budget_label": "500", "family": "null", "title": "N0", "is_null": True, "n": 9, "f1": 44.4, "em": 33.3, "tokens_per_correct": 350.0, "usd_per_100_correct": 0.0301, "comparisons": {}, "rank": 3},
    ],
    "floors": {"n0_f1": 44.4, "n2_f1_by_budget": {}},
    "frontiers": {"f1_vs_usd": [{"arm": "bm25", "budget": 500}, {"arm": "synapse_lean", "budget": 500}], "f1_vs_tokens": []},
    "notes": ["ingest cost unknown: amortized cost-of-pass for graph arms is not computed"],
}

LAB_BOARD_RETRIEVE = {
    "mode": "retrieve",
    "n_questions": 9,
    "effect_floor_points": 11.11,
    "rows": [
        {"arm": "null_closed_book", "budget": None, "budget_label": "default", "family": "null", "is_null": True, "n": 9, "context_tokens_mean": 0.0, "units_mean": 0.0, "containment": 0.0, "recall_permissive": None, "recall_strict": None},
        {"arm": "bm25", "budget": None, "budget_label": "default", "family": "passage", "is_null": False, "n": 9, "context_tokens_mean": 734.2, "units_mean": 8.0, "containment": 88.9, "recall_permissive": None, "recall_strict": None},
    ],
    "floors": {"n0_f1": None, "n2_f1_by_budget": {}},
    "frontiers": {"f1_vs_usd": [], "f1_vs_tokens": []},
}

LAB_ROWS = [
    {"arm": "bm25", "budget": 500, "qid": f"demo-{i:02d}", "question": f"q{i}", "gold": ["1843"], "context_tokens": 480, "units_used": 3, "containment": True, "answer": "1843", "em": 1.0, "f1": 1.0}
    for i in range(1, 4)
]


def lab_estimate_payload(body: dict[str, Any]) -> dict[str, Any]:
    """A deterministic stand-in for the backend estimator: $0.001 point / $0.002 upper per cell."""
    mode = body.get("mode", "retrieve")
    arms = body.get("arms") or [a["name"] for a in LAB_ARMS["arms"]]
    budgets = body.get("budgets") or [None]
    paid = mode != "retrieve"
    cells = [
        {"arm": a, "budget": b, "calls": 9 if paid else 0, "prompt_tokens": 4500 if paid else 0, "completion_tokens": 72 if paid else 0, "upper_prompt_tokens": 4950 if paid else 0, "upper_completion_tokens": 4896 if paid else 0, "point_usd": 0.001 if paid else 0.0, "upper_usd": 0.002 if paid else 0.0, "context_tokens": float(b or 4000) if paid else 0.0, "measured": False, "note": ""}
        for a in arms for b in budgets
    ]
    upper = sum(c["upper_usd"] for c in cells)
    max_usd = body.get("max_usd")
    refuse = max_usd is not None and upper > max_usd
    return {
        "mode": mode,
        "reader_model": body.get("reader_model", "gpt-5-nano"),
        "batch": mode == "batch",
        "n_questions": 9,
        "phases": [{"phase": "reader", "model": "gpt-5-nano", "batch": mode == "batch", "calls": sum(c["calls"] for c in cells), "point_usd": sum(c["point_usd"] for c in cells), "upper_usd": upper}] if paid else [],
        "cells": cells,
        "total_point_usd": sum(c["point_usd"] for c in cells),
        "total_upper_usd": upper,
        "max_usd": max_usd,
        "refuse": refuse,
        "refuse_reason": f"upper bound ${upper:.4f} exceeds the cap ${max_usd:.4f}" if refuse else None,
        "tokenizer": "tiktoken:o200k_base",
        "reasoning_allowance": 512 if paid else 0,
        "max_output_tokens": 544,
        "price_dates": {"table": "2026-07-21"},
        "assumptions": ["prices hand-recorded"] if paid else ["retrieve-only mode never calls a model: the estimate is $0"],
        "pricing_url": "https://openai.com/api/pricing/",
        "batch_multiplier": 0.5,
        "dataset": {"name": "demo", "split": "test", "n": 9, "source": "demo_qa.json", "sha256": "abc"},
        "measured_from": {"run_id": body["measured_run_id"], "cells": [], "unmeasured": []} if body.get("measured_run_id") else None,
        "config_hash": "0123456789abcdef",
        "cells_count": len(cells),
    }


def lab_detail(run_id: str, mode: str, status: str, *, cap: float | None = 1.0, pending: dict[str, Any] | None = None) -> dict[str, Any]:
    board = None if status in ("batch_submitted", "refused") else copy.deepcopy(LAB_BOARD_RETRIEVE if mode == "retrieve" else LAB_BOARD_READ)
    manifest = {
        "run_id": run_id,
        "status": status,
        "created_at": "2026-09-25T10:15:00+00:00",
        "config": {"mode": mode, "reader_model": "gpt-5-nano", "arms": ["null_closed_book", "bm25", "synapse_lean"], "budgets": [500]},
        "dataset": {"name": "demo", "split": "test", "n": 9},
        "code": {"git_sha": "c6a6e35f00d1", "git_dirty": True, "synapse_version": "0.4.0"},
        "actual": {"reader": {"usd": 0.0021 if mode != "retrieve" else 0.0}},
        "estimate": {"read": {"total_point_usd": 0.002, "total_upper_usd": 0.009}, "read_pending": pending or {"requests": 0, "upper_usd": 0.0, "spent_usd": 0.0021}},
        "budget_cap_usd": cap,
    }
    if status == "aborted":
        manifest["abort_reason"] = "spend cap: 5 request(s) not sent"
    return {
        "run_id": run_id,
        "status": status,
        "active": False,
        "job_id": None,
        "manifest": manifest,
        "leaderboard": board,
        "frontiers": (board or {}).get("frontiers"),
        "floors": (board or {}).get("floors"),
        "rows": {"total": len(LAB_ROWS), "offset": 0, "limit": 0, "rows": []},
        "report": f"# Synapse Lab run `{run_id}`\n\n| Arm | Budget | F1 |\n|---|---|---|\n| bm25 | 500 | 66.7 |\n" if board else None,
    }


def _guidance_payload(name: str, body: dict[str, Any]) -> dict[str, Any]:
    """A deterministic stand-in for ``procedural_guidance.guide()``."""
    trajectory = body.get("trajectory") or []
    mode = body.get("mode") or "raw"
    if not trajectory:
        node, method = "Start", "start"
    else:
        last = str(trajectory[-1].get("action") or "")
        node, method = (last, "exact") if last in {n["id"] for n in PROC_GRAPH["nodes"]} else (None, "none")
    context = "" if mode == "none" else f"Active Cognitive Node: [{node}] (Type: ACTION)" if node else PROC_TEXT
    targets = [e["target"] for e in PROC_GRAPH["edges"] if e["source"] == node]
    return {
        "graph": name,
        "version": 3,
        "active_node": node,
        "localization": method,
        "scope": "local" if node else "full",
        "context": context,
        "guidance": "Read the entity's relations next." if mode == "generative" else None,
        "next_actions": targets,
        "usage": {"context_chars": len(context), "context_tokens_est": len(context) // 4, "llm_calls": 1 if mode == "generative" else 0, "cached": False},
    }


@dataclass
class FakeBackend:
    """Routes requests like the FastAPI app; ``calls`` records what the client sent."""

    calls: list[Call] = field(default_factory=list)
    streams: list[ChunkedStream] = field(default_factory=list)
    chunk_size: int = 4096
    unreachable: bool = False
    failures: dict[str, tuple[int, Any]] = field(default_factory=dict)
    chat_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "citations", "data": RETRIEVAL["citations"]},
            {"type": "paths", "data": RETRIEVAL["paths"]},
            {"type": "sources", "data": RETRIEVAL["sources"]},
            {"type": "token", "data": "Babbage "},
            {"type": "token", "data": "designed it — Ünïcödé ✓"},
            {"type": "done", "usage": {"context_chars": 83, "context_tokens_est": 21, "answer_chars": 32}},
        ]
    )
    ingest_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "progress", "stage": "extracting", "processed": 0, "total": 3},
            {"type": "progress", "stage": "extracting", "processed": 3, "total": 3},
            {"type": "community_progress", "stage": "summarizing", "processed": 2, "total": 2},
            {"type": "done", "data": INGEST_DONE},
        ]
    )
    rebuild_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "progress", "stage": "clustering"},
            {"type": "done", "data": {"communities": 2, "summarized": 2, "modularity": 0.41}},
        ]
    )
    evolve_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "progress", "stage": "baseline", "round": 0, "processed": 3, "total": 3, "llm_calls": 12},
            {"type": "progress", "stage": "baseline", "round": 0, "score": 0.4444},
            {"type": "progress", "stage": "rollout", "round": 1, "processed": 10, "total": 10, "llm_calls": 52},
            {"type": "progress", "stage": "refine", "round": 1, "llm_calls": 52},
            {"type": "progress", "stage": "accepted", "round": 1, "score": 0.5556, "previous_score": 0.4444, "version": 4, "change": "+1 node, +1 edge"},
            {"type": "progress", "stage": "rejected", "round": 2, "reason": "structural", "edits": {"add_nodes": [{"id": "x"}]}, "diagnostics": ["node 'x' cannot reach a terminal"]},
            {"type": "done", "data": EVOLVE_REPORT},
        ]
    )
    procedures: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {PROC_NAME: {**copy.deepcopy(PROC_GRAPH), "version": 3, "score": 0.5556}}
    )
    lab_runs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            LAB_RUN_ID: lab_detail(LAB_RUN_ID, "realtime", "done"),
            LAB_ABORTED_ID: lab_detail(LAB_ABORTED_ID, "realtime", "aborted", cap=0.5, pending={"requests": 5, "upper_usd": 0.004, "spent_usd": 0.5}),
            LAB_BATCH_ID: lab_detail(LAB_BATCH_ID, "batch", "batch_submitted"),
        }
    )
    # Overrides the events of the next run/resume stream (None: derived from the run's mode).
    lab_events: list[dict[str, Any]] | None = None
    # The final status a resumed run reaches (None: "done").
    lab_resume_status: str | None = None
    lab_uploads: list[bytes] = field(default_factory=list)

    def _stream(self, events: list[dict[str, Any]]) -> httpx.Response:
        stream = ChunkedStream(split_bytes(sse(*events), self.chunk_size))
        self.streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        body = await request.aread()
        path = request.url.path
        raw_path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        self.calls.append(
            Call(
                request.method,
                path,
                dict(request.headers),
                body,
                dict(request.url.params),
                raw_path,
            )
        )
        if path in self.failures:
            status, detail = self.failures[path]
            if detail is None:
                return httpx.Response(status, text="<html>gateway timeout</html>")
            return httpx.Response(status, json={"detail": detail})

        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "service": "synapse-backend"})
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready", "neo4j": "up", "llm_provider": "gemini"})
        if path == "/api/about":
            return httpx.Response(200, json=ABOUT)
        if path == "/api/retrieve":
            payload = json.loads(body)
            result = json.loads(json.dumps(RETRIEVAL))
            budget = payload.get("max_context_chars")
            if budget is not None and budget < len(result["context"]):
                result["context"] = result["context"][:budget] + f"\n…[context truncated to {budget} chars]"
                result["usage"]["truncated"] = True
            result["usage"]["context_chars"] = len(result["context"])
            return httpx.Response(200, json=result)
        if path == "/api/chat":
            return self._stream(self.chat_events)
        if path == "/api/upload":
            return httpx.Response(
                200,
                json={"job_id": "job_000007", "filename": "history.pdf", "total_chunks": 3, "status": "processing"},
            )
        if path == "/api/upload/job_000007/events":
            return self._stream(self.ingest_events)
        if path == "/api/graph-data":
            return httpx.Response(200, json=GRAPH)
        if path == "/api/communities":
            return httpx.Response(200, json=COMMUNITIES)
        if path == "/api/communities/rebuild":
            return httpx.Response(200, json={"job_id": "job_000008", "status": "processing"})
        if path == "/api/communities/rebuild/job_000008/events":
            return self._stream(self.rebuild_events)
        if path == "/api/graph" and request.method == "DELETE":
            return httpx.Response(200, json={"status": "success", "message": "Knowledge graph cleared (procedural memory kept)"})
        if path == "/api/agent/ask" and request.method == "POST":
            payload = json.loads(body)
            if payload.get("graph", PROC_NAME) is not None and payload.get("graph", PROC_NAME) not in self.procedures:
                return httpx.Response(404, json={"detail": f"Procedural graph '{payload['graph']}' not found."})
            result = copy.deepcopy(AGENT_RESULT)
            if payload.get("graph", PROC_NAME) is None:
                result["graph"] = None
                for step in result["steps"]:
                    step["localization"] = None
            # Like the router: only a run steered by a graph can be recorded against it.
            result["recorded"] = bool(payload.get("record")) and result["graph"] is not None
            return httpx.Response(200, json=result)
        if raw_path == "/api/procedures" or raw_path.startswith("/api/procedures/"):
            return self._procedures(request.method, raw_path, request.url.params, body)
        if raw_path.startswith("/api/lab/"):
            return self._lab(request.method, raw_path, request.url.params, body)
        return httpx.Response(404, json={"detail": "Not Found"})

    def _procedures(self, method: str, raw_path: str, params: Any, body: bytes) -> httpx.Response:
        """``/api/procedures…`` routed on the ENCODED path, so a name may hold a ``/``."""
        segments = [unquote(s) for s in raw_path.split("/")[3:]]  # after "", "api", "procedures"
        payload = json.loads(body) if body else {}
        if not segments and method == "GET":
            return httpx.Response(200, json={"graphs": [PROC_SUMMARY] if PROC_NAME in self.procedures else []})
        if len(segments) == 3 and segments[0] == "evolve" and segments[2] == "events":
            if segments[1] == "job_000009":
                return self._stream(self.evolve_events)
            return self._stream([{"type": "error", "data": "Unknown job id."}])

        name, rest = segments[0], "/".join(segments[1:])
        if method == "PUT" and not rest:
            if payload.get("name") not in (None, name):
                return httpx.Response(422, json={"detail": {"message": "Graph name in body does not match the path", "diagnostics": []}})
            if not payload.get("nodes"):
                return httpx.Response(
                    422,
                    json={"detail": {"message": "Invalid procedural graph", "diagnostics": ["missing Start node", "no terminal node"]}},
                )
            previous = self.procedures.get(name, {}).get("version", 0)
            self.procedures[name] = {**payload, "version": previous + 1, "score": None}
            return httpx.Response(200, json={"name": name, "version": previous + 1})
        if name not in self.procedures:
            return httpx.Response(404, json={"detail": f"Procedural graph '{name}' not found."})
        stored = self.procedures[name]
        if method == "GET" and not rest:
            if params.get("format") == "text":
                return httpx.Response(200, json={"text": PROC_TEXT})
            return httpx.Response(200, json=stored)
        if method == "DELETE" and not rest:
            del self.procedures[name]
            return httpx.Response(200, json={"status": "success"})
        if method == "GET" and rest == "graph-data":
            return httpx.Response(200, json=PROC_GRAPH_DATA)
        if method == "POST" and rest == "guidance":
            return httpx.Response(200, json=_guidance_payload(name, payload))
        if method == "POST" and rest == "trajectories":
            return httpx.Response(200, json={"status": "recorded"})
        if method == "GET" and rest == "versions":
            return httpx.Response(200, json=PROC_VERSIONS)
        if method == "POST" and rest == "rollback":
            stored["version"] += 1
            return httpx.Response(200, json={"name": name, "version": stored["version"]})
        if method == "GET" and rest == "rejections":
            return httpx.Response(200, json=PROC_REJECTIONS)
        if method == "POST" and rest == "evolve":
            return httpx.Response(200, json={"job_id": "job_000009", "status": "processing"})
        return httpx.Response(404, json={"detail": "Not Found"})

    def _lab_done(self, run_id: str, status: str) -> dict[str, Any]:
        detail = self.lab_runs.get(run_id) or lab_detail(run_id, "retrieve", status)
        manifest = detail["manifest"]
        return {
            "run_id": run_id,
            "status": status,
            "mode": manifest["config"]["mode"],
            "dataset": manifest["dataset"],
            "reason": manifest.get("abort_reason") if status == "aborted" else None,
            "phases": {"retrieve": "done", "read": "done", "score": "done"},
            "spent_usd": manifest["actual"]["reader"]["usd"],
            "cap_usd": manifest["budget_cap_usd"],
            "pending": manifest["estimate"]["read_pending"],
            "estimate_upper_usd": 0.009,
        }

    def _lab_stream(self, run_id: str) -> httpx.Response:
        if self.lab_events is not None:
            return self._stream(self.lab_events)
        detail = self.lab_runs.get(run_id)
        if detail is None:
            return self._stream([{"type": "error", "data": "Unknown Lab run."}])
        mode = detail["manifest"]["config"]["mode"]
        events: list[dict[str, Any]] = [
            {"type": "phase", "phase": "retrieve", "status": "running", "total": 27, "run_id": run_id},
            *({"type": "progress", "phase": "retrieve", "done": i, "total": 27, "arm": "bm25", "run_id": run_id} for i in range(1, 28)),
            {"type": "phase", "phase": "retrieve", "status": "done", "run_id": run_id},
        ]
        if mode == "realtime":
            events += [
                {"type": "phase", "phase": "read", "status": "running", "total": 18, "run_id": run_id},
                {"type": "progress", "phase": "read", "done": 18, "total": 18, "spent_usd": 0.0021, "run_id": run_id},
            ]
        if mode == "batch":
            events.append({"type": "batch_submitted", "requests": 18, "batch": {"batches": ["b1"]}, "run_id": run_id})
        return self._stream([*events, {"type": "done", "data": self._lab_done(run_id, detail["status"])}])

    def _lab(self, method: str, raw_path: str, params: Any, body: bytes) -> httpx.Response:
        """``/api/lab/…`` routed like routers/lab.py (the run id is ONE encoded segment)."""
        segments = [unquote(s) for s in raw_path.split("/")[3:]]
        head = segments[0] if segments else ""
        if method == "GET" and head == "arms":
            return httpx.Response(200, json=LAB_ARMS)
        if method == "GET" and head == "models":
            return httpx.Response(200, json=LAB_MODELS)
        if method == "GET" and head == "datasets":
            return httpx.Response(200, json=LAB_DATASETS)
        if method == "POST" and head == "qa-files":
            self.lab_uploads.append(body)
            if b"replace" not in body and b"taken.jsonl" in body:
                return httpx.Response(409, json={"detail": "A different QA file named 'taken.jsonl' already exists; past runs refer to it by content hash. Upload under another name, or set replace=true."})
            return httpx.Response(200, json={"status": "created", "id": "qa-file:mine.jsonl", "name": "qa-file", "file": "mine.jsonl", "n": 3, "splits": {"test": 2, "train": 1}, "bytes": 300, "sha256": "f" * 64})
        payload = json.loads(body) if body else {}
        if method == "POST" and head == "estimate":
            return httpx.Response(200, json=lab_estimate_payload(payload))
        if head != "runs":
            return httpx.Response(404, json={"detail": "Not Found"})
        if len(segments) == 1 and method == "GET":
            runs = [
                {"run_id": rid, "status": d["status"], "created_at": d["manifest"]["created_at"], "mode": d["manifest"]["config"]["mode"], "dataset": "demo", "n": 9, "arms": d["manifest"]["config"]["arms"], "budgets": [500, None], "reader_model": "gpt-5-nano", "run_dir": f"/srv/lab_runs/{rid}", "active": False}
                for rid, d in self.lab_runs.items()
            ]
            return httpx.Response(200, json={"runs": runs})
        if len(segments) == 1 and method == "POST":
            mode = payload.get("mode", "retrieve")
            if mode != "retrieve" and payload.get("max_usd") is None:
                return httpx.Response(422, json={"detail": {"message": "A paid run needs max_usd", "diagnostics": ["set max_usd"]}})
            estimate = lab_estimate_payload(payload)
            if estimate["refuse"]:
                return httpx.Response(422, json={"detail": {"message": "Refused: the estimated upper bound breaks the spend cap", "diagnostics": [estimate["refuse_reason"]], "estimate": estimate}})
            run_id = payload.get("run_id") or "20260925-120000-d4e5f6"
            status = "batch_submitted" if mode == "batch" else "done"
            self.lab_runs[run_id] = lab_detail(run_id, mode, status)
            return httpx.Response(200, json={"job_id": "job_000010", "run_id": run_id, "status": "running", "mode": mode, "estimate": estimate, "events": f"/api/lab/runs/{run_id}/events"})
        run_id, rest = segments[1], "/".join(segments[2:])
        if method == "GET" and rest == "events":
            return self._lab_stream(run_id)
        if run_id not in self.lab_runs:
            return httpx.Response(404, json={"detail": f"Unknown Lab run {run_id!r}."})
        detail = self.lab_runs[run_id]
        if method == "GET" and not rest:
            shown = copy.deepcopy(detail)
            limit, offset = int(params.get("limit", 50)), int(params.get("offset", 0))
            rows = [r for r in LAB_ROWS if params.get("arm") in (None, r["arm"])]
            shown["rows"] = {"total": len(rows), "offset": offset, "limit": limit, "rows": rows[offset : offset + limit]}
            return httpx.Response(200, json=shown)
        if method == "POST" and rest == "resume":
            previous = detail["status"]
            status = self.lab_resume_status or "done"
            mode = detail["manifest"]["config"]["mode"]
            self.lab_runs[run_id] = lab_detail(run_id, mode, status, cap=payload.get("max_usd") or detail["manifest"]["budget_cap_usd"])
            return httpx.Response(200, json={"job_id": "job_000011", "run_id": run_id, "status": "running", "previous_status": previous, "events": f"/api/lab/runs/{run_id}/events"})
        return httpx.Response(404, json={"detail": "Not Found"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs: Any) -> SynapseClient:
        if kwargs.get("base_url") is None:  # the CLI passes base_url=None for "use env"
            kwargs["base_url"] = BASE_URL
        kwargs.setdefault("timeout", 5)
        return SynapseClient(transport=self.transport, **kwargs)

    def paths(self, method: str | None = None) -> list[str]:
        return [c.path for c in self.calls if method is None or c.method == method]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not inherit the developer's SYNAPSE_* settings."""
    for name in list(os.environ):
        if name.startswith("SYNAPSE_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "history.pdf"
    path.write_bytes(b"%PDF-1.4\n%fake\n")
    return path
