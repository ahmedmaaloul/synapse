# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedural Guidance (the online half of Procedural Graphs)

At every step of a multi-step task the agent is shown the part of its
Procedural Graph that matters *now* (Lu, Chen, Wu, Arık — "Procedural Graphs:
Self-Evolving Execution Structures for LLM Agents", arXiv:2609.09153, §3.2):

  1. localize — u_t = Match(a_{t-1}, V): which node the agent's last action
     corresponds to (``Start`` before any action);
  2. scope    — G_t = N_h(u_t): that node plus its directed outgoing
     transitions up to h = 2 hops, or the full graph when nothing matched;
  3. guide    — the paper hands (G_t, query, the last w = 3 steps) to a guidance
     LLM Ψ whose output is appended to the solver prompt. The solver still
     chooses its own action: this is soft steering, not a controller.

The graph is frozen while it is being used; learning happens offline
(``procedural_evolution``).

Three things here are Synapse's, not the paper's. Each is a testable
hypothesis, not a claim:

  • **Raw local guidance (the default).** ``mode="raw"`` returns the serialized
    local subgraph itself and makes ZERO extra LLM calls. The paper measures
    what its guidance LLM costs (Table 3: +33.4% / +55.4% total tokens on
    GDPval / ALFWorld; Table 9: 10,116 vs 4,003 tokens per HotpotQA question)
    but never tests "localized subgraph, injected raw", the cheapest cell of
    its own design space. Synapse's retrieval endpoints already hand context to
    the caller's model rather than paying for a second generation (see
    docs/finops.md); this is the same stance.
    ``mode="generative"`` is the paper's configuration.
  • **A localization cascade** instead of "exact match or full graph":
    ``start`` → ``exact`` → ``normalized`` (case, punctuation and arguments
    ignored, from ``procedural_graph.localize``) → ``semantic`` (cosine
    between the last action + observation and the node descriptions, at least
    ``procedural_semantic_threshold``) → ``none``, which falls back to the
    full graph. The paper's own ablation (Table 3) finds the full graph both
    worse and costlier than the local one, so a miss is worth one embedding.
    The semantic step only runs for a step that HAS an action (a failed parse
    falls back to the full graph, as the paper's Match would), and it never
    lands on ``Start`` or on a terminal node.
  • **A guidance cache** (generative mode). The key is (graph, version, active
    node, scope, sha1 of the rendered prompt). The prompt already contains the
    query and the last w steps, so equal keys mean the guidance LLM would read
    the same input. A repeated situation is then answered without a second
    call. The paper lists "reuse guidance across steps" as future work.

Never fatal: an embedding failure skips the semantic step, and an unknown graph
raises ``ProceduralGraphNotFound`` for the API to turn into a 404.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.services import procedural_store as store
from app.services.chat_engine import estimate_tokens
from app.services.llm_provider import get_chat_llm, get_embeddings
from app.services.procedural_graph import (
    START,
    Localization,
    ProceduralGraph,
    localize,
    serialize_full,
    serialize_local,
)
from app.services.procedural_store import ProceduralGraphNotFound

logger = logging.getLogger(__name__)

__all__ = [
    "GUIDANCE_MODES",
    "GUIDANCE_PROMPT",
    "ProceduralGraphNotFound",
    "clear_caches",
    "format_step_action",
    "format_window",
    "guide",
    "llm_usage",
    "response_text",
]

GUIDANCE_MODES: tuple[str, ...] = ("none", "raw", "generative")

#: Characters of the last observation embedded for semantic localization.
SEMANTIC_OBSERVATION_CHARS = 500
#: Characters of each observation shown to the guidance LLM in the w-step window.
WINDOW_OBSERVATION_CHARS = 800
#: Node-description embedding sets kept in-process: one per (graph, version,
#: content). Bounded because an evolution run localizes on many unsaved
#: candidate graphs.
NODE_EMBEDDING_CACHE_SIZE = 32

# Adapted from the paper's "Guidance Generation Prompt" (Appendix B.5). The
# wording is Synapse's; the slots and the instruction are the paper's. The
# local and full variants differ ONLY in the graph-context slot, as in the
# paper, so a local-vs-full comparison measures scope and nothing else.
GUIDANCE_PROMPT = """You are an expert cognitive architect and execution guide for an AI agent \
solving the task: {task_description}

Here is {graph_context_desc}:
{graph_context}

Here is the current active query / observation:
{query}

Here is the agent's recent execution trajectory (most recent last):
{recent_context}

Analyze this {graph_source} in the context of the agent's current progress. Using the \
condition, guidance and pitfalls attributes carried by the edges in the graph context, write \
clear, detailed and actionable guidance that tells the agent exactly which step or strategy to \
pursue next, which pitfalls to avoid, and how to recover from recent failures if there were any. \
Include the specific tools, command patterns or arguments defined in the graph context whenever \
they are relevant to the next steps."""

_LOCAL_CONTEXT_DESC = (
    "the localized Procedural Graph around the agent's current step (the active node and its "
    "directed transitions)"
)
_FULL_CONTEXT_DESC = (
    "the complete Procedural Graph governing the task structure and strategic guidance"
)
_FALLBACK_TASK = "the multi-step task described by the Procedural Graph below"

# (graph, version, active node, scope, sha1(prompt)) → guidance text.
_guidance_cache: OrderedDict[tuple, str] = OrderedDict()
# (graph, version, sha1(node texts)) → (node ids, vectors).
_node_embedding_cache: OrderedDict[tuple, tuple[list[str], list[list[float]]]] = OrderedDict()


def clear_caches() -> None:
    """Forget cached guidance and node embeddings (tests; after a model switch)."""
    _guidance_cache.clear()
    _node_embedding_cache.clear()


# ── LLM response helpers (shared with the agent and the evolution loop) ──
def response_text(response: Any) -> str:
    """The text of a LangChain response (``str`` or a list of content blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def llm_usage(response: Any, prompt_text: str) -> dict:
    """``{input_tokens, output_tokens, estimated}`` for one model call.

    Measured when the provider reports it (``usage_metadata``, else the raw
    ``response_metadata`` token block), otherwise estimated at chars / 4 and
    flagged, the same rule as ``benchmarks/public/cost.py``. A block that
    reports zero for both counts is treated as absent, because recording 0
    tokens would under-report the bill.
    """
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        tokens_in, tokens_out = (
            _int(metadata.get("input_tokens")),
            _int(metadata.get("output_tokens")),
        )
        if tokens_in or tokens_out:
            return {"input_tokens": tokens_in, "output_tokens": tokens_out, "estimated": False}
    raw = getattr(response, "response_metadata", None)
    if isinstance(raw, Mapping):
        block = raw.get("token_usage") or raw.get("usage") or {}
        if isinstance(block, Mapping):
            tokens_in = _int(block.get("prompt_tokens", block.get("input_tokens")))
            tokens_out = _int(block.get("completion_tokens", block.get("output_tokens")))
            if tokens_in or tokens_out:
                return {"input_tokens": tokens_in, "output_tokens": tokens_out, "estimated": False}
    return {
        "input_tokens": estimate_tokens(prompt_text),
        "output_tokens": estimate_tokens(response_text(response)),
        "estimated": True,
    }


# ── Trajectory formatting ────────────────────────────
def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def format_step_action(step: Mapping[str, Any]) -> str:
    """``tool(arg="value")`` when the step carries ``args``, else the bare action."""
    action = str(step.get("action") or "").strip()
    args = step.get("args")
    if action and isinstance(args, Mapping) and args:
        rendered = ", ".join(f"{key}={_quote(value)}" for key, value in args.items())
        return f"{action}({rendered})"
    return action


def _quote(value: Any) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def format_window(trajectory: Sequence[Mapping[str, Any]], window: int) -> str:
    """The last ``window`` steps, numbered by their position in the whole run."""
    if window <= 0 or not trajectory:
        return "(no steps taken yet)"
    recent = list(trajectory)[-window:]
    offset = len(trajectory) - len(recent)
    blocks = []
    for position, step in enumerate(recent, start=offset + 1):
        lines = [f"Step {position}:"]
        if step.get("thought"):
            lines.append(f"Thought: {_flat(step['thought'])}")
        lines.append(f"Action: {format_step_action(step) or '(no valid action)'}")
        observation = _flat(step.get("observation"))
        if observation:
            lines.append(f"Observation: {_clip(observation, WINDOW_OBSERVATION_CHARS)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ── Localization cascade ─────────────────────────────
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _node_text(graph: ProceduralGraph, node_id: str) -> str:
    node = graph.nodes[node_id]
    return f"{node.id} {node.description}".strip()


def _semantic_candidates(graph: ProceduralGraph) -> list[str]:
    """Nodes the semantic step may localize on: all but ``Start`` and the terminals.

    ``Start`` means "no action taken yet"; once the trajectory has a step,
    being localized back on it would be wrong. A terminal (out-degree 0, e.g.
    ``End``) means the procedure is over: its context tells the solver there
    is nothing left to do. A similarity score is not evidence enough for that,
    and in scratch mode (``Start → End``) ``End`` would be the ONLY candidate,
    so every match would steer the agent to stop. Excluding both leaves the
    paper's behaviour when nothing fits: the full graph.
    """
    terminals = set(graph.terminals())
    return [node_id for node_id in graph.nodes if node_id != START and node_id not in terminals]


async def _node_embeddings(
    graph: ProceduralGraph, name: str, version: int | None, embeddings: Any
) -> tuple[list[str], list[list[float]]]:
    """Embeddings of the semantic candidates, computed once per graph content.

    The key carries a fingerprint of the node texts as well as (name,
    version): evolution localizes on unsaved candidates, which all have
    version None.
    """
    ids = _semantic_candidates(graph)
    texts = [_node_text(graph, node_id) for node_id in ids]
    fingerprint = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()
    key = (name, version, fingerprint)
    cached = _node_embedding_cache.get(key)
    if cached is not None:
        _node_embedding_cache.move_to_end(key)
        return cached
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts) if texts else []
    entry = (ids, [list(v) for v in vectors])
    _node_embedding_cache[key] = entry
    while len(_node_embedding_cache) > NODE_EMBEDDING_CACHE_SIZE:
        _node_embedding_cache.popitem(last=False)
    return entry


async def _semantic_match(
    graph: ProceduralGraph,
    name: str,
    version: int | None,
    action: str,
    observation: str,
    threshold: float,
) -> Localization | None:
    """Best node by cosine similarity, or ``None`` below threshold or on any failure."""
    text = f"{action} {observation[:SEMANTIC_OBSERVATION_CHARS]}".strip()
    if not text or not _semantic_candidates(graph):
        return None
    try:
        embeddings = get_embeddings()
        ids, vectors = await _node_embeddings(graph, name, version, embeddings)
        if not ids or len(vectors) != len(ids):
            return None
        query_vector = await asyncio.to_thread(embeddings.embed_query, text)
    except Exception as e:  # noqa: BLE001 - localization degrades, guidance must not fail
        logger.info("Semantic localization skipped (embedding failed): %s", e)
        return None
    best_id, best_score = None, -1.0
    for node_id, vector in zip(ids, vectors, strict=True):
        similarity = _cosine(query_vector, vector)
        if similarity > best_score:
            best_id, best_score = node_id, similarity
    if best_id is None or best_score < threshold:
        return None
    return Localization(best_id, "semantic", round(best_score, 4))


async def _localize(
    graph: ProceduralGraph,
    name: str,
    version: int | None,
    last_step: Mapping[str, Any] | None,
    *,
    semantic: bool,
    threshold: float,
) -> Localization:
    """start → exact → normalized → semantic → none (see the module docstring)."""
    if last_step is None:
        return localize(graph, None)
    action = str(last_step.get("action") or "").strip()
    if not action:
        # A step without an action is a failed parse or a rejected call: its
        # observation is an error message (the Navigator's lists every tool
        # name), not a tool result. The paper's Match finds nothing for it and
        # uses the full graph. Embedding the error text would instead place
        # the agent on a node it never reached.
        return Localization(None, "none")
    found = localize(graph, action)
    if found.node_id is not None:
        return found
    if semantic:
        observation = str(last_step.get("observation") or "")
        found = await _semantic_match(graph, name, version, action, observation, threshold)
        if found is not None:
            return found
    return Localization(None, "none")


# ── Guidance ─────────────────────────────────────────
def _next_actions(graph: ProceduralGraph, node_id: str | None) -> list[str]:
    if node_id is None or node_id not in graph.nodes:
        return []
    return list(dict.fromkeys(edge.target for edge in graph.out_edges(node_id)))


def _cache_get(key: tuple) -> str | None:
    value = _guidance_cache.get(key)
    if value is not None:
        _guidance_cache.move_to_end(key)
    return value


def _cache_put(key: tuple, value: str, size: int) -> None:
    if size <= 0:
        return
    _guidance_cache[key] = value
    _guidance_cache.move_to_end(key)
    while len(_guidance_cache) > size:
        _guidance_cache.popitem(last=False)


async def guide(
    name: str,
    *,
    query: str,
    trajectory: Sequence[Mapping[str, Any]] | None,
    mode: str | None = None,
    hops: int | None = None,
    window: int | None = None,
    graph: ProceduralGraph | None = None,
    version: int | None = None,
    full_graph: bool = False,
    llm: Any = None,
) -> dict:
    """Situational guidance for the agent's next step.

    ``trajectory`` items are ``{"action", "observation"?, "thought"?, "args"?}``,
    oldest first. ``graph`` (with its ``version``) skips the store read: the
    agent loads its graph once per run, and evolution evaluates unsaved
    candidates. ``full_graph=True`` forces full-graph scope, which is the
    paper's Table 3 ablation. ``llm`` replaces the guidance model
    (``get_chat_llm(temperature=0)`` by default).

    Returns ``{graph, version, active_node, localization, localization_score,
    scope, context, guidance, next_actions, usage}``. ``usage`` holds
    ``context_chars``, ``context_tokens_est``, ``llm_calls``, ``cached``,
    ``input_tokens``, ``output_tokens`` and ``estimated``.

    * ``mode="none"``: no context, no LLM, no embedding call. Localization is
      still reported, from name matching only.
    * ``mode="raw"``: ``context`` is the serialized subgraph; ``guidance`` is None.
    * ``mode="generative"``: ``guidance`` comes from one LLM call, or from the
      cache when the same situation was seen before.

    Raises ``ProceduralGraphNotFound`` when ``graph`` is None and nothing is
    stored under ``name``.
    """
    settings = get_settings()
    mode = mode or settings.procedural_guidance_mode
    if mode not in GUIDANCE_MODES:
        raise ValueError(
            f"unknown guidance mode {mode!r} (expected one of {', '.join(GUIDANCE_MODES)})"
        )
    hops = settings.procedural_hops if hops is None else max(0, int(hops))
    window = settings.procedural_window if window is None else max(0, int(window))

    if graph is None:
        loaded = await store.load_graph_with_meta(name)
        if loaded is None:
            raise ProceduralGraphNotFound(name)
        graph, meta = loaded
        version = meta.get("version")

    steps = [step for step in (trajectory or []) if isinstance(step, Mapping)]
    location = await _localize(
        graph,
        name,
        version,
        steps[-1] if steps else None,
        # The semantic step costs an embedding call. Skip it when its answer
        # cannot change the context: no guidance, or a forced full graph.
        semantic=mode != "none" and not full_graph,
        threshold=float(settings.procedural_semantic_threshold),
    )
    scope = "local" if location.node_id is not None and not full_graph else "full"

    usage = {
        "context_chars": 0,
        "context_tokens_est": 0,
        "llm_calls": 0,
        "cached": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated": False,
    }
    result = {
        "graph": name,
        "version": version,
        "active_node": location.node_id,
        "localization": location.method,
        "localization_score": location.score,
        "scope": scope,
        "context": "",
        "guidance": None,
        "next_actions": _next_actions(graph, location.node_id),
        "usage": usage,
    }
    if mode == "none":
        return result

    if scope == "local":
        context = serialize_local(graph, location.node_id, hops)
    else:
        context = serialize_full(graph)
    result["context"] = context
    usage["context_chars"] = len(context)
    usage["context_tokens_est"] = estimate_tokens(context)
    if mode == "raw":
        return result

    prompt = GUIDANCE_PROMPT.format(
        task_description=_flat(graph.description) or _FALLBACK_TASK,
        graph_context_desc=_LOCAL_CONTEXT_DESC if scope == "local" else _FULL_CONTEXT_DESC,
        graph_context=context,
        query=query,
        recent_context=format_window(steps, window),
        graph_source="localized subgraph" if scope == "local" else "complete Procedural Graph",
    )
    key = (
        name,
        version,
        location.node_id,
        scope,
        hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
    )
    cached = _cache_get(key)
    if cached is not None:
        result["guidance"] = cached
        usage["cached"] = True
        return result

    model = llm if llm is not None else get_chat_llm(temperature=0)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    text = response_text(response).strip()
    usage["llm_calls"] = 1
    usage.update(llm_usage(response, prompt))
    result["guidance"] = text
    _cache_put(key, text, int(settings.procedural_guidance_cache_size))
    return result
