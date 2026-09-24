# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — GraphRAG Navigator (a ReAct agent that walks the knowledge graph)

``/api/chat`` answers in one shot: retrieve a subgraph, then generate. That is
the right shape for most questions, but it leaves no room for a *strategy*:
which entity to look up first, when a bridge entity has been found, when the
evidence is enough to commit. The Navigator answers step by step instead. Each
turn is one LLM call that writes a ``Thought`` and one ``Action``, and the
action runs one of six deterministic tools over Synapse's own graph:

    search_entities(query)    ranked entity names, types and descriptions
    neighbors(entity)         the entity's direct relations, with direction
    read_sources(entity)      the source passages it was extracted from
    search_passages(query)    semantic search over the source passages
    find_path(source, target) the shortest relation chain linking two entities
    answer(text)              submit the final answer (ends the episode)

The tools make no LLM call. They are built from the chat engine's retrieval
primitives (hybrid seeding, neighbourhood expansion, reasoning paths) and the
chunk store, so the agent sees exactly the evidence ``/api/chat`` would.

This agent is also where procedural memory becomes *measurable*. With a
Procedural Graph (Lu, Chen, Wu, Arık, arXiv:2609.09153) each prompt carries a
``Procedural Graph Guidance:`` block, built by ``procedural_guidance.guide`` from
the agent's last action. Evolution learns the navigator's strategy from QA pairs
scored by EM/F1. Semantic memory (entities) and procedural memory (how to walk
them) live in the same Neo4j.

The solver prompt is adapted from the paper's ReAct template (Appendix B.5):
exactly one Thought and one Action per turn, no simulated Observation. The
action parser is deliberately tolerant (optional quotes, one positional
argument, stray markdown). A turn that still cannot be executed gets a
corrective observation and counts as a ``parse_failure``, which is a column of
the paper's Table 9.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import zip_longest
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.neo4j_driver import execute_query
from app.services import chat_engine, chunk_store
from app.services import procedural_store as store
from app.services.llm_provider import ProviderConfigError, get_chat_llm
from app.services.procedural_graph import ProceduralGraph, normalize_action
from app.services.procedural_guidance import (
    GUIDANCE_MODES,
    format_step_action,
    guide,
    llm_usage,
    response_text,
)
from app.services.procedural_store import ProceduralGraphNotFound

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[str]]

# ── Tool result sizes (the observation cap applies on top) ──
SEARCH_K = 5  # entities returned by search_entities
READ_SOURCES_K = 3  # passages returned by read_sources
PASSAGES_K = 3  # passages returned by search_passages
PASSAGE_CHARS = 500  # per passage, so several fit under the observation cap
DESCRIPTION_CHARS = 200  # per entity description in search results
MAX_RELATIONS = 40  # relations listed by neighbors

INVALID_ACTION_FORMAT = 'Invalid action format. Use: Action: tool_name(arg="value")'


# ── Tool catalog ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ToolSpec:
    """How a tool is presented to the model: its name, argument names and purpose."""

    name: str
    params: tuple[str, ...]
    description: str

    @property
    def signature(self) -> str:
        args = ", ".join(f'{param}="..."' for param in self.params)
        return f"{self.name}({args})"


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "search_entities",
            ("query",),
            "find entities whose name or description matches; returns names, types and "
            "short descriptions. Use a specific name, not the whole question.",
        ),
        ToolSpec(
            "neighbors",
            ("entity",),
            "list the entity's direct relations (one hop), with their direction. Use the "
            "exact name returned by search_entities.",
        ),
        ToolSpec(
            "read_sources",
            ("entity",),
            "read the source passages the entity was extracted from (dates, roles, exact "
            "titles often only appear there).",
        ),
        ToolSpec(
            "search_passages",
            ("query",),
            "semantic search over the source passages, for when entity search misses or a "
            "specific phrase, date or number is needed.",
        ),
        ToolSpec(
            "find_path",
            ("source", "target"),
            "find the shortest chain of relations connecting two entities.",
        ),
        ToolSpec(
            "answer",
            ("text",),
            "submit the final answer and stop. Give the shortest exact form: a name, date, "
            "number, or yes/no.",
        ),
    )
}

#: The navigator's action vocabulary. Identical, by test, to the ACTION node ids
#: and ``tools`` of the ``graphrag-navigator`` expert prior, so localization by
#: action name can hit every tool node.
TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SPECS)
ANSWER_TOOL = "answer"

#: The task, as the guidance and refiner prompts describe it.
TASK_DESCRIPTION = (
    "Answer a (possibly multi-hop) question with a short exact answer by navigating a "
    "knowledge graph of entities, relations and source passages with the tools "
    + ", ".join(TOOL_NAMES)
    + "."
)


# ── Tools (deterministic, no LLM) ────────────────────
RESOLVE_ENTITY_QUERY = """
MATCH (n:Entity)
WHERE n.name = $name OR toLower(n.name) = toLower($name)
RETURN n.name AS name, n.type AS type, n.description AS description
ORDER BY CASE WHEN n.name = $name THEN 0 ELSE 1 END, n.name
LIMIT 1
"""


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _clean_arg(value: Any) -> str:
    return _flat(value).strip("\"'` ")


async def _resolve_entity(name: str) -> dict | None:
    """The stored entity for ``name``: exact match first, then case-insensitive.

    Models retype names ("ada lovelace"), and failing a lookup over letter case
    would waste a whole LLM turn.
    """
    name = _clean_arg(name)
    if not name:
        return None
    rows = await execute_query(RESOLVE_ENTITY_QUERY, {"name": name})
    return rows[0] if rows else None


def _not_found(name: str) -> str:
    return (
        f'No entity named "{_clean_arg(name)}" in the knowledge graph. Use search_entities to '
        "find its exact name."
    )


def _merge_seed_rows(*channels: Sequence[dict]) -> list[dict]:
    """Interleave ranked seed rows from several channels, deduped by name.

    The same interleaving as ``chat_engine._rank_seed_names``: a strong lexical
    hit is not buried behind many weaker semantic ones. Rows are kept whole
    because the agent needs the type and description too.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for group in zip_longest(*channels):
        for row in group:
            name = (row or {}).get("name")
            if name and name not in seen:
                seen.add(name)
                merged.append(row)
    return merged


def _format_passages(chunks: Sequence[dict]) -> list[str]:
    lines = []
    for position, chunk in enumerate(chunks, start=1):
        source = chunk.get("document") or "unknown source"
        text = _clip(_flat(chunk.get("text")), PASSAGE_CHARS)
        lines.append(f"[{position}] {source} (chunk {chunk.get('index', 0)}): {text}")
    return lines


async def search_entities(query: str) -> str:
    """Hybrid (vector + full-text) entity search: top names, types, descriptions."""
    query = _clean_arg(query)
    if not query:
        return "search_entities needs a non-empty query."
    vector_rows = await chat_engine._seeds_by_vector(query, SEARCH_K)
    keyword_rows = await chat_engine._seeds_by_keyword(query, SEARCH_K)
    rows = _merge_seed_rows(vector_rows or [], keyword_rows or [])[:SEARCH_K]
    if not rows:
        return f'No entities match "{query}". Try a shorter name or search_passages.'
    lines = [f'Entities matching "{query}":']
    for position, row in enumerate(rows, start=1):
        description = _clip(_flat(row.get("description")), DESCRIPTION_CHARS)
        line = f"{position}. {row['name']} ({row.get('type') or 'UNKNOWN'})"
        lines.append(f"{line}: {description}" if description else line)
    return "\n".join(lines)


async def neighbors(entity: str) -> str:
    """The entity's one-hop relations, rendered with their real direction."""
    node = await _resolve_entity(entity)
    if node is None:
        return _not_found(entity)
    name = node["name"]
    edges = await chat_engine._neighborhood_edges([name], 1)
    lines = [f"Entity: {name} ({node.get('type') or 'UNKNOWN'})"]
    if node.get("description"):
        lines.append(f"Description: {_flat(node['description'])}")
    if not edges:
        lines.append("No relations are recorded for this entity.")
        return "\n".join(lines)
    lines.append(f"Relations ({len(edges)}):")
    for edge in edges[:MAX_RELATIONS]:
        lines.append(f"- {edge['source']} -[{edge.get('rel') or 'RELATED_TO'}]-> {edge['target']}")
    if len(edges) > MAX_RELATIONS:
        lines.append(f"(+{len(edges) - MAX_RELATIONS} more relations not shown)")
    return "\n".join(lines)


async def read_sources(entity: str) -> str:
    """The source passages the entity was extracted from, most relevant first."""
    node = await _resolve_entity(entity)
    if node is None:
        return _not_found(entity)
    chunks = await chunk_store.chunks_for_entities([node["name"]], READ_SOURCES_K)
    if not chunks:
        return f'No source passages are linked to "{node["name"]}".'
    return "\n".join([f"Sources for {node['name']}:", *_format_passages(chunks)])


async def search_passages(query: str) -> str:
    """Semantic search over the stored source passages."""
    query = _clean_arg(query)
    if not query:
        return "search_passages needs a non-empty query."
    chunks = await chunk_store.search_chunks(query, PASSAGES_K)
    if not chunks:
        return f'No passages match "{query}".'
    return "\n".join([f'Passages matching "{query}":', *_format_passages(chunks)])


async def find_path(source: str, target: str) -> str:
    """Shortest relation chain(s) between two entities (the chat engine's paths)."""
    start, goal = await _resolve_entity(source), await _resolve_entity(target)
    if start is None:
        return _not_found(source)
    if goal is None:
        return _not_found(target)
    if start["name"] == goal["name"]:
        return f"{start['name']} and {goal['name']} are the same entity."
    settings = get_settings()
    hops = max(1, min(int(settings.retrieval_max_hops), chat_engine.MAX_HOPS_CAP))
    names = [start["name"], goal["name"]]
    edges = await chat_engine._neighborhood_edges(names, hops)
    paths = chat_engine.build_reasoning_paths(names, edges, settings)
    if not paths:
        return (
            f"No path found between {start['name']} and {goal['name']} within "
            f"{2 * hops} hops. This does not mean they are unrelated: look up each one separately."
        )
    return "\n".join(
        [f"Paths between {start['name']} and {goal['name']}:"] + [f"- {p['text']}" for p in paths]
    )


DEFAULT_TOOLS: dict[str, ToolFn] = {
    "search_entities": search_entities,
    "neighbors": neighbors,
    "read_sources": read_sources,
    "search_passages": search_passages,
    "find_path": find_path,
}


# ── Action parsing ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class ParsedAction:
    """One model turn, parsed. ``error`` is None if and only if it can be executed."""

    thought: str
    tool: str | None
    args: dict[str, str]
    raw_action: str
    error: str | None = None


_OBSERVATION_LINE = re.compile(
    r"^[\s*_`#>-]*Observation\s*[*_`]*\s*:", re.IGNORECASE | re.MULTILINE
)
_ACTION_LINE = re.compile(r"^[\s*_`#>-]*Action\s*[*_`]*\s*:[*_`\t ]*", re.IGNORECASE | re.MULTILINE)
_THOUGHT_LINE = re.compile(
    r"^[\s*_`#>-]*Thought\s*[*_`]*\s*:[*_`\t ]*", re.IGNORECASE | re.MULTILINE
)
_CALL = re.compile(r"^`*\s*([A-Za-z_][\w.\-]*)\s*\((.*)\)\s*`*\s*[.;]?\s*$", re.DOTALL)
_KWARG = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*)$", re.DOTALL)


def _split_top_level(raw: str) -> list[str]:
    """Split call arguments on commas outside quotes and brackets."""
    pieces: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    escaped = False
    for ch in raw:
        if quote:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            pieces.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    pieces.append("".join(current).strip())
    return [piece for piece in pieces if piece]


def _unquote(value: str) -> str:
    value = value.strip().strip("`").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        return re.sub(r"\\(.)", r"\1", inner).strip()
    return value


def _bind_args(raw: str, params: tuple[str, ...]) -> tuple[dict[str, str], list[str], str | None]:
    """Map ``raw`` call arguments onto ``params``.

    Returns ``(args, missing, error)``. Tolerated: keyword or positional
    arguments, quoted or not, and, for a one-argument tool, a mistyped keyword
    name or unquoted commas (``answer(Paris, France)``).
    """
    raw = raw.strip()
    if not params:
        return {}, [], None
    pieces = _split_top_level(raw)
    keywords: dict[str, str] = {}
    unknown: dict[str, str] = {}
    positional: list[str] = []
    for piece in pieces:
        match = _KWARG.match(piece)
        if match and piece.lstrip()[:1] not in "\"'":
            target = keywords if match.group(1) in params else unknown
            target[match.group(1)] = _unquote(match.group(2))
        else:
            positional.append(_unquote(piece))

    if len(params) == 1:
        param = params[0]
        if positional and (len(pieces) > 1):
            # Unquoted commas split one value into several pieces:
            # answer(Paris, France) or answer(text=Paris, France).
            prefix = re.match(rf"^\s*{re.escape(param)}\s*=\s*", raw)
            value = _unquote(raw[prefix.end() :] if prefix else raw)
        elif param in keywords:
            value = keywords[param]  # extra unknown keywords (k=5) are ignored
        elif positional:
            value = positional[0]
        elif len(unknown) == 1:
            value = next(iter(unknown.values()))  # a mistyped keyword name
        else:
            value = ""
        return ({param: value}, [], None) if value else ({}, [param], None)

    args = {key: value for key, value in keywords.items() if value}

    free = [p for p in params if p not in args]
    if len(positional) > len(free):
        return args, [], f"too many arguments: expected {', '.join(params)}"
    for param, value in zip(free, positional, strict=False):
        if value:
            args[param] = value
    missing = [p for p in params if p not in args]
    return args, missing, None


def parse_action(text: str, specs: Mapping[str, ToolSpec] | None = None) -> ParsedAction:
    """Parse one ``Thought: … / Action: tool(arg="…")`` turn.

    Anything from a line starting with ``Observation:`` on is dropped (the model
    simulating the environment), and so is a second ``Thought:`` after the
    action. Tool names are matched after ``normalize_action``, so
    ``SearchEntities`` and ``search-entities`` both resolve to
    ``search_entities``.
    """
    specs = TOOL_SPECS if specs is None else specs
    text = text or ""
    cut = _OBSERVATION_LINE.search(text)
    if cut:
        text = text[: cut.start()]
    action_match = _ACTION_LINE.search(text)
    if action_match is None:
        thought = _THOUGHT_LINE.sub("", text, count=1).strip()
        return ParsedAction(thought, None, {}, _clip(_flat(text), 300), INVALID_ACTION_FORMAT)

    thought = _THOUGHT_LINE.sub("", text[: action_match.start()], count=1).strip()
    rest = text[action_match.end() :]
    next_thought = _THOUGHT_LINE.search(rest)
    if next_thought:
        rest = rest[: next_thought.start()]
    rest = rest.strip()
    raw_action = _clip(rest.splitlines()[0].strip() if rest else "", 300)

    first_line = rest.splitlines()[0].strip() if rest else ""
    call = _CALL.match(first_line) or _CALL.match(rest)
    if call is None:
        return ParsedAction(thought, None, {}, raw_action, INVALID_ACTION_FORMAT)

    name = normalize_action(call.group(1))
    spec = specs.get(name)
    if spec is None:
        available = ", ".join(specs)
        return ParsedAction(
            thought,
            None,
            {},
            raw_action,
            f"Unknown tool '{call.group(1)}'. Available tools: {available}. "
            f'Use: Action: tool_name(arg="value")',
        )
    args, missing, error = _bind_args(call.group(2), spec.params)
    if error or missing:
        detail = error or f"missing {', '.join(missing)}"
        return ParsedAction(
            thought,
            None,
            {},
            raw_action,
            f"Invalid arguments for {name} ({detail}). Use: Action: {spec.signature}",
        )
    return ParsedAction(thought, name, args, raw_action)


# ── Prompt (adapted from the paper's ReAct solver template, Appendix B.5) ──
SYSTEM_PROMPT = """You are the Synapse GraphRAG Navigator. You answer a question by navigating a \
knowledge graph built from the user's documents: entities, the relations between them, and the \
source passages they were extracted from.

Tools (the only actions you can take):
{tool_list}

Rules:
- Ground the answer in tool observations; never answer from prior knowledge.
- Use entity names exactly as the tools return them.
- Finish with answer(text="...") giving the shortest exact answer: a name, date, number, or \
yes/no. No full sentences."""

# Printed above raw (serialized-subgraph) guidance so the solver knows how to
# read it. Generated guidance is already prose and is injected as-is.
RAW_GUIDANCE_PREAMBLE = (
    "(The recommended procedure around your current step. Take the transition whose condition "
    "matches your situation, follow its guidance and avoid its pitfalls. It is advice: you still "
    "choose the action.)"
)

SOLVER_TEMPLATE = """{system_prompt}
{guidance_block}
You must interleave Thought and Action. Your output format must be exactly:
Thought: <your reasoning about what to do next>
Action: <tool_name>(arg="value")

Example:
Thought: I need the entity's exact name before I can read its relations.
Action: search_entities(query="Ada Lovelace")

DO NOT write any "Observation:" block or any subsequent steps. Only output exactly one Thought \
and one Action. Do NOT simulate the tools' responses.

Question: {question}

Current Trajectory:
{trajectory}

Thought:"""


def _tool_list(specs: Mapping[str, ToolSpec]) -> str:
    return "\n".join(f"- {spec.signature}: {spec.description}" for spec in specs.values())


def _render_trajectory(steps: Sequence[Mapping[str, Any]]) -> str:
    if not steps:
        return "(no steps yet)"
    blocks = []
    for step in steps:
        action = step.get("action")
        shown = format_step_action(step) if action else step.get("raw_action") or "(unparseable)"
        lines = []
        if step.get("thought"):
            lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {shown}")
        if action != ANSWER_TOOL:
            lines.append(f"Observation: {step.get('observation', '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    steps: Sequence[Mapping[str, Any]],
    *,
    specs: Mapping[str, ToolSpec] | None = None,
    guidance: str = "",
    raw_guidance: bool = True,
) -> str:
    """The full solver prompt for the next turn."""
    specs = TOOL_SPECS if specs is None else specs
    guidance_block = ""
    if guidance.strip():
        body = f"{RAW_GUIDANCE_PREAMBLE}\n{guidance}" if raw_guidance else guidance
        guidance_block = f"\nProcedural Graph Guidance:\n{body}\n"
    return SOLVER_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT.format(tool_list=_tool_list(specs)),
        guidance_block=guidance_block,
        question=question,
        trajectory=_render_trajectory(steps),
    )


# ── The run ──────────────────────────────────────────
@dataclass
class AgentResult:
    """One Navigator episode. ``to_dict`` is the ``POST /api/agent/ask`` body.

    ``usage.llm_calls`` counts EVERY model call of the run (solver turns plus
    generative-guidance calls); ``usage.guidance_llm_calls`` is the guidance
    subset. ``usage.estimated`` is True when any call's tokens were estimated
    at chars / 4 because the provider reported none. ``graph`` is None when no
    procedural guidance was used.
    """

    question: str
    answer: str | None
    steps: list[dict]
    stopped: str  # "answer" | "max_steps" | "error"
    parse_failures: int
    usage: dict
    graph: dict | None
    latency_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _truncate_observation(text: str, max_chars: int) -> str:
    marker = " …[observation truncated]"
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)].rstrip() + marker


def _resolve_specs(tools: Mapping[str, ToolFn]) -> dict[str, ToolSpec]:
    specs = {}
    for name in tools:
        specs[name] = TOOL_SPECS.get(name) or ToolSpec(name, ("query",), "custom tool")
    specs[ANSWER_TOOL] = TOOL_SPECS[ANSWER_TOOL]
    return specs


async def run_agent(
    question: str,
    *,
    graph_name: str | None = None,
    guidance: str | None = None,
    max_steps: int | None = None,
    llm: Any = None,
    tools: Mapping[str, ToolFn] | None = None,
    graph: ProceduralGraph | None = None,
    graph_version: int | None = None,
    full_graph: bool = False,
    guidance_llm: Any = None,
) -> AgentResult:
    """Answer ``question`` by walking the knowledge graph, one Thought/Action per LLM call.

    ``graph_name`` names a stored Procedural Graph (loaded ONCE per run);
    ``None`` runs without procedural memory. ``graph`` (with
    ``graph_version``) supplies an in-memory graph instead, which is how
    evolution evaluates an unsaved candidate. ``guidance`` defaults to
    ``procedural_guidance_mode``, and ``"none"`` disables the graph.
    ``full_graph`` forces full-graph guidance (the paper's ablation).
    ``llm``, ``guidance_llm`` and ``tools`` are injectable: ``tools``
    REPLACES the default registry (``answer`` is always available), so a test
    can never reach the database through a tool it forgot to fake.

    Raises ``ProceduralGraphNotFound`` for an unknown ``graph_name`` and
    ``ProviderConfigError`` when no model is configured. Any other model
    failure ends the run with ``stopped="error"``.
    """
    settings = get_settings()
    started = time.perf_counter()
    mode = guidance or settings.procedural_guidance_mode
    if mode not in GUIDANCE_MODES:
        raise ValueError(
            f"unknown guidance mode {mode!r} (expected one of {', '.join(GUIDANCE_MODES)})"
        )
    steps_cap = settings.agent_max_steps if max_steps is None else max(1, int(max_steps))
    observation_cap = int(settings.agent_observation_max_chars)

    use_graph = mode != "none" and (graph is not None or bool(graph_name))
    if use_graph and graph is None:
        loaded = await store.load_graph_with_meta(graph_name)
        if loaded is None:
            raise ProceduralGraphNotFound(graph_name)
        graph, meta = loaded
        graph_version = meta.get("version")
    if use_graph:
        graph_name = graph_name or graph.name

    registry = dict(DEFAULT_TOOLS if tools is None else tools)
    registry.pop(ANSWER_TOOL, None)
    specs = _resolve_specs(registry)
    model = llm if llm is not None else get_chat_llm(temperature=settings.agent_temperature)

    steps: list[dict] = []
    usage = {
        "llm_calls": 0,
        "guidance_llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated": False,
        "context_chars": 0,
    }
    answer: str | None = None
    stopped = "max_steps"
    error: str | None = None
    parse_failures = 0

    for _ in range(steps_cap):
        injected, injected_raw = "", True
        localization, active_node, guidance_error = None, None, None
        if use_graph:
            try:
                advice = await guide(
                    graph_name,
                    query=question,
                    trajectory=steps,
                    mode=mode,
                    graph=graph,
                    version=graph_version,
                    full_graph=full_graph,
                    llm=guidance_llm,
                )
            except ProviderConfigError:
                raise
            except Exception as e:  # noqa: BLE001 - a guidance failure must not end the run
                logger.warning("⚠️ Procedural guidance failed; continuing without it: %s", e)
                guidance_error = f"{type(e).__name__}: {e}"
            else:
                # Generated guidance when there is some; otherwise the raw subgraph.
                injected_raw = not advice["guidance"]
                injected = advice["guidance"] or advice["context"]
                localization, active_node = advice["localization"], advice["active_node"]
                advice_usage = advice["usage"]
                usage["llm_calls"] += advice_usage["llm_calls"]
                usage["guidance_llm_calls"] += advice_usage["llm_calls"]
                usage["input_tokens"] += advice_usage["input_tokens"]
                usage["output_tokens"] += advice_usage["output_tokens"]
                usage["estimated"] = usage["estimated"] or advice_usage["estimated"]
        usage["context_chars"] += len(injected)

        prompt = build_prompt(
            question, steps, specs=specs, guidance=injected, raw_guidance=injected_raw
        )
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
        except ProviderConfigError:
            raise
        except Exception as e:  # noqa: BLE001 - reported on the result, not raised
            logger.warning("⚠️ Navigator LLM call failed: %s", e)
            stopped, error = "error", f"{type(e).__name__}: {e}"
            break
        call_usage = llm_usage(response, prompt)
        usage["llm_calls"] += 1
        usage["input_tokens"] += call_usage["input_tokens"]
        usage["output_tokens"] += call_usage["output_tokens"]
        usage["estimated"] = usage["estimated"] or call_usage["estimated"]

        parsed = parse_action(response_text(response), specs)
        step: dict[str, Any] = {
            "thought": parsed.thought,
            "action": parsed.tool,
            "args": dict(parsed.args),
            "observation": "",
            "guidance_context_chars": len(injected),
            "localization": localization,
            "active_node": active_node,
        }
        if guidance_error:
            step["guidance_error"] = guidance_error
        steps.append(step)

        if parsed.error:
            parse_failures += 1
            step["raw_action"] = parsed.raw_action
            step["observation"] = parsed.error
            continue
        if parsed.tool == ANSWER_TOOL:
            answer, stopped = parsed.args["text"], "answer"
            break
        try:
            observation = await registry[parsed.tool](**parsed.args)
        except Exception as e:  # noqa: BLE001 - the agent reads the error and adapts
            logger.info("Navigator tool %s failed: %s", parsed.tool, e)
            observation = f"Tool error ({parsed.tool}): {type(e).__name__}: {e}"
        step["observation"] = _truncate_observation(str(observation or ""), observation_cap)

    return AgentResult(
        question=question,
        answer=answer,
        steps=steps,
        stopped=stopped,
        parse_failures=parse_failures,
        usage=usage,
        graph={"name": graph_name, "version": graph_version} if use_graph else None,
        latency_s=round(time.perf_counter() - started, 3),
        error=error,
    )
