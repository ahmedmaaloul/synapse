# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedural Graphs (the data structure)

GraphRAG gives an agent *semantic* memory: what is true about the corpus
(entities, relations, source passages). It says nothing about *procedural*
memory: how to go about answering. Which tool comes first, when a bridge entity
has been found, when to stop searching and commit to an answer. In the CoALA
framing these are different memories, and an agent that re-derives its strategy
on every question pays for it in wasted tool calls and wrong turns.

This module implements the structure introduced by

    Lu, Chen, Wu, Arık — "Procedural Graphs: Self-Evolving Execution Structures
    for LLM Agents", Google, arXiv:2609.09153 (2026).
    https://arxiv.org/abs/2609.09153

A Procedural Graph G = (V, R, E, Φ) is a small directed attributed graph:

  • nodes abstract a tool ACTION, a REASONING step or a STATUS; ``Start``
    initialises localisation (a0 = Start);
  • edges are triplets (u, r, v) over the relation vocabulary LEADS_TO /
    TRIGGERS / PROVIDES_INPUT_FOR / CONVERGES_TO, each carrying the attributes
    Φ(e) = {condition, guidance, pitfalls};
  • ONLINE, the agent is localised on the node matching its last action and
    shown the directed h-hop neighbourhood of that node (``serialize_local``);
  • OFFLINE, a refiner LLM proposes JSON edits which ``prepare_candidate``
    applies exactly like PrepareCandidate in the paper's Algorithm 1 (line 10).

Everything here is PURE: no I/O, no LLM, no clock. Every rule the evolution loop
depends on (edit order, cycle repair, the terminal-reachability check) is
therefore unit-testable in isolation and deterministic run to run. Persistence
lives in ``procedural_store``; the online guidance service and the evolution
loop are built on top of this module.

Where Synapse departs from, or adds to, the paper it says so inline: the
``normalized`` localisation fallback, printing the relation label in the local
serialization, and upsert semantics for ``add_nodes``.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

# ── Vocabulary (paper §3.1) ──────────────────────────
NODE_TYPES: tuple[str, ...] = ("ACTION", "REASONING", "STATUS")
RELATIONS: tuple[str, ...] = ("LEADS_TO", "TRIGGERS", "PROVIDES_INPUT_FOR", "CONVERGES_TO")
CYCLE_POLICIES: tuple[str, ...] = ("forbid", "allow")
START = "Start"
END = "End"
DEFAULT_RELATION = "LEADS_TO"

#: The four arrays the refiner may return (paper, Appendix B.5 refiner prompt).
EDIT_KEYS: tuple[str, ...] = ("add_nodes", "delete_nodes", "add_edges", "delete_edges")

#: Graph names travel in URL paths (``/api/procedures/{name}``) and in the
#: ``"<graph>::<id>"`` node uid, so they are restricted to a slug. A ``/`` would
#: break routing; a ``:`` would make the uid ambiguous.
GRAPH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

CyclePolicy = Literal["forbid", "allow"]
LocalizationMethod = Literal["start", "exact", "normalized", "semantic", "none"]


class InvalidProceduralGraph(ValueError):
    """A graph, or a graph JSON document, failed structural validation.

    ``diagnostics`` is the full list of problems, so an API can return all of
    them at once instead of making the caller fix one error per round-trip.
    """

    def __init__(
        self, diagnostics: Iterable[str], message: str = "Invalid procedural graph"
    ) -> None:
        self.diagnostics = list(diagnostics)
        self.message = message
        shown = "; ".join(self.diagnostics[:5])
        extra = len(self.diagnostics) - 5
        suffix = f" (+{extra} more)" if extra > 0 else ""
        super().__init__(f"{message}: {shown}{suffix}" if shown else message)


# ── Data model ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ProcNode:
    """A node: an abstracted tool ACTION, a REASONING step or a STATUS."""

    id: str
    type: str
    description: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "description": self.description}


@dataclass(frozen=True, slots=True)
class ProcEdge:
    """A triplet (source, relation, target) with its attributes Φ(e).

    ``condition`` is a natural-language precondition, or ``None`` when the
    transition is unconditional (the paper's ``null``).
    """

    source: str
    target: str
    relation: str = DEFAULT_RELATION
    condition: str | None = None
    guidance: str = ""
    pitfalls: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        """Edge identity: two edges with the same triplet are the same edge."""
        return (self.source, self.relation, self.target)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "condition": self.condition,
            "guidance": self.guidance,
            "pitfalls": self.pitfalls,
        }


@dataclass(slots=True)
class ProceduralGraph:
    """G = (V, R, E, Φ) plus the metadata the refiner prompt needs.

    ``nodes`` keeps insertion order, which is the authoring order: it is what
    makes ``to_dict`` and the serializers byte-stable across save/load.
    ``tools`` is the action catalog ACTION nodes are expected to match (only a
    *warning* when violated, exactly as in the paper, where tool-catalog
    membership is a refiner-prompt requirement, not a structural check).
    """

    name: str
    nodes: dict[str, ProcNode]
    edges: list[ProcEdge]
    description: str = ""
    cycle_policy: str = "forbid"
    tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.tools = tuple(self.tools)

    def copy(self) -> ProceduralGraph:
        """An independent copy. Nodes/edges are frozen, so a shallow copy is safe."""
        return ProceduralGraph(
            name=self.name,
            nodes=dict(self.nodes),
            edges=list(self.edges),
            description=self.description,
            cycle_policy=self.cycle_policy,
            tools=tuple(self.tools),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "cycle_policy": self.cycle_policy,
            "tools": list(self.tools),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str | None = None) -> ProceduralGraph:
        """Build a graph from its JSON shape (see ``to_dict``).

        ``name`` overrides (or supplies) the name in ``data``: an API path
        parameter wins over the body. Unknown top-level keys are ignored, so the
        output of ``GET /api/procedures/{name}`` (which adds version/score) can
        be edited and PUT back as-is.

        Raises :class:`InvalidProceduralGraph` only for problems that make the
        document impossible to represent (not an object, missing or duplicate
        ids, non-text attribute values). *Content* problems (unknown types or
        relations, dangling endpoints, cycles) produce a graph object that
        :func:`validate` then reports on, so every problem is listed together.
        Types and relations are case-normalized (``"action"`` → ``"ACTION"``,
        ``"leads to"`` → ``"LEADS_TO"``).
        """
        if not isinstance(data, Mapping):
            raise InvalidProceduralGraph(["graph JSON must be an object"])
        problems: list[str] = []

        raw_name = name if name is not None else data.get("name")
        graph_name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not graph_name:
            problems.append("graph 'name' is missing")

        description = _text_or_problem(data.get("description"), "description", problems)
        raw_policy = data.get("cycle_policy") or "forbid"
        cycle_policy = raw_policy.strip().lower() if isinstance(raw_policy, str) else "forbid"
        if not isinstance(raw_policy, str):
            problems.append("'cycle_policy' must be a string ('forbid' or 'allow')")

        tools: list[str] = []
        raw_tools = data.get("tools", [])
        if raw_tools is None:
            raw_tools = []
        if not isinstance(raw_tools, list):
            problems.append("'tools' must be a list of tool names")
        else:
            for i, tool in enumerate(raw_tools):
                if isinstance(tool, str) and tool.strip():
                    if tool.strip() not in tools:
                        tools.append(tool.strip())
                else:
                    problems.append(f"tools[{i}] must be a non-empty string")

        nodes: dict[str, ProcNode] = {}
        raw_nodes = data.get("nodes", [])
        if not isinstance(raw_nodes, list):
            problems.append("'nodes' must be a list")
            raw_nodes = []
        for i, raw in enumerate(raw_nodes):
            where = f"nodes[{i}]"
            if not isinstance(raw, Mapping):
                problems.append(f"{where} must be an object")
                continue
            node_id = _clean_id(raw.get("id"))
            if node_id is None:
                problems.append(f"{where} has no 'id'")
                continue
            if node_id in nodes:
                problems.append(f"{where}: duplicate node id '{node_id}'")
                continue
            node_type = _clean_type(raw.get("type")) or ""
            node_description = _text_or_problem(
                raw.get("description"), f"{where}.description", problems
            )
            nodes[node_id] = ProcNode(node_id, node_type, node_description)

        edges: list[ProcEdge] = []
        raw_edges = data.get("edges", [])
        if not isinstance(raw_edges, list):
            problems.append("'edges' must be a list")
            raw_edges = []
        for i, raw in enumerate(raw_edges):
            edge = _parse_edge(raw, f"edges[{i}]", problems)
            if edge is not None:
                edges.append(edge)

        if problems:
            raise InvalidProceduralGraph(problems, "Malformed procedural graph JSON")
        return cls(
            name=graph_name,
            nodes=nodes,
            edges=edges,
            description=description,
            cycle_policy=cycle_policy,
            tools=tuple(tools),
        )

    def out_edges(self, node_id: str) -> list[ProcEdge]:
        """Outgoing transitions of ``node_id``, in edge-list order."""
        return [edge for edge in self.edges if edge.source == node_id]

    def terminals(self) -> list[str]:
        """Nodes with out-degree 0, in node order.

        The paper's terminal is *any* zero-out-degree node, not specifically
        ``End`` (Appendix B.6).
        """
        sources = {edge.source for edge in self.edges}
        return [node_id for node_id in self.nodes if node_id not in sources]


@dataclass(frozen=True, slots=True)
class Localization:
    """Where the agent is in the graph: u_t = Match(a_{t-1}, V).

    ``node_id`` is ``None`` when nothing matched (``method == "none"``), which
    the guidance layer answers with the full graph. ``score`` carries the cosine
    similarity when the (embedding-based) ``semantic`` step matched.
    """

    node_id: str | None
    method: str
    score: float | None = None

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "method": self.method, "score": self.score}


# ── Parsing helpers (shared by from_dict and apply_edits) ──
class _Malformed(ValueError):
    """A field value that cannot be turned into text (an object, a number list…)."""


def _clean_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _clean_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().upper()


def _clean_relation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"[\s-]+", "_", value.strip()).upper()


def _clean_text(value: Any) -> str:
    """Text attribute → str. ``None`` → ``""``; a list of strings is joined.

    LLM refiners regularly return ``pitfalls`` as a list of bullet strings; that
    is a formatting quirk, not a structural error, so it is joined, not rejected.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool | int | float):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "; ".join(item.strip() for item in value if item.strip())
    raise _Malformed(type(value).__name__)


def _clean_condition(value: Any) -> str | None:
    """Condition → text, or ``None`` for an unconditional transition.

    ``""``, ``"null"`` and ``"none"`` all mean "unconditional": models emit the
    JSON literal as a string often enough that treating it as a precondition
    named "null" would be absurd.
    """
    text = _clean_text(value)
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _text_or_problem(value: Any, where: str, problems: list[str]) -> str:
    try:
        return _clean_text(value)
    except _Malformed as e:
        problems.append(f"{where} must be text, got {e}")
        return ""


def _parse_edge(raw: Any, where: str, problems: list[str]) -> ProcEdge | None:
    """Parse one edge object; append to ``problems`` and return None if malformed.

    A missing relation defaults to LEADS_TO; an unknown one is kept verbatim so
    :func:`validate` (or ``apply_edits``) can name it in its diagnostic.
    """
    if not isinstance(raw, Mapping):
        problems.append(f"{where} must be an object")
        return None
    source = _clean_id(raw.get("source"))
    target = _clean_id(raw.get("target"))
    if source is None or target is None:
        problems.append(f"{where} needs non-empty 'source' and 'target'")
        return None
    raw_relation = raw.get("relation")
    relation = DEFAULT_RELATION if raw_relation is None else _clean_relation(raw_relation)
    if relation is None:
        problems.append(f"{where}: relation must be a string")
        return None
    try:
        condition = _clean_condition(raw.get("condition"))
        guidance = _clean_text(raw.get("guidance"))
        pitfalls = _clean_text(raw.get("pitfalls"))
    except _Malformed as e:
        problems.append(f"{where}: condition/guidance/pitfalls must be text, got {e}")
        return None
    return ProcEdge(source, target, relation, condition, guidance, pitfalls)


def edge_label(edge: ProcEdge) -> str:
    """Human-readable ``[A] -RELATION→ [B]`` used in notes and diagnostics."""
    return f"[{edge.source}] -{edge.relation}→ [{edge.target}]"


# ── Construction ─────────────────────────────────────
def skeleton(name: str, tools: Iterable[str] = (), description: str = "") -> ProceduralGraph:
    """The paper's scratch starting point: ``Start → End`` and nothing else."""
    return ProceduralGraph(
        name=name,
        nodes={
            START: ProcNode(START, "STATUS", "The task has been received; no action taken yet."),
            END: ProcNode(END, "STATUS", "The task is complete."),
        },
        edges=[ProcEdge(START, END, DEFAULT_RELATION)],
        description=description,
        tools=tuple(tools),
    )


# ── Edits (refiner output) ───────────────────────────
def apply_edits(graph: ProceduralGraph, edits: Any) -> tuple[ProceduralGraph, list[str]]:
    """Apply refiner edits to a COPY of ``graph``; return ``(graph, diagnostics)``.

    Order is the paper's: ``delete_edges`` → ``delete_nodes`` → ``add_nodes`` →
    ``add_edges``. So a refiner can delete every edge between two nodes and
    re-add the one it wants to keep in the same response.

    Every malformed entry is skipped and reported (non-object, missing
    id/source/target, unknown node type or relation, an endpoint that does not
    exist after the deletions and additions, unknown top-level keys). The input
    graph is never mutated.
    """
    candidate, diagnostics, _notes = _apply_edits(graph, edits)
    return candidate, diagnostics


def _apply_edits(
    graph: ProceduralGraph, edits: Any
) -> tuple[ProceduralGraph, list[str], list[str]]:
    """``apply_edits`` plus informational notes (no-ops, upserts, cascades)."""
    candidate = graph.copy()
    diagnostics: list[str] = []
    notes: list[str] = []

    if not isinstance(edits, Mapping):
        diagnostics.append(
            f"edits must be a JSON object with {', '.join(EDIT_KEYS)}; "
            f"got {type(edits).__name__}"
        )
        return candidate, diagnostics, notes

    for key in edits:
        if key not in EDIT_KEYS:
            diagnostics.append(f"unknown edit key '{key}' (expected {', '.join(EDIT_KEYS)})")

    sections: dict[str, list] = {}
    for key in EDIT_KEYS:
        value = edits.get(key)
        if value is None:
            sections[key] = []
        elif isinstance(value, list):
            sections[key] = value
        else:
            diagnostics.append(f"'{key}' must be a list")
            sections[key] = []

    if not any(sections.values()):
        notes.append("no edits proposed: the candidate is identical to the retained graph")

    # 1. delete_edges — removes ALL edges between source and target,
    #    regardless of relation (paper, Appendix B.5).
    for i, entry in enumerate(sections["delete_edges"]):
        where = f"delete_edges[{i}]"
        if not isinstance(entry, Mapping):
            diagnostics.append(f"{where} must be an object with 'source' and 'target'")
            continue
        source, target = _clean_id(entry.get("source")), _clean_id(entry.get("target"))
        if source is None or target is None:
            diagnostics.append(f"{where} needs non-empty 'source' and 'target'")
            continue
        kept = [e for e in candidate.edges if not (e.source == source and e.target == target)]
        if len(kept) == len(candidate.edges):
            notes.append(f"{where}: no edge [{source}]→[{target}] to delete (no-op)")
        candidate.edges = kept

    # 2. delete_nodes — incident edges go with the node (they would dangle).
    for i, entry in enumerate(sections["delete_nodes"]):
        where = f"delete_nodes[{i}]"
        raw_id = entry.get("id") if isinstance(entry, Mapping) else entry
        node_id = _clean_id(raw_id)
        if node_id is None:
            diagnostics.append(f"{where} must be a node id")
            continue
        if node_id not in candidate.nodes:
            notes.append(f"{where}: no node [{node_id}] to delete (no-op)")
            continue
        del candidate.nodes[node_id]
        kept = [e for e in candidate.edges if node_id not in (e.source, e.target)]
        dropped = len(candidate.edges) - len(kept)
        if dropped:
            notes.append(f"{where}: deleting [{node_id}] also removed {dropped} incident edge(s)")
        candidate.edges = kept

    # 3. add_nodes — an existing id is UPDATED (upsert). The paper's prompt only
    #    says "add"; refiners routinely re-emit an existing node to reword its
    #    description, and rejecting that as a duplicate would waste a round.
    for i, entry in enumerate(sections["add_nodes"]):
        where = f"add_nodes[{i}]"
        if not isinstance(entry, Mapping):
            diagnostics.append(f"{where} must be an object with 'id', 'type', 'description'")
            continue
        node_id = _clean_id(entry.get("id"))
        if node_id is None:
            diagnostics.append(f"{where} has no 'id'")
            continue
        existing = candidate.nodes.get(node_id)
        raw_type = entry.get("type")
        if raw_type is None:
            if existing is None:
                diagnostics.append(f"{where}: new node '{node_id}' has no 'type'")
                continue
            node_type = existing.type
        else:
            node_type = _clean_type(raw_type)
            if node_type not in NODE_TYPES:
                diagnostics.append(
                    f"{where}: node '{node_id}' has invalid type {raw_type!r} "
                    f"(expected one of {', '.join(NODE_TYPES)})"
                )
                continue
        if "description" in entry:
            try:
                description = _clean_text(entry.get("description"))
            except _Malformed as e:
                diagnostics.append(f"{where}: description must be text, got {e}")
                continue
        else:
            description = existing.description if existing else ""
        candidate.nodes[node_id] = ProcNode(node_id, node_type, description)
        if existing is not None:
            notes.append(f"{where}: updated existing node [{node_id}]")

    # 4. add_edges — endpoints must exist AFTER the deletions and additions.
    for i, entry in enumerate(sections["add_edges"]):
        where = f"add_edges[{i}]"
        entry_problems: list[str] = []
        edge = _parse_edge(entry, where, entry_problems)
        if edge is None:
            diagnostics.extend(entry_problems)
            continue
        if isinstance(entry, Mapping) and entry.get("relation") is None:
            notes.append(f"{where}: no relation given, defaulted to {DEFAULT_RELATION}")
        if edge.relation not in RELATIONS:
            diagnostics.append(
                f"{where}: invalid relation {edge.relation!r} "
                f"(expected one of {', '.join(RELATIONS)})"
            )
            continue
        missing = [n for n in dict.fromkeys((edge.source, edge.target)) if n not in candidate.nodes]
        if missing:
            diagnostics.append(
                f"{where}: edge [{edge.source}]→[{edge.target}] references missing "
                f"node(s): {', '.join(missing)}"
            )
            continue
        for j, current in enumerate(candidate.edges):
            if current.key == edge.key:
                candidate.edges[j] = edge
                notes.append(f"{where}: updated existing edge {edge_label(edge)}")
                break
        else:
            candidate.edges.append(edge)

    return candidate, diagnostics, notes


# ── Structure ────────────────────────────────────────
def _adjacency(graph: ProceduralGraph) -> dict[str, list[ProcEdge]]:
    out: dict[str, list[ProcEdge]] = {}
    for edge in graph.edges:
        out.setdefault(edge.source, []).append(edge)
    return out


def _dfs_roots(graph: ProceduralGraph) -> list[str]:
    """Start first (the paper's a0), then every other node in sorted order.

    Edge endpoints that are not nodes are included so a cycle through a dangling
    id is still visited — validate() reports the dangling id separately.
    """
    others = set(graph.nodes)
    for edge in graph.edges:
        others.update((edge.source, edge.target))
    others.discard(START)
    return ([START] if START in graph.nodes else []) + sorted(others)


def _walk_back_edges(graph: ProceduralGraph) -> tuple[list[int], list[str] | None]:
    """Iterative DFS; return (indices of back edges, the first cycle found).

    Iterative rather than recursive so a large refiner-produced graph cannot hit
    Python's recursion limit. Edges are followed in edge-list order, so the
    result is fully determined by the graph.
    """
    out: dict[str, list[int]] = {}
    for idx, edge in enumerate(graph.edges):
        out.setdefault(edge.source, []).append(idx)

    white, gray, black = 0, 1, 2
    color: dict[str, int] = {}
    back_edges: list[int] = []
    first_cycle: list[str] | None = None

    for root in _dfs_roots(graph):
        if color.get(root, white) != white:
            continue
        color[root] = gray
        stack: list[tuple[str, Any]] = [(root, iter(out.get(root, ())))]
        while stack:
            node, pending = stack[-1]
            descended = False
            for idx in pending:
                target = graph.edges[idx].target
                state = color.get(target, white)
                if state == gray:
                    back_edges.append(idx)
                    if first_cycle is None:
                        path = [n for n, _ in stack]
                        first_cycle = path[path.index(target) :] + [target]
                elif state == white:
                    color[target] = gray
                    stack.append((target, iter(out.get(target, ()))))
                    descended = True
                    break
            if not descended:
                color[node] = black
                stack.pop()
    return sorted(back_edges), first_cycle


def find_cycle(graph: ProceduralGraph) -> list[str] | None:
    """One directed cycle as a node path (first == last), or ``None``."""
    return _walk_back_edges(graph)[1]


def repair_cycles(graph: ProceduralGraph) -> tuple[ProceduralGraph, list[ProcEdge]]:
    """Remove cycle-closing (back) edges; return ``(acyclic copy, removed edges)``.

    Deterministic DFS from ``Start``, then any remaining nodes in sorted order.
    Removing every DFS back edge always yields a DAG, and rooting the search at
    ``Start`` means the edge that is cut is the one that *returns* toward the
    beginning of the procedure — the forward path the agent walks survives.
    This is the paper's "removes detected cycle-closing edges before validation"
    when cycles are disallowed (Appendix B.6).
    """
    back_edges, _ = _walk_back_edges(graph)
    repaired = graph.copy()
    if not back_edges:
        return repaired, []
    cut = set(back_edges)
    removed = [graph.edges[i] for i in back_edges]
    repaired.edges = [edge for i, edge in enumerate(graph.edges) if i not in cut]
    return repaired, removed


def _reaches_terminal(graph: ProceduralGraph) -> set[str]:
    """Nodes with a directed path to some terminal (reverse BFS from terminals)."""
    incoming: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.source in graph.nodes and edge.target in graph.nodes:
            incoming.setdefault(edge.target, []).append(edge.source)
    seen = set(graph.terminals())
    queue = deque(seen)
    while queue:
        node = queue.popleft()
        for source in incoming.get(node, ()):
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return seen


def _reachable_from(graph: ProceduralGraph, root: str) -> set[str]:
    out = _adjacency(graph)
    seen = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for edge in out.get(node, ()):
            if edge.target not in seen:
                seen.add(edge.target)
                queue.append(edge.target)
    return seen


def validate(graph: ProceduralGraph) -> list[str]:
    """Structural failures only; ``[]`` means the graph is admissible.

    Checks: a valid name and cycle policy, ``Start`` present, valid node types
    and relations, every edge endpoint exists, at least one terminal
    (out-degree 0), every node has a directed path to a terminal, and, when
    ``cycle_policy == "forbid"``, no cycle. Tool-catalog membership and
    reachability *from* Start are deliberately not failures; see
    :func:`warnings` (the paper's validator does not enforce them either).
    """
    problems: list[str] = []
    if not isinstance(graph.name, str) or not GRAPH_NAME_PATTERN.match(graph.name):
        problems.append(
            f"invalid graph name {graph.name!r}: use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    if graph.cycle_policy not in CYCLE_POLICIES:
        problems.append(
            f"invalid cycle_policy {graph.cycle_policy!r} (expected one of "
            f"{', '.join(CYCLE_POLICIES)})"
        )
    if START not in graph.nodes:
        problems.append(f"missing the '{START}' node (localisation starts there)")

    for key, node in graph.nodes.items():
        if key != node.id:
            problems.append(f"node stored under key '{key}' has id '{node.id}'")
        if node.type not in NODE_TYPES:
            problems.append(
                f"node [{node.id}] has invalid type {node.type!r} "
                f"(expected one of {', '.join(NODE_TYPES)})"
            )

    for edge in graph.edges:
        if edge.relation not in RELATIONS:
            problems.append(
                f"edge {edge_label(edge)} has invalid relation {edge.relation!r} "
                f"(expected one of {', '.join(RELATIONS)})"
            )
        missing = [n for n in dict.fromkeys((edge.source, edge.target)) if n not in graph.nodes]
        if missing:
            problems.append(f"edge {edge_label(edge)} references missing node(s): {', '.join(missing)}")

    if graph.nodes:
        if not graph.terminals():
            problems.append("no terminal node (every node has an outgoing transition)")
        else:
            reaching = _reaches_terminal(graph)
            stuck = [node_id for node_id in graph.nodes if node_id not in reaching]
            if stuck:
                problems.append(f"no directed path to a terminal node from: {', '.join(stuck)}")

    if graph.cycle_policy == "forbid":
        cycle = find_cycle(graph)
        if cycle:
            problems.append(f"cycle not allowed (cycle_policy='forbid'): {' → '.join(cycle)}")
    return problems


def warnings(graph: ProceduralGraph) -> list[str]:
    """Non-fatal quality issues worth surfacing to a human or the refiner.

    * ACTION nodes whose id is not in ``graph.tools`` (only when ``tools`` is
      set) — localisation matches on the action name, so such a node can never
      become the active node;
    * nodes unreachable from ``Start`` — dead weight in every full-graph prompt;
    * transitions with empty guidance — the attribute the paper finds most
      consistently populated and the one the solver actually reads.
    """
    notes: list[str] = []
    if graph.tools:
        tools = set(graph.tools)
        for node in graph.nodes.values():
            if node.type == "ACTION" and node.id not in tools:
                notes.append(
                    f"ACTION node [{node.id}] is not one of the tools "
                    f"({', '.join(graph.tools)}); the agent can never be localised on it"
                )
    if START in graph.nodes:
        reachable = _reachable_from(graph, START)
        unreachable = [node_id for node_id in graph.nodes if node_id not in reachable]
        if unreachable:
            notes.append(f"unreachable from {START}: {', '.join(unreachable)}")
    for edge in graph.edges:
        if not edge.guidance.strip():
            notes.append(f"transition {edge_label(edge)} has no guidance")
    return notes


def prepare_candidate(
    graph: ProceduralGraph, edits: Any
) -> tuple[ProceduralGraph | None, list[str], list[str]]:
    """Algorithm 1, line 10: ``(G_cand, d_k) ← PrepareCandidate(G_{k-1}, ΔG_k, c)``.

    Returns ``(candidate, diagnostics, notes)``:

    1. apply the edits to a copy in the paper's order (``apply_edits``);
    2. if ``graph.cycle_policy == "forbid"``, remove cycle-closing edges
       (``repair_cycles``) — each removal is listed in ``notes``;
    3. run the structural checks (``validate``).

    Any diagnostic from step 1 or 3 is a structural failure, and the candidate
    is then returned as ``None``. That is the paper's "G_cand may be
    unavailable" made strict: a partially-applied candidate is never handed to
    the caller, so it cannot be evaluated, saved or diffed by mistake. The
    checks in step 3 still run after an edit failure so the refiner's rejection
    memory receives *every* problem with its proposal in one round, not just the
    first. ``diagnostics == []`` if and only if a candidate is returned.
    """
    candidate, diagnostics, notes = _apply_edits(graph, edits)
    if graph.cycle_policy == "forbid":
        candidate, removed = repair_cycles(candidate)
        notes.extend(f"removed cycle-closing edge {edge_label(edge)}" for edge in removed)
    diagnostics.extend(validate(candidate))
    if diagnostics:
        return None, diagnostics, notes
    return candidate, diagnostics, notes


# ── Online localisation ──────────────────────────────
_ACTION_PREFIX = re.compile(r"^\s*action\s*:\s*", re.IGNORECASE)
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_action(text: str | None) -> str:
    """Canonical snake_case action name: arguments stripped, case/punctuation folded.

    ``'search_entities(query="x")'`` → ``"search_entities"``;
    ``"Action: FindPath(source=...)"`` → ``"find_path"``;
    ``"Search-Entities"`` → ``"search_entities"``.
    """
    if not text:
        return ""
    name = _ACTION_PREFIX.sub("", str(text)).split("(", 1)[0].strip()
    name = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", name))
    return _NON_ALNUM.sub("_", name.lower()).strip("_")


def _match_key(text: str) -> str:
    """Underscore-insensitive form of ``normalize_action`` (``searchentities``)."""
    return normalize_action(text).replace("_", "")


def localize(graph: ProceduralGraph, last_action: str | None) -> Localization:
    """u_t = Match(a_{t-1}, V), with Synapse's normalized fallback.

    * ``start``      — no previous action (empty trajectory): a0 = Start;
    * ``exact``      — the action string equals a node id (the paper's Match);
    * ``normalized`` — equal after ``normalize_action`` on both sides, ignoring
      underscores (Synapse's addition: models write ``Search_Entities`` or
      include the arguments, and an exact-only match would then fall back to
      the full graph, which the paper's Table 3 shows is worse *and* costlier);
    * ``none``       — no match; the caller uses the full graph. (The
      embedding-based ``semantic`` step lives in the guidance service, which
      has embeddings — this module stays pure.)

    Among several normalized matches an ACTION node wins, then node order.
    """
    raw = (last_action or "").strip()
    if not raw:
        if START in graph.nodes:
            return Localization(START, "start")
        return Localization(None, "none")
    if raw in graph.nodes:
        return Localization(raw, "exact")
    key = _match_key(raw)
    if key:
        matches = [node for node in graph.nodes.values() if _match_key(node.id) == key]
        if matches:
            best = next((n for n in matches if n.type == "ACTION"), matches[0])
            return Localization(best.id, "normalized")
    return Localization(None, "none")


# ── Local neighbourhood N_h(u) ───────────────────────
def local_transitions(
    graph: ProceduralGraph, node_id: str, hops: int = 2
) -> list[tuple[int, ProcEdge]]:
    """``(hop, edge)`` pairs of N_h(node): directed OUTGOING transitions, BFS by hop.

    Hop 1 is the node's own outgoing edges; hop k expands the targets first
    reached at hop k-1. Each node is expanded at most once, so cycles (under
    ``cycle_policy == "allow"``) cannot loop. ``hops < 0`` is treated as 0.
    Raises ``KeyError`` for an unknown node.
    """
    if node_id not in graph.nodes:
        raise KeyError(node_id)
    out = _adjacency(graph)
    seen = {node_id}
    frontier = [node_id]
    result: list[tuple[int, ProcEdge]] = []
    for hop in range(1, max(0, int(hops)) + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for edge in out.get(node, ()):
                result.append((hop, edge))
                if edge.target not in seen:
                    seen.add(edge.target)
                    next_frontier.append(edge.target)
        frontier = next_frontier
        if not frontier:
            break
    return result


def neighborhood(graph: ProceduralGraph, node_id: str, hops: int = 2) -> ProceduralGraph:
    """G_t = N_h(u_t) as a (sub)graph: the node plus its outgoing h-hop horizon.

    Edges are ordered by hop. The hop number of each edge is available from
    :func:`local_transitions`, which this is built on and which
    :func:`serialize_local` uses to group the output.
    """
    transitions = local_transitions(graph, node_id, hops)
    keep = {node_id}
    for _, edge in transitions:
        keep.update((edge.source, edge.target))
    return ProceduralGraph(
        name=graph.name,
        nodes={nid: node for nid, node in graph.nodes.items() if nid in keep},
        edges=[edge for _, edge in transitions],
        description=graph.description,
        cycle_policy=graph.cycle_policy,
        tools=graph.tools,
    )


# ── Serialization (what the solver / guidance LLM reads) ──
def _flat(text: str) -> str:
    """Collapse whitespace so multi-line attributes cannot break the layout."""
    return " ".join(text.split())


def _transition_lines(edge: ProcEdge) -> list[str]:
    # Synapse prints the relation label; the paper's serializer does not
    # (Appendix B.5, "The two stored relation labels ... are not printed").
    # It costs a few tokens and tells the reader whether a transition is a
    # sequence step (LEADS_TO), a fallback (TRIGGERS) or a data dependency.
    condition = _flat(edge.condition) if edge.condition else "unconditional"
    lines = [
        f"- Transition: [{edge.source}]→[{edge.target}] "
        f"(Relation: {edge.relation}; Condition: {condition})"
    ]
    if edge.guidance.strip():
        lines.append(f"  * Guidance: {_flat(edge.guidance)}")
    if edge.pitfalls.strip():
        lines.append(f"  * Pitfalls to Avoid: {_flat(edge.pitfalls)}")
    return lines


def _hop_header(hop: int) -> str:
    if hop == 1:
        return "Immediate Transition Options (Hop 1):"
    if hop == 2:
        return "Subsequent Horizon (Hop 2):"
    return f"Further Horizon (Hop {hop}):"


def serialize_local(graph: ProceduralGraph, node_id: str, hops: int = 2) -> str:
    """The paper's local serializer format (Appendix B.5), relation label added.

    ::

        Active Cognitive Node: [X] (Type: T)
        Description: ...
        Immediate Transition Options (Hop 1):
        - Transition: [X]→[B] (Relation: R; Condition: C)
          * Guidance: ...
          * Pitfalls to Avoid: ...
        Subsequent Horizon (Hop 2):
        - Transition: [B]→[C] (...)

    Deterministic: the same graph, node and hops always yield the same string
    (the guidance cache keys on it). Empty sections beyond hop 1 are omitted.
    """
    node = graph.nodes[node_id]
    lines = [f"Active Cognitive Node: [{node.id}] (Type: {node.type})"]
    if node.description.strip():
        lines.append(f"Description: {_flat(node.description)}")
    by_hop: dict[int, list[ProcEdge]] = {}
    for hop, edge in local_transitions(graph, node_id, hops):
        by_hop.setdefault(hop, []).append(edge)
    max_hops = max(0, int(hops))
    if max_hops >= 1:
        lines.append(_hop_header(1))
        if not by_hop.get(1):
            lines.append("- none: this is a terminal node; the procedure ends here")
        for edge in by_hop.get(1, []):
            lines.extend(_transition_lines(edge))
    for hop in range(2, max_hops + 1):
        if by_hop.get(hop):
            lines.append(_hop_header(hop))
            for edge in by_hop[hop]:
                lines.extend(_transition_lines(edge))
    return "\n".join(lines)


def serialize_full(graph: ProceduralGraph) -> str:
    """Every node, then every transition in the local serializer's edge format.

    This is the paper's full-graph guidance context (its "no match" fallback and
    its Table 3 ablation).
    """
    lines = [f"Procedural Graph: [{graph.name}]"]
    if graph.description.strip():
        lines.append(f"Description: {_flat(graph.description)}")
    lines.append("Nodes:")
    for node in graph.nodes.values():
        detail = f": {_flat(node.description)}" if node.description.strip() else ""
        lines.append(f"- [{node.id}] (Type: {node.type}){detail}")
    lines.append("Transitions:")
    if not graph.edges:
        lines.append("- none")
    for edge in graph.edges:
        lines.extend(_transition_lines(edge))
    return "\n".join(lines)


# ── Diff (version notes, UI) ─────────────────────────
def graph_diff(old: ProceduralGraph, new: ProceduralGraph) -> dict:
    """Structural difference ``old → new``.

    Nodes are identified by id, edges by their triplet (source, relation,
    target). ``changed_edges`` are same-triplet edges whose condition, guidance
    or pitfalls differ; ``changed_nodes`` (an addition to the five keys the
    evolution report needs) are same-id nodes whose type or description differ.
    Output order follows ``new`` for additions/changes and ``old`` for removals.
    """
    old_edges = {edge.key: edge for edge in old.edges}
    new_edges = {edge.key: edge for edge in new.edges}
    changed_edges = []
    for key, edge in new_edges.items():
        before = old_edges.get(key)
        if before is not None and before != edge:
            changed_edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "before": _edge_attributes(before),
                    "after": _edge_attributes(edge),
                }
            )
    return {
        "added_nodes": [n.to_dict() for nid, n in new.nodes.items() if nid not in old.nodes],
        "removed_nodes": [n.to_dict() for nid, n in old.nodes.items() if nid not in new.nodes],
        "changed_nodes": [
            {"id": nid, "before": old.nodes[nid].to_dict(), "after": n.to_dict()}
            for nid, n in new.nodes.items()
            if nid in old.nodes and old.nodes[nid] != n
        ],
        "added_edges": [e.to_dict() for key, e in new_edges.items() if key not in old_edges],
        "removed_edges": [e.to_dict() for key, e in old_edges.items() if key not in new_edges],
        "changed_edges": changed_edges,
    }


def _edge_attributes(edge: ProcEdge) -> dict:
    return {"condition": edge.condition, "guidance": edge.guidance, "pitfalls": edge.pitfalls}


def summarize_diff(diff: Mapping[str, Any]) -> str:
    """One line for a version note: ``"+2 nodes, -1 edge, 3 edges changed"``."""

    def count(key: str) -> int:
        return len(diff.get(key) or [])

    def plural(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    parts = []
    if count("added_nodes"):
        parts.append(f"+{plural(count('added_nodes'), 'node')}")
    if count("removed_nodes"):
        parts.append(f"-{plural(count('removed_nodes'), 'node')}")
    if count("changed_nodes"):
        parts.append(f"{plural(count('changed_nodes'), 'node')} changed")
    if count("added_edges"):
        parts.append(f"+{plural(count('added_edges'), 'edge')}")
    if count("removed_edges"):
        parts.append(f"-{plural(count('removed_edges'), 'edge')}")
    if count("changed_edges"):
        parts.append(f"{plural(count('changed_edges'), 'edge')} changed")
    return ", ".join(parts) if parts else "no structural change"
