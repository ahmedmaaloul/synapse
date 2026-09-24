# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedural Store (procedural memory in Neo4j)

Procedural Graphs (Lu, Chen, Wu, Arık, arXiv:2609.09153; data structure in
``procedural_graph``) live in the SAME Neo4j as the knowledge graph, under
their own labels. Semantic memory (entities) and procedural memory (how to
navigate them) sit side by side, can be inspected with the same tools, and are
backed up together.

Model::

    (:ProcedureGraph {name, version, score, description, cycle_policy, tools,
                      node_count, edge_count, created_at, updated_at})
    (:Procedure {uid, graph, id, type, description, ord})        uid = "<graph>::<id>"
    (:Procedure)-[:TRANSITION {relation, condition, guidance, pitfalls, ord}]->(:Procedure)
    (:ProcedureVersion {uid, graph, version, score, accepted, created_at, note,
                        graph_json, edits_json, diff_json})    uid = "<graph>::v<version>"
    (:ProcedureRejection {graph, round, reason, score, diagnostics_json,
                          edits_json, created_at})
    (:ProcedureTrajectory {graph, version, query, steps_json, score, source,
                           created_at})

Why it is shaped this way:

  • **Live nodes + immutable snapshots.** The live ``Procedure`` nodes are the
    current graph, queryable and visualisable; every save also writes a
    ``ProcedureVersion`` holding the full graph JSON, the edits that produced it
    and the diff from its predecessor. Rollback re-saves a snapshot as a NEW
    version, so history is append-only and a rollback is itself undoable.
  • **One transaction per save** (``execute_write_batch``). Replacing the live
    graph, bumping the version counter and writing the snapshot either all
    commit or none do; a crash cannot leave a half-replaced graph or a version
    counter out of step with its snapshot.
  • **The version number is computed inside that transaction** (``MERGE`` on
    the ``ProcedureGraph`` + ``SET version = version + 1``), and
    ``ProcedureVersion.uid`` is unique, so two racing saves cannot both write
    "v3": the loser's transaction fails.
  • **Synthetic ``uid``s** because Neo4j 5 Community has single-property
    uniqueness constraints only (no node keys).
  • **``ord`` properties** preserve authoring order, so a graph serializes to
    the same bytes after a save/load round-trip (cache keys and diffs rely on
    it).
  • **Timestamps are ISO-8601 strings** written from Python: sortable and
    JSON-serializable as-is (a driver ``DateTime`` is not).
  • Every version is ``accepted = true``: rejected candidates never become
    versions; they are ``ProcedureRejection`` rows (the paper's rejection
    memory, H_rejected, made durable for audit).

``DELETE /api/graph`` keeps all of this (``graph_schema.PROCEDURAL_LABELS``),
and ``/api/graph-data`` never renders it (it is scoped to ``:Entity``).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.neo4j_driver import execute_query, execute_write_batch
from app.services.procedural_graph import (
    GRAPH_NAME_PATTERN,
    InvalidProceduralGraph,
    ProceduralGraph,
    graph_diff,
    validate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InvalidProceduralGraph",
    "PRIORS_DIR",
    "ProceduralGraphNotFound",
    "delete_graph",
    "ensure_default_graphs",
    "get_meta",
    "get_version",
    "list_graphs",
    "list_priors",
    "list_rejections",
    "list_trajectories",
    "list_versions",
    "load_graph",
    "load_graph_with_meta",
    "load_prior",
    "record_rejection",
    "record_trajectory",
    "rollback",
    "save_graph",
]

#: Hand-designed expert priors (G0), one ``<name>.json`` per graph. A path next
#: to the package, not importlib.resources: the backend is not pip-installed —
#: the image copies ``backend/`` wholesale, so ``app/data`` ships with the code.
PRIORS_DIR = Path(__file__).resolve().parent.parent / "data" / "procedural"


class ProceduralGraphNotFound(LookupError):
    """No stored graph (or no such version of it) under that name."""

    def __init__(self, name: str, version: int | None = None) -> None:
        self.name = name
        self.version = version
        if version is None:
            message = f"Procedural graph '{name}' not found"
        else:
            message = f"Procedural graph '{name}' has no version {version}"
        super().__init__(message)


# ── Cypher: reads ────────────────────────────────────
LIST_GRAPHS_QUERY = """
MATCH (g:ProcedureGraph)
RETURN g.name AS name, g.version AS version, g.score AS score,
       g.node_count AS nodes, g.edge_count AS edges,
       g.updated_at AS updated_at, g.description AS description
ORDER BY name ASC
"""

GET_META_QUERY = """
MATCH (g:ProcedureGraph {name: $name})
RETURN g.name AS name, g.version AS version, g.score AS score,
       g.description AS description, g.cycle_policy AS cycle_policy,
       g.tools AS tools, g.node_count AS nodes, g.edge_count AS edges,
       g.created_at AS created_at, g.updated_at AS updated_at
"""

# One round-trip for the whole live graph. Nodes and transitions are collected
# in ``ord`` order so the loaded graph serializes exactly like the saved one.
# ``collect`` drops nulls, which is what makes a node-less or edge-less graph
# come back as empty lists rather than ``[null]``.
LOAD_GRAPH_QUERY = """
MATCH (g:ProcedureGraph {name: $name})
OPTIONAL MATCH (n:Procedure {graph: $name})
WITH g, n
ORDER BY n.ord ASC, n.id ASC
WITH g, collect(n {.id, .type, .description}) AS nodes
OPTIONAL MATCH (a:Procedure {graph: $name})-[t:TRANSITION]->(b:Procedure {graph: $name})
WITH g, nodes, a, t, b
ORDER BY t.ord ASC
RETURN g {.name, .version, .score, .description, .cycle_policy, .tools,
          .created_at, .updated_at, nodes: g.node_count, edges: g.edge_count} AS meta,
       nodes,
       collect(CASE WHEN t IS NULL THEN NULL ELSE {
           source: a.id, target: b.id, relation: t.relation,
           condition: t.condition, guidance: t.guidance, pitfalls: t.pitfalls
       } END) AS edges
"""

LIST_VERSIONS_QUERY = """
MATCH (v:ProcedureVersion {graph: $name})
RETURN v.version AS version, v.score AS score, v.accepted AS accepted,
       v.created_at AS created_at, v.note AS note, v.diff_json AS diff_json
ORDER BY v.version DESC
LIMIT $limit
"""

GET_VERSION_QUERY = """
MATCH (v:ProcedureVersion {uid: $uid})
RETURN v.version AS version, v.score AS score, v.note AS note,
       v.created_at AS created_at, v.graph_json AS graph_json
"""

LIST_REJECTIONS_QUERY = """
MATCH (r:ProcedureRejection {graph: $name})
RETURN r.round AS round, r.reason AS reason, r.score AS score,
       r.diagnostics_json AS diagnostics_json, r.edits_json AS edits_json,
       r.created_at AS created_at
ORDER BY r.created_at DESC
LIMIT $limit
"""

LIST_TRAJECTORIES_QUERY = """
MATCH (t:ProcedureTrajectory {graph: $name})
RETURN t.version AS version, t.query AS query, t.steps_json AS steps_json,
       t.score AS score, t.source AS source, t.created_at AS created_at
ORDER BY t.created_at DESC
LIMIT $limit
"""

# ── Cypher: the save transaction (run in this order, in ONE batch) ──
# 1. Bump the version under the ProcedureGraph's write lock. ``coalesce`` makes
#    the first save version 1.
UPSERT_GRAPH_QUERY = """
MERGE (g:ProcedureGraph {name: $name})
ON CREATE SET g.created_at = $now
SET g.version = coalesce(g.version, 0) + 1,
    g.score = $score,
    g.description = $description,
    g.cycle_policy = $cycle_policy,
    g.tools = $tools,
    g.node_count = $node_count,
    g.edge_count = $edge_count,
    g.updated_at = $now
RETURN g.version AS version
"""

# 2. Drop the live graph (its TRANSITIONs go with it).
WIPE_LIVE_QUERY = """
MATCH (n:Procedure {graph: $name})
DETACH DELETE n
"""

# 3. Recreate the nodes...
WRITE_NODES_QUERY = """
UNWIND $nodes AS node
CREATE (:Procedure {uid: node.uid, graph: $name, id: node.id, type: node.type,
                    description: node.description, ord: node.ord})
"""

# 4. ...and the transitions, matched on the uniquely-indexed uid.
WRITE_EDGES_QUERY = """
UNWIND $edges AS edge
MATCH (a:Procedure {uid: edge.source_uid})
MATCH (b:Procedure {uid: edge.target_uid})
CREATE (a)-[:TRANSITION {relation: edge.relation, condition: edge.condition,
                         guidance: edge.guidance, pitfalls: edge.pitfalls,
                         ord: edge.ord}]->(b)
"""

# 5. Snapshot, numbered by the counter statement 1 just bumped.
WRITE_VERSION_QUERY = """
MATCH (g:ProcedureGraph {name: $name})
CREATE (v:ProcedureVersion {
    uid: $name + '::v' + toString(g.version),
    graph: $name, version: g.version, score: $score, accepted: true,
    created_at: $now, note: $note,
    graph_json: $graph_json, edits_json: $edits_json, diff_json: $diff_json
})
RETURN v.version AS version
"""

# ── Cypher: other writes ─────────────────────────────
DELETE_GRAPH_QUERIES: tuple[str, ...] = (
    "MATCH (n:Procedure {graph: $name}) DETACH DELETE n",
    "MATCH (v:ProcedureVersion {graph: $name}) DETACH DELETE v",
    "MATCH (r:ProcedureRejection {graph: $name}) DETACH DELETE r",
    "MATCH (t:ProcedureTrajectory {graph: $name}) DETACH DELETE t",
    """
    OPTIONAL MATCH (g:ProcedureGraph {name: $name})
    DETACH DELETE g
    RETURN count(g) AS deleted
    """,
)

WRITE_REJECTION_QUERY = """
CREATE (:ProcedureRejection {
    graph: $name, round: $round, reason: $reason, score: $score,
    diagnostics_json: $diagnostics_json, edits_json: $edits_json, created_at: $now
})
"""

WRITE_TRAJECTORY_QUERY = """
CREATE (:ProcedureTrajectory {
    graph: $name, version: $version, query: $query, steps_json: $steps_json,
    score: $score, source: $source, created_at: $now
})
"""


# ── Helpers ──────────────────────────────────────────
def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dumps(value: Any) -> str | None:
    """JSON for a ``*_json`` property; ``None`` stays ``None`` (no property)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any = None) -> Any:
    """Parse a ``*_json`` property, tolerating a missing or corrupt value."""
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except ValueError:
        logger.warning("⚠️ Unparseable JSON property in procedural store; ignoring it")
        return default


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _uid(graph: str, node_id: str) -> str:
    return f"{graph}::{node_id}"


def _version_uid(graph: str, version: int) -> str:
    return f"{graph}::v{version}"


def _meta_from_row(row: dict) -> dict:
    return {
        "name": row.get("name"),
        "version": _as_int(row.get("version")),
        "score": _as_float(row.get("score")),
        "description": row.get("description") or "",
        "cycle_policy": row.get("cycle_policy") or "forbid",
        "tools": list(row.get("tools") or []),
        "nodes": _as_int(row.get("nodes")),
        "edges": _as_int(row.get("edges")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


# ── Reads ────────────────────────────────────────────
async def list_graphs() -> list[dict]:
    """Every stored graph: ``{name, version, score, nodes, edges, updated_at, description}``."""
    rows = await execute_query(LIST_GRAPHS_QUERY)
    return [
        {
            "name": row.get("name"),
            "version": _as_int(row.get("version")),
            "score": _as_float(row.get("score")),
            "nodes": _as_int(row.get("nodes")),
            "edges": _as_int(row.get("edges")),
            "updated_at": row.get("updated_at"),
            "description": row.get("description") or "",
        }
        for row in rows
    ]


async def get_meta(name: str) -> dict | None:
    """The ``ProcedureGraph`` record (version, score, counts, timestamps) or ``None``."""
    rows = await execute_query(GET_META_QUERY, {"name": name})
    return _meta_from_row(rows[0]) if rows else None


async def load_graph_with_meta(name: str) -> tuple[ProceduralGraph, dict] | None:
    """The live graph and its metadata in ONE read, or ``None`` if absent.

    Callers that need the version too (the guidance cache key, the GET
    endpoint) should use this rather than ``load_graph`` + ``get_meta``: two
    reads could straddle a concurrent save and pair a graph with the wrong
    version number.
    """
    rows = await execute_query(LOAD_GRAPH_QUERY, {"name": name})
    if not rows or not rows[0].get("meta"):
        return None
    row = rows[0]
    meta = _meta_from_row(row["meta"])
    graph = ProceduralGraph.from_dict(
        {
            "name": name,
            "description": meta["description"],
            "cycle_policy": meta["cycle_policy"],
            "tools": meta["tools"],
            "nodes": [n for n in row.get("nodes") or [] if n],
            "edges": [e for e in row.get("edges") or [] if e],
        },
        name=name,
    )
    return graph, meta


async def load_graph(name: str) -> ProceduralGraph | None:
    """The live graph (one read), or ``None`` if no graph has that name."""
    loaded = await load_graph_with_meta(name)
    return loaded[0] if loaded else None


async def list_versions(name: str, limit: int = 200) -> list[dict]:
    """Newest first: ``{version, score, accepted, created_at, note, diff}``."""
    rows = await execute_query(LIST_VERSIONS_QUERY, {"name": name, "limit": int(limit)})
    return [
        {
            "version": _as_int(row.get("version")),
            "score": _as_float(row.get("score")),
            "accepted": bool(row.get("accepted", True)),
            "created_at": row.get("created_at"),
            "note": row.get("note") or "",
            "diff": _loads(row.get("diff_json"), default={}),
        }
        for row in rows
    ]


async def _read_version(name: str, version: int) -> dict:
    rows = await execute_query(GET_VERSION_QUERY, {"uid": _version_uid(name, int(version))})
    if not rows or not rows[0].get("graph_json"):
        raise ProceduralGraphNotFound(name, int(version))
    return rows[0]


async def get_version(name: str, version: int) -> ProceduralGraph:
    """The graph exactly as saved in version ``version``.

    Raises :class:`ProceduralGraphNotFound` when that version does not exist.
    """
    row = await _read_version(name, version)
    return ProceduralGraph.from_dict(json.loads(row["graph_json"]), name=name)


async def list_rejections(name: str, limit: int = 20) -> list[dict]:
    """Newest first: ``{round, reason, score, diagnostics, edits, created_at}``."""
    rows = await execute_query(LIST_REJECTIONS_QUERY, {"name": name, "limit": int(limit)})
    return [
        {
            "round": row.get("round"),
            "reason": row.get("reason") or "",
            "score": _as_float(row.get("score")),
            "diagnostics": _loads(row.get("diagnostics_json"), default=[]),
            "edits": _loads(row.get("edits_json")),
            "created_at": row.get("created_at"),
        }
        for row in rows
    ]


async def list_trajectories(name: str, limit: int = 50) -> list[dict]:
    """Newest first: ``{version, query, steps, score, source, created_at}``."""
    rows = await execute_query(LIST_TRAJECTORIES_QUERY, {"name": name, "limit": int(limit)})
    return [
        {
            "version": row.get("version"),
            "query": row.get("query") or "",
            "steps": _loads(row.get("steps_json"), default=[]),
            "score": _as_float(row.get("score")),
            "source": row.get("source") or "",
            "created_at": row.get("created_at"),
        }
        for row in rows
    ]


# ── Writes ───────────────────────────────────────────
async def save_graph(
    graph: ProceduralGraph,
    *,
    score: float | None = None,
    note: str = "",
    edits: dict | None = None,
    previous: ProceduralGraph | None = None,
) -> int:
    """Validate, replace the live graph and append a version; return its number.

    Versions start at 1 and increase by one per save (computed inside the
    transaction, see the module docstring). ``score`` is the validation score
    that justified this version, or ``None`` when unmeasured (a hand edit, the
    seeded prior); it replaces the graph's current score either way, because a
    score belongs to the graph it was measured on.

    ``previous`` is the graph this one replaces, used only for ``diff_json``.
    The evolution loop has it in hand; when omitted, the stored graph is loaded
    (one extra read), and a first save diffs against the empty graph.

    Raises :class:`InvalidProceduralGraph` before touching the database when
    ``validate`` reports anything.
    """
    diagnostics = validate(graph)
    if diagnostics:
        raise InvalidProceduralGraph(diagnostics)

    name = graph.name
    if previous is None:
        previous = await load_graph(name)
    diff = graph_diff(previous or ProceduralGraph(name=name, nodes={}, edges=[]), graph)
    now = _now()
    score = _as_float(score)

    nodes = [
        {
            "uid": _uid(name, node.id),
            "id": node.id,
            "type": node.type,
            "description": node.description,
            "ord": i,
        }
        for i, node in enumerate(graph.nodes.values())
    ]
    edges = [
        {
            "source_uid": _uid(name, edge.source),
            "target_uid": _uid(name, edge.target),
            "relation": edge.relation,
            "condition": edge.condition,
            "guidance": edge.guidance,
            "pitfalls": edge.pitfalls,
            "ord": i,
        }
        for i, edge in enumerate(graph.edges)
    ]
    statements: list[tuple[str, dict]] = [
        (
            UPSERT_GRAPH_QUERY,
            {
                "name": name,
                "now": now,
                "score": score,
                "description": graph.description,
                "cycle_policy": graph.cycle_policy,
                "tools": list(graph.tools),
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        ),
        (WIPE_LIVE_QUERY, {"name": name}),
        (WRITE_NODES_QUERY, {"name": name, "nodes": nodes}),
        (WRITE_EDGES_QUERY, {"name": name, "edges": edges}),
        (
            WRITE_VERSION_QUERY,
            {
                "name": name,
                "now": now,
                "score": score,
                "note": note,
                "graph_json": _dumps(graph.to_dict()),
                "edits_json": _dumps(edits),
                "diff_json": _dumps(diff),
            },
        ),
    ]
    results = await execute_write_batch(statements)
    for rows in (results[-1] if results else [], results[0] if results else []):
        if rows and rows[0].get("version") is not None:
            version = int(rows[0]["version"])
            logger.info("🧭 Saved procedural graph %s v%s (%s)", name, version, note or "no note")
            return version
    raise RuntimeError(f"saving procedural graph '{name}' returned no version number")


async def rollback(name: str, version: int) -> int:
    """Re-save version ``version`` as a NEW version (note ``"rollback to vN"``).

    History stays append-only, so the rollback itself can be rolled back. The
    old version's score is carried over: it was measured on this exact graph.
    Raises :class:`ProceduralGraphNotFound` if that version does not exist.
    """
    row = await _read_version(name, version)
    graph = ProceduralGraph.from_dict(json.loads(row["graph_json"]), name=name)
    return await save_graph(
        graph, score=_as_float(row.get("score")), note=f"rollback to v{int(version)}"
    )


async def delete_graph(name: str) -> bool:
    """Delete a graph and everything recorded for it (all five labels), atomically.

    Returns whether a ``ProcedureGraph`` by that name existed.
    """
    results = await execute_write_batch([(query, {"name": name}) for query in DELETE_GRAPH_QUERIES])
    last = results[-1] if results else []
    return bool(last and _as_int(last[0].get("deleted")))


async def record_rejection(
    name: str,
    *,
    round: int,
    reason: str,
    score: float | None = None,
    diagnostics: list[str] | None = None,
    edits: Any = None,
) -> None:
    """Persist one rejected candidate (the paper's rejection memory, for audit).

    ``reason`` is e.g. ``"structural"`` (``diagnostics`` set, no validation
    rollout) or ``"score"`` (``score`` below the retained graph's). Deliberately
    does not require the graph to exist: a ``scratch`` evolution run on a new
    name has nothing stored until its first accepted candidate.
    """
    await execute_write_batch(
        [
            (
                WRITE_REJECTION_QUERY,
                {
                    "name": name,
                    "round": int(round),
                    "reason": reason,
                    "score": _as_float(score),
                    "diagnostics_json": _dumps(list(diagnostics or [])),
                    "edits_json": _dumps(edits),
                    "now": _now(),
                },
            )
        ]
    )


async def record_trajectory(
    name: str,
    *,
    version: int | None,
    query: str,
    steps: list[dict],
    score: float | None,
    source: str,
) -> None:
    """Persist one agent trajectory (``steps`` as JSON) against a graph version.

    ``score`` is ``None`` when the run was not graded (e.g. ``/api/agent/ask``
    with ``record=true``). Like rejections, the graph need not exist yet.
    """
    await execute_write_batch(
        [
            (
                WRITE_TRAJECTORY_QUERY,
                {
                    "name": name,
                    "version": None if version is None else int(version),
                    "query": query,
                    "steps_json": _dumps(list(steps)),
                    "score": _as_float(score),
                    "source": source,
                    "now": _now(),
                },
            )
        ]
    )


# ── Expert priors ────────────────────────────────────
def list_priors() -> list[str]:
    """Names of the bundled expert priors (``app/data/procedural/*.json``)."""
    if not PRIORS_DIR.is_dir():
        return []
    return sorted(path.stem for path in PRIORS_DIR.glob("*.json"))


def load_prior(name: str) -> ProceduralGraph:
    """Load a bundled prior by name.

    The name is checked against the graph-name slug *before* building a path,
    so ``"../../etc/passwd"`` can never be read. Raises
    :class:`ProceduralGraphNotFound` for an unknown prior.
    """
    if not GRAPH_NAME_PATTERN.match(name or ""):
        raise ProceduralGraphNotFound(name)
    path = PRIORS_DIR / f"{name}.json"
    if not path.is_file():
        raise ProceduralGraphNotFound(name)
    return ProceduralGraph.from_dict(json.loads(path.read_text(encoding="utf-8")), name=name)


async def ensure_default_graphs() -> list[str]:
    """Seed every bundled prior that is not stored yet; return the names seeded.

    Called once from the app lifespan (after ``ensure_schema``). Best-effort
    like the schema bootstrap: a cold or unreachable database is logged and
    skipped, never fatal, and an existing graph (evolved or hand-edited) is
    never overwritten. Two workers booting at the same instant could both seed
    (v1 and v2 of the same prior); harmless, and the version history shows it.
    """
    seeded: list[str] = []
    for name in list_priors():
        try:
            if await get_meta(name) is not None:
                continue
            await save_graph(load_prior(name), note="seeded from the expert prior")
            seeded.append(name)
            logger.info("🧭 Seeded procedural graph %s from its expert prior", name)
        except Exception as e:  # noqa: BLE001 - startup must not fail on this
            logger.warning("⚠️ Could not seed procedural graph %s: %s", name, e)
    return seeded
