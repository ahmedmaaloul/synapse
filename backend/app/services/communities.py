# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Community Detection & Summarization

The half of GraphRAG that plain vector RAG fundamentally cannot do.

Vector RAG answers *local* questions ("what did Ada work on?") by retrieving the
few chunks nearest the query. It cannot answer *global* ones ("what are the main
themes across this corpus?") because no single chunk contains the answer — it is
a property of the graph as a whole.

This module builds that capability:

  1. LOAD      — pull the entity graph out of Neo4j in one round-trip and build an
                 undirected, weighted ``networkx`` graph (nodes = entity names).
  2. CLUSTER   — Louvain community detection with a fixed seed, so results are
                 reproducible run to run. Modularity is reported as a real
                 quality signal for the partition.
  3. SUMMARIZE — each community's members + internal relationships are handed to
                 the LLM, which returns a short title and a 2-3 sentence summary.
                 Calls run concurrently under a semaphore.
  4. PERSIST   — every community becomes a ``(:Community)`` node carrying an
                 embedding of "title + summary", linked to its members by
                 ``(:Entity)-[:IN_COMMUNITY]->(:Community)``, so communities are
                 themselves vector-searchable ("global search").

Neo4j here runs *without* the GDS plugin, so clustering happens in Python — never
via ``gds.*`` procedures. Every step degrades gracefully: an empty graph returns
zeros, a failed LLM call falls back to a derived title, and missing vector support
falls back to plain writes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, get_settings
from app.neo4j_driver import execute_query
from app.services.graph_schema import COMMUNITY_VECTOR_INDEX
from app.services.llm_provider import get_chat_llm, get_embeddings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], Awaitable[None]]

#: Louvain is stochastic; a fixed seed makes every run of the same graph produce
#: the same partition, which is what makes summaries stable and cacheable.
LOUVAIN_SEED = 42

# ── Cypher ───────────────────────────────────────────
#: One round-trip: every entity-to-entity edge plus the endpoint metadata the
#: summarizer needs. Entities with no edges cannot form a community anyway (a
#: singleton is dropped by ``community_min_size``), so this single query is
#: sufficient — and far cheaper than fetching all nodes separately.
LOAD_GRAPH_QUERY = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE a.name IS NOT NULL AND b.name IS NOT NULL AND a.name <> b.name
RETURN a.name AS source,
       b.name AS target,
       coalesce(r.type, type(r)) AS rel_type,
       a.type AS source_type,
       a.description AS source_description,
       b.type AS target_type,
       b.description AS target_description
"""

WIPE_COMMUNITIES_QUERY = """
MATCH (c:Community)
DETACH DELETE c
"""

WRITE_COMMUNITY_QUERY = """
CREATE (c:Community {id: $id, title: $title, summary: $summary, size: $size})
WITH c
CALL {
    WITH c
    WITH c WHERE $embedding IS NOT NULL
    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
}
WITH c
UNWIND $members AS member
MATCH (e:Entity {name: member})
MERGE (e)-[:IN_COMMUNITY]->(c)
"""

WRITE_COMMUNITY_FALLBACK_QUERY = """
CREATE (c:Community {id: $id, title: $title, summary: $summary, size: $size})
WITH c
UNWIND $members AS member
MATCH (e:Entity {name: member})
MERGE (e)-[:IN_COMMUNITY]->(c)
"""

LIST_COMMUNITIES_QUERY = """
MATCH (c:Community)
OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
WITH c, collect(e.name) AS members
RETURN c.id AS id, c.title AS title, c.summary AS summary,
       c.size AS size, members AS members
ORDER BY size DESC, title ASC
LIMIT $limit
"""

SEARCH_COMMUNITIES_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(node)
WITH node, score, collect(e.name) AS members
RETURN node.id AS id, node.title AS title, node.summary AS summary,
       node.size AS size, members AS members, score AS score
ORDER BY score DESC
"""

# ── Summarization prompt ─────────────────────────────
SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledge-graph analyst. You are given one cluster of \
closely connected entities extracted from a document corpus, together with the \
relationships between them. Describe what this cluster is *about*.

Return ONLY valid JSON with this exact structure (no markdown fences, no prose):
{{
  "title": "Short Topic Title",
  "summary": "Two to three sentences describing the theme."
}}

RULES:
1. "title" must be at most 6 words, specific and human-readable — a theme, not a list.
2. "summary" must be 2-3 sentences describing what connects these entities and why
   the cluster matters. Name the most important entities explicitly.
3. Use ONLY the entities and relationships provided. Never invent facts.
4. No markdown, no code fences, no keys other than "title" and "summary".""",
        ),
        (
            "human",
            """Entities in this cluster:
{members}

Relationships between them:
{relationships}

Summarize this cluster as JSON.""",
        ),
    ]
)


# ── Graph loading (pure helpers) ─────────────────────
def _clean(value: object) -> str:
    """Coerce a Neo4j value to a trimmed string ('' for None)."""
    return "" if value is None else str(value).strip()


def build_graph_from_rows(
    rows: list[dict],
) -> tuple[nx.Graph, dict[str, dict], list[dict]]:
    """Build an undirected weighted graph from raw edge rows.

    Pure and synchronous so it can be tested without a database.

    Returns:
        ``(graph, entity_meta, edges)`` where ``entity_meta`` maps entity name ->
        ``{"type", "description"}`` and ``edges`` is the deduplicated list of
        ``{"source", "target", "type"}`` triples used to describe a community to
        the LLM. Parallel edges between the same pair increase the edge weight,
        so strongly connected entities are more likely to cluster together.
    """
    graph = nx.Graph()
    meta: dict[str, dict] = {}
    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def sort_key(row: dict) -> tuple[str, str, str]:
        return (
            _clean(row.get("source")),
            _clean(row.get("target")),
            _clean(row.get("rel_type")),
        )

    # Sorting makes graph construction order-independent, which — together with
    # LOUVAIN_SEED — is what actually makes the partition reproducible.
    for row in sorted(rows, key=sort_key):
        source = _clean(row.get("source"))
        target = _clean(row.get("target"))
        if not source or not target or source == target:
            continue

        meta.setdefault(
            source,
            {
                "type": _clean(row.get("source_type")),
                "description": _clean(row.get("source_description")),
            },
        )
        meta.setdefault(
            target,
            {
                "type": _clean(row.get("target_type")),
                "description": _clean(row.get("target_description")),
            },
        )

        rel_type = _clean(row.get("rel_type")) or "RELATED_TO"
        key = (source, target, rel_type)
        if key not in seen:
            seen.add(key)
            edges.append({"source": source, "target": target, "type": rel_type})

        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1.0
        else:
            graph.add_edge(source, target, weight=1.0)

    return graph, meta, edges


class GraphLoadError(RuntimeError):
    """The entity graph could not be read from Neo4j.

    Deliberately distinct from "the graph is empty". The two used to be conflated
    — a failed read returned ``[]`` — and that is dangerous now that the
    zero-community path *wipes* the previous run's ``:Community`` nodes: one
    transient Neo4j hiccup would silently destroy every community in the
    database. An empty result means "there is nothing to cluster" and licenses
    the wipe; this exception means "we know nothing" and must abort instead.
    """


async def _fetch_graph_rows() -> list[dict]:
    """Fetch every entity-to-entity edge from Neo4j (one round-trip).

    Returns ``[]`` only when the graph genuinely has no entity-to-entity edges,
    which is a valid state for this pipeline. A read that *fails* raises
    :class:`GraphLoadError` so the caller can tell the two apart.
    """
    try:
        rows = await execute_query(LOAD_GRAPH_QUERY)
    except Exception as e:
        raise GraphLoadError(f"could not load graph for community detection: {e}") from e
    return rows or []


# ── Detection (pure) ─────────────────────────────────
def detect_communities(
    graph: nx.Graph, settings: Settings | None = None
) -> tuple[list[list[str]], float]:
    """Partition ``graph`` with Louvain and return ``(communities, modularity)``.

    Pure and synchronous. Communities smaller than ``community_min_size`` are
    dropped as noise, but modularity is computed over the *full* partition —
    modularity is only defined for a complete partition of the graph, and the
    full-partition value is the honest quality signal for the clustering.

    Members are ordered most-central-first (degree), and communities largest-first,
    so downstream truncation keeps the most informative entities.
    """
    settings = settings or get_settings()

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return [], 0.0

    partition = nx.algorithms.community.louvain_communities(
        graph,
        resolution=settings.community_resolution,
        seed=LOUVAIN_SEED,
        weight="weight",
    )

    try:
        score = nx.algorithms.community.modularity(
            graph, partition, resolution=settings.community_resolution, weight="weight"
        )
    except Exception as e:  # noqa: BLE001 - never fail the pipeline on a metric
        logger.warning("⚠️ Modularity computation failed: %s", e)
        score = 0.0

    degrees = dict(graph.degree())
    kept = [
        sorted(group, key=lambda n: (-degrees.get(n, 0), n))
        for group in partition
        if len(group) >= settings.community_min_size
    ]
    kept.sort(key=lambda members: (-len(members), members[0] if members else ""))

    return kept, round(float(score), 4)


def _build_and_detect(
    rows: list[dict], settings: Settings
) -> tuple[nx.Graph, dict[str, dict], list[dict], list[list[str]], float]:
    """Build the networkx graph and partition it — the whole CPU-bound stretch.

    Both halves are pure and synchronous, and both scale with corpus size, so they
    are kept together in one function: ``detect_and_summarize`` can then push the
    entire stretch onto a worker thread in a single hop instead of two. Running on
    a worker thread changes nothing about the result — the work is single-threaded
    and seeded (``LOUVAIN_SEED``), so the partition stays byte-identical.
    """
    graph, meta, edges = build_graph_from_rows(rows)
    found, modularity = detect_communities(graph, settings)
    return graph, meta, edges, found, modularity


def community_id(members: list[str]) -> str:
    """Stable, content-derived id — unchanged members means an unchanged id."""
    digest = hashlib.sha1("|".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"com-{digest[:12]}"


# ── LLM summarization ────────────────────────────────
def _parse_summary_json(text: str) -> dict:
    """Tolerantly parse ``{"title", "summary"}`` out of an LLM response.

    Mirrors the strategy of ``graph_builder._parse_llm_json`` — strip fences, try
    a direct parse, then fall back to the outermost ``{...}`` block — but that
    helper hard-codes the entity/relationship shape, so it cannot be reused here.
    Always returns the expected keys so a garbage response can never crash a run.
    """
    empty = {"title": "", "summary": ""}
    if not text:
        return empty

    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning("Failed to parse community summary JSON: %s", text[:200])
            return empty
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse community summary JSON (brace fallback): %s", text[:200]
            )
            return empty

    if not isinstance(parsed, dict):
        return empty
    return {
        "title": _clean(parsed.get("title")),
        "summary": _clean(parsed.get("summary")),
    }


def _format_members(members: list[str], meta: dict[str, dict]) -> str:
    lines = []
    for name in members:
        info = meta.get(name, {})
        entity_type = info.get("type") or "UNKNOWN"
        line = f"- {name} ({entity_type})"
        description = info.get("description")
        if description:
            line += f": {description}"
        lines.append(line)
    return "\n".join(lines)


def _format_relationships(members: list[str], edges: list[dict], limit: int = 60) -> str:
    member_set = set(members)
    lines = [
        f"- {e['source']} --[{e['type']}]--> {e['target']}"
        for e in edges
        if e["source"] in member_set and e["target"] in member_set
    ]
    if not lines:
        return "(no relationships recorded between these entities)"
    return "\n".join(lines[:limit])


def _fallback_title(members: list[str]) -> str:
    """Derive a readable title from the most central members (LLM unavailable)."""
    return " / ".join(members[:3]) if members else "Untitled Community"


async def _summarize_community(
    chain,
    members: list[str],
    meta: dict[str, dict],
    edges: list[dict],
    *,
    settings: Settings,
) -> dict:
    """Ask the LLM for a title + summary for one community. Never raises."""
    shown = members[: settings.community_max_members_in_summary]
    try:
        response = await asyncio.wait_for(
            chain.ainvoke(
                {
                    "members": _format_members(shown, meta),
                    "relationships": _format_relationships(shown, edges),
                }
            ),
            timeout=settings.extraction_timeout,
        )
        parsed = _parse_summary_json(getattr(response, "content", "") or "")
    except TimeoutError:
        logger.warning("   Community summary TIMEOUT (%s members)", len(members))
        parsed = {"title": "", "summary": ""}
    except Exception as e:  # noqa: BLE001 - one bad community must not fail the run
        logger.warning("   Community summary failed: %s", e)
        parsed = {"title": "", "summary": ""}

    title = parsed["title"] or _fallback_title(shown)
    summary = parsed["summary"] or (
        f"A cluster of {len(members)} closely related entities including "
        f"{', '.join(shown[:5])}."
    )
    return {
        "id": community_id(members),
        "title": title,
        "summary": summary,
        "size": len(members),
        "members": members,
        "llm_ok": bool(parsed["title"] or parsed["summary"]),
    }


async def _embed_communities(communities: list[dict]) -> list[list[float] | None]:
    """Embed "title + summary" per community so communities are searchable.

    Best-effort: on any failure every embedding is ``None`` and the communities
    are still persisted (they remain reachable via the full-text index).
    """
    if not communities:
        return []
    texts = [f"{c['title']} — {c['summary']}".strip(" —") for c in communities]
    try:
        embeddings = get_embeddings()
        # LangChain embeddings are sync; keep them off the event loop.
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ Community embedding skipped (%s); no vector search", e)
        return [None] * len(communities)
    if len(vectors) != len(communities):
        logger.warning("⚠️ Embedding count mismatch; skipping community vectors")
        return [None] * len(communities)
    return [list(v) for v in vectors]


# ── Persistence ──────────────────────────────────────
async def _wipe_communities() -> None:
    """Delete every ``:Community`` node from the previous run.

    MUST run on *every* rebuild, including one that produces no communities at
    all — otherwise stale themes from an earlier run survive and the UI keeps
    showing clusters the graph no longer contains. Best-effort: a failure here is
    logged, never raised, so it cannot abort a rebuild.
    """
    try:
        await execute_query(WIPE_COMMUNITIES_QUERY)
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ Could not clear previous communities: %s", e)


async def _persist_communities(
    communities: list[dict], vectors: list[list[float] | None]
) -> int:
    """Wipe prior communities and write the new ones. Idempotent by construction.

    The wipe happens here — as late as possible — so the previous run's
    communities stay queryable while the (slow) LLM summarization is still in
    flight. The zero-community path in ``detect_and_summarize`` never reaches this
    function and so wipes explicitly instead.
    """
    await _wipe_communities()

    written = 0
    for community, vector in zip(communities, vectors, strict=False):
        params = {
            "id": community["id"],
            "title": community["title"],
            "summary": community["summary"],
            "size": community["size"],
            "members": community["members"],
            "embedding": vector,
        }
        try:
            await execute_query(WRITE_COMMUNITY_QUERY, params)
            written += 1
        except Exception:  # noqa: BLE001 - older Neo4j lacks vector procedures
            try:
                await execute_query(WRITE_COMMUNITY_FALLBACK_QUERY, params)
                written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "⚠️ Community write failed for %s: %s", community["id"], e
                )
    return written


# ── Public API ───────────────────────────────────────
async def detect_and_summarize(on_progress: ProgressCallback | None = None) -> dict:
    """Detect communities, summarize them with the LLM, and persist them.

    Returns ``{"communities": int, "summarized": int, "modularity": float}``.
    Safe on an empty or edgeless graph — it returns zeros rather than raising, and
    still clears the previous run's communities so no stale theme survives.

    A graph that cannot be *read* returns the same zeros but leaves the database
    completely untouched: see :class:`GraphLoadError`.
    """
    settings = get_settings()

    async def report(event: dict) -> None:
        if on_progress is not None:
            await on_progress(event)

    await report({"type": "progress", "stage": "loading_graph"})
    try:
        rows = await _fetch_graph_rows()
    except GraphLoadError as e:
        # We have not learned that the graph is empty — only that we could not
        # look. Falling through would hit the zero-community wipe below and let
        # one failed query delete every community the corpus has. Abort clean:
        # the existing communities stay queryable and the next run rebuilds them.
        logger.warning("⚠️ Community detection aborted (%s); communities left untouched", e)
        await report({"type": "progress", "stage": "summarizing", "processed": 0, "total": 0})
        return {"communities": 0, "summarized": 0, "modularity": 0.0}

    # Graph construction and Louvain are CPU-bound and can run for seconds on a
    # large corpus. Done inline they would stall the FastAPI event loop for that
    # whole time — no other request, and no progress event, gets through. One
    # ``to_thread`` hop keeps the loop responsive; the result is unchanged.
    graph, meta, edges, communities, modularity = await asyncio.to_thread(
        _build_and_detect, rows, settings
    )
    logger.info(
        "🧩 Louvain: %s communities (min size %s) over %s nodes / %s edges, modularity %.4f",
        len(communities),
        settings.community_min_size,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        modularity,
    )

    if not communities:
        # Nothing qualifies this run — but the previous run's :Community nodes are
        # still in the database, and this path never reaches _persist_communities.
        # Without this wipe, clearing the graph (or raising community_min_size)
        # would leave the UI showing themes that no longer exist anywhere.
        await _wipe_communities()
        await report({"type": "progress", "stage": "summarizing", "processed": 0, "total": 0})
        return {"communities": 0, "summarized": 0, "modularity": modularity}

    total = len(communities)
    await report(
        {"type": "progress", "stage": "summarizing", "processed": 0, "total": total}
    )

    try:
        llm = get_chat_llm(
            json_mode=True, temperature=settings.community_summary_temperature
        )
        chain = SUMMARY_PROMPT | llm
    except Exception as e:  # noqa: BLE001 - no LLM configured: still persist clusters
        logger.warning("⚠️ Community summarization unavailable (%s); using fallbacks", e)
        chain = None

    semaphore = asyncio.Semaphore(settings.extraction_concurrency)
    completed = 0
    completed_lock = asyncio.Lock()

    async def summarize(members: list[str]) -> dict:
        nonlocal completed
        if chain is None:
            result = {
                "id": community_id(members),
                "title": _fallback_title(members),
                "summary": f"A cluster of {len(members)} closely related entities.",
                "size": len(members),
                "members": members,
                "llm_ok": False,
            }
        else:
            async with semaphore:
                result = await _summarize_community(
                    chain, members, meta, edges, settings=settings
                )
        async with completed_lock:
            completed += 1
            await report(
                {
                    "type": "progress",
                    "stage": "summarizing",
                    "processed": completed,
                    "total": total,
                    "title": result["title"],
                }
            )
        return result

    summarized = await asyncio.gather(*(summarize(m) for m in communities))
    summarized_ok = sum(1 for c in summarized if c["llm_ok"])

    await report({"type": "progress", "stage": "embedding_communities", "total": total})
    vectors = await _embed_communities(summarized)

    await report({"type": "progress", "stage": "writing_communities", "total": total})
    written = await _persist_communities(summarized, vectors)
    logger.info("✅ Persisted %s/%s communities", written, total)

    return {
        "communities": written,
        "summarized": summarized_ok,
        "modularity": modularity,
    }


def _row_to_community(row: dict) -> dict:
    """Normalize a Neo4j row to the contracted community shape."""
    members = [m for m in (row.get("members") or []) if m]
    size = row.get("size")
    return {
        "id": _clean(row.get("id")),
        "title": _clean(row.get("title")),
        "summary": _clean(row.get("summary")),
        "size": int(size) if size is not None else len(members),
        "members": members,
    }


async def get_community_summaries(limit: int = 10) -> list[dict]:
    """Return the largest communities as ``{id, title, summary, size, members}``."""
    try:
        rows = await execute_query(LIST_COMMUNITIES_QUERY, {"limit": limit})
    except Exception as e:  # noqa: BLE001 - communities may not exist yet
        logger.info("Community summaries unavailable: %s", e)
        return []
    return [_row_to_community(row) for row in rows or []]


async def search_communities(query: str, k: int = 5) -> list[dict]:
    """Vector-search community summaries — the "global search" retrieval path.

    Falls back to the largest communities when embeddings or the vector index are
    unavailable, so thematic questions still get corpus-level context.
    """
    try:
        embeddings = get_embeddings()
        vector = await asyncio.to_thread(embeddings.embed_query, query)
    except Exception as e:  # noqa: BLE001
        logger.info("Community vector search skipped (embedding failed): %s", e)
        return await get_community_summaries(k)

    try:
        rows = await execute_query(
            SEARCH_COMMUNITIES_QUERY,
            {"index": COMMUNITY_VECTOR_INDEX, "k": k, "vec": vector},
        )
    except Exception as e:  # noqa: BLE001 - index missing / no embeddings written
        logger.info("Community vector index unavailable (%s); listing largest", e)
        return await get_community_summaries(k)

    results = []
    for row in rows or []:
        community = _row_to_community(row)
        if row.get("score") is not None:
            community["score"] = float(row["score"])
        results.append(community)
    return results[:k]
