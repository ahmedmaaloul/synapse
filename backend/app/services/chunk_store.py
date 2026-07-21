# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Chunk Store (source text units)

Extraction distills prose into ~15-word entity ``description`` fields and then
throws the source text away. An honest benchmark showed exactly what that costs:
entity-only GraphRAG *lost* to plain passage vector RAG on raw fact recall
(87.5% / 64.3% vs 89.3% / 71.4%) even though the controlled ablation proved the
graph structure itself is worth +39.4pp recall (48.1% -> 87.5%, permissive rule;
see benchmarks/results.md). The graph was never the problem
— the input was. We were retrieving lossy summaries while the baseline retrieved
the actual prose.

This module is the fix, and it is what production GraphRAG does: keep every
source chunk as a first-class **text unit**, link it to the entities that were
extracted *from it*, and make it retrievable both semantically (vector index)
and structurally (walk from a seed entity to the passages that mention it). The
chat engine can then ground an answer in the graph *and* the original evidence.

Model::

    (:Chunk {id, text, document, index, embedding})
    (:Entity)-[:MENTIONED_IN]->(:Chunk)

``id`` is a content hash of ``document + index + text``, so re-ingesting the same
document MERGEs onto the same nodes instead of duplicating the corpus.

**:Chunk nodes are storage, not graph.** They must never reach the force-graph
visualization — ``/api/graph-data`` is scoped to ``(n:Entity)`` and stays that
way (see ``test_chunk_store.py``); a chunk rendered as a node would be a wall of
text wired to a dozen entities.

Every read degrades gracefully: a missing vector index, a cold database or an
unavailable embedding backend yields ``[]`` rather than an error, exactly like
``chat_engine``'s seed retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from app.neo4j_driver import execute_query
from app.services.graph_schema import CHUNK_VECTOR_INDEX
from app.services.llm_provider import get_embeddings

logger = logging.getLogger(__name__)

#: Separator used when hashing the id components. A control character that can
#: never appear in extracted PDF prose, so ``("doc", 1, "a")`` and
#: ``("doc", 1, "a")`` are the only inputs that can collide — i.e. identical
#: chunks, which is precisely when we *want* the same id.
_ID_SEP = "\x1f"

#: Length of the hex digest kept for the chunk id. 32 hex chars = 128 bits of a
#: SHA-256 digest: collision-free in practice for any realistic corpus, and short
#: enough to stay readable in Cypher output and test assertions.
_ID_LENGTH = 32


# ── Identity ─────────────────────────────────────────
def chunk_id(document_name: str, index: int, text: str) -> str:
    """Deterministic content-addressed id for a chunk.

    Re-ingesting an unchanged document produces the same ids, so the MERGEs in
    :func:`store_chunks` update in place instead of piling up duplicate copies of
    the corpus. Editing the text (or moving it to another position/document)
    changes the id, which is the intended behaviour: it is a different text unit.
    """
    payload = _ID_SEP.join([str(document_name), str(index), str(text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_LENGTH]


# ── Cypher ───────────────────────────────────────────
# MERGE on the content-addressed id is what makes re-ingestion idempotent.
WRITE_CHUNK_QUERY = """
MERGE (c:Chunk {id: $id})
SET c.text = $text,
    c.document = $document,
    c.index = $index
WITH c
CALL {
    WITH c
    WITH c WHERE $embedding IS NOT NULL
    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
}
RETURN c.id AS id
"""

# Identical minus the vector property — for Neo4j builds without the vector
# procedures. The chunk is still stored and still reachable from its entities.
WRITE_CHUNK_FALLBACK_QUERY = """
MERGE (c:Chunk {id: $id})
SET c.text = $text,
    c.document = $document,
    c.index = $index
RETURN c.id AS id
"""

# MATCH (not MERGE) on the entity: the nodes were written moments earlier by
# graph_builder. MERGEing here would resurrect an entity that resolution had
# deliberately just merged away, silently re-introducing the duplicate.
LINK_CHUNK_QUERY = """
MATCH (c:Chunk {id: $id})
UNWIND $names AS name
MATCH (e:Entity {name: name})
MERGE (e)-[:MENTIONED_IN]->(c)
RETURN count(*) AS links
"""

SEARCH_CHUNKS_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $vec)
YIELD node, score
RETURN node.id AS id, node.text AS text, node.document AS document,
       node.index AS index, score
"""

# Ranked, *then* truncated. The previous version ordered by (document, index)
# before applying LIMIT, so whenever more chunks mentioned the seeds than the
# limit allowed, Neo4j kept the alphabetically-first document and discarded the
# rest — relevance never entered, and the one retrieval channel plain vector RAG
# cannot reproduce was decided by filename.
#
# ``$names`` arrives in relevance order (see :func:`chunks_for_entities`), so the
# position of a name in that list *is* its rank. UNWINDing the positions rather
# than the names lets both ranking signals be computed in one pass:
#   seed_mentions   — how many distinct seeds the chunk mentions. A chunk
#                     covering three of the question's entities is far more
#                     likely to contain the multi-hop link than one covering one.
#   best_seed_rank  — the best (lowest) position among the seeds it mentions, so
#                     a chunk about the top-ranked seed beats an equally broad
#                     chunk about weaker ones.
# (document, index, id) is the final tie-break: purely for determinism, never a
# relevance signal. ``id`` is included because a re-chunked document can leave
# two chunks sharing (document, index) with different text.
CHUNKS_FOR_ENTITIES_QUERY = """
UNWIND range(0, size($names) - 1) AS seed_rank
MATCH (e:Entity {name: $names[seed_rank]})-[:MENTIONED_IN]->(c:Chunk)
WITH c, count(DISTINCT seed_rank) AS seed_mentions, min(seed_rank) AS best_seed_rank
RETURN c.id AS id, c.text AS text, c.document AS document, c.index AS index,
       seed_mentions, best_seed_rank
ORDER BY seed_mentions DESC, best_seed_rank ASC, document ASC, index ASC, id ASC
LIMIT $limit
"""


# ── Helpers ──────────────────────────────────────────
async def _embed_chunks(chunks: list[str]) -> list[list[float] | None]:
    """Embed every chunk in ONE batched call, off the event loop.

    Best-effort by design: if the embedding backend is missing or throws, each
    chunk gets ``None`` and is stored without a vector. The text unit is still
    persisted and still retrievable through its entities — degraded, not lost.
    """
    if not chunks:
        return []
    try:
        embeddings = get_embeddings()
        # LangChain embeddings are sync; keep them off the event loop.
        vectors = await asyncio.to_thread(embeddings.embed_documents, list(chunks))
    except Exception as e:  # noqa: BLE001 - retrieval degrades, ingestion must not fail
        logger.warning("⚠️ Chunk embedding skipped (%s); chunks stored without vectors", e)
        return [None] * len(chunks)
    if len(vectors) != len(chunks):
        logger.warning("⚠️ Chunk embedding count mismatch; storing chunks without vectors")
        return [None] * len(chunks)
    return [list(v) for v in vectors]


def _clean_names(names: list[str] | None) -> list[str]:
    """Trimmed, de-duplicated, order-preserving entity names."""
    seen: dict[str, None] = {}
    for raw in names or []:
        name = str(raw).strip()
        if name:
            seen.setdefault(name, None)
    return list(seen)


def _row_to_chunk(row: dict, *, score: float | None = None) -> dict:
    """Normalize a driver row to the shared excerpt shape."""
    return {
        "id": str(row.get("id") or ""),
        "text": str(row.get("text") or ""),
        "document": str(row.get("document") or ""),
        "index": int(row.get("index") or 0),
        "score": float(row.get("score", 0.0) or 0.0) if score is None else score,
    }


def _as_int(value: object, default: int) -> int:
    """Coerce a driver value to ``int``, falling back on missing/garbage input.

    Explicitly *not* ``int(value or default)``: a legitimate ``0`` (the top seed
    rank) must survive, and ``or`` would silently demote it to the default.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _rank_key(chunk: dict) -> tuple[int, int, str, int, str]:
    """Sort key mirroring ``CHUNKS_FOR_ENTITIES_QUERY``'s ORDER BY.

    Relevance first (more seeds covered, then a better best seed), then
    (document, index, id) purely to keep the output deterministic — the driver
    is free to hand rows back in any order it likes.
    """
    return (
        -chunk["seed_mentions"],
        chunk["best_seed_rank"],
        chunk["document"],
        chunk["index"],
        chunk["id"],
    )


# ── Write path ───────────────────────────────────────
async def store_chunks(
    document_name: str,
    chunks: list[str],
    entity_names_per_chunk: list[list[str]],
) -> dict:
    """Persist source chunks as ``:Chunk`` nodes and link their entities.

    Args:
        document_name: Source file the chunks came from.
        chunks: The raw text units, in document order.
        entity_names_per_chunk: For each chunk, the **canonical** entity names
            extracted from it. Callers must apply entity resolution's alias map
            first — linking a name that resolution merged away leaves the chunk
            orphaned, because the node no longer exists.

    Returns:
        ``{"chunks": <nodes written>, "links": <entity→chunk edges>}``
    """
    if not chunks:
        return {"chunks": 0, "links": 0}

    vectors = await _embed_chunks(chunks)
    chunks_written = 0
    links_written = 0

    for index, text in enumerate(chunks):
        names_for_chunk = (
            entity_names_per_chunk[index] if index < len(entity_names_per_chunk) else []
        )
        params = {
            "id": chunk_id(document_name, index, text),
            "text": text,
            "document": document_name,
            "index": index,
            "embedding": vectors[index] if index < len(vectors) else None,
        }
        try:
            await execute_query(WRITE_CHUNK_QUERY, params)
        except Exception:  # noqa: BLE001 - older Neo4j lacks the vector procedures
            try:
                await execute_query(WRITE_CHUNK_FALLBACK_QUERY, params)
            except Exception as e:  # noqa: BLE001
                logger.warning("⚠️ Chunk write failed (%s #%s): %s", document_name, index, e)
                continue
        chunks_written += 1

        names = _clean_names(names_for_chunk)
        if not names:
            continue
        try:
            rows = await execute_query(LINK_CHUNK_QUERY, {"id": params["id"], "names": names})
        except Exception as e:  # noqa: BLE001 - the chunk itself is already safe
            logger.warning("⚠️ Chunk link failed (%s #%s): %s", document_name, index, e)
            continue
        # Prefer the count the database actually reports; fall back to the number
        # of links requested when the driver returns nothing to count.
        links_written += int(rows[0].get("links") or 0) if rows else len(names)

    logger.info(
        "🧾 Stored %s source chunks (%s entity links) for %s",
        chunks_written,
        links_written,
        document_name,
    )
    return {"chunks": chunks_written, "links": links_written}


# ── Read path ────────────────────────────────────────
async def search_chunks(query: str, k: int) -> list[dict]:
    """Semantic search over the stored text units (best-effort).

    Returns ``[]`` — never raises — when embeddings or the vector index are
    unavailable, so a cold or old database degrades to graph-only retrieval
    instead of breaking the answer.
    """
    if not str(query or "").strip() or k <= 0:
        return []

    try:
        embeddings = get_embeddings()
        vector = await asyncio.to_thread(embeddings.embed_query, query)
    except Exception as e:  # noqa: BLE001
        logger.warning("Chunk vector search skipped (embedding failed): %s", e)
        return []

    try:
        rows = await execute_query(
            SEARCH_CHUNKS_QUERY, {"index": CHUNK_VECTOR_INDEX, "k": int(k), "vec": vector}
        )
    except Exception as e:  # noqa: BLE001 - no chunk vector index yet
        logger.info("Chunk vector search unavailable (%s); skipping source excerpts", e)
        return []

    return [_row_to_chunk(row) for row in rows or []]


async def chunks_for_entities(names: list[str], limit: int) -> list[dict]:
    """Source excerpts that mention any of ``names`` — the structural read path.

    This is the half plain vector RAG cannot do: having found the right entities
    through the graph (including via multi-hop expansion), pull the *prose those
    entities were extracted from*.

    Args:
        names: Seed entity names **in relevance order** — the caller's ranking
            (``chat_engine`` passes its vector/keyword-fused seeds) is used as a
            tie-break, so passing an unordered list degrades the ranking to
            "coverage only" rather than breaking it.
        limit: Maximum excerpts to return, applied *after* ranking.

    Returns:
        Excerpts ordered by relevance, each carrying the evidence for its own
        position on top of the shared ``{"id", "text", "document", "index",
        "score"}`` shape:

        * ``seed_mentions`` — how many of ``names`` the chunk mentions.
        * ``best_seed_rank`` — position of the best seed it mentions (0 = the
          top-ranked seed); ``len(names)`` when the database reported none.

    Ranking is ``(-seed_mentions, best_seed_rank, document, index, id)``, applied
    in Cypher *before* ``LIMIT`` and re-applied here so the result cannot depend
    on driver row order. Note the consequence: excerpts now come back in
    relevance order, **not** document order.
    """
    wanted = _clean_names(names)
    if not wanted or limit <= 0:
        return []

    try:
        rows = await execute_query(
            CHUNKS_FOR_ENTITIES_QUERY, {"names": wanted, "limit": int(limit)}
        )
    except Exception as e:  # noqa: BLE001 - no chunks stored yet
        logger.info("Chunk lookup unavailable (%s); skipping source excerpts", e)
        return []

    # A row that carries no ranking at all sorts behind every genuinely ranked
    # chunk instead of jumping the queue on a falsy 0.
    unranked = len(wanted)

    by_id: dict[str, dict] = {}
    for row in rows or []:
        chunk = _row_to_chunk(row, score=0.0)
        chunk["seed_mentions"] = max(0, _as_int(row.get("seed_mentions"), 0))
        chunk["best_seed_rank"] = _as_int(row.get("best_seed_rank"), unranked)
        first = by_id.get(chunk["id"])
        if first is None:
            by_id[chunk["id"]] = chunk
            continue
        # The aggregation makes duplicate ids impossible in Cypher, but a driver
        # that ever repeats a chunk must not change the answer: keep the
        # strongest evidence seen for it rather than whichever row landed first.
        first["seed_mentions"] = max(first["seed_mentions"], chunk["seed_mentions"])
        first["best_seed_rank"] = min(first["best_seed_rank"], chunk["best_seed_rank"])

    return sorted(by_id.values(), key=_rank_key)[:limit]
