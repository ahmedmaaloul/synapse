# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Chunk Store (source text units)

Extraction distills prose into ~15-word entity ``description`` fields and then
throws the source text away. An honest benchmark showed exactly what that costs:
entity-only GraphRAG *lost* to plain passage vector RAG on raw fact recall
(87.5% / 64.3% vs 89.3% / 71.4%) even though the controlled ablation proved the
graph structure itself is worth +37.6pp recall. The graph was never the problem
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

CHUNKS_FOR_ENTITIES_QUERY = """
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.name IN $names
RETURN DISTINCT c.id AS id, c.text AS text, c.document AS document,
       c.index AS index
ORDER BY document, index
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

    Results are de-duplicated by chunk id and ordered by ``(document, index)`` so
    the excerpts read in document order and the output is stable regardless of
    how the driver happens to return rows.
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

    by_id: dict[str, dict] = {}
    for row in rows or []:
        chunk = _row_to_chunk(row, score=0.0)
        # An entity can mention a chunk more than once across the MATCH; the
        # first row wins so the result is independent of driver row order.
        by_id.setdefault(chunk["id"], chunk)

    ordered = sorted(by_id.values(), key=lambda c: (c["document"], c["index"], c["id"]))
    return ordered[:limit]
