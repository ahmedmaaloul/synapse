# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Graph Router

Serves graph data for the force-graph visualization, exposes the Louvain
*communities* (the corpus-level "themes" that power global search), and supports
clearing the DB.

Community rebuilds are long-running (Louvain + one LLM summary per community),
so they run as a background job on the same ``job_bus`` + SSE pattern as
ingestion: ``POST /api/communities/rebuild`` returns a ``job_id`` and the client
watches ``GET /api/communities/rebuild/{job_id}/events``.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.neo4j_driver import execute_query
from app.services.communities import detect_and_summarize, get_community_summaries
from app.services.jobs import job_bus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/graph-data")
async def get_graph_data():
    """Fetch the ENTITY graph, formatted for react-force-graph.

    Deliberately scoped to ``:Entity`` — on both ends of the relationship match,
    which is what keeps the non-entity nodes *and* their edges out:

    * Community detection writes ``(:Community)`` nodes joined by
      ``[:IN_COMMUNITY]``.
    * Source-chunk retrieval writes ``(:Chunk)`` nodes joined by
      ``[:MENTIONED_IN]``.

    Neither carries a ``name``/``type``, so an unscoped ``MATCH (n)`` would
    render them as unnamed grey blobs wired to every member — and chunks would
    additionally ship whole paragraphs of source prose to the browser. Themes
    are surfaced by ``/api/communities``; source excerpts ride the chat
    endpoint's ``sources`` event instead.
    """
    nodes_raw = await execute_query(
        """
        MATCH (n:Entity)
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
    )
    rels_raw = await execute_query(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN elementId(a) AS source, elementId(b) AS target,
               type(r) AS type, properties(r) AS properties
        """
    )

    nodes = [
        {
            "id": str(n["id"]),
            "label": n["properties"].get("name", n["labels"][0] if n["labels"] else "Unknown"),
            # Color by the domain `type` property (PERSON, TOOL, ...), not the
            # Neo4j label (which is "Entity" for every node).
            "type": n["properties"].get("type", n["labels"][0] if n["labels"] else "Unknown"),
            # Never ship embedding vectors to the browser — large and useless there.
            "properties": {k: v for k, v in n["properties"].items() if k != "embedding"},
        }
        for n in nodes_raw
    ]

    links = [
        {
            "source": str(r["source"]),
            "target": str(r["target"]),
            # Prefer the semantic type stored on the relationship, else the Neo4j type.
            "type": r.get("properties", {}).get("type", r["type"]),
            "properties": r.get("properties", {}),
        }
        for r in rels_raw
    ]

    return {"nodes": nodes, "links": links}


@router.get("/communities")
async def list_communities(limit: int = Query(20, ge=1, le=100)):
    """Return the largest community summaries — the corpus' detected themes.

    Never fails on a cold database: ``get_community_summaries`` returns ``[]``
    when nothing has been clustered yet, so the UI can render an empty state
    instead of an error.
    """
    communities = await get_community_summaries(limit=limit)
    return {"communities": communities, "count": len(communities)}


async def _rebuild_communities(job_id: str) -> None:
    """Background task: re-run Louvain + summarization, streaming progress."""
    try:

        async def on_progress(event: dict) -> None:
            await job_bus.publish(job_id, event)

        result = await detect_and_summarize(on_progress=on_progress)

        await job_bus.publish(job_id, {"type": "done", "data": result})
        logger.info(
            "✅ Rebuilt %s communities (modularity %s)",
            result["communities"],
            result["modularity"],
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("❌ Community rebuild failed")
        await job_bus.publish(job_id, {"type": "error", "data": str(e)})


@router.post("/communities/rebuild")
async def rebuild_communities(background_tasks: BackgroundTasks):
    """Kick off community detection in the background; returns a ``job_id``."""
    job_id = job_bus.create()
    background_tasks.add_task(_rebuild_communities, job_id)
    return {"job_id": job_id, "status": "processing"}


@router.get("/communities/rebuild/{job_id}/events")
async def community_rebuild_events(job_id: str):
    """SSE stream of progress events for a community-rebuild job."""

    async def stream():
        async for event in job_bus.subscribe(job_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/graph")
async def clear_graph():
    """Delete all nodes and relationships in the database."""
    try:
        await execute_query("MATCH (n) DETACH DELETE n")
        return {"status": "success", "message": "Knowledge graph cleared"}
    except Exception as e:
        logger.error("Error clearing graph: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear graph: {e}") from e
