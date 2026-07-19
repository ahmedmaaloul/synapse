"""
Project Synapse — Graph Router

Serves graph data for the force-graph visualization and supports clearing the DB.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.neo4j_driver import execute_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/graph-data")
async def get_graph_data():
    """Fetch all nodes and relationships, formatted for react-force-graph."""
    nodes_raw = await execute_query(
        """
        MATCH (n)
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
    )
    rels_raw = await execute_query(
        """
        MATCH (a)-[r]->(b)
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


@router.delete("/graph")
async def clear_graph():
    """Delete all nodes and relationships in the database."""
    try:
        await execute_query("MATCH (n) DETACH DELETE n")
        return {"status": "success", "message": "Knowledge graph cleared"}
    except Exception as e:
        logger.error("Error clearing graph: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear graph: {e}") from e
