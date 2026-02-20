"""
Project Synapse — Graph Router
Serves the knowledge graph data for frontend visualization.
"""

from fastapi import APIRouter
from app.neo4j_driver import execute_query

router = APIRouter()


@router.get("/graph-data")
async def get_graph_data():
    """
    Fetch all nodes and relationships from Neo4j
    for the force-graph visualization.
    """
    # Fetch nodes
    nodes_query = """
    MATCH (n)
    RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
    """
    nodes_raw = await execute_query(nodes_query)

    # Fetch relationships
    rels_query = """
    MATCH (a)-[r]->(b)
    RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type, properties(r) AS properties
    """
    rels_raw = await execute_query(rels_query)

    # Format for react-force-graph
    nodes = [
        {
            "id": str(n["id"]),
            "label": n["properties"].get("name", n["labels"][0] if n["labels"] else "Unknown"),
            "type": n["labels"][0] if n["labels"] else "Unknown",
            "properties": n["properties"],
        }
        for n in nodes_raw
    ]

    links = [
        {
            "source": str(r["source"]),
            "target": str(r["target"]),
            "type": r["type"],
            "properties": r.get("properties", {}),
        }
        for r in rels_raw
    ]

    return {"nodes": nodes, "links": links}


from fastapi import APIRouter, HTTPException

@router.delete("/graph")
async def clear_graph():
    """
    Delete all nodes and relationships in the Neo4j database.
    """
    query = "MATCH (n) DETACH DELETE n"
    try:
        await execute_query(query)
        return {"status": "success", "message": "Knowledge graph cleared"}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error clearing graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear graph: {str(e)}")
