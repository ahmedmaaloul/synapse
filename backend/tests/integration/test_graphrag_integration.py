"""End-to-end GraphRAG data-layer test against a REAL Neo4j.

Exercises the parts that mocks can't prove: vector index creation, writing node
embeddings via ``db.create.setNodeVectorProperty``, and hybrid retrieval. Uses
the offline ``fake`` embedder so NO LLM or cloud key is required.

Run it:
    docker compose up -d neo4j
    SYNAPSE_IT=1 EMBEDDING_PROVIDER=fake NEO4J_URI=bolt://localhost:7687 \\
        pytest tests/integration -v

Skipped automatically unless SYNAPSE_IT=1.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SYNAPSE_IT") != "1",
        reason="integration test — set SYNAPSE_IT=1 with a live Neo4j to run",
    ),
]


@pytest.fixture
async def clean_db():
    from app.neo4j_driver import close_driver, execute_query

    await execute_query("MATCH (n) DETACH DELETE n")
    yield
    await execute_query("MATCH (n) DETACH DELETE n")
    await close_driver()


async def test_full_ingest_and_retrieve(clean_db):
    from app.services import graph_builder
    from app.services.chat_engine import retrieve_subgraph
    from app.services.graph_schema import ensure_schema

    await ensure_schema()

    entities = [
        {"name": "Ada Lovelace", "type": "PERSON", "description": "First programmer"},
        {"name": "Analytical Engine", "type": "PROJECT", "description": "Mechanical computer"},
        {"name": "Mathematics", "type": "SKILL", "description": "Study of numbers"},
    ]
    relationships = [
        {"source": "Ada Lovelace", "target": "Analytical Engine", "type": "WORKED_ON", "description": "wrote algorithms"},
        {"source": "Ada Lovelace", "target": "Mathematics", "type": "HAS_SKILL", "description": "expert"},
    ]

    embeddings = await graph_builder._embed_entities(entities)
    assert embeddings, "fake embedder should produce vectors"

    nodes = await graph_builder._write_entities(entities, "history.pdf", embeddings)
    assert nodes == 3

    rels = await graph_builder._write_relationships(relationships)
    assert rels == 2

    # Vector + graph retrieval should surface Ada and her neighborhood.
    context, citations = await retrieve_subgraph("Who worked on the analytical engine?")
    names = {c["name"] for c in citations}
    assert "Analytical Engine" in names or "Ada Lovelace" in names
    assert "WORKED_ON" in context or "Analytical Engine" in context


async def test_graph_data_endpoint(clean_db):
    from app.routers.graph import get_graph_data
    from app.services import graph_builder

    await graph_builder._write_entities(
        [{"name": "Docker", "type": "TOOL", "description": "containers"}], "d.pdf", {}
    )
    data = await get_graph_data()
    labels = {n["label"] for n in data["nodes"]}
    assert "Docker" in labels
    # Type must be the domain type property, not the Neo4j "Entity" label.
    docker_node = next(n for n in data["nodes"] if n["label"] == "Docker")
    assert docker_node["type"] == "TOOL"
    # Embedding vectors must never be shipped to the client.
    assert "embedding" not in docker_node["properties"]
