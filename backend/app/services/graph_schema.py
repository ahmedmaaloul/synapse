"""
Project Synapse — Graph Schema Bootstrap

Idempotently creates the Neo4j indexes/constraints that power retrieval:

  • a uniqueness constraint on ``Entity.name`` (fast MERGE + no dup nodes)
  • a full-text index over name+description (keyword / hybrid retrieval)
  • a vector index over the entity embedding (semantic retrieval)

Called once at startup. Each statement is ``IF NOT EXISTS`` so re-running is safe,
and failures are logged but non-fatal — the app still serves keyword retrieval if,
say, the Neo4j build is too old for vector indexes.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.neo4j_driver import execute_query

logger = logging.getLogger(__name__)

ENTITY_NAME_CONSTRAINT = "entity_name_unique"
ENTITY_FULLTEXT_INDEX = "entity_fulltext"
ENTITY_VECTOR_INDEX = "entity_embedding"


async def ensure_schema() -> None:
    """Create constraints/indexes if they don't already exist."""
    settings = get_settings()

    statements: list[tuple[str, str]] = [
        (
            "name uniqueness constraint",
            f"""
            CREATE CONSTRAINT {ENTITY_NAME_CONSTRAINT} IF NOT EXISTS
            FOR (n:Entity) REQUIRE n.name IS UNIQUE
            """,
        ),
        (
            "full-text index",
            f"""
            CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name, n.description]
            """,
        ),
        (
            "vector index",
            f"""
            CREATE VECTOR INDEX {ENTITY_VECTOR_INDEX} IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {settings.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }} }}
            """,
        ),
    ]

    for label, stmt in statements:
        try:
            await execute_query(stmt)
            logger.info("✅ Ensured Neo4j %s", label)
        except Exception as e:  # noqa: BLE001 - non-fatal; degrade gracefully
            logger.warning("⚠️ Could not create %s: %s", label, e)
