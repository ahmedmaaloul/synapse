# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Graph Schema Bootstrap

Idempotently creates the Neo4j indexes/constraints that power retrieval:

  • a uniqueness constraint on ``Entity.name`` (fast MERGE + no dup nodes)
  • a full-text index over name+description (keyword / hybrid retrieval)
  • a vector index over the entity embedding (semantic retrieval)
  • a vector + full-text index over ``Community`` title/summary, which is what
    makes corpus-level "global search" over community summaries possible
  • a vector + full-text index over ``Chunk.text`` — the stored source text
    units, so an answer can be grounded in the original prose and not only in
    the 15-word entity descriptions extraction distilled out of it

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
COMMUNITY_FULLTEXT_INDEX = "community_fulltext"
COMMUNITY_VECTOR_INDEX = "community_embedding"
CHUNK_FULLTEXT_INDEX = "chunk_fulltext"
CHUNK_VECTOR_INDEX = "chunk_embedding"
CHUNK_ID_CONSTRAINT = "chunk_id_unique"


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
            # chunk_store relies on `MERGE (c:Chunk {id: ...})` being idempotent
            # across re-ingests. Without a constraint that is only a convention:
            # concurrent ingests can race and create duplicates, and every MERGE
            # does a label scan instead of an index lookup.
            "chunk id uniqueness constraint",
            f"""
            CREATE CONSTRAINT {CHUNK_ID_CONSTRAINT} IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
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
        (
            "community full-text index",
            f"""
            CREATE FULLTEXT INDEX {COMMUNITY_FULLTEXT_INDEX} IF NOT EXISTS
            FOR (c:Community) ON EACH [c.title, c.summary]
            """,
        ),
        (
            "community vector index",
            f"""
            CREATE VECTOR INDEX {COMMUNITY_VECTOR_INDEX} IF NOT EXISTS
            FOR (c:Community) ON (c.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {settings.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }} }}
            """,
        ),
        (
            "chunk vector index",
            f"""
            CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {settings.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }} }}
            """,
        ),
        (
            "chunk full-text index",
            f"""
            CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """,
        ),
    ]

    for label, stmt in statements:
        try:
            await execute_query(stmt)
            logger.info("✅ Ensured Neo4j %s", label)
        except Exception as e:  # noqa: BLE001 - non-fatal; degrade gracefully
            logger.warning("⚠️ Could not create %s: %s", label, e)
