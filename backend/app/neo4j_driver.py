# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Neo4j Driver
Manages the Neo4j async driver lifecycle.
"""

import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    """Get or create the Neo4j async driver."""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver() -> None:
    """Close the Neo4j driver connection."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def execute_query(query: str, parameters: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return the results as a list of dicts."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        records = await result.data()
        return records


async def verify_connectivity() -> bool:
    """Return True if Neo4j is reachable (used by the readiness probe)."""
    try:
        driver = await get_driver()
        await driver.verify_connectivity()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Neo4j connectivity check failed: %s", e)
        return False
