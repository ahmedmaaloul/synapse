"""
Project Synapse — Neo4j Driver
Manages the Neo4j async driver lifecycle.
"""

from neo4j import AsyncGraphDatabase, AsyncDriver
from app.config import get_settings

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
