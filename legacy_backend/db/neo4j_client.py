"""
Neo4j database client.
All Neo4j driver interactions go through this module — never create a driver elsewhere.
Always use async sessions and parameterized queries (never string interpolation).
"""

import logging
from neo4j import AsyncGraphDatabase, AsyncDriver
from config import settings

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    """Return the shared async Neo4j driver, creating it on first call."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver() -> None:
    """Close the Neo4j driver. Call on application shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def ensure_indexes() -> None:
    """
    Create required Neo4j vector and full-text indexes if they don't exist.
    Called once on application startup.
    """
    driver = await get_driver()
    async with driver.session() as session:
        # Vector index: Function embeddings (1536d cosine)
        await session.run("""
            CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
            FOR (f:Function) ON f.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """, dims=settings.embedding_dimensions)

        # Vector index: Chunk embeddings (1536d cosine)
        await session.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """, dims=settings.embedding_dimensions)

        # Full-text index: Function name, docstring, code
        await session.run("""
            CREATE FULLTEXT INDEX function_text IF NOT EXISTS
            FOR (f:Function) ON EACH [f.name, f.docstring, f.code]
        """)

        # Regular b-tree index on codebase_id for fast filtering
        await session.run("""
            CREATE INDEX function_codebase IF NOT EXISTS
            FOR (f:Function) ON (f.codebase_id)
        """)

    logger.info("Neo4j indexes ensured")


async def run_query(cypher: str, params: dict | None = None) -> list[dict]:
    """
    Execute a read Cypher query and return results as a list of record dicts.
    Always uses parameterized queries — never string interpolation.
    """
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(cypher, parameters=(params or {}))
        records = await result.data()
    return records
