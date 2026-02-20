"""
Neo4j data loader.
Writes graph nodes and relationships to Neo4j using async sessions
and parameterized Cypher queries (MERGE for idempotent re-ingestion).
Never use string interpolation in Cypher — always use $params.
"""

import logging
from neo4j import AsyncSession
from db.neo4j_client import get_driver

logger = logging.getLogger(__name__)

# Batch sizes for Cypher UNWIND
_NODE_BATCH_SIZE = 200
_REL_BATCH_SIZE = 500


# ─────────────────────────────────────────────
# Per-label Cypher MERGE templates
# ─────────────────────────────────────────────

_MERGE_FUNCTION = """
UNWIND $rows AS row
MERGE (n:Function {id: row.id})
SET n.name        = row.name,
    n.simple_name = row.simple_name,
    n.file        = row.file,
    n.start_line  = row.start_line,
    n.end_line    = row.end_line,
    n.docstring   = row.docstring,
    n.code        = row.code,
    n.complexity  = row.complexity,
    n.loc         = row.loc,
    n.class_name  = row.class_name,
    n.codebase_id = row.codebase_id,
    n.embedding   = row.embedding
RETURN count(n) AS count
"""

_MERGE_CLASS = """
UNWIND $rows AS row
MERGE (n:Class {id: row.id})
SET n.name        = row.name,
    n.file        = row.file,
    n.start_line  = row.start_line,
    n.end_line    = row.end_line,
    n.docstring   = row.docstring,
    n.methods     = row.methods,
    n.codebase_id = row.codebase_id
RETURN count(n) AS count
"""

_MERGE_FILE = """
UNWIND $rows AS row
MERGE (n:File {id: row.id})
SET n.path        = row.path,
    n.language    = row.language,
    n.loc         = row.loc,
    n.codebase_id = row.codebase_id
RETURN count(n) AS count
"""

_MERGE_MODULE = """
UNWIND $rows AS row
MERGE (n:Module {id: row.id})
SET n.name        = row.name,
    n.type        = row.type,
    n.codebase_id = row.codebase_id
RETURN count(n) AS count
"""

# Relationship MERGE: lookup source + target by id, then MERGE the rel
_MERGE_RELATIONSHIP = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
CALL apoc.merge.relationship(src, row.rel_type, {}, row.props, tgt)
YIELD rel
RETURN count(rel) AS count
"""

# Fallback without APOC — individual relationship types
_MERGE_REL_CONTAINS = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:CONTAINS]->(tgt)
RETURN count(r) AS count
"""
_MERGE_REL_HAS_METHOD = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:HAS_METHOD]->(tgt)
RETURN count(r) AS count
"""
_MERGE_REL_CALLS = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:CALLS {line_number: row.props.line_number}]->(tgt)
RETURN count(r) AS count
"""
_MERGE_REL_IMPORTS = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:IMPORTS]->(tgt)
RETURN count(r) AS count
"""
_MERGE_REL_INHERITS = """
UNWIND $rows AS row
MATCH (src {id: row.source_id})
MATCH (tgt {id: row.target_id})
MERGE (src)-[r:INHERITS]->(tgt)
RETURN count(r) AS count
"""

_REL_CYPHER_MAP = {
    "CONTAINS":  _MERGE_REL_CONTAINS,
    "HAS_METHOD": _MERGE_REL_HAS_METHOD,
    "CALLS":     _MERGE_REL_CALLS,
    "IMPORTS":   _MERGE_REL_IMPORTS,
    "INHERITS":  _MERGE_REL_INHERITS,
}

_LABEL_CYPHER_MAP = {
    "Function": _MERGE_FUNCTION,
    "Class":    _MERGE_CLASS,
    "File":     _MERGE_FILE,
    "Module":   _MERGE_MODULE,
}


def _batches(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


async def load_nodes(session: AsyncSession, nodes: list[dict]) -> int:
    """
    Batch-load nodes grouped by label using MERGE.
    Returns total count of nodes upserted.
    """
    # Group by label
    by_label: dict[str, list[dict]] = {}
    for node in nodes:
        label = node.get("label", "Unknown")
        by_label.setdefault(label, []).append(node)

    total = 0
    for label, label_nodes in by_label.items():
        cypher = _LABEL_CYPHER_MAP.get(label)
        if not cypher:
            logger.warning(f"No MERGE template for label '{label}', skipping {len(label_nodes)} nodes")
            continue
        for batch in _batches(label_nodes, _NODE_BATCH_SIZE):
            result = await session.run(cypher, rows=batch)
            record = await result.single()
            if record:
                total += record["count"]

    return total


async def load_relationships(session: AsyncSession, relationships: list[tuple]) -> int:
    """
    Batch-load relationships grouped by type using MERGE.
    Each tuple: (source_id, rel_type, target_id, props_dict)
    Returns total count of relationships upserted.
    """
    # Group by rel_type
    by_type: dict[str, list[dict]] = {}
    for source_id, rel_type, target_id, props in relationships:
        row = {
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
            "props": props or {},
        }
        by_type.setdefault(rel_type, []).append(row)

    total = 0
    for rel_type, rows in by_type.items():
        cypher = _REL_CYPHER_MAP.get(rel_type)
        if not cypher:
            logger.warning(f"No MERGE template for relationship '{rel_type}', skipping {len(rows)}")
            continue
        for batch in _batches(rows, _REL_BATCH_SIZE):
            try:
                result = await session.run(cypher, rows=batch)
                record = await result.single()
                if record:
                    total += record["count"]
            except Exception as exc:
                logger.warning(f"Relationship batch load failed for {rel_type}: {exc}")

    return total


async def load_graph(nodes: list[dict], relationships: list[tuple]) -> dict:
    """
    Load a full graph (nodes + relationships) into Neo4j.
    Returns { "nodes_created": int, "relationships_created": int }.
    """
    driver = await get_driver()
    async with driver.session() as session:
        nodes_created = await load_nodes(session, nodes)
        rels_created = await load_relationships(session, relationships)

    logger.info(f"Loaded {nodes_created} nodes, {rels_created} relationships into Neo4j")
    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
    }


async def delete_codebase(codebase_id: str) -> None:
    """
    Delete all nodes and relationships for a given codebase_id.
    Useful for re-ingestion without stale data.
    """
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            "MATCH (n {codebase_id: $cid}) DETACH DELETE n",
            cid=codebase_id,
        )
    logger.info(f"Deleted all nodes for codebase '{codebase_id}'")
