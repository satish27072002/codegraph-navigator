"""
Graph expander.
Starting from seed nodes returned by hybrid search,
traverse the graph via CALLS, IMPORTS, and INHERITS relationships
up to `hops` steps to gather richer context for LLM generation.
Returns nodes and edges in React Flow format.
"""

import logging
from db.neo4j_client import run_query

logger = logging.getLogger(__name__)

# Relationship types to traverse during expansion
_EXPAND_REL_TYPES = ["CALLS", "IMPORTS", "INHERITS", "HAS_METHOD"]

# Node type -> frontend type string mapping
_LABEL_TO_TYPE = {
    "Function": "Function",
    "Class":    "Class",
    "File":     "File",
    "Module":   "Module",
    "Chunk":    "Function",   # treat chunks as functions for display
}

# Graph expansion Cypher: standard variable-length path — no APOC required.
# Traverses CALLS, IMPORTS, INHERITS, HAS_METHOD up to 2 hops from seed nodes.
# Uses DISTINCT to deduplicate nodes when multiple paths lead to the same node.
_EXPAND_CYPHER_STANDARD = """
MATCH (seed)
WHERE seed.id IN $seed_ids
MATCH (seed)-[:CALLS|IMPORTS|INHERITS|HAS_METHOD*1..2]->(neighbor)
WITH DISTINCT neighbor
RETURN
    neighbor.id          AS id,
    labels(neighbor)[0]  AS label,
    neighbor.name        AS name,
    coalesce(neighbor.file, neighbor.path, '') AS file,
    neighbor.start_line  AS start_line,
    neighbor.end_line    AS end_line
"""

_EXPAND_EDGES_CYPHER = """
MATCH (src)-[r:CALLS|IMPORTS|INHERITS|HAS_METHOD|CONTAINS]->(tgt)
WHERE src.id IN $node_ids AND tgt.id IN $node_ids
RETURN
    src.id      AS source_id,
    tgt.id      AS target_id,
    type(r)     AS rel_type,
    r.line_number AS line_number
"""


def _make_node(row: dict, highlighted: bool) -> dict:
    """Convert a Neo4j record into a React Flow node dict."""
    label = row.get("label") or "Function"
    node_id = row.get("id") or ""
    return {
        "id": node_id,
        "type": _LABEL_TO_TYPE.get(label, "Function"),
        "name": row.get("name") or node_id,
        "file": row.get("file") or "",
        "highlighted": highlighted,
    }


def _make_edge(row: dict) -> dict:
    """Convert a relationship record into a React Flow edge dict."""
    return {
        "id": f"{row['source_id']}-{row['rel_type']}-{row['target_id']}",
        "source": row["source_id"],
        "target": row["target_id"],
        "type": row["rel_type"],
        "line_number": row.get("line_number"),
    }


async def expand_graph(
    seed_node_ids: list[str],
    hops: int = 2,
) -> dict:
    """
    Traverse the Neo4j graph from seed_node_ids up to `hops` steps.

    Returns React Flow compatible:
      {
        "nodes": [{ id, type, name, file, highlighted }],
        "edges": [{ id, source, target, type, line_number }],
      }

    Seed nodes are marked highlighted=True.
    Neighbour nodes are marked highlighted=False.
    """
    if not seed_node_ids:
        return {"nodes": [], "edges": []}

    seed_set = set(seed_node_ids)
    all_nodes: dict[str, dict] = {}   # id -> node dict
    all_edges: list[dict] = []

    # ── Fetch seed nodes first ─────────────────────────────────────────
    seed_rows = await run_query(
        """
        MATCH (n)
        WHERE n.id IN $ids
        RETURN
            n.id         AS id,
            labels(n)[0] AS label,
            n.name       AS name,
            coalesce(n.file, n.path, '') AS file,
            n.start_line AS start_line,
            n.end_line   AS end_line
        """,
        params={"ids": seed_node_ids},
    )
    for row in seed_rows:
        node = _make_node(row, highlighted=True)
        all_nodes[node["id"]] = node

    if hops == 0:
        return {"nodes": list(all_nodes.values()), "edges": []}

    # ── Expand neighbours (standard Cypher, no APOC required) ─────────
    try:
        expand_rows = await run_query(
            _EXPAND_CYPHER_STANDARD,
            params={"seed_ids": seed_node_ids},
        )
    except Exception as exc:
        logger.warning(f"Graph expansion failed: {exc}")
        expand_rows = []

    for row in expand_rows:
        node_id = row.get("id")
        if not node_id or node_id in all_nodes:
            continue
        highlighted = node_id in seed_set
        node = _make_node(row, highlighted=highlighted)
        all_nodes[node_id] = node

    # ── Fetch edges between all collected nodes ────────────────────────
    all_node_ids = list(all_nodes.keys())
    if len(all_node_ids) > 1:
        try:
            edge_rows = await run_query(
                _EXPAND_EDGES_CYPHER,
                params={"node_ids": all_node_ids},
            )
            for row in edge_rows:
                all_edges.append(_make_edge(row))
        except Exception as exc:
            logger.warning(f"Edge fetch failed: {exc}")

    logger.info(
        f"Graph expansion: {len(seed_node_ids)} seeds, {hops} hops "
        f"-> {len(all_nodes)} nodes, {len(all_edges)} edges"
    )

    return {
        "nodes": list(all_nodes.values()),
        "edges": all_edges,
    }


async def get_node_neighbourhood(node_id: str) -> dict:
    """
    Return the immediate neighbourhood (1 hop) of a single node.
    Used by GET /graph/{node_id}.
    """
    return await expand_graph([node_id], hops=1)
