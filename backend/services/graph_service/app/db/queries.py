"""Database query logic and Cypher statements for graph_service."""
from collections.abc import Mapping
from typing import Any

from neo4j import Session


def get_kg_status(session: Session, repo_id: str) -> dict[str, int]:
    """Retrieve indexing status of the knowledge graph for a repository."""
    record = session.run(
        """
        MATCH (n {repo_id: $repo_id})
        WHERE n:KGDocument OR n:KGChunk OR n:KGEntity
        RETURN
          count(CASE WHEN n:KGDocument THEN 1 END) AS docs,
          count(CASE WHEN n:KGChunk THEN 1 END) AS chunks,
          count(CASE WHEN n:KGEntity THEN 1 END) AS entities,
          count(CASE WHEN n:KGChunk AND n.embedding IS NOT NULL THEN 1 END) AS embedded_chunks,
          count(CASE WHEN n:KGEntity AND n.embedding IS NOT NULL THEN 1 END) AS embedded_entities
        """,
        repo_id=repo_id,
    ).single()

    rel_record = session.run(
        """
        MATCH (:KGEntity {repo_id: $repo_id})-[r:KG_RELATION]->(:KGEntity {repo_id: $repo_id})
        RETURN count(r) AS relations
        """,
        repo_id=repo_id,
    ).single()

    return {
        "docs": int(record["docs"]) if record else 0,
        "chunks": int(record["chunks"]) if record else 0,
        "entities": int(record["entities"]) if record else 0,
        "relations": int(rel_record["relations"]) if rel_record else 0,
        "embedded_chunks": int(record["embedded_chunks"]) if record else 0,
        "embedded_entities": int(record["embedded_entities"]) if record else 0,
    }


def fetch_kg_subgraph(
    session: Session, repo_id: str, entity_names: list[str], hops: int, limit: int
) -> dict[str, list[dict[str, Any]]]:
    """Extract a structural knowledge subgraph around the provided entities."""
    if not entity_names:
        return {"nodes": [], "edges": []}

    query = f"""
    MATCH (seed:KGEntity {{repo_id: $repo_id}})
    WHERE toLower(seed.name) IN $names
    
    // Expand graph around seeds up to `hops` relationships
    CALL apoc.path.subgraphAll(seed, {{
        relationshipFilter: "KG_RELATION>|<KG_RELATION",
        minLevel: 0,
        maxLevel: {hops},
        limit: {limit}
    }}) YIELD nodes, relationships
    
    // Extract unique nodes from subgraph
    UNWIND nodes AS n
    WITH DISTINCT n, relationships
    
    // Map nodes back directly to their properties
    WITH collect({{
        id: n.entity_id,
        name: n.name,
        type: n.type,
        kind: "entity"
    }}) AS result_nodes, relationships
    
    // Extract unique edges from relationships
    UNWIND relationships AS r
    WITH result_nodes, DISTINCT r
    
    WITH result_nodes, collect({{
        source: startNode(r).entity_id,
        target: endNode(r).entity_id,
        label: "linked",
        relation_type: r.relation_type,
        confidence: r.confidence,
        evidence: r.evidence,
        evidence_chunk_id: r.evidence_chunk_id
    }}) AS result_edges
    
    RETURN result_nodes AS nodes, result_edges AS edges
    """
    
    record = session.run(
        query, repo_id=repo_id, names=[n.lower() for n in entity_names]
    ).single()

    if not record:
        return {"nodes": [], "edges": []}

    return {
        "nodes": record["nodes"] or [],
        "edges": record["edges"] or [],
    }
