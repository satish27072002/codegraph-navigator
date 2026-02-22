from __future__ import annotations

from collections.abc import Iterable
from typing import Any


KG_CHUNK_VECTOR_INDEX = "kg_chunk_embedding"
KG_ENTITY_VECTOR_INDEX = "kg_entity_embedding"


def ensure_kg_schema(session) -> None:
    session.run(
        "CREATE CONSTRAINT kg_repo_repo_id IF NOT EXISTS "
        "FOR (r:Repo) REQUIRE r.repo_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT kg_document_doc_id IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT kg_chunk_chunk_id IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT kg_entity_entity_id IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_name IF NOT EXISTS "
        "FOR (e:Entity) ON (e.name)"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_type IF NOT EXISTS "
        "FOR (e:Entity) ON (e.type)"
    ).consume()
    session.run(
        "CREATE INDEX kg_document_repo_id IF NOT EXISTS "
        "FOR (d:Document) ON (d.repo_id)"
    ).consume()
    session.run(
        "CREATE INDEX kg_chunk_repo_id IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.repo_id)"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_repo_id IF NOT EXISTS "
        "FOR (e:Entity) ON (e.repo_id)"
    ).consume()


def ensure_kg_vector_indexes(session, chunk_dims: int | None, entity_dims: int | None) -> None:
    if chunk_dims and chunk_dims > 0:
        session.run(
            f"CREATE VECTOR INDEX {KG_CHUNK_VECTOR_INDEX} IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {chunk_dims}, `vector.similarity_function`: 'cosine'}}}}"
        ).consume()
    if entity_dims and entity_dims > 0:
        session.run(
            f"CREATE VECTOR INDEX {KG_ENTITY_VECTOR_INDEX} IF NOT EXISTS "
            "FOR (e:Entity) ON (e.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {entity_dims}, `vector.similarity_function`: 'cosine'}}}}"
        ).consume()


def upsert_repo_documents(session, repo_id: str, documents: list[dict[str, Any]]) -> None:
    session.run(
        """
        MERGE (r:Repo {repo_id: $repo_id})
        SET r.repo_id = $repo_id
        """,
        repo_id=repo_id,
    ).consume()

    if not documents:
        return

    session.run(
        """
        MERGE (r:Repo {repo_id: $repo_id})
        WITH r
        UNWIND $documents AS doc
        MERGE (d:Document {doc_id: doc.doc_id, repo_id: $repo_id})
        SET
          d.repo_id = $repo_id,
          d.path = coalesce(doc.path, ""),
          d.language = coalesce(doc.language, "")
        MERGE (r)-[:HAS_DOCUMENT]->(d)
        """,
        repo_id=repo_id,
        documents=documents,
    ).consume()


def upsert_chunks(session, repo_id: str, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    session.run(
        """
        UNWIND $chunks AS chunk
        MATCH (d:Document {doc_id: chunk.doc_id, repo_id: $repo_id})
        MERGE (c:Chunk {chunk_id: chunk.chunk_id, repo_id: $repo_id})
        SET
          c.repo_id = $repo_id,
          c.doc_id = chunk.doc_id,
          c.index = chunk.index,
          c.text = chunk.text
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        repo_id=repo_id,
        chunks=chunks,
    ).consume()


def upsert_entities(session, repo_id: str, entities: list[dict[str, Any]]) -> None:
    if not entities:
        return

    session.run(
        """
        UNWIND $entities AS entity
        MERGE (e:Entity {entity_id: entity.entity_id, repo_id: $repo_id})
        SET
          e.repo_id = $repo_id,
          e.name = coalesce(entity.name, ""),
          e.type = coalesce(entity.type, "unknown")
        """,
        repo_id=repo_id,
        entities=entities,
    ).consume()


def upsert_mentions(session, repo_id: str, mentions: list[dict[str, Any]]) -> None:
    if not mentions:
        return

    session.run(
        """
        UNWIND $mentions AS mention
        MATCH (e:Entity {entity_id: mention.entity_id, repo_id: $repo_id})
        MATCH (c:Chunk {chunk_id: mention.chunk_id, repo_id: $repo_id})
        MERGE (e)-[m:MENTIONED_IN]->(c)
        SET m.evidence = coalesce(mention.evidence, "")
        """,
        repo_id=repo_id,
        mentions=mentions,
    ).consume()


def upsert_relations(session, repo_id: str, relations: list[dict[str, Any]]) -> None:
    if not relations:
        return

    session.run(
        """
        UNWIND $relations AS rel
        MATCH (src:Entity {entity_id: rel.source_entity_id, repo_id: $repo_id})
        MATCH (dst:Entity {entity_id: rel.target_entity_id, repo_id: $repo_id})
        MERGE (src)-[r:RELATED_TO {
          relation_type: rel.relation_type,
          evidence_chunk_id: rel.evidence_chunk_id
        }]->(dst)
        SET
          r.confidence = rel.confidence,
          r.evidence = coalesce(rel.evidence, "")
        """,
        repo_id=repo_id,
        relations=relations,
    ).consume()


def set_chunk_embeddings(session, repo_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id, repo_id: $repo_id})
        SET c.embedding = row.embedding
        """,
        repo_id=repo_id,
        rows=rows,
    ).consume()


def set_entity_embeddings(session, repo_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id, repo_id: $repo_id})
        SET e.embedding = row.embedding
        """,
        repo_id=repo_id,
        rows=rows,
    ).consume()


def get_kg_status(session, repo_id: str) -> dict[str, int]:
    doc_rec = session.run(
        """
        MATCH (d:Document {repo_id: $repo_id})
        RETURN count(d) AS docs
        """,
        repo_id=repo_id,
    ).single()
    chunk_rec = session.run(
        """
        MATCH (c:Chunk {repo_id: $repo_id})
        RETURN
          count(c) AS chunks,
          count(CASE WHEN c.embedding IS NOT NULL THEN 1 END) AS embedded_chunks
        """,
        repo_id=repo_id,
    ).single()
    ent_rec = session.run(
        """
        MATCH (e:Entity {repo_id: $repo_id})
        RETURN
          count(e) AS entities,
          count(CASE WHEN e.embedding IS NOT NULL THEN 1 END) AS embedded_entities
        """,
        repo_id=repo_id,
    ).single()
    rel_rec = session.run(
        """
        MATCH (:Entity {repo_id: $repo_id})-[r:RELATED_TO]->(:Entity {repo_id: $repo_id})
        RETURN count(r) AS relations
        """,
        repo_id=repo_id,
    ).single()

    return {
        "docs": int(doc_rec["docs"]) if doc_rec else 0,
        "chunks": int(chunk_rec["chunks"]) if chunk_rec else 0,
        "entities": int(ent_rec["entities"]) if ent_rec else 0,
        "relations": int(rel_rec["relations"]) if rel_rec else 0,
        "embedded_chunks": int(chunk_rec["embedded_chunks"]) if chunk_rec else 0,
        "embedded_entities": int(ent_rec["embedded_entities"]) if ent_rec else 0,
    }


def fetch_subgraph(
    session,
    *,
    repo_id: str,
    entity_names: Iterable[str],
    hops: int,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    normalized_names = sorted({name.strip().lower() for name in entity_names if name.strip()})
    if not normalized_names:
        return {"nodes": [], "edges": []}

    seed_record = session.run(
        """
        MATCH (seed:Entity {repo_id: $repo_id})
        WHERE toLower(seed.name) IN $entity_names
        RETURN collect(DISTINCT seed.entity_id) AS seed_ids
        """,
        repo_id=repo_id,
        entity_names=normalized_names,
    ).single()
    seed_ids = (seed_record.get("seed_ids", []) if seed_record else []) or []
    if not seed_ids:
        return {"nodes": [], "edges": []}

    entity_record = session.run(
        f"""
        MATCH (seed:Entity {{repo_id: $repo_id}})
        WHERE seed.entity_id IN $seed_ids
        OPTIONAL MATCH (seed)-[:RELATED_TO*1..{hops}]-(other:Entity {{repo_id: $repo_id}})
        WITH collect(DISTINCT seed) + collect(DISTINCT other) AS raw_entities
        UNWIND raw_entities AS ent
        WITH DISTINCT ent
        WHERE ent IS NOT NULL
        RETURN collect(ent)[..$limit] AS entities
        """,
        repo_id=repo_id,
        seed_ids=seed_ids,
        limit=limit,
    ).single()
    if not entity_record:
        return {"nodes": [], "edges": []}
    raw_entities = entity_record.get("entities", []) or []

    entity_nodes: list[dict[str, Any]] = []
    entity_ids: list[str] = []
    for node in raw_entities:
        entity_id = str(node.get("entity_id", ""))
        if not entity_id:
            continue
        entity_ids.append(entity_id)
        entity_nodes.append(
            {
                "id": entity_id,
                "kind": "entity",
                "repo_id": str(node.get("repo_id", "")),
                "name": str(node.get("name", "")),
                "type": str(node.get("type", "")),
            }
        )
    if not entity_ids:
        return {"nodes": [], "edges": []}

    rel_record = session.run(
        """
        MATCH (src:Entity {repo_id: $repo_id})-[r:RELATED_TO]->(dst:Entity {repo_id: $repo_id})
        WHERE src.entity_id IN $entity_ids AND dst.entity_id IN $entity_ids
        RETURN collect(DISTINCT {
          source_id: src.entity_id,
          target_id: dst.entity_id,
          relation_type: coalesce(r.relation_type, 'related_to'),
          confidence: toFloat(coalesce(r.confidence, 0.0)),
          evidence_chunk_id: coalesce(r.evidence_chunk_id, '')
        })[..$limit] AS rels
        """,
        repo_id=repo_id,
        entity_ids=entity_ids,
        limit=limit,
    ).single()
    raw_rels = rel_record.get("rels", []) if rel_record else []

    edges: list[dict[str, Any]] = []
    for rel in raw_rels:
        source_id = str(rel.get("source_id", ""))
        target_id = str(rel.get("target_id", ""))
        if not source_id or not target_id:
            continue
        relation_type = str(rel.get("relation_type", "related_to"))
        confidence = float(rel.get("confidence", 0.0) or 0.0)
        evidence_chunk_id = str(rel.get("evidence_chunk_id", ""))
        edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "confidence": confidence,
                "evidence_chunk_id": evidence_chunk_id,
                # Compatibility fields for existing consumers that read source/target/type.
                "source": source_id,
                "target": target_id,
                "type": "RELATED_TO",
            }
        )

    return {"nodes": entity_nodes, "edges": edges}
