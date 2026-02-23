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
        "FOR (d:KGDocument) REQUIRE d.doc_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT kg_chunk_chunk_id IF NOT EXISTS "
        "FOR (c:KGChunk) REQUIRE c.chunk_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE CONSTRAINT kg_entity_entity_id IF NOT EXISTS "
        "FOR (e:KGEntity) REQUIRE e.entity_id IS UNIQUE"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_name IF NOT EXISTS "
        "FOR (e:KGEntity) ON (e.name)"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_type IF NOT EXISTS "
        "FOR (e:KGEntity) ON (e.type)"
    ).consume()
    session.run(
        "CREATE INDEX kg_document_repo_id IF NOT EXISTS "
        "FOR (d:KGDocument) ON (d.repo_id)"
    ).consume()
    session.run(
        "CREATE INDEX kg_chunk_repo_id IF NOT EXISTS "
        "FOR (c:KGChunk) ON (c.repo_id)"
    ).consume()
    session.run(
        "CREATE INDEX kg_entity_repo_id IF NOT EXISTS "
        "FOR (e:KGEntity) ON (e.repo_id)"
    ).consume()


def ensure_kg_vector_indexes(session, chunk_dims: int | None, entity_dims: int | None) -> None:
    if chunk_dims and chunk_dims > 0:
        session.run(
            f"CREATE VECTOR INDEX {KG_CHUNK_VECTOR_INDEX} IF NOT EXISTS "
            "FOR (c:KGChunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {chunk_dims}, `vector.similarity_function`: 'cosine'}}}}"
        ).consume()
    if entity_dims and entity_dims > 0:
        session.run(
            f"CREATE VECTOR INDEX {KG_ENTITY_VECTOR_INDEX} IF NOT EXISTS "
            "FOR (e:KGEntity) ON (e.embedding) "
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
        MERGE (d:KGDocument {doc_id: doc.doc_id, repo_id: $repo_id})
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
        MATCH (d:KGDocument {doc_id: chunk.doc_id, repo_id: $repo_id})
        MERGE (c:KGChunk {chunk_id: chunk.chunk_id, repo_id: $repo_id})
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
        MERGE (e:KGEntity {entity_id: entity.entity_id, repo_id: $repo_id})
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
        MATCH (e:KGEntity {entity_id: mention.entity_id, repo_id: $repo_id})
        MATCH (c:KGChunk {chunk_id: mention.chunk_id, repo_id: $repo_id})
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
        MATCH (src:KGEntity {entity_id: rel.source_entity_id, repo_id: $repo_id})
        MATCH (dst:KGEntity {entity_id: rel.target_entity_id, repo_id: $repo_id})
        MERGE (src)-[r:KG_RELATION {
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
        MATCH (c:KGChunk {chunk_id: row.chunk_id, repo_id: $repo_id})
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
        MATCH (e:KGEntity {entity_id: row.entity_id, repo_id: $repo_id})
        SET e.embedding = row.embedding
        """,
        repo_id=repo_id,
        rows=rows,
    ).consume()



