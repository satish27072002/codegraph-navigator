"""Knowledge Graph API routes – load, status, subgraph."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import ValidationError

from app.models import (
    KGLoadRequest,
    KGLoadResponse,
    KGStatusResponse,
    KGSubgraphRequest,
    KGSubgraphResponse,
)
from app.db.queries import fetch_kg_subgraph, get_kg_status
from app.kg_store import (
    ensure_kg_schema,
    ensure_kg_vector_indexes,
    set_chunk_embeddings,
    set_entity_embeddings,
    upsert_chunks,
    upsert_entities,
    upsert_mentions,
    upsert_relations,
    upsert_repo_documents,
)

logger = logging.getLogger("graph_service")
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers – late-imported from app.main to avoid circular imports.
# ---------------------------------------------------------------------------
def _m():  # noqa: ANN202
    """Return the main module, imported lazily."""
    import app.main as main  # noqa: F811
    return main


# ---------------------------------------------------------------------------
# /kg/load
# ---------------------------------------------------------------------------
@router.post("/kg/load", response_model=KGLoadResponse)
def kg_load(raw_payload: dict[str, Any] | None = Body(default=None)) -> KGLoadResponse:
    m = _m()

    if raw_payload is None:
        raise HTTPException(status_code=400, detail="repo_id is required for /kg/load")
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="kg/load payload must be a JSON object")

    repo_raw = raw_payload.get("repo_id")
    if repo_raw is None or not str(repo_raw).strip():
        raise HTTPException(status_code=400, detail="repo_id is required for /kg/load")

    try:
        payload = KGLoadRequest.model_validate(raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid /kg/load payload: {exc.errors()}") from exc

    repo_id = str(payload.repo_id)
    if not repo_id.strip():
        raise HTTPException(status_code=400, detail="repo_id is required for /kg/load")
    logger.info(
        "kg.load.received",
        extra={"repo_id": repo_id, "documents": len(payload.documents)},
    )

    docs: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    entity_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    entity_ref_by_name: dict[str, tuple[str, str]] = {}
    mentions: dict[tuple[str, str], dict[str, Any]] = {}
    relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def _ensure_entity_local(name: str, entity_type: str) -> str:
        normalized_name = name.strip()
        normalized_type = entity_type.strip() or "unknown"
        if not normalized_name:
            return ""

        key = (normalized_name.lower(), normalized_type.lower())
        if key not in entity_by_key:
            entity_by_key[key] = {
                "entity_id": m._stable_id(repo_id, normalized_name.lower(), normalized_type.lower()),
                "repo_id": repo_id,
                "name": normalized_name,
                "type": normalized_type,
            }
        entity_id = str(entity_by_key[key]["entity_id"])

        name_key = normalized_name.lower()
        existing = entity_ref_by_name.get(name_key)
        if existing is None:
            entity_ref_by_name[name_key] = (entity_id, normalized_type.lower())
        else:
            _, existing_type = existing
            if existing_type == "unknown" and normalized_type.lower() != "unknown":
                entity_ref_by_name[name_key] = (entity_id, normalized_type.lower())

        return entity_id

    for doc in payload.documents:
        path = doc.path.strip()
        if not path:
            continue
        doc_id = m._stable_id(repo_id, path)
        docs.append(
            {
                "doc_id": doc_id,
                "path": path,
                "language": doc.language.strip(),
            }
        )

        doc_chunks = m._chunk_text(doc.text)
        for chunk_index, chunk_text in enumerate(doc_chunks):
            chunk_id = m._stable_id(repo_id, doc_id, str(chunk_index), chunk_text)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "repo_id": repo_id,
                    "doc_id": doc_id,
                    "index": chunk_index,
                    "text": chunk_text,
                }
            )

            extracted_entities, extracted_relations = m._llm_extract_for_chunk(
                repo_id=repo_id,
                path=path,
                chunk_text=chunk_text,
            )

            chunk_evidence = chunk_text[:280]
            for item in extracted_entities:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                entity_type = str(item.get("type", "unknown")).strip() or "unknown"
                entity_id = _ensure_entity_local(name, entity_type)
                mentions[(entity_id, chunk_id)] = {
                    "entity_id": entity_id,
                    "chunk_id": chunk_id,
                    "evidence": chunk_evidence,
                }

            for rel in extracted_relations:
                source_name = str(rel.get("source", "")).strip()
                target_name = str(rel.get("target", "")).strip()
                if not source_name or not target_name:
                    continue
                relation_type = str(rel.get("relation_type", "related_to")).strip() or "related_to"
                confidence = float(rel.get("confidence", 0.5) or 0.5)
                confidence = max(0.0, min(confidence, 1.0))
                evidence = str(rel.get("evidence", "")).strip() or chunk_evidence

                source_entity_id = entity_ref_by_name.get(source_name.lower(), ("", ""))[0]
                target_entity_id = entity_ref_by_name.get(target_name.lower(), ("", ""))[0]
                if not source_entity_id:
                    source_entity_id = _ensure_entity_local(source_name, "unknown")
                if not target_entity_id:
                    target_entity_id = _ensure_entity_local(target_name, "unknown")
                rel_key = (source_entity_id, target_entity_id, relation_type.lower(), chunk_id)
                relations[rel_key] = {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                    "relation_type": relation_type,
                    "confidence": confidence,
                    "evidence": evidence,
                    "evidence_chunk_id": chunk_id,
                }
                mentions[(source_entity_id, chunk_id)] = {
                    "entity_id": source_entity_id,
                    "chunk_id": chunk_id,
                    "evidence": evidence,
                }
                mentions[(target_entity_id, chunk_id)] = {
                    "entity_id": target_entity_id,
                    "chunk_id": chunk_id,
                    "evidence": evidence,
                }

    entities = list(entity_by_key.values())
    mention_rows = list(mentions.values())
    relation_rows = list(relations.values())

    # Defensive consistency checks to prevent writing a KG under the wrong repo_id.
    if any(str(item.get("repo_id", "")) != repo_id for item in chunks):
        raise HTTPException(status_code=500, detail="internal error: chunk repo_id mismatch before write")
    if any(str(item.get("repo_id", "")) != repo_id for item in entities):
        raise HTTPException(status_code=500, detail="internal error: entity repo_id mismatch before write")

    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        ensure_kg_schema(session)
        upsert_repo_documents(session, repo_id, docs)
        upsert_chunks(session, repo_id, chunks)
        upsert_entities(session, repo_id, entities)
        upsert_mentions(session, repo_id, mention_rows)
        upsert_relations(session, repo_id, relation_rows)

        if m.ENABLE_EMBEDDINGS and chunks:
            chunk_dims: int | None = None
            for idx in range(0, len(chunks), m.EMBEDDING_BATCH_SIZE):
                batch = chunks[idx : idx + m.EMBEDDING_BATCH_SIZE]
                vectors = m._openai_embed([str(item.get("text", "")) for item in batch])
                if vectors and chunk_dims is None:
                    chunk_dims = len(vectors[0])
                set_chunk_embeddings(
                    session,
                    repo_id,
                    [
                        {"chunk_id": batch[i]["chunk_id"], "embedding": vectors[i]}
                        for i in range(min(len(batch), len(vectors)))
                    ],
                )

            entity_dims: int | None = None
            if entities:
                for idx in range(0, len(entities), m.EMBEDDING_BATCH_SIZE):
                    batch = entities[idx : idx + m.EMBEDDING_BATCH_SIZE]
                    vectors = m._openai_embed(
                        [f"{item.get('name', '')}\n{item.get('type', '')}" for item in batch]
                    )
                    if vectors and entity_dims is None:
                        entity_dims = len(vectors[0])
                    set_entity_embeddings(
                        session,
                        repo_id,
                        [
                            {"entity_id": batch[i]["entity_id"], "embedding": vectors[i]}
                            for i in range(min(len(batch), len(vectors)))
                        ],
                    )
            ensure_kg_vector_indexes(session, chunk_dims, entity_dims)

    response = KGLoadResponse(
        docs=len(docs),
        chunks=len(chunks),
        entities=len(entities),
        relations=len(relation_rows),
    )
    logger.info(
        "kg.load.written",
        extra={
            "repo_id": repo_id,
            "docs": response.docs,
            "chunks": response.chunks,
            "entities": response.entities,
            "relations": response.relations,
        },
    )
    return response


# ---------------------------------------------------------------------------
# /kg/status
# ---------------------------------------------------------------------------
@router.get("/kg/status", response_model=KGStatusResponse)
def kg_status(repo_id: str = Query(...)) -> KGStatusResponse:
    m = _m()
    repo = str(repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        status = get_kg_status(session, repo)
    return KGStatusResponse(repo_id=repo, **status)


# ---------------------------------------------------------------------------
# /kg/subgraph
# ---------------------------------------------------------------------------
@router.post("/kg/subgraph", response_model=KGSubgraphResponse)
def kg_subgraph(payload: KGSubgraphRequest) -> KGSubgraphResponse:
    m = _m()
    repo = str(payload.repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        graph = fetch_kg_subgraph(
            session,
            repo_id=repo,
            entity_names=payload.entity_names,
            hops=payload.hops,
            limit=payload.limit,
        )
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    logger.info(
        "kg.subgraph.result",
        extra={
            "repo_id": repo,
            "entity_names_count": len(payload.entity_names),
            "hops": payload.hops,
            "nodes_count": len(nodes),
            "edges_count": len(edges),
        },
    )
    return KGSubgraphResponse(nodes=nodes, edges=edges)
