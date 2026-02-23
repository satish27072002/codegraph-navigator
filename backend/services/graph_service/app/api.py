"""Graph API routes – search, embed, expand, load, subgraph, status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from neo4j.exceptions import Neo4jError

from app.models import (
    EmbeddingStatusResponse,
    ExpandResponse,
    GraphEmbedRequest,
    GraphExpandRequest,
    GraphLoadRequest,
    GraphRepoDefaultSearchRequest,
    GraphSearchRequest,
    GraphVectorSearchRequest,
    RepoStatusResponse,
    SearchHit,
    SearchResponse,
    SubgraphResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers – late-imported from app.main to avoid circular imports.
# Each route function calls _m() to grab the main module on demand.
# ---------------------------------------------------------------------------
def _m():  # noqa: ANN202
    """Return the main module, imported lazily."""
    import app.main as main  # noqa: F811
    return main


# ---------------------------------------------------------------------------
# /graph/load
# ---------------------------------------------------------------------------
@router.post("/graph/load")
def graph_load(payload: GraphLoadRequest) -> dict[str, int | bool]:
    m = _m()
    facts = m._load_facts(Path(payload.facts_path))
    if str(payload.repo_id) != str(facts.get("repo_id", "")):
        raise HTTPException(status_code=400, detail="repo_id does not match facts payload")

    nodes = facts.get("nodes", [])
    edges = facts.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise HTTPException(status_code=400, detail="facts nodes/edges must be arrays")

    repo_id = str(payload.repo_id)
    edges_by_type: dict[str, list[dict]] = {key: [] for key in m.REL_MAP}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        edge_type = str(edge.get("type", "")).lower()
        if edge_type in edges_by_type:
            edges_by_type[edge_type].append(edge)

    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        m._ensure_schema(session)
        m._upsert_nodes(session, repo_id, nodes)
        for edge_type, rel_name in m.REL_MAP.items():
            m._upsert_edges(session, repo_id, edges_by_type[edge_type], rel_name)

    return {
        "ok": True,
        "nodes_upserted": len(nodes),
        "edges_upserted": sum(len(group) for group in edges_by_type.values()),
    }


# ---------------------------------------------------------------------------
# /graph/embed
# ---------------------------------------------------------------------------
@router.post("/graph/embed")
def graph_embed(payload: GraphEmbedRequest) -> dict[str, int | str | bool]:
    import time
    m = _m()
    repo_id = str(payload.repo_id)
    started = time.perf_counter()

    def _duration_ms() -> int:
        return int(round((time.perf_counter() - started) * 1000))

    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        total_nodes, embedded_nodes = m._repo_embedding_counts(session, repo_id)

        if not m.ENABLE_EMBEDDINGS:
            return {
                "ok": True,
                "enabled": False,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": m.OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }

        if total_nodes > 0 and embedded_nodes == total_nodes:
            dims = m._ensure_vector_index_for_repo(session, repo_id)
            response: dict[str, int | str | bool] = {
                "ok": True,
                "enabled": True,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": m.OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }
            if dims is not None:
                response["dimensions"] = dims
            return response

        rows = m._repo_nodes_for_embedding(session, repo_id)
        if not rows:
            return {
                "ok": True,
                "enabled": True,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": m.OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }

        text_rows = [{"id": row["id"], "text": m._embedding_text(row)} for row in rows]
        dims: int | None = None
        for idx in range(0, len(text_rows), m.EMBEDDING_BATCH_SIZE):
            batch = text_rows[idx : idx + m.EMBEDDING_BATCH_SIZE]
            embeddings = m._openai_embed([item["text"] for item in batch])
            if embeddings and dims is None:
                dims = len(embeddings[0])
            m._set_embeddings(
                session,
                repo_id,
                [{"id": batch[i]["id"], "embedding": embeddings[i]} for i in range(len(batch))],
            )

        if dims is None or dims <= 0:
            raise HTTPException(status_code=500, detail="failed to compute embedding dimensions")
        m._ensure_vector_index(session, dims)
        total_nodes, embedded_nodes = m._repo_embedding_counts(session, repo_id)

    return {
        "ok": True,
        "enabled": True,
        "embedded_nodes": embedded_nodes,
        "total_nodes": total_nodes,
        "skipped": False,
        "model": m.OPENAI_EMBED_MODEL,
        "dimensions": dims,
        "duration_ms": _duration_ms(),
    }


# ---------------------------------------------------------------------------
# /graph/embeddings/status
# ---------------------------------------------------------------------------
@router.get("/graph/embeddings/status", response_model=EmbeddingStatusResponse)
def graph_embedding_status(repo_id: str = Query(...)) -> EmbeddingStatusResponse:
    m = _m()
    repo = str(repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        record = session.run(
            """
            MATCH (n:CodeNode {repo_id: $repo_id})
            RETURN
              count(n) AS total_nodes,
              count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS embedded_nodes
            """,
            repo_id=repo,
        ).single()

    total_nodes = int(record["total_nodes"]) if record else 0
    embedded_nodes = int(record["embedded_nodes"]) if record else 0
    return EmbeddingStatusResponse(
        repo_id=repo,
        embeddings_exist=total_nodes > 0 and embedded_nodes > 0,
        embedded_nodes=embedded_nodes,
    )


# ---------------------------------------------------------------------------
# /graph/repo/status
# ---------------------------------------------------------------------------
@router.get("/graph/repo/status", response_model=RepoStatusResponse)
def graph_repo_status(repo_id: str = Query(...)) -> RepoStatusResponse:
    m = _m()
    repo = str(repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        node_record = session.run(
            """
            MATCH (n:CodeNode {repo_id: $repo_id})
            RETURN
              count(n) AS indexed_node_count,
              count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS embedded_nodes
            """,
            repo_id=repo,
        ).single()
        edge_record = session.run(
            """
            MATCH (:CodeNode {repo_id: $repo_id})-[r:CONTAINS|IMPORTS|CALLS]->(:CodeNode {repo_id: $repo_id})
            RETURN count(r) AS indexed_edge_count
            """,
            repo_id=repo,
        ).single()

    indexed_node_count = int(node_record["indexed_node_count"]) if node_record else 0
    embedded_nodes = int(node_record["embedded_nodes"]) if node_record else 0
    indexed_edge_count = int(edge_record["indexed_edge_count"]) if edge_record else 0
    return RepoStatusResponse(
        repo_id=repo,
        indexed_node_count=indexed_node_count,
        indexed_edge_count=indexed_edge_count,
        embedded_nodes=embedded_nodes,
        embeddings_exist=indexed_node_count > 0 and embedded_nodes > 0,
    )


# ---------------------------------------------------------------------------
# /graph/search/fulltext
# ---------------------------------------------------------------------------
@router.post("/graph/search/fulltext", response_model=SearchResponse)
def graph_search_fulltext(payload: GraphSearchRequest) -> SearchResponse:
    m = _m()
    repo_id = str(payload.repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        try:
            m._ensure_schema(session)
        except Neo4jError as exc:
            detail = m._neo4j_error_summary(exc)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to ensure fulltext index before search. "
                    "Verify Neo4j permissions for CREATE INDEX and connectivity. "
                    f"Detail: {detail}"
                ),
            ) from exc

        try:
            hits = m._run_fulltext_query(
                session,
                repo_id=repo_id,
                query_text=payload.query,
                top_k=payload.top_k,
            )
        except Neo4jError as exc:
            if m._is_missing_fulltext_index_error(exc):
                try:
                    m._ensure_schema(session)
                    hits = m._run_fulltext_query(
                        session,
                        repo_id=repo_id,
                        query_text=payload.query,
                        top_k=payload.top_k,
                    )
                except Neo4jError as retry_exc:
                    detail = m._neo4j_error_summary(retry_exc)
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Fulltext index '{m.FULLTEXT_INDEX_NAME}' is missing or unavailable after re-creation attempt. "
                            "Ensure Neo4j can create indexes and retry. "
                            f"Detail: {detail}"
                        ),
                    ) from retry_exc
            else:
                detail = m._neo4j_error_summary(exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Fulltext search failed on index '{m.FULLTEXT_INDEX_NAME}'. "
                        f"Detail: {detail}"
                    ),
                ) from exc
    return SearchResponse(hits=hits)


# ---------------------------------------------------------------------------
# /graph/search/vector
# ---------------------------------------------------------------------------
@router.post("/graph/search/vector", response_model=SearchResponse)
def graph_search_vector(payload: GraphVectorSearchRequest) -> SearchResponse:
    m = _m()
    repo_id = str(payload.repo_id)
    if not payload.embedding:
        return SearchResponse(hits=[])

    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        dims = m._ensure_vector_index_for_repo(session, repo_id)
        try:
            result = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                WHERE node.repo_id = $repo_id
                RETURN node, score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                index_name=m.VECTOR_INDEX_NAME,
                repo_id=repo_id,
                embedding=payload.embedding,
                top_k=payload.top_k,
            )
            hits = [SearchHit(node=m._node_to_payload(row["node"]), score=float(row["score"])) for row in result]
        except Neo4jError:
            if dims is None:
                hits = []
            else:
                fallback = session.run(
                    """
                    MATCH (node:CodeNode {repo_id: $repo_id})
                    WHERE node.embedding IS NOT NULL AND size(node.embedding) = $dims
                    WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """,
                    repo_id=repo_id,
                    embedding=payload.embedding,
                    dims=dims,
                    top_k=payload.top_k,
                )
                hits = [SearchHit(node=m._node_to_payload(row["node"]), score=float(row["score"])) for row in fallback]

    return SearchResponse(hits=hits)


# ---------------------------------------------------------------------------
# /graph/search/default
# ---------------------------------------------------------------------------
@router.post("/graph/search/default", response_model=SearchResponse)
def graph_search_default(payload: GraphRepoDefaultSearchRequest) -> SearchResponse:
    m = _m()
    repo_id = str(payload.repo_id)

    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH (node:CodeNode {repo_id: $repo_id})
            WITH
              node,
              CASE node.type
                WHEN 'file' THEN 4.0
                WHEN 'function' THEN 3.0
                WHEN 'class' THEN 2.0
                WHEN 'module' THEN 1.5
                ELSE 1.0
              END AS type_score,
              CASE
                WHEN node.code_snippet IS NOT NULL AND node.code_snippet <> '' THEN 1.0
                ELSE 0.0
              END AS snippet_score
            RETURN node, (type_score + snippet_score) AS score
            ORDER BY score DESC, node.path ASC, node.name ASC
            LIMIT $top_k
            """,
            repo_id=repo_id,
            top_k=payload.top_k,
        )
        hits = [SearchHit(node=m._node_to_payload(row["node"]), score=float(row["score"])) for row in result]

    return SearchResponse(hits=hits)


# ---------------------------------------------------------------------------
# /graph/expand
# ---------------------------------------------------------------------------
@router.post("/graph/expand", response_model=ExpandResponse)
def graph_expand(payload: GraphExpandRequest) -> ExpandResponse:
    m = _m()
    repo_id = str(payload.repo_id)
    unique_ids = sorted({node_id for node_id in payload.node_ids if node_id})
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        return m._expand(session, repo_id, unique_ids, payload.hops)


# ---------------------------------------------------------------------------
# /graph/subgraph
# ---------------------------------------------------------------------------
@router.get("/graph/subgraph", response_model=SubgraphResponse)
def graph_subgraph(
    repo_id: str = Query(...),
    node_id: str = Query(...),
    hops: int = Query(1, ge=1, le=4),
) -> SubgraphResponse:
    m = _m()
    repo = str(repo_id)
    with m._require_driver().session(database=m.NEO4J_DATABASE) as session:
        root_exists = session.run(
            "MATCH (n:CodeNode {repo_id: $repo_id, id: $node_id}) RETURN n LIMIT 1",
            repo_id=repo,
            node_id=node_id,
        ).single()
        if root_exists is None:
            raise HTTPException(status_code=404, detail="node not found")
        expanded = m._expand(session, repo, [node_id], hops)

    return SubgraphResponse(nodes=expanded.nodes, edges=expanded.edges)