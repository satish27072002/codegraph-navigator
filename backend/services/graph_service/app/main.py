from __future__ import annotations

import json
import logging
import os
import random
import re
import socket
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import Body, FastAPI, HTTPException, Query, Response
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field, ValidationError

from .kg_store import (
    ensure_kg_schema,
    ensure_kg_vector_indexes,
    fetch_subgraph as fetch_kg_subgraph,
    get_kg_status,
    set_chunk_embeddings,
    set_entity_embeddings,
    upsert_chunks,
    upsert_entities,
    upsert_mentions,
    upsert_relations,
    upsert_repo_documents,
)


NEO4J_URL = os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI") or "bolt://neo4j:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_EMBED_TIMEOUT_SEC = int(os.getenv("OPENAI_EMBED_TIMEOUT_SEC", os.getenv("OPENAI_TIMEOUT_SEC", "30")))
OPENAI_EMBED_MAX_RETRIES = max(1, int(os.getenv("OPENAI_EMBED_MAX_RETRIES", "8")))
OPENAI_EMBED_BACKOFF_BASE_SEC = float(os.getenv("OPENAI_EMBED_BACKOFF_BASE_SEC", "0.5"))
OPENAI_EMBED_BACKOFF_MAX_SEC = float(os.getenv("OPENAI_EMBED_BACKOFF_MAX_SEC", "10"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
EMBEDDING_MAX_TEXT_CHARS = int(os.getenv("EMBEDDING_MAX_TEXT_CHARS", "12000"))
ENABLE_EMBEDDINGS = os.getenv("ENABLE_EMBEDDINGS", "true").lower() not in {"0", "false", "no", "off"}
DEBUG_ENV = os.getenv("DEBUG_ENV", "false").lower() in {"1", "true", "yes", "on"}
EMBEDDING_TEXT_FIELDS = [
    field.strip()
    for field in os.getenv("EMBEDDING_TEXT_FIELDS", "name,path,code_snippet").split(",")
    if field.strip()
]
OPENAI_EMBEDDING_DIMENSIONS = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8004")
LLM_SERVICE_TIMEOUT_SEC = int(os.getenv("LLM_SERVICE_TIMEOUT_SEC", "60"))
KG_CHUNK_SIZE_CHARS = max(200, int(os.getenv("KG_CHUNK_SIZE_CHARS", "1800")))
KG_CHUNK_OVERLAP_CHARS = max(0, int(os.getenv("KG_CHUNK_OVERLAP_CHARS", "200")))
KG_EXTRACT_RETRIES = max(1, int(os.getenv("KG_EXTRACT_RETRIES", "2")))

REL_MAP = {
    "contains": "CONTAINS",
    "imports": "IMPORTS",
    "calls": "CALLS",
}
VECTOR_INDEX_NAME = "code_node_embedding"
FULLTEXT_INDEX_NAME = "code_node_fulltext"
OPENAI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

app = FastAPI(title="graph_service")
driver: Driver | None = None
logger = logging.getLogger("graph_service")
STARTUP_CONFIG_ERROR: str | None = None


class GraphLoadRequest(BaseModel):
    repo_id: uuid.UUID
    facts_path: str


class GraphEmbedRequest(BaseModel):
    repo_id: uuid.UUID


class GraphSearchRequest(BaseModel):
    repo_id: uuid.UUID
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class GraphRepoDefaultSearchRequest(BaseModel):
    repo_id: uuid.UUID
    top_k: int = Field(default=10, ge=1, le=100)


class GraphVectorSearchRequest(BaseModel):
    repo_id: uuid.UUID
    embedding: list[float]
    top_k: int = Field(default=10, ge=1, le=100)


class GraphExpandRequest(BaseModel):
    repo_id: uuid.UUID
    node_ids: list[str]
    hops: int = Field(default=1, ge=1, le=3)


class GraphNode(BaseModel):
    id: str
    repo_id: str
    type: str
    name: str
    path: str
    code_snippet: str


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class SearchHit(BaseModel):
    node: GraphNode
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ExpandResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class EmbeddingStatusResponse(BaseModel):
    repo_id: str
    embeddings_exist: bool
    embedded_nodes: int


class RepoStatusResponse(BaseModel):
    repo_id: str
    indexed_node_count: int
    indexed_edge_count: int
    embedded_nodes: int
    embeddings_exist: bool


class SubgraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class KGDocumentInput(BaseModel):
    path: str
    language: str = ""
    text: str = Field(min_length=1)


class KGLoadRequest(BaseModel):
    repo_id: uuid.UUID
    documents: list[KGDocumentInput]


class KGLoadResponse(BaseModel):
    docs: int
    chunks: int
    entities: int
    relations: int


class KGStatusResponse(BaseModel):
    repo_id: str
    docs: int
    chunks: int
    entities: int
    relations: int
    embedded_chunks: int
    embedded_entities: int


class KGSubgraphRequest(BaseModel):
    repo_id: uuid.UUID
    entity_names: list[str]
    hops: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=100, ge=1, le=1000)


class KGSubgraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def _require_driver() -> Driver:
    if driver is None:
        raise HTTPException(status_code=500, detail="neo4j driver not initialized")
    return driver


def _validate_embedding_startup_config() -> str | None:
    if not ENABLE_EMBEDDINGS:
        return None

    api_key = OPENAI_API_KEY.strip()
    if not api_key:
        logger.error("startup config invalid: ENABLE_EMBEDDINGS=true but OPENAI_API_KEY is missing/empty")
        return "ENABLE_EMBEDDINGS=true but OPENAI_API_KEY is missing/empty"

    if len(api_key) < 10:
        logger.error(
            "startup config invalid: OPENAI_API_KEY is too short",
            extra={"openai_api_key_present": True, "openai_api_key_length": len(api_key)},
        )
        return "ENABLE_EMBEDDINGS=true but OPENAI_API_KEY is too short"

    return None


def _ensure_embedding_config_ready() -> None:
    if STARTUP_CONFIG_ERROR:
        raise HTTPException(status_code=503, detail=f"embedding startup config invalid: {STARTUP_CONFIG_ERROR}")


def _load_facts(facts_path: Path) -> dict:
    if not facts_path.exists():
        raise HTTPException(status_code=400, detail=f"facts file not found: {facts_path}")
    try:
        payload = json.loads(facts_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid facts json: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="facts payload must be a json object")
    return payload


def _node_to_payload(node) -> GraphNode:
    return GraphNode(
        id=str(node.get("id", "")),
        repo_id=str(node.get("repo_id", "")),
        type=str(node.get("type", "")),
        name=str(node.get("name", "")),
        path=str(node.get("path", "")),
        code_snippet=str(node.get("code_snippet", "")),
    )


def _edges_to_payloads(rels) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for rel in rels:
        if isinstance(rel, dict):
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            rel_type = str(rel.get("type", "")).lower()
        else:
            source = str(rel.start_node.get("id", ""))
            target = str(rel.end_node.get("id", ""))
            rel_type = str(rel.type).lower()

        if not source or not target:
            continue
        edges.append(GraphEdge(source=source, target=target, type=rel_type or "related_to"))
    return edges


def _ensure_schema(session) -> None:
    session.run(
        "CREATE CONSTRAINT code_node_repo_id_id IF NOT EXISTS "
        "FOR (n:CodeNode) REQUIRE (n.repo_id, n.id) IS UNIQUE"
    ).consume()
    session.run(
        "CREATE INDEX code_node_repo_id IF NOT EXISTS "
        "FOR (n:CodeNode) ON (n.repo_id)"
    ).consume()
    session.run(
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
        "FOR (n:CodeNode) ON EACH [n.name, n.path, n.code_snippet]"
    ).consume()


def _ensure_vector_index(session, dimensions: int) -> None:
    if dimensions <= 0:
        raise HTTPException(status_code=400, detail="embedding dimensions must be > 0")
    session.run(
        f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
        "FOR (n:CodeNode) ON (n.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
    ).consume()


def _neo4j_error_summary(exc: Exception) -> str:
    code = getattr(exc, "code", "")
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 400:
        message = f"{message[:397]}..."
    if code:
        return f"{code}: {message}"
    return message or exc.__class__.__name__


def _is_missing_fulltext_index_error(exc: Exception) -> bool:
    haystack = f"{getattr(exc, 'code', '')} {exc}".lower()
    if FULLTEXT_INDEX_NAME.lower() not in haystack:
        return False
    missing_markers = (
        "not found",
        "no such",
        "does not exist",
        "unknown index",
        "index not found",
    )
    return any(marker in haystack for marker in missing_markers)


_LUCENE_SPECIAL_CHARS = re.compile(r'([+\-!(){}\[\]^"~*?:\\/|&])')


def _normalize_fulltext_query(query_text: str) -> str:
    # Escape Lucene-reserved characters so user input like "src/sample" won't crash parsing.
    compact = " ".join((query_text or "").split()).strip()
    if not compact:
        return ""
    return _LUCENE_SPECIAL_CHARS.sub(r"\\\1", compact)


def _run_fulltext_query(session, *, repo_id: str, query_text: str, top_k: int) -> list[SearchHit]:
    normalized_query = _normalize_fulltext_query(query_text)
    if not normalized_query:
        return []

    result = session.run(
        """
        CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $top_k})
        YIELD node, score
        WHERE node.repo_id = $repo_id
        RETURN node, score
        ORDER BY score DESC
        LIMIT $top_k
        """,
        index_name=FULLTEXT_INDEX_NAME,
        repo_id=repo_id,
        query_text=normalized_query,
        top_k=top_k,
    )
    return [SearchHit(node=_node_to_payload(row["node"]), score=float(row["score"])) for row in result]


def _upsert_nodes(session, repo_id: str, nodes: list[dict]) -> None:
    if not nodes:
        return
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:CodeNode {repo_id: $repo_id, id: node.id})
        SET
          n.type = coalesce(node.type, ""),
          n.name = coalesce(node.name, ""),
          n.path = coalesce(node.path, ""),
          n.code_snippet = coalesce(node.code_snippet, "")
        """,
        repo_id=repo_id,
        nodes=nodes,
    ).consume()


def _upsert_edges(session, repo_id: str, edges: list[dict], rel_type: str) -> None:
    if not edges:
        return
    query = f"""
        UNWIND $edges AS edge
        MATCH (src:CodeNode {{repo_id: $repo_id, id: edge.source}})
        MATCH (dst:CodeNode {{repo_id: $repo_id, id: edge.target}})
        MERGE (src)-[r:{rel_type}]->(dst)
    """
    session.run(query, repo_id=repo_id, edges=edges).consume()


def _embedding_text(row: dict) -> str:
    parts: list[str] = []
    for field in EMBEDDING_TEXT_FIELDS:
        value = row.get(field)
        if value is not None:
            text = str(value).strip()
            if text:
                parts.append(text)
    if not parts:
        parts.append(str(row.get("id", "")))
    return "\n".join(parts)[:EMBEDDING_MAX_TEXT_CHARS]


def _openai_embed(inputs: list[str]) -> list[list[float]]:
    if not inputs:
        return []
    _ensure_embedding_config_ready()
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for embedding operations.",
        )

    payload: dict[str, object] = {
        "model": OPENAI_EMBED_MODEL,
        "input": inputs,
    }
    if OPENAI_EMBEDDING_DIMENSIONS and OPENAI_EMBED_MODEL.startswith("text-embedding-3"):
        payload["dimensions"] = int(OPENAI_EMBEDDING_DIMENSIONS)

    payload_bytes = json.dumps(payload).encode("utf-8")
    data: dict | None = None
    last_error: str = "unknown embedding error"
    attempt = 0

    while attempt < OPENAI_EMBED_MAX_RETRIES:
        attempt += 1
        req = urlrequest.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=OPENAI_EMBED_TIMEOUT_SEC) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            break
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            status = int(exc.code)
            last_error = f"http {status}: {detail or 'no response body'}"

            if 400 <= status < 500 and status != 429:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "OpenAI embedding request rejected with non-retryable client error "
                        f"{status}: {detail or 'no response body'}"
                    ),
                ) from exc

            if status not in OPENAI_RETRYABLE_STATUS_CODES or attempt >= OPENAI_EMBED_MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "OpenAI embedding failed after "
                        f"{attempt} attempt(s); last error {status}: {detail or 'no response body'}"
                    ),
                ) from exc

            capped = min(
                OPENAI_EMBED_BACKOFF_MAX_SEC,
                OPENAI_EMBED_BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
            )
            sleep_sec = random.uniform(0.0, max(0.0, capped))
            logger.warning(
                "openai.embed.retry attempt=%s/%s status=%s sleep_sec=%.2f reason=%s",
                attempt,
                OPENAI_EMBED_MAX_RETRIES,
                status,
                sleep_sec,
                detail or "no response body",
            )
            time.sleep(sleep_sec)
        except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            reason_text = str(reason).strip() or exc.__class__.__name__
            last_error = f"network: {reason_text}"

            if attempt >= OPENAI_EMBED_MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "OpenAI embedding unavailable after "
                        f"{attempt} attempt(s): {reason_text}"
                    ),
                ) from exc

            capped = min(
                OPENAI_EMBED_BACKOFF_MAX_SEC,
                OPENAI_EMBED_BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
            )
            sleep_sec = random.uniform(0.0, max(0.0, capped))
            logger.warning(
                "openai.embed.retry attempt=%s/%s network_error=%s sleep_sec=%.2f",
                attempt,
                OPENAI_EMBED_MAX_RETRIES,
                reason_text,
                sleep_sec,
            )
            time.sleep(sleep_sec)
        except json.JSONDecodeError as exc:
            last_error = f"invalid json from OpenAI: {exc}"
            if attempt >= OPENAI_EMBED_MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "OpenAI embedding returned invalid JSON after "
                        f"{attempt} attempt(s): {exc}"
                    ),
                ) from exc
            capped = min(
                OPENAI_EMBED_BACKOFF_MAX_SEC,
                OPENAI_EMBED_BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
            )
            sleep_sec = random.uniform(0.0, max(0.0, capped))
            logger.warning(
                "openai.embed.retry attempt=%s/%s parse_error=%s sleep_sec=%.2f",
                attempt,
                OPENAI_EMBED_MAX_RETRIES,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    if data is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenAI embedding failed after "
                f"{OPENAI_EMBED_MAX_RETRIES} attempt(s): {last_error}"
            ),
        )

    if not isinstance(data, dict) or "data" not in data:
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI embeddings API")

    items = sorted(data["data"], key=lambda item: item.get("index", 0))
    embeddings = [item.get("embedding", []) for item in items]
    if len(embeddings) != len(inputs):
        raise HTTPException(status_code=502, detail="OpenAI embedding response size mismatch")
    return embeddings


def _repo_nodes_for_embedding(session, repo_id: str) -> list[dict]:
    result = session.run(
        """
        MATCH (n:CodeNode {repo_id: $repo_id})
        WHERE n.embedding IS NULL
        RETURN n.id AS id, n.name AS name, n.path AS path, n.code_snippet AS code_snippet
        ORDER BY n.id
        """,
        repo_id=repo_id,
    )
    return [dict(record) for record in result]


def _repo_embedding_counts(session, repo_id: str) -> tuple[int, int]:
    record = session.run(
        """
        MATCH (n:CodeNode {repo_id: $repo_id})
        RETURN
          count(n) AS total_nodes,
          count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS embedded_nodes
        """,
        repo_id=repo_id,
    ).single()
    total_nodes = int(record["total_nodes"]) if record else 0
    embedded_nodes = int(record["embedded_nodes"]) if record else 0
    return total_nodes, embedded_nodes


def _repo_embedding_dimensions(session, repo_id: str) -> int | None:
    record = session.run(
        """
        MATCH (n:CodeNode {repo_id: $repo_id})
        WHERE n.embedding IS NOT NULL
        RETURN size(n.embedding) AS dims
        LIMIT 1
        """,
        repo_id=repo_id,
    ).single()
    if not record:
        return None
    dims = record.get("dims")
    if dims is None:
        return None
    value = int(dims)
    return value if value > 0 else None


def _ensure_vector_index_for_repo(session, repo_id: str) -> int | None:
    dims = _repo_embedding_dimensions(session, repo_id)
    if dims is None:
        return None
    _ensure_vector_index(session, dims)
    return dims


def _set_embeddings(session, repo_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MATCH (n:CodeNode {repo_id: $repo_id, id: row.id})
        SET n.embedding = row.embedding
        """,
        repo_id=repo_id,
        rows=rows,
    ).consume()


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return sha256(joined.encode("utf-8")).hexdigest()


def _chunk_text(text: str) -> list[str]:
    content = text.strip()
    if not content:
        return []
    if len(content) <= KG_CHUNK_SIZE_CHARS:
        return [content]

    chunks: list[str] = []
    start = 0
    step = max(1, KG_CHUNK_SIZE_CHARS - KG_CHUNK_OVERLAP_CHARS)
    while start < len(content):
        end = min(len(content), start + KG_CHUNK_SIZE_CHARS)
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start += step
    return chunks


def _normalize_kg_extract_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relationships", [])
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    if isinstance(raw_entities, list):
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(raw.get("type", "unknown")).strip() or "unknown"
            entities.append({"name": name, "type": entity_type})

    if isinstance(raw_relations, list):
        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not source or not target:
                continue
            relation_type = str(raw.get("relation_type", "related_to")).strip() or "related_to"
            evidence = str(raw.get("evidence", "")).strip()
            confidence_raw = raw.get("confidence", 0.5)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(confidence, 1.0))
            relations.append(
                {
                    "source": source,
                    "target": target,
                    "relation_type": relation_type,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )
    return entities, relations


def _llm_extract_for_chunk(*, repo_id: str, path: str, chunk_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint = f"{LLM_SERVICE_URL.rstrip('/')}/kg/extract"
    payload = json.dumps(
        {
            "repo_id": repo_id,
            "path": path,
            "chunk_text": chunk_text,
        }
    ).encode("utf-8")

    last_error = "unknown extraction error"
    for attempt in range(1, KG_EXTRACT_RETRIES + 1):
        req = urlrequest.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=LLM_SERVICE_TIMEOUT_SEC) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("response is not a json object")
            entities, relations = _normalize_kg_extract_payload(parsed)
            return entities, relations
        except (urlerror.HTTPError, urlerror.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            if isinstance(exc, urlerror.HTTPError):
                detail = exc.read().decode("utf-8", errors="ignore")
                last_error = f"http {exc.code}: {detail or 'no response body'}"
            if attempt >= KG_EXTRACT_RETRIES:
                break
            time.sleep(min(1.0, 0.25 * attempt))

    logger.warning(
        "kg.extract.failed repo_id=%s path=%s attempts=%s detail=%s",
        repo_id,
        path,
        KG_EXTRACT_RETRIES,
        last_error,
    )
    return [], []


def _expand(session, repo_id: str, node_ids: list[str], hops: int) -> ExpandResponse:
    if not node_ids:
        return ExpandResponse(nodes=[], edges=[])

    nodes_query = f"""
        MATCH (seed:CodeNode {{repo_id: $repo_id}})
        WHERE seed.id IN $node_ids
        OPTIONAL MATCH (seed)-[:CONTAINS|IMPORTS|CALLS*0..{hops}]-(n:CodeNode {{repo_id: $repo_id}})
        WITH collect(DISTINCT seed) + collect(DISTINCT n) AS all_nodes
        UNWIND all_nodes AS node
        WITH DISTINCT node
        WHERE node IS NOT NULL
        RETURN collect(node) AS nodes
    """
    node_record = session.run(nodes_query, repo_id=repo_id, node_ids=node_ids).single()
    raw_nodes = node_record["nodes"] if node_record and node_record["nodes"] else []

    if not raw_nodes:
        return ExpandResponse(nodes=[], edges=[])

    expanded_ids = [str(node.get("id", "")) for node in raw_nodes]
    rel_record = session.run(
        """
        MATCH (a:CodeNode {repo_id: $repo_id})-[r:CONTAINS|IMPORTS|CALLS]->(b:CodeNode {repo_id: $repo_id})
        WHERE a.id IN $node_ids AND b.id IN $node_ids
        RETURN collect(DISTINCT {source: a.id, target: b.id, type: toLower(type(r))}) AS rels
        """,
        repo_id=repo_id,
        node_ids=expanded_ids,
    ).single()
    rels = rel_record["rels"] if rel_record and rel_record["rels"] else []

    return ExpandResponse(
        nodes=[_node_to_payload(node) for node in raw_nodes],
        edges=_edges_to_payloads(rels),
    )


@app.on_event("startup")
def startup() -> None:
    global driver, STARTUP_CONFIG_ERROR
    STARTUP_CONFIG_ERROR = _validate_embedding_startup_config()
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            _ensure_schema(session)
            ensure_kg_schema(session)
    except Neo4jError as exc:
        detail = _neo4j_error_summary(exc)
        logger.exception("neo4j.schema_init_failed", extra={"detail": detail})
        raise RuntimeError(
            "Failed to initialize Neo4j schema (constraints/indexes). "
            "Ensure Neo4j is reachable and the configured user can CREATE INDEX/CONSTRAINT. "
            f"Detail: {detail}"
        ) from exc


@app.on_event("shutdown")
def shutdown() -> None:
    global driver
    if driver is not None:
        driver.close()
        driver = None


@app.get("/health")
async def health(response: Response) -> dict[str, object]:
    ok = STARTUP_CONFIG_ERROR is None
    if not ok:
        response.status_code = 503
    return {"ok": ok, "config_error": STARTUP_CONFIG_ERROR}


@app.get("/debug/env")
def debug_env() -> dict[str, object]:
    if not DEBUG_ENV:
        raise HTTPException(status_code=404, detail="not found")

    api_key = OPENAI_API_KEY.strip()
    return {
        "ok": True,
        "enable_embeddings": ENABLE_EMBEDDINGS,
        "startup_config_ok": STARTUP_CONFIG_ERROR is None,
        "startup_config_error": STARTUP_CONFIG_ERROR,
        "openai_api_key_present": bool(api_key),
        "openai_api_key_length": len(api_key),
        "openai_api_key_length_valid": len(api_key) >= 10 if api_key else False,
        "openai_embed_model": OPENAI_EMBED_MODEL,
        "openai_embed_timeout_sec": OPENAI_EMBED_TIMEOUT_SEC,
        "openai_embed_max_retries": OPENAI_EMBED_MAX_RETRIES,
    }


@app.post("/graph/load")
def graph_load(payload: GraphLoadRequest) -> dict[str, int | bool]:
    facts = _load_facts(Path(payload.facts_path))
    if str(payload.repo_id) != str(facts.get("repo_id", "")):
        raise HTTPException(status_code=400, detail="repo_id does not match facts payload")

    nodes = facts.get("nodes", [])
    edges = facts.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise HTTPException(status_code=400, detail="facts nodes/edges must be arrays")

    repo_id = str(payload.repo_id)
    edges_by_type: dict[str, list[dict]] = {key: [] for key in REL_MAP}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        edge_type = str(edge.get("type", "")).lower()
        if edge_type in edges_by_type:
            edges_by_type[edge_type].append(edge)

    with _require_driver().session(database=NEO4J_DATABASE) as session:
        _ensure_schema(session)
        _upsert_nodes(session, repo_id, nodes)
        for edge_type, rel_name in REL_MAP.items():
            _upsert_edges(session, repo_id, edges_by_type[edge_type], rel_name)

    return {
        "ok": True,
        "nodes_upserted": len(nodes),
        "edges_upserted": sum(len(group) for group in edges_by_type.values()),
    }


@app.post("/kg/load", response_model=KGLoadResponse)
def kg_load(raw_payload: dict[str, Any] | None = Body(default=None)) -> KGLoadResponse:
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

    def _ensure_entity(name: str, entity_type: str) -> str:
        normalized_name = name.strip()
        normalized_type = entity_type.strip() or "unknown"
        if not normalized_name:
            return ""

        key = (normalized_name.lower(), normalized_type.lower())
        if key not in entity_by_key:
            entity_by_key[key] = {
                "entity_id": _stable_id(repo_id, normalized_name.lower(), normalized_type.lower()),
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
        doc_id = _stable_id(repo_id, path)
        docs.append(
            {
                "doc_id": doc_id,
                "path": path,
                "language": doc.language.strip(),
            }
        )

        doc_chunks = _chunk_text(doc.text)
        for chunk_index, chunk_text in enumerate(doc_chunks):
            chunk_id = _stable_id(repo_id, doc_id, str(chunk_index), chunk_text)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "repo_id": repo_id,
                    "doc_id": doc_id,
                    "index": chunk_index,
                    "text": chunk_text,
                }
            )

            extracted_entities, extracted_relations = _llm_extract_for_chunk(
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
                entity_id = _ensure_entity(name, entity_type)
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
                    source_entity_id = _ensure_entity(source_name, "unknown")
                if not target_entity_id:
                    target_entity_id = _ensure_entity(target_name, "unknown")
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

    with _require_driver().session(database=NEO4J_DATABASE) as session:
        ensure_kg_schema(session)
        upsert_repo_documents(session, repo_id, docs)
        upsert_chunks(session, repo_id, chunks)
        upsert_entities(session, repo_id, entities)
        upsert_mentions(session, repo_id, mention_rows)
        upsert_relations(session, repo_id, relation_rows)

        if ENABLE_EMBEDDINGS and chunks:
            chunk_dims: int | None = None
            for idx in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[idx : idx + EMBEDDING_BATCH_SIZE]
                vectors = _openai_embed([str(item.get("text", "")) for item in batch])
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
                for idx in range(0, len(entities), EMBEDDING_BATCH_SIZE):
                    batch = entities[idx : idx + EMBEDDING_BATCH_SIZE]
                    vectors = _openai_embed(
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


@app.get("/kg/status", response_model=KGStatusResponse)
def kg_status(repo_id: uuid.UUID = Query(...)) -> KGStatusResponse:
    repo = str(repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
        status = get_kg_status(session, repo)
    return KGStatusResponse(repo_id=repo, **status)


@app.post("/kg/subgraph", response_model=KGSubgraphResponse)
def kg_subgraph(payload: KGSubgraphRequest) -> KGSubgraphResponse:
    repo = str(payload.repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
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


@app.post("/graph/embed")
def graph_embed(payload: GraphEmbedRequest) -> dict[str, int | str | bool]:
    repo_id = str(payload.repo_id)
    started = time.perf_counter()

    def _duration_ms() -> int:
        return int(round((time.perf_counter() - started) * 1000))

    with _require_driver().session(database=NEO4J_DATABASE) as session:
        total_nodes, embedded_nodes = _repo_embedding_counts(session, repo_id)

        if not ENABLE_EMBEDDINGS:
            return {
                "ok": True,
                "enabled": False,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }

        # Idempotency short-circuit: do nothing when every node is already embedded.
        if total_nodes > 0 and embedded_nodes == total_nodes:
            dims = _ensure_vector_index_for_repo(session, repo_id)
            response: dict[str, int | str | bool] = {
                "ok": True,
                "enabled": True,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }
            if dims is not None:
                response["dimensions"] = dims
            return response

        rows = _repo_nodes_for_embedding(session, repo_id)
        if not rows:
            return {
                "ok": True,
                "enabled": True,
                "embedded_nodes": embedded_nodes,
                "total_nodes": total_nodes,
                "skipped": True,
                "model": OPENAI_EMBED_MODEL,
                "duration_ms": _duration_ms(),
            }

        text_rows = [{"id": row["id"], "text": _embedding_text(row)} for row in rows]
        dims: int | None = None
        for idx in range(0, len(text_rows), EMBEDDING_BATCH_SIZE):
            batch = text_rows[idx : idx + EMBEDDING_BATCH_SIZE]
            embeddings = _openai_embed([item["text"] for item in batch])
            if embeddings and dims is None:
                dims = len(embeddings[0])
            _set_embeddings(
                session,
                repo_id,
                [{"id": batch[i]["id"], "embedding": embeddings[i]} for i in range(len(batch))],
            )

        if dims is None or dims <= 0:
            raise HTTPException(status_code=500, detail="failed to compute embedding dimensions")
        _ensure_vector_index(session, dims)
        total_nodes, embedded_nodes = _repo_embedding_counts(session, repo_id)

    return {
        "ok": True,
        "enabled": True,
        "embedded_nodes": embedded_nodes,
        "total_nodes": total_nodes,
        "skipped": False,
        "model": OPENAI_EMBED_MODEL,
        "dimensions": dims,
        "duration_ms": _duration_ms(),
    }


@app.get("/graph/embeddings/status", response_model=EmbeddingStatusResponse)
def graph_embedding_status(repo_id: uuid.UUID = Query(...)) -> EmbeddingStatusResponse:
    repo = str(repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
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


@app.get("/graph/repo/status", response_model=RepoStatusResponse)
def graph_repo_status(repo_id: uuid.UUID = Query(...)) -> RepoStatusResponse:
    repo = str(repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
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


@app.post("/graph/search/fulltext", response_model=SearchResponse)
def graph_search_fulltext(payload: GraphSearchRequest) -> SearchResponse:
    repo_id = str(payload.repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
        try:
            _ensure_schema(session)
        except Neo4jError as exc:
            detail = _neo4j_error_summary(exc)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to ensure fulltext index before search. "
                    "Verify Neo4j permissions for CREATE INDEX and connectivity. "
                    f"Detail: {detail}"
                ),
            ) from exc

        try:
            hits = _run_fulltext_query(
                session,
                repo_id=repo_id,
                query_text=payload.query,
                top_k=payload.top_k,
            )
        except Neo4jError as exc:
            if _is_missing_fulltext_index_error(exc):
                try:
                    _ensure_schema(session)
                    hits = _run_fulltext_query(
                        session,
                        repo_id=repo_id,
                        query_text=payload.query,
                        top_k=payload.top_k,
                    )
                except Neo4jError as retry_exc:
                    detail = _neo4j_error_summary(retry_exc)
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Fulltext index '{FULLTEXT_INDEX_NAME}' is missing or unavailable after re-creation attempt. "
                            "Ensure Neo4j can create indexes and retry. "
                            f"Detail: {detail}"
                        ),
                    ) from retry_exc
            else:
                detail = _neo4j_error_summary(exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Fulltext search failed on index '{FULLTEXT_INDEX_NAME}'. "
                        f"Detail: {detail}"
                    ),
                ) from exc
    return SearchResponse(hits=hits)


@app.post("/graph/search/vector", response_model=SearchResponse)
def graph_search_vector(payload: GraphVectorSearchRequest) -> SearchResponse:
    repo_id = str(payload.repo_id)
    if not payload.embedding:
        return SearchResponse(hits=[])

    with _require_driver().session(database=NEO4J_DATABASE) as session:
        dims = _ensure_vector_index_for_repo(session, repo_id)
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
                index_name=VECTOR_INDEX_NAME,
                repo_id=repo_id,
                embedding=payload.embedding,
                top_k=payload.top_k,
            )
            hits = [SearchHit(node=_node_to_payload(row["node"]), score=float(row["score"])) for row in result]
        except Neo4jError:
            # Fallback when vector index isn't available yet: brute-force cosine on repo embeddings.
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
                hits = [SearchHit(node=_node_to_payload(row["node"]), score=float(row["score"])) for row in fallback]

    return SearchResponse(hits=hits)


@app.post("/graph/search/default", response_model=SearchResponse)
def graph_search_default(payload: GraphRepoDefaultSearchRequest) -> SearchResponse:
    repo_id = str(payload.repo_id)

    with _require_driver().session(database=NEO4J_DATABASE) as session:
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
        hits = [SearchHit(node=_node_to_payload(row["node"]), score=float(row["score"])) for row in result]

    return SearchResponse(hits=hits)


@app.post("/graph/expand", response_model=ExpandResponse)
def graph_expand(payload: GraphExpandRequest) -> ExpandResponse:
    repo_id = str(payload.repo_id)
    unique_ids = sorted({node_id for node_id in payload.node_ids if node_id})
    with _require_driver().session(database=NEO4J_DATABASE) as session:
        return _expand(session, repo_id, unique_ids, payload.hops)


@app.get("/graph/subgraph", response_model=SubgraphResponse)
def graph_subgraph(
    repo_id: uuid.UUID = Query(...),
    node_id: str = Query(...),
    hops: int = Query(1, ge=1, le=4),
) -> SubgraphResponse:
    repo = str(repo_id)
    with _require_driver().session(database=NEO4J_DATABASE) as session:
        root_exists = session.run(
            "MATCH (n:CodeNode {repo_id: $repo_id, id: $node_id}) RETURN n LIMIT 1",
            repo_id=repo,
            node_id=node_id,
        ).single()
        if root_exists is None:
            raise HTTPException(status_code=404, detail="node not found")
        expanded = _expand(session, repo, [node_id], hops)

    return SubgraphResponse(nodes=expanded.nodes, edges=expanded.edges)
