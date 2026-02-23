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

from fastapi import FastAPI, HTTPException, Response
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from app.api import router as graph_router
from app.kg_api import router as kg_router

from app.models import (
    ExpandResponse,
    GraphEdge,
    GraphNode,
    SearchHit,
)
from app.kg_store import (
    ensure_kg_schema,
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

app.include_router(graph_router)
app.include_router(kg_router)

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



