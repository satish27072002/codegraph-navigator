"""
Hybrid retriever.
Combines Neo4j vector search (semantic similarity) and full-text search
(keyword matching). Results are merged using Reciprocal Rank Fusion (RRF)
and returned ranked by combined relevance score.
"""

import asyncio
import logging
from db.neo4j_client import run_query

logger = logging.getLogger(__name__)

# RRF constant — controls the balance between rank position and score
_RRF_K = 60


# ─────────────────────────────────────────────
# Vector search
# ─────────────────────────────────────────────

_VECTOR_SEARCH_CYPHER = """
CALL db.index.vector.queryNodes('function_embeddings', $top_k, $embedding)
YIELD node AS f, score
WHERE f.codebase_id = $codebase_id
RETURN
    f.id          AS id,
    f.name        AS name,
    f.file        AS file,
    f.start_line  AS start_line,
    f.end_line    AS end_line,
    f.code        AS code,
    f.docstring   AS docstring,
    f.complexity  AS complexity,
    score         AS vector_score
ORDER BY score DESC
LIMIT $top_k
"""


async def _vector_search(
    embedding: list[float],
    codebase_id: str,
    top_k: int,
) -> list[dict]:
    try:
        rows = await run_query(
            _VECTOR_SEARCH_CYPHER,
            params={
                "embedding": embedding,
                "codebase_id": codebase_id,
                "top_k": top_k * 2,  # fetch more than needed for RRF merging
            },
        )
        return rows
    except Exception as exc:
        logger.warning(f"Vector search failed: {exc}")
        return []


# ─────────────────────────────────────────────
# Full-text search
# ─────────────────────────────────────────────

_FULLTEXT_SEARCH_CYPHER = """
CALL db.index.fulltext.queryNodes('function_text', $query)
YIELD node AS f, score
WHERE f.codebase_id = $codebase_id
RETURN
    f.id          AS id,
    f.name        AS name,
    f.file        AS file,
    f.start_line  AS start_line,
    f.end_line    AS end_line,
    f.code        AS code,
    f.docstring   AS docstring,
    f.complexity  AS complexity,
    score         AS fulltext_score
ORDER BY score DESC
LIMIT $top_k
"""


async def _fulltext_search(
    query_text: str,
    codebase_id: str,
    top_k: int,
) -> list[dict]:
    # Escape Lucene special characters to avoid query parse errors
    escaped = _escape_lucene(query_text)
    try:
        rows = await run_query(
            _FULLTEXT_SEARCH_CYPHER,
            params={
                "query": escaped,
                "codebase_id": codebase_id,
                "top_k": top_k * 2,
            },
        )
        return rows
    except Exception as exc:
        logger.warning(f"Full-text search failed: {exc}")
        return []


def _escape_lucene(text: str) -> str:
    """Escape special Lucene query syntax characters."""
    specials = r'+-&&||!(){}[]^"~*?:\/'
    escaped = ""
    for ch in text:
        if ch in specials:
            escaped += "\\" + ch
        else:
            escaped += ch
    return escaped


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def _rrf_merge(
    vector_results: list[dict],
    fulltext_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank_in_list1) + 1/(k + rank_in_list2)
    """
    scores: dict[str, float] = {}
    node_data: dict[str, dict] = {}

    for rank, row in enumerate(vector_results, start=1):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (_RRF_K + rank)
        if node_id not in node_data:
            node_data[node_id] = row

    for rank, row in enumerate(fulltext_results, start=1):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (_RRF_K + rank)
        if node_id not in node_data:
            node_data[node_id] = row

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for node_id, rrf_score in ranked:
        row = dict(node_data[node_id])
        row["relevance_score"] = round(rrf_score, 4)
        results.append(row)

    return results


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

async def hybrid_search(
    question_embedding: list[float],
    question_text: str,
    codebase_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Run vector + full-text search in parallel, merge via RRF.
    Returns a list of node dicts ranked by combined relevance_score.

    Each result dict contains:
      id, name, file, start_line, end_line, code, docstring, complexity, relevance_score
    """
    vector_task = _vector_search(question_embedding, codebase_id, top_k)
    fulltext_task = _fulltext_search(question_text, codebase_id, top_k)

    vector_results, fulltext_results = await asyncio.gather(vector_task, fulltext_task)

    merged = _rrf_merge(vector_results, fulltext_results, top_k)

    retrieval_desc = []
    if vector_results:
        retrieval_desc.append("vector")
    if fulltext_results:
        retrieval_desc.append("full-text")

    logger.info(
        f"Hybrid search: {len(vector_results)} vector, {len(fulltext_results)} fulltext "
        f"-> {len(merged)} merged results"
    )

    return merged
