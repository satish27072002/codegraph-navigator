"""
Parent document retriever.
Embeds small child chunks for precise retrieval, but returns the
larger parent chunk (or the full function code) as context to the LLM.
This pattern gives dense retrieval (small chunks) + rich context (large chunks).
"""

import logging
from db.neo4j_client import run_query

logger = logging.getLogger(__name__)

# Child chunk size in characters
_CHILD_CHUNK_SIZE = 300
# Overlap between consecutive child chunks
_CHILD_CHUNK_OVERLAP = 50


def chunk_code(code: str, parent_id: str) -> list[dict]:
    """
    Split a code string into overlapping child chunks.
    Returns a list of chunk dicts, each containing:
      { text, index, parent_id, embedding: [] }
    """
    chunks: list[dict] = []
    start = 0
    index = 0
    while start < len(code):
        end = min(start + _CHILD_CHUNK_SIZE, len(code))
        chunk_text = code[start:end]
        chunks.append({
            "text": chunk_text,
            "index": index,
            "parent_id": parent_id,
            "embedding": [],
        })
        index += 1
        start += _CHILD_CHUNK_SIZE - _CHILD_CHUNK_OVERLAP
        if start >= len(code):
            break
    return chunks


_GET_PARENT_CYPHER = """
MATCH (child:Chunk)-[:HAS_PARENT]->(parent:Chunk)
WHERE child.id IN $child_ids
RETURN DISTINCT
    parent.id   AS id,
    parent.text AS text,
    parent.index AS index
"""

_GET_FUNCTION_CODE_CYPHER = """
MATCH (f:Function)
WHERE f.id IN $ids
RETURN
    f.id        AS id,
    f.name      AS name,
    f.file      AS file,
    f.start_line AS start_line,
    f.end_line  AS end_line,
    f.code      AS code,
    f.docstring AS docstring,
    f.complexity AS complexity
"""


async def get_parent_context(child_node_ids: list[str]) -> list[dict]:
    """
    Given child chunk IDs, fetch their parent chunks for richer LLM context.
    Falls back to the function nodes themselves if no Chunk hierarchy exists.
    """
    if not child_node_ids:
        return []

    try:
        parent_rows = await run_query(
            _GET_PARENT_CYPHER,
            params={"child_ids": child_node_ids},
        )
        if parent_rows:
            return parent_rows
    except Exception as exc:
        logger.debug(f"Parent chunk fetch skipped: {exc}")

    # Fallback: return the function nodes directly
    try:
        fn_rows = await run_query(
            _GET_FUNCTION_CODE_CYPHER,
            params={"ids": child_node_ids},
        )
        return fn_rows
    except Exception as exc:
        logger.warning(f"Function code fetch failed: {exc}")
        return []


async def get_rich_context(search_results: list[dict]) -> list[dict]:
    """
    Given hybrid search results (function node dicts), enrich each with
    the full parent context if available.

    For portfolio-level complexity: if the function code is short (<300 chars),
    return it as-is. If longer, try to fetch parent chunks for richer context.
    Returns the same list, potentially with enriched 'code' fields.
    """
    if not search_results:
        return []

    # Find functions with long code that may benefit from parent context
    ids_needing_parent = [
        r["id"] for r in search_results
        if len(r.get("code") or "") > _CHILD_CHUNK_SIZE
    ]

    if not ids_needing_parent:
        return search_results

    parent_rows = await get_parent_context(ids_needing_parent)
    parent_map = {row["id"]: row for row in parent_rows}

    enriched = []
    for result in search_results:
        if result["id"] in parent_map:
            enriched_result = dict(result)
            parent = parent_map[result["id"]]
            enriched_result["code"] = parent.get("text") or parent.get("code") or result.get("code", "")
            enriched.append(enriched_result)
        else:
            enriched.append(result)

    return enriched
