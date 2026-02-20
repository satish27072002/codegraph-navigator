"""
POST /query — Full pipeline:
  1. Classify question (structural vs semantic)
  2a. Structural -> text2Cypher -> execute Cypher -> format results as context
  2b. Semantic   -> embed question -> hybrid search -> graph expansion -> parent context
  3. Assemble context string
  4. Generate answer with gpt-4o
  5. Return QueryResponse with answer, sources, graph, retrieval_method, cypher_used
"""

import logging
import time
from fastapi import APIRouter, HTTPException

from models.schemas import QueryRequest, QueryResponse, SourceReference, GraphData, GraphNode, GraphEdge
from services.embeddings.embedder import embed_text
from services.retrieval.hybrid_retriever import hybrid_search
from services.retrieval.graph_expander import expand_graph
from services.retrieval.parent_retriever import get_rich_context
from services.retrieval.text2cypher import is_structural_question, answer_structural_question
from services.llm.query_engine import assemble_context, assemble_cypher_context, generate_answer
from db.neo4j_client import run_query as neo4j_run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


def _build_sources(search_results: list[dict]) -> list[SourceReference]:
    """Convert hybrid search result dicts into SourceReference objects."""
    sources = []
    for r in search_results:
        sources.append(SourceReference(
            name=r.get("name") or r.get("id") or "",
            file=r.get("file") or "",
            start_line=r.get("start_line") or 0,
            end_line=r.get("end_line") or 0,
            code=r.get("code") or "",
            relevance_score=float(r.get("relevance_score") or 0.0),
        ))
    return sources


def _build_graph_data(graph_dict: dict) -> GraphData:
    """Convert raw graph dict into typed GraphData."""
    nodes = [
        GraphNode(
            id=n["id"],
            type=n.get("type", "Function"),
            name=n.get("name", ""),
            file=n.get("file", ""),
            highlighted=n.get("highlighted", False),
        )
        for n in graph_dict.get("nodes", [])
    ]
    edges = [
        GraphEdge(
            id=e["id"],
            source=e["source"],
            target=e["target"],
            type=e.get("type", "CALLS"),
        )
        for e in graph_dict.get("edges", [])
    ]
    return GraphData(nodes=nodes, edges=edges)


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a natural language question about the ingested codebase.
    Routes structural questions through text2Cypher,
    semantic questions through hybrid search + graph expansion.
    """
    t_start = time.monotonic()
    logger.info(f"Query: '{request.question}' | codebase={request.codebase_id}")

    cypher_used: str | None = None
    retrieval_method = "hybrid + graph expansion"

    # ── Route: structural vs semantic ──────────────────────────────────
    if is_structural_question(request.question):
        # ── Path A: text2Cypher ────────────────────────────────────────
        retrieval_method = "text2cypher"
        try:
            t2c = await answer_structural_question(request.question, request.codebase_id)
            cypher_used = t2c.get("cypher") or None
            cypher_results = t2c.get("results") or []
            context = assemble_cypher_context(cypher_results, cypher_used or "")
            answer = await generate_answer(request.question, context)

            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(f"text2Cypher answered in {elapsed_ms}ms")

            # Build graph from Cypher result node names
            graph_dict: dict = {"nodes": [], "edges": []}
            try:
                # Extract string values from Cypher results (column names vary by query)
                result_names: list[str] = []
                for r in (cypher_results or []):
                    for val in r.values():
                        if isinstance(val, str) and val:
                            result_names.append(val)

                if result_names:
                    records = await neo4j_run_query(
                        """
                        MATCH (f:Function)
                        WHERE f.name IN $names AND f.codebase_id = $codebase_id
                        RETURN f.id AS id
                        LIMIT 20
                        """,
                        {"names": result_names, "codebase_id": request.codebase_id},
                    )
                    seed_ids = [r["id"] for r in records if r.get("id")]
                    if seed_ids:
                        graph_dict = await expand_graph(seed_ids, hops=request.hops)
                        logger.info(f"text2Cypher graph: {len(graph_dict.get('nodes', []))} nodes")
            except Exception as exc:
                logger.warning(f"text2Cypher graph build failed: {exc}")

            return QueryResponse(
                answer=answer,
                sources=[],
                graph=_build_graph_data(graph_dict),
                retrieval_method=retrieval_method,
                cypher_used=cypher_used,
            )
        except Exception as exc:
            logger.warning(f"text2Cypher failed, falling back to hybrid: {exc}")
            retrieval_method = "hybrid + graph expansion (text2cypher fallback)"

    # ── Path B: hybrid search + graph expansion ─────────────────────────
    try:
        # 1. Embed the question
        question_embedding = await embed_text(request.question)
    except Exception as exc:
        logger.warning(f"Embedding failed: {exc} — using empty vector")
        question_embedding = []

    # 2. Hybrid search
    try:
        search_results = await hybrid_search(
            question_embedding=question_embedding,
            question_text=request.question,
            codebase_id=request.codebase_id,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error(f"Hybrid search failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")

    if not search_results:
        return QueryResponse(
            answer=(
                "I couldn't find relevant code for your question. "
                "Make sure the codebase has been ingested via POST /ingest."
            ),
            sources=[],
            graph=GraphData(),
            retrieval_method=retrieval_method,
            cypher_used=None,
        )

    # 3. Graph expansion from seed nodes
    seed_ids = [r["id"] for r in search_results]
    try:
        graph_dict = await expand_graph(seed_ids, hops=request.hops)
    except Exception as exc:
        logger.warning(f"Graph expansion failed: {exc}")
        graph_dict = {"nodes": [], "edges": []}

    # 4. Enrich sources with parent context
    try:
        enriched_sources = await get_rich_context(search_results)
    except Exception as exc:
        logger.warning(f"Parent context enrichment failed: {exc}")
        enriched_sources = search_results

    # 5. Assemble context + generate answer
    context = assemble_context(enriched_sources)
    try:
        answer = await generate_answer(request.question, context)
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        f"Query answered in {elapsed_ms}ms | "
        f"{len(search_results)} sources | "
        f"{len(graph_dict.get('nodes', []))} graph nodes"
    )

    if search_results and graph_dict.get("nodes"):
        retrieval_method = "hybrid + graph expansion"
    elif search_results:
        retrieval_method = "hybrid"

    return QueryResponse(
        answer=answer,
        sources=_build_sources(enriched_sources),
        graph=_build_graph_data(graph_dict),
        retrieval_method=retrieval_method,
        cypher_used=cypher_used,
    )
