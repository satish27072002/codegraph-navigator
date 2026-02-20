"""
GET /graph/{node_id} — Return the immediate neighbourhood of a graph node
(the node itself plus all directly connected nodes and edges).
Used by the frontend when a user clicks a node to expand the graph canvas.
"""

import logging
from fastapi import APIRouter, HTTPException
from models.schemas import GraphResponse, GraphNode, GraphEdge
from services.retrieval.graph_expander import get_node_neighbourhood

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/{node_id}", response_model=GraphResponse)
async def get_node_graph(node_id: str) -> GraphResponse:
    """
    Return the node and its immediate neighbours (1 hop).
    Called when the user clicks a node in the React Flow canvas.
    """
    try:
        graph_dict = await get_node_neighbourhood(node_id)
    except Exception as exc:
        logger.error(f"Graph neighbourhood query failed for node '{node_id}': {exc}")
        raise HTTPException(status_code=500, detail=f"Graph query error: {exc}")

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

    return GraphResponse(nodes=nodes, edges=edges)
