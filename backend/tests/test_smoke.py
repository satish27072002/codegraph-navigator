"""
Smoke tests — verify that the app modules import and basic logic works
without needing a live Neo4j or OpenAI connection.
"""

import pytest


def test_import_config():
    """Config module loads with env defaults."""
    from config import settings
    assert settings.neo4j_user == "neo4j"
    assert settings.llm_model_fast == "gpt-4o-mini"


def test_import_schemas():
    """Schema models parse correctly."""
    from models.schemas import QueryRequest, GraphData, GraphNode, GraphEdge

    req = QueryRequest(question="How does auth work?", codebase_id="default")
    assert req.question == "How does auth work?"
    assert req.top_k == 5
    assert req.hops == 2

    graph = GraphData(nodes=[], edges=[])
    assert graph.nodes == []


def test_structural_question_detection():
    """is_structural_question correctly classifies queries."""
    from services.retrieval.text2cypher import is_structural_question

    assert is_structural_question("What calls run_query?") is True
    assert is_structural_question("How many functions are in auth.py?") is True
    assert is_structural_question("How does authentication work?") is False
    assert is_structural_question("Explain the payment flow") is False


def test_graph_node_builder():
    """_build_graph_data builds correct typed objects."""
    from routers.query import _build_graph_data

    graph_dict = {
        "nodes": [
            {"id": "default:function:auth.py:login:10", "type": "Function",
             "name": "login", "file": "auth.py", "highlighted": True}
        ],
        "edges": [
            {"id": "a-CALLS-b", "source": "a", "target": "b", "type": "CALLS"}
        ],
    }
    result = _build_graph_data(graph_dict)
    assert len(result.nodes) == 1
    assert result.nodes[0].name == "login"
    assert result.nodes[0].highlighted is True
    assert len(result.edges) == 1
    assert result.edges[0].type == "CALLS"
