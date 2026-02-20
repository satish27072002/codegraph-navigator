"""
Pydantic request/response schemas for all API endpoints.
All API schemas are defined here — never inline in routers.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Graph data structures
# Must always match { nodes: [], edges: [] } — consumed directly by React Flow
# ─────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    type: Literal["Function", "Class", "File", "Module"]
    name: str
    file: str = ""
    highlighted: bool = False


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal["CALLS", "IMPORTS", "INHERITS", "CONTAINS", "HAS_METHOD"]


class GraphData(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Source reference
# ─────────────────────────────────────────────

class SourceReference(BaseModel):
    name: str
    file: str
    start_line: int
    end_line: int
    code: str
    relevance_score: float


# ─────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    codebase_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    hops: int = Field(default=2, ge=0, le=4)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    graph: GraphData = Field(default_factory=GraphData)
    retrieval_method: str
    cypher_used: str | None = None


# ─────────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    repo_path: str = Field(..., description="Local folder path")
    codebase_id: str = Field(..., description="Unique identifier for this codebase")
    language: Literal["python"] = "python"


class GithubIngestRequest(BaseModel):
    github_url: str = Field(..., description="Public GitHub repo URL (https://github.com/...)")
    codebase_id: str = Field(..., description="Unique identifier for this codebase")
    language: Literal["python"] = "python"


class IngestResponse(BaseModel):
    status: str
    nodes_created: int
    relationships_created: int


# ─────────────────────────────────────────────
# GET /graph/{node_id}
# ─────────────────────────────────────────────

class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
