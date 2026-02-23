from typing import Any
import uuid

from pydantic import BaseModel, Field


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
