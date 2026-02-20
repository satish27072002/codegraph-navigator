"""
OpenAI embedding service.
Generates embeddings using text-embedding-3-small (1536 dimensions).
IMPORTANT: Never switch embedding models mid-project — all stored vectors
must use the same model or similarity search breaks.
"""

import asyncio
import logging
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# OpenAI allows up to 2048 texts per batch call; we stay conservative
_EMBED_BATCH_SIZE = 100


def get_client() -> AsyncOpenAI:
    """Return the shared OpenAI async client (lazy init)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_embed_text(node: dict) -> str:
    """
    Build a rich text representation of a Function node for embedding.
    Combines name, docstring, and code for dense semantic representation.
    """
    parts = []
    if node.get("name"):
        parts.append(f"function: {node['name']}")
    if node.get("docstring"):
        parts.append(f"docstring: {node['docstring']}")
    if node.get("code"):
        # Only embed the first 1000 chars of code to keep token count manageable
        parts.append(f"code:\n{node['code'][:1000]}")
    return "\n".join(parts)


async def embed_text(text: str) -> list[float]:
    """
    Generate a single embedding vector (1536 floats) for a text string.
    """
    if not text.strip():
        return [0.0] * settings.embedding_dimensions

    client = get_client()
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Splits into batches of _EMBED_BATCH_SIZE and calls the API in sequence.
    Returns embeddings in the same order as the input.
    """
    if not texts:
        return []

    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i:i + _EMBED_BATCH_SIZE]
        # Replace empty strings — OpenAI rejects them
        safe_batch = [t if t.strip() else " " for t in batch]
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=safe_batch,
        )
        # API returns embeddings in the same order as the input
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def embed_nodes(nodes: list[dict]) -> list[dict]:
    """
    Given a list of node dicts, embed all Function nodes in batch
    and attach the embedding vector to each node dict.
    Non-Function nodes are returned unchanged.

    Returns the same list with embeddings filled in.
    """
    function_nodes = [n for n in nodes if n.get("label") == "Function"]

    if not function_nodes:
        return nodes

    logger.info(f"Generating embeddings for {len(function_nodes)} function nodes...")

    texts = [_build_embed_text(n) for n in function_nodes]

    try:
        embeddings = await embed_batch(texts)
    except Exception as exc:
        logger.warning(f"Batch embedding failed: {exc} — nodes will have empty embeddings")
        return nodes

    # Build a lookup from node id -> embedding
    embed_map = {
        fn["id"]: emb
        for fn, emb in zip(function_nodes, embeddings)
    }

    # Attach embeddings back to the original node list
    for node in nodes:
        if node.get("label") == "Function" and node["id"] in embed_map:
            node["embedding"] = embed_map[node["id"]]

    logger.info("Embeddings complete")
    return nodes
