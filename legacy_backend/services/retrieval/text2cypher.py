"""
text2Cypher service.
For structural questions (counts, impact analysis, dependency chains),
generates a Cypher query using gpt-4o and executes it against Neo4j.
"""

import logging
import re
from openai import AsyncOpenAI
from config import settings
from db.neo4j_client import run_query
from services.llm.prompts import TEXT2CYPHER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# Neo4j schema description — injected into the text2Cypher system prompt
NEO4J_SCHEMA = """
Node Labels and key properties:
  Function: id, name, simple_name, file, start_line, end_line, docstring, code, complexity, loc, class_name, codebase_id
  Class:    id, name, file, start_line, end_line, docstring, methods[], codebase_id
  File:     id, path, language, loc, codebase_id
  Module:   id, name, type (internal|external), codebase_id

Relationship Types:
  (File)-[:CONTAINS]->(Function)
  (File)-[:CONTAINS]->(Class)
  (Class)-[:HAS_METHOD]->(Function)
  (Function)-[:CALLS {line_number}]->(Function)
  (Function)-[:IMPORTS]->(Module)
  (Class)-[:INHERITS]->(Class)

Always filter by codebase_id = $codebase_id unless told otherwise.
"""

# Keywords that indicate a structural (count/graph/dependency) question
_STRUCTURAL_KEYWORDS = [
    r"\bhow many\b", r"\bcount\b", r"\blist all\b", r"\bshow all\b",
    r"\bwhat (calls|imports|inherits|depends|uses)\b",
    r"\bwho calls\b", r"\bwhat calls\b", r"\bcallers of\b",
    r"\bwhat imports\b", r"\bwhat depends\b",
    r"\ball functions\b", r"\ball classes\b", r"\ball files\b",
    r"\bmost complex\b", r"\bhighest complexity\b",
    r"\bdependency\b", r"\bdependencies\b",
    r"\bimpact of\b", r"\bbreaks if\b",
]
_STRUCTURAL_RE = re.compile("|".join(_STRUCTURAL_KEYWORDS), re.IGNORECASE)


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def is_structural_question(question: str) -> bool:
    """
    Heuristic: return True if the question is better answered via Cypher
    (counts, listings, dependency analysis) rather than vector/semantic search.
    """
    return bool(_STRUCTURAL_RE.search(question))


def _clean_cypher(raw: str) -> str:
    """
    Strip markdown code fences and whitespace from LLM-generated Cypher.
    e.g. ```cypher\nMATCH ...\n``` -> MATCH ...
    """
    raw = raw.strip()
    # Remove ```cypher ... ``` or ``` ... ```
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _sanitize_cypher(cypher: str) -> str:
    """
    Basic safety check: block destructive operations.
    Raises ValueError if the query contains mutations.
    """
    upper = cypher.upper()
    forbidden = ["CREATE ", "MERGE ", "DELETE ", "DETACH ", "SET ", "REMOVE ", "DROP "]
    for kw in forbidden:
        if kw in upper:
            raise ValueError(f"Generated Cypher contains forbidden keyword: {kw.strip()}")
    return cypher


async def generate_cypher(question: str, codebase_id: str) -> str:
    """
    Use gpt-4o to generate a parameterized Cypher query for the given question.
    Returns the raw Cypher string (stripped of markdown fences).
    """
    system = TEXT2CYPHER_SYSTEM_PROMPT.format(schema=NEO4J_SCHEMA)
    user = (
        f"Question: {question}\n\n"
        f"The codebase_id is '{codebase_id}'. "
        f"Generate a single Cypher MATCH query that answers this question. "
        f"Use $codebase_id as a parameter. Return only the Cypher, no explanation."
    )

    client = get_client()
    response = await client.chat.completions.create(
        model=settings.llm_model_fast,  # gpt-4o-mini: cheaper, well within TPM limits
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,  # deterministic for Cypher generation
        max_tokens=500,
    )

    raw = response.choices[0].message.content or ""
    cypher = _clean_cypher(raw)
    cypher = _sanitize_cypher(cypher)

    logger.info(f"Generated Cypher: {cypher[:120]}...")
    return cypher


async def execute_cypher(cypher: str, params: dict | None = None) -> list[dict]:
    """
    Execute a Cypher query against Neo4j and return results as a list of dicts.
    Always uses parameterized queries — never string interpolation.
    """
    try:
        results = await run_query(cypher, params=params or {})
        return results
    except Exception as exc:
        logger.warning(f"Cypher execution failed: {exc}")
        return []


async def answer_structural_question(question: str, codebase_id: str) -> dict:
    """
    Full text2Cypher pipeline:
      1. Generate Cypher from question
      2. Execute against Neo4j
      3. Return { "cypher": str, "results": list[dict] }
    """
    try:
        cypher = await generate_cypher(question, codebase_id)
        results = await execute_cypher(cypher, params={"codebase_id": codebase_id})
        return {"cypher": cypher, "results": results}
    except Exception as exc:
        logger.warning(f"text2Cypher pipeline failed: {exc}")
        return {"cypher": "", "results": []}
