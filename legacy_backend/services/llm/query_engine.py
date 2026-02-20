"""
LLM query engine.
Assembles context from retrieval results and calls gpt-4o to generate the final answer.
All prompt strings come from prompts.py — never hardcode them here.
"""

import logging
from openai import AsyncOpenAI
from config import settings
from services.llm.prompts import QUERY_SYSTEM_PROMPT, QUERY_USER_TEMPLATE

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# Max total context characters sent to the LLM
# (gpt-4o has 128k context, but we cap for cost and latency)
_MAX_CONTEXT_CHARS = 24_000


def get_client() -> AsyncOpenAI:
    """Return the shared OpenAI async client (lazy init)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def assemble_context(sources: list[dict]) -> str:
    """
    Format retrieved source nodes into a structured context string for the LLM.
    Each source is rendered as a named code block with file + line info.
    Truncates to _MAX_CONTEXT_CHARS to avoid token overflow.
    """
    if not sources:
        return "No relevant code found."

    parts: list[str] = []
    total_chars = 0

    for i, src in enumerate(sources, start=1):
        name = src.get("name") or src.get("id") or f"source_{i}"
        file_path = src.get("file") or ""
        start = src.get("start_line") or ""
        end = src.get("end_line") or ""
        code = src.get("code") or src.get("text") or ""
        docstring = src.get("docstring") or ""
        score = src.get("relevance_score", 0)

        header = f"### {name}"
        if file_path:
            header += f"  ({file_path}"
            if start and end:
                header += f" : {start}–{end}"
            header += ")"
        if score:
            header += f"  [score: {score:.3f}]"

        block_parts = [header]
        if docstring:
            block_parts.append(f"Docstring: {docstring}")
        if code:
            block_parts.append(f"```python\n{code}\n```")

        block = "\n".join(block_parts)

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            parts.append(f"### {name}  (truncated — context limit reached)")
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def assemble_cypher_context(cypher_results: list[dict], cypher: str) -> str:
    """Format text2Cypher results as a context string for the LLM."""
    if not cypher_results:
        return f"Cypher query returned no results.\nQuery used:\n```cypher\n{cypher}\n```"

    lines = [f"Cypher query results ({len(cypher_results)} rows):"]
    for row in cypher_results[:50]:   # cap at 50 rows for context
        lines.append(str(row))

    lines.append(f"\nQuery used:\n```cypher\n{cypher}\n```")
    return "\n".join(lines)


async def generate_answer(question: str, context: str) -> str:
    """
    Call gpt-4o with the assembled context and return the answer as a Markdown string.
    Uses QUERY_SYSTEM_PROMPT and QUERY_USER_TEMPLATE from prompts.py.
    """
    user_message = QUERY_USER_TEMPLATE.format(
        question=question,
        context=context,
    )

    client = get_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    answer = response.choices[0].message.content or "No answer generated."
    return answer
