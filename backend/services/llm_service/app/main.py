from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from codegraph_shared.openai_utils import chat as _shared_chat
from codegraph_shared.kg_normalize import normalize_kg_extract


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
MAX_CONTEXT_SNIPPETS = int(os.getenv("MAX_CONTEXT_SNIPPETS", "8"))
MAX_SNIPPET_CHARS = int(os.getenv("MAX_LLM_SNIPPET_CHARS", "1200"))
MAX_GRAPH_EDGES_FOR_PROMPT = int(os.getenv("MAX_GRAPH_EDGES_FOR_PROMPT", "40"))

app = FastAPI(title="llm_service")
logger = logging.getLogger("llm_service")


class AnswerRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieval_pack: dict[str, Any]
    kg_context: dict[str, Any] | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[str]
    graph: dict[str, Any]
    warning: str | None = None


class KGChunkExtractRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    doc_path: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class KGExtractResponse(BaseModel):
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


def _sorted_snippets(retrieval_pack: dict[str, Any]) -> list[dict[str, Any]]:
    snippets = retrieval_pack.get("snippets", [])
    if not isinstance(snippets, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw in snippets:
        if not isinstance(raw, dict):
            continue
        snippet_id = str(raw.get("id", "")).strip()
        if not snippet_id:
            continue
        normalized.append(
            {
                "id": snippet_id,
                "name": str(raw.get("name", "")),
                "path": str(raw.get("path", "")),
                "type": str(raw.get("type", "")),
                "code_snippet": str(raw.get("code_snippet", "")),
                "score": float(raw.get("score", 0.0) or 0.0),
            }
        )

    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _fallback_answer(question: str, retrieval_pack: dict[str, Any]) -> AnswerResponse:
    snippets = _sorted_snippets(retrieval_pack)[:MAX_CONTEXT_SNIPPETS]
    nodes = _normalized_nodes(retrieval_pack)
    if not snippets and not nodes:
        return AnswerResponse(
            answer=(
                "No indexed snippets were retrieved for this repository yet. "
                "Run ingest/indexing and retry the query."
            ),
            citations=[],
            graph={"nodes": [], "edges": []},
            warning="OPENAI_API_KEY missing; returned deterministic fallback answer.",
        )

    answer = _deterministic_summary_answer(question, retrieval_pack, snippets)
    citations = [snippet["id"] for snippet in snippets[:5]]
    graph = _build_graph_payload(retrieval_pack, kg_context=None)
    return AnswerResponse(
        answer=answer,
        citations=citations,
        graph=graph,
        warning="OPENAI_API_KEY missing; returned deterministic fallback answer.",
    )


def _build_context(snippets: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for snippet in snippets[:MAX_CONTEXT_SNIPPETS]:
        code = snippet["code_snippet"][:MAX_SNIPPET_CHARS]
        chunks.append(
            "\n".join(
                [
                    f"id: {snippet['id']}",
                    f"path: {snippet['path']}",
                    f"name: {snippet['name']}",
                    f"type: {snippet['type']}",
                    f"score: {snippet['score']}",
                    "snippet:",
                    code,
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def _extract_ids(text: str, allowed: set[str]) -> list[str]:
    ids: list[str] = []
    for token in re.findall(r"[a-f0-9]{32,64}", text.lower()):
        if token in allowed and token not in ids:
            ids.append(token)
    return ids


def _normalized_nodes(retrieval_pack: dict[str, Any]) -> list[dict[str, str]]:
    nodes = retrieval_pack.get("nodes", [])
    if not isinstance(nodes, list):
        return []

    normalized: list[dict[str, str]] = []
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            continue
        normalized.append(
            {
                "id": node_id,
                "name": str(raw.get("name", "")).strip(),
                "type": str(raw.get("type", "")).strip(),
                "path": str(raw.get("path", "")).strip(),
            }
        )
    return normalized


def _normalized_edges(retrieval_pack: dict[str, Any]) -> list[dict[str, str]]:
    edges = retrieval_pack.get("edges", [])
    if not isinstance(edges, list):
        return []

    normalized: list[dict[str, str]] = []
    for raw in edges:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        if not source or not target:
            continue
        normalized.append(
            {
                "source": source,
                "target": target,
                "type": str(raw.get("type", "")).strip().lower() or "related_to",
            }
        )
    return normalized


def _graph_context_summary(retrieval_pack: dict[str, Any], kg_context: dict[str, Any] | None) -> str:
    nodes = _normalized_nodes(retrieval_pack)
    edges = _normalized_edges(retrieval_pack)

    lines: list[str] = []

    if nodes or edges:
        node_by_id = {node["id"]: node for node in nodes}
        type_counts: dict[str, int] = {}
        for node in nodes:
            node_type = node["type"] or "unknown"
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        top_types = ", ".join(
            f"{node_type}:{count}"
            for node_type, count in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[:4]
        ) or "n/a"

        edge_lines: list[str] = []
        for edge in edges[:MAX_GRAPH_EDGES_FOR_PROMPT]:
            source = node_by_id.get(edge["source"], {"name": edge["source"]})
            target = node_by_id.get(edge["target"], {"name": edge["target"]})
            edge_lines.append(f"{source['name']} -[{edge['type']}]-> {target['name']}")

        lines += [
            f"Code graph nodes: {len(nodes)}",
            f"Code graph edges: {len(edges)}",
            f"Node types: {top_types}",
        ]
        if edge_lines:
            lines.append("Key code relationships:")
            lines.extend(edge_lines)

    if kg_context:
        linked = kg_context.get("linked_entities", [])
        evidence = kg_context.get("evidence", [])
        subgraph = kg_context.get("subgraph", {})
        kg_edges = subgraph.get("edges", []) if isinstance(subgraph, dict) else []

        if linked:
            entity_str = ", ".join(
                f"{e.get('name', '')} ({e.get('type', '')})" for e in linked[:10]
            )
            lines.append(f"Semantic entities: {entity_str}")

        if kg_edges:
            lines.append("Semantic relationships:")
            for edge in kg_edges[:MAX_GRAPH_EDGES_FOR_PROMPT]:
                if isinstance(edge, dict):
                    lines.append(
                        f"{edge.get('source', '?')} -[{edge.get('type', 'related')}]-> {edge.get('target', '?')}"
                    )

        if evidence:
            lines.append("Supporting evidence:")
            for ev in evidence[:5]:
                if isinstance(ev, dict):
                    lines.append(f"[{ev.get('doc_path', '')}] {str(ev.get('text', ''))[:300]}")

    return "\n".join(lines) if lines else "No graph context available."


def _build_graph_payload(
    retrieval_pack: dict[str, Any],
    kg_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a unified graph payload merging code graph nodes/edges with KG entities/relations."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    # Code graph nodes
    for raw in retrieval_pack.get("nodes", []):
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            continue
        nodes_by_id[node_id] = {
            "id": node_id,
            "type": str(raw.get("type", "file")).strip() or "file",
            "label": str(raw.get("name", node_id)).strip(),
            "path": str(raw.get("path", "")).strip() or None,
        }

    # Code graph edges
    edge_counter = 0
    for raw in retrieval_pack.get("edges", []):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        if not source or not target:
            continue
        edges.append({
            "id": f"ce_{edge_counter}",
            "source": source,
            "target": target,
            "label": str(raw.get("type", "related")).strip().lower() or "related",
        })
        edge_counter += 1

    if kg_context:
        linked_entities = kg_context.get("linked_entities", [])
        subgraph = kg_context.get("subgraph", {})
        kg_edges = subgraph.get("edges", []) if isinstance(subgraph, dict) else []

        # KG entity nodes — add as "concept" type if no matching code node
        name_to_id: dict[str, str] = {}
        for node in nodes_by_id.values():
            name_to_id[node["label"].lower()] = node["id"]

        for entity in linked_entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            existing_id = name_to_id.get(name.lower())
            if existing_id:
                # Merge: mark existing code node as also being a semantic entity
                name_to_id[name.lower()] = existing_id
            else:
                concept_id = f"kg_{name.lower().replace(' ', '_')}"
                if concept_id not in nodes_by_id:
                    nodes_by_id[concept_id] = {
                        "id": concept_id,
                        "type": "concept",
                        "label": name,
                        "path": None,
                    }
                name_to_id[name.lower()] = concept_id

        # KG relation edges
        kg_edge_counter = 0
        for edge in kg_edges:
            if not isinstance(edge, dict):
                continue
            source_name = str(edge.get("source", "")).strip()
            target_name = str(edge.get("target", "")).strip()
            if not source_name or not target_name:
                continue
            source_id = name_to_id.get(source_name.lower())
            target_id = name_to_id.get(target_name.lower())
            if source_id and target_id:
                edges.append({
                    "id": f"ke_{kg_edge_counter}",
                    "source": source_id,
                    "target": target_id,
                    "label": str(edge.get("type", "related")).strip().lower() or "related",
                })
                kg_edge_counter += 1

        # Evidence nodes — top-scoring KG evidence items added as "evidence" type
        evidence_items = kg_context.get("evidence", [])
        top_evidence = sorted(
            [e for e in evidence_items if isinstance(e, dict)],
            key=lambda e: float(e.get("score", 0)),
            reverse=True,
        )[:6]

        # Snapshot of code node (id, path) pairs to connect evidence to
        code_node_paths = [
            (nid, node.get("path"))
            for nid, node in nodes_by_id.items()
            if node.get("type") not in ("evidence", "concept")
        ]

        ev_edge_counter = 0
        for ev in top_evidence:
            chunk_id = str(ev.get("chunk_id", "")).strip()
            text = str(ev.get("text", "")).strip()
            doc_path = str(ev.get("doc_path", "")).strip() or None
            if not chunk_id or not text:
                continue

            ev_node_id = f"ev_{chunk_id}"
            if ev_node_id not in nodes_by_id:
                label = (text[:58] + "…") if len(text) > 58 else text
                nodes_by_id[ev_node_id] = {
                    "id": ev_node_id,
                    "type": "evidence",
                    "label": label,
                    "path": doc_path,
                }

            # Connect evidence node to its source file node (matched by path)
            if doc_path:
                for code_nid, code_path in code_node_paths:
                    if code_path and code_path == doc_path:
                        edges.append({
                            "id": f"ev_link_{ev_edge_counter}",
                            "source": code_nid,
                            "target": ev_node_id,
                            "label": "supports",
                        })
                        ev_edge_counter += 1
                        break

    # Snippet-based evidence nodes — always generated from highest-scoring retrieved snippets.
    # This ensures evidence nodes appear even when KG extraction was not run during ingestion.
    raw_snippets = retrieval_pack.get("snippets", [])
    top_snippets = sorted(
        [s for s in raw_snippets if isinstance(s, dict) and str(s.get("code_snippet", "")).strip()],
        key=lambda s: float(s.get("score", 0)),
        reverse=True,
    )[:4]

    snip_edge_counter = 0
    for snip in top_snippets:
        snip_id = str(snip.get("id", "")).strip()
        code = str(snip.get("code_snippet", "")).strip()
        path = str(snip.get("path", "")).strip() or None
        if not snip_id or not code:
            continue

        ev_node_id = f"snip_{snip_id}"
        if ev_node_id not in nodes_by_id:
            label = (code[:58] + "…") if len(code) > 58 else code
            nodes_by_id[ev_node_id] = {
                "id": ev_node_id,
                "type": "evidence",
                "label": label,
                "path": path,
            }

        # Connect evidence node to its parent code node
        parent_id: str | None = None
        if snip_id in nodes_by_id and nodes_by_id[snip_id].get("type") not in ("evidence", "concept"):
            parent_id = snip_id
        elif path:
            for nid, node in nodes_by_id.items():
                if (
                    node.get("path") == path
                    and node.get("type") not in ("evidence", "concept")
                    and not nid.startswith(("snip_", "ev_", "kg_"))
                ):
                    parent_id = nid
                    break

        if parent_id:
            edges.append({
                "id": f"snip_link_{snip_edge_counter}",
                "source": parent_id,
                "target": ev_node_id,
                "label": "supports",
            })
            snip_edge_counter += 1

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
    }


def _deterministic_summary_answer(
    question: str,
    retrieval_pack: dict[str, Any],
    snippets: list[dict[str, Any]] | None = None,
) -> str:
    chosen = snippets if snippets is not None else _sorted_snippets(retrieval_pack)[:MAX_CONTEXT_SNIPPETS]
    nodes = _normalized_nodes(retrieval_pack)
    edges = _normalized_edges(retrieval_pack)

    if not chosen and not nodes:
        return (
            "No indexed snippets were retrieved for this repository yet. "
            "Run ingest/indexing and retry the query."
        )

    highlights: list[str] = []
    for snippet in chosen[:4]:
        highlights.append(
            f"- {snippet['name']} ({snippet['path'] or '<no path>'}) [{snippet['type']}] score={snippet['score']:.3f}"
        )

    relation_lines: list[str] = []
    node_by_id = {node["id"]: node for node in nodes}
    for edge in edges[:6]:
        source_name = node_by_id.get(edge["source"], {"name": edge["source"]})["name"]
        target_name = node_by_id.get(edge["target"], {"name": edge["target"]})["name"]
        relation_lines.append(f"- {source_name} --{edge['type']}--> {target_name}")

    lines = [f"Best-effort answer for: {question}"]
    if highlights:
        lines.append("Most relevant code anchors:")
        lines.extend(highlights)
    if relation_lines:
        lines.append("Observed graph relationships:")
        lines.extend(relation_lines)
    lines.append(
        f"Retrieved context size: snippets={len(chosen)}, nodes={len(nodes)}, edges={len(edges)}."
    )
    return "\n".join(lines)


def _looks_low_confidence(answer: str) -> bool:
    text = answer.strip().lower()
    if not text:
        return True
    weak_markers = (
        "i'm unsure",
        "i am unsure",
        "not enough context",
        "cannot determine",
        "no context",
        "can't determine",
    )
    return any(marker in text for marker in weak_markers)


def _build_kg_extract_messages(
    repo_id: str, doc_path: str, chunk_text: str, *, strict_retry: bool
) -> list[dict[str, str]]:
    relation_types = "defines, uses, depends_on, calls, inherits, part_of, related"
    base_instructions = (
        "Extract repository-meaningful entities and relationships from this chunk. "
        "Prioritize modules, classes, functions, concepts, components, configs, and services. "
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "entities": [{"name": "...", "type": "..."}],\n'
        '  "relationships": [{"source": "...", "target": "...", "relation_type": "...", "confidence": 0.0, "evidence": "..."}]\n'
        "}\n"
        f"Use short relation_type values only from: {relation_types}. "
        "Confidence must be a float from 0.0 to 1.0. "
        "Evidence must be a short snippet copied from the chunk."
    )
    strict_suffix = ""
    if strict_retry:
        strict_suffix = (
            "\nCRITICAL: output must be valid JSON object only, with no markdown, "
            "no code fences, and no explanatory text."
        )
    user_prompt = (
        f"{base_instructions}{strict_suffix}\n\n"
        f"repo_id: {repo_id}\n"
        f"path: {doc_path}\n"
        f"chunk:\n{chunk_text}"
    )
    return [
        {
            "role": "system",
            "content": "You are a strict JSON information extraction engine for code intelligence.",
        },
        {"role": "user", "content": user_prompt},
    ]


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("no json object found in model content")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("parsed content is not a json object")
    return parsed


def _heuristic_kg_extract(chunk_text: str) -> KGExtractResponse:
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    evidence = chunk_text[:240]

    for name in re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", chunk_text, flags=re.MULTILINE):
        entities[(name.lower(), "class")] = {"name": name, "type": "class"}
    for name in re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", chunk_text, flags=re.MULTILINE):
        entities[(name.lower(), "function")] = {"name": name, "type": "function"}
    for name in re.findall(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)", chunk_text, flags=re.MULTILINE):
        entities[(name.lower(), "module")] = {"name": name, "type": "module"}
    for name in re.findall(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import", chunk_text, flags=re.MULTILINE):
        entities[(name.lower(), "module")] = {"name": name, "type": "module"}

    if not entities:
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", chunk_text):
            entities[(token.lower(), "symbol")] = {"name": token, "type": "symbol"}
            if len(entities) >= 8:
                break

    entity_list = list(entities.values())
    for idx in range(len(entity_list) - 1):
        relationships.append(
            {
                "source": entity_list[idx]["name"],
                "target": entity_list[idx + 1]["name"],
                "relation_type": "related",
                "confidence": 0.2,
                "evidence": evidence,
            }
        )

    return KGExtractResponse(entities=entity_list, relationships=relationships[:12])


def _openai_kg_extract(repo_id: str, doc_path: str, chunk_text: str) -> KGExtractResponse:
    parse_error: Exception | None = None
    for strict_retry in (False, True):
        messages = _build_kg_extract_messages(repo_id, doc_path, chunk_text, strict_retry=strict_retry)
        try:
            content = _shared_chat(
                messages,
                model=OPENAI_CHAT_MODEL,
                api_key=OPENAI_API_KEY,
                timeout=float(OPENAI_TIMEOUT_SEC),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = _parse_json_object(content)
            entities, relationships = normalize_kg_extract(parsed)
            return KGExtractResponse(entities=entities, relationships=relationships)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            parse_error = exc
            if strict_retry:
                break
            logger.warning("kg.extract.parse_failed first_attempt retrying_with_strict_prompt error=%s", exc)
            continue

    raise HTTPException(status_code=502, detail=f"LLM extraction JSON parse failed after retry: {parse_error}")


def _openai_answer(
    question: str,
    retrieval_pack: dict[str, Any],
    kg_context: dict[str, Any] | None,
) -> AnswerResponse:
    snippets = _sorted_snippets(retrieval_pack)[:MAX_CONTEXT_SNIPPETS]
    context = _build_context(snippets)
    graph_context = _graph_context_summary(retrieval_pack, kg_context)
    allowed_ids = {snippet["id"] for snippet in snippets}

    prompt = (
        "You are answering repository questions using retrieved code structure and semantic graph context. "
        "Return strict JSON with keys: answer (string) and citations (array of snippet ids from the code context). "
        "Only cite ids from the provided context. "
        "When context exists, give a best-effort explanation instead of saying there is no context.\n\n"
        f"Question:\n{question}\n\n"
        f"Code Structure Context:\n{context}\n"
        f"\nGraph and Semantic Context:\n{graph_context}\n"
    )

    try:
        content = _shared_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Be concise, factual, and cite provided snippet ids. "
                        "Answer with practical explanation: repository purpose, key components, and how components connect. "
                        "Never claim there is no context when snippets or graph relationships are present."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=OPENAI_CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            timeout=float(OPENAI_TIMEOUT_SEC),
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except HTTPException as exc:
        raise exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"answer": content, "citations": _extract_ids(content, allowed_ids)}

    answer = str(parsed.get("answer", "")).strip() or "No answer generated."
    citations_raw = parsed.get("citations", [])
    citations: list[str] = []
    if isinstance(citations_raw, list):
        for item in citations_raw:
            token = str(item).strip().lower()
            if token in allowed_ids and token not in citations:
                citations.append(token)

    if not citations:
        for snippet in snippets[:3]:
            citations.append(snippet["id"])

    warning: str | None = None
    if snippets and _looks_low_confidence(answer):
        deterministic = _deterministic_summary_answer(question, retrieval_pack, snippets)
        answer = f"{deterministic}\n\nModel response note:\n{answer}"
        warning = "LLM returned low-confidence wording; appended deterministic retrieval summary."

    graph = _build_graph_payload(retrieval_pack, kg_context)

    return AnswerResponse(answer=answer, citations=citations, graph=graph, warning=warning)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest) -> AnswerResponse:
    if not OPENAI_API_KEY:
        return _fallback_answer(payload.question, payload.retrieval_pack)
    return _openai_answer(payload.question, payload.retrieval_pack, payload.kg_context)


@app.post("/extract/kg", response_model=KGExtractResponse)
def extract_kg(payload: KGChunkExtractRequest) -> KGExtractResponse:
    if not OPENAI_API_KEY:
        return _heuristic_kg_extract(payload.text)
    try:
        return _openai_kg_extract(payload.repo_id, payload.doc_path, payload.text)
    except HTTPException as exc:
        logger.warning(
            "extract.kg.openai_failed repo_id=%s doc_path=%s chunk_id=%s detail=%s; using heuristic fallback",
            payload.repo_id,
            payload.doc_path,
            payload.chunk_id,
            exc.detail,
        )
        return _heuristic_kg_extract(payload.text)
