"""
Graph builder.
Takes the output of parse_repository() and flattens it into
a canonical list of nodes + relationships ready for Neo4j loading.
The parser already builds these structures, so this module provides
a thin normalization + validation layer.
"""

import logging

logger = logging.getLogger(__name__)

# Max code size stored per function (avoids giant blobs in Neo4j)
MAX_CODE_CHARS = 8_000


def _truncate(text: str, max_chars: int = MAX_CODE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n# ... (truncated)"


def build_graph(parsed_data: dict, codebase_id: str) -> dict:
    """
    Normalise parsed AST data into Neo4j-ready node and relationship lists.

    Input: output of parse_repository()
    Returns:
      {
        "nodes":         list[dict]  — each has "id", "label", and label-specific fields,
        "relationships": list[tuple] — (source_id, rel_type, target_id, props_dict),
        "stats": {
            "nodes": int,
            "relationships": int,
        }
      }
    """
    nodes: list[dict] = []

    # ── Function nodes ─────────────────────────────────────────────────
    for fn in parsed_data.get("functions", []):
        nodes.append({
            "id": fn["id"],
            "label": "Function",
            "name": fn["name"],
            "simple_name": fn.get("simple_name", fn["name"]),
            "file": fn["file"],
            "start_line": fn["start_line"],
            "end_line": fn["end_line"],
            "docstring": fn.get("docstring", ""),
            "code": _truncate(fn.get("code", "")),
            "complexity": fn.get("complexity", 1),
            "loc": fn.get("loc", 0),
            "class_name": fn.get("class_name", ""),
            "codebase_id": codebase_id,
            "embedding": fn.get("embedding", []),
        })

    # ── Class nodes ────────────────────────────────────────────────────
    for cls in parsed_data.get("classes", []):
        nodes.append({
            "id": cls["id"],
            "label": "Class",
            "name": cls["name"],
            "file": cls["file"],
            "start_line": cls["start_line"],
            "end_line": cls["end_line"],
            "docstring": cls.get("docstring", ""),
            "methods": cls.get("methods", []),
            "codebase_id": codebase_id,
        })

    # ── File nodes ─────────────────────────────────────────────────────
    for f in parsed_data.get("files", []):
        nodes.append({
            "id": f["id"],
            "label": "File",
            "path": f["path"],
            "language": f.get("language", "python"),
            "loc": f.get("loc", 0),
            "codebase_id": codebase_id,
        })

    # ── Module nodes ───────────────────────────────────────────────────
    for mod in parsed_data.get("modules", []):
        nodes.append({
            "id": mod["id"],
            "label": "Module",
            "name": mod["name"],
            "type": mod.get("type", "external"),
            "codebase_id": codebase_id,
        })

    relationships = parsed_data.get("relationships", [])

    logger.info(
        f"Graph built: {len(nodes)} nodes, {len(relationships)} relationships "
        f"for codebase '{codebase_id}'"
    )

    return {
        "nodes": nodes,
        "relationships": relationships,
        "stats": {
            "nodes": len(nodes),
            "relationships": len(relationships),
        },
    }
