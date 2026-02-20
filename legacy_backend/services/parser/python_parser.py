"""
Python source file parser.
Walks a directory, finds all .py files, and uses ast_extractor to produce
structured node data ready for Neo4j loading.
"""

import logging
from pathlib import Path
from typing import Generator

from services.parser.ast_extractor import (
    extract_from_file,
    FunctionNode,
    ClassNode,
    ImportNode,
    CallRelationship,
)

logger = logging.getLogger(__name__)

# Directories to skip during file discovery
_SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", ".env",
    "node_modules", ".next", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.egg-info",
}


def find_python_files(repo_path: str) -> Generator[Path, None, None]:
    """
    Recursively yield all .py files under repo_path,
    skipping common non-source directories.
    """
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    for path in root.rglob("*.py"):
        # Skip any path containing a skipped directory segment
        parts = set(path.parts)
        if any(skip in parts for skip in _SKIP_DIRS):
            continue
        # Skip files in hidden directories
        if any(part.startswith(".") for part in path.parts[len(root.parts):]):
            continue
        yield path


def parse_repository(repo_path: str, codebase_id: str) -> dict:
    """
    Parse an entire Python repository.

    Returns:
      {
        "functions":     list of dicts (ready for Neo4j),
        "classes":       list of dicts (ready for Neo4j),
        "files":         list of dicts (ready for Neo4j),
        "modules":       list of dicts (ready for Neo4j),
        "relationships": list of (source_id, rel_type, target_id, props) tuples,
        "stats": {
            "files_parsed": int,
            "functions_found": int,
            "classes_found": int,
            "calls_found": int,
            "errors": int,
        }
      }
    """
    root = Path(repo_path)

    all_functions: list[dict] = []
    all_classes: list[dict] = []
    all_files: list[dict] = []
    all_modules: dict[str, dict] = {}   # deduplicated by module name
    relationships: list[tuple] = []

    # Track function names per file for call resolution
    # qualified_name -> node_id
    func_name_to_id: dict[str, str] = {}

    stats = {
        "files_parsed": 0,
        "functions_found": 0,
        "classes_found": 0,
        "calls_found": 0,
        "errors": 0,
    }

    py_files = list(find_python_files(repo_path))

    for file_path in py_files:
        rel_path = str(file_path.relative_to(root))
        file_id = f"{codebase_id}:file:{rel_path}"

        try:
            result = extract_from_file(file_path)
        except Exception as exc:
            logger.warning(f"Failed to parse {file_path}: {exc}")
            stats["errors"] += 1
            continue

        functions: list[FunctionNode] = result["functions"]
        classes: list[ClassNode] = result["classes"]
        imports: list[ImportNode] = result["imports"]
        calls: list[CallRelationship] = result["calls"]

        # ── File node ──────────────────────────────────────────────────
        loc = sum(1 for _ in file_path.open())
        file_node = {
            "id": file_id,
            "label": "File",
            "path": rel_path,
            "language": "python",
            "loc": loc,
            "codebase_id": codebase_id,
        }
        all_files.append(file_node)

        # ── Function nodes ─────────────────────────────────────────────
        for fn in functions:
            fn_id = f"{codebase_id}:function:{rel_path}:{fn.qualified_name}:{fn.start_line}"
            func_name_to_id[fn.qualified_name] = fn_id
            func_name_to_id[fn.name] = fn_id  # also index by simple name for call resolution

            fn_dict = {
                "id": fn_id,
                "label": "Function",
                "name": fn.qualified_name,
                "simple_name": fn.name,
                "file": rel_path,
                "start_line": fn.start_line,
                "end_line": fn.end_line,
                "docstring": fn.docstring,
                "code": fn.code,
                "complexity": fn.complexity,
                "loc": fn.loc,
                "class_name": fn.class_name or "",
                "codebase_id": codebase_id,
                "embedding": [],  # filled in Week 2
            }
            all_functions.append(fn_dict)

            # File CONTAINS Function
            relationships.append((file_id, "CONTAINS", fn_id, {}))

        # ── Class nodes ────────────────────────────────────────────────
        for cls in classes:
            cls_id = f"{codebase_id}:class:{rel_path}:{cls.name}:{cls.start_line}"
            cls_dict = {
                "id": cls_id,
                "label": "Class",
                "name": cls.name,
                "file": rel_path,
                "start_line": cls.start_line,
                "end_line": cls.end_line,
                "docstring": cls.docstring,
                "methods": cls.methods,
                "codebase_id": codebase_id,
            }
            all_classes.append(cls_dict)

            # File CONTAINS Class
            relationships.append((file_id, "CONTAINS", cls_id, {}))

            # Class HAS_METHOD Function
            for method_name in cls.methods:
                qualified = f"{cls.name}.{method_name}"
                method_id = func_name_to_id.get(qualified) or func_name_to_id.get(method_name)
                if method_id:
                    relationships.append((cls_id, "HAS_METHOD", method_id, {}))

        # ── Module nodes (imports) ─────────────────────────────────────
        for imp in imports:
            mod_id = f"{codebase_id}:module:{imp.name}"
            if mod_id not in all_modules:
                all_modules[mod_id] = {
                    "id": mod_id,
                    "label": "Module",
                    "name": imp.name,
                    "type": imp.type,
                    "codebase_id": codebase_id,
                }

        # ── Call relationships (resolved in second pass below) ─────────
        for call in calls:
            relationships.append((
                call.caller_name,       # will be resolved to ID in second pass
                "_CALL_PENDING",        # placeholder
                call.callee_name,
                {"line_number": call.line_number},
            ))

        stats["files_parsed"] += 1
        stats["functions_found"] += len(functions)
        stats["classes_found"] += len(classes)
        stats["calls_found"] += len(calls)

    # ── Second pass: resolve CALLS relationships ────────────────────────
    resolved_relationships: list[tuple] = []
    for rel in relationships:
        source_id, rel_type, target_id, props = rel
        if rel_type == "_CALL_PENDING":
            # source_id is caller qualified name, target_id is callee simple name
            actual_caller_id = func_name_to_id.get(source_id)
            actual_callee_id = func_name_to_id.get(target_id)
            if actual_caller_id and actual_callee_id and actual_caller_id != actual_callee_id:
                resolved_relationships.append((
                    actual_caller_id, "CALLS", actual_callee_id, props
                ))
            # If we can't resolve, skip the call edge (callee may be external)
        else:
            resolved_relationships.append(rel)

    logger.info(
        f"Parsed {stats['files_parsed']} files, "
        f"{stats['functions_found']} functions, "
        f"{stats['classes_found']} classes, "
        f"{stats['calls_found']} raw calls -> "
        f"{sum(1 for r in resolved_relationships if r[1] == 'CALLS')} resolved"
    )

    return {
        "functions": all_functions,
        "classes": all_classes,
        "files": all_files,
        "modules": list(all_modules.values()),
        "relationships": resolved_relationships,
        "stats": stats,
    }
