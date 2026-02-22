from __future__ import annotations

import ast
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Final


IGNORED_DIRS: Final[set[str]] = {
    ".git",
    "node_modules",
    "venv",
    "__pycache__",
    "dist",
    "build",
}
DEFAULT_MAX_SNIPPET_CHARS: Final[int] = int(os.getenv("MAX_SNIPPET_CHARS", "2000"))


def _stable_node_id(repo_id: uuid.UUID, path: str, symbol: str, node_type: str) -> str:
    raw = f"{repo_id}|{path}|{symbol}|{node_type}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _truncate_snippet(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_python_files(repo_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(repo_dir, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(root) / filename)
    files.sort()
    return files


def _source_for_node(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return segment

    if not hasattr(node, "lineno"):
        return ""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", lineno)
    if lineno is None or end_lineno is None:
        return ""
    lines = source.splitlines(keepends=True)
    return "".join(lines[max(0, lineno - 1) : end_lineno])


def _call_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        left = _call_name(expr.value)
        if left:
            return f"{left}.{expr.attr}"
        return expr.attr
    return None


def _collect_call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Call):
            name = _call_name(candidate.func)
            if name:
                names.append(name)
    return names


def build_graph_facts(
    repo_id: uuid.UUID,
    repo_dir: Path,
    *,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> dict:
    if not repo_dir.exists():
        raise RuntimeError(f"Repo directory not found: {repo_dir}")

    nodes_by_id: dict[str, dict] = {}
    edges_set: set[tuple[str, str, str]] = set()

    def add_node(
        *,
        node_type: str,
        name: str,
        path: str,
        symbol: str,
        code_snippet: str,
    ) -> str:
        node_id = _stable_node_id(repo_id, path, symbol, node_type)
        if node_id not in nodes_by_id:
            nodes_by_id[node_id] = {
                "id": node_id,
                "type": node_type,
                "name": name,
                "path": path,
                "code_snippet": _truncate_snippet(code_snippet, max_snippet_chars),
            }
        return node_id

    def add_edge(source: str, target: str, edge_type: str) -> None:
        edges_set.add((source, target, edge_type))

    for py_file in _iter_python_files(repo_dir):
        rel_path = _relative_posix(py_file, repo_dir)
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        file_node_id = add_node(
            node_type="file",
            name=py_file.name,
            path=rel_path,
            symbol=rel_path,
            code_snippet=source,
        )

        top_level_nodes: list[tuple[str, str, ast.AST]] = []
        for top in tree.body:
            if isinstance(top, ast.ClassDef):
                top_level_nodes.append(("class", top.name, top))
            if isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef):
                top_level_nodes.append(("function", top.name, top))

        local_symbol_ids: dict[str, str] = {}
        for node_type, symbol_name, node in top_level_nodes:
            node_id = add_node(
                node_type=node_type,
                name=symbol_name,
                path=rel_path,
                symbol=symbol_name,
                code_snippet=_source_for_node(source, node),
            )
            add_edge(file_node_id, node_id, "contains")
            local_symbol_ids[symbol_name] = node_id

        for candidate in ast.walk(tree):
            if isinstance(candidate, ast.Import):
                for alias in candidate.names:
                    module_name = alias.name
                    module_id = add_node(
                        node_type="module",
                        name=module_name,
                        path="<external>",
                        symbol=module_name,
                        code_snippet="",
                    )
                    add_edge(file_node_id, module_id, "imports")
            if isinstance(candidate, ast.ImportFrom):
                module_base = f"{'.' * candidate.level}{candidate.module or ''}"
                for alias in candidate.names:
                    if alias.name == "*":
                        module_name = module_base
                    elif module_base:
                        module_name = f"{module_base}.{alias.name}"
                    else:
                        module_name = alias.name
                    module_id = add_node(
                        node_type="module",
                        name=module_name,
                        path="<external>",
                        symbol=module_name,
                        code_snippet="",
                    )
                    add_edge(file_node_id, module_id, "imports")

        for node_type, symbol_name, node in top_level_nodes:
            source_id = local_symbol_ids[symbol_name]
            for call_name in _collect_call_names(node):
                target_id = local_symbol_ids.get(call_name)
                if target_id is None:
                    target_id = add_node(
                        node_type="function",
                        name=call_name,
                        path="<external>",
                        symbol=call_name,
                        code_snippet="",
                    )
                add_edge(source_id, target_id, "calls")

        for top in tree.body:
            if isinstance(top, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for call_name in _collect_call_names(top):
                target_id = local_symbol_ids.get(call_name)
                if target_id is None:
                    target_id = add_node(
                        node_type="function",
                        name=call_name,
                        path="<external>",
                        symbol=call_name,
                        code_snippet="",
                    )
                add_edge(file_node_id, target_id, "calls")

    nodes = sorted(nodes_by_id.values(), key=lambda node: node["id"])
    edges = [
        {"source": source, "target": target, "type": edge_type}
        for source, target, edge_type in sorted(edges_set, key=lambda edge: (edge[0], edge[1], edge[2]))
    ]
    return {
        "repo_id": str(repo_id),
        "nodes": nodes,
        "edges": edges,
    }


def write_graph_facts(
    repo_id: uuid.UUID,
    repo_dir: Path,
    *,
    artifacts_root: Path,
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> Path:
    facts = build_graph_facts(repo_id, repo_dir, max_snippet_chars=max_snippet_chars)
    output_dir = artifacts_root / str(repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph_facts.json"
    output_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return output_path
