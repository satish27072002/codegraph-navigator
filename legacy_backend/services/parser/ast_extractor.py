"""
AST extraction using tree-sitter.
Takes a single Python source file and extracts:
  - Function definitions (name, docstring, code, start_line, end_line, complexity)
  - Class definitions (name, docstring, methods, start_line, end_line)
  - Import statements (module name, type: internal | external)
  - Call relationships (caller function -> callee function, line_number)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
    PY_LANGUAGE = Language(tspython.language())
    _PARSER_AVAILABLE = True
except Exception:
    _PARSER_AVAILABLE = False

# Well-known standard library and common third-party module prefixes
_STDLIB_PREFIXES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "pathlib",
    "typing", "collections", "itertools", "functools", "io", "abc",
    "logging", "threading", "multiprocessing", "subprocess", "socket",
    "http", "urllib", "email", "html", "xml", "csv", "hashlib",
    "base64", "uuid", "copy", "shutil", "tempfile", "glob", "fnmatch",
    "inspect", "traceback", "warnings", "contextlib", "dataclasses",
    "enum", "struct", "array", "queue", "heapq", "bisect", "weakref",
    "gc", "platform", "signal", "ctypes", "ast", "dis", "tokenize",
    "unittest", "asyncio", "concurrent", "string", "textwrap",
    "difflib", "pprint", "reprlib", "numbers", "decimal", "fractions",
    "random", "statistics", "pickle", "shelve", "sqlite3", "zlib",
    "gzip", "zipfile", "tarfile", "configparser", "argparse", "getopt",
}

_THIRD_PARTY_PREFIXES = {
    "fastapi", "uvicorn", "pydantic", "starlette", "sqlalchemy",
    "alembic", "celery", "redis", "pymongo", "motor", "httpx",
    "requests", "aiohttp", "flask", "django", "tornado", "sanic",
    "neo4j", "openai", "anthropic", "langchain", "numpy", "pandas",
    "scipy", "sklearn", "torch", "tensorflow", "keras", "PIL",
    "cv2", "matplotlib", "seaborn", "pytest", "click", "typer",
    "dotenv", "yaml", "toml", "boto3", "google", "azure", "stripe",
    "tree_sitter", "tree_sitter_python",
}


@dataclass
class FunctionNode:
    name: str
    file: str
    start_line: int
    end_line: int
    docstring: str
    code: str
    complexity: int
    loc: int
    class_name: Optional[str] = None   # set if this is a method
    embedding: list[float] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassNode:
    name: str
    file: str
    start_line: int
    end_line: int
    docstring: str
    methods: list[str] = field(default_factory=list)


@dataclass
class ImportNode:
    name: str
    type: str  # "internal" | "external"


@dataclass
class CallRelationship:
    caller_name: str     # qualified name of caller
    callee_name: str     # best-effort name of callee
    line_number: int


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _get_text(node: "Node", source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(node: "Node", source: bytes) -> str:
    """Extract the docstring from a function or class body node."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for expr in stmt.children:
                        if expr.type == "string":
                            raw = _get_text(expr, source)
                            for q in ('"""', "'''", '"', "'"):
                                if raw.startswith(q) and raw.endswith(q) and len(raw) > len(q) * 2:
                                    return raw[len(q):-len(q)].strip()
                            return raw.strip("\"'").strip()
                break
    return ""


def _compute_complexity(node: "Node") -> int:
    """
    Approximate cyclomatic complexity by counting decision points:
    if, elif, for, while, and, or, except, with, assert, comprehensions.
    """
    DECISION_TYPES = {
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "with_statement", "assert_statement",
        "boolean_operator",
        "list_comprehension", "set_comprehension", "dictionary_comprehension",
        "generator_expression",
    }
    count = 1  # base complexity

    def walk(n: "Node") -> None:
        nonlocal count
        if n.type in DECISION_TYPES:
            count += 1
        for child in n.children:
            walk(child)

    walk(node)
    return count


def _extract_calls(func_node: "Node", source: bytes, caller_name: str) -> list[CallRelationship]:
    """Walk a function body and collect all call expressions."""
    calls: list[CallRelationship] = []

    def walk(n: "Node") -> None:
        if n.type == "call":
            func_child = n.child_by_field_name("function")
            if func_child is not None:
                callee = _get_text(func_child, source)
                # Strip attribute access: "self.foo" -> "foo", "obj.bar.baz" -> "baz"
                if "." in callee:
                    callee = callee.split(".")[-1]
                line_number = n.start_point[0] + 1
                calls.append(CallRelationship(
                    caller_name=caller_name,
                    callee_name=callee,
                    line_number=line_number,
                ))
        for child in n.children:
            walk(child)

    walk(func_node)
    return calls


def _classify_import(module_name: str, file_path: str) -> str:
    """Return 'external' for stdlib/third-party, 'internal' for same-project imports."""
    root = module_name.lstrip(".").split(".")[0]
    if root in _STDLIB_PREFIXES or root in _THIRD_PARTY_PREFIXES:
        return "external"
    if module_name.startswith("."):
        return "internal"
    return "internal"


def _extract_imports(tree_root: "Node", source: bytes, file_path: str) -> list[ImportNode]:
    imports: list[ImportNode] = []
    for node in tree_root.children:
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    name = _get_text(child, source)
                    imports.append(ImportNode(
                        name=name,
                        type=_classify_import(name, file_path),
                    ))
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node:
                name = _get_text(module_node, source)
                imports.append(ImportNode(
                    name=name,
                    type=_classify_import(name, file_path),
                ))
    return imports


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def extract_from_file(file_path: Path) -> dict:
    """
    Parse a single .py file and return:
      {
        "functions": list[FunctionNode],
        "classes":   list[ClassNode],
        "imports":   list[ImportNode],
        "calls":     list[CallRelationship],
      }
    Falls back to regex-based extraction if tree-sitter is unavailable.
    """
    source_bytes = file_path.read_bytes()
    file_str = str(file_path)

    if _PARSER_AVAILABLE:
        return _extract_with_tree_sitter(source_bytes, file_str)
    else:
        return _extract_with_regex(source_bytes.decode("utf-8", errors="replace"), file_str)


def _extract_with_tree_sitter(source: bytes, file_str: str) -> dict:
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source)
    root = tree.root_node

    functions: list[FunctionNode] = []
    classes: list[ClassNode] = []
    calls: list[CallRelationship] = []
    imports = _extract_imports(root, source, file_str)

    def process_function(node: "Node", class_name: Optional[str] = None) -> FunctionNode:
        name_node = node.child_by_field_name("name")
        name = _get_text(name_node, source) if name_node else "<unknown>"
        docstring = _get_docstring(node, source)
        code = _get_text(node, source)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        complexity = _compute_complexity(node)
        loc = end_line - start_line + 1

        fn = FunctionNode(
            name=name,
            file=file_str,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            code=code,
            complexity=complexity,
            loc=loc,
            class_name=class_name,
        )
        calls.extend(_extract_calls(node, source, fn.qualified_name))
        return fn

    def process_class(node: "Node") -> ClassNode:
        name_node = node.child_by_field_name("name")
        class_name = _get_text(name_node, source) if name_node else "<unknown>"
        docstring = _get_docstring(node, source)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        method_names: list[str] = []

        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    fn_node = None
                    if stmt.type == "function_definition":
                        fn_node = stmt
                    elif stmt.type == "decorated_definition":
                        for d_child in stmt.children:
                            if d_child.type == "function_definition":
                                fn_node = d_child
                                break
                    if fn_node is not None:
                        method = process_function(fn_node, class_name=class_name)
                        functions.append(method)
                        method_names.append(method.name)

        return ClassNode(
            name=class_name,
            file=file_str,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            methods=method_names,
        )

    for node in root.children:
        if node.type == "function_definition":
            functions.append(process_function(node))
        elif node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    functions.append(process_function(child))
                    break
                elif child.type == "class_definition":
                    classes.append(process_class(child))
                    break
        elif node.type == "class_definition":
            classes.append(process_class(node))

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "calls": calls,
    }


def _extract_with_regex(source: str, file_str: str) -> dict:
    """Fallback extraction using regex when tree-sitter is unavailable."""
    functions: list[FunctionNode] = []
    classes: list[ClassNode] = []
    imports: list[ImportNode] = []
    calls: list[CallRelationship] = []
    lines = source.splitlines()

    # Extract imports
    for line in lines:
        m = re.match(r"^import\s+([\w.]+)", line)
        if m:
            name = m.group(1)
            imports.append(ImportNode(name=name, type=_classify_import(name, file_str)))
            continue
        m = re.match(r"^from\s+([\w.]+)\s+import", line)
        if m:
            name = m.group(1)
            imports.append(ImportNode(name=name, type=_classify_import(name, file_str)))

    # Extract function definitions (handles both top-level and indented)
    func_re = re.compile(r"^( *)def\s+(\w+)\s*\(", re.MULTILINE)
    for m in func_re.finditer(source):
        start_line = source[:m.start()].count("\n") + 1
        name = m.group(2)
        indent_len = len(m.group(1))
        end_line = start_line
        for i in range(start_line, len(lines)):
            line = lines[i]
            stripped = line.lstrip()
            cur_indent = len(line) - len(stripped)
            if stripped and cur_indent <= indent_len and i > start_line - 1:
                if re.match(r"(def |class |\w)", stripped) and i != start_line - 1:
                    break
            end_line = i + 1

        code = "\n".join(lines[start_line - 1:end_line])
        functions.append(FunctionNode(
            name=name,
            file=file_str,
            start_line=start_line,
            end_line=end_line,
            docstring="",
            code=code,
            complexity=1,
            loc=end_line - start_line + 1,
        ))

    # Extract class definitions
    class_re = re.compile(r"^class\s+(\w+)", re.MULTILINE)
    for m in class_re.finditer(source):
        start_line = source[:m.start()].count("\n") + 1
        name = m.group(1)
        classes.append(ClassNode(
            name=name,
            file=file_str,
            start_line=start_line,
            end_line=start_line,
            docstring="",
            methods=[],
        ))

    return {
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "calls": calls,
    }
