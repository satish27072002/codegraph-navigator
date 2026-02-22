from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import _is_missing_fulltext_index_error, _normalize_fulltext_query


class _FakeNeo4jError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_detects_missing_fulltext_index_error() -> None:
    exc = _FakeNeo4jError(
        "Neo.DatabaseError.Schema.IndexNotFound",
        "No such fulltext schema index: code_node_fulltext",
    )
    assert _is_missing_fulltext_index_error(exc) is True


def test_ignores_non_missing_index_error() -> None:
    exc = _FakeNeo4jError(
        "Neo.ClientError.Statement.SyntaxError",
        "Invalid input near CALL db.index.fulltext.queryNodes",
    )
    assert _is_missing_fulltext_index_error(exc) is False


def test_normalize_fulltext_query_escapes_lucene_special_chars() -> None:
    assert _normalize_fulltext_query("src/sample") == r"src\/sample"
    assert _normalize_fulltext_query("name:(foo)") == r"name\:\(foo\)"
    assert _normalize_fulltext_query("a && b || c") == r"a \&\& b \|\| c"


def test_normalize_fulltext_query_compacts_whitespace() -> None:
    assert _normalize_fulltext_query("  alpha   beta  ") == "alpha beta"
    assert _normalize_fulltext_query("   ") == ""
