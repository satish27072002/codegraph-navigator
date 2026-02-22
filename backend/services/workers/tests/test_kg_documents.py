from __future__ import annotations

import sys
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import job_runner


def test_build_kg_documents_respects_allowlist_and_ignored_dirs(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_dir / "README.md").write_text("# ignore\n", encoding="utf-8")
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "x.py").write_text("print('nope')\n", encoding="utf-8")
    (repo_dir / "__pycache__").mkdir()
    (repo_dir / "__pycache__" / "cache.py").write_text("print('nope')\n", encoding="utf-8")

    docs = job_runner._build_kg_documents_from_repo(
        uuid.UUID("33333333-3333-3333-3333-333333333333"),
        repo_dir,
    )
    paths = {doc["path"] for doc in docs}

    assert "main.py" in paths
    assert "README.md" not in paths
    assert "node_modules/x.py" not in paths
    assert "__pycache__/cache.py" not in paths


def test_build_kg_documents_skips_oversized_files(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "huge.py").write_text("x" * 4000, encoding="utf-8")

    monkeypatch.setattr(job_runner, "KG_MAX_FILE_BYTES", 32)
    docs = job_runner._build_kg_documents_from_repo(
        uuid.UUID("44444444-4444-4444-4444-444444444444"),
        repo_dir,
    )
    assert docs == []
