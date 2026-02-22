from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from job_runner import _safe_extract_zip


def test_safe_zip_extracts_files(tmp_path: Path) -> None:
    zip_path = tmp_path / "safe.zip"
    extract_dir = tmp_path / "repo"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("src/main.py", "print('ok')\n")

    _safe_extract_zip(zip_path, extract_dir)

    extracted = extract_dir / "src" / "main.py"
    assert extracted.exists()
    assert extracted.read_text() == "print('ok')\n"


def test_safe_zip_blocks_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "unsafe.zip"
    extract_dir = tmp_path / "repo"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "nope")

    with pytest.raises(RuntimeError, match="Unsafe ZIP member path"):
        _safe_extract_zip(zip_path, extract_dir)

    assert not (tmp_path / "evil.txt").exists()
