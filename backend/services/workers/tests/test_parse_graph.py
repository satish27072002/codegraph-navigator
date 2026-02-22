from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parse_graph import build_graph_facts, write_graph_facts


FIXTURE_PROJECT = Path(__file__).resolve().parent / "fixtures" / "python_project"


def _prepare_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    shutil.copytree(FIXTURE_PROJECT, repo_dir)
    return repo_dir


def test_build_graph_facts_contains_expected_nodes_and_edges(tmp_path: Path) -> None:
    repo_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    repo_dir = _prepare_repo(tmp_path)

    facts = build_graph_facts(repo_id, repo_dir, max_snippet_chars=120)
    assert json.loads(json.dumps(facts))["repo_id"] == str(repo_id)

    nodes = facts["nodes"]
    edges = facts["edges"]
    assert nodes
    assert edges

    assert all(len(node["code_snippet"]) <= 120 for node in nodes)
    assert all("node_modules/" not in node["path"] for node in nodes)
    assert all("__pycache__/" not in node["path"] for node in nodes)

    by_name_type_path = {(node["name"], node["type"], node["path"]): node for node in nodes}
    main_file = by_name_type_path[("main.py", "file", "main.py")]
    greet_fn = by_name_type_path[("greet", "function", "main.py")]
    greeter_cls = by_name_type_path[("Greeter", "class", "main.py")]
    os_module = by_name_type_path[("os", "module", "<external>")]
    helper_fn = by_name_type_path[("helper", "function", "<external>")]

    edge_set = {(edge["source"], edge["target"], edge["type"]) for edge in edges}
    assert (main_file["id"], greet_fn["id"], "contains") in edge_set
    assert (main_file["id"], greeter_cls["id"], "contains") in edge_set
    assert (main_file["id"], os_module["id"], "imports") in edge_set
    assert (greet_fn["id"], helper_fn["id"], "calls") in edge_set

    facts_again = build_graph_facts(repo_id, repo_dir, max_snippet_chars=120)
    assert facts == facts_again


def test_write_graph_facts_emits_json_file(tmp_path: Path) -> None:
    repo_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    repo_dir = _prepare_repo(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    output_path = write_graph_facts(repo_id, repo_dir, artifacts_root=artifacts_root, max_snippet_chars=40)

    assert output_path == artifacts_root / str(repo_id) / "graph_facts.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["repo_id"] == str(repo_id)
    assert isinstance(payload["nodes"], list)
    assert isinstance(payload["edges"], list)
    assert all(len(node["code_snippet"]) <= 40 for node in payload["nodes"])
