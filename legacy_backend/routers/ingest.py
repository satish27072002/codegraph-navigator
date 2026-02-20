"""
POST /ingest         — Parse a local codebase path
POST /ingest/github  — Clone a public GitHub repo and parse it
POST /ingest/zip     — Upload a ZIP archive and parse it

All three paths share the same pipeline:
  parse_repository → build_graph → embed_nodes → load_graph
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import GithubIngestRequest, IngestRequest, IngestResponse
from services.embeddings.embedder import embed_nodes
from services.graph.builder import build_graph
from services.graph.neo4j_loader import load_graph
from services.parser.python_parser import parse_repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


# ── Shared pipeline ────────────────────────────────────────────────────────

async def _run_pipeline(repo_path: str, codebase_id: str) -> IngestResponse:
    """Parse, embed, and load a local directory into Neo4j."""

    # Step 1: Parse
    logger.info(f"[ingest] Step 1: Parsing — path={repo_path}, codebase_id={codebase_id}")
    try:
        parsed = parse_repository(repo_path, codebase_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Parsing failed")
        raise HTTPException(status_code=500, detail=f"Parsing error: {exc}")

    files_parsed = parsed["stats"]["files_parsed"]
    logger.info(f"[ingest] Step 1 done: files_parsed={files_parsed}")

    if files_parsed == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No Python files found in '{repo_path}'. Make sure the ZIP contains .py source files.",
        )

    # Step 2: Build graph
    graph_data = build_graph(parsed, codebase_id)
    logger.info(
        f"[ingest] Step 2 done: nodes={len(graph_data['nodes'])}, "
        f"relationships={len(graph_data['relationships'])}"
    )

    # Step 3: Embed Function nodes (non-fatal if OpenAI is unavailable)
    logger.info(f"[ingest] Step 3: Embedding {len(graph_data['nodes'])} nodes...")
    try:
        graph_data["nodes"] = await embed_nodes(graph_data["nodes"])
        logger.info("[ingest] Step 3 done: embeddings generated")
    except Exception as exc:
        logger.warning(f"[ingest] Step 3 warning: embedding failed (continuing without embeddings): {exc}")

    # Step 4: Load into Neo4j
    logger.info("[ingest] Step 4: Loading into Neo4j...")
    try:
        result = await load_graph(graph_data["nodes"], graph_data["relationships"])
    except Exception as exc:
        logger.exception("Neo4j load failed")
        raise HTTPException(status_code=500, detail=f"Neo4j load error: {exc}")

    logger.info(
        f"[ingest] Step 4 done: nodes_created={result['nodes_created']}, "
        f"relationships_created={result['relationships_created']}"
    )

    return IngestResponse(
        status="ok",
        nodes_created=result["nodes_created"],
        relationships_created=result["relationships_created"],
    )


# ── POST /ingest (local path) ──────────────────────────────────────────────

@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest a Python codebase from a local filesystem path.
    The path must be accessible from within the Docker container.
    """
    logger.info(
        f"Ingest (local) started: codebase_id={request.codebase_id}, path={request.repo_path}"
    )
    return await _run_pipeline(request.repo_path, request.codebase_id)


# ── POST /ingest/github ───────────────────────────────────────────────────

@router.post("/github", response_model=IngestResponse)
async def ingest_github(request: GithubIngestRequest) -> IngestResponse:
    """
    Clone a public GitHub repository (shallow, depth=1) into a temp directory,
    ingest it, then delete the temp directory.
    """
    url = request.github_url.strip()
    if not url.startswith("https://github.com/"):
        raise HTTPException(
            status_code=422,
            detail="URL must start with https://github.com/",
        )

    logger.info(f"Ingest (GitHub) started: codebase_id={request.codebase_id}, url={url}")

    tmpdir = tempfile.mkdtemp(prefix="codegraph_github_")
    try:
        import git  # gitpython — imported here to avoid startup cost when unused

        try:
            git.Repo.clone_from(url, tmpdir, depth=1, no_single_branch=True)
        except git.GitCommandError as exc:
            logger.error(f"git clone failed: {exc}")
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to clone repository: {stderr}",
            )

        return await _run_pipeline(tmpdir, request.codebase_id)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── POST /ingest/zip ──────────────────────────────────────────────────────

MAX_ZIP_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("/zip", response_model=IngestResponse)
async def ingest_zip(
    file: UploadFile = File(..., description="ZIP archive of a Python codebase"),
    codebase_id: str = Form(default="default"),
    language: str = Form(default="python"),  # noqa: ARG001 — reserved for future multi-language support
) -> IngestResponse:
    """
    Accept a ZIP file upload, extract it into a temp directory, ingest, then clean up.
    Maximum file size: 100 MB.
    """
    if not (file.filename or "").endswith(".zip"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a .zip archive")

    logger.info(
        f"Ingest (ZIP) started: codebase_id={codebase_id}, filename={file.filename}"
    )

    tmpdir = tempfile.mkdtemp(prefix="codegraph_zip_")
    try:
        contents = await file.read()
        if len(contents) > MAX_ZIP_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"ZIP file too large ({len(contents) // (1024 * 1024)} MB). "
                    "Maximum is 100 MB."
                ),
            )

        zip_path = Path(tmpdir) / "upload.zip"
        zip_path.write_bytes(contents)

        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Skip absolute paths and path-traversal entries for safety
                for member in zf.namelist():
                    if member.startswith("/") or ".." in member:
                        continue
                    zf.extract(member, extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is not a valid ZIP archive",
            )

        return await _run_pipeline(str(extract_dir), codebase_id)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
