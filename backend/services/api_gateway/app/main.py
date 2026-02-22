from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import engine, get_db_session
from .logging_config import configure_logging, request_id_context
from .models import Base, Job
from .queue import enqueue_kg_ingest_job, enqueue_pipeline_job
from .schemas import (
    JobCreatedResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    RepoStatusResponse,
)


configure_logging()
logger = logging.getLogger("api_gateway")
settings = get_settings()

app = FastAPI(title="api_gateway")

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    token = request_id_context.set(request_id)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "request.failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        request_id_context.reset(token)
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request.completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    request_id_context.reset(token)
    return response


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.ready", extra={"database_url": settings.database_url})


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


def _clean_upstream_detail(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "no upstream detail"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]

    if isinstance(parsed, dict):
        if "detail" in parsed:
            return str(parsed["detail"])[:500]
        if "message" in parsed:
            return str(parsed["message"])[:500]
    return text[:500]


def _post_json(url: str, payload: dict) -> dict:
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=settings.service_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        upstream_status = int(exc.code)
        clean = _clean_upstream_detail(detail)
        raise HTTPException(
            status_code=upstream_status,
            detail=f"upstream POST {url} failed ({upstream_status}): {clean}",
        ) from exc
    except urlerror.URLError as exc:
        raise HTTPException(status_code=502, detail=f"upstream POST {url} unavailable: {exc.reason}") from exc


def _get_json(url: str, params: dict[str, str]) -> dict:
    query = urlparse.urlencode(params)
    full_url = f"{url}?{query}" if query else url
    req = urlrequest.Request(full_url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=settings.service_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        upstream_status = int(exc.code)
        clean = _clean_upstream_detail(detail)
        raise HTTPException(
            status_code=upstream_status,
            detail=f"upstream GET {full_url} failed ({upstream_status}): {clean}",
        ) from exc
    except urlerror.URLError as exc:
        raise HTTPException(status_code=502, detail=f"upstream GET {full_url} unavailable: {exc.reason}") from exc


def _post_json_passthrough(
    url: str,
    body: bytes,
    *,
    content_type: str,
) -> tuple[int, object]:
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": content_type or "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=settings.service_timeout_sec) as response:
            raw = response.read().decode("utf-8")
            payload: object = json.loads(raw) if raw else {}
            return int(response.status), payload
    except urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            payload: object = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw or f"upstream POST {url} failed with status {exc.code}"}
        return int(exc.code), payload
    except urlerror.URLError as exc:
        raise HTTPException(status_code=502, detail=f"upstream POST {url} unavailable: {exc.reason}") from exc


async def _create_pipeline_job(
    session: AsyncSession,
    *,
    repo_id: uuid.UUID,
    job_type: str,
) -> Job:
    job = Job(
        repo_id=repo_id,
        job_type=job_type,
        status="queued",
        progress=0,
        current_step="INGEST",
        attempts=0,
        error=None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


@app.post("/ingest/zip", response_model=JobCreatedResponse)
async def ingest_zip(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> JobCreatedResponse:
    repo_id = uuid.uuid4()
    job = await _create_pipeline_job(
        session,
        repo_id=repo_id,
        job_type="PIPELINE_INGEST_ZIP",
    )

    uploads_dir = Path(settings.data_dir) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / f"{repo_id}.zip"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await file.close()

    logger.info(
        "job.created",
        extra={
            "job_id": str(job.job_id),
            "repo_id": str(repo_id),
            "job_type": job.job_type,
            "source": "zip",
            "saved_to": os.fspath(target_path),
        },
    )
    try:
        enqueue_pipeline_job(job.job_id)
    except Exception:
        logger.exception(
            "job.enqueue_failed",
            extra={"job_id": str(job.job_id), "repo_id": str(repo_id)},
        )

    return JobCreatedResponse(job_id=job.job_id, repo_id=repo_id)


@app.post("/ingest/kg/zip", response_model=JobCreatedResponse)
async def ingest_kg_zip(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> JobCreatedResponse:
    repo_id = uuid.uuid4()
    job = await _create_pipeline_job(
        session,
        repo_id=repo_id,
        job_type="PIPELINE_KG_INGEST_ZIP",
    )

    uploads_dir = Path(settings.data_dir) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / f"{repo_id}.zip"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await file.close()

    logger.info(
        "job.created",
        extra={
            "job_id": str(job.job_id),
            "repo_id": str(repo_id),
            "job_type": job.job_type,
            "source": "kg_zip",
            "saved_to": os.fspath(target_path),
        },
    )
    try:
        enqueue_kg_ingest_job(job_id=job.job_id, repo_id=repo_id)
        logger.info(
            "queued run_kg_ingest",
            extra={
                "job_id": str(job.job_id),
                "repo_id": str(repo_id),
                "task_name": "pipeline.run_kg_ingest",
            },
        )
    except Exception:
        logger.exception(
            "job.enqueue_failed",
            extra={"job_id": str(job.job_id), "repo_id": str(repo_id)},
        )

    return JobCreatedResponse(job_id=job.job_id, repo_id=repo_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> JobStatusResponse:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse.model_validate(job)


@app.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(
    repo_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_db_session),
) -> list[JobStatusResponse]:
    stmt = select(Job).where(Job.repo_id == repo_id).order_by(desc(Job.created_at))
    rows = (await session.scalars(stmt)).all()
    return [JobStatusResponse.model_validate(row) for row in rows]


@app.post("/query", response_model=QueryResponse)
def query_repo(payload: QueryRequest) -> QueryResponse:
    retrieval_url = f"{settings.retrieval_service_url.rstrip('/')}/retrieve"
    llm_url = f"{settings.llm_service_url.rstrip('/')}/answer"

    retrieval_pack = _post_json(
        retrieval_url,
        {
            "repo_id": str(payload.repo_id),
            "question": payload.question,
        },
    )
    llm_response = _post_json(
        llm_url,
        {
            "repo_id": str(payload.repo_id),
            "question": payload.question,
            "retrieval_pack": retrieval_pack,
        },
    )

    citations_raw = llm_response.get("citations", [])
    citations = [str(item) for item in citations_raw] if isinstance(citations_raw, list) else []
    warning = llm_response.get("warning")

    return QueryResponse(
        answer=str(llm_response.get("answer", "")),
        citations=citations,
        warning=str(warning) if warning is not None else None,
        retrieval_pack=retrieval_pack,
    )


@app.post("/kg/query")
async def kg_query_proxy(request: Request):
    retrieval_url = f"{settings.retrieval_service_url.rstrip('/')}/kg/query"
    status_code, payload = _post_json_passthrough(
        retrieval_url,
        await request.body(),
        content_type=request.headers.get("content-type", "application/json"),
    )
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/repos/{repo_id}/status", response_model=RepoStatusResponse)
def repo_status(repo_id: uuid.UUID) -> RepoStatusResponse:
    graph_url = f"{settings.graph_service_url.rstrip('/')}/graph/repo/status"
    payload = _get_json(graph_url, {"repo_id": str(repo_id)})
    return RepoStatusResponse.model_validate(payload)
