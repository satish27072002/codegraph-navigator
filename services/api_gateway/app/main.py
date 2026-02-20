from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import engine, get_db_session
from .logging_config import configure_logging, request_id_context
from .models import Base, Job
from .queue import enqueue_pipeline_job
from .schemas import IngestGithubRequest, JobCreatedResponse, JobStatusResponse


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


@app.post("/ingest/github", response_model=JobCreatedResponse)
async def ingest_github(
    payload: IngestGithubRequest,
    session: AsyncSession = Depends(get_db_session),
) -> JobCreatedResponse:
    repo_id = uuid.uuid4()
    job = await _create_pipeline_job(
        session,
        repo_id=repo_id,
        job_type="PIPELINE_INGEST_GITHUB",
    )

    logger.info(
        "job.created",
        extra={
            "job_id": str(job.job_id),
            "repo_id": str(repo_id),
            "job_type": job.job_type,
            "source": "github",
            "repo_url": payload.repo_url,
            "branch": payload.branch,
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
    filename = Path(file.filename or "upload.zip").name
    target_path = uploads_dir / f"{job.job_id}_{filename}"
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
            "uploaded_file": filename,
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
