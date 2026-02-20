from __future__ import annotations

import logging
import time
import uuid
from typing import Final

from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select

from celery_app import celery_app
from db import SessionLocal
from models import Job


logger = logging.getLogger(__name__)

MAX_ATTEMPTS: Final[int] = 3
STEP_DELAY_SECONDS: Final[float] = 1.0
PIPELINE_STEPS: Final[list[tuple[str, int]]] = [
    ("INGEST", 25),
    ("PARSE", 50),
    ("LOAD_GRAPH", 75),
    ("EMBED", 100),
]


def _get_locked_job(session, job_uuid: uuid.UUID) -> Job | None:
    stmt = select(Job).where(Job.job_id == job_uuid).with_for_update()
    return session.execute(stmt).scalar_one_or_none()


@celery_app.task(bind=True, name="pipeline.run_job")
def run_pipeline_job(self, job_id: str) -> dict[str, str]:
    job_uuid = uuid.UUID(job_id)

    with SessionLocal.begin() as session:
        job = _get_locked_job(session, job_uuid)
        if job is None:
            logger.warning("job not found", extra={"job_id": job_id})
            return {"status": "missing"}

        if job.status == "completed":
            logger.info("job already completed", extra={"job_id": job_id})
            return {"status": "completed"}

        if job.status == "running":
            logger.info("job already running", extra={"job_id": job_id})
            return {"status": "running"}

        job.status = "running"
        job.error = None
        job.current_step = "INGEST"
        if job.progress < 0:
            job.progress = 0

    try:
        for step_name, progress in PIPELINE_STEPS:
            time.sleep(STEP_DELAY_SECONDS)
            with SessionLocal.begin() as session:
                job = _get_locked_job(session, job_uuid)
                if job is None:
                    return {"status": "missing"}
                if job.status == "completed":
                    return {"status": "completed"}

                job.current_step = step_name
                job.progress = progress

        with SessionLocal.begin() as session:
            job = _get_locked_job(session, job_uuid)
            if job is None:
                return {"status": "missing"}

            job.status = "completed"
            job.progress = 100
            job.current_step = "EMBED"
            job.error = None

        logger.info("job completed", extra={"job_id": job_id})
        return {"status": "completed"}

    except Exception as exc:
        retry_job = False

        with SessionLocal.begin() as session:
            job = _get_locked_job(session, job_uuid)
            if job is None:
                return {"status": "missing"}

            job.attempts = (job.attempts or 0) + 1
            job.error = str(exc)

            if job.attempts >= MAX_ATTEMPTS:
                job.status = "failed"
                retry_job = False
            else:
                job.status = "queued"
                retry_job = True

        if retry_job:
            logger.warning(
                "job failed, retrying",
                extra={"job_id": job_id, "attempts": job.attempts},
            )
            try:
                raise self.retry(exc=exc, countdown=2)
            except MaxRetriesExceededError:
                with SessionLocal.begin() as session:
                    job = _get_locked_job(session, job_uuid)
                    if job is not None:
                        job.status = "failed"
                        job.error = str(exc)
                return {"status": "failed"}

        logger.exception("job failed", extra={"job_id": job_id})
        return {"status": "failed"}
