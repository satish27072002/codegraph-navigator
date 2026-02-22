from __future__ import annotations

import uuid

from celery import Celery

from .config import get_settings


settings = get_settings()
celery_client = Celery("api_gateway", broker=settings.redis_url)


def enqueue_pipeline_job(
    job_id: uuid.UUID,
) -> None:
    celery_client.send_task(
        "pipeline.run_job",
        args=[str(job_id)],
        queue="pipeline",
    )


def enqueue_kg_ingest_job(
    *,
    job_id: uuid.UUID,
    repo_id: uuid.UUID,
) -> None:
    celery_client.send_task(
        "pipeline.run_kg_ingest",
        args=[str(job_id), str(repo_id)],
        queue="pipeline",
    )
