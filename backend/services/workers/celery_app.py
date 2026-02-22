from __future__ import annotations

import os

from celery import Celery
from kombu import Queue


broker_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("workers", broker=broker_url)
celery_app.conf.update(
    task_default_queue="pipeline",
    task_queues=(Queue("pipeline"),),
    task_track_started=True,
)

import job_runner  # noqa: E402,F401
