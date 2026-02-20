from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class IngestGithubRequest(BaseModel):
    repo_url: str
    branch: str | None = None


class JobCreatedResponse(BaseModel):
    job_id: uuid.UUID
    repo_id: uuid.UUID


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID
    repo_id: uuid.UUID
    job_type: str
    status: str
    progress: int
    current_step: str
    attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime
