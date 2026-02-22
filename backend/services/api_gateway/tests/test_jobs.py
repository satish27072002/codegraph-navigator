from __future__ import annotations

import asyncio
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


TEST_DB_PATH = Path("./test_api_gateway.db")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{TEST_DB_PATH}")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
TEST_DATA_DIR = ROOT_DIR / ".test_data"
os.environ.setdefault("DATA_DIR", str(TEST_DATA_DIR))

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402


def _reset_tables() -> None:
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)

    async def _run() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_run())


def _zip_payload() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("main.py", "print('ok')\n")
    return buffer.getvalue()


def test_create_zip_job_and_fetch_status() -> None:
    _reset_tables()

    with TestClient(app) as client:
        create_response = client.post(
            "/ingest/zip",
            files={"file": ("repo.zip", _zip_payload(), "application/zip")},
        )
        assert create_response.status_code == 200
        payload = create_response.json()

        job_id = payload["job_id"]
        repo_id = payload["repo_id"]

        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()

        assert status_payload["job_id"] == job_id
        assert status_payload["repo_id"] == repo_id
        assert status_payload["job_type"] == "PIPELINE_INGEST_ZIP"
        assert status_payload["status"] == "queued"
        assert status_payload["progress"] == 0
        assert status_payload["current_step"] == "INGEST"
        assert status_payload["attempts"] == 0
        assert status_payload["error"] is None


def test_list_jobs_for_repo() -> None:
    _reset_tables()

    with TestClient(app) as client:
        create_response = client.post(
            "/ingest/zip",
            files={"file": ("repo.zip", _zip_payload(), "application/zip")},
        )
        assert create_response.status_code == 200
        payload = create_response.json()

        jobs_response = client.get("/jobs", params={"repo_id": payload["repo_id"]})
        assert jobs_response.status_code == 200
        jobs = jobs_response.json()

        assert len(jobs) == 1
        assert jobs[0]["job_id"] == payload["job_id"]
        assert jobs[0]["repo_id"] == payload["repo_id"]


def test_create_kg_zip_job_and_fetch_status() -> None:
    _reset_tables()

    with TestClient(app) as client:
        create_response = client.post(
            "/ingest/kg/zip",
            files={"file": ("repo.zip", _zip_payload(), "application/zip")},
        )
        assert create_response.status_code == 200
        payload = create_response.json()

        status_response = client.get(f"/jobs/{payload['job_id']}")
        assert status_response.status_code == 200
        status_payload = status_response.json()

        assert status_payload["job_id"] == payload["job_id"]
        assert status_payload["repo_id"] == payload["repo_id"]
        assert status_payload["job_type"] == "PIPELINE_KG_INGEST_ZIP"
        assert status_payload["status"] == "queued"
        assert status_payload["progress"] == 0
        assert status_payload["current_step"] == "INGEST"
