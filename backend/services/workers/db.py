from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DEFAULT_DB_URL = "postgresql://jobs:jobs@postgres:5432/jobs"


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return url


database_url = _normalize_database_url(
    os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or DEFAULT_DB_URL
)

engine = create_engine(database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
