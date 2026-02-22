from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATABASE_URL = "postgresql+asyncpg://jobs:jobs@postgres:5432/jobs"


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    neo4j_url: str
    cors_origins: list[str]
    data_dir: str
    retrieval_service_url: str
    llm_service_url: str
    graph_service_url: str
    service_timeout_sec: int


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _parse_cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_settings() -> Settings:
    raw_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or DEFAULT_DATABASE_URL
    return Settings(
        database_url=_normalize_database_url(raw_url),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        neo4j_url=os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI") or "bolt://neo4j:7687",
        cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS", "")),
        data_dir=os.getenv("DATA_DIR", "/data"),
        retrieval_service_url=os.getenv("RETRIEVAL_SERVICE_URL", "http://retrieval_service:8003"),
        llm_service_url=os.getenv("LLM_SERVICE_URL", "http://llm_service:8004"),
        graph_service_url=os.getenv("GRAPH_SERVICE_URL", "http://graph_service:8002"),
        service_timeout_sec=int(os.getenv("SERVICE_TIMEOUT_SEC", "30")),
    )
