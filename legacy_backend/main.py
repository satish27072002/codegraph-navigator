"""
CodeGraph Navigator — FastAPI application entry point.
Registers all routers, startup/shutdown lifecycle hooks, and configures the app.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import query, ingest, graph
from db.neo4j_client import get_driver, close_driver, ensure_indexes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────
    logger.info("Starting up CodeGraph Navigator...")
    driver = await get_driver()
    await driver.verify_connectivity()
    logger.info("Neo4j connected")
    await ensure_indexes()
    logger.info("Neo4j indexes ensured")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("Shutting down — closing Neo4j driver...")
    await close_driver()


app = FastAPI(
    title="CodeGraph Navigator",
    description="Graph RAG system for codebase understanding",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(graph.router)


@app.get("/health")
async def health() -> dict:
    """Health check — confirms the API is running."""
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    """Extended status — confirms Neo4j connectivity."""
    try:
        driver = await get_driver()
        await driver.verify_connectivity()
        neo4j_ok = True
    except Exception as exc:
        logger.warning(f"Neo4j status check failed: {exc}")
        neo4j_ok = False

    return {
        "api": "ok",
        "neo4j": "ok" if neo4j_ok else "unavailable",
    }
