"""
Project Synapse — FastAPI Application

Entry point: logging, CORS, lifespan (Neo4j driver + schema bootstrap), routers,
and health/readiness probes.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("synapse")

from app.config import get_settings  # noqa: E402
from app.neo4j_driver import close_driver, get_driver, verify_connectivity  # noqa: E402
from app.routers import chat, graph, upload  # noqa: E402
from app.services.graph_schema import ensure_schema  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown: init the Neo4j driver and ensure the graph schema."""
    settings = get_settings()
    logger.info("🧠 %s starting up (LLM provider: %s)", settings.app_name, settings.llm_provider)
    await get_driver()
    logger.info("✅ Neo4j driver initialized")
    await ensure_schema()
    yield
    await close_driver()
    logger.info("🛑 Neo4j driver closed")


app = FastAPI(
    title="Project Synapse API",
    description="Interactive GraphRAG Knowledge Explorer",
    version="0.2.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Log request latency and expose it via the X-Process-Time header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
    if not request.url.path.startswith(("/health", "/api/graph-data")):
        logger.info("%s %s → %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(graph.router, prefix="/api", tags=["Graph"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])


@app.get("/health", tags=["Health"])
async def health():
    """Liveness probe — is the process up?"""
    return {"status": "ok", "service": "synapse-backend"}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe — can we reach Neo4j?"""
    ok = await verify_connectivity()
    return {
        "status": "ready" if ok else "degraded",
        "neo4j": "up" if ok else "down",
        "llm_provider": settings.llm_provider,
    }
