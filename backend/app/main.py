# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — FastAPI Application

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


APP_VERSION = "0.2.0"
AUTHOR = "Ahmed Maaloul"
AUTHOR_EMAIL = "maaloulahmed25@gmail.com"
REPO_URL = "https://github.com/ahmedmaaloul/synapse"
LICENSE_ID = "AGPL-3.0-or-later"

app = FastAPI(
    title="Synapse API",
    description="Interactive GraphRAG Knowledge Explorer",
    version=APP_VERSION,
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


@app.get("/api/about", tags=["About"])
async def about():
    """Project, authorship and licensing metadata.

    Also satisfies AGPL-3.0 §13: users interacting with this instance over a
    network are told exactly where to obtain the corresponding source.
    """
    return {
        "name": settings.app_name,
        "version": APP_VERSION,
        "description": "Interactive GraphRAG Knowledge Explorer",
        "author": AUTHOR,
        "author_email": AUTHOR_EMAIL,
        "repository": REPO_URL,
        "source_code": REPO_URL,
        "license": LICENSE_ID,
        "license_url": f"{REPO_URL}/blob/main/LICENSE",
        "commercial_license": {
            "required_for": "closed-source, proprietary or SaaS use",
            "contact": AUTHOR_EMAIL,
            "granted_by": AUTHOR,
        },
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe — can we reach Neo4j?"""
    ok = await verify_connectivity()
    return {
        "status": "ready" if ok else "degraded",
        "neo4j": "up" if ok else "down",
        "llm_provider": settings.llm_provider,
    }
