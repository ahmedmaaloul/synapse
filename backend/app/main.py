"""
Project Synapse — FastAPI Application
Main entry point with CORS, lifespan, and router registration.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging so all our loggers output at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from app.config import get_settings
from app.neo4j_driver import get_driver, close_driver
from app.routers import upload, graph, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle — init and close Neo4j driver."""
    settings = get_settings()
    print(f"🧠 {settings.app_name} starting up...")
    await get_driver()
    print("✅ Neo4j driver initialized")
    yield
    await close_driver()
    print("🛑 Neo4j driver closed")


app = FastAPI(
    title="Project Synapse API",
    description="Interactive GraphRAG Knowledge Explorer",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(graph.router, prefix="/api", tags=["Graph"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])


# ── Health Check ─────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "synapse-backend"}
