"""Shared pytest fixtures — hermetic, no real DB / network / LLM."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

# Force offline-safe providers before any app import reads settings.
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("LLM_PROVIDER", "ollama")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test sees a fresh, uncached Settings object."""
    from app.config import get_settings
    from app.services import llm_provider

    get_settings.cache_clear()
    llm_provider._load_embeddings.cache_clear()
    yield
    get_settings.cache_clear()
    llm_provider._load_embeddings.cache_clear()


QueryHandler = Callable[[str, dict], list]


@pytest.fixture
def fake_neo4j(monkeypatch):
    """Patch ``execute_query`` everywhere it's imported.

    Usage:
        def test_x(fake_neo4j):
            calls = fake_neo4j(lambda q, p: [{"name": "Ada"}] if "MATCH" in q else [])
            ... run code ...
            assert calls  # list of (query, params) actually executed
    """

    def install(handler: QueryHandler):
        calls: list[tuple[str, dict]] = []

        async def fake_execute_query(query: str, parameters: dict | None = None):
            params = parameters or {}
            calls.append((query, params))
            result = handler(query, params)
            return result if result is not None else []

        for module_path in (
            "app.neo4j_driver",
            "app.services.graph_builder",
            "app.services.chat_engine",
            "app.services.graph_schema",
            "app.routers.graph",
        ):
            import importlib

            module = importlib.import_module(module_path)
            if hasattr(module, "execute_query"):
                monkeypatch.setattr(module, "execute_query", fake_execute_query)

        return calls

    return install
