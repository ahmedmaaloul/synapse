# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Shared pytest fixtures — hermetic, no real DB / network / LLM."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from types import ModuleType

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

# Modules that are known to do ``from app.neo4j_driver import execute_query``.
# Importing them guarantees they are in ``sys.modules`` before we sweep it, so a
# test that only touches one of them still gets *every* binding patched. The
# sweep below is what actually makes the fixture correct — this list only fixes
# import order, so forgetting to extend it degrades nothing for modules the test
# has already imported.
_DB_BOUND_MODULES = (
    "app.neo4j_driver",
    "app.routers.graph",
    "app.services.chat_engine",
    "app.services.communities",
    "app.services.entity_resolution",
    "app.services.graph_builder",
    "app.services.graph_schema",
)


def _modules_binding_execute_query() -> list[ModuleType]:
    """Every loaded ``app.*`` module that holds an ``execute_query`` attribute.

    ``from x import y`` copies the reference, so patching ``app.neo4j_driver``
    alone leaves every importer still calling the real driver. Sweeping
    ``sys.modules`` means a newly added service is covered automatically instead
    of silently talking to a DB that isn't there.
    """
    for module_path in _DB_BOUND_MODULES:
        importlib.import_module(module_path)

    return [
        module
        for name, module in list(sys.modules.items())
        if (name == "app" or name.startswith("app."))
        and isinstance(module, ModuleType)
        and hasattr(module, "execute_query")
    ]


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

        for module in _modules_binding_execute_query():
            monkeypatch.setattr(module, "execute_query", fake_execute_query)

        return calls

    return install
