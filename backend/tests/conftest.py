# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
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


@pytest.fixture(autouse=True)
def _no_lab_calibration_file(monkeypatch):
    """Never read the user's ``backend/lab_runs/calibration.json``.

    ``estimate.load_calibration()`` (the router's estimate, ``ingest.plan_ingest``)
    reads that file when it exists, so a measured calibration on the developer's
    machine would change every priced ingest a test asserts. Point it at a path
    that cannot exist; a test that wants a calibration passes one explicitly.
    """
    from pathlib import Path

    from app.lab import estimate

    monkeypatch.setattr(estimate, "CALIBRATION_PATH", Path(os.devnull) / "calibration.json")


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
    "app.services.procedural_store",
)


def _modules_binding(attribute: str) -> list[ModuleType]:
    """Every loaded ``app.*`` module that holds ``attribute`` (a driver function).

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
        and hasattr(module, attribute)
    ]


def _modules_binding_execute_query() -> list[ModuleType]:
    """Every loaded ``app.*`` module that holds an ``execute_query`` attribute."""
    return _modules_binding("execute_query")


class RecordedCalls(list):
    """``(query, params)`` for every statement run, in execution order.

    A plain list (tests compare it with ``==``) plus ``batches``: one list per
    ``execute_write_batch`` call, holding that transaction's statements. Batch
    statements ALSO appear in the flat list, in order, so a handler-driven test
    reads the same either way; ``batches`` is what lets a test assert that
    writes which must be atomic really were sent as ONE transaction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[tuple[str, dict]]] = []


@pytest.fixture
def fake_neo4j(monkeypatch):
    """Patch ``execute_query`` AND ``execute_write_batch`` everywhere they're imported.

    Both route every statement through the same ``handler`` and record into the
    same ``calls`` list; a write batch returns one handler result per statement
    (the real driver's shape) and is also recorded in ``calls.batches``.

    Usage:
        def test_x(fake_neo4j):
            calls = fake_neo4j(lambda q, p: [{"name": "Ada"}] if "MATCH" in q else [])
            ... run code ...
            assert calls  # list of (query, params) actually executed
            assert len(calls.batches) == 1  # one write transaction
    """

    def install(handler: QueryHandler):
        calls = RecordedCalls()

        async def fake_execute_query(query: str, parameters: dict | None = None):
            params = parameters or {}
            calls.append((query, params))
            result = handler(query, params)
            return result if result is not None else []

        async def fake_execute_write_batch(statements: list[tuple[str, dict]]):
            batch: list[tuple[str, dict]] = []
            # Registered before running, so a handler that raises mid-batch
            # still leaves the statements it saw on record.
            calls.batches.append(batch)
            results: list[list] = []
            for query, parameters in statements:
                params = parameters or {}
                calls.append((query, params))
                batch.append((query, params))
                result = handler(query, params)
                results.append(result if result is not None else [])
            return results

        for module in _modules_binding("execute_query"):
            monkeypatch.setattr(module, "execute_query", fake_execute_query)
        for module in _modules_binding("execute_write_batch"):
            monkeypatch.setattr(module, "execute_write_batch", fake_execute_write_batch)

        return calls

    return install
