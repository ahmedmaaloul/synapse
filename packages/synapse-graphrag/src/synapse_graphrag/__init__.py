# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
synapse-graphrag — MCP server, CLI and async client for Synapse.

Synapse turns documents into a Neo4j knowledge graph and answers questions with
GraphRAG. This package is the *client side*: a thin HTTP layer over the backend
API so any LLM host — Claude Code, Claude Desktop, Cursor, VS Code, a script —
can pull budgeted graph context (``synapse_retrieve``) and write the answer
itself, without paying for a second generation call.

Public API::

    from synapse_graphrag import SynapseClient

    async with SynapseClient("http://localhost:8000") as client:
        retrieval = await client.retrieve("Who invented the Analytical Engine?")
        print(retrieval.context, retrieval.usage)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synapse-graphrag")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.0.0"

from synapse_graphrag.client import (  # noqa: E402 — needs __version__ first
    Answer,
    IngestResult,
    Retrieval,
    SynapseClient,
    SynapseConnectionError,
    SynapseError,
    run,
)

__all__ = [
    "Answer",
    "IngestResult",
    "Retrieval",
    "SynapseClient",
    "SynapseConnectionError",
    "SynapseError",
    "__version__",
    "run",
]
