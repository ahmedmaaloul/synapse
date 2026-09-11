# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Shared fixtures — a fake Synapse backend behind ``httpx.MockTransport``.

Hermetic by construction: no network, no Neo4j, no LLM. The fake speaks the
backend's exact wire format (JSON bodies, ``data: {json}\\n\\n`` SSE frames,
FastAPI-style ``{"detail": ...}`` errors) so the client, the MCP server and
the CLI are all exercised end to end against the real request/response shapes.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from synapse_graphrag.client import SynapseClient

BASE_URL = "http://synapse.test"


def sse(*events: dict[str, Any]) -> bytes:
    """Encode events exactly as the backend does."""
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def split_bytes(payload: bytes, size: int) -> list[bytes]:
    return [payload[i : i + size] for i in range(0, len(payload), size)] or [b""]


class ChunkedStream(httpx.AsyncByteStream):
    """Chunked body that records who closed it.

    ``closed_by_reader`` is True when ``aclose`` ran in the task that read the
    stream — a deterministic, consumer-driven close — and False when the event
    loop's asyncgen finaliser got to it first, which is the race the client
    guards against with ``contextlib.aclosing``.
    """

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = list(chunks)
        self.reader: asyncio.Task | None = None
        self.closed_by_reader: bool | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.reader = asyncio.current_task()
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed_by_reader = asyncio.current_task() is self.reader
        await asyncio.sleep(0)  # a real socket close suspends too


@dataclass
class Call:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    params: dict[str, str]

    @property
    def json(self) -> Any:
        return json.loads(self.body) if self.body else None


GRAPH = {
    "nodes": [
        {"id": "1", "label": "Charles Babbage", "type": "PERSON", "properties": {"description": "Mathematician"}},
        {"id": "2", "label": "Analytical Engine", "type": "TOOL", "properties": {}},
        {"id": "3", "label": "Ada Lovelace", "type": "PERSON", "properties": {"description": "First programmer"}},
        {"id": "4", "label": "Lonely Node", "type": "CONCEPT", "properties": {}},
    ],
    "links": [
        {"source": "1", "target": "2", "type": "DESIGNED", "properties": {}},
        {"source": "3", "target": "2", "type": "WROTE_PROGRAMS_FOR", "properties": {}},
        {"source": "3", "target": "1", "type": "COLLABORATED_WITH", "properties": {}},
    ],
}

RETRIEVAL = {
    "mode": "local",
    "context": "Charles Babbage designed the Analytical Engine.\nAda Lovelace wrote programs for it.",
    "citations": [
        {"name": "Charles Babbage", "type": "PERSON", "kind": "entity"},
        {"name": "Analytical Engine", "type": "TOOL", "kind": "entity"},
    ],
    "paths": [{"nodes": ["Ada Lovelace", "Analytical Engine"], "hops": 1}],
    "sources": [{"id": "c1", "document": "history.pdf", "index": 0, "text": "Babbage designed…"}],
    "usage": {
        "context_chars": 83,
        "context_tokens_est": 21,
        "truncated": False,
        "citations": 2,
        "paths": 1,
        "sources": 1,
    },
}

ABOUT = {
    "name": "Synapse",
    "version": "0.4.0",
    "llm_provider": "gemini",
    "embedding_provider": "fastembed",
    "repository": "https://github.com/ahmedmaaloul/synapse",
    "license": "PolyForm-Noncommercial-1.0.0",
}

COMMUNITIES = {
    "communities": [
        {"id": "c-1", "title": "Victorian computing", "summary": "Babbage and Lovelace.", "size": 3, "members": ["Charles Babbage", "Ada Lovelace", "Analytical Engine"]},
    ],
    "count": 1,
}

INGEST_DONE = {
    "filename": "history.pdf",
    "chunks_processed": 3,
    "nodes_created": 12,
    "relationships_created": 18,
    "entities_extracted": 14,
    "unique_entities": 12,
    "entities_merged": 2,
    "chunks_stored": 3,
    "communities": 2,
    "communities_summarized": 2,
    "modularity": 0.41,
}


@dataclass
class FakeBackend:
    """Routes requests like the FastAPI app; ``calls`` records what the client sent."""

    calls: list[Call] = field(default_factory=list)
    streams: list[ChunkedStream] = field(default_factory=list)
    chunk_size: int = 4096
    unreachable: bool = False
    failures: dict[str, tuple[int, Any]] = field(default_factory=dict)
    chat_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "citations", "data": RETRIEVAL["citations"]},
            {"type": "paths", "data": RETRIEVAL["paths"]},
            {"type": "sources", "data": RETRIEVAL["sources"]},
            {"type": "token", "data": "Babbage "},
            {"type": "token", "data": "designed it — Ünïcödé ✓"},
            {"type": "done", "usage": {"context_chars": 83, "context_tokens_est": 21, "answer_chars": 32}},
        ]
    )
    ingest_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "progress", "stage": "extracting", "processed": 0, "total": 3},
            {"type": "progress", "stage": "extracting", "processed": 3, "total": 3},
            {"type": "community_progress", "stage": "summarizing", "processed": 2, "total": 2},
            {"type": "done", "data": INGEST_DONE},
        ]
    )
    rebuild_events: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "progress", "stage": "clustering"},
            {"type": "done", "data": {"communities": 2, "summarized": 2, "modularity": 0.41}},
        ]
    )

    def _stream(self, events: list[dict[str, Any]]) -> httpx.Response:
        stream = ChunkedStream(split_bytes(sse(*events), self.chunk_size))
        self.streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        body = await request.aread()
        path = request.url.path
        self.calls.append(
            Call(
                request.method,
                path,
                dict(request.headers),
                body,
                dict(request.url.params),
            )
        )
        if path in self.failures:
            status, detail = self.failures[path]
            if detail is None:
                return httpx.Response(status, text="<html>gateway timeout</html>")
            return httpx.Response(status, json={"detail": detail})

        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "service": "synapse-backend"})
        if path == "/health/ready":
            return httpx.Response(200, json={"status": "ready", "neo4j": "up", "llm_provider": "gemini"})
        if path == "/api/about":
            return httpx.Response(200, json=ABOUT)
        if path == "/api/retrieve":
            payload = json.loads(body)
            result = json.loads(json.dumps(RETRIEVAL))
            budget = payload.get("max_context_chars")
            if budget is not None and budget < len(result["context"]):
                result["context"] = result["context"][:budget] + f"\n…[context truncated to {budget} chars]"
                result["usage"]["truncated"] = True
            result["usage"]["context_chars"] = len(result["context"])
            return httpx.Response(200, json=result)
        if path == "/api/chat":
            return self._stream(self.chat_events)
        if path == "/api/upload":
            return httpx.Response(
                200,
                json={"job_id": "job_000007", "filename": "history.pdf", "total_chunks": 3, "status": "processing"},
            )
        if path == "/api/upload/job_000007/events":
            return self._stream(self.ingest_events)
        if path == "/api/graph-data":
            return httpx.Response(200, json=GRAPH)
        if path == "/api/communities":
            return httpx.Response(200, json=COMMUNITIES)
        if path == "/api/communities/rebuild":
            return httpx.Response(200, json={"job_id": "job_000008", "status": "processing"})
        if path == "/api/communities/rebuild/job_000008/events":
            return self._stream(self.rebuild_events)
        if path == "/api/graph" and request.method == "DELETE":
            return httpx.Response(200, json={"status": "success", "message": "Knowledge graph cleared"})
        return httpx.Response(404, json={"detail": "Not Found"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs: Any) -> SynapseClient:
        if kwargs.get("base_url") is None:  # the CLI passes base_url=None for "use env"
            kwargs["base_url"] = BASE_URL
        kwargs.setdefault("timeout", 5)
        return SynapseClient(transport=self.transport, **kwargs)

    def paths(self, method: str | None = None) -> list[str]:
        return [c.path for c in self.calls if method is None or c.method == method]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not inherit the developer's SYNAPSE_* settings."""
    for name in list(os.environ):
        if name.startswith("SYNAPSE_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "history.pdf"
    path.write_bytes(b"%PDF-1.4\n%fake\n")
    return path
