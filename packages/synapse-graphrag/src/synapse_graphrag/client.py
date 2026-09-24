# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — async Python client

One implementation of the backend's wire format, shared by the MCP server and
the CLI: the JSON endpoints, the Server-Sent-Events streams behind ``/api/chat``
and the ingestion / community-rebuild / evolution jobs, and ``POST /api/retrieve``
— the retrieval-only endpoint that returns *budgeted* GraphRAG context so a host
that already has an LLM can write the answer itself.

Procedural memory (``/api/procedures``, ``/api/agent/ask``) sits next to it: a
procedural graph is *how* to work through a multi-step task (steps, and
transitions carrying a condition, guidance and pitfalls), after Lu, Chen, Wu,
Arık, "Procedural Graphs: Self-Evolving Execution Structures for LLM Agents"
(arXiv:2609.09153). The client reads, edits, versions and evolves those graphs
and asks for step-local guidance; it never runs an agent itself.

Why a client over HTTP rather than importing the engine: the backend needs
Neo4j, an embedding model and an LLM provider; a host only needs ``httpx``.

Every failure surfaces as :class:`SynapseError` — HTTP errors carry the
backend's ``detail`` and status, an unreachable backend raises the
:class:`SynapseConnectionError` subclass with a message that says where it
looked. Nothing here logs; callers decide what to show.

Configuration is read from the environment when an argument is omitted:
``SYNAPSE_URL`` (default ``http://localhost:8000``), ``SYNAPSE_API_KEY`` (sent
as ``Authorization: Bearer …``; the Synapse backend itself ignores it, an
authenticating proxy in front of it will not), ``SYNAPSE_TIMEOUT`` (seconds),
``SYNAPSE_MAX_CONTEXT_CHARS`` (default retrieval budget; ``0`` disables it,
anything else must be at least ``MIN_CONTEXT_CHARS``, the backend's floor) and
``SYNAPSE_CACHE_TTL`` (MCP-side retrieval cache, seconds; ``0`` disables).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from types import EllipsisType
from typing import Any, TypeVar
from urllib.parse import quote

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_CONTEXT_CHARS = 6000
# The backend rejects smaller budgets (``POST /api/retrieve``: ``max_context_chars >= 200``).
MIN_CONTEXT_CHARS = 200
DEFAULT_CACHE_TTL = 300.0
CONNECT_TIMEOUT = 10.0
# Floor on the timeout of requests that make the backend run an agent (the
# navigator's ``agent_ask``, an evolution job): several LLM calls in a row can
# outlast SYNAPSE_TIMEOUT's default, and a client that gives up does not stop
# the backend, which keeps working and spending. See ``env_long_timeout``.
LONG_RUNNING_TIMEOUT = 600.0

# Sync or async callable receiving one event dict; see ``SynapseClient.ingest_pdf``.
ProgressCallback = Callable[[dict[str, Any]], Any]


# ── Errors ───────────────────────────────────────────────────────────────────
class SynapseError(Exception):
    """The backend answered with an error, or something else went wrong.

    ``status`` is the HTTP status (``None`` for transport-level failures) and
    ``detail`` the backend's ``detail`` field when it sent JSON, else the body,
    rendered as one readable line. ``payload`` keeps that ``detail`` as sent —
    e.g. ``{"message": …, "diagnostics": [...]}`` for a procedural graph the
    backend refused — so callers can act on its structure; ``None`` otherwise.
    """

    def __init__(self, status: int | None, detail: str, payload: Any = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.payload = payload

    def __str__(self) -> str:
        if self.status is None:
            return self.detail
        return f"HTTP {self.status}: {self.detail}"


class SynapseConnectionError(SynapseError):
    """The backend could not be reached at all (refused, DNS, connect timeout)."""


# ── Environment ──────────────────────────────────────────────────────────────
def _env_number(name: str, default: float, *, integer: bool = False) -> float | int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default) if integer else default
    try:
        return int(raw) if integer else float(raw)
    except ValueError as exc:
        kind = "an integer" if integer else "a number"
        raise SynapseError(None, f"{name} must be {kind}, got {raw!r}") from exc


def env_url() -> str:
    """``SYNAPSE_URL`` without a trailing slash, defaulting to localhost."""
    return (os.environ.get("SYNAPSE_URL") or DEFAULT_URL).rstrip("/") or DEFAULT_URL


def env_api_key() -> str | None:
    return os.environ.get("SYNAPSE_API_KEY") or None


def env_timeout() -> float:
    return float(_env_number("SYNAPSE_TIMEOUT", DEFAULT_TIMEOUT))


def env_long_timeout() -> float:
    """``max(SYNAPSE_TIMEOUT, LONG_RUNNING_TIMEOUT)``: the timeout for agent runs.

    A timeout there only abandons the answer: the backend finishes the run and
    pays for it anyway, so waiting longer is the cheaper failure.
    """
    return max(env_timeout(), LONG_RUNNING_TIMEOUT)


def env_max_context_chars() -> int | None:
    """Default retrieval budget; ``0`` or a negative value means *no budget*.

    A positive value below :data:`MIN_CONTEXT_CHARS` is rejected here rather
    than sent: the backend would answer every retrieval with a 422.
    """
    value = int(_env_number("SYNAPSE_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS, integer=True))
    if value <= 0:
        return None
    if value < MIN_CONTEXT_CHARS:
        raise SynapseError(
            None,
            f"SYNAPSE_MAX_CONTEXT_CHARS must be 0 (no budget) or at least "
            f"{MIN_CONTEXT_CHARS}, got {value}",
        )
    return value


def env_cache_ttl() -> float:
    return float(_env_number("SYNAPSE_CACHE_TTL", DEFAULT_CACHE_TTL))


# ── Results ──────────────────────────────────────────────────────────────────
def _as_list(value: Any) -> list[dict[str, Any]]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class Retrieval:
    """``POST /api/retrieve`` result: context plus provenance and ``usage``."""

    mode: str
    context: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Retrieval:
        return cls(
            mode=str(payload.get("mode") or "local"),
            context=str(payload.get("context") or ""),
            citations=_as_list(payload.get("citations")),
            paths=_as_list(payload.get("paths")),
            sources=_as_list(payload.get("sources")),
            usage=_as_dict(payload.get("usage")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "context": self.context,
            "citations": list(self.citations),
            "paths": list(self.paths),
            "sources": list(self.sources),
            "usage": dict(self.usage),
        }


@dataclass(slots=True)
class Answer:
    """Aggregated ``/api/chat`` stream: the answer text plus what grounded it."""

    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "paths": list(self.paths),
            "sources": list(self.sources),
            "usage": dict(self.usage),
        }


@dataclass(slots=True)
class IngestResult:
    """The ingestion job's ``done`` payload, plus the ``job_id`` it ran under."""

    job_id: str
    filename: str
    chunks_processed: int
    nodes_created: int
    relationships_created: int
    communities: int
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, job_id: str, payload: dict[str, Any]) -> IngestResult:
        return cls(
            job_id=job_id,
            filename=str(payload.get("filename") or ""),
            chunks_processed=int(payload.get("chunks_processed") or 0),
            nodes_created=int(payload.get("nodes_created") or 0),
            relationships_created=int(payload.get("relationships_created") or 0),
            communities=int(payload.get("communities") or 0),
            data=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **self.data}


# ── Server-Sent Events ───────────────────────────────────────────────────────
async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Yield one decoded JSON object per SSE frame.

    The backend writes ``data: {json}\\n\\n`` frames, but HTTP delivers bytes in
    arbitrary pieces — a frame can arrive split anywhere, including inside a
    multi-byte character. Bytes are buffered and cut on newlines, so decoding
    only ever sees whole lines. ``:`` comments and non-``data`` fields are
    ignored; multi-line ``data`` is joined with newlines per the SSE spec, and
    ``\\r\\n`` line endings are accepted.
    """
    buffer = b""
    data_lines: list[str] = []

    def take(line: bytes) -> dict[str, Any] | None:
        """Feed one line; return a frame when a blank line closes it."""
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            if not data_lines:
                return None
            raw = "\n".join(data_lines)
            data_lines.clear()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SynapseError(None, f"Malformed SSE frame: {raw[:200]!r}") from exc
        if line.startswith(b":"):
            return None
        name, _, value = line.partition(b":")
        if name == b"data":
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value.decode("utf-8"))
        return None

    async for chunk in chunks:
        buffer += chunk
        while (newline := buffer.find(b"\n")) >= 0:
            line, buffer = buffer[:newline], buffer[newline + 1 :]
            frame = take(line)
            if frame is not None:
                yield frame

    # A stream that ends without a trailing blank line still holds a frame.
    if buffer:
        take(buffer)
    if data_lines:
        frame = take(b"")
        if frame is not None:
            yield frame


async def _notify(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def _format_detail(payload: Any) -> str:
    """Render FastAPI's ``detail`` — a string, a list of validation errors, or a
    ``{"message", "diagnostics"}`` object (an invalid procedural graph)."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and "message" in payload:
        message = str(payload.get("message") or "")
        raw = payload.get("diagnostics")
        diagnostics = [str(d) for d in raw] if isinstance(raw, list) else []
        if not diagnostics:
            return message
        return f"{message}: {'; '.join(diagnostics)}" if message else "; ".join(diagnostics)
    if isinstance(payload, list):
        parts = []
        for err in payload:
            if isinstance(err, dict):
                loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
                parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
            else:
                parts.append(str(err))
        return "; ".join(parts)
    return json.dumps(payload)


# ── Client ───────────────────────────────────────────────────────────────────
class SynapseClient:
    """Async client for one Synapse backend.

    Use it as an async context manager (or call :meth:`aclose`). ``transport``
    lets tests plug an ``httpx.MockTransport`` in; everything else comes from
    the arguments or, when omitted, the environment (see the module docstring).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or env_url()).rstrip("/")
        self.api_key = api_key if api_key is not None else env_api_key()
        self.timeout = float(timeout) if timeout is not None else env_timeout()

        from synapse_graphrag import __version__  # lazy: avoids an import cycle

        headers = {"User-Agent": f"synapse-graphrag/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=min(CONNECT_TIMEOUT, self.timeout)),
            transport=transport,
        )

    async def __aenter__(self) -> SynapseClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── plumbing ─────────────────────────────────────────────────────────
    def _connection_error(self, exc: Exception) -> SynapseConnectionError:
        reason = str(exc) or type(exc).__name__  # httpx timeouts often carry no message
        return SynapseConnectionError(
            None,
            f"Cannot reach Synapse at {self.base_url}: {reason}. Is it running? (`make up`)",
        )

    def _transport_error(self, exc: httpx.TransportError) -> SynapseError:
        """One readable sentence per transport failure.

        Only a connection that never opened earns the "is it running?" hint. A
        backend that answered and then stalled (read timeout — ingestion and
        ``ask`` can be slow) or dropped the stream half-way is a different
        problem, and the message should send people to a different fix.
        """
        if isinstance(exc, httpx.ConnectTimeout):
            return self._connection_error(exc)
        if isinstance(exc, httpx.TimeoutException):
            return SynapseError(
                None,
                f"Synapse at {self.base_url} did not respond within {self.timeout:g}s "
                "(raise SYNAPSE_TIMEOUT).",
            )
        if isinstance(exc, httpx.RemoteProtocolError):
            reason = f": {exc}" if str(exc) else ""
            return SynapseError(
                None, f"Synapse at {self.base_url} closed the connection mid-stream{reason}."
            )
        return self._connection_error(exc)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail: str = response.text
        raw_detail: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "detail" in payload:
            raw_detail = payload["detail"]
            detail = _format_detail(raw_detail)
        raise SynapseError(response.status_code, detail or response.reason_phrase, raw_detail)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise self._transport_error(exc) from exc
        except httpx.HTTPError as exc:
            raise SynapseError(None, str(exc)) from exc
        self._raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise SynapseError(
                response.status_code, f"Expected JSON from {path}, got: {response.text[:200]!r}"
            ) from exc

    async def _stream(self, method: str, path: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        headers = {"Accept": "text/event-stream", **kwargs.pop("headers", {})}
        try:
            async with self._http.stream(method, path, headers=headers, **kwargs) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)
                # ``aclosing`` closes the inner generators when this one is
                # closed, so an early exit upstream tears the chain down in
                # order instead of leaving it to the event loop's finaliser.
                async with (
                    aclosing(response.aiter_bytes()) as chunks,
                    aclosing(iter_sse(chunks)) as events,
                ):
                    async for event in events:
                        yield event
        except httpx.TransportError as exc:
            raise self._transport_error(exc) from exc
        except httpx.HTTPError as exc:
            raise SynapseError(None, str(exc)) from exc

    async def _follow_job(
        self,
        events_path: str,
        on_progress: ProgressCallback | None,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        """Consume a job's SSE stream until ``done``; forward the rest to ``on_progress``.

        ``timeout`` overrides the client's for this stream only (its read
        timeout is the longest silence tolerated between two events).
        """
        extra: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        async with aclosing(self._stream("GET", events_path, **extra)) as events:
            async for event in events:
                kind = event.get("type")
                if kind == "done":
                    return _as_dict(event.get("data"))
                if kind == "error":
                    raise SynapseError(None, str(event.get("data") or "Job failed."))
                await _notify(on_progress, event)
        raise SynapseError(None, f"Event stream {events_path} ended without a done event.")

    # ── health & metadata ────────────────────────────────────────────────
    async def health(self) -> dict[str, Any]:
        """``GET /health`` — liveness."""
        return _as_dict(await self._request("GET", "/health"))

    async def ready(self) -> dict[str, Any]:
        """``GET /health/ready`` — Neo4j reachability and the configured LLM provider."""
        return _as_dict(await self._request("GET", "/health/ready"))

    async def about(self) -> dict[str, Any]:
        """``GET /api/about`` — version, providers, licensing."""
        return _as_dict(await self._request("GET", "/api/about"))

    # ── retrieval & answers ──────────────────────────────────────────────
    async def retrieve(
        self, query: str, k: int = 8, max_context_chars: int | None = None
    ) -> Retrieval:
        """``POST /api/retrieve`` — GraphRAG context, optionally cut to a character budget.

        No LLM generation happens on the backend; only embedding + graph
        queries. ``usage`` reports ``context_chars``, ``context_tokens_est``
        and whether the budget truncated the context.
        """
        body: dict[str, Any] = {"query": query, "k": k}
        if max_context_chars is not None:
            body["max_context_chars"] = max_context_chars
        payload = await self._request("POST", "/api/retrieve", json=body)
        return Retrieval.from_payload(_as_dict(payload))

    async def ask_stream(
        self, query: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """``POST /api/chat`` — yield the raw events (``citations``, ``token``, … ``done``).

        Stopping early (``break``) leaves the generator suspended; iterate it
        under ``contextlib.aclosing`` so the HTTP stream is closed right away
        rather than by the event loop at shutdown. :meth:`ask` does this.
        """
        turns = [
            {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))}
            for turn in (history or [])
        ]
        async with aclosing(
            self._stream("POST", "/api/chat", json={"query": query, "history": turns})
        ) as events:
            async for event in events:
                yield event

    async def ask(self, query: str, history: list[dict[str, str]] | None = None) -> Answer:
        """Run :meth:`ask_stream` to completion and aggregate it into an :class:`Answer`."""
        tokens: list[str] = []
        citations: list[dict[str, Any]] = []
        paths: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        done = False
        async with aclosing(self.ask_stream(query, history)) as events:
            async for event in events:
                kind = event.get("type")
                if kind == "token":
                    tokens.append(str(event.get("data", "")))
                elif kind == "citations":
                    citations = _as_list(event.get("data"))
                elif kind == "paths":
                    paths = _as_list(event.get("data"))
                elif kind == "sources":
                    sources = _as_list(event.get("data"))
                elif kind == "done":
                    usage = _as_dict(event.get("usage"))
                    done = True
                    break
                elif kind == "error":
                    raise SynapseError(None, str(event.get("data") or "Generation failed."))
        if not done:
            # A stream that stops without ``done`` is a partial answer, not an answer.
            raise SynapseError(None, "Chat stream ended without a done event.")
        text = "".join(tokens)
        usage.setdefault("answer_chars", len(text))
        return Answer(text=text, citations=citations, paths=paths, sources=sources, usage=usage)

    # ── ingestion ────────────────────────────────────────────────────────
    async def ingest_pdf(
        self,
        path: str | Path,
        theme: str = "Generic",
        on_progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Upload a PDF and follow its job until ``done``.

        ``on_progress`` receives every backend event (``progress``,
        ``community_progress``) plus one client-side ``accepted`` event carrying
        the upload response (``job_id``, ``total_chunks``) so a UI can announce
        the job before the first progress frame arrives.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise SynapseError(None, f"File not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise SynapseError(None, f"Only PDF files are supported: {file_path.name}")

        content = await asyncio.to_thread(file_path.read_bytes)
        accepted = _as_dict(
            await self._request(
                "POST",
                "/api/upload",
                files={"file": (file_path.name, content, "application/pdf")},
                data={"theme": theme},
            )
        )
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise SynapseError(None, f"Upload accepted without a job_id: {accepted}")
        await _notify(on_progress, {"type": "accepted", **accepted})
        done = await self._follow_job(f"/api/upload/{job_id}/events", on_progress)
        return IngestResult.from_payload(job_id, done)

    # ── graph & communities ──────────────────────────────────────────────
    async def graph(self) -> dict[str, Any]:
        """``GET /api/graph-data`` — ``{"nodes": [...], "links": [...]}``."""
        return _as_dict(await self._request("GET", "/api/graph-data"))

    async def communities(self, limit: int = 20) -> dict[str, Any]:
        """``GET /api/communities`` — the largest community summaries (corpus themes)."""
        return _as_dict(await self._request("GET", "/api/communities", params={"limit": limit}))

    async def rebuild_communities(
        self, on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """``POST /api/communities/rebuild`` and follow the job; returns its ``done`` payload."""
        accepted = _as_dict(await self._request("POST", "/api/communities/rebuild"))
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise SynapseError(None, f"Rebuild accepted without a job_id: {accepted}")
        done = await self._follow_job(f"/api/communities/rebuild/{job_id}/events", on_progress)
        return {"job_id": job_id, **done}

    async def clear_graph(self) -> dict[str, Any]:
        """``DELETE /api/graph`` — remove the knowledge graph (entities, relationships,
        communities, source excerpts). Irreversible. Backends with procedural memory
        keep their procedural graphs; :meth:`delete_procedure` removes one."""
        return _as_dict(await self._request("DELETE", "/api/graph"))

    # ── procedural memory ────────────────────────────────────────────────
    async def procedures(self) -> dict[str, Any]:
        """``GET /api/procedures`` — ``{"graphs": [{name, version, score, nodes, edges,
        updated_at, description}]}``."""
        return _as_dict(await self._request("GET", "/api/procedures"))

    async def procedure(self, name: str) -> dict[str, Any]:
        """``GET /api/procedures/{name}`` — the graph JSON (``name``, ``description``,
        ``cycle_policy``, ``tools``, ``nodes``, ``edges``) plus ``version`` and ``score``."""
        return _as_dict(await self._request("GET", _procedure_path(name)))

    async def procedure_text(self, name: str) -> str:
        """``GET /api/procedures/{name}?format=text`` — the whole graph serialized the way
        an LLM reads it (every node, then every transition with its condition,
        guidance and pitfalls)."""
        payload = _as_dict(
            await self._request("GET", _procedure_path(name), params={"format": "text"})
        )
        return str(payload.get("text") or "")

    async def procedure_graph_data(self, name: str) -> dict[str, Any]:
        """``GET /api/procedures/{name}/graph-data`` — ``{"nodes", "links", "version",
        "score"}`` in the force-graph shape the UI draws."""
        return _as_dict(await self._request("GET", _procedure_path(name, "/graph-data")))

    async def put_procedure(self, name: str, graph: dict[str, Any]) -> dict[str, Any]:
        """``PUT /api/procedures/{name}`` — store ``graph`` as a new version.

        The backend validates it first (a ``Start`` node, a terminal, every node
        able to reach a terminal, no cycle unless ``cycle_policy`` allows it…);
        a refusal raises :class:`SynapseError` with status 422, the diagnostics
        joined in ``detail`` and listed in ``payload["diagnostics"]``. The body's
        ``name`` must match ``name`` or be absent. Returns ``{"name", "version"}``.
        """
        if not isinstance(graph, dict):
            raise SynapseError(None, "A procedural graph must be a JSON object.")
        return _as_dict(await self._request("PUT", _procedure_path(name), json=graph))

    async def delete_procedure(self, name: str) -> dict[str, Any]:
        """``DELETE /api/procedures/{name}`` — the graph with its versions, rejections and
        recorded trajectories. Irreversible."""
        return _as_dict(await self._request("DELETE", _procedure_path(name)))

    async def procedure_guidance(
        self,
        name: str,
        query: str,
        trajectory: Iterable[dict[str, Any]] | None = None,
        mode: str | None = None,
        hops: int | None = None,
        window: int | None = None,
    ) -> dict[str, Any]:
        """``POST /api/procedures/{name}/guidance`` — step-local procedural guidance.

        ``trajectory`` is the steps taken so far, oldest first, each
        ``{"action": str, "observation"?: str}`` (other keys — an agent step's
        ``thought`` or ``args`` — are dropped, so a navigator trace can be passed
        as is). The backend localises the last action to a node and returns
        ``{graph, version, active_node, localization, scope, context, guidance,
        next_actions, usage}``. ``mode``: ``"raw"`` (the subgraph as text, no LLM
        call), ``"generative"`` (one backend LLM call, cached) or ``"none"``;
        omitted, the backend's default applies.
        """
        body: dict[str, Any] = {"query": query, "trajectory": _trajectory_steps(trajectory)}
        if mode is not None:
            body["mode"] = mode
        if hops is not None:
            body["hops"] = hops
        if window is not None:
            body["window"] = window
        return _as_dict(
            await self._request("POST", _procedure_path(name, "/guidance"), json=body)
        )

    async def record_trajectory(
        self,
        name: str,
        query: str,
        steps: Iterable[dict[str, Any]],
        score: float,
        source: str = "client",
    ) -> dict[str, Any]:
        """``POST /api/procedures/{name}/trajectories`` — store one scored run.

        ``score`` is in ``[0, 1]`` (checked here, before any request) and
        ``source`` says who recorded it. Recorded runs are kept for audit and
        future learning; the evolution loop does not read them yet (it learns
        from its own rollouts). An honest score matters more than a high one.
        """
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise SynapseError(None, f"score must be a number in [0, 1], got {score!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise SynapseError(None, f"score must be in [0, 1], got {value:g}")
        body = {
            "query": query,
            "steps": [dict(step) for step in steps],
            "score": value,
            "source": source,
        }
        return _as_dict(
            await self._request("POST", _procedure_path(name, "/trajectories"), json=body)
        )

    async def procedure_versions(self, name: str) -> dict[str, Any]:
        """``GET /api/procedures/{name}/versions`` — ``{"versions": [{version, score,
        accepted, created_at, note, diff}]}``."""
        return _as_dict(await self._request("GET", _procedure_path(name, "/versions")))

    async def rollback_procedure(self, name: str, version: int) -> dict[str, Any]:
        """``POST /api/procedures/{name}/rollback`` — re-save ``version`` as a NEW version
        (history is never rewritten). Returns ``{"name", "version"}``."""
        return _as_dict(
            await self._request(
                "POST", _procedure_path(name, "/rollback"), json={"version": int(version)}
            )
        )

    async def procedure_rejections(self, name: str, limit: int = 20) -> dict[str, Any]:
        """``GET /api/procedures/{name}/rejections`` — the evolution loop's rejection
        memory, newest first: ``{"rejections": [...]}``."""
        return _as_dict(
            await self._request(
                "GET", _procedure_path(name, "/rejections"), params={"limit": limit}
            )
        )

    async def evolve_procedure(
        self,
        name: str,
        train: Iterable[dict[str, Any]],
        val: Iterable[dict[str, Any]],
        *,
        rounds: int | None = None,
        batch_size: int | None = None,
        mode: str = "static",
        metric: str = "f1",
        guidance: str = "raw",
        max_llm_calls: int | None = None,
        on_progress: ProgressCallback | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """``POST /api/procedures/{name}/evolve`` and follow the job to its report.

        Offline self-evolution (Algorithm 1 of the paper): roll the navigator
        agent out on ``train`` batches, have an LLM propose graph edits, keep a
        candidate only if its ``val`` score does not drop. It COSTS BACKEND LLM
        CALLS — every rollout step, every generative guidance call and every
        refinement; ``max_llm_calls`` caps them (omitted: the backend's budget).
        A cap too small for the baseline plus one full round is refused before
        anything is spent: status 422, the smallest sufficient cap in
        ``payload["minimum_max_llm_calls"]``.

        ``mode="scratch"`` on a name that already holds a graph is refused
        (status 409) unless ``replace=True``: the stored graph is then scored
        once on ``val`` (it counts in the budget) and only a candidate scoring
        at least as well can overwrite it. ``replace`` is sent only when true.

        ``train`` / ``val`` are ``{"question", "answer"}`` items (other keys are
        dropped). ``on_progress`` receives a client-side ``accepted`` event with
        the ``job_id``, then every backend ``progress`` event. Returns the report
        (``baseline_score``, ``final_score``, ``final_version``, ``rounds``,
        ``stopped``, ``llm_calls``, ``effect_floor``…) with the ``job_id``.

        The event stream has no idle timeout: a rollout batch or a refinement can
        be minutes apart. Stopping to listen does not stop the job — the backend
        runs it to completion or to its budget.
        """
        body: dict[str, Any] = {
            "train": qa_items(train, "train"),
            "val": qa_items(val, "val"),
            "mode": mode,
            "metric": metric,
            "guidance": guidance,
        }
        if rounds is not None:
            body["rounds"] = rounds
        if batch_size is not None:
            body["batch_size"] = batch_size
        if max_llm_calls is not None:
            body["max_llm_calls"] = max_llm_calls
        if replace:
            body["replace"] = True
        accepted = _as_dict(
            await self._request("POST", _procedure_path(name, "/evolve"), json=body)
        )
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise SynapseError(None, f"Evolution accepted without a job_id: {accepted}")
        await _notify(on_progress, {"type": "accepted", **accepted})
        done = await self._follow_job(
            f"/api/procedures/evolve/{quote(job_id, safe='')}/events",
            on_progress,
            timeout=httpx.Timeout(None, connect=min(CONNECT_TIMEOUT, self.timeout)),
        )
        return {"job_id": job_id, **done}

    # ── navigator agent ──────────────────────────────────────────────────
    async def agent_ask(
        self,
        query: str,
        graph: str | None | EllipsisType = ...,
        guidance: str | None = None,
        max_steps: int | None = None,
        record: bool = False,
    ) -> dict[str, Any]:
        """``POST /api/agent/ask`` — the backend's GraphRAG Navigator answers by walking
        the knowledge graph with deterministic tools, one LLM call per step.

        ``graph``: ``...`` (default) lets the backend use its default procedural
        graph, ``None`` runs without one, a name picks one. ``guidance``:
        ``"none"`` / ``"raw"`` / ``"generative"`` (omitted: the backend default).
        ``record=True`` stores the trajectory, unscored, when a procedural graph
        was used. Returns the agent result: ``answer``, ``steps`` [{thought,
        action, args, observation, guidance_context_chars, localization}],
        ``stopped``, ``parse_failures``, ``usage``, ``graph``, ``latency_s`` and
        ``recorded``. Not streamed: the request waits for the whole run, so give
        the client a long timeout (the CLI uses :func:`env_long_timeout`). A
        client that times out or disconnects does not stop the run; the backend
        keeps working, and spending, until it ends.
        """
        body: dict[str, Any] = {"query": query, "record": bool(record)}
        if graph is not ...:
            body["graph"] = graph
        if guidance is not None:
            body["guidance"] = guidance
        if max_steps is not None:
            body["max_steps"] = max_steps
        return _as_dict(await self._request("POST", "/api/agent/ask", json=body))


def _procedure_path(name: str, suffix: str = "") -> str:
    """``/api/procedures/{name}{suffix}`` with ``name`` percent-encoded as ONE segment."""
    if not isinstance(name, str) or not name.strip():
        raise SynapseError(None, "A procedural graph name is required.")
    return f"/api/procedures/{quote(name, safe='')}{suffix}"


def _trajectory_steps(steps: Iterable[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Reduce steps to the guidance endpoint's ``{"action", "observation"?}`` shape."""
    reduced: list[dict[str, str]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            raise SynapseError(None, f"Each trajectory step must be an object, got {step!r}")
        entry = {"action": str(step.get("action") or "")}
        observation = step.get("observation")
        if observation is not None:
            entry["observation"] = str(observation)
        reduced.append(entry)
    return reduced


def qa_items(items: Iterable[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Validate ``{"question", "answer"}`` items before anything is sent (or spent).

    ``answer`` is one gold answer or a list of acceptable ones (the backend
    scores against the best match); every other key is dropped.
    """
    reduced: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SynapseError(None, f"{label}[{index}] must be an object, got {item!r}")
        question = str(item.get("question") or "").strip()
        raw = item.get("answer")
        answer: str | list[str]
        if isinstance(raw, list):
            answer = [str(a).strip() for a in raw if a is not None and str(a).strip()]
        else:
            answer = "" if raw is None else str(raw).strip()
        if not question or not answer:
            raise SynapseError(
                None, f"{label}[{index}] needs a non-empty 'question' and 'answer'"
            )
        reduced.append({"question": question, "answer": answer})
    if not reduced:
        raise SynapseError(None, f"{label} must hold at least one {{question, answer}} item")
    return reduced


T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine to completion — the CLI's bridge into the async client."""
    return asyncio.run(coro)
