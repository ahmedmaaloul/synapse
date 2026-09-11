# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — MCP server

Exposes a Synapse knowledge graph to any Model Context Protocol host: Claude
Code, Claude Desktop, Cursor, VS Code, Windsurf, or anything that speaks stdio
or streamable HTTP.

The design centre is FinOps. The host already pays for a model, so the
flagship tool ``synapse_retrieve`` returns *budgeted* GraphRAG context
(``max_context_chars``) with ``usage`` accounting and lets the host's model
write the answer — no second generation bill. ``synapse_ask`` exists for hosts
that want the backend's own grounded answer and accept the extra LLM call.
Identical retrievals within ``SYNAPSE_CACHE_TTL`` seconds are served from an
in-process cache (``usage.cached``), because agents re-ask the same question
several times per task.

Error policy — one rule, applied everywhere: an anticipated failure (backend
unreachable, HTTP error, bad path, unknown job…) is raised as the SDK's
``ToolError``. The SDK returns it as ``CallToolResult(is_error=True)`` whose
text is the message, so the model reads one readable sentence, never a
traceback. Tools never return ``{"error": …}`` dicts; a *refusal* (clearing
the graph without ``confirm``) is a normal result that says so.

stdout is the stdio transport, so logging goes to stderr only.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from synapse_graphrag import __version__
from synapse_graphrag.client import (
    MIN_CONTEXT_CHARS,
    SynapseClient,
    SynapseError,
    env_cache_ttl,
    env_max_context_chars,
    env_url,
)

logger = logging.getLogger("synapse_graphrag.mcp")

SERVER_NAME = "synapse"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CACHE_MAX_ENTRIES = 256

ClientFactory = Callable[[], SynapseClient]

# Module level on purpose: with ``from __future__ import annotations`` the SDK
# resolves parameter annotations against the function's globals.
QueryArg = Annotated[str, Field(min_length=1, description="The question or topic to look up.")]

INSTRUCTIONS = """\
Synapse is a GraphRAG knowledge graph built from the user's documents (entities,
relationships, community summaries and source excerpts stored in Neo4j).

Prefer `synapse_retrieve`: it returns graph context within a character budget and
you write the answer yourself — no second LLM call is billed. Answer only from the
returned context, cite the entity names you relied on, and say plainly when the
graph does not contain the answer. Use `synapse_ask` only when the user explicitly
wants Synapse's own generated answer. `synapse_find_entities`, `synapse_communities`
and `synapse_graph_stats` are cheap ways to discover what the graph knows before
retrieving. `synapse_ingest_pdf` adds documents; `synapse_clear_graph` deletes
everything and requires confirm=true."""

ANSWER_WITH_GRAPH_PROMPT = """\
Answer the question below using the Synapse knowledge graph.

1. Call the `synapse_retrieve` tool with the question as `query` (raise `k` for broad
   questions, lower `max_context_chars` when tokens are scarce).
2. Answer ONLY from the returned `context`. Do not add outside knowledge.
3. Cite the entities you relied on by name — the `citations` list gives their names
   and types; mention reasoning `paths` when they connect the answer.
4. If the context does not contain the answer, say so explicitly and suggest what
   document or entity would need to be ingested. Never guess.
5. Note `usage.truncated`: if true, the context was cut to the budget — say the
   answer may be partial, or retrieve again with a larger `max_context_chars`.

Question: {question}"""

SAFETY_BRIEF_PROMPT = """\
Write an AI-safety brief on the topic below from the Synapse knowledge graph. Corpora ingested
with the "AI Safety" theme carry RISK, FAILURE_MODE, MITIGATION, EVALUATION, BENCHMARK, INCIDENT
and POLICY entities, and every RISK / FAILURE_MODE description starts with "demonstrated:" or
"hypothesised:".

1. Call `synapse_communities` once to see the corpus' themes, then `synapse_retrieve` with the
   topic as `query` (raise `k` to 12 for a broad topic; retrieve again with a narrower query for
   each risk or mitigation you need evidence on).
2. Structure the brief as: **Claims** · **Evidence** (label each item `demonstrated` or
   `hypothesised`, as the entity descriptions do) · **Mitigations and their evaluation status**
   (which MITIGATES edges also have an EVALUATED_BY or MEASURES path, and which do not) ·
   **Open questions** · **Not in the corpus**.
3. Every line must cite an entity name from `citations` or quote a `sources` excerpt. Nothing
   in the brief may come from outside the returned context.
4. Put anything the graph cannot support under "Not in the corpus" instead of filling it in,
   and say which documents would need to be ingested.
5. If `usage.truncated` is true, say the brief may be partial.

Topic: {topic}"""


# ── Retrieval cache ──────────────────────────────────────────────────────────
class _TTLCache:
    """Tiny (key → payload) cache with per-entry expiry; ``ttl <= 0`` disables it."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._items: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        if self.ttl <= 0:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() >= expires_at:
            self._items.pop(key, None)
            return None
        return copy.deepcopy(value)

    def set(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        if self.ttl <= 0:
            return
        if len(self._items) >= CACHE_MAX_ENTRIES:
            oldest = min(self._items, key=lambda k: self._items[k][0])
            self._items.pop(oldest, None)
        self._items[key] = (time.monotonic() + self.ttl, copy.deepcopy(value))

    def clear(self) -> None:
        self._items.clear()


# ── Pure graph helpers (shared with the CLI) ─────────────────────────────────
def _degrees(graph: dict[str, Any]) -> Counter[str]:
    degrees: Counter[str] = Counter()
    for link in graph.get("links") or []:
        degrees[str(link.get("source"))] += 1
        degrees[str(link.get("target"))] += 1
    return degrees


def find_entities(graph: dict[str, Any], query: str, limit: int = 20) -> dict[str, Any]:
    """Case-insensitive substring match over node labels and types of ``/api/graph-data``."""
    needle = query.strip().lower()
    degrees = _degrees(graph)
    matches: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        label = str(node.get("label") or "")
        kind = str(node.get("type") or "")
        if needle and needle not in label.lower() and needle not in kind.lower():
            continue
        node_id = str(node.get("id"))
        entry: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "type": kind,
            "degree": degrees.get(node_id, 0),
        }
        description = (node.get("properties") or {}).get("description")
        if description:
            entry["description"] = str(description)
        matches.append(entry)
    matches.sort(key=lambda e: (-e["degree"], e["label"].lower()))
    return {
        "query": query,
        "total_matches": len(matches),
        "returned": min(limit, len(matches)),
        "entities": matches[:limit],
    }


def graph_stats(graph: dict[str, Any], top: int = 5) -> dict[str, Any]:
    """Node/edge counts and per-type breakdowns of a ``/api/graph-data`` payload."""
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    degrees = _degrees(graph)
    by_id = {str(n.get("id")): n for n in nodes}
    top_entities = [
        {
            "label": str(by_id[node_id].get("label") or ""),
            "type": str(by_id[node_id].get("type") or ""),
            "degree": degree,
        }
        for node_id, degree in degrees.most_common()
        if node_id in by_id
    ][:top]
    return {
        "nodes": len(nodes),
        "edges": len(links),
        "isolated_nodes": sum(1 for n in nodes if degrees.get(str(n.get("id")), 0) == 0),
        "entity_types": dict(Counter(str(n.get("type") or "Unknown") for n in nodes).most_common()),
        "relationship_types": dict(
            Counter(str(link.get("type") or "RELATED_TO") for link in links).most_common()
        ),
        "top_entities": top_entities,
    }


# ── Server ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _connected(client_factory: ClientFactory) -> AsyncIterator[SynapseClient]:
    """One client per tool call; every client failure becomes a readable ``ToolError``."""
    try:
        async with client_factory() as client:
            yield client
    except SynapseError as exc:
        raise ToolError(str(exc)) from exc


def _env_budget() -> int | None:
    """``SYNAPSE_MAX_CONTEXT_CHARS`` as a ``ToolError`` when it is malformed.

    Read per call, outside ``_connected`` (the cache key needs it before any
    HTTP happens), so the env error has to be mapped here to stay readable.
    """
    try:
        return env_max_context_chars()
    except SynapseError as exc:
        raise ToolError(str(exc)) from exc


def build_server(
    client_factory: ClientFactory | None = None,
    *,
    cache_ttl: float | None = None,
) -> MCPServer:
    """Create the ``synapse`` MCP server.

    ``client_factory`` builds the :class:`SynapseClient` used for each call
    (default: one configured from the environment) — tests pass a factory
    wired to an ``httpx.MockTransport``. ``cache_ttl`` overrides
    ``SYNAPSE_CACHE_TTL`` for the retrieval cache.
    """
    factory: ClientFactory = client_factory or SynapseClient
    cache = _TTLCache(env_cache_ttl() if cache_ttl is None else cache_ttl)

    server = MCPServer(
        name=SERVER_NAME,
        title="Synapse GraphRAG",
        version=__version__,
        instructions=INSTRUCTIONS,
        website_url="https://github.com/ahmedmaaloul/synapse",
    )

    @server.tool(
        name="synapse_retrieve",
        description=(
            "Retrieve GraphRAG context for a question from the Synapse knowledge graph "
            "WITHOUT generating an answer — you write the answer from the returned context. "
            "Returns: mode (local entity subgraph or global community summaries), context "
            "(text cut to max_context_chars), citations (entity names/types to cite), "
            "reasoning paths, source excerpts, and usage (context_chars, context_tokens_est, "
            "truncated, cached). Cheap: embedding + graph queries only, no LLM call on the "
            "backend; identical calls are cached for a few minutes."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_retrieve(
        query: QueryArg,
        k: Annotated[int, Field(ge=1, le=20, description="Number of seed entities (1-20).")] = 8,
        max_context_chars: Annotated[
            int | None,
            Field(
                ge=MIN_CONTEXT_CHARS,
                description=(
                    "Character budget for the context (default: SYNAPSE_MAX_CONTEXT_CHARS, "
                    "6000). Lower it when tokens are scarce."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        budget = max_context_chars if max_context_chars is not None else _env_budget()
        key = (query, k, budget)
        cached = cache.get(key)
        if cached is not None:
            cached["usage"]["cached"] = True
            return cached
        async with _connected(factory) as client:
            retrieval = await client.retrieve(query, k=k, max_context_chars=budget)
        payload = retrieval.to_dict()
        payload["usage"] = {**payload["usage"], "cached": False}
        cache.set(key, payload)
        return payload

    @server.tool(
        name="synapse_ask",
        description=(
            "Ask Synapse for a complete, grounded answer generated by the backend's own LLM. "
            "This costs a SECOND LLM call (the backend's) on top of yours — prefer "
            "synapse_retrieve and answer from its context unless the user explicitly wants "
            "Synapse's answer. Returns text, citations, paths, sources and usage."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=False, open_world_hint=False),
    )
    async def synapse_ask(
        query: QueryArg,
        history: Annotated[
            list[dict[str, str]] | None,
            Field(description="Prior turns as [{role: user|assistant, content: str}]."),
        ] = None,
    ) -> dict[str, Any]:
        async with _connected(factory) as client:
            answer = await client.ask(query, history)
        return answer.to_dict()

    @server.tool(
        name="synapse_ingest_pdf",
        description=(
            "Ingest a local PDF into the knowledge graph: parse, extract entities and "
            "relationships with the backend's LLM, embed, write to Neo4j and re-detect "
            "communities. Slow (minutes for large files) and it costs backend LLM calls. "
            "Returns the job summary (nodes/relationships created, communities)."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False),
    )
    async def synapse_ingest_pdf(
        path: Annotated[str, Field(description="Absolute or relative path to a .pdf file on this machine.")],
        theme: Annotated[str, Field(description="Domain hint for extraction, e.g. 'Legal', 'Medical'.")] = "Generic",
    ) -> dict[str, Any]:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise ToolError(f"File not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ToolError(f"Only PDF files can be ingested, got: {file_path.name}")
        async with _connected(factory) as client:
            result = await client.ingest_pdf(file_path, theme=theme)
        return result.to_dict()

    @server.tool(
        name="synapse_communities",
        description=(
            "List the corpus' detected themes: Louvain communities with an LLM-written title "
            "and summary, largest first. Use it to learn what the graph is about before asking."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_communities(
        limit: Annotated[int, Field(ge=1, le=100, description="How many communities (1-100).")] = 10,
    ) -> dict[str, Any]:
        async with _connected(factory) as client:
            return await client.communities(limit=limit)

    @server.tool(
        name="synapse_find_entities",
        description=(
            "Find entities whose name or type contains the query (case-insensitive substring). "
            "Returns id, label, type, description when present, and degree (number of "
            "relationships), best-connected first."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_find_entities(
        query: Annotated[str, Field(min_length=1, description="Substring to match against names and types.")],
        limit: Annotated[int, Field(ge=1, le=200, description="Maximum entities to return.")] = 20,
    ) -> dict[str, Any]:
        async with _connected(factory) as client:
            graph = await client.graph()
        return find_entities(graph, query, limit)

    @server.tool(
        name="synapse_graph_stats",
        description=(
            "Size and shape of the knowledge graph: node and edge counts, entities per type, "
            "relationships per type, isolated nodes and the best-connected entities."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_graph_stats() -> dict[str, Any]:
        async with _connected(factory) as client:
            graph = await client.graph()
        return graph_stats(graph)

    @server.tool(
        name="synapse_status",
        description=(
            "Check the Synapse backend: liveness, readiness (Neo4j up?), version, and which "
            "LLM / embedding providers it runs. Call this first when another tool fails."
        ),
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_status() -> dict[str, Any]:
        async with _connected(factory) as client:
            health = await client.health()
            ready = await client.ready()
            about = await client.about()
            url = client.base_url
        return {
            "url": url,
            "health": health.get("status"),
            "ready": ready.get("status"),
            "neo4j": ready.get("neo4j"),
            "version": about.get("version"),
            "llm_provider": about.get("llm_provider"),
            "embedding_provider": about.get("embedding_provider"),
            "client_version": __version__,
        }

    @server.tool(
        name="synapse_clear_graph",
        description=(
            "DELETE every node and relationship in the knowledge graph. Irreversible. "
            "Refuses unless confirm=true; ask the user before confirming."
        ),
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False),
    )
    async def synapse_clear_graph(
        confirm: Annotated[bool, Field(description="Must be true to actually delete the graph.")] = False,
    ) -> dict[str, Any]:
        if not confirm:
            return {
                "cleared": False,
                "message": (
                    "Refused: this deletes the whole knowledge graph and cannot be undone. "
                    "Call again with confirm=true once the user has agreed."
                ),
            }
        async with _connected(factory) as client:
            result = await client.clear_graph()
        cache.clear()
        return {"cleared": True, **result}

    @server.resource(
        "synapse://about",
        name="about",
        title="About this Synapse instance",
        description="Backend metadata: version, providers, authorship and license (JSON).",
        mime_type="application/json",
    )
    async def about_resource() -> str:
        try:
            async with factory() as client:
                about = await client.about()
        except SynapseError as exc:
            raise ResourceError(str(exc)) from exc
        return json.dumps(about, indent=2)

    @server.prompt(
        name="answer_with_graph",
        title="Answer with the knowledge graph",
        description="Retrieve budgeted context with synapse_retrieve and answer only from it, citing entities.",
    )
    def answer_with_graph(question: str) -> str:
        return ANSWER_WITH_GRAPH_PROMPT.format(question=question)

    @server.prompt(
        name="safety_brief",
        title="AI-safety brief from the knowledge graph",
        description=(
            "Retrieve evidence on a topic and write a structured, cited AI-safety brief: "
            "claims, evidence (demonstrated vs hypothesised), mitigations and their evaluation "
            "status, open questions, and what is not in the corpus."
        ),
    )
    def safety_brief(topic: str) -> str:
        return SAFETY_BRIEF_PROMPT.format(topic=topic)

    return server


# ── Entry point ──────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapse-mcp",
        description="Run the Synapse MCP server (stdio by default, or streamable HTTP).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio for local hosts (Claude Code, Cursor…); streamable-http exposes /mcp.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address for streamable-http.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for streamable-http.")
    parser.add_argument("--url", help="Synapse backend URL (overrides SYNAPSE_URL).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _uvicorn_logs_to_stderr() -> None:
    """Uvicorn's default access log targets stdout; keep every log line on stderr.

    The SDK builds ``uvicorn.Config`` with uvicorn's module-level default
    ``LOGGING_CONFIG``, so mutating that dict before ``run`` is the only hook.
    """
    try:
        from uvicorn.config import LOGGING_CONFIG
    except ImportError:  # pragma: no cover — uvicorn ships with mcp on every platform we support
        return
    LOGGING_CONFIG["handlers"]["access"]["stream"] = "ext://sys.stderr"


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``synapse-mcp``). Logs to stderr; stdout is the transport."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    if args.url:
        os.environ["SYNAPSE_URL"] = args.url

    try:
        server = build_server()
        if args.transport == "stdio":
            logger.info("Synapse MCP server %s on stdio → backend %s", __version__, env_url())
            server.run(transport="stdio")
        else:
            _uvicorn_logs_to_stderr()
            logger.info(
                "Synapse MCP server %s on http://%s:%s/mcp → backend %s",
                __version__,
                args.host,
                args.port,
                env_url(),
            )
            server.run(transport="streamable-http", host=args.host, port=args.port)
    except SynapseError as exc:  # a malformed SYNAPSE_* value: one line, like the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
