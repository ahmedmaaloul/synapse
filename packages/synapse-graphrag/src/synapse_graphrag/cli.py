# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — command-line interface (``synapse-graphrag``)

A thin argparse front-end over :class:`synapse_graphrag.client.SynapseClient`
for people and scripts: check the backend, ask a question, pull budgeted
context, ingest a PDF with live progress, list themes, print graph stats —
plus ``mcp`` (runs the MCP server) and ``install-config`` (prints the exact
snippet each MCP host needs).

Failure contract: a backend error or an unreachable backend prints ONE line
to stderr and exits 1. Users never see a traceback for a server that is
simply not running. Machine-readable output (``retrieve --json``,
``install-config``) goes to stdout alone; commentary goes to stderr so the
output can be piped into a file or ``jq``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from contextlib import aclosing
from typing import Any

from synapse_graphrag import __version__
from synapse_graphrag.client import (
    DEFAULT_MAX_CONTEXT_CHARS,
    SynapseClient,
    SynapseError,
    env_max_context_chars,
    env_url,
    run,
)

ClientFactory = Callable[..., SynapseClient]

MCP_CLIENTS = ("claude-code", "claude-desktop", "cursor", "vscode", "windsurf")

CONFIG_LOCATIONS = {
    "claude-desktop": (
        "macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json\n"
        "Windows: %APPDATA%\\Claude\\claude_desktop_config.json\n"
        "Linux:   ~/.config/Claude/claude_desktop_config.json"
    ),
    "cursor": "~/.cursor/mcp.json (global) or .cursor/mcp.json (per project)",
    "vscode": ".vscode/mcp.json (workspace) — or add it under \"mcp.servers\" in settings.json",
    "windsurf": "~/.codeium/windsurf/mcp_config.json",
}


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _note(message: str) -> None:
    print(message, file=sys.stderr)


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_status(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            health = await client.health()
            ready = await client.ready()
            about = await client.about()
            return {"url": client.base_url, "health": health, "ready": ready, "about": about}

    info = run(go())
    ready = info["ready"]
    about = info["about"]
    print(f"Synapse at {info['url']}")
    print(f"  health:    {info['health'].get('status', '?')}")
    print(f"  ready:     {ready.get('status', '?')} (neo4j {ready.get('neo4j', '?')})")
    print(f"  version:   {about.get('version', '?')}")
    print(f"  llm:       {about.get('llm_provider', '?')}")
    print(f"  embedding: {about.get('embedding_provider', '?')}")
    return 0 if ready.get("status") == "ready" else 1


def cmd_ask(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        done = False
        async with make_client(base_url=args.url) as client:
            # ``aclosing``: leaving the loop early must close the HTTP stream
            # here, not in the event loop's finaliser (which races and prints).
            async with aclosing(client.ask_stream(args.query)) as events:
                async for event in events:
                    kind = event.get("type")
                    if kind == "token":
                        print(event.get("data", ""), end="", flush=True)
                    elif kind == "citations":
                        citations = list(event.get("data") or [])
                    elif kind == "error":
                        print(flush=True)
                        raise SynapseError(None, str(event.get("data") or "Generation failed."))
                    elif kind == "done":
                        done = True
                        break
        print(flush=True)
        if not done:
            raise SynapseError(None, "Chat stream ended without a done event.")
        return citations

    citations = run(go())
    if citations:
        names = ", ".join(str(c.get("name", "?")) for c in citations)
        _note(f"\nCitations: {names}")
    return 0


def cmd_retrieve(args: argparse.Namespace, make_client: ClientFactory) -> int:
    # Same default as the MCP server: --budget, else $SYNAPSE_MAX_CONTEXT_CHARS
    # (6000); 0 on either means "no budget".
    budget = args.budget if args.budget is not None else env_max_context_chars()
    if budget is not None and budget <= 0:
        budget = None

    async def go() -> Any:
        async with make_client(base_url=args.url) as client:
            return await client.retrieve(args.query, k=args.k, max_context_chars=budget)

    retrieval = run(go())
    if args.json:
        print(json.dumps(retrieval.to_dict(), indent=2, ensure_ascii=False))
        return 0

    usage = retrieval.usage
    header = (
        f"mode={retrieval.mode} · {usage.get('context_chars', len(retrieval.context))} chars "
        f"(~{usage.get('context_tokens_est', '?')} tokens) · {len(retrieval.citations)} citations"
    )
    if usage.get("truncated"):
        header += " · truncated"
    _note(header)
    print(retrieval.context)
    if retrieval.citations:
        names = ", ".join(str(c.get("name", "?")) for c in retrieval.citations)
        _note(f"\nCitations: {names}")
    return 0


def _print_progress(event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "accepted":
        _note(
            f"Uploaded {event.get('filename')}: {event.get('total_chunks')} chunks "
            f"(job {event.get('job_id')})"
        )
        return
    stage = event.get("stage", kind)
    processed, total = event.get("processed"), event.get("total")
    label = "communities" if kind == "community_progress" else "ingest"
    if processed is not None and total:
        _note(f"[{label}] {stage} {processed}/{total}")
    else:
        _note(f"[{label}] {stage}")


def cmd_ingest(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> Any:
        async with make_client(base_url=args.url) as client:
            return await client.ingest_pdf(args.file, theme=args.theme, on_progress=_print_progress)

    result = run(go())
    print(
        f"Ingested {result.filename}: {result.chunks_processed} chunks → "
        f"{result.nodes_created} nodes, {result.relationships_created} relationships, "
        f"{result.communities} communities"
    )
    return 0


def cmd_communities(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.communities(limit=args.limit)

    payload = run(go())
    communities = payload.get("communities") or []
    if not communities:
        print(
            "No communities yet — ingest a document, or rebuild them with "
            "POST /api/communities/rebuild (the UI's rebuild button; needs an LLM key)."
        )
        return 0
    for community in communities:
        title = community.get("title") or community.get("id") or "Untitled"
        size = community.get("size") or len(community.get("members") or [])
        print(f"• {title} ({size} entities)")
        summary = community.get("summary")
        if summary:
            print(f"    {summary}")
    return 0


def cmd_stats(args: argparse.Namespace, make_client: ClientFactory) -> int:
    from synapse_graphrag.mcp_server import graph_stats

    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.graph()

    stats = graph_stats(run(go()))
    print(f"Nodes: {stats['nodes']}   Edges: {stats['edges']}   Isolated: {stats['isolated_nodes']}")
    if stats["entity_types"]:
        print("Entity types:")
        for kind, count in stats["entity_types"].items():
            print(f"  {kind:<20} {count}")
    if stats["relationship_types"]:
        print("Relationship types:")
        for kind, count in stats["relationship_types"].items():
            print(f"  {kind:<20} {count}")
    if stats["top_entities"]:
        print("Most connected:")
        for entity in stats["top_entities"]:
            print(f"  {entity['label']} ({entity['type']}) — degree {entity['degree']}")
    return 0


def cmd_mcp(args: argparse.Namespace, make_client: ClientFactory) -> int:
    from synapse_graphrag.mcp_server import main as mcp_main

    forwarded = ["--transport", args.transport, "--host", args.host, "--port", str(args.port)]
    if args.url:
        forwarded += ["--url", args.url]
    return mcp_main(forwarded)


def cmd_install_config(args: argparse.Namespace, make_client: ClientFactory) -> int:
    url = args.url or env_url()
    env = {"SYNAPSE_URL": url}
    if args.budget:
        env["SYNAPSE_MAX_CONTEXT_CHARS"] = str(args.budget)

    if args.client == "claude-code":
        env_flags = " ".join(f"-e {key}={value}" for key, value in env.items())
        _note("Run this in your project (add `--scope user` to enable it everywhere):")
        print(f"claude mcp add synapse {env_flags} -- uvx synapse-graphrag mcp")
        return 0

    entry: dict[str, Any] = {"command": "uvx", "args": ["synapse-graphrag", "mcp"], "env": env}
    if args.client == "vscode":
        config: dict[str, Any] = {"servers": {"synapse": {"type": "stdio", **entry}}}
    else:
        config = {"mcpServers": {"synapse": entry}}

    _note(f"Add this to your {args.client} MCP config:\n{CONFIG_LOCATIONS[args.client]}\n")
    print(json.dumps(config, indent=2))
    return 0


# ── Parser ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapse-graphrag",
        description="Talk to a Synapse GraphRAG backend: retrieve budgeted context, ask, ingest, "
        "inspect the graph, or run the MCP server.",
    )
    parser.add_argument(
        "--url", default=None, help="Synapse backend URL (default: $SYNAPSE_URL or localhost:8000)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("status", help="Health, readiness, version and providers of the backend.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("ask", help="Stream a grounded answer generated by the backend's LLM.")
    p.add_argument("query")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser(
        "retrieve", help="Print GraphRAG context for a question without generating an answer."
    )
    p.add_argument("query")
    p.add_argument("--k", type=int, default=8, help="Seed entities (1-20, default 8).")
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        metavar="CHARS",
        help=f"Max context characters (default: $SYNAPSE_MAX_CONTEXT_CHARS, else "
        f"{DEFAULT_MAX_CONTEXT_CHARS}; 0 disables the budget).",
    )
    p.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("ingest", help="Ingest a PDF and follow the job's progress.")
    p.add_argument("file", metavar="FILE.pdf")
    p.add_argument("--theme", default="Generic", help="Domain hint for extraction.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("communities", help="List the corpus' detected themes.")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_communities)

    p = sub.add_parser("stats", help="Node/edge counts and type breakdowns.")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("mcp", help="Run the MCP server (same as `synapse-mcp`).")
    p.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser(
        "install-config", help="Print the MCP config snippet / command for a given client."
    )
    p.add_argument("--client", choices=MCP_CLIENTS, required=True)
    # SUPPRESS keeps the global --url when this one is omitted (argparse lets a
    # subparser default overwrite the parent's value otherwise).
    p.add_argument("--url", default=argparse.SUPPRESS, help="Backend URL to embed.")
    p.add_argument("--budget", type=int, default=None, help="SYNAPSE_MAX_CONTEXT_CHARS to embed.")
    p.set_defaults(func=cmd_install_config)

    return parser


def main(argv: list[str] | None = None, *, client_factory: ClientFactory | None = None) -> int:
    """Console entry point (``synapse-graphrag``); returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    make_client = client_factory or SynapseClient
    try:
        return int(args.func(args, make_client))
    except SynapseError as exc:
        _err(str(exc))
        return 1
    except KeyboardInterrupt:
        _note("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
