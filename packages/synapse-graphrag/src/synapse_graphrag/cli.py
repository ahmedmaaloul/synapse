# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — command-line interface (``synapse-graphrag``)

A thin argparse front-end over :class:`synapse_graphrag.client.SynapseClient`
for people and scripts: check the backend, ask a question, pull budgeted
context, ingest a PDF with live progress, list themes, print graph stats —
plus ``mcp`` (runs the MCP server) and ``install-config`` (prints the exact
snippet each MCP host needs).

Procedural memory has its own commands: ``procedures`` (list / show / export /
import / versions / rollback / guide), ``agent`` (the backend's GraphRAG
Navigator, with its step trace) and ``evolve`` (offline self-evolution of a
procedural graph, after Lu et al., arXiv:2609.09153). ``evolve`` is the one
command here that can spend hundreds of backend LLM calls, so it prints an
upper-bound estimate first and refuses to start without ``--yes`` or a "y" typed
at an interactive prompt — never on a piped or closed stdin. Before asking, it
also refuses a ``--max-llm-calls`` too small for the baseline plus one full
round (naming the smallest one that runs a round, computed as the backend does)
and a ``--mode scratch`` run that would overwrite a stored graph without
``--replace``. ``agent`` and ``evolve`` wait ``max(SYNAPSE_TIMEOUT, 600)``
seconds: giving up sooner would not stop the backend, which keeps working and
spending until the run ends.

``lab`` is the Synapse Lab: compare retrieval approaches side by side on your
own questions, scored on quality AND cost next to three evidence floors
(``arms`` / ``models`` / ``datasets`` / ``upload`` / ``estimate`` / ``run`` /
``runs`` / ``show`` / ``resume``). ``--mode retrieve`` (the default) calls no
model and costs nothing. ``lab run`` follows the same consent rule as
``evolve``: it prints the estimate first, refuses a run whose upper bound breaks
``--max-usd`` (required for the paid modes), and starts nothing without
``--yes`` or an interactive "y". ``lab resume`` asks the same way when
resuming would spend. ``lab estimate`` exits 1 when the estimate is a refusal.

Failure contract: a backend error or an unreachable backend prints ONE line
to stderr and exits 1. Users never see a traceback for a server that is
simply not running. Machine-readable output (``retrieve --json``,
``install-config``, ``--json`` flags, ``procedures export``) goes to stdout
alone; commentary goes to stderr so the output can be piped into a file or
``jq``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any

from synapse_graphrag import __version__
from synapse_graphrag.client import (
    DEFAULT_MAX_CONTEXT_CHARS,
    LAB_MODES,
    LAB_ORDERS,
    SynapseClient,
    SynapseError,
    env_long_timeout,
    env_max_context_chars,
    env_url,
    lab_request,
    parse_budget,
    parse_budgets,
    qa_items,
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

# The backend's defaults (AGENT_MAX_STEPS, EVOLUTION_DEFAULT_ROUNDS,
# EVOLUTION_DEFAULT_BATCH_SIZE). ``evolve`` always sends rounds and batch size
# explicitly, and sends the estimate the user consented to as max_llm_calls
# unless --max-llm-calls names another cap. The backend cannot then spend more
# than was agreed, even if its own AGENT_MAX_STEPS or EVOLUTION_MAX_LLM_CALLS
# is larger than assumed here: it stops cleanly (stopped: budget) instead.
AGENT_MAX_STEPS = 8
EVOLVE_ROUNDS = 3
EVOLVE_BATCH_SIZE = 10
# The backend's bound on a request's max_llm_calls.
MAX_LLM_CALLS_LIMIT = 100_000
OBSERVATION_PREVIEW_CHARS = 240


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _int_range(low: int, high: int | None = None) -> Callable[[str], int]:
    """argparse ``type=`` for an integer in ``[low, high]`` (``high=None``: unbounded)."""

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
        if value < low or (high is not None and value > high):
            bounds = f"{low}-{high}" if high is not None else f">= {low}"
            raise argparse.ArgumentTypeError(f"must be {bounds}, got {value}")
        return value

    return parse


def _flat(text: Any) -> str:
    return " ".join(str(text).split())


def _preview(text: Any, limit: int) -> str:
    flat = _flat(text)
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _score(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _long_client(args: argparse.Namespace, make_client: ClientFactory) -> SynapseClient:
    """A client for requests that make the backend run an agent (``agent``, ``evolve``).

    Its timeout is ``max(SYNAPSE_TIMEOUT, 600)``: timing out would not stop
    the run, only lose its result after the backend has paid for it.
    """
    return make_client(base_url=args.url, timeout=env_long_timeout())


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


# ── Procedural memory ────────────────────────────────────────────────────────
def _count(value: Any) -> int:
    if isinstance(value, list | dict):
        return len(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _diff_summary(diff: Any) -> str:
    """``+1 node, +2 edges, ~1 edge`` from a version's ``graph_diff``.

    An empty or missing diff (the first version has nothing to compare with)
    prints nothing; a diff whose lists are all empty is a text-only change.
    """
    if not isinstance(diff, dict) or not diff:
        return ""
    parts = []
    for key, sign, noun in (
        ("added_nodes", "+", "node"),
        ("removed_nodes", "-", "node"),
        ("changed_nodes", "~", "node"),
        ("added_edges", "+", "edge"),
        ("removed_edges", "-", "edge"),
        ("changed_edges", "~", "edge"),
    ):
        n = _count(diff.get(key))
        if n:
            parts.append(f"{sign}{n} {noun}{'' if n == 1 else 's'}")
    return ", ".join(parts) if parts else "no structural change"


def _localization_method(value: Any) -> str:
    """The localization method, whether the backend sent a string or an object."""
    if isinstance(value, dict):
        return str(value.get("method") or "")
    return str(value) if value else ""


def _format_action(action: Any, args: Any) -> str:
    name = str(action or "?")
    if isinstance(args, dict):
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    elif isinstance(args, list):
        inner = ", ".join(json.dumps(v, ensure_ascii=False) for v in args)
    elif args is None or args == "":
        inner = ""
    else:
        inner = json.dumps(args, ensure_ascii=False)
    return f"{name}({inner})"


def _print_procedure(graph: dict[str, Any]) -> None:
    """Human-readable view of a graph; ``--text`` shows what an LLM reads instead."""
    name = graph.get("name") or "?"
    print(
        f"{name}  v{graph.get('version', '?')} · score {_score(graph.get('score'))} · "
        f"cycle policy {graph.get('cycle_policy') or 'forbid'}"
    )
    if graph.get("description"):
        print(f"  {_flat(graph['description'])}")
    tools = graph.get("tools") or []
    if tools:
        print(f"Tools: {', '.join(str(t) for t in tools)}")
    nodes = graph.get("nodes") or []
    print(f"Nodes ({len(nodes)}):")
    for node in nodes:
        detail = f" — {_flat(node['description'])}" if node.get("description") else ""
        print(f"  [{node.get('id')}] {node.get('type', '?')}{detail}")
    edges = graph.get("edges") or []
    print(f"Transitions ({len(edges)}):")
    for edge in edges:
        condition = f"  if {_flat(edge['condition'])}" if edge.get("condition") else ""
        print(
            f"  [{edge.get('source')}] —{edge.get('relation', 'LEADS_TO')}→ "
            f"[{edge.get('target')}]{condition}"
        )
        if edge.get("guidance"):
            print(f"      guidance: {_flat(edge['guidance'])}")
        if edge.get("pitfalls"):
            print(f"      pitfalls: {_flat(edge['pitfalls'])}")


def _read_json_file(path: str) -> Any:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SynapseError(None, f"Cannot read {path}: {exc.strerror or exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SynapseError(None, f"{path} is not valid JSON: {exc}") from exc


def cmd_procedures(args: argparse.Namespace, make_client: ClientFactory) -> int:
    action = args.proc_command or "list"
    handler = {
        "list": _procedures_list,
        "show": _procedures_show,
        "export": _procedures_export,
        "import": _procedures_import,
        "versions": _procedures_versions,
        "rollback": _procedures_rollback,
        "guide": _procedures_guide,
    }[action]
    return handler(args, make_client)


def _procedures_list(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.procedures()

    graphs = run(go()).get("graphs") or []
    if not graphs:
        print(
            "No procedural graphs yet — the backend seeds its bundled priors "
            "(graphrag-navigator and mcp-host) at startup when PROCEDURAL_ENABLED is true "
            "(the default)."
        )
        return 0
    for graph in graphs:
        print(
            f"• {graph.get('name')}  v{graph.get('version', '?')} · score "
            f"{_score(graph.get('score'))} · {graph.get('nodes', '?')} nodes, "
            f"{graph.get('edges', '?')} edges · updated {graph.get('updated_at') or '—'}"
        )
        if graph.get("description"):
            print(f"    {_preview(graph['description'], 160)}")
    return 0


def _procedures_show(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> Any:
        async with make_client(base_url=args.url) as client:
            if args.text:
                return await client.procedure_text(args.name)
            return await client.procedure(args.name)

    result = run(go())
    if args.text:
        print(result)
    else:
        _print_procedure(result)
    return 0


def _procedures_export(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.procedure(args.name)

    graph = run(go())
    text = json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        try:
            Path(args.output).write_text(text, encoding="utf-8")
        except OSError as exc:
            raise SynapseError(None, f"Cannot write {args.output}: {exc.strerror or exc}") from exc
        _note(f"Exported {args.name} v{graph.get('version', '?')} → {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def _procedures_import(args: argparse.Namespace, make_client: ClientFactory) -> int:
    graph = _read_json_file(args.file)
    if not isinstance(graph, dict):
        raise SynapseError(None, f"{args.file} must hold one JSON object (a procedural graph).")
    # ``version`` and ``score`` are assigned by the backend; an ``export`` carries
    # them for provenance, so a round trip must not send them back.
    graph = {k: v for k, v in graph.items() if k not in ("version", "score")}
    name = args.name or graph.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SynapseError(None, f'{args.file} has no "name"; pass --name.')
    graph["name"] = name

    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.put_procedure(name, graph)

    result = run(go())
    print(f"Imported {result.get('name') or name} as v{result.get('version', '?')}")
    return 0


def _procedures_versions(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.procedure_versions(args.name)

    versions = run(go()).get("versions") or []
    if not versions:
        print(f"No versions recorded for {args.name}.")
        return 0
    for version in versions:
        status = "accepted" if version.get("accepted", True) else "rejected"
        note = f" · {version['note']}" if version.get("note") else ""
        diff = _diff_summary(version.get("diff"))
        label = f"v{version.get('version', '?')}"
        print(
            f"{label:<5}score {_score(version.get('score'))} · {status} · "
            f"{version.get('created_at') or '—'}{note}" + (f" ({diff})" if diff else "")
        )
    return 0


def _procedures_rollback(args: argparse.Namespace, make_client: ClientFactory) -> int:
    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.rollback_procedure(args.name, args.version)

    result = run(go())
    print(
        f"Rolled {args.name} back to v{args.version} — saved as v{result.get('version', '?')} "
        "(history is kept)"
    )
    return 0


def _procedures_guide(args: argparse.Namespace, make_client: ClientFactory) -> int:
    if args.observation is not None and not args.last_action:
        raise SynapseError(None, "--observation describes a step: pass --last-action too.")
    trajectory: list[dict[str, str]] = []
    if args.last_action:
        step = {"action": args.last_action}
        if args.observation is not None:
            step["observation"] = args.observation
        trajectory.append(step)

    async def go() -> dict[str, Any]:
        async with make_client(base_url=args.url) as client:
            return await client.procedure_guidance(
                args.name, args.query, trajectory, mode=args.mode
            )

    result = run(go())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    usage = result.get("usage") or {}
    context = str(result.get("context") or "")
    header = (
        f"{result.get('graph', args.name)} v{result.get('version', '?')} · node "
        f"{result.get('active_node') or '—'} ({_localization_method(result.get('localization')) or '?'})"
        f" · scope {result.get('scope', '?')} · {usage.get('context_chars', len(context))} chars "
        f"(~{usage.get('context_tokens_est', '?')} tokens) · {usage.get('llm_calls', 0)} LLM calls"
    )
    if usage.get("cached"):
        header += " · cached"
    _note(header)
    if context:
        print(context, flush=True)
    if result.get("guidance"):
        print(("\n" if context else "") + "Guidance:\n" + str(result["guidance"]), flush=True)
    next_actions = result.get("next_actions") or []
    if next_actions:
        _note(f"\nNext: {', '.join(str(a) for a in next_actions)}")
    return 0


def _usage_line(result: dict[str, Any]) -> str:
    usage = result.get("usage") or {}
    parts = [
        f"{usage.get('llm_calls', '?')} LLM calls ({usage.get('guidance_llm_calls', 0)} for guidance)",
        f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out tokens"
        + (" (estimated)" if usage.get("estimated") else ""),
    ]
    if result.get("parse_failures"):
        parts.append(f"{result['parse_failures']} parse failures")
    if isinstance(result.get("latency_s"), int | float):
        parts.append(f"{result['latency_s']:.1f}s")
    return "Usage: " + " · ".join(parts)


def cmd_agent(args: argparse.Namespace, make_client: ClientFactory) -> int:
    # ``...`` = let the backend use its PROCEDURAL_DEFAULT_GRAPH; None = no graph.
    graph: Any = None if args.no_graph else (args.graph or ...)

    async def go() -> dict[str, Any]:
        async with _long_client(args, make_client) as client:
            return await client.agent_ask(
                args.query,
                graph=graph,
                guidance=args.guidance,
                max_steps=args.max_steps,
                record=args.record,
            )

    result = run(go())
    answer = result.get("answer")
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if answer is not None else 1

    steps = result.get("steps") or []
    used = result.get("graph")
    if isinstance(used, dict) and used.get("name"):
        _note(f"Navigator · procedural graph {used['name']} v{used.get('version', '?')}")
    else:
        _note("Navigator · no procedural graph")
    for index, step in enumerate(steps, 1):
        method = _localization_method(step.get("localization"))
        _note(f"\nStep {index}" + (f"  [PG: {method}]" if method else ""))
        if step.get("thought"):
            _note(f"  Thought: {_flat(step['thought'])}")
        _note(f"  Action: {_format_action(step.get('action'), step.get('args'))}")
        if step.get("observation"):
            _note(f"  Observation: {_preview(step['observation'], OBSERVATION_PREVIEW_CHARS)}")
    if answer is not None:
        _note("\nAnswer:")
        print(answer, flush=True)
    else:
        _note(
            f"\nNo answer: the navigator stopped ({result.get('stopped', '?')}) "
            f"after {len(steps)} steps."
        )
    _note(_usage_line(result))
    if args.record:
        if result.get("recorded"):
            _note("Trajectory recorded (unscored).")
        else:
            _note(
                "Trajectory not recorded: the run used no procedural graph, or the backend "
                "could not store it."
            )
    return 0 if answer is not None else 1


def _load_qa(path: str, split: str | None, flag: str) -> list[dict[str, Any]]:
    """A JSON array or JSONL file of ``{question, answer}`` items, optionally one split."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SynapseError(None, f"Cannot read {path}: {exc.strerror or exc}") from exc
    items: list[Any]
    if raw.lstrip().startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SynapseError(None, f"{path} is not valid JSON: {exc}") from exc
    else:
        items = []
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SynapseError(None, f"{path}:{number} is not valid JSON: {exc.msg}") from exc
    splits = sorted(
        {str(i["split"]) for i in items if isinstance(i, dict) and i.get("split") is not None}
    )
    if split is not None:
        items = [i for i in items if isinstance(i, dict) and str(i.get("split")) == split]
        if not items:
            found = ", ".join(splits) or "none"
            raise SynapseError(None, f"{path} has no items in split {split!r} (splits: {found}).")
    elif len(splits) > 1:
        # A file with train / val / test mixed would silently train on the test set.
        raise SynapseError(
            None, f"{path} mixes splits ({', '.join(splits)}); choose one with {flag}."
        )
    return qa_items(items, path if split is None else f"{path} ({split})")


def _per_rollout(guidance: str, max_steps: int) -> int:
    """A rollout's worst case: ``max_steps`` solver calls, doubled with generative
    guidance (the backend's guidance cache can only lower this)."""
    return max(1, max_steps) * (2 if guidance == "generative" else 1)


def evolution_estimate(
    rounds: int,
    batch_size: int,
    val_size: int,
    guidance: str,
    max_steps: int = AGENT_MAX_STEPS,
    *,
    stored_eval: bool = False,
) -> dict[str, int]:
    """Upper bound on the backend LLM calls of one evolution run.

    Baseline: one rollout per validation question, and as many again when
    ``--mode scratch --replace`` scores the stored graph it may overwrite
    (``stored_eval``). Each round: ``batch_size`` training rollouts, one
    refiner call, and — unless the candidate is rejected structurally — one
    rollout per validation question. A rollout is at most ``max_steps`` solver
    calls, doubled when every step also asks for generative guidance.
    """
    rollouts = rounds * (batch_size + val_size) + val_size * (2 if stored_eval else 1)
    per_rollout = _per_rollout(guidance, max_steps)
    return {
        "rollouts": rollouts,
        "per_rollout": per_rollout,
        "refiner_calls": rounds,
        "llm_calls": rollouts * per_rollout + rounds,
    }


def evolution_minimum(
    batch_size: int,
    train_size: int,
    val_size: int,
    guidance: str,
    max_steps: int = AGENT_MAX_STEPS,
    *,
    stored_eval: bool = False,
) -> int:
    """The smallest ``--max-llm-calls`` the backend accepts: baseline + ONE full round.

    Mirrors ``minimum_llm_calls`` in the backend's ``procedural_evolution``
    (the backend refuses a smaller cap with a 422 before spending anything):
    the worst case of the baseline (plus the stored graph's evaluation with
    ``stored_eval``), round 1's batch — capped by the training set, as the
    backend strides it — and its validation, in rollouts, plus one refiner call.
    """
    batch = max(1, min(batch_size, train_size))
    rollouts = val_size * (2 if stored_eval else 1) + batch + val_size
    return rollouts * _per_rollout(guidance, max_steps) + 1


def _stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _confirm(prompt: str) -> bool:
    print(prompt, end="", file=sys.stderr, flush=True)
    try:
        reply = sys.stdin.readline()
    except (OSError, ValueError):
        return False
    return reply.strip().lower() in ("y", "yes")


def _scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return str(len(value))
    return _preview(value, 120)


def _stored_procedure(
    args: argparse.Namespace, make_client: ClientFactory
) -> dict[str, Any] | None:
    """The graph stored under ``args.name``, or ``None`` (a read: nothing is spent)."""

    async def go() -> dict[str, Any] | None:
        async with _long_client(args, make_client) as client:
            try:
                return await client.procedure(args.name)
            except SynapseError as exc:
                if exc.status == 404:
                    return None
                raise

    return run(go())


def cmd_evolve(args: argparse.Namespace, make_client: ClientFactory) -> int:
    train = _load_qa(args.train, args.train_split, "--train-split")
    val = _load_qa(args.val, args.val_split, "--val-split")
    overlap = {i["question"].casefold() for i in train} & {i["question"].casefold() for i in val}
    if args.replace and args.mode != "scratch":
        raise SynapseError(
            None,
            "--replace only applies to --mode scratch (static mode builds on the stored graph).",
        )

    # Scratch mode must not overwrite a stored graph by accident, and replacing
    # one costs its evaluation: find out which case this is before any estimate.
    existing = _stored_procedure(args, make_client) if args.mode == "scratch" else None
    if existing is not None and not args.replace:
        raise SynapseError(
            None,
            f"procedural graph {args.name!r} already exists (v{existing.get('version', '?')}, "
            f"score {_score(existing.get('score'))}); --mode scratch would replace it. Re-run "
            "with --replace to score it once on the validation set and overwrite it only with "
            "a candidate at least as good, or choose another NAME.",
        )
    stored_eval = existing is not None

    # Refuse, before asking for consent, a cap that cannot run one full round:
    # the backend would refuse it too (422), and a baseline alone decides nothing.
    minimum = evolution_minimum(
        args.batch_size, len(train), len(val), args.guidance, stored_eval=stored_eval
    )
    if minimum > MAX_LLM_CALLS_LIMIT:
        raise SynapseError(
            None,
            f"one round of this run needs up to {minimum} LLM calls, above the backend's limit "
            f"of {MAX_LLM_CALLS_LIMIT}; use fewer validation questions or a smaller --batch-size.",
        )
    if args.max_llm_calls is not None and args.max_llm_calls < minimum:
        stored_part = f" + {len(val)} stored graph" if stored_eval else ""
        raise SynapseError(
            None,
            f"--max-llm-calls {args.max_llm_calls} cannot pay for the baseline and one full "
            f"round, whose worst case is ({len(val)} baseline{stored_part} + "
            f"{min(args.batch_size, len(train))} train + {len(val)} val) agent runs × "
            f"{_per_rollout(args.guidance, AGENT_MAX_STEPS)} calls + 1 refinement; the smallest "
            f"--max-llm-calls that runs one round is {minimum} "
            f"(assuming AGENT_MAX_STEPS={AGENT_MAX_STEPS}).",
        )

    estimate = evolution_estimate(
        args.rounds, args.batch_size, len(val), args.guidance, stored_eval=stored_eval
    )
    _note(
        f"Evolving {args.name!r} (mode {args.mode}, metric {args.metric}, guidance "
        f"{args.guidance}) on {len(train)} train / {len(val)} validation questions."
    )
    stored_runs = f" + {len(val)} stored graph" if stored_eval else ""
    _note(
        f"This spends backend LLM calls. Upper bound ≈ {estimate['rollouts']} agent runs "
        f"({args.rounds} rounds × ({args.batch_size} train + {len(val)} val) + {len(val)} "
        f"baseline{stored_runs}) × {estimate['per_rollout']} calls + "
        f"{estimate['refiner_calls']} refinements = {estimate['llm_calls']} calls "
        f"(assuming AGENT_MAX_STEPS={AGENT_MAX_STEPS})."
    )
    if args.max_llm_calls is not None:
        max_llm_calls = args.max_llm_calls
        _note(f"Hard cap: the backend stops cleanly at {max_llm_calls} calls (--max-llm-calls).")
    else:
        # The consented bound IS the cap: sent explicitly, never left to backend defaults.
        # It always covers ``minimum`` (one round is part of the bound).
        max_llm_calls = min(estimate["llm_calls"], MAX_LLM_CALLS_LIMIT)
        _note(
            f"Hard cap: the backend stops cleanly at {max_llm_calls} calls, the bound above "
            "(set another with --max-llm-calls)."
        )
    _note(
        f"A candidate is kept iff its validation score does not drop; with {len(val)} "
        f"validation questions the score moves in steps of {1 / len(val):.3f}, so the "
        "accept/reject trail is a search trace, not a significance test."
    )
    if overlap:
        _note(
            f"warning: {len(overlap)} validation question(s) also appear in train — the "
            "acceptance gate would reward memorising them."
        )
    if existing is not None:
        _note(
            f"mode scratch --replace: starts from an empty Start→End skeleton; the stored "
            f"v{existing.get('version', '?')} (score {_score(existing.get('score'))}) is scored "
            "once on the validation set first, and only a candidate scoring at least as well "
            "overwrites it."
        )
    elif args.mode == "scratch":
        _note(
            f"mode scratch: starts from an empty Start→End skeleton; nothing is saved as "
            f"{args.name!r} unless a candidate is accepted."
        )

    if not args.yes:
        if not _stdin_is_interactive():
            raise SynapseError(
                None,
                "refusing to start a paid evolution run without consent: re-run with --yes, "
                "or from an interactive terminal.",
            )
        if not _confirm("Start it? [y/N] "):
            _note("Cancelled; nothing was sent to the backend.")
            return 1

    started: dict[str, str] = {}

    def on_progress(event: dict[str, Any]) -> None:
        if event.get("type") == "accepted":
            started["job_id"] = str(event.get("job_id") or "")
            _note(
                f"Evolution job {started['job_id']} started "
                "(Ctrl-C stops listening, not the job)."
            )
            return
        stage = event.get("stage") or event.get("type")
        where = f"round {event['round']} " if event.get("round") not in (None, "") else ""
        extras = " ".join(
            f"{key}={_scalar(value)}"
            for key, value in event.items()
            if key not in ("type", "stage", "round") and not isinstance(value, dict)
        )
        _note(f"[evolve] {where}{stage}" + (f" {extras}" if extras else ""))

    async def go() -> dict[str, Any]:
        async with _long_client(args, make_client) as client:
            return await client.evolve_procedure(
                args.name,
                train,
                val,
                rounds=args.rounds,
                batch_size=args.batch_size,
                mode=args.mode,
                metric=args.metric,
                guidance=args.guidance,
                max_llm_calls=max_llm_calls,
                on_progress=on_progress,
                replace=args.replace,
            )

    try:
        report = run(go())
    except KeyboardInterrupt:
        job = started.get("job_id")
        if job:
            _note(
                f"interrupted: job {job} keeps running on the backend until it finishes or "
                "reaches its LLM-call budget."
            )
        else:
            _note("interrupted")
        return 130

    _print_evolution_report(report)
    return 0


def _print_evolution_report(report: dict[str, Any]) -> None:
    print(
        f"Evolved {report.get('graph', '?')} ({report.get('mode', '?')}): "
        f"{report.get('rounds_run', '?')} rounds run, stopped: {report.get('stopped', '?')} · "
        f"{report.get('llm_calls', '?')} LLM calls"
    )
    version = report.get("final_version")
    # Null when nothing was saved: a scratch run that accepted no candidate.
    saved = f"v{version}" if version is not None else "nothing saved"
    print(
        f"  baseline {_score(report.get('baseline_score'))} → final "
        f"{_score(report.get('final_score'))} ({saved})"
    )
    if report.get("stored_score") is not None:
        # scratch --replace: the graph that was stored scored this on validation,
        # and no candidate below it could overwrite it.
        print(
            f"  stored v{report.get('stored_version', '?')} scored "
            f"{_score(report['stored_score'])}: the score a candidate had to reach to replace it"
        )
    for entry in report.get("rounds") or []:
        verdict = "accepted" if entry.get("accepted") else "rejected"
        line = (
            f"  round {entry.get('round', '?')}: train {_score(entry.get('train_mean'))} · "
            f"candidate {_score(entry.get('candidate_score'))} · {verdict}"
        )
        if entry.get("reason"):
            line += f" — {_flat(entry['reason'])}"
        diagnostics = entry.get("diagnostics") or []
        if diagnostics:
            line += f" ({'; '.join(_preview(d, 120) for d in diagnostics[:3])})"
        if entry.get("accepted") and entry.get("diff"):
            line += f" [{_diff_summary(entry['diff'])}]"
        print(line)
    if report.get("effect_floor") is not None:
        print(
            f"Effect floor: {_score(report['effect_floor'])} (one validation question); "
            "smaller differences are noise, and accept/reject on a small validation set is a "
            "search trace, not significance."
        )


# ── Synapse Lab ──────────────────────────────────────────────────────────────
LAB_FLOOR_NOTE = (
    "Containment and permissive recall have no lower bound: the vocabulary null (N1) scores "
    "high on both while carrying no question-specific evidence. Read every row against the "
    "floors."
)


def _usd(value: Any) -> str:
    if value is None:
        return "—"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    if amount == 0:
        return "$0"
    if abs(amount) < 0.01:
        return f"${amount:.4f}"
    return f"${amount:,.2f}"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _budget_label(budget: Any) -> str:
    return "default" if budget is None else str(budget)


def _dataset_label(ds: dict[str, Any]) -> str:
    """``demo`` / ``hotpotqa`` / ``qa-file:<name>`` — the id a run was given."""
    name = str(ds.get("name") or "?")
    return f"qa-file:{ds['source']}" if name == "qa-file" and ds.get("source") else name


def _table(headers: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    """Plain-text columns: ``right`` holds the indexes of right-aligned (numeric) columns."""
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.rjust(w) if i in right else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()

    return [line(headers), *(line(r) for r in rows)]


def _csv(raw: str) -> list[str]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _budgets_arg(raw: str) -> list[int | None]:
    try:
        return parse_budgets(raw)
    except SynapseError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _cell_arg(raw: str) -> tuple[str, int | None]:
    """``ARM@BUDGET`` (``ARM`` alone = its default context)."""
    arm, _, budget = raw.partition("@")
    if not arm.strip():
        raise argparse.ArgumentTypeError(f"expected ARM@BUDGET, got {raw!r}")
    try:
        return arm.strip(), parse_budget(budget or None)
    except SynapseError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _usd_arg(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an amount in USD, got {raw!r}") from exc
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw}")
    return value


def cmd_lab(args: argparse.Namespace, make_client: ClientFactory) -> int:
    action = args.lab_command or "arms"
    handler = {
        "arms": _lab_arms,
        "models": _lab_models,
        "datasets": _lab_datasets,
        "upload": _lab_upload,
        "estimate": _lab_estimate,
        "run": _lab_run,
        "runs": _lab_runs,
        "show": _lab_show,
        "resume": _lab_resume,
    }[action]
    return handler(args, make_client)


def _lab_get(args: argparse.Namespace, make_client: ClientFactory, method: str, *a: Any, **kw: Any) -> Any:
    async def go() -> Any:
        async with make_client(base_url=args.url) as client:
            return await getattr(client, method)(*a, **kw)

    return run(go())


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _lab_arms(args: argparse.Namespace, make_client: ClientFactory) -> int:
    payload = _lab_get(args, make_client, "lab_arms")
    if getattr(args, "json", False):
        _print_json(payload)
        return 0
    arms = payload.get("arms") or []
    for family in payload.get("families") or []:
        members = [a for a in arms if a.get("family") == family.get("family")]
        if not members:
            continue
        print(family.get("title") or family.get("family"))
        for arm in members:
            calls = arm.get("retrieval_llm_calls", 0)
            extras = [f"{calls} retrieval LLM call{'' if calls == 1 else 's'}"]
            if arm.get("needs_graph"):
                extras.append("needs the extracted graph")
            print(f"  {arm.get('name', '?'):<18} {arm.get('title', '')} · {' · '.join(extras)}")
            if arm.get("description"):
                print(f"      {_flat(arm['description'])}")
            source = arm.get("source") or {}
            if source.get("citation") or source.get("url"):
                print(f"      source: {source.get('citation', '')} {source.get('url', '')}".rstrip())
    batch = "" if payload.get("batch_available", True) else " (batch is not available on this backend)"
    print(f"Modes: retrieve (free, the default) · realtime · batch{batch}")
    return 0


def _lab_models(args: argparse.Namespace, make_client: ClientFactory) -> int:
    payload = _lab_get(args, make_client, "lab_models")
    if args.json:
        _print_json(payload)
        return 0
    rows = [
        [
            str(m.get("name")),
            _num(m.get("input_usd_per_1m"), 3),
            _num(m.get("output_usd_per_1m"), 3),
            _num(m.get("batch_input_usd_per_1m"), 3),
            _num(m.get("batch_output_usd_per_1m"), 3),
            "yes" if m.get("reasoning") else "no",
            str(m.get("price_checked_on") or "—"),
        ]
        for m in payload.get("models") or []
    ]
    headers = ["MODEL", "IN $/1M", "OUT $/1M", "BATCH IN", "BATCH OUT", "REASONING", "PRICE AS OF"]
    for line in _table(headers, rows, right={1, 2, 3, 4}):
        print(line)
    _note(
        f"Default reader: {payload.get('default', '?')}. Prices are hand-recorded, not live — "
        f"verify at {payload.get('pricing_url', 'the provider')}. A reasoning model bills its "
        "hidden reasoning tokens as output."
    )
    return 0


def _lab_datasets(args: argparse.Namespace, make_client: ClientFactory) -> int:
    payload = _lab_get(args, make_client, "lab_datasets")
    if args.json:
        _print_json(payload)
        return 0
    for dataset in payload.get("datasets") or []:
        splits = dataset.get("splits") or {}
        detail = f"{dataset['n']} questions" if dataset.get("n") is not None else ""
        if splits:
            detail += " · splits " + ", ".join(f"{k} {v}" for k, v in splits.items())
        if dataset.get("graph_paragraphs") is not None:
            detail = f"{dataset['graph_paragraphs']} paragraphs in the graph"
        print(f"• {dataset.get('id')}  {dataset.get('title', '')}" + (f" · {detail}" if detail else ""))
    for dataset in payload.get("unavailable") or []:
        print(f"✗ {dataset.get('id')}: {dataset.get('reason', 'unavailable')}")
    return 0


def _lab_upload(args: argparse.Namespace, make_client: ClientFactory) -> int:
    result = _lab_get(
        args, make_client, "lab_upload_qa_file", args.file, name=args.name, replace=args.replace
    )
    splits = result.get("splits") or {}
    split_text = (" · splits " + ", ".join(f"{k} {v}" for k, v in splits.items())) if splits else ""
    verb = {"unchanged": "Already stored (unchanged)", "replaced": "Replaced"}.get(
        str(result.get("status")), "Stored"
    )
    print(f"{verb}: {result.get('file')} — {result.get('n')} questions{split_text}")
    _note(f"Use it with --dataset {result.get('id')}")
    return 0


def _lab_body(args: argparse.Namespace) -> dict[str, Any]:
    ingest: dict[str, Any] | None = None
    if args.ingest or args.ingest_paragraphs is not None:
        ingest = {"model": args.ingest_model, "batch": not args.ingest_realtime}
        if args.ingest_paragraphs is not None:
            ingest["paragraphs"] = args.ingest_paragraphs
    return lab_request(
        dataset=args.dataset,
        arms=args.arms,
        budgets=args.budgets,
        mode=args.mode,
        max_usd=args.max_usd,
        reader_model=args.reader_model,
        split=args.split,
        n=args.n,
        offset=args.offset,
        sample_seed=args.sample_seed,
        k=args.k,
        passage_k=args.passage_k,
        seed=args.seed,
        order=args.order,
        reasoning_allowance=args.reasoning_allowance,
        max_concurrency=args.concurrency,
        extra_cells=args.extra_cell,
        ingest=ingest,
        ingest_usd=args.ingest_usd,
        measured_run_id=args.measured_from,
        run_id=getattr(args, "run_id", None),
    )


def _print_estimate(est: dict[str, Any], emit: Callable[[str], None]) -> None:
    ds = est.get("dataset") or {}
    reader = str(est.get("reader_model") or "?")
    if est.get("reasoning_allowance"):
        reader += f" (reasoning model, {est['reasoning_allowance']} reasoning tokens allowed)"
    batch = f" · Batch ×{est.get('batch_multiplier', 0.5)}" if est.get("batch") else ""
    emit(
        f"Estimate · {_dataset_label(ds)} ({ds.get('split') or 'all'}, n={ds.get('n', '?')}) · "
        f"mode {est.get('mode')}{batch} · reader {reader} · tokenizer {est.get('tokenizer', '?')}"
    )
    measured = est.get("measured_from")
    if measured:
        emit(
            f"Priced on the measured contexts of run {measured.get('run_id')} "
            f"({len(measured.get('cells') or [])} cells measured, "
            f"{len(measured.get('unmeasured') or [])} bounded by their budget)."
        )
    cells = est.get("cells") or []
    if est.get("mode") == "retrieve":
        emit(
            f"Retrieve-only: {len(cells)} (arm, budget) cells × {ds.get('n', '?')} questions; "
            "no model is called — $0."
        )
    else:
        rows = [
            [
                str(c.get("arm")),
                _budget_label(c.get("budget")),
                f"{int(c.get('calls') or 0):,}",
                _num(c.get("context_tokens"), 0) + ("" if c.get("measured") else " ≤"),
                f"{int(c.get('prompt_tokens') or 0):,}",
                f"{int(c.get('upper_completion_tokens') or 0):,}",
                _usd(c.get("point_usd")),
                _usd(c.get("upper_usd")),
            ]
            for c in cells
        ]
        headers = ["ARM", "BUDGET", "CALLS", "CONTEXT", "PROMPT TOK", "OUTPUT CAP", "POINT", "UPPER"]
        for line in _table(headers, rows, right={2, 3, 4, 5, 6, 7}):
            emit("  " + line)
        for phase in est.get("phases") or []:
            emit(
                f"  {phase.get('phase')}: {int(phase.get('calls') or 0):,} calls · "
                f"{_usd(phase.get('point_usd'))} point / {_usd(phase.get('upper_usd'))} upper"
            )
        for assumption in est.get("assumptions") or []:
            emit(f"  – {assumption}")
    cap = est.get("max_usd")
    emit(
        f"Total: {_usd(est.get('total_point_usd'))} point · {_usd(est.get('total_upper_usd'))} "
        f"upper bound · cap {_usd(cap) if cap is not None else 'none'}"
    )
    if est.get("refuse"):
        emit(f"REFUSED: {est.get('refuse_reason')}")


def _lab_estimate(args: argparse.Namespace, make_client: ClientFactory) -> int:
    body = _lab_body(args)
    est = _lab_get(args, make_client, "lab_estimate", body)
    if args.json:
        _print_json(est)
    else:
        _print_estimate(est, print)
    return 1 if est.get("refuse") else 0


def _comparison(row: dict[str, Any], key: str) -> str:
    c = (row.get("comparisons") or {}).get(key)
    if not c or c.get("diff") is None:
        return "—"
    return f"{float(c['diff']):+.1f}" + ("" if c.get("reportable") else " (n.s.)")


def _print_leaderboard(board: dict[str, Any]) -> None:
    rows = board.get("rows") or []
    if board.get("mode") == "read":
        headers = [
            "ARM", "BUDGET", "F1", "EM", "TOK/CORRECT", "$/100 CORRECT", "vs N0", "vs N2", "PREMIUM",
        ]
        table = [
            [
                f"{r.get('arm')} (floor)" if r.get("is_null") else str(r.get("arm")),
                str(r.get("budget_label") or _budget_label(r.get("budget"))),
                _num(r.get("f1")),
                _num(r.get("em")),
                _num(r.get("tokens_per_correct"), 0),
                _usd(r.get("usd_per_100_correct")),
                _comparison(r, "gain_above_n0"),
                _comparison(r, "gain_above_n2"),
                _comparison(r, "graph_premium"),
            ]
            for r in rows
        ]
        for line in _table(headers, table, right={2, 3, 4, 5, 6, 7, 8}):
            print(line)
        print(
            f"Effect floor: {_num(board.get('effect_floor_points'), 2)} points (one question). A "
            "difference counts only above it AND with a 95% paired-bootstrap CI excluding 0; "
            "(n.s.) otherwise."
        )
        frontier = (board.get("frontiers") or {}).get("f1_vs_usd") or []
        if frontier:
            print(
                "Pareto frontier (F1 vs $): "
                + ", ".join(f"{p.get('arm')}@{_budget_label(p.get('budget'))}" for p in frontier)
            )
    else:
        headers = ["ARM", "BUDGET", "CONTEXT TOK", "UNITS", "CONTAINS %", "RECALL %", "STRICT %"]
        table = [
            [
                f"{r.get('arm')} (floor)" if r.get("is_null") else str(r.get("arm")),
                str(r.get("budget_label") or _budget_label(r.get("budget"))),
                _num(r.get("context_tokens_mean"), 0),
                _num(r.get("units_mean")),
                _num(r.get("containment")),
                _num(r.get("recall_permissive")),
                _num(r.get("recall_strict")),
            ]
            for r in rows
        ]
        for line in _table(headers, table, right={2, 3, 4, 5, 6}):
            print(line)
        print("Retrieve-only: no model was called ($0). " + LAB_FLOOR_NOTE)
    for note in board.get("notes") or []:
        print(f"Note: {note}")


def _print_run(detail: dict[str, Any]) -> None:
    manifest = detail.get("manifest") or {}
    cfg = manifest.get("config") or {}
    ds = manifest.get("dataset") or {}
    mode = cfg.get("mode")
    running = " (running)" if detail.get("active") else ""
    reader = f" · reader {cfg.get('reader_model')}" if mode != "retrieve" else ""
    print(
        f"Lab run {detail.get('run_id')} · {detail.get('status')}{running} · mode {mode} · "
        f"{_dataset_label(ds)} ({ds.get('split') or 'all'}, n={ds.get('n')}){reader}"
    )
    code = manifest.get("code") or {}
    sha = str(code.get("git_sha") or "unknown")[:10]
    print(
        f"  code {sha}{' (dirty)' if code.get('git_dirty') else ''} · Synapse "
        f"{code.get('synapse_version') or 'unknown'} · created {manifest.get('created_at') or '—'}"
    )
    if mode != "retrieve":
        actual = ((manifest.get("actual") or {}).get("reader") or {}).get("usd")
        estimate = (manifest.get("estimate") or {}).get("read") or (
            (manifest.get("estimate") or {}).get("pre_run") or {}
        )
        print(
            f"  spend {_usd(actual)} actual · estimate {_usd(estimate.get('total_point_usd'))} "
            f"point / {_usd(estimate.get('total_upper_usd'))} upper · cap "
            f"{_usd(manifest.get('budget_cap_usd'))}"
        )
    reason = manifest.get("refuse_reason") or manifest.get("abort_reason") or manifest.get("error")
    if detail.get("status") in ("refused", "aborted", "failed") and reason:
        print(f"  {detail.get('status')}: {reason}")
    board = detail.get("leaderboard")
    if board is None:
        print("Not scored yet.")
        return
    _print_leaderboard(board)


def _print_rows(page: dict[str, Any]) -> None:
    rows = page.get("rows") or []
    if not rows:
        return
    read = any("f1" in r for r in rows)
    if read:
        headers = ["ARM", "BUDGET", "QID", "F1", "EM", "ANSWER", "GOLD"]
        table = [
            [
                str(r.get("arm")), _budget_label(r.get("budget")), str(r.get("qid")),
                _num((r.get("f1") or 0) * 100), _num((r.get("em") or 0) * 100),
                _preview(r.get("answer") if r.get("answer") is not None else r.get("read_error"), 40),
                _preview(", ".join(str(g) for g in r.get("gold") or []), 40),
            ]
            for r in rows
        ]
        right = {3, 4}
    else:
        headers = ["ARM", "BUDGET", "QID", "CONTEXT TOK", "UNITS", "CONTAINS", "GOLD"]
        table = [
            [
                str(r.get("arm")), _budget_label(r.get("budget")), str(r.get("qid")),
                str(r.get("context_tokens", "—")), str(r.get("units_used", "—")),
                "yes" if r.get("containment") else "no",
                _preview(", ".join(str(g) for g in r.get("gold") or []), 40),
            ]
            for r in rows
        ]
        right = {3, 4}
    print(f"\nRows {page.get('offset', 0) + 1}–{page.get('offset', 0) + len(rows)} of {page.get('total')}:")
    for line in _table(headers, table, right=right):
        print(line)


def _print_lab_event(event: dict[str, Any], started: dict[str, str]) -> None:
    kind = event.get("type")
    if kind == "accepted":
        started["run_id"] = str(event.get("run_id") or "")
        _note(
            f"Lab run {started['run_id']} started (job {event.get('job_id')}); Ctrl-C stops "
            "listening, not the run."
        )
    elif kind == "phase":
        total = f" ({event['total']} items)" if event.get("total") else ""
        _note(f"[{event.get('phase')}] {event.get('status')}{total}")
    elif kind == "progress":
        done, total = event.get("done"), event.get("total")
        if not total or done is None:
            return
        # About ten lines per phase, whatever its size: every ceil(total / 10) items.
        if done == total or done % max(1, -(-int(total) // 10)) == 0:
            spent = f" · spent {_usd(event['spent_usd'])}" if event.get("spent_usd") is not None else ""
            _note(f"[{event.get('phase')}] {done}/{total}{spent}")
    elif kind == "batch_submitted":
        _note(f"[read] batch submitted: {event.get('requests')} requests")
    elif kind == "batch_status":
        status = event.get("status") or {}
        _note(f"[batch] {status.get('status', '?') if isinstance(status, dict) else status}")
    elif kind in ("refused", "aborted"):
        _note(f"[read] {kind}: {event.get('reason')}")


def _follow_lab(
    args: argparse.Namespace, make_client: ClientFactory, start: Callable[[SynapseClient, Any], Any]
) -> int:
    """Run ``start(client, on_progress)`` (start/resume + follow), then report the outcome."""
    started: dict[str, str] = {"run_id": getattr(args, "run_id", None) or ""}

    def on_progress(event: dict[str, Any]) -> None:
        _print_lab_event(event, started)

    async def go() -> dict[str, Any]:
        async with _long_client(args, make_client) as client:
            return await start(client, on_progress)

    try:
        result = run(go())
    except KeyboardInterrupt:
        run_id = started.get("run_id")
        if run_id:
            _note(
                f"interrupted: run {run_id} keeps going on the backend; read it back with "
                f"`synapse-graphrag lab show {run_id}`."
            )
        else:
            _note("interrupted")
        return 130

    status = result.get("status")
    run_id = result.get("run_id")
    ok = status in ("done", "batch_submitted")
    if args.json:
        _print_json(result)
        return 0 if ok else 1
    if status == "done":
        _print_run(_lab_get(args, make_client, "lab_show", run_id, limit=0))
    elif status == "batch_submitted":
        print(
            f"Run {run_id}: batch submitted and not finished yet. Collect it with "
            f"`synapse-graphrag lab resume {run_id}` (OpenAI's window is 24h)."
        )
    else:
        reason = f" — {result['reason']}" if result.get("reason") else ""
        print(f"Run {run_id}: {status}{reason}")
        if status in ("aborted", "refused"):
            print(
                f"Nothing past the cap was sent. Continue with `synapse-graphrag lab resume "
                f"{run_id} --max-usd X`; answers already paid for are kept."
            )
    return 0 if ok else 1


def _consent(prompt: str, what: str) -> bool:
    """``--yes`` or an interactive y; a piped or closed stdin is never consent."""
    if not _stdin_is_interactive():
        raise SynapseError(
            None,
            f"refusing to {what} without consent: re-run with --yes, or from an interactive "
            "terminal.",
        )
    return _confirm(prompt)


def _lab_run(args: argparse.Namespace, make_client: ClientFactory) -> int:
    body = _lab_body(args)
    paid = body["mode"] != "retrieve"
    if paid and args.max_usd is None:
        raise SynapseError(
            None,
            f"--mode {body['mode']} calls the backend's reader model: pass --max-usd, the hard "
            "cap on what the run may spend (--mode retrieve is free).",
        )
    est = _lab_get(args, make_client, "lab_estimate", body)
    _print_estimate(est, _note)
    if est.get("refuse"):
        raise SynapseError(None, f"refused before anything was started: {est.get('refuse_reason')}")
    if paid:
        _note(
            f"This run SPENDS: up to {_usd(est.get('total_upper_usd'))} (upper bound), hard cap "
            f"{_usd(body.get('max_usd'))}. The backend re-checks the cap on the measured contexts "
            "before reading and stops cleanly before crossing it."
        )
        if body["mode"] == "batch":
            _note(
                "Batch mode: the run retrieves, submits OpenAI batches (half price, 24h window) "
                "and parks; `synapse-graphrag lab resume RUN_ID` collects the answers."
            )
    else:
        _note(
            "Retrieve-only: no model is called and nothing is spent; the backend only reads the "
            "knowledge graph."
        )
    if not args.yes and not _consent("Start it? [y/N] ", "start a Lab run"):
        _note("Cancelled; no run was started.")
        return 1

    async def start(client: SynapseClient, on_progress: Any) -> dict[str, Any]:
        return await client.lab_run(body, on_progress=on_progress)

    return _follow_lab(args, make_client, start)


def _lab_runs(args: argparse.Namespace, make_client: ClientFactory) -> int:
    payload = _lab_get(args, make_client, "lab_runs")
    if args.json:
        _print_json(payload)
        return 0
    runs = payload.get("runs") or []
    if not runs:
        print(
            "No Lab runs yet — `synapse-graphrag lab run` starts one (retrieve-only by default: "
            "free, no model called)."
        )
        return 0
    for entry in runs:
        budgets = ", ".join(_budget_label(b) for b in entry.get("budgets") or [])
        reader = f" · reader {entry.get('reader_model')}" if entry.get("mode") != "retrieve" else ""
        running = " (running)" if entry.get("active") else ""
        print(
            f"• {entry.get('run_id')}  {entry.get('status')}{running} · {entry.get('mode')} · "
            f"{entry.get('dataset')} n={entry.get('n')} · {len(entry.get('arms') or [])} arms × "
            f"[{budgets}]{reader} · {entry.get('created_at') or '—'}"
        )
    return 0


def _lab_show(args: argparse.Namespace, make_client: ClientFactory) -> int:
    detail = _lab_get(
        args, make_client, "lab_show", args.run_id, limit=args.rows, arm=args.arm, budget=args.budget
    )
    if args.json:
        _print_json(detail)
        return 0
    _print_run(detail)
    _print_rows(detail.get("rows") or {})
    return 0


def _lab_resume(args: argparse.Namespace, make_client: ClientFactory) -> int:
    detail = _lab_get(args, make_client, "lab_show", args.run_id, limit=0)
    manifest = detail.get("manifest") or {}
    status = detail.get("status")
    mode = (manifest.get("config") or {}).get("mode")
    spends = mode != "retrieve" and status != "batch_submitted" and (
        status != "done" or args.retry_failed
    )
    if spends:
        pending = (manifest.get("estimate") or {}).get("read_pending") or {}
        cap = args.max_usd if args.max_usd is not None else manifest.get("budget_cap_usd")
        _note(
            f"Resuming {args.run_id} ({status}): the {mode} reader reads what is left — this "
            f"SPENDS. Last check: {pending.get('requests', '?')} pending requests, upper bound "
            f"{_usd(pending.get('upper_usd'))}, {_usd(pending.get('spent_usd'))} already spent; "
            f"cap {_usd(cap)}. The backend re-checks the cap before reading."
        )
        if not args.yes and not _consent("Resume it? [y/N] ", "resume a paid Lab run"):
            _note("Cancelled; nothing was resumed.")
            return 1
    elif status == "batch_submitted":
        _note(
            f"Polling the batch of {args.run_id}; no new spend (the batch and its cap were set "
            "when the run started)."
        )
    else:
        _note(f"Re-scoring {args.run_id} ({status}); no model is called.")

    async def start(client: SynapseClient, on_progress: Any) -> dict[str, Any]:
        return await client.lab_resume(
            args.run_id,
            max_usd=args.max_usd,
            retry_failed=args.retry_failed,
            on_progress=on_progress,
        )

    return _follow_lab(args, make_client, start)


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

    _add_procedure_parsers(sub)
    _add_lab_parsers(sub)
    return parser


def _add_procedure_parsers(sub: Any) -> None:
    p = sub.add_parser(
        "procedures",
        help="Procedural graphs: list, show, export, import, versions, rollback, guide.",
        description="Procedural memory: step-by-step strategies the backend keeps next to the "
        "knowledge graph. Without an action, lists them.",
    )
    p.set_defaults(func=cmd_procedures, proc_command=None)
    actions = p.add_subparsers(dest="proc_command", metavar="ACTION")

    actions.add_parser("list", help="Every procedural graph with its version and score.")
    a = actions.add_parser("show", help="Nodes and transitions of one graph.")
    a.add_argument("name", metavar="NAME")
    a.add_argument(
        "--text", action="store_true", help="Print the graph exactly as an LLM reads it."
    )
    a = actions.add_parser("export", help="Write a graph as JSON (stdout, or -o FILE).")
    a.add_argument("name", metavar="NAME")
    a.add_argument("-o", "--output", metavar="FILE", default=None)
    a = actions.add_parser(
        "import", help="Store a graph JSON file as a new version (the backend validates it)."
    )
    a.add_argument("file", metavar="FILE")
    a.add_argument("--name", default=None, help="Graph name (default: the file's \"name\").")
    a = actions.add_parser("versions", help="Version history with scores and diffs.")
    a.add_argument("name", metavar="NAME")
    a = actions.add_parser("rollback", help="Re-save an old version as the newest one.")
    a.add_argument("name", metavar="NAME")
    a.add_argument("version", metavar="VERSION", type=_int_range(1))
    a = actions.add_parser(
        "guide", help="Step-local guidance: the subgraph around your last action."
    )
    a.add_argument("name", metavar="NAME")
    a.add_argument("--query", required=True, help="The task or question being worked on.")
    a.add_argument("--last-action", default=None, help="The step just taken (omit at the start).")
    a.add_argument("--observation", default=None, help="What the last action returned.")
    a.add_argument(
        "--mode",
        choices=("raw", "generative"),
        default="raw",
        help="raw (default): the subgraph as text, no LLM call; generative: one backend LLM call.",
    )
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")

    p = sub.add_parser(
        "agent",
        help="Ask the backend's GraphRAG Navigator agent; prints its step trace (backend LLM).",
        description="Ask the backend's GraphRAG Navigator: it answers by walking the knowledge "
        "graph, one backend LLM call per step (two with generative guidance), and prints its "
        "step trace. The request waits up to max($SYNAPSE_TIMEOUT, 600) seconds. If the client "
        "disconnects or times out, the backend keeps working (and spending) until the run ends.",
    )
    p.add_argument("query")
    which = p.add_mutually_exclusive_group()
    which.add_argument(
        "--graph", default=None, metavar="NAME", help="Procedural graph (default: the backend's)."
    )
    which.add_argument("--no-graph", action="store_true", help="Run without a procedural graph.")
    p.add_argument("--guidance", choices=("none", "raw", "generative"), default=None)
    p.add_argument("--max-steps", type=_int_range(1, 20), default=None, metavar="N")
    p.add_argument(
        "--record", action="store_true", help="Store the trajectory (unscored) on the backend."
    )
    p.add_argument("--json", action="store_true", help="Emit the full JSON result.")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser(
        "evolve",
        help="Self-evolve a procedural graph from QA pairs (costs backend LLM calls; asks first).",
        description="Self-evolve a procedural graph from QA pairs (Algorithm 1 of "
        "arXiv:2609.09153). It costs backend LLM calls: an upper bound is printed and nothing "
        "starts without --yes or an interactive y. Requests wait up to max($SYNAPSE_TIMEOUT, "
        "600) seconds and the progress stream has no idle timeout. If the client disconnects "
        "(Ctrl-C included), the backend keeps working (and spending, up to the cap) until the "
        "run ends.",
    )
    p.add_argument("name", metavar="NAME")
    p.add_argument("--train", required=True, metavar="FILE", help="JSON array or JSONL of {question, answer}.")
    p.add_argument("--val", required=True, metavar="FILE", help="Validation set, same format.")
    p.add_argument("--train-split", default=None, metavar="SPLIT", help="Keep only items whose \"split\" is SPLIT.")
    p.add_argument("--val-split", default=None, metavar="SPLIT", help="Same, for --val.")
    p.add_argument("--rounds", type=_int_range(1, 20), default=EVOLVE_ROUNDS, metavar="N")
    p.add_argument("--batch-size", type=_int_range(1, 100), default=EVOLVE_BATCH_SIZE, metavar="N")
    p.add_argument("--mode", choices=("static", "scratch"), default="static")
    p.add_argument("--metric", choices=("f1", "em"), default="f1")
    p.add_argument("--guidance", choices=("raw", "generative"), default="raw")
    p.add_argument(
        "--max-llm-calls",
        # Bounded like the backend's own field: a larger cap would be refused with a 422
        # after consent was asked for a cap that could never be sent.
        type=_int_range(1, MAX_LLM_CALLS_LIMIT),
        default=None,
        metavar="N",
        help=f"Hard cap on backend LLM calls, 1-{MAX_LLM_CALLS_LIMIT} (the backend's limit; "
        "default: the printed upper bound). It must pay for the baseline plus one full round; "
        "a smaller cap is refused with the minimum.",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="With --mode scratch on a NAME that already holds a graph: score that graph once on "
        "the validation set (counted in the budget) and overwrite it only with a candidate that "
        "scores at least as well. Without it, such a run is refused.",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Start without the confirmation prompt.")
    p.set_defaults(func=cmd_evolve)


def _add_lab_run_flags(a: argparse.ArgumentParser) -> None:
    """The run configuration shared by ``lab estimate`` and ``lab run``."""
    a.add_argument(
        "--dataset",
        default="demo",
        metavar="ID",
        help="demo (default), hotpotqa, or qa-file:NAME for an uploaded file (see `lab datasets`).",
    )
    a.add_argument("--split", default=None, help="demo: train | val | test (default) | all; qa-file: a split field value.")
    a.add_argument("--n", type=_int_range(1, 100_000), default=None, metavar="N", help="Cap the number of questions.")
    a.add_argument("--offset", type=_int_range(0), default=None, metavar="N", help="hotpotqa: skip the first N sampled questions (disjoint samples).")
    a.add_argument("--sample-seed", type=int, default=None, metavar="SEED", help="hotpotqa: the sampling seed.")
    a.add_argument("--arms", type=_csv, default=None, metavar="LIST", help="Comma-separated arms (default: every arm; see `lab arms`).")
    a.add_argument(
        "--budgets",
        type=_budgets_arg,
        default=None,
        metavar="LIST",
        help="Context budgets in tokens, e.g. 500,1k,2k,4k,8k,default (default: default = each arm's own context).",
    )
    a.add_argument(
        "--extra-cell",
        type=_cell_arg,
        action="append",
        default=None,
        metavar="ARM@BUDGET",
        help="An extra (arm, budget) cell beyond the grid, e.g. synapse_d@default (repeatable).",
    )
    a.add_argument("--k", type=_int_range(1, 100), default=None, metavar="N", help="Seeds / passages per arm (default 8).")
    a.add_argument(
        "--passage-k", type=_int_range(1, 100), default=None, metavar="N",
        help="Exactly N passages for passage-ranking arms at every budget (default: as many as "
        "each capped budget holds; k at the default context).",
    )
    a.add_argument("--seed", type=int, default=None, help="Run seed (reader seed and N2's random draw).")
    a.add_argument("--reader-model", default=None, metavar="MODEL", help="Reader model (default: the backend's, gpt-5-nano; see `lab models`).")
    a.add_argument(
        "--mode",
        choices=LAB_MODES,
        default="retrieve",
        help="retrieve (default): free, no model called, retrieval-level columns only; "
        "realtime: the reader answers now; batch: OpenAI Batch, half price, collected by `lab resume`.",
    )
    a.add_argument("--max-usd", type=_usd_arg, default=None, metavar="USD", help="Hard spend cap; required to run a paid mode.")
    a.add_argument("--order", choices=LAB_ORDERS, default=None, help="Packing placement: score (default) or PathRAG's ascending.")
    a.add_argument("--reasoning-allowance", type=_int_range(0, 100_000), default=None, metavar="N", help="Reasoning tokens allowed per call on a reasoning model.")
    a.add_argument("--concurrency", type=_int_range(1, 32), default=None, metavar="N", help="Realtime requests in flight (default 4).")
    a.add_argument("--ingest", action="store_true", help="hotpotqa: also price the ingest of the sample's corpus.")
    a.add_argument("--ingest-paragraphs", type=_int_range(0), default=None, metavar="N", help="Price an ingest of N paragraphs next to the run.")
    a.add_argument("--ingest-model", default="gpt-4o-mini", metavar="MODEL", help="Extraction model for the ingest estimate.")
    a.add_argument("--ingest-realtime", action="store_true", help="Price the ingest at realtime rates (default: Batch).")
    a.add_argument("--ingest-usd", type=_usd_arg, default=None, metavar="USD", help="The corpus' MEASURED ingest spend (amortized cost-of-pass).")
    a.add_argument("--measured-from", default=None, metavar="RUN_ID", help="Price on the measured contexts of an earlier retrieve run of this configuration.")


def _add_lab_parsers(sub: Any) -> None:
    p = sub.add_parser(
        "lab",
        help="Synapse Lab: compare retrieval approaches on your data, on quality AND cost.",
        description="Synapse Lab: compare retrieval approaches (arms) side by side on your own "
        "questions, scored on answer quality AND cost next to three evidence floors "
        "(closed-book, vocabulary, random context). `--mode retrieve` (the default) is free: no "
        "model is called. `run` always prints the estimate first and starts nothing without "
        "--yes or an interactive y; a paid mode also needs --max-usd. Without an action, lists "
        "the arms.",
    )
    p.set_defaults(func=cmd_lab, lab_command=None, json=False)
    actions = p.add_subparsers(dest="lab_command", metavar="ACTION")

    a = actions.add_parser("arms", help="The arms, grouped: evidence floors, passage baselines, graph arms.")
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    a = actions.add_parser("models", help="Reader models with their $/1M tokens (realtime and Batch).")
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    a = actions.add_parser("datasets", help="Datasets a run can use now, and why others cannot.")
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")
    a = actions.add_parser("upload", help="Store a QA file (JSON array or JSONL of {question, answer}).")
    a.add_argument("file", metavar="FILE")
    a.add_argument("--name", default=None, help="Store it under another file name.")
    a.add_argument("--replace", action="store_true", help="Overwrite a different file stored under the same name.")

    a = actions.add_parser(
        "estimate",
        help="Price a run for free: point estimate and upper bound per arm × budget (exit 1 if refused).",
    )
    _add_lab_run_flags(a)
    a.add_argument("--json", action="store_true", help="Emit the full estimate JSON.")

    a = actions.add_parser(
        "run",
        help="Start a run: prints the estimate, asks for consent, then follows its progress.",
        description="Start a Lab run. The estimate is printed first; nothing starts without "
        "--yes or an interactive y (never on a piped stdin), and a paid mode (realtime, batch) "
        "needs --max-usd, which the backend enforces. Ctrl-C stops listening, not the run.",
    )
    _add_lab_run_flags(a)
    a.add_argument("--run-id", default=None, metavar="ID", help="Name the run (default: timestamp + config hash).")
    a.add_argument("-y", "--yes", action="store_true", help="Start without the confirmation prompt.")
    a.add_argument("--json", action="store_true", help="Emit the final summary as JSON.")

    a = actions.add_parser("runs", help="Every run, newest first.")
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")

    a = actions.add_parser("show", help="A run's leaderboard (and, with --rows, per-question rows).")
    a.add_argument("run_id", metavar="RUN_ID")
    a.add_argument("--rows", type=_int_range(0, 1000), default=0, metavar="N", help="Also print N per-question rows.")
    a.add_argument("--arm", default=None, help="Rows of one arm only.")
    a.add_argument("--budget", default=None, help="Rows of one budget only (a token count or default).")
    a.add_argument("--json", action="store_true", help="Emit the full JSON payload.")

    a = actions.add_parser(
        "resume",
        help="Collect a batch run, or finish an aborted/refused one (asks first when that spends).",
    )
    a.add_argument("run_id", metavar="RUN_ID")
    a.add_argument("--max-usd", type=_usd_arg, default=None, metavar="USD", help="A new cap for the rest of the run.")
    a.add_argument("--retry-failed", action="store_true", help="Re-read requests that came back with an error.")
    a.add_argument("-y", "--yes", action="store_true", help="Resume without the confirmation prompt.")
    a.add_argument("--json", action="store_true", help="Emit the final summary as JSON.")


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
