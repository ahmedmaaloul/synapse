# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Zero-API-Key Demo Seeder

Loads a hand-curated, pre-extracted knowledge graph (``demo_graph.json`` — a
brief history of AI, from Babbage's Analytical Engine to GraphRAG) straight into
Neo4j so a fresh clone shows a real, populated graph in well under a minute.

Why it matters: the extraction step is the only part of the pipeline that needs
an LLM. By shipping the *result* of extraction we skip that step entirely — no
Gemini / Anthropic / Ollama key required — while still exercising the project's
real services (schema bootstrap, embedding, node + relationship writers). What
you see after seeding is produced by the same code paths a live ingest uses.

Usage (from ``backend/``):

    python -m scripts.seed_demo                 # seed, keeping existing data
    python -m scripts.seed_demo --clear         # wipe Synapse's own nodes first
    EMBEDDING_PROVIDER=fake python -m scripts.seed_demo   # fully offline, no model download

Exit codes: 0 success · 1 bad/missing demo file · 2 Neo4j unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from app import neo4j_driver
from app.config import get_settings
from app.services import graph_builder
from app.services.graph_schema import ensure_schema

logger = logging.getLogger("scripts.seed_demo")

DEMO_GRAPH_PATH = Path(__file__).with_name("demo_graph.json")

# Entity types the frontend has a colour for (see frontend/src/app/lib/constants.ts).
# Keeping the demo inside this set guarantees a legible, fully-coloured graph.
ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "PERSON",
        "ORGANIZATION",
        "COMPANY",
        "UNIVERSITY",
        "CONCEPT",
        "PROJECT",
        "SKILL",
        "TOOL",
        "FRAMEWORK",
        "DATABASE",
        "EVENT",
        "LOCATION",
        "THING",
    }
)

#: Every node label Synapse itself writes. ``--clear`` must cover all of
#: them: it used to delete only ``:Entity``, which left the previous run's
#: ``:Community`` nodes behind — and since ``make demo`` is the README's headline
#: command, a re-seed was the most likely way anyone saw ghost themes in the UI.
#: Keep this list in step with any new label the backend starts writing.
SYNAPSE_LABELS: tuple[str, ...] = ("Entity", "Community", "Chunk")

# --clear wipes Synapse's own nodes (and their relationships) for a clean demo.
# Scoped by label on purpose: a bare `MATCH (n) DETACH DELETE n` would empty a
# database Synapse happens to share with something else.
CLEAR_QUERY = (
    f"MATCH (n) WHERE {' OR '.join(f'n:{label}' for label in SYNAPSE_LABELS)} DETACH DELETE n"
)


class DemoGraphError(RuntimeError):
    """Raised when ``demo_graph.json`` is missing or structurally invalid."""


# ── Loading & validation ─────────────────────────────────
def validate_demo_graph(payload: Any) -> list[str]:
    """Return a list of human-readable problems with a demo graph payload.

    An empty list means the payload is safe to load. Checks: required keys,
    per-entity name/type/description, allowed types, duplicate names, and that
    every relationship endpoint resolves to a declared entity.
    """
    problems: list[str] = []

    if not isinstance(payload, dict):
        return ["top-level JSON must be an object"]

    entities = payload.get("entities")
    relationships = payload.get("relationships")
    if not isinstance(entities, list) or not entities:
        problems.append("'entities' must be a non-empty list")
        entities = []
    if not isinstance(relationships, list):
        problems.append("'relationships' must be a list")
        relationships = []

    names: set[str] = set()
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            problems.append(f"entity #{i} is not an object")
            continue
        name = str(entity.get("name", "")).strip()
        etype = str(entity.get("type", "")).strip()
        description = str(entity.get("description", "")).strip()
        label = name or f"#{i}"
        if not name:
            problems.append(f"entity #{i} has no name")
        elif name in names:
            problems.append(f"duplicate entity name: {name!r}")
        else:
            names.add(name)
        if not etype:
            problems.append(f"entity {label!r} has no type")
        elif etype not in ALLOWED_ENTITY_TYPES:
            problems.append(f"entity {label!r} has unsupported type {etype!r}")
        if not description:
            problems.append(f"entity {label!r} has no description")

    for i, rel in enumerate(relationships):
        if not isinstance(rel, dict):
            problems.append(f"relationship #{i} is not an object")
            continue
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        rel_type = str(rel.get("type", "")).strip()
        if not rel_type:
            problems.append(f"relationship #{i} ({source} -> {target}) has no type")
        for endpoint, role in ((source, "source"), (target, "target")):
            if not endpoint:
                problems.append(f"relationship #{i} has an empty {role}")
            elif endpoint not in names:
                problems.append(
                    f"relationship #{i} {role} {endpoint!r} is not a declared entity"
                )

    return problems


def load_demo_graph(path: Path = DEMO_GRAPH_PATH) -> dict:
    """Read and validate the demo graph, raising ``DemoGraphError`` on any issue."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise DemoGraphError(f"Could not read demo graph at {path}: {e}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DemoGraphError(f"{path} is not valid JSON: {e}") from e

    problems = validate_demo_graph(payload)
    if problems:
        listed = "\n  - ".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n  ... and {len(problems) - 20} more"
        raise DemoGraphError(f"{path} is invalid:\n  - {listed}{more}")

    return payload


# ── Seeding ──────────────────────────────────────────────
async def seed(
    payload: dict,
    *,
    clear: bool = False,
    document_name: str | None = None,
    echo: bool = True,
) -> dict:
    """Write the demo graph to Neo4j using the project's own services.

    Reuses ``ensure_schema`` plus ``graph_builder._embed_entities`` /
    ``_write_entities`` / ``_write_relationships`` so seeding exercises exactly
    the production write path — only the LLM extraction step is pre-computed.
    """

    def say(message: str) -> None:
        if echo:
            print(message)

    entities: list[dict] = payload["entities"]
    relationships: list[dict] = payload["relationships"]
    doc = document_name or payload.get("document_name") or "Synapse Demo Graph"

    say(f"📚 Demo graph: {len(entities)} entities, {len(relationships)} relationships")

    say("🧱 Ensuring Neo4j schema (constraint, full-text index, vector index)...")
    await ensure_schema()

    cleared = False
    if clear:
        say(f"🧨 Clearing existing Synapse nodes ({', '.join(':' + n for n in SYNAPSE_LABELS)})...")
        await neo4j_driver.execute_query(CLEAR_QUERY)
        cleared = True

    settings = get_settings()
    say(f"🧠 Embedding entities with EMBEDDING_PROVIDER={settings.embedding_provider}...")
    embeddings_by_name = await graph_builder._embed_entities(entities)
    if embeddings_by_name:
        say(f"   ✅ Embedded {len(embeddings_by_name)} entities")
    else:
        say("   ⚠️  No embeddings produced — graph still loads, retrieval falls back to keywords")

    say("✍️  Writing nodes...")
    nodes_created = await graph_builder._write_entities(entities, doc, embeddings_by_name)

    say("🔗 Writing relationships...")
    rels_created = await graph_builder._write_relationships(relationships)

    return {
        "document_name": doc,
        "entities_in_file": len(entities),
        "relationships_in_file": len(relationships),
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "embedded": len(embeddings_by_name),
        "cleared": cleared,
    }


def _print_summary(result: dict) -> None:
    print("\n" + "─" * 58)
    print("✅ Demo graph loaded")
    print(f"   document      : {result['document_name']}")
    print(f"   nodes written : {result['nodes_created']}/{result['entities_in_file']}")
    print(f"   edges written : {result['relationships_created']}/{result['relationships_in_file']}")
    print(f"   embedded      : {result['embedded']}")
    print("─" * 58)
    print("Next: open http://localhost:3000 and ask the graph a question,")
    print("e.g. \"How does AlexNet connect to the Transformer?\"\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed_demo",
        description="Seed Neo4j with Synapse's zero-API-key demo knowledge graph.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "delete every node Synapse writes "
            f"({', '.join(':' + n for n in SYNAPSE_LABELS)}) before seeding"
        ),
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEMO_GRAPH_PATH,
        help=f"path to a demo graph JSON file (default: {DEMO_GRAPH_PATH.name})",
    )
    parser.add_argument(
        "--document-name",
        default=None,
        help="override the source document label stored on every node",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the JSON file and exit without touching Neo4j",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


async def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    echo = not args.quiet
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    # The driver logs a server-side deprecation notice for every single write;
    # it's harmless and would bury the progress output under 150 identical lines.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        payload = load_demo_graph(args.file)
    except DemoGraphError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if args.validate_only:
        if echo:
            print(
                f"✅ {args.file} is valid: "
                f"{len(payload['entities'])} entities, {len(payload['relationships'])} relationships"
            )
        return 0

    settings = get_settings()
    if not await neo4j_driver.verify_connectivity():
        print(
            f"❌ Cannot reach Neo4j at {settings.neo4j_uri}.\n"
            "   Start it first:  docker compose up -d neo4j\n"
            "   Then re-run:     python -m scripts.seed_demo --clear\n"
            "   (override with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)",
            file=sys.stderr,
        )
        return 2

    try:
        result = await seed(
            payload,
            clear=args.clear,
            document_name=args.document_name,
            echo=echo,
        )
    finally:
        await neo4j_driver.close_driver()

    if echo:
        _print_summary(result)

    if result["nodes_created"] == 0:
        print("❌ No nodes were written — check the Neo4j logs.", file=sys.stderr)
        return 2
    return 0


def run() -> int:
    return asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(run())
