# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the zero-API-key demo seeder.

Two concerns:
  1. ``demo_graph.json`` is well-formed — no dangling relationship endpoints,
     no duplicate names, every entity typed with a colour the frontend knows.
  2. ``seed()`` drives the real service write path, with Neo4j faked out.

Everything here is hermetic: no network, no database, no LLM (conftest pins
EMBEDDING_PROVIDER=fake).
"""

from __future__ import annotations

import collections
import json
import re

import pytest

from app import neo4j_driver
from scripts import seed_demo
from scripts.seed_demo import (
    ALLOWED_ENTITY_TYPES,
    DEMO_GRAPH_PATH,
    DemoGraphError,
    load_demo_graph,
    validate_demo_graph,
)


@pytest.fixture(scope="module")
def demo() -> dict:
    return json.loads(DEMO_GRAPH_PATH.read_text(encoding="utf-8"))


# ── The shipped dataset ──────────────────────────────────
class TestDemoGraphFile:
    def test_file_exists_and_parses(self, demo):
        assert isinstance(demo, dict)
        assert demo["document_name"]

    def test_passes_full_validation(self):
        # Raises DemoGraphError with a detailed report if anything is off.
        payload = load_demo_graph()
        assert payload["entities"]

    def test_entity_names_are_unique(self, demo):
        names = [e["name"] for e in demo["entities"]]
        dupes = [n for n, c in collections.Counter(names).items() if c > 1]
        assert dupes == []

    def test_every_entity_has_name_type_description(self, demo):
        for entity in demo["entities"]:
            assert entity.get("name", "").strip(), entity
            assert entity.get("type", "").strip(), entity
            assert entity.get("description", "").strip(), entity

    def test_every_type_is_renderable_by_the_frontend(self, demo):
        types = {e["type"] for e in demo["entities"]}
        assert types <= ALLOWED_ENTITY_TYPES, types - ALLOWED_ENTITY_TYPES

    def test_every_relationship_endpoint_resolves(self, demo):
        names = {e["name"] for e in demo["entities"]}
        dangling = [
            (r["source"], r["type"], r["target"])
            for r in demo["relationships"]
            if r["source"] not in names or r["target"] not in names
        ]
        assert dangling == []

    def test_relationships_are_typed_and_described(self, demo):
        for rel in demo["relationships"]:
            assert rel.get("type", "").strip(), rel
            assert rel.get("description", "").strip(), rel

    def test_no_self_loops(self, demo):
        assert [r for r in demo["relationships"] if r["source"] == r["target"]] == []

    def test_no_duplicate_edges(self, demo):
        keys = [(r["source"], r["type"], r["target"]) for r in demo["relationships"]]
        dupes = [k for k, c in collections.Counter(keys).items() if c > 1]
        assert dupes == []

    def test_is_substantial(self, demo):
        assert len(demo["entities"]) >= 35
        assert len(demo["relationships"]) >= 50

    def test_every_entity_is_connected(self, demo):
        """No orphan nodes — an isolated dot in the graph view looks broken."""
        linked = {r["source"] for r in demo["relationships"]}
        linked |= {r["target"] for r in demo["relationships"]}
        orphans = sorted({e["name"] for e in demo["entities"]} - linked)
        assert orphans == []

    def test_not_a_star_topology(self, demo):
        """Rich clustering, not one hub: no node owns more than a third of edges."""
        degree: collections.Counter[str] = collections.Counter()
        for rel in demo["relationships"]:
            degree[rel["source"]] += 1
            degree[rel["target"]] += 1
        _, top = degree.most_common(1)[0]
        assert top < len(demo["relationships"]) / 3

    def test_has_multi_hop_paths(self, demo):
        """A 3+ hop path must exist, otherwise GraphRAG has nothing to traverse."""
        adjacency: dict[str, set[str]] = collections.defaultdict(set)
        for rel in demo["relationships"]:
            adjacency[rel["source"]].add(rel["target"])
            adjacency[rel["target"]].add(rel["source"])

        start = "Alan Turing"
        seen = {start}
        frontier = {start}
        for _ in range(3):
            frontier = {n for node in frontier for n in adjacency[node]} - seen
            seen |= frontier
        assert frontier, "no nodes exactly 3 hops from Alan Turing"


# ── Validator ────────────────────────────────────────────
def _valid_payload() -> dict:
    return {
        "entities": [
            {"name": "Ada Lovelace", "type": "PERSON", "description": "First programmer."},
            {"name": "Analytical Engine", "type": "PROJECT", "description": "Mechanical computer."},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "type": "WROTE_PROGRAM_FOR",
                "description": "Wrote the first algorithm.",
            }
        ],
    }


class TestValidateDemoGraph:
    def test_accepts_a_good_payload(self):
        assert validate_demo_graph(_valid_payload()) == []

    def test_rejects_non_object(self):
        assert validate_demo_graph([1, 2, 3])

    def test_rejects_empty_entities(self):
        payload = _valid_payload()
        payload["entities"] = []
        assert any("non-empty" in p for p in validate_demo_graph(payload))

    def test_rejects_duplicate_names(self):
        payload = _valid_payload()
        payload["entities"].append(
            {"name": "Ada Lovelace", "type": "PERSON", "description": "Again."}
        )
        assert any("duplicate" in p for p in validate_demo_graph(payload))

    def test_rejects_unknown_type(self):
        payload = _valid_payload()
        payload["entities"][0]["type"] = "WIZARD"
        assert any("unsupported type" in p for p in validate_demo_graph(payload))

    def test_rejects_missing_description(self):
        payload = _valid_payload()
        payload["entities"][0]["description"] = "  "
        assert any("no description" in p for p in validate_demo_graph(payload))

    def test_rejects_dangling_relationship_endpoint(self):
        payload = _valid_payload()
        payload["relationships"][0]["target"] = "Ghost Node"
        assert any("not a declared entity" in p for p in validate_demo_graph(payload))

    def test_rejects_untyped_relationship(self):
        payload = _valid_payload()
        payload["relationships"][0]["type"] = ""
        assert any("has no type" in p for p in validate_demo_graph(payload))


class TestLoadDemoGraph:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DemoGraphError, match="Could not read"):
            load_demo_graph(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(DemoGraphError, match="not valid JSON"):
            load_demo_graph(path)

    def test_structurally_invalid(self, tmp_path):
        path = tmp_path / "bad.json"
        payload = _valid_payload()
        payload["relationships"][0]["source"] = "Nobody"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(DemoGraphError, match="not a declared entity"):
            load_demo_graph(path)


# ── The loader against a fake Neo4j ──────────────────────
def _handler(query: str, params: dict):
    return []


class TestSeed:
    async def test_writes_expected_node_and_edge_counts(self, fake_neo4j, capsys):
        calls = fake_neo4j(_handler)
        payload = load_demo_graph()

        result = await seed_demo.seed(payload)

        assert result["nodes_created"] == len(payload["entities"])
        assert result["relationships_created"] == len(payload["relationships"])
        assert result["entities_in_file"] == len(payload["entities"])
        assert result["relationships_in_file"] == len(payload["relationships"])
        assert result["cleared"] is False

        merges = [q for q, _ in calls if "MERGE (n:Entity {name: $name})" in q]
        assert len(merges) == len(payload["entities"])
        apoc = [q for q, _ in calls if "apoc.merge.relationship" in q]
        assert len(apoc) == len(payload["relationships"])

        assert "Demo graph:" in capsys.readouterr().out

    async def test_ensures_schema_before_writing(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        await seed_demo.seed(_valid_payload(), echo=False)

        queries = [q for q, _ in calls]
        assert any("CREATE CONSTRAINT" in q for q in queries)
        assert any("CREATE FULLTEXT INDEX" in q for q in queries)
        assert any("CREATE VECTOR INDEX" in q for q in queries)
        first_write = next(i for i, q in enumerate(queries) if "MERGE (n:Entity" in q)
        assert queries.index(next(q for q in queries if "CREATE CONSTRAINT" in q)) < first_write

    async def test_embeds_every_entity(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        payload = _valid_payload()

        result = await seed_demo.seed(payload, echo=False)

        assert result["embedded"] == len(payload["entities"])
        node_params = [p for q, p in calls if "MERGE (n:Entity {name: $name})" in q]
        assert all(p["embedding"] for p in node_params)
        assert len(node_params[0]["embedding"]) == 384

    async def test_stores_document_name_on_every_node(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        await seed_demo.seed(_valid_payload(), document_name="My Demo", echo=False)
        node_params = [p for q, p in calls if "MERGE (n:Entity {name: $name})" in q]
        assert {p["document"] for p in node_params} == {"My Demo"}

    async def test_clear_flag_wipes_first(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        result = await seed_demo.seed(_valid_payload(), clear=True, echo=False)

        queries = [q for q, _ in calls]
        assert seed_demo.CLEAR_QUERY in queries
        assert result["cleared"] is True
        assert queries.index(seed_demo.CLEAR_QUERY) < next(
            i for i, q in enumerate(queries) if "MERGE (n:Entity" in q
        )

    async def test_no_clear_by_default(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        await seed_demo.seed(_valid_payload(), echo=False)
        assert seed_demo.CLEAR_QUERY not in [q for q, _ in calls]

    async def test_falls_back_to_related_to_without_apoc(self, fake_neo4j):
        def no_apoc(query: str, params: dict):
            if "apoc.merge.relationship" in query:
                raise RuntimeError("Unknown procedure apoc.merge.relationship")
            return []

        calls = fake_neo4j(no_apoc)
        result = await seed_demo.seed(_valid_payload(), echo=False)

        assert result["relationships_created"] == 1
        assert any("MERGE (a)-[r:RELATED_TO]->(b)" in q for q, _ in calls)


class TestClearCoversEveryLabelSynapseWrites:
    """``--clear`` used to be ``MATCH (n:Entity) DETACH DELETE n``.

    That left every :Community node from the previous run in the database, so a
    re-seed produced a fresh graph decorated with themes describing entities that
    no longer existed — and ``make demo`` is the README's headline command, which
    made this the most likely way anyone ever saw ghost themes.
    """

    def test_clear_query_removes_every_synapse_label(self):
        for label in ("Entity", "Community", "Chunk"):
            assert label in seed_demo.SYNAPSE_LABELS
            assert f"n:{label}" in seed_demo.CLEAR_QUERY
        assert "DETACH DELETE n" in seed_demo.CLEAR_QUERY

    def test_clear_query_is_scoped_and_never_a_whole_database_wipe(self):
        """Synapse may share a database; the delete must name the labels it owns."""
        assert "MATCH (n) DETACH DELETE n" not in seed_demo.CLEAR_QUERY
        assert "WHERE" in seed_demo.CLEAR_QUERY
        # Every label mentioned in the query is one we declared.
        mentioned = set(re.findall(r"n:(\w+)", seed_demo.CLEAR_QUERY))
        assert mentioned == set(seed_demo.SYNAPSE_LABELS)

    async def test_clearing_wipes_stale_communities_in_the_same_query(self, fake_neo4j):
        calls = fake_neo4j(_handler)
        await seed_demo.seed(_valid_payload(), clear=True, echo=False)

        clears = [q for q, _ in calls if "DETACH DELETE n" in q]
        assert len(clears) == 1  # one scoped round-trip, not one per label
        assert "n:Community" in clears[0]
        assert "n:Chunk" in clears[0]


# ── CLI ──────────────────────────────────────────────────
class TestMain:
    async def test_validate_only_needs_no_database(self, capsys):
        assert await seed_demo.main(["--validate-only"]) == 0
        assert "is valid" in capsys.readouterr().out

    async def test_bad_file_exits_1(self, tmp_path, capsys):
        missing = tmp_path / "gone.json"
        assert await seed_demo.main(["--file", str(missing)]) == 1
        assert "Could not read" in capsys.readouterr().err

    async def test_unreachable_neo4j_exits_2_with_help(self, monkeypatch, capsys):
        async def unreachable() -> bool:
            return False

        monkeypatch.setattr(neo4j_driver, "verify_connectivity", unreachable)

        assert await seed_demo.main([]) == 2
        err = capsys.readouterr().err
        assert "Cannot reach Neo4j" in err
        assert "docker compose up -d neo4j" in err

    async def test_happy_path_exits_0(self, monkeypatch, fake_neo4j, capsys):
        fake_neo4j(_handler)

        async def reachable() -> bool:
            return True

        async def noop() -> None:
            return None

        monkeypatch.setattr(neo4j_driver, "verify_connectivity", reachable)
        monkeypatch.setattr(neo4j_driver, "close_driver", noop)

        assert await seed_demo.main(["--clear"]) == 0
        out = capsys.readouterr().out
        assert "Demo graph loaded" in out
