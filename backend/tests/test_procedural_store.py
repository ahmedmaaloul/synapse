# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for procedural memory persistence, the write-batch driver call, the
``fake_neo4j`` write-batch coverage, the schema bootstrap and the startup seed.

Fully hermetic. ``_MemoryNeo4j`` answers the store's OWN Cypher constants from
Python state, so save → load → rollback → delete round-trips exercise every
parameter the store sends and every row shape it parses. It does not execute
Cypher: the statements' semantics against a real Neo4j are integration-test
territory (``SYNAPSE_IT=1``).
"""

from __future__ import annotations

import json
import logging
import sys
from types import ModuleType

import pytest

from app import neo4j_driver
from app.services import graph_schema
from app.services import procedural_graph as pg
from app.services import procedural_store as store
from app.services.procedural_graph import InvalidProceduralGraph, ProceduralGraph

# ── A tiny stand-in for Neo4j that speaks the store's queries ──


class _MemoryNeo4j:
    """Answers exactly the Cypher constants ``procedural_store`` defines.

    Any other query is a test failure: the store must not grow an untested
    statement silently.
    """

    def __init__(self) -> None:
        self.graphs: dict[str, dict] = {}
        self.live: dict[str, tuple[list[dict], list[dict]]] = {}
        self.versions: dict[str, dict] = {}
        self.rejections: list[dict] = []
        self.trajectories: list[dict] = []

    def __call__(self, query: str, p: dict) -> list[dict]:
        if query == store.UPSERT_GRAPH_QUERY:
            g = self.graphs.setdefault(p["name"], {"name": p["name"], "created_at": p["now"]})
            g.update(
                version=g.get("version", 0) + 1,
                score=p["score"],
                description=p["description"],
                cycle_policy=p["cycle_policy"],
                tools=list(p["tools"]),
                node_count=p["node_count"],
                edge_count=p["edge_count"],
                updated_at=p["now"],
            )
            return [{"version": g["version"]}]
        if query == store.WIPE_LIVE_QUERY:
            self.live.pop(p["name"], None)
            return []
        if query == store.WRITE_NODES_QUERY:
            self.live[p["name"]] = ([dict(n) for n in p["nodes"]], [])
            return []
        if query == store.WRITE_EDGES_QUERY:
            nodes, edges = self.live[p["name"]]
            by_uid = {n["uid"]: n for n in nodes}
            for e in p["edges"]:
                if e["source_uid"] in by_uid and e["target_uid"] in by_uid:
                    edges.append(
                        {
                            "source": by_uid[e["source_uid"]]["id"],
                            "target": by_uid[e["target_uid"]]["id"],
                            **{k: e[k] for k in ("relation", "condition", "guidance", "pitfalls", "ord")},
                        }
                    )
            return []
        if query == store.WRITE_VERSION_QUERY:
            g = self.graphs[p["name"]]
            uid = f"{p['name']}::v{g['version']}"
            assert uid not in self.versions, "procedure_version_uid_unique violated"
            self.versions[uid] = {
                "graph": p["name"],
                "version": g["version"],
                "score": p["score"],
                "accepted": True,
                "created_at": p["now"],
                "note": p["note"],
                "graph_json": p["graph_json"],
                "edits_json": p["edits_json"],
                "diff_json": p["diff_json"],
            }
            return [{"version": g["version"]}]
        if query == store.GET_META_QUERY:
            g = self.graphs.get(p["name"])
            return [self._meta_row(g)] if g else []
        if query == store.LIST_GRAPHS_QUERY:
            return [self._meta_row(self.graphs[name]) for name in sorted(self.graphs)]
        if query == store.LOAD_GRAPH_QUERY:
            g = self.graphs.get(p["name"])
            if not g:
                return []
            nodes, edges = self.live.get(p["name"], ([], []))
            meta = self._meta_row(g)
            return [
                {
                    "meta": meta,
                    "nodes": [
                        {k: n[k] for k in ("id", "type", "description")}
                        for n in sorted(nodes, key=lambda n: n["ord"])
                    ],
                    "edges": [
                        {k: e[k] for k in ("source", "target", "relation", "condition", "guidance", "pitfalls")}
                        for e in sorted(edges, key=lambda e: e["ord"])
                    ],
                }
            ]
        if query == store.LIST_VERSIONS_QUERY:
            rows = [v for v in self.versions.values() if v["graph"] == p["name"]]
            rows.sort(key=lambda v: v["version"], reverse=True)
            return rows[: p["limit"]]
        if query == store.GET_VERSION_QUERY:
            v = self.versions.get(p["uid"])
            return [v] if v else []
        if query in store.DELETE_GRAPH_QUERIES:
            name = p["name"]
            if "Procedure {graph" in query:
                self.live.pop(name, None)
            elif "ProcedureVersion" in query:
                self.versions = {k: v for k, v in self.versions.items() if v["graph"] != name}
            elif "ProcedureRejection" in query:
                self.rejections = [r for r in self.rejections if r["graph"] != name]
            elif "ProcedureTrajectory" in query:
                self.trajectories = [t for t in self.trajectories if t["graph"] != name]
            elif "ProcedureGraph" in query:
                return [{"deleted": 1 if self.graphs.pop(name, None) else 0}]
            return []
        if query == store.WRITE_REJECTION_QUERY:
            self.rejections.append({"graph": p["name"], **{k: v for k, v in p.items() if k != "name"}})
            return []
        if query == store.LIST_REJECTIONS_QUERY:
            rows = [
                {**r, "created_at": r["now"]} for r in reversed(self.rejections) if r["graph"] == p["name"]
            ]
            return rows[: p["limit"]]
        if query == store.WRITE_TRAJECTORY_QUERY:
            self.trajectories.append({"graph": p["name"], **{k: v for k, v in p.items() if k != "name"}})
            return []
        if query == store.LIST_TRAJECTORIES_QUERY:
            rows = [
                {**t, "created_at": t["now"]}
                for t in reversed(self.trajectories)
                if t["graph"] == p["name"]
            ]
            return rows[: p["limit"]]
        raise AssertionError(f"unexpected query sent by procedural_store:\n{query}")

    @staticmethod
    def _meta_row(g: dict) -> dict:
        return {
            "name": g["name"],
            "version": g["version"],
            "score": g.get("score"),
            "description": g["description"],
            "cycle_policy": g["cycle_policy"],
            "tools": g["tools"],
            "nodes": g["node_count"],
            "edges": g["edge_count"],
            "created_at": g["created_at"],
            "updated_at": g["updated_at"],
        }


@pytest.fixture
def memory_neo4j(fake_neo4j):
    db = _MemoryNeo4j()
    calls = fake_neo4j(db)
    return db, calls


def _nav(name: str = "nav") -> ProceduralGraph:
    graph = pg.skeleton(name, tools=("search", "answer"), description="demo graph")
    edits = {
        "delete_edges": [{"source": "Start", "target": "End"}],
        "add_nodes": [
            {"id": "search", "type": "ACTION", "description": "Look it up."},
            {"id": "answer", "type": "ACTION", "description": "Reply."},
        ],
        "add_edges": [
            {"source": "Start", "target": "search", "guidance": "Search first."},
            {
                "source": "search",
                "target": "answer",
                "relation": "PROVIDES_INPUT_FOR",
                "condition": "When found",
                "guidance": "Answer briefly.",
                "pitfalls": "No guessing.",
            },
            {"source": "answer", "target": "End", "guidance": "Stop."},
        ],
    }
    candidate, diagnostics, _ = pg.prepare_candidate(graph, edits)
    assert diagnostics == [] and candidate is not None
    return candidate


# ── 1. fake_neo4j covers execute_write_batch ─────────
class TestFakeNeo4jWriteBatchCoverage:
    """Mirrors test_packaging.TestFakeNeo4jCoverage for the transactional writer."""

    @pytest.mark.parametrize("module_path", ["app.neo4j_driver", "app.services.procedural_store"])
    async def test_every_importer_is_patched(self, module_path, fake_neo4j):
        import importlib

        module = importlib.import_module(module_path)
        real = module.execute_write_batch

        calls = fake_neo4j(lambda q, p: [{"ok": True}])

        assert module.execute_write_batch is not real, (
            f"{module_path}.execute_write_batch still points at the real driver"
        )
        result = await module.execute_write_batch([("CREATE (a)", {"x": 1}), ("CREATE (b)", None)])
        assert result == [[{"ok": True}], [{"ok": True}]]
        assert calls == [("CREATE (a)", {"x": 1}), ("CREATE (b)", {})]
        assert calls.batches == [[("CREATE (a)", {"x": 1}), ("CREATE (b)", {})]]

    async def test_procedural_store_execute_query_is_patched(self, fake_neo4j):
        real = store.execute_query
        calls = fake_neo4j(lambda q, p: [{"n": 1}])
        assert store.execute_query is not real
        assert await store.execute_query("RETURN 1") == [{"n": 1}]
        assert calls == [("RETURN 1", {})]

    async def test_single_queries_and_batches_share_one_ordered_log(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: None)
        await neo4j_driver.execute_query("READ 1")
        assert await neo4j_driver.execute_write_batch([("W1", {}), ("W2", {})]) == [[], []]
        await neo4j_driver.execute_query("READ 2")
        await neo4j_driver.execute_write_batch([("W3", {})])
        assert [q for q, _ in calls] == ["READ 1", "W1", "W2", "READ 2", "W3"]
        assert [[q for q, _ in b] for b in calls.batches] == [["W1", "W2"], ["W3"]]

    async def test_a_newly_added_module_is_covered_automatically(self, fake_neo4j, monkeypatch):
        newcomer = ModuleType("app.services.brand_new_writer")
        newcomer.execute_write_batch = neo4j_driver.execute_write_batch  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "app.services.brand_new_writer", newcomer)

        calls = fake_neo4j(lambda q, p: [{"seen": 1}])

        assert await newcomer.execute_write_batch([("MERGE (n)", {})]) == [[{"seen": 1}]]
        assert calls.batches == [[("MERGE (n)", {})]]

    async def test_a_raising_handler_leaves_the_statements_it_saw_on_record(self, fake_neo4j):
        def handler(q, p):
            if q == "BOOM":
                raise RuntimeError("tx failed")
            return []

        calls = fake_neo4j(handler)
        with pytest.raises(RuntimeError):
            await neo4j_driver.execute_write_batch([("OK", {}), ("BOOM", {}), ("NEVER", {})])
        assert calls.batches == [[("OK", {}), ("BOOM", {})]]

    async def test_patches_are_undone_after_the_test(self):
        assert store.execute_write_batch.__module__ == "app.neo4j_driver"
        assert store.execute_query.__module__ == "app.neo4j_driver"


# ── 2. The real execute_write_batch ──────────────────
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    async def data(self):
        return self._rows


class _FakeTx:
    def __init__(self, log):
        self.log = log

    async def run(self, query, parameters):
        self.log.append(("run", query, parameters))
        return _FakeResult([{"echo": query}])


class _FakeSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.driver.log.append(("close",))

    async def execute_write(self, work):
        result = None
        for _ in range(self.driver.attempts):  # a managed tx may re-run the work
            self.driver.log.append(("begin",))
            result = await work(_FakeTx(self.driver.log))
        self.driver.log.append(("commit",))
        return result


class _FakeDriver:
    def __init__(self, attempts: int = 1):
        self.attempts = attempts
        self.log: list[tuple] = []
        self.sessions = 0

    def session(self):
        self.sessions += 1
        return _FakeSession(self)


class TestExecuteWriteBatch:
    @pytest.fixture
    def driver(self, monkeypatch):
        def install(attempts: int = 1) -> _FakeDriver:
            fake = _FakeDriver(attempts)

            async def get_driver():
                return fake

            monkeypatch.setattr(neo4j_driver, "get_driver", get_driver)
            return fake

        return install

    async def test_runs_every_statement_in_one_managed_write_transaction(self, driver):
        fake = driver()
        results = await neo4j_driver.execute_write_batch([("A", {"x": 1}), ("B", None)])
        assert results == [[{"echo": "A"}], [{"echo": "B"}]]
        assert fake.sessions == 1
        assert fake.log == [
            ("begin",),
            ("run", "A", {"x": 1}),
            ("run", "B", {}),
            ("commit",),
            ("close",),
        ]

    async def test_a_retried_transaction_does_not_duplicate_results(self, driver):
        driver(attempts=2)
        results = await neo4j_driver.execute_write_batch([("A", {}), ("B", {})])
        assert results == [[{"echo": "A"}], [{"echo": "B"}]]

    async def test_empty_batch_opens_no_session(self, driver):
        fake = driver()
        assert await neo4j_driver.execute_write_batch([]) == []
        assert fake.sessions == 0


# ── 3. Schema ────────────────────────────────────────
class TestProceduralSchema:
    def test_labels(self):
        assert graph_schema.PROCEDURAL_LABELS == (
            "Procedure",
            "ProcedureGraph",
            "ProcedureVersion",
            "ProcedureRejection",
            "ProcedureTrajectory",
        )

    async def test_ensure_schema_also_ensures_procedural_schema(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        await graph_schema.ensure_schema()
        text = "\n".join(q for q, _ in calls)
        assert "entity_name_unique" in text  # the existing schema still runs
        assert "CREATE CONSTRAINT procedure_uid_unique IF NOT EXISTS" in text
        assert "FOR (p:Procedure) REQUIRE p.uid IS UNIQUE" in text
        assert "FOR (g:ProcedureGraph) REQUIRE g.name IS UNIQUE" in text
        assert "FOR (v:ProcedureVersion) REQUIRE v.uid IS UNIQUE" in text
        for label in ("Procedure", "ProcedureVersion", "ProcedureRejection", "ProcedureTrajectory"):
            assert f":{label}) ON (" in text

    async def test_uses_community_edition_constraints_only(self, fake_neo4j):
        """Neo4j 5 Community: single-property uniqueness, no node keys / composites."""
        calls = fake_neo4j(lambda q, p: [])
        await graph_schema.ensure_procedural_schema()
        text = "\n".join(q for q, _ in calls)
        assert "NODE KEY" not in text.upper()
        assert "REQUIRE (" not in text
        assert all("IF NOT EXISTS" in q for q, _ in calls)

    async def test_a_failing_statement_is_not_fatal(self, fake_neo4j, caplog):
        def handler(q, p):
            if "procedure_uid_unique" in q:
                raise RuntimeError("old neo4j")
            return []

        calls = fake_neo4j(handler)
        with caplog.at_level(logging.WARNING):
            await graph_schema.ensure_procedural_schema()
        assert len(calls) == 7  # every statement still attempted
        assert "old neo4j" in caplog.text


# ── 4. Save / load ───────────────────────────────────
class TestSaveGraph:
    async def test_invalid_graph_raises_before_touching_the_db(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        bad = pg.skeleton("nav")
        del bad.nodes["Start"]
        with pytest.raises(InvalidProceduralGraph) as err:
            await store.save_graph(bad)
        assert any("Start" in d for d in err.value.diagnostics)
        assert calls == []

    async def test_one_atomic_batch_with_the_expected_statements(self, fake_neo4j):
        graph = _nav()
        calls = fake_neo4j(lambda q, p: [{"version": 4}] if "RETURN" in q and "version" in q else [])

        version = await store.save_graph(
            graph, score=0.75, note="evolution round 2", edits={"add_nodes": []}, previous=pg.skeleton("nav")
        )

        assert version == 4
        assert len(calls.batches) == 1
        assert len(calls) == 5, "with `previous` given, no read may happen"
        queries = [q for q, _ in calls.batches[0]]
        assert queries == [
            store.UPSERT_GRAPH_QUERY,
            store.WIPE_LIVE_QUERY,
            store.WRITE_NODES_QUERY,
            store.WRITE_EDGES_QUERY,
            store.WRITE_VERSION_QUERY,
        ]
        upsert, _, nodes, edges, snapshot = (p for _, p in calls.batches[0])
        assert upsert["score"] == 0.75
        assert upsert["tools"] == ["search", "answer"]
        assert (upsert["node_count"], upsert["edge_count"]) == (len(graph.nodes), len(graph.edges))
        assert [n["uid"] for n in nodes["nodes"]] == [f"nav::{nid}" for nid in graph.nodes]
        assert [n["ord"] for n in nodes["nodes"]] == list(range(len(graph.nodes)))
        assert edges["edges"][1] == {
            "source_uid": "nav::search",
            "target_uid": "nav::answer",
            "relation": "PROVIDES_INPUT_FOR",
            "condition": "When found",
            "guidance": "Answer briefly.",
            "pitfalls": "No guessing.",
            "ord": 1,
        }
        assert edges["edges"][0]["condition"] is None
        assert ProceduralGraph.from_dict(json.loads(snapshot["graph_json"])) == graph
        assert json.loads(snapshot["edits_json"]) == {"add_nodes": []}
        diff = json.loads(snapshot["diff_json"])
        assert {n["id"] for n in diff["added_nodes"]} == {"search", "answer"}
        assert snapshot["note"] == "evolution round 2"
        assert snapshot["now"] == upsert["now"]

    async def test_version_is_numbered_inside_the_transaction(self):
        """No read-then-write race: the counter is bumped by the batch itself."""
        assert "SET g.version = coalesce(g.version, 0) + 1" in store.UPSERT_GRAPH_QUERY
        assert "version: g.version" in store.WRITE_VERSION_QUERY
        assert "uid: $name + '::v' + toString(g.version)" in store.WRITE_VERSION_QUERY

    async def test_without_previous_the_stored_graph_is_read_for_the_diff(self, memory_neo4j):
        db, calls = memory_neo4j
        await store.save_graph(pg.skeleton("nav"))
        await store.save_graph(_nav())
        assert [q for q, _ in calls].count(store.LOAD_GRAPH_QUERY) == 2
        [latest, first] = await store.list_versions("nav")
        assert {n["id"] for n in first["diff"]["added_nodes"]} == {"Start", "End"}
        assert {n["id"] for n in latest["diff"]["added_nodes"]} == {"search", "answer"}
        assert latest["diff"]["removed_edges"][0]["target"] == "End"

    async def test_missing_version_row_is_an_error_not_a_silent_zero(self, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        with pytest.raises(RuntimeError, match="returned no version"):
            await store.save_graph(_nav(), previous=_nav())


class TestRoundTrip:
    async def test_save_then_load_is_exact(self, memory_neo4j):
        graph = _nav()
        assert await store.save_graph(graph, score=0.5) == 1
        loaded = await store.load_graph("nav")
        assert loaded == graph
        assert pg.serialize_full(loaded) == pg.serialize_full(graph)

    async def test_the_prior_round_trips_exactly(self, memory_neo4j):
        prior = store.load_prior("graphrag-navigator")
        await store.save_graph(prior)
        assert await store.load_graph("graphrag-navigator") == prior

    async def test_load_graph_with_meta_is_one_read(self, memory_neo4j):
        db, calls = memory_neo4j
        await store.save_graph(_nav(), score=0.25)
        before = len(calls)
        graph, meta = await store.load_graph_with_meta("nav")
        assert len(calls) == before + 1
        assert graph == _nav()
        assert meta["version"] == 1 and meta["score"] == 0.25
        assert (meta["nodes"], meta["edges"]) == (len(graph.nodes), len(graph.edges))
        assert meta["tools"] == ["search", "answer"]

    async def test_absent_graph(self, memory_neo4j):
        assert await store.load_graph("ghost") is None
        assert await store.load_graph_with_meta("ghost") is None
        assert await store.get_meta("ghost") is None

    async def test_null_entries_from_optional_matches_are_ignored(self, fake_neo4j):
        graph = _nav()
        row = {
            "meta": {"name": "nav", "version": 1, "tools": ["search", "answer"],
                     "description": "demo graph", "cycle_policy": "forbid"},
            "nodes": [n.to_dict() for n in graph.nodes.values()] + [None],
            "edges": [e.to_dict() for e in graph.edges] + [None],
        }
        fake_neo4j(lambda q, p: [row])
        assert await store.load_graph("nav") == graph

    async def test_list_graphs_and_meta(self, memory_neo4j):
        await store.save_graph(_nav("b-graph"), score=0.4)
        await store.save_graph(pg.skeleton("a-graph", description="x"))
        graphs = await store.list_graphs()
        assert [g["name"] for g in graphs] == ["a-graph", "b-graph"]
        assert set(graphs[0]) == {"name", "version", "score", "nodes", "edges", "updated_at", "description"}
        assert graphs[0]["score"] is None and graphs[1]["score"] == 0.4
        assert (graphs[1]["nodes"], graphs[1]["edges"]) == (4, 3)
        meta = await store.get_meta("b-graph")
        assert meta["version"] == 1 and meta["cycle_policy"] == "forbid"
        assert meta["created_at"] and meta["updated_at"]


# ── 5. Versions & rollback ───────────────────────────
class TestVersions:
    async def test_versions_count_up_from_one_newest_first(self, memory_neo4j):
        assert await store.save_graph(pg.skeleton("nav"), note="seed") == 1
        assert await store.save_graph(_nav(), score=0.6, note="evolution round 1") == 2
        versions = await store.list_versions("nav")
        assert [v["version"] for v in versions] == [2, 1]
        assert set(versions[0]) == {"version", "score", "accepted", "created_at", "note", "diff"}
        assert versions[0]["score"] == 0.6 and versions[1]["score"] is None
        assert versions[0]["note"] == "evolution round 1"
        assert all(v["accepted"] is True for v in versions)

    async def test_get_version_returns_the_snapshot(self, memory_neo4j):
        await store.save_graph(pg.skeleton("nav"))
        await store.save_graph(_nav())
        assert await store.get_version("nav", 1) == pg.skeleton("nav")
        assert await store.get_version("nav", 2) == _nav()

    async def test_missing_version(self, memory_neo4j):
        await store.save_graph(pg.skeleton("nav"))
        with pytest.raises(store.ProceduralGraphNotFound) as err:
            await store.get_version("nav", 9)
        assert err.value.name == "nav" and err.value.version == 9
        assert isinstance(err.value, LookupError)

    async def test_rollback_appends_a_new_version_with_the_old_score(self, memory_neo4j):
        await store.save_graph(pg.skeleton("nav"), score=0.3)
        await store.save_graph(_nav(), score=0.1)
        assert await store.rollback("nav", 1) == 3
        assert await store.load_graph("nav") == pg.skeleton("nav")
        latest = (await store.list_versions("nav"))[0]
        assert latest["note"] == "rollback to v1"
        assert latest["score"] == 0.3
        assert (await store.get_meta("nav"))["score"] == 0.3
        # History is append-only: the rolled-back-from version is still there.
        assert await store.get_version("nav", 2) == _nav()

    async def test_rollback_to_a_missing_version_writes_nothing(self, memory_neo4j):
        db, calls = memory_neo4j
        await store.save_graph(pg.skeleton("nav"))
        with pytest.raises(store.ProceduralGraphNotFound):
            await store.rollback("nav", 5)
        assert len(calls.batches) == 1  # only the original save


# ── 6. Delete, rejections, trajectories ──────────────
class TestDeleteAndRecords:
    async def test_delete_removes_all_five_labels_in_one_batch(self, memory_neo4j):
        db, calls = memory_neo4j
        await store.save_graph(_nav())
        await store.record_rejection("nav", round=1, reason="score", score=0.1)
        await store.record_trajectory("nav", version=1, query="q", steps=[], score=1.0, source="api")
        await store.save_graph(_nav("other"))

        assert await store.delete_graph("nav") is True

        batch = calls.batches[-1]
        assert [q for q, _ in batch] == list(store.DELETE_GRAPH_QUERIES)
        assert all(p == {"name": "nav"} for _, p in batch)
        text = "\n".join(store.DELETE_GRAPH_QUERIES)
        for label in graph_schema.PROCEDURAL_LABELS:
            assert f":{label} " in text or f":{label})" in text or f":{label} {{" in text
        assert await store.load_graph("nav") is None
        assert await store.list_versions("nav") == []
        assert await store.list_rejections("nav") == []
        assert await store.list_trajectories("nav") == []
        assert await store.load_graph("other") == _nav("other")  # other graphs untouched

    async def test_delete_of_an_absent_graph_reports_false(self, memory_neo4j):
        assert await store.delete_graph("ghost") is False

    async def test_rejections_round_trip_newest_first(self, memory_neo4j):
        db, calls = memory_neo4j
        await store.record_rejection(
            "nav", round=1, reason="structural", diagnostics=["missing the 'Start' node"],
            edits={"delete_nodes": ["Start"]},
        )
        await store.record_rejection("nav", round=2, reason="score", score=0.2, edits={"add_nodes": []})
        assert len(calls.batches) == 2
        rejections = await store.list_rejections("nav")
        assert [r["round"] for r in rejections] == [2, 1]
        assert set(rejections[0]) == {"round", "reason", "score", "diagnostics", "edits", "created_at"}
        assert rejections[1]["diagnostics"] == ["missing the 'Start' node"]
        assert rejections[1]["edits"] == {"delete_nodes": ["Start"]}
        assert rejections[1]["score"] is None and rejections[0]["score"] == 0.2
        assert await store.list_rejections("nav", limit=1) == rejections[:1]

    async def test_rejection_does_not_require_a_stored_graph(self, memory_neo4j):
        """A scratch evolution run on a new name has nothing stored yet."""
        await store.record_rejection("brand-new", round=1, reason="structural", diagnostics=["x"])
        assert len(await store.list_rejections("brand-new")) == 1

    async def test_trajectories_round_trip(self, memory_neo4j):
        steps = [{"thought": "t", "action": "search_entities", "args": {"query": "Ada"},
                  "observation": "Ada Lovelace (PERSON)"}]
        await store.record_trajectory("nav", version=2, query="Who?", steps=steps, score=None,
                                      source="api")
        await store.record_trajectory("nav", version=2, query="When?", steps=[], score=0.5,
                                      source="evolution")
        [latest, first] = await store.list_trajectories("nav")
        assert latest["query"] == "When?" and latest["source"] == "evolution"
        assert first["steps"] == steps and first["score"] is None and first["version"] == 2
        assert set(first) == {"version", "query", "steps", "score", "source", "created_at"}

    async def test_corrupt_json_properties_degrade_to_defaults(self, fake_neo4j):
        fake_neo4j(lambda q, p: [{"round": 1, "reason": "x", "diagnostics_json": "{not json",
                                   "edits_json": None, "created_at": "t"}])
        [rejection] = await store.list_rejections("nav")
        assert rejection["diagnostics"] == [] and rejection["edits"] is None


# ── 7. Priors & startup seeding ──────────────────────
class TestPriors:
    def test_the_navigator_prior_ships(self):
        assert "graphrag-navigator" in store.list_priors()
        prior = store.load_prior("graphrag-navigator")
        assert prior.name == "graphrag-navigator"
        assert pg.validate(prior) == []

    def test_the_mcp_host_prior_ships(self):
        """The graph MCP hosts are steered by: its ACTION ids are the synapse_* tools."""
        assert "mcp-host" in store.list_priors()
        prior = store.load_prior("mcp-host")
        assert prior.name == "mcp-host"
        assert all(tool.startswith("synapse_") for tool in prior.tools)

    def test_every_bundled_prior_is_valid_with_no_warnings(self):
        """A prior is seeded as v1 of a live graph, so each must be admissible and clean."""
        names = store.list_priors()
        assert {"graphrag-navigator", "mcp-host"} <= set(names)
        for name in names:
            prior = store.load_prior(name)
            assert prior.name == name, name  # the file name is the graph name
            assert pg.validate(prior) == [], name
            assert pg.warnings(prior) == [], name

    def test_the_prior_lives_inside_the_app_package(self):
        """backend/Dockerfile copies backend/ wholesale, so app/data ships in the image."""
        assert store.PRIORS_DIR.parts[-3:] == ("app", "data", "procedural")
        assert (store.PRIORS_DIR / "graphrag-navigator.json").is_file()

    @pytest.mark.parametrize("name", ["../../etc/passwd", "nope", "", "a/b"])
    def test_unknown_or_unsafe_names(self, name):
        with pytest.raises(store.ProceduralGraphNotFound):
            store.load_prior(name)

    def test_errors_are_reexported(self):
        assert store.InvalidProceduralGraph is pg.InvalidProceduralGraph


class TestEnsureDefaultGraphs:
    async def test_seeds_every_absent_prior_once(self, memory_neo4j):
        # Every file in the priors directory, not only the navigator: MCP hosts
        # default to mcp-host, so it must exist on a fresh database too.
        assert await store.ensure_default_graphs() == store.list_priors()
        assert {"graphrag-navigator", "mcp-host"} <= set(store.list_priors())
        assert await store.ensure_default_graphs() == []
        for name in ("graphrag-navigator", "mcp-host"):
            meta = await store.get_meta(name)
            assert meta["version"] == 1 and meta["score"] is None
            [version] = await store.list_versions(name)
            assert version["note"] == "seeded from the expert prior"
            assert await store.load_graph(name) == store.load_prior(name)

    async def test_never_overwrites_an_existing_graph(self, memory_neo4j):
        evolved = store.load_prior("graphrag-navigator")
        evolved.description = "evolved"
        await store.save_graph(evolved, score=0.9)
        # Only the priors still absent are seeded.
        absent = [name for name in store.list_priors() if name != "graphrag-navigator"]
        assert await store.ensure_default_graphs() == absent
        assert await store.ensure_default_graphs() == []
        assert (await store.load_graph("graphrag-navigator")).description == "evolved"

    async def test_a_database_failure_is_logged_not_raised(self, fake_neo4j, caplog):
        def down(q, p):
            raise ConnectionError("neo4j unreachable")

        fake_neo4j(down)
        with caplog.at_level(logging.WARNING):
            assert await store.ensure_default_graphs() == []
        assert "neo4j unreachable" in caplog.text


class TestLifespanSeeding:
    @pytest.fixture
    def lifespan_calls(self, monkeypatch):
        from app import main

        seen: list[str] = []

        async def fake_get_driver():
            return object()

        async def fake_schema():
            seen.append("schema")

        async def fake_seed():
            seen.append("seed")
            return []

        async def fake_close():
            seen.append("close")

        monkeypatch.setattr(main, "get_driver", fake_get_driver)
        monkeypatch.setattr(main, "ensure_schema", fake_schema)
        monkeypatch.setattr(main, "ensure_default_graphs", fake_seed)
        monkeypatch.setattr(main, "close_driver", fake_close)
        return main, seen

    async def test_seeds_after_the_schema_when_enabled(self, lifespan_calls):
        main, seen = lifespan_calls
        async with main.lifespan(main.app):
            seen.append("serving")
        assert seen == ["schema", "seed", "serving", "close"]

    async def test_skipped_when_procedural_memory_is_disabled(self, lifespan_calls, monkeypatch):
        main, seen = lifespan_calls
        monkeypatch.setenv("PROCEDURAL_ENABLED", "false")
        async with main.lifespan(main.app):
            seen.append("serving")
        assert seen == ["schema", "serving", "close"]
