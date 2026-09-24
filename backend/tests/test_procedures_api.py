# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""API tests for ``/api/procedures/*`` and ``/api/agent/ask``: the HTTP contract
the client package, the MCP server, the CLI and the UI code against.

The TestClient is created without the context-manager form, so the lifespan
(Neo4j + schema + seeding) never runs. Most tests swap the procedural store for
an in-memory ``FakeStore``; a few go through the REAL store over ``fake_neo4j``
to prove the wiring. The agent and the evolution loop are replaced by fakes:
no test can reach a database or an LLM provider.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routers import procedures as procedures_router
from app.services import procedural_guidance as pgd
from app.services import procedural_store as store
from app.services.graph_agent import AgentResult
from app.services.llm_provider import ProviderConfigError
from app.services.procedural_graph import ProceduralGraph, serialize_full
from app.services.procedural_store import ProceduralGraphNotFound, load_prior


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolation():
    pgd.clear_caches()
    procedures_router._running_evolutions.clear()
    yield
    pgd.clear_caches()
    procedures_router._running_evolutions.clear()


def sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")
    ]


class FakeStore:
    """An in-memory procedural store with the real module's function signatures."""

    def __init__(self) -> None:
        self.graphs: dict[str, dict] = {}
        self.trajectories: list[dict] = []
        self.rejections: list[dict] = [
            {
                "round": 1,
                "reason": "structural",
                "score": None,
                "diagnostics": ["x"],
                "edits": {},
                "created_at": "t",
            }
        ]
        self.rejection_limits: list[int] = []

    def add(self, graph: ProceduralGraph, version: int = 1, score: float | None = 0.5) -> None:
        self.graphs[graph.name] = {
            "graph": graph,
            "version": version,
            "score": score,
            "history": [(version, graph)],
        }

    def install(self, monkeypatch) -> FakeStore:
        async def list_graphs():
            return [
                {
                    "name": n,
                    "version": g["version"],
                    "score": g["score"],
                    "nodes": len(g["graph"].nodes),
                    "edges": len(g["graph"].edges),
                    "updated_at": "t",
                    "description": g["graph"].description,
                }
                for n, g in sorted(self.graphs.items())
            ]

        async def load_graph_with_meta(name):
            g = self.graphs.get(name)
            return (g["graph"], {"version": g["version"], "score": g["score"]}) if g else None

        async def get_meta(name):
            g = self.graphs.get(name)
            return {"name": name, "version": g["version"], "score": g["score"]} if g else None

        async def save_graph(graph, *, score=None, note="", edits=None, previous=None):
            from app.services.procedural_graph import InvalidProceduralGraph, validate

            problems = validate(graph)
            if problems:
                raise InvalidProceduralGraph(problems)
            g = self.graphs.get(graph.name)
            version = (g["version"] if g else 0) + 1
            history = (g["history"] if g else []) + [(version, graph)]
            self.graphs[graph.name] = {
                "graph": graph,
                "version": version,
                "score": score,
                "history": history,
                "note": note,
            }
            return version

        async def delete_graph(name):
            return self.graphs.pop(name, None) is not None

        async def record_trajectory(name, *, version, query, steps, score, source):
            self.trajectories.append(
                {
                    "name": name,
                    "version": version,
                    "query": query,
                    "steps": steps,
                    "score": score,
                    "source": source,
                }
            )

        async def list_versions(name, limit=200):
            g = self.graphs.get(name)
            if not g:
                return []
            return [
                {
                    "version": v,
                    "score": None,
                    "accepted": True,
                    "created_at": "t",
                    "note": "",
                    "diff": {},
                }
                for v, _ in reversed(g["history"])
            ]

        async def rollback(name, version):
            g = self.graphs.get(name)
            snapshot = dict(g["history"]).get(version) if g else None
            if snapshot is None:
                raise ProceduralGraphNotFound(name, version)
            return await save_graph(snapshot, note=f"rollback to v{version}")

        async def list_rejections(name, limit=20):
            self.rejection_limits.append(limit)
            return self.rejections[:limit]

        for fn in (
            list_graphs,
            load_graph_with_meta,
            get_meta,
            save_graph,
            delete_graph,
            record_trajectory,
            list_versions,
            rollback,
            list_rejections,
        ):
            monkeypatch.setattr(store, fn.__name__, fn)
        return self


@pytest.fixture
def fake_store(monkeypatch):
    fs = FakeStore().install(monkeypatch)
    fs.add(load_prior("graphrag-navigator"), version=3, score=0.72)
    return fs


# ── Graph CRUD ───────────────────────────────────────
class TestGraphCrud:
    def test_list_goes_through_the_real_store(self, client, fake_neo4j):
        row = {
            "name": "graphrag-navigator",
            "version": 2,
            "score": None,
            "nodes": 11,
            "edges": 14,
            "updated_at": "2026-09-24T00:00:00+00:00",
            "description": "d",
        }
        calls = fake_neo4j(lambda q, p: [row] if q == store.LIST_GRAPHS_QUERY else [])
        r = client.get("/api/procedures")
        assert r.status_code == 200
        assert r.json() == {"graphs": [row]}
        assert [q for q, _ in calls] == [store.LIST_GRAPHS_QUERY]

    def test_get_returns_the_editable_json_with_version_and_score(self, client, fake_store):
        r = client.get("/api/procedures/graphrag-navigator")
        assert r.status_code == 200
        body = r.json()
        prior = load_prior("graphrag-navigator")
        assert body == {**prior.to_dict(), "version": 3, "score": 0.72}

    def test_get_as_text(self, client, fake_store):
        r = client.get("/api/procedures/graphrag-navigator", params={"format": "text"})
        assert r.json() == {"text": serialize_full(load_prior("graphrag-navigator"))}

    def test_unknown_graph_is_404(self, client, fake_store):
        for path in ("/api/procedures/ghost", "/api/procedures/ghost/graph-data"):
            r = client.get(path)
            assert r.status_code == 404
            assert "ghost" in r.json()["detail"]

    def test_graph_data_shape(self, client, fake_store):
        body = client.get("/api/procedures/graphrag-navigator/graph-data").json()
        prior = load_prior("graphrag-navigator")
        assert body["version"] == 3 and body["score"] == 0.72
        assert len(body["nodes"]) == len(prior.nodes) and len(body["links"]) == len(prior.edges)
        start = next(n for n in body["nodes"] if n["id"] == "Start")
        end = next(n for n in body["nodes"] if n["id"] == "End")
        assert start == {
            **start,
            "label": "Start",
            "type": "STATUS",
            "is_start": True,
            "is_terminal": False,
        }
        assert end["is_terminal"] is True and end["is_start"] is False
        assert set(body["links"][0]) == {
            "source",
            "target",
            "relation",
            "condition",
            "guidance",
            "pitfalls",
        }

    def test_put_saves_a_new_version(self, client, fake_store):
        graph = load_prior("graphrag-navigator").to_dict()
        graph["description"] = "edited"
        r = client.put("/api/procedures/graphrag-navigator", json=graph)
        assert r.status_code == 200
        assert r.json() == {"name": "graphrag-navigator", "version": 4}
        assert fake_store.graphs["graphrag-navigator"]["graph"].description == "edited"

    def test_put_accepts_a_body_without_a_name_and_the_get_output_as_is(self, client, fake_store):
        body = client.get("/api/procedures/graphrag-navigator").json()
        body.pop("name")
        r = client.put("/api/procedures/copy-of-nav", json=body)
        assert r.json() == {"name": "copy-of-nav", "version": 1}

    def test_put_with_a_mismatched_name_is_422(self, client, fake_store):
        graph = load_prior("graphrag-navigator").to_dict()
        r = client.put("/api/procedures/other", json=graph)
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["message"] == "Graph name in the body does not match the path"
        assert detail["diagnostics"]

    def test_put_invalid_graph_lists_every_diagnostic(self, client, fake_store):
        graph = {
            "nodes": [{"id": "A", "type": "ACTION"}, {"id": "B", "type": "WIZARD"}],
            "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "A"}],
        }
        r = client.put("/api/procedures/broken", json=graph)
        assert r.status_code == 422
        diagnostics = r.json()["detail"]["diagnostics"]
        assert any("Start" in d for d in diagnostics)
        assert any("WIZARD" in d for d in diagnostics)
        assert any("cycle" in d for d in diagnostics)
        assert "broken" not in fake_store.graphs

    def test_put_malformed_json_shape_is_422(self, client, fake_store):
        r = client.put("/api/procedures/bad", json={"nodes": "not a list"})
        assert r.status_code == 422
        assert r.json()["detail"]["message"] == "Malformed procedural graph JSON"

    def test_put_invalid_goes_nowhere_near_the_database(self, client, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        r = client.put("/api/procedures/broken", json={"nodes": [], "edges": []})
        assert r.status_code == 422
        assert calls == []

    def test_put_valid_is_one_write_transaction_in_the_real_store(self, client, fake_neo4j):
        def handler(query, params):
            if query in (store.UPSERT_GRAPH_QUERY, store.WRITE_VERSION_QUERY):
                return [{"version": 1}]
            return []

        calls = fake_neo4j(handler)
        graph = load_prior("graphrag-navigator").to_dict()
        r = client.put("/api/procedures/graphrag-navigator", json=graph)
        assert r.json() == {"name": "graphrag-navigator", "version": 1}
        assert len(calls.batches) == 1 and len(calls.batches[0]) == 5

    def test_delete(self, client, fake_store):
        r = client.delete("/api/procedures/graphrag-navigator")
        assert r.json() == {"status": "success", "deleted": True}
        assert client.delete("/api/procedures/graphrag-navigator").json() == {
            "status": "success",
            "deleted": False,
        }

    def test_delete_is_one_batch_over_all_five_labels_in_the_real_store(self, client, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [{"deleted": 1}] if "count(g)" in q else [])
        assert client.delete("/api/procedures/nav").json()["deleted"] is True
        assert len(calls.batches) == 1
        assert [q for q, _ in calls.batches[0]] == list(store.DELETE_GRAPH_QUERIES)


# ── Guidance + trajectories ──────────────────────────
class TestGuidanceEndpoint:
    def test_raw_guidance_after_an_action(self, client, fake_store):
        r = client.post(
            "/api/procedures/graphrag-navigator/guidance",
            json={
                "query": "Who wrote the first program?",
                "trajectory": [{"action": "search_entities", "observation": "1. Ada"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["localization"] == "exact"
        assert body["active_node"] == "search_entities"
        assert body["scope"] == "local"
        assert body["guidance"] is None
        assert body["version"] == 3
        assert body["context"].startswith("Active Cognitive Node: [search_entities]")
        assert set(body["next_actions"]) == {"neighbors", "read_sources", "search_passages"}
        assert body["usage"]["llm_calls"] == 0

    def test_a_navigator_trace_with_null_fields_replays(self, client, fake_store):
        # /api/agent/ask records a failed parse as action null; replaying the
        # trace over HTTP must work, and that step falls back to the full graph.
        r = client.post(
            "/api/procedures/graphrag-navigator/guidance",
            json={
                "query": "Q?",
                "trajectory": [
                    {"action": "search_entities", "observation": None},
                    {"thought": "hm", "action": None, "args": {}, "observation": "Invalid"},
                ],
            },
        )
        assert r.status_code == 200
        assert (r.json()["localization"], r.json()["scope"]) == ("none", "full")

    def test_empty_trajectory_localizes_on_start(self, client, fake_store):
        body = client.post(
            "/api/procedures/graphrag-navigator/guidance", json={"query": "Q?"}
        ).json()
        assert (body["active_node"], body["localization"]) == ("Start", "start")

    def test_hops_and_mode_are_passed_through(self, client, fake_store):
        body = client.post(
            "/api/procedures/graphrag-navigator/guidance",
            json={"query": "Q?", "mode": "none", "hops": 1},
        ).json()
        assert body["context"] == "" and body["guidance"] is None

    def test_unknown_graph_is_404(self, client, fake_store):
        r = client.post("/api/procedures/ghost/guidance", json={"query": "Q?"})
        assert r.status_code == 404

    def test_generative_without_a_provider_is_503(self, client, fake_store, monkeypatch):
        def no_key(**kwargs):
            raise ProviderConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

        monkeypatch.setattr(pgd, "get_chat_llm", no_key)
        r = client.post(
            "/api/procedures/graphrag-navigator/guidance",
            json={"query": "Q?", "mode": "generative"},
        )
        assert r.status_code == 503
        assert "OPENAI_API_KEY is empty" in r.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {"query": "Q?", "hops": 9},
            {"query": "Q?", "window": -1},
            {"query": "Q?", "mode": "loud"},
        ],
    )
    def test_invalid_input_is_422(self, client, fake_store, payload):
        assert (
            client.post("/api/procedures/graphrag-navigator/guidance", json=payload).status_code
            == 422
        )


class TestTrajectories:
    def test_record_against_the_current_version(self, client, fake_store):
        steps = [{"action": "search_entities", "observation": "x"}]
        r = client.post(
            "/api/procedures/graphrag-navigator/trajectories",
            json={"query": "Q?", "steps": steps, "score": 0.5},
        )
        assert r.json() == {"status": "recorded"}
        assert fake_store.trajectories == [
            {
                "name": "graphrag-navigator",
                "version": 3,
                "query": "Q?",
                "steps": steps,
                "score": 0.5,
                "source": "api",
            }
        ]

    def test_unknown_graph_is_404(self, client, fake_store):
        r = client.post(
            "/api/procedures/ghost/trajectories", json={"query": "Q?", "steps": [], "score": 1}
        )
        assert r.status_code == 404
        assert fake_store.trajectories == []

    @pytest.mark.parametrize("score", [-0.1, 1.5, None])
    def test_score_must_be_in_0_1(self, client, fake_store, score):
        r = client.post(
            "/api/procedures/graphrag-navigator/trajectories",
            json={"query": "Q?", "steps": [], "score": score},
        )
        assert r.status_code == 422


# ── Versions, rollback, rejections ───────────────────
class TestVersions:
    def test_versions_newest_first(self, client, fake_store):
        client.put(
            "/api/procedures/graphrag-navigator", json=load_prior("graphrag-navigator").to_dict()
        )
        versions = client.get("/api/procedures/graphrag-navigator/versions").json()["versions"]
        assert [v["version"] for v in versions] == [4, 3]
        assert set(versions[0]) == {"version", "score", "accepted", "created_at", "note", "diff"}

    def test_versions_of_an_unknown_graph_is_404(self, client, fake_store):
        assert client.get("/api/procedures/ghost/versions").status_code == 404

    def test_rollback(self, client, fake_store):
        r = client.post("/api/procedures/graphrag-navigator/rollback", json={"version": 3})
        assert r.json() == {"name": "graphrag-navigator", "version": 4}

    def test_rollback_to_a_missing_version_is_404(self, client, fake_store):
        r = client.post("/api/procedures/graphrag-navigator/rollback", json={"version": 99})
        assert r.status_code == 404
        assert "no version 99" in r.json()["detail"]

    def test_rollback_needs_a_positive_version(self, client, fake_store):
        assert (
            client.post(
                "/api/procedures/graphrag-navigator/rollback", json={"version": 0}
            ).status_code
            == 422
        )

    def test_rejections_of_an_unknown_graph_is_404(self, client, fake_store):
        fake_store.rejections = []
        assert client.get("/api/procedures/ghost/rejections").status_code == 404

    def test_rejections_of_a_never_saved_scratch_graph_are_listed(self, client, fake_store):
        # A scratch run that accepted nothing leaves rejections and no graph.
        r = client.get("/api/procedures/scratchy/rejections")
        assert r.status_code == 200 and r.json() == {"rejections": fake_store.rejections}

    def test_rejections(self, client, fake_store):
        r = client.get("/api/procedures/graphrag-navigator/rejections", params={"limit": 5})
        assert r.json() == {"rejections": fake_store.rejections}
        assert fake_store.rejection_limits == [5]
        assert (
            client.get(
                "/api/procedures/graphrag-navigator/rejections", params={"limit": 0}
            ).status_code
            == 422
        )


# ── Evolution jobs ───────────────────────────────────
QA = [{"question": "Who?", "answer": "Ada"}]
REPORT = {
    "graph": "graphrag-navigator",
    "stopped": "completed",
    "baseline_score": 0.5,
    "final_score": 0.6,
}


@pytest.fixture
def fake_evolve(monkeypatch):
    """Replace the loop with a recorder; let the provider check pass."""
    calls: list[dict] = []

    async def evolve(name, *, on_progress=None, **params):
        calls.append({"name": name, **params})
        await on_progress(
            {"type": "progress", "stage": "baseline", "round": 0, "processed": 1, "total": 1}
        )
        await on_progress(
            {"type": "progress", "stage": "accepted", "round": 1, "score": 0.6, "version": 4}
        )
        return REPORT

    monkeypatch.setattr(procedures_router, "evolve", evolve)
    monkeypatch.setattr(procedures_router, "get_chat_llm", lambda **kw: object())
    return calls


class TestEvolveEndpoint:
    def test_starts_a_job_and_streams_progress_then_the_report(
        self, client, fake_store, fake_evolve
    ):
        r = client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "processing" and body["job_id"].startswith("job_")

        stream = client.get(f"/api/procedures/evolve/{body['job_id']}/events")
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert stream.text.startswith("data: ")
        assert stream.text.endswith("\n\n")
        events = sse_events(stream.text)
        assert [e["type"] for e in events] == ["progress", "progress", "done"]
        assert events[-1]["data"] == REPORT

    def test_defaults_come_from_settings(self, client, fake_store, fake_evolve):
        client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA})
        settings = get_settings()
        (call,) = fake_evolve
        assert call == {
            "name": "graphrag-navigator",
            "train": QA,
            "val": QA,
            "rounds": settings.evolution_default_rounds,
            "batch_size": settings.evolution_default_batch_size,
            "mode": "static",
            "metric": "f1",
            "guidance": "raw",
            "max_llm_calls": settings.evolution_max_llm_calls,
            "replace": False,
        }

    def test_explicit_parameters_are_passed_through(self, client, fake_store, fake_evolve):
        payload = {
            "train": QA,
            "val": [{"question": "Q", "answer": ["a", "b"]}],
            "rounds": 2,
            "batch_size": 5,
            "mode": "scratch",
            "metric": "em",
            "guidance": "generative",
            "max_llm_calls": 50,
        }
        client.post("/api/procedures/new-graph/evolve", json=payload)
        (call,) = fake_evolve
        assert {
            k: call[k]
            for k in ("rounds", "batch_size", "mode", "metric", "guidance", "max_llm_calls")
        } == {
            "rounds": 2,
            "batch_size": 5,
            "mode": "scratch",
            "metric": "em",
            "guidance": "generative",
            "max_llm_calls": 50,
        }
        assert call["val"] == [{"question": "Q", "answer": ["a", "b"]}]

    def test_blank_alternative_golds_are_dropped(self, client, fake_store, fake_evolve):
        val = [{"question": "Q", "answer": [" Ada ", "", "Lovelace"]}]
        client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": val})
        (call,) = fake_evolve
        assert call["val"] == [{"question": "Q", "answer": ["Ada", "Lovelace"]}]

    def test_a_failure_streams_an_error_event(self, client, fake_store, monkeypatch):
        async def failing(name, **kwargs):
            raise RuntimeError("refiner exploded")

        monkeypatch.setattr(procedures_router, "evolve", failing)
        monkeypatch.setattr(procedures_router, "get_chat_llm", lambda **kw: object())
        job_id = client.post(
            "/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA}
        ).json()["job_id"]
        events = sse_events(client.get(f"/api/procedures/evolve/{job_id}/events").text)
        assert events == [{"type": "error", "data": "refiner exploded"}]
        assert "graphrag-navigator" not in procedures_router._running_evolutions

    @pytest.mark.parametrize(
        "payload",
        [
            {"train": [], "val": QA},
            {"train": QA, "val": []},
            {"train": QA, "val": QA, "rounds": 0},
            {"train": QA, "val": QA, "rounds": 21},
            {"train": QA, "val": QA, "batch_size": 0},
            {"train": QA, "val": QA, "batch_size": 101},
            {"train": QA, "val": QA, "mode": "online"},
            {"train": [{"question": "", "answer": "x"}], "val": QA},
            # A blank gold can only ever score 0: refused before anything is spent.
            {"train": [{"question": "q1", "answer": []}], "val": QA},
            {"train": QA, "val": [{"question": "q2", "answer": ""}]},
            {"train": QA, "val": [{"question": "q2", "answer": "   "}]},
            {"train": [{"question": "q1", "answer": ["", " "]}], "val": QA},
        ],
    )
    def test_invalid_requests_are_422(self, client, fake_store, fake_evolve, payload):
        assert (
            client.post("/api/procedures/graphrag-navigator/evolve", json=payload).status_code
            == 422
        )
        assert fake_evolve == []

    def test_static_mode_on_an_unknown_graph_is_404(self, client, fake_store, fake_evolve):
        r = client.post("/api/procedures/ghost/evolve", json={"train": QA, "val": QA})
        assert r.status_code == 404
        assert fake_evolve == []

    def test_scratch_mode_needs_a_valid_name(self, client, fake_store, fake_evolve):
        r = client.post(
            "/api/procedures/bad name!/evolve", json={"train": QA, "val": QA, "mode": "scratch"}
        )
        assert r.status_code == 422

    def test_scratch_on_an_existing_graph_is_409_without_replace(
        self, client, fake_store, fake_evolve
    ):
        r = client.post(
            "/api/procedures/graphrag-navigator/evolve",
            json={"train": QA, "val": QA, "mode": "scratch"},
        )
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "'graphrag-navigator' already exists (v3)" in detail and "replace=true" in detail
        assert fake_evolve == []
        # Refused ahead of the budget check: replace changes the minimum.
        r = client.post(
            "/api/procedures/graphrag-navigator/evolve",
            json={"train": QA, "val": QA, "mode": "scratch", "max_llm_calls": 1},
        )
        assert r.status_code == 409

    def test_scratch_with_replace_starts_and_passes_replace_on(
        self, client, fake_store, fake_evolve
    ):
        r = client.post(
            "/api/procedures/graphrag-navigator/evolve",
            json={"train": QA, "val": QA, "mode": "scratch", "replace": True},
        )
        assert r.status_code == 200
        (call,) = fake_evolve
        assert (call["mode"], call["replace"]) == ("scratch", True)

    @pytest.mark.parametrize(
        ("payload", "rollouts", "per_rollout"),
        [
            # (|val| baseline + |B_1| train + |val| validation) × worst case + 1 refiner.
            ({}, 1 + 1 + 1, "raw"),
            ({"guidance": "generative"}, 1 + 1 + 1, "generative"),
            # The stored graph a scratch run may replace is scored once more on val.
            ({"mode": "scratch", "replace": True}, 1 + 1 + 1 + 1, "raw"),
        ],
    )
    def test_a_cap_below_one_round_is_422_with_the_minimum(
        self, client, fake_store, fake_evolve, payload, rollouts, per_rollout
    ):
        steps = get_settings().agent_max_steps
        minimum = rollouts * steps * (2 if per_rollout == "generative" else 1) + 1
        url = "/api/procedures/graphrag-navigator/evolve"
        body = {"train": QA, "val": QA, **payload}
        r = client.post(url, json={**body, "max_llm_calls": minimum - 1})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["minimum_max_llm_calls"] == minimum
        assert detail["message"] == "max_llm_calls cannot pay for the baseline and one full round"
        assert f"set max_llm_calls to at least {minimum}" in detail["diagnostics"][0]
        assert fake_evolve == []
        assert client.post(url, json={**body, "max_llm_calls": minimum}).status_code == 200
        assert fake_evolve[0]["max_llm_calls"] == minimum

    def test_the_default_cap_is_checked_too(self, client, fake_store, fake_evolve, monkeypatch):
        monkeypatch.setattr(get_settings(), "evolution_max_llm_calls", 10)
        r = client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA})
        assert r.status_code == 422
        assert r.json()["detail"]["minimum_max_llm_calls"] == 3 * get_settings().agent_max_steps + 1
        assert fake_evolve == []

    def test_no_provider_is_503_before_anything_is_spent(
        self, client, fake_store, fake_evolve, monkeypatch
    ):
        def no_key(**kwargs):
            raise ProviderConfigError("LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty.")

        monkeypatch.setattr(procedures_router, "get_chat_llm", no_key)
        r = client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA})
        assert r.status_code == 503
        assert "GOOGLE_API_KEY" in r.json()["detail"]
        assert fake_evolve == []

    def test_one_run_per_graph_at_a_time(self, client, fake_store, fake_evolve):
        procedures_router._running_evolutions.add("graphrag-navigator")
        r = client.post("/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA})
        assert r.status_code == 409

    def test_the_lock_is_released_when_the_job_ends(self, client, fake_store, fake_evolve):
        job_id = client.post(
            "/api/procedures/graphrag-navigator/evolve", json={"train": QA, "val": QA}
        ).json()["job_id"]
        client.get(f"/api/procedures/evolve/{job_id}/events")
        assert procedures_router._running_evolutions == set()

    def test_unknown_job(self, client):
        events = sse_events(client.get("/api/procedures/evolve/job_999999/events").text)
        assert events == [{"type": "error", "data": "Unknown job id."}]


# ── The Navigator ────────────────────────────────────
def _result(graph):
    return AgentResult(
        question="Who?",
        answer="Ada",
        steps=[
            {
                "thought": "t",
                "action": "answer",
                "args": {"text": "Ada"},
                "observation": "",
                "guidance_context_chars": 10,
                "localization": "start",
                "active_node": "Start",
            }
        ],
        stopped="answer",
        parse_failures=0,
        usage={
            "llm_calls": 1,
            "guidance_llm_calls": 0,
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated": False,
            "context_chars": 10,
        },
        graph=graph,
        latency_s=0.1,
    )


@pytest.fixture
def fake_agent(monkeypatch):
    calls: list[dict] = []

    async def run_agent(question, **kwargs):
        calls.append({"question": question, **kwargs})
        graph = {"name": kwargs["graph_name"], "version": 3} if kwargs["graph_name"] else None
        return _result(graph)

    monkeypatch.setattr(procedures_router, "run_agent", run_agent)
    return calls


class TestAgentAsk:
    def test_defaults_to_the_default_graph(self, client, fake_store, fake_agent):
        r = client.post("/api/agent/ask", json={"query": "Who?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Ada" and body["stopped"] == "answer"
        assert body["graph"] == {"name": "graphrag-navigator", "version": 3}
        assert body["recorded"] is False
        assert fake_agent == [
            {
                "question": "Who?",
                "graph_name": get_settings().procedural_default_graph,
                "guidance": None,
                "max_steps": None,
            }
        ]
        assert set(body) >= {
            "answer",
            "steps",
            "stopped",
            "parse_failures",
            "usage",
            "graph",
            "latency_s",
        }

    def test_explicit_null_graph_runs_without_procedural_memory(
        self, client, fake_store, fake_agent
    ):
        body = client.post("/api/agent/ask", json={"query": "Who?", "graph": None}).json()
        assert fake_agent[0]["graph_name"] is None
        assert body["graph"] is None

    def test_default_graph_is_off_when_procedural_memory_is_disabled(
        self, client, fake_store, fake_agent, monkeypatch
    ):
        monkeypatch.setattr(get_settings(), "procedural_enabled", False)
        client.post("/api/agent/ask", json={"query": "Who?"})
        assert fake_agent[0]["graph_name"] is None

    def test_options_are_passed_through(self, client, fake_store, fake_agent):
        client.post(
            "/api/agent/ask",
            json={"query": "Who?", "graph": "custom", "guidance": "generative", "max_steps": 4},
        )
        assert fake_agent[0] == {
            "question": "Who?",
            "graph_name": "custom",
            "guidance": "generative",
            "max_steps": 4,
        }

    def test_record_stores_the_trajectory_with_a_null_score(self, client, fake_store, fake_agent):
        body = client.post("/api/agent/ask", json={"query": "Who?", "record": True}).json()
        assert body["recorded"] is True
        (trajectory,) = fake_store.trajectories
        assert trajectory["score"] is None and trajectory["source"] == "agent"
        assert trajectory["version"] == 3 and trajectory["steps"] == body["steps"]

    def test_record_without_a_graph_records_nothing(self, client, fake_store, fake_agent):
        body = client.post(
            "/api/agent/ask", json={"query": "Who?", "graph": None, "record": True}
        ).json()
        assert body["recorded"] is False and fake_store.trajectories == []

    def test_no_provider_is_503(self, client, fake_store, monkeypatch):
        async def no_key(question, **kwargs):
            raise ProviderConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

        monkeypatch.setattr(procedures_router, "run_agent", no_key)
        r = client.post("/api/agent/ask", json={"query": "Who?"})
        assert r.status_code == 503
        assert "OPENAI_API_KEY" in r.json()["detail"]

    def test_unknown_graph_is_404(self, client, fake_store, monkeypatch):
        async def missing(question, **kwargs):
            raise ProceduralGraphNotFound(kwargs["graph_name"])

        monkeypatch.setattr(procedures_router, "run_agent", missing)
        assert (
            client.post("/api/agent/ask", json={"query": "Who?", "graph": "ghost"}).status_code
            == 404
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {"query": "Q", "max_steps": 0},
            {"query": "Q", "max_steps": 21},
            {"query": "Q", "guidance": "loud"},
        ],
    )
    def test_invalid_input_is_422(self, client, fake_agent, payload):
        assert client.post("/api/agent/ask", json=payload).status_code == 422
        assert fake_agent == []
