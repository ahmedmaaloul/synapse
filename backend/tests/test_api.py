# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""API-level tests with the FastAPI TestClient (no real DB / LLM).

The client is created WITHOUT the context-manager form so the app lifespan
(Neo4j connect + schema bootstrap) does not run — each endpoint's DB access is
mocked via the ``fake_neo4j`` fixture instead.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app
from app.routers import chat as chat_router
from app.routers import graph as graph_router
from app.routers import upload as upload_router


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def stub_communities(monkeypatch):
    """Replace ``detect_and_summarize`` in both routers with a recording stub.

    Returns ``(events, calls)``: the progress events the stub emitted and the
    number of times it ran. Keeps the ingest path from touching a real DB/LLM.
    """
    calls: list[dict] = []

    def install(result=None, *, raises: Exception | None = None):
        payload = result or {"communities": 2, "summarized": 2, "modularity": 0.42}

        async def fake_detect(on_progress=None):
            calls.append(payload)
            if raises is not None:
                raise raises
            if on_progress is not None:
                await on_progress(
                    {"type": "progress", "stage": "summarizing", "processed": 1, "total": 2}
                )
            return payload

        monkeypatch.setattr(upload_router, "detect_and_summarize", fake_detect)
        monkeypatch.setattr(graph_router, "detect_and_summarize", fake_detect)
        return calls

    return install


def sse_events(text: str) -> list[dict]:
    """Parse the ``data: {json}`` frames out of an SSE response body."""
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _stub_ingest(monkeypatch) -> None:
    """Make ``POST /api/upload`` parse + extract without a PDF, an LLM or a DB."""
    monkeypatch.setattr(
        upload_router, "extract_text_from_pdf", lambda b: ["chunk a", "chunk b"]
    )

    async def fake_build(chunks, filename, theme="Generic", on_progress=None):
        return {
            "nodes_created": 2,
            "relationships_created": 1,
            "entities_extracted": 2,
            "unique_entities": 2,
        }

    monkeypatch.setattr(upload_router, "build_knowledge_graph", fake_build)


def _upload(client) -> dict:
    r = client.post(
        "/api/upload",
        files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"theme": "Personal CV / Resume"},
    )
    assert r.status_code == 200
    return r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_reports_neo4j(client, monkeypatch):
    async def fake_ok():
        return True

    monkeypatch.setattr(main, "verify_connectivity", fake_ok)
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["neo4j"] == "up"
    assert body["status"] == "ready"


def test_about_reports_authorship_and_license(client):
    r = client.get("/api/about")
    assert r.status_code == 200
    body = r.json()
    assert body["author"] == "Ahmed Maaloul"
    assert body["license"] == "AGPL-3.0-or-later"
    assert "Ahmed Maaloul" in r.text
    assert "AGPL-3.0-or-later" in r.text
    # AGPL §13: network users must be pointed at the corresponding source.
    assert body["source_code"] == "https://github.com/ahmedmaaloul/synapse"
    assert body["commercial_license"]["contact"] == "ahmed.maaloul@proton.me"
    assert body["llm_provider"] and body["embedding_provider"]


def test_process_time_header_present(client):
    r = client.get("/health")
    assert "X-Process-Time" in r.headers


class TestUpload:
    def test_rejects_non_pdf(self, client):
        r = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"theme": "Generic"},
        )
        assert r.status_code == 400
        assert "PDF" in r.json()["detail"]

    def test_accepts_pdf_and_returns_job_id(self, client, monkeypatch, stub_communities):
        stub_communities()
        _stub_ingest(monkeypatch)

        r = client.post(
            "/api/upload",
            files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"theme": "Personal CV / Resume"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"].startswith("job_")
        assert body["total_chunks"] == 2
        assert body["status"] == "processing"

    def test_empty_pdf_text_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(upload_router, "extract_text_from_pdf", lambda b: [])
        r = client.post(
            "/api/upload",
            files={"file": ("blank.pdf", b"%PDF-1.4", "application/pdf")},
            data={"theme": "Generic"},
        )
        assert r.status_code == 400

    def test_ingest_rebuilds_communities_and_reports_counts(
        self, client, monkeypatch, stub_communities
    ):
        calls = stub_communities()
        _stub_ingest(monkeypatch)

        job_id = _upload(client)["job_id"]
        events = sse_events(client.get(f"/api/upload/{job_id}/events").text)

        # Community progress is republished under its own type so it cannot
        # rewind the extraction progress bar.
        assert any(e["type"] == "community_progress" for e in events)
        progress_stages = [e.get("stage") for e in events if e["type"] == "progress"]
        assert "summarizing" not in progress_stages

        done = events[-1]
        assert done["type"] == "done"
        assert done["data"]["nodes_created"] == 2
        assert done["data"]["communities"] == 2
        assert done["data"]["communities_summarized"] == 2
        assert done["data"]["modularity"] == 0.42
        assert len(calls) == 1

    def test_ingest_skips_communities_when_disabled(
        self, client, monkeypatch, stub_communities
    ):
        from app.config import get_settings

        calls = stub_communities()
        _stub_ingest(monkeypatch)
        monkeypatch.setenv("COMMUNITY_DETECTION_ENABLED", "false")
        get_settings.cache_clear()

        job_id = _upload(client)["job_id"]
        events = sse_events(client.get(f"/api/upload/{job_id}/events").text)

        assert calls == []
        assert events[-1]["data"]["communities"] == 0
        assert events[-1]["data"]["modularity"] == 0.0

    def test_community_failure_does_not_lose_the_ingest(
        self, client, monkeypatch, stub_communities
    ):
        stub_communities(raises=RuntimeError("louvain exploded"))
        _stub_ingest(monkeypatch)

        job_id = _upload(client)["job_id"]
        events = sse_events(client.get(f"/api/upload/{job_id}/events").text)

        done = events[-1]
        assert done["type"] == "done"
        assert done["data"]["nodes_created"] == 2  # the graph is still reported
        assert done["data"]["communities"] == 0


class TestChat:
    def test_streams_sse_events(self, client, monkeypatch):
        async def fake_gen(question, history):
            yield {"type": "citations", "data": [{"name": "Ada", "type": "PERSON"}]}
            yield {"type": "token", "data": "Hello"}
            yield {"type": "done"}

        monkeypatch.setattr(chat_router, "generate_rag_response", fake_gen)

        r = client.post("/api/chat", json={"query": "who is ada?", "history": []})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = [
            json.loads(line[len("data: "):])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [e["type"] for e in events] == ["citations", "token", "done"]

    def test_rejects_empty_query(self, client):
        r = client.post("/api/chat", json={"query": ""})
        assert r.status_code == 422  # pydantic min_length


# ── A database holding everything the visualization must NOT show ───────────
# Community detection writes (:Community) nodes joined by [:IN_COMMUNITY], and
# source-chunk retrieval writes (:Chunk) nodes joined by [:MENTIONED_IN]. Both
# hang off entities and neither carries a name/type, so leaking them into
# /api/graph-data renders unnamed grey blobs wired to half the graph — and, for
# chunks, ships whole paragraphs and their embedding vectors to the browser.
MIXED_NODES = [
    {
        "id": "e1",
        "labels": ["Entity"],
        "properties": {"name": "Ada", "type": "PERSON", "embedding": [0.1, 0.2]},
    },
    {
        "id": "e2",
        "labels": ["Entity"],
        "properties": {"name": "Analytical Engine", "type": "TOOL"},
    },
    {
        "id": "c1",
        "labels": ["Community"],
        "properties": {
            "title": "Computing pioneers",
            "summary": "Ada and the engine.",
            "embedding": [0.3],
        },
    },
    {
        "id": "k1",
        "labels": ["Chunk"],
        "properties": {
            "text": "Ada Lovelace wrote the first published algorithm.",
            "document": "history.pdf",
            "index": 0,
            "embedding": [0.4],
        },
    },
]

MIXED_RELS = [
    {"source": "e1", "target": "e2", "type": "WORKED_ON", "properties": {}},
    {"source": "e1", "target": "k1", "type": "MENTIONED_IN", "properties": {}},
    {"source": "e2", "target": "k1", "type": "MENTIONED_IN", "properties": {}},
    {"source": "e1", "target": "c1", "type": "IN_COMMUNITY", "properties": {}},
]

_NODE_MATCH = re.compile(r"MATCH \(\w+(?::(\w+))?\)")
_REL_MATCH = re.compile(r"MATCH \(\w+(?::(\w+))?\)-\[\w+(?::(\w+))?\]->\(\w+(?::(\w+))?\)")


def _mixed_graph_handler(query: str, params: dict) -> list[dict]:
    """Emulate Neo4j label matching over that mixed database.

    The *fake* does the filtering, not the assertions: drop the ``:Entity``
    scoping from the router's queries and this handler faithfully hands back the
    community and chunk rows, so the tests below fail instead of silently
    passing on a post-filter that isn't there.
    """
    by_id = {n["id"]: n for n in MIXED_NODES}

    if "elementId(n) AS id" in query:
        (label,) = _NODE_MATCH.search(query).groups()
        return [n for n in MIXED_NODES if label is None or label in n["labels"]]

    if "elementId(a) AS source" in query:
        a_label, rel_type, b_label = _REL_MATCH.search(query).groups()
        return [
            r
            for r in MIXED_RELS
            if (a_label is None or a_label in by_id[r["source"]]["labels"])
            and (b_label is None or b_label in by_id[r["target"]]["labels"])
            and (rel_type is None or rel_type == r["type"])
        ]

    return []


class TestGraph:
    def test_graph_data_shape(self, client, fake_neo4j):
        def handler(query, params):
            if "elementId(n) AS id" in query:
                return [
                    {
                        "id": "1",
                        "labels": ["Entity"],
                        "properties": {"name": "Ada", "type": "PERSON"},
                    }
                ]
            if "elementId(a) AS source" in query:
                return [
                    {"source": "1", "target": "2", "type": "WORKED_ON", "properties": {}}
                ]
            return []

        fake_neo4j(handler)
        r = client.get("/api/graph-data")
        assert r.status_code == 200
        data = r.json()
        assert data["nodes"][0]["label"] == "Ada"
        assert data["links"][0]["type"] == "WORKED_ON"

    def test_graph_data_excludes_chunk_and_community_nodes(self, client, fake_neo4j):
        calls = fake_neo4j(_mixed_graph_handler)

        r = client.get("/api/graph-data")
        assert r.status_code == 200
        nodes = r.json()["nodes"]

        assert [n["label"] for n in nodes] == ["Ada", "Analytical Engine"]
        assert [n["type"] for n in nodes] == ["PERSON", "TOOL"]
        for node in nodes:
            # Source prose and community summaries are separate products
            # (/api/chat sources, /api/communities) — not graph payload.
            assert "text" not in node["properties"]
            assert "summary" not in node["properties"]
            # Embedding vectors are large and useless in the browser.
            assert "embedding" not in node["properties"]
        assert "Ada Lovelace wrote" not in r.text
        assert "Computing pioneers" not in r.text
        # And it holds because the *query* is scoped, not because of a filter
        # applied afterwards that a future refactor could quietly drop.
        assert "MATCH (n:Entity)" in calls[0][0]

    def test_graph_data_excludes_mentioned_in_and_community_edges(
        self, client, fake_neo4j
    ):
        calls = fake_neo4j(_mixed_graph_handler)

        r = client.get("/api/graph-data")
        assert r.status_code == 200
        body = r.json()
        links = body["links"]

        assert [link["type"] for link in links] == ["WORKED_ON"]
        # No edge may dangle into a node the visualization never received.
        node_ids = {n["id"] for n in body["nodes"]}
        assert all(
            link["source"] in node_ids and link["target"] in node_ids for link in links
        )
        assert "MATCH (a:Entity)-[r]->(b:Entity)" in calls[1][0]

    def test_clear_graph(self, client, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        r = client.delete("/api/graph")
        assert r.status_code == 200
        assert r.json()["status"] == "success"


SAMPLE_COMMUNITY = {
    "id": "com-abc123",
    "title": "Distributed Data Platforms",
    "summary": "Kafka, Spark and Airflow form the ingestion backbone.",
    "size": 3,
    "members": ["Kafka", "Spark", "Airflow"],
}


class TestCommunities:
    def test_lists_community_summaries(self, client, monkeypatch):
        async def fake_summaries(limit=10):
            return [SAMPLE_COMMUNITY]

        monkeypatch.setattr(graph_router, "get_community_summaries", fake_summaries)

        r = client.get("/api/communities")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["communities"][0]["title"] == "Distributed Data Platforms"
        assert body["communities"][0]["members"] == ["Kafka", "Spark", "Airflow"]

    def test_limit_is_passed_through(self, client, monkeypatch):
        seen: list[int] = []

        async def fake_summaries(limit=10):
            seen.append(limit)
            return []

        monkeypatch.setattr(graph_router, "get_community_summaries", fake_summaries)

        r = client.get("/api/communities?limit=5")
        assert r.status_code == 200
        assert seen == [5]
        assert r.json() == {"communities": [], "count": 0}

    @pytest.mark.parametrize("limit", [0, -1, 500])
    def test_rejects_out_of_range_limit(self, client, limit):
        r = client.get(f"/api/communities?limit={limit}")
        assert r.status_code == 422

    def test_empty_database_is_not_an_error(self, client, monkeypatch):
        async def fake_summaries(limit=10):
            return []

        monkeypatch.setattr(graph_router, "get_community_summaries", fake_summaries)

        r = client.get("/api/communities")
        assert r.status_code == 200
        assert r.json()["communities"] == []

    def test_rebuild_returns_a_job_id(self, client, stub_communities):
        stub_communities()
        r = client.post("/api/communities/rebuild")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"].startswith("job_")
        assert body["status"] == "processing"

    def test_rebuild_streams_progress_then_done(self, client, stub_communities):
        calls = stub_communities()
        job_id = client.post("/api/communities/rebuild").json()["job_id"]

        r = client.get(f"/api/communities/rebuild/{job_id}/events")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = sse_events(r.text)
        assert [e["type"] for e in events] == ["progress", "done"]
        assert events[0]["stage"] == "summarizing"
        assert events[-1]["data"] == {
            "communities": 2,
            "summarized": 2,
            "modularity": 0.42,
        }
        assert len(calls) == 1

    def test_rebuild_failure_streams_an_error_event(self, client, stub_communities):
        stub_communities(raises=RuntimeError("no graph"))
        job_id = client.post("/api/communities/rebuild").json()["job_id"]

        events = sse_events(client.get(f"/api/communities/rebuild/{job_id}/events").text)
        assert events[-1]["type"] == "error"
        assert "no graph" in events[-1]["data"]

    def test_unknown_rebuild_job_yields_an_error(self, client):
        events = sse_events(client.get("/api/communities/rebuild/job_999999/events").text)
        assert events == [{"type": "error", "data": "Unknown job id."}]
