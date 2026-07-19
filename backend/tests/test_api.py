"""API-level tests with the FastAPI TestClient (no real DB / LLM).

The client is created WITHOUT the context-manager form so the app lifespan
(Neo4j connect + schema bootstrap) does not run — each endpoint's DB access is
mocked via the ``fake_neo4j`` fixture instead.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app
from app.routers import chat as chat_router
from app.routers import upload as upload_router


@pytest.fixture
def client():
    return TestClient(app)


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

    def test_accepts_pdf_and_returns_job_id(self, client, monkeypatch):
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

    def test_clear_graph(self, client, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        r = client.delete("/api/graph")
        assert r.status_code == 200
        assert r.json()["status"] == "success"
