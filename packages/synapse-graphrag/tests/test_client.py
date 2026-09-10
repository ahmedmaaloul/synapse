# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""``SynapseClient`` against the fake backend: wire format, SSE, errors, config."""

from __future__ import annotations

import json
from contextlib import aclosing

import httpx
import pytest

from synapse_graphrag import Answer, IngestResult, Retrieval, SynapseClient, SynapseError, run
from synapse_graphrag.client import (
    SynapseConnectionError,
    env_max_context_chars,
    env_timeout,
    env_url,
    iter_sse,
)
from tests.conftest import BASE_URL, INGEST_DONE, RETRIEVAL, FakeBackend, split_bytes, sse


# ── SSE parser ───────────────────────────────────────────────────────────────
async def _collect(chunks: list[bytes]) -> list[dict]:
    async def gen():
        for chunk in chunks:
            yield chunk

    return [event async for event in iter_sse(gen())]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16, 100_000])
async def test_sse_frames_survive_any_chunk_boundary(size: int):
    events = [{"type": "token", "data": "héllo ✓"}, {"type": "done"}, {"type": "x", "data": [1, 2]}]
    assert await _collect(split_bytes(sse(*events), size)) == events


async def test_sse_ignores_comments_other_fields_and_crlf():
    raw = (
        b": keep-alive\r\n"
        b"event: token\r\n"
        b"id: 7\r\n"
        b'data: {"type": "token",\r\n'
        b'data:  "data": "two lines"}\r\n'
        b"\r\n"
        b'data: {"type": "done"}'  # no trailing blank line at EOF
    )
    assert await _collect([raw]) == [{"type": "token", "data": "two lines"}, {"type": "done"}]


async def test_sse_malformed_frame_is_a_synapse_error():
    with pytest.raises(SynapseError, match="Malformed SSE frame"):
        await _collect([b"data: {not json}\n\n"])


# ── retrieve ─────────────────────────────────────────────────────────────────
async def test_retrieve_sends_budget_and_parses_payload(backend: FakeBackend):
    async with backend.client() as client:
        retrieval = await client.retrieve("Who designed the Analytical Engine?", k=5, max_context_chars=2000)

    call = backend.calls[0]
    assert (call.method, call.path) == ("POST", "/api/retrieve")
    assert call.json == {"query": "Who designed the Analytical Engine?", "k": 5, "max_context_chars": 2000}
    assert call.headers["content-type"] == "application/json"
    assert isinstance(retrieval, Retrieval)
    assert retrieval.mode == "local"
    assert retrieval.context == RETRIEVAL["context"]
    assert [c["name"] for c in retrieval.citations] == ["Charles Babbage", "Analytical Engine"]
    assert retrieval.paths == RETRIEVAL["paths"]
    assert retrieval.sources == RETRIEVAL["sources"]
    assert retrieval.usage["truncated"] is False
    assert retrieval.to_dict()["usage"]["context_tokens_est"] == 21


async def test_retrieve_omits_budget_when_none_and_reports_truncation(backend: FakeBackend):
    async with backend.client() as client:
        untouched = await client.retrieve("q")
        cut = await client.retrieve("q", max_context_chars=20)

    assert "max_context_chars" not in backend.calls[0].json
    assert untouched.usage["truncated"] is False
    assert cut.usage["truncated"] is True
    assert "[context truncated to 20 chars]" in cut.context


# ── ask / ask_stream ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("chunk_size", [1, 5, 64, 4096])
async def test_ask_aggregates_split_frames(chunk_size: int):
    backend = FakeBackend(chunk_size=chunk_size)
    async with backend.client() as client:
        answer = await client.ask("q", history=[{"role": "user", "content": "hi", "extra": "dropped"}])

    call = backend.calls[0]
    assert (call.method, call.path) == ("POST", "/api/chat")
    assert call.json == {"query": "q", "history": [{"role": "user", "content": "hi"}]}
    assert call.headers["accept"] == "text/event-stream"
    assert isinstance(answer, Answer)
    assert answer.text == "Babbage designed it — Ünïcödé ✓"
    assert [c["name"] for c in answer.citations] == ["Charles Babbage", "Analytical Engine"]
    assert answer.paths == RETRIEVAL["paths"]
    assert answer.sources == RETRIEVAL["sources"]
    assert answer.usage == {"context_chars": 83, "context_tokens_est": 21, "answer_chars": 32}


async def test_ask_without_usage_on_done_still_counts_answer_chars(backend: FakeBackend):
    backend.chat_events = [{"type": "token", "data": "abc"}, {"type": "done"}]
    async with backend.client() as client:
        answer = await client.ask("q")
    assert answer.text == "abc"
    assert answer.usage == {"answer_chars": 3}


async def test_ask_raises_on_error_event(backend: FakeBackend):
    backend.chat_events = [
        {"type": "citations", "data": []},
        {"type": "error", "data": "Generation failed: no API key"},
    ]
    async with backend.client() as client:
        with pytest.raises(SynapseError, match="Generation failed: no API key"):
            await client.ask("q")


async def test_ask_stream_yields_raw_events(backend: FakeBackend):
    async with backend.client() as client:
        events = [event async for event in client.ask_stream("q")]
    assert [e["type"] for e in events] == ["citations", "paths", "sources", "token", "token", "done"]


# ── ingest ───────────────────────────────────────────────────────────────────
async def test_ingest_pdf_uploads_multipart_then_follows_events(backend: FakeBackend, pdf_path):
    seen: list[dict] = []

    async def on_progress(event: dict) -> None:
        seen.append(event)

    async with backend.client() as client:
        result = await client.ingest_pdf(pdf_path, theme="History", on_progress=on_progress)

    upload = backend.calls[0]
    assert (upload.method, upload.path) == ("POST", "/api/upload")
    assert upload.headers["content-type"].startswith("multipart/form-data")
    assert b'name="file"; filename="history.pdf"' in upload.body
    assert b"%PDF-1.4" in upload.body
    assert b'name="theme"\r\n\r\nHistory' in upload.body
    assert backend.paths("GET") == ["/api/upload/job_000007/events"]

    assert isinstance(result, IngestResult)
    assert result.job_id == "job_000007"
    assert (result.filename, result.chunks_processed) == ("history.pdf", 3)
    assert (result.nodes_created, result.relationships_created, result.communities) == (12, 18, 2)
    assert result.data == INGEST_DONE
    assert result.to_dict() == {"job_id": "job_000007", **INGEST_DONE}

    assert seen[0] == {"type": "accepted", "job_id": "job_000007", "filename": "history.pdf", "total_chunks": 3, "status": "processing"}
    assert [e["type"] for e in seen[1:]] == ["progress", "progress", "community_progress"]


async def test_ingest_pdf_sync_progress_callback_and_error_event(backend: FakeBackend, pdf_path):
    backend.ingest_events = [
        {"type": "progress", "stage": "extracting", "processed": 0, "total": 3},
        {"type": "error", "data": "LLM extraction failed"},
    ]
    seen: list[str] = []
    async with backend.client() as client:
        with pytest.raises(SynapseError, match="LLM extraction failed"):
            await client.ingest_pdf(pdf_path, on_progress=lambda e: seen.append(e["type"]))
    assert seen == ["accepted", "progress"]


async def test_ingest_pdf_rejects_missing_or_non_pdf_files(backend: FakeBackend, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    async with backend.client() as client:
        with pytest.raises(SynapseError, match="File not found"):
            await client.ingest_pdf(tmp_path / "missing.pdf")
        with pytest.raises(SynapseError, match="Only PDF files"):
            await client.ingest_pdf(other)
    assert backend.calls == []


async def test_ingest_pdf_upload_rejected_by_backend(backend: FakeBackend, pdf_path):
    backend.failures["/api/upload"] = (400, "Could not extract text from PDF (is it a scanned image?).")
    async with backend.client() as client:
        with pytest.raises(SynapseError) as info:
            await client.ingest_pdf(pdf_path)
    assert info.value.status == 400
    assert "scanned image" in info.value.detail


# ── graph, communities, clear ────────────────────────────────────────────────
async def test_graph_communities_rebuild_and_clear(backend: FakeBackend):
    progress: list[str] = []
    async with backend.client() as client:
        graph = await client.graph()
        communities = await client.communities(limit=5)
        rebuilt = await client.rebuild_communities(on_progress=lambda e: progress.append(e["stage"]))
        cleared = await client.clear_graph()

    assert len(graph["nodes"]) == 4 and len(graph["links"]) == 3
    assert communities["count"] == 1
    assert backend.calls[1].params == {"limit": "5"}
    assert rebuilt == {"job_id": "job_000008", "communities": 2, "summarized": 2, "modularity": 0.41}
    assert progress == ["clustering"]
    assert cleared["status"] == "success"
    assert [(c.method, c.path) for c in backend.calls] == [
        ("GET", "/api/graph-data"),
        ("GET", "/api/communities"),
        ("POST", "/api/communities/rebuild"),
        ("GET", "/api/communities/rebuild/job_000008/events"),
        ("DELETE", "/api/graph"),
    ]


async def test_health_ready_about(backend: FakeBackend):
    async with backend.client() as client:
        assert (await client.health())["status"] == "ok"
        assert (await client.ready())["neo4j"] == "up"
        assert (await client.about())["version"] == "0.4.0"


# ── headers & configuration ──────────────────────────────────────────────────
async def test_api_key_becomes_bearer_header(backend: FakeBackend):
    async with backend.client(api_key="sk-test") as client:
        await client.health()
    headers = backend.calls[0].headers
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["user-agent"].startswith("synapse-graphrag/")


async def test_no_authorization_header_without_key(backend: FakeBackend):
    async with backend.client() as client:
        await client.health()
    assert "authorization" not in backend.calls[0].headers


async def test_api_key_and_url_come_from_env(backend: FakeBackend, monkeypatch):
    monkeypatch.setenv("SYNAPSE_API_KEY", "from-env")
    monkeypatch.setenv("SYNAPSE_URL", "http://env.test/")
    monkeypatch.setenv("SYNAPSE_TIMEOUT", "7.5")
    client = SynapseClient(transport=backend.transport)
    assert client.base_url == "http://env.test"
    assert client.api_key == "from-env"
    assert client.timeout == 7.5
    async with client:
        await client.health()
    assert backend.calls[0].headers["authorization"] == "Bearer from-env"


def test_env_defaults_and_validation(monkeypatch):
    assert env_url() == "http://localhost:8000"
    assert env_timeout() == 120.0
    assert env_max_context_chars() == 6000
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "0")
    assert env_max_context_chars() is None
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "-5")
    assert env_max_context_chars() is None
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "200")
    assert env_max_context_chars() == 200
    # 1..199 would make the backend 422 every retrieval: refuse it up front.
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "199")
    with pytest.raises(SynapseError, match=r"must be 0 \(no budget\) or at least 200, got 199"):
        env_max_context_chars()
    monkeypatch.setenv("SYNAPSE_TIMEOUT", "soon")
    with pytest.raises(SynapseError, match="SYNAPSE_TIMEOUT must be a number"):
        env_timeout()


# ── error mapping ────────────────────────────────────────────────────────────
async def test_http_error_with_json_detail(backend: FakeBackend):
    backend.failures["/api/retrieve"] = (500, "Failed to query the knowledge graph.")
    async with backend.client() as client:
        with pytest.raises(SynapseError) as info:
            await client.retrieve("q")
    assert info.value.status == 500
    assert info.value.detail == "Failed to query the knowledge graph."
    assert str(info.value) == "HTTP 500: Failed to query the knowledge graph."


async def test_http_422_validation_list_is_flattened(backend: FakeBackend):
    backend.failures["/api/retrieve"] = (
        422,
        [
            {"loc": ["body", "k"], "msg": "Input should be less than or equal to 20", "type": "less_than_equal"},
            {"loc": ["body", "query"], "msg": "String should have at least 1 character", "type": "string_too_short"},
        ],
    )
    async with backend.client() as client:
        with pytest.raises(SynapseError) as info:
            await client.retrieve("", k=21)
    assert info.value.status == 422
    assert info.value.detail == (
        "k: Input should be less than or equal to 20; query: String should have at least 1 character"
    )


async def test_http_error_without_json_uses_body(backend: FakeBackend):
    backend.failures["/health"] = (504, None)
    async with backend.client() as client:
        with pytest.raises(SynapseError) as info:
            await client.health()
    assert info.value.status == 504
    assert "gateway timeout" in info.value.detail


async def test_streaming_http_error_is_mapped(backend: FakeBackend):
    backend.failures["/api/chat"] = (503, "LLM provider unavailable")
    async with backend.client() as client:
        with pytest.raises(SynapseError) as info:
            await client.ask("q")
    assert (info.value.status, info.value.detail) == (503, "LLM provider unavailable")


async def test_unreachable_backend_is_a_connection_error(backend: FakeBackend):
    backend.unreachable = True
    async with backend.client() as client:
        with pytest.raises(SynapseConnectionError) as info:
            await client.retrieve("q")
        with pytest.raises(SynapseConnectionError):
            await client.ask("q")
    assert info.value.status is None
    assert str(info.value).startswith(f"Cannot reach Synapse at {BASE_URL}: ")
    assert "Is it running?" in str(info.value)


async def test_non_json_success_body_is_an_error(backend: FakeBackend):
    backend.failures["/api/about"] = (200, None)
    async with backend.client() as client:
        with pytest.raises(SynapseError, match="Expected JSON"):
            await client.about()


def test_run_helper_drives_a_coroutine(backend: FakeBackend):
    async def go():
        async with backend.client() as client:
            return await client.health()

    assert run(go()) == {"status": "ok", "service": "synapse-backend"}


def test_results_round_trip_through_json():
    retrieval = Retrieval.from_payload(RETRIEVAL)
    assert json.loads(json.dumps(retrieval.to_dict())) == RETRIEVAL
    assert Retrieval.from_payload({}).mode == "local"


# ── transport failures ───────────────────────────────────────────────────────
async def test_read_timeout_names_the_timeout_not_a_dead_backend():
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    async with SynapseClient(BASE_URL, timeout=5, transport=httpx.MockTransport(stall)) as client:
        with pytest.raises(SynapseError) as info:
            await client.health()
    assert not isinstance(info.value, SynapseConnectionError)
    assert str(info.value) == f"Synapse at {BASE_URL} did not respond within 5s (raise SYNAPSE_TIMEOUT)."


async def test_connect_timeout_is_a_connection_error_with_a_reason():
    def never_opens(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("", request=request)  # httpx leaves the message empty

    async with SynapseClient(BASE_URL, timeout=5, transport=httpx.MockTransport(never_opens)) as client:
        with pytest.raises(SynapseConnectionError) as info:
            await client.health()
    assert str(info.value) == f"Cannot reach Synapse at {BASE_URL}: ConnectTimeout. Is it running? (`make up`)"


async def test_stream_dropped_mid_way_is_reported_as_such():
    class Dropped(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"type": "citations", "data": []}, {"type": "token", "data": "half"})
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Dropped())

    async with SynapseClient(BASE_URL, timeout=5, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SynapseError, match="closed the connection mid-stream: peer closed") as info:
            await client.ask("q")
    assert not isinstance(info.value, SynapseConnectionError)


# ── stream lifecycle ─────────────────────────────────────────────────────────
async def test_ask_stream_without_done_is_an_error_not_a_partial_answer(backend: FakeBackend):
    backend.chat_events = [{"type": "citations", "data": []}, {"type": "token", "data": "partial"}]
    async with backend.client() as client:
        with pytest.raises(SynapseError, match="Chat stream ended without a done event"):
            await client.ask("q")


async def test_ask_closes_the_http_stream_in_the_reader_task(backend: FakeBackend):
    """Leaving on ``done`` must tear the generator chain down right there.

    Left to the event loop's shutdown finaliser, the nested generators race and
    ``asyncio.run`` prints ``RuntimeError: aclose(): asynchronous generator is
    already running`` after every otherwise successful CLI ``ask``.
    """
    async with backend.client() as client:
        await client.ask("q")
        assert backend.streams[-1].closed_by_reader is True

        async with aclosing(client.ask_stream("q")) as events:  # the documented early-exit form
            async for _ in events:
                break
        assert backend.streams[-1].closed_by_reader is True

        with pytest.raises(SynapseError, match="Generation failed"):
            backend.chat_events = [{"type": "error", "data": "Generation failed: no key"}]
            await client.ask("q")
        assert backend.streams[-1].closed_by_reader is True


async def test_follow_job_closes_the_event_stream_in_the_reader_task(backend: FakeBackend, pdf_path):
    async with backend.client() as client:
        await client.ingest_pdf(pdf_path)
        assert backend.streams[-1].closed_by_reader is True
        await client.rebuild_communities()
        assert backend.streams[-1].closed_by_reader is True
