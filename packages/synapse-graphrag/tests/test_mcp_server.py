# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The MCP server: tool catalogue, in-process calls, and a real client session.

In-process ``server.call_tool`` raises ``ToolError`` for anticipated failures;
over the protocol the same failure is a ``CallToolResult(is_error=True)`` whose
text is the message. Both paths are covered so the error policy documented in
``mcp_server.py`` is what a host actually observes.
"""

from __future__ import annotations

import json

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.memory import create_client_server_memory_streams

from synapse_graphrag import __version__
from synapse_graphrag.mcp_server import (
    SERVER_NAME,
    build_server,
    find_entities,
    graph_stats,
    main,
)
from tests.conftest import GRAPH, INGEST_DONE, FakeBackend

TOOLS = {
    "synapse_retrieve": {"readOnlyHint": True, "idempotentHint": True},
    "synapse_ask": {"readOnlyHint": True},
    "synapse_ingest_pdf": {"readOnlyHint": False, "destructiveHint": False},
    "synapse_communities": {"readOnlyHint": True},
    "synapse_find_entities": {"readOnlyHint": True},
    "synapse_graph_stats": {"readOnlyHint": True},
    "synapse_status": {"readOnlyHint": True},
    "synapse_clear_graph": {"destructiveHint": True},
}


def make_server(backend: FakeBackend, **kwargs):
    return build_server(client_factory=backend.client, **kwargs)


def _structured(result) -> dict:
    assert result.is_error is False, result.content
    assert result.structured_content is not None
    # The text block carries the same JSON for hosts without structured output.
    assert json.loads(result.content[0].text) == result.structured_content
    return result.structured_content


# ── catalogue ────────────────────────────────────────────────────────────────
async def test_list_tools_names_and_annotations(backend: FakeBackend):
    server = make_server(backend)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == set(TOOLS)
    for name, expected in TOOLS.items():
        hints = tools[name].annotations.model_dump(by_alias=True)
        for key, value in expected.items():
            assert hints[key] is value, f"{name}.{key}"
        assert tools[name].description, name
    retrieve = tools["synapse_retrieve"].input_schema["properties"]
    assert retrieve["k"] == {"default": 8, "minimum": 1, "maximum": 20, "title": "K", "type": "integer", "description": retrieve["k"]["description"]}
    assert retrieve["max_context_chars"]["default"] is None
    assert tools["synapse_retrieve"].output_schema["type"] == "object"
    assert server.name == SERVER_NAME
    assert server.version == __version__
    assert "synapse_retrieve" in server.instructions


async def test_resource_and_prompt(backend: FakeBackend):
    server = make_server(backend)
    resources = await server.list_resources()
    assert [str(r.uri) for r in resources] == ["synapse://about"]
    contents = list(await server.read_resource("synapse://about"))
    assert contents[0].mime_type == "application/json"
    assert json.loads(contents[0].content)["version"] == "0.4.0"

    prompts = await server.list_prompts()
    assert [p.name for p in prompts] == ["answer_with_graph", "safety_brief"]
    assert [a.name for a in prompts[0].arguments] == ["question"]
    prompt = await server.get_prompt("answer_with_graph", {"question": "Who is Ada?"})
    text = prompt.messages[0].content.text
    assert "synapse_retrieve" in text and text.endswith("Question: Who is Ada?")
    assert "does not contain the answer" in text

    brief = await server.get_prompt("safety_brief", {"topic": "sandbagging"})
    brief_text = brief.messages[0].content.text
    assert "Not in the corpus" in brief_text and brief_text.endswith("Topic: sandbagging")
    assert "synapse_retrieve" in brief_text and "synapse_communities" in brief_text


# ── retrieve & cache ─────────────────────────────────────────────────────────
async def test_retrieve_returns_budgeted_context_and_caches(backend: FakeBackend):
    server = make_server(backend, cache_ttl=300)
    first = _structured(await server.call_tool("synapse_retrieve", {"query": "Babbage", "k": 3}))
    second = _structured(await server.call_tool("synapse_retrieve", {"query": "Babbage", "k": 3}))
    other = _structured(await server.call_tool("synapse_retrieve", {"query": "Babbage", "k": 4}))

    assert backend.paths() == ["/api/retrieve", "/api/retrieve"]  # 2nd call was a cache hit
    assert backend.calls[0].json == {"query": "Babbage", "k": 3, "max_context_chars": 6000}
    assert first["mode"] == "local"
    assert first["context"].startswith("Charles Babbage designed")
    assert first["usage"]["cached"] is False
    assert second["usage"]["cached"] is True
    assert {k: v for k, v in second.items() if k != "usage"} == {k: v for k, v in first.items() if k != "usage"}
    assert other["usage"]["cached"] is False


async def test_retrieve_budget_from_env_and_explicit_override(backend: FakeBackend, monkeypatch):
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "2500")
    server = make_server(backend, cache_ttl=0)
    await server.call_tool("synapse_retrieve", {"query": "q"})
    cut = _structured(await server.call_tool("synapse_retrieve", {"query": "q", "max_context_chars": 200}))
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "0")
    await server.call_tool("synapse_retrieve", {"query": "q"})

    bodies = [c.json for c in backend.calls]
    assert bodies[0]["max_context_chars"] == 2500
    assert bodies[1]["max_context_chars"] == 200
    assert "max_context_chars" not in bodies[2]  # 0 disables the budget
    assert cut["usage"]["cached"] is False


async def test_cache_disabled_with_ttl_zero(backend: FakeBackend):
    server = make_server(backend, cache_ttl=0)
    await server.call_tool("synapse_retrieve", {"query": "q"})
    result = _structured(await server.call_tool("synapse_retrieve", {"query": "q"}))
    assert len(backend.calls) == 2
    assert result["usage"]["cached"] is False


async def test_retrieve_validates_arguments_before_calling_the_backend(backend: FakeBackend):
    server = make_server(backend)
    with pytest.raises(ToolError, match="less than or equal to 20"):
        await server.call_tool("synapse_retrieve", {"query": "q", "k": 21})
    with pytest.raises(ToolError):
        await server.call_tool("synapse_retrieve", {"query": ""})
    assert backend.calls == []


# ── other tools ──────────────────────────────────────────────────────────────
async def test_ask_returns_the_generated_answer(backend: FakeBackend):
    server = make_server(backend)
    result = _structured(
        await server.call_tool(
            "synapse_ask", {"query": "q", "history": [{"role": "user", "content": "earlier"}]}
        )
    )
    assert result["text"] == "Babbage designed it — Ünïcödé ✓"
    assert result["usage"]["answer_chars"] == 32
    assert backend.calls[0].json["history"] == [{"role": "user", "content": "earlier"}]


async def test_status(backend: FakeBackend):
    server = make_server(backend)
    result = _structured(await server.call_tool("synapse_status", {}))
    assert result == {
        "url": "http://synapse.test",
        "health": "ok",
        "ready": "ready",
        "neo4j": "up",
        "version": "0.4.0",
        "llm_provider": "gemini",
        "embedding_provider": "fastembed",
        "client_version": __version__,
    }
    assert backend.paths() == ["/health", "/health/ready", "/api/about"]


async def test_find_entities(backend: FakeBackend):
    server = make_server(backend)
    result = _structured(await server.call_tool("synapse_find_entities", {"query": "ada"}))
    assert result["total_matches"] == 1
    assert result["entities"] == [
        {"id": "3", "label": "Ada Lovelace", "type": "PERSON", "degree": 2, "description": "First programmer"}
    ]
    by_type = _structured(await server.call_tool("synapse_find_entities", {"query": "PERSON", "limit": 1}))
    assert by_type["total_matches"] == 2 and by_type["returned"] == 1
    assert by_type["entities"][0]["label"] in {"Ada Lovelace", "Charles Babbage"}  # both degree 2
    none = _structured(await server.call_tool("synapse_find_entities", {"query": "zzz"}))
    assert none["entities"] == []


def test_find_entities_orders_by_degree_then_label():
    result = find_entities(GRAPH, "e")  # matches every node; three share degree 2
    assert [e["label"] for e in result["entities"]] == [
        "Ada Lovelace",
        "Analytical Engine",
        "Charles Babbage",
        "Lonely Node",
    ]


async def test_graph_stats(backend: FakeBackend):
    server = make_server(backend)
    result = _structured(await server.call_tool("synapse_graph_stats", {}))
    assert result["nodes"] == 4 and result["edges"] == 3 and result["isolated_nodes"] == 1
    assert result["entity_types"] == {"PERSON": 2, "TOOL": 1, "CONCEPT": 1}
    assert result["relationship_types"] == {"DESIGNED": 1, "WROTE_PROGRAMS_FOR": 1, "COLLABORATED_WITH": 1}
    assert result["top_entities"][0]["degree"] == 2
    assert graph_stats({"nodes": [], "links": []})["nodes"] == 0


async def test_communities(backend: FakeBackend):
    server = make_server(backend)
    result = _structured(await server.call_tool("synapse_communities", {"limit": 3}))
    assert result["count"] == 1
    assert backend.calls[0].params == {"limit": "3"}


async def test_ingest_pdf(backend: FakeBackend, pdf_path, tmp_path):
    server = make_server(backend)
    result = _structured(await server.call_tool("synapse_ingest_pdf", {"path": str(pdf_path), "theme": "History"}))
    assert result == {"job_id": "job_000007", **INGEST_DONE}
    assert b"History" in backend.calls[0].body

    with pytest.raises(ToolError, match="File not found"):
        await server.call_tool("synapse_ingest_pdf", {"path": str(tmp_path / "nope.pdf")})
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(ToolError, match="Only PDF files"):
        await server.call_tool("synapse_ingest_pdf", {"path": str(txt)})
    assert backend.paths("POST") == ["/api/upload"]


async def test_clear_graph_refuses_without_confirm(backend: FakeBackend):
    server = make_server(backend)
    refused = _structured(await server.call_tool("synapse_clear_graph", {}))
    assert refused["cleared"] is False and "confirm=true" in refused["message"]
    assert backend.calls == []

    cleared = _structured(await server.call_tool("synapse_clear_graph", {"confirm": True}))
    assert cleared["cleared"] is True and cleared["status"] == "success"
    assert backend.paths("DELETE") == ["/api/graph"]


async def test_clear_graph_invalidates_the_retrieval_cache(backend: FakeBackend):
    server = make_server(backend, cache_ttl=300)
    await server.call_tool("synapse_retrieve", {"query": "q"})
    await server.call_tool("synapse_clear_graph", {"confirm": True})
    result = _structured(await server.call_tool("synapse_retrieve", {"query": "q"}))
    assert result["usage"]["cached"] is False
    assert backend.paths("POST") == ["/api/retrieve", "/api/retrieve"]


# ── error policy ─────────────────────────────────────────────────────────────
async def test_unreachable_backend_is_a_readable_tool_error(backend: FakeBackend):
    backend.unreachable = True
    server = make_server(backend)
    for name, args in [
        ("synapse_status", {}),
        ("synapse_retrieve", {"query": "q"}),
        ("synapse_graph_stats", {}),
    ]:
        with pytest.raises(ToolError) as info:
            await server.call_tool(name, args)
        message = str(info.value)
        assert "Cannot reach Synapse at http://synapse.test" in message, name
        assert "Is it running?" in message
        assert "Traceback" not in message


async def test_backend_http_error_is_a_tool_error(backend: FakeBackend):
    backend.failures["/api/retrieve"] = (500, "Failed to query the knowledge graph.")
    server = make_server(backend)
    with pytest.raises(ToolError, match="HTTP 500: Failed to query the knowledge graph."):
        await server.call_tool("synapse_retrieve", {"query": "q"})


async def test_malformed_env_budget_is_a_readable_tool_error(backend: FakeBackend, monkeypatch):
    """Env parsing runs before ``_connected``; its errors must still be ToolErrors, not tracebacks."""
    server = make_server(backend, cache_ttl=0)
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "abc")
    with pytest.raises(ToolError, match="SYNAPSE_MAX_CONTEXT_CHARS must be an integer, got 'abc'"):
        await server.call_tool("synapse_retrieve", {"query": "q"})
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "100")  # the backend's floor is 200
    with pytest.raises(ToolError, match=r"must be 0 \(no budget\) or at least 200, got 100"):
        await server.call_tool("synapse_retrieve", {"query": "q"})
    assert backend.calls == []
    # An explicit argument never consults the env.
    result = _structured(await server.call_tool("synapse_retrieve", {"query": "q", "max_context_chars": 250}))
    assert result["usage"]["cached"] is False
    assert backend.calls[0].json["max_context_chars"] == 250


# ── end to end over the protocol ─────────────────────────────────────────────
async def test_end_to_end_client_session(backend: FakeBackend):
    server = make_server(backend)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:

            async def serve() -> None:
                await server._lowlevel_server.run(
                    server_read,
                    server_write,
                    server._lowlevel_server.create_initialization_options(),
                    raise_exceptions=True,
                )

            tg.start_soon(serve)
            async with ClientSession(client_read, client_write) as session:
                init = await session.initialize()
                assert init.server_info.name == SERVER_NAME
                assert init.server_info.version == __version__
                assert "synapse_retrieve" in (init.instructions or "")

                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == set(TOOLS)

                status = await session.call_tool("synapse_status", {})
                assert status.is_error is False
                assert status.structured_content["version"] == "0.4.0"
                assert status.structured_content["llm_provider"] == "gemini"

                backend.unreachable = True
                failed = await session.call_tool("synapse_retrieve", {"query": "q"})
                assert failed.is_error is True
                assert "Cannot reach Synapse at http://synapse.test" in failed.content[0].text
                backend.unreachable = False

                resource = await session.read_resource("synapse://about")
                assert json.loads(resource.contents[0].text)["license"] == "PolyForm-Noncommercial-1.0.0"

                prompt = await session.get_prompt("answer_with_graph", {"question": "Why?"})
                assert prompt.messages[0].content.text.endswith("Question: Why?")
            tg.cancel_scope.cancel()


# ── entry point ──────────────────────────────────────────────────────────────
def test_main_parses_args_and_runs_the_chosen_transport(monkeypatch):
    runs: list[tuple] = []

    class FakeServer:
        def run(self, transport, **kwargs):
            runs.append((transport, kwargs))

    monkeypatch.setattr("synapse_graphrag.mcp_server.build_server", lambda: FakeServer())
    assert main([]) == 0
    assert main(["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9000", "--url", "http://x:1"]) == 0
    assert runs == [("stdio", {}), ("streamable-http", {"host": "0.0.0.0", "port": 9000})]
    import os

    assert os.environ["SYNAPSE_URL"] == "http://x:1"
    from uvicorn.config import LOGGING_CONFIG

    assert LOGGING_CONFIG["handlers"]["access"]["stream"] == "ext://sys.stderr"


def test_main_reports_a_malformed_env_value_on_one_line(capsys, monkeypatch):
    monkeypatch.setenv("SYNAPSE_CACHE_TTL", "abc")
    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is the stdio transport
    assert captured.err.strip() == "error: SYNAPSE_CACHE_TTL must be a number, got 'abc'"


def test_main_version_and_help(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"synapse-mcp {__version__}"
    with pytest.raises(SystemExit):
        main(["--transport", "carrier-pigeon"])
