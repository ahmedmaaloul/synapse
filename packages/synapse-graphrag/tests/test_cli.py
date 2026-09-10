# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""``synapse-graphrag`` CLI through ``main([...])`` with a fake backend."""

from __future__ import annotations

import json
import logging

import pytest

from synapse_graphrag import __version__
from synapse_graphrag.cli import MCP_CLIENTS, main
from tests.conftest import RETRIEVAL, FakeBackend


def run_cli(backend: FakeBackend, *argv: str) -> int:
    return main(list(argv), client_factory=backend.client)


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"synapse-graphrag {__version__}"


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "retrieve" in out and "install-config" in out


# ── install-config ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("client", [c for c in MCP_CLIENTS if c != "claude-code"])
def test_install_config_emits_valid_json_per_client(client: str, capsys):
    assert main(["install-config", "--client", client, "--url", "http://kg.internal:8000", "--budget", "4000"]) == 0
    captured = capsys.readouterr()
    config = json.loads(captured.out)  # stdout is pure JSON; notes go to stderr
    entry = config["servers"]["synapse"] if client == "vscode" else config["mcpServers"]["synapse"]
    assert entry["command"] == "uvx"
    assert entry["args"] == ["synapse-graphrag", "mcp"]
    assert entry["env"] == {"SYNAPSE_URL": "http://kg.internal:8000", "SYNAPSE_MAX_CONTEXT_CHARS": "4000"}
    if client == "vscode":
        assert entry["type"] == "stdio"
    assert captured.err.strip()


def test_install_config_claude_code_prints_the_add_command(capsys):
    assert main(["install-config", "--client", "claude-code"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == "claude mcp add synapse -e SYNAPSE_URL=http://localhost:8000 -- uvx synapse-graphrag mcp"


def test_install_config_uses_global_url_and_env(capsys, monkeypatch):
    assert main(["--url", "http://global:8000", "install-config", "--client", "cursor"]) == 0
    assert json.loads(capsys.readouterr().out)["mcpServers"]["synapse"]["env"] == {"SYNAPSE_URL": "http://global:8000"}
    monkeypatch.setenv("SYNAPSE_URL", "http://from-env:8000")
    assert main(["install-config", "--client", "claude-code"]) == 0
    assert "SYNAPSE_URL=http://from-env:8000" in capsys.readouterr().out


# ── retrieve ─────────────────────────────────────────────────────────────────
def test_retrieve_json(backend: FakeBackend, capsys):
    assert run_cli(backend, "retrieve", "Who designed the Analytical Engine?", "--k", "3", "--budget", "1000", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == RETRIEVAL
    assert backend.calls[0].json == {"query": "Who designed the Analytical Engine?", "k": 3, "max_context_chars": 1000}


def test_retrieve_budget_defaults_to_env_like_the_mcp_server(backend: FakeBackend, monkeypatch):
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "300")
    assert run_cli(backend, "retrieve", "q", "--json") == 0
    assert run_cli(backend, "retrieve", "q", "--budget", "500", "--json") == 0
    assert run_cli(backend, "retrieve", "q", "--budget", "0", "--json") == 0  # 0 = no budget
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "0")
    assert run_cli(backend, "retrieve", "q", "--json") == 0
    monkeypatch.delenv("SYNAPSE_MAX_CONTEXT_CHARS")
    assert run_cli(backend, "retrieve", "q", "--json") == 0
    assert [c.json.get("max_context_chars") for c in backend.calls] == [300, 500, None, None, 6000]


def test_retrieve_rejects_a_malformed_env_budget_with_one_line(backend: FakeBackend, capsys, monkeypatch):
    monkeypatch.setenv("SYNAPSE_MAX_CONTEXT_CHARS", "100")
    assert run_cli(backend, "retrieve", "q") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        "error: SYNAPSE_MAX_CONTEXT_CHARS must be 0 (no budget) or at least 200, got 100"
    )
    assert backend.calls == []


def test_retrieve_plain_text(backend: FakeBackend, capsys):
    assert run_cli(backend, "retrieve", "q", "--budget", "20") == 0
    captured = capsys.readouterr()
    assert "[context truncated to 20 chars]" in captured.out
    assert "truncated" in captured.err and "Citations: Charles Babbage, Analytical Engine" in captured.err


# ── other commands ───────────────────────────────────────────────────────────
def test_status(backend: FakeBackend, capsys):
    assert run_cli(backend, "status") == 0
    out = capsys.readouterr().out
    assert "Synapse at http://synapse.test" in out
    assert "ready:     ready (neo4j up)" in out
    assert "llm:       gemini" in out


def test_ask_streams_tokens(backend: FakeBackend, capsys):
    assert run_cli(backend, "ask", "q") == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Babbage designed it — Ünïcödé ✓"
    assert "Citations: Charles Babbage, Analytical Engine" in captured.err


def test_ask_error_event_exits_1(backend: FakeBackend, capsys):
    backend.chat_events = [{"type": "error", "data": "Generation failed: no API key"}]
    assert run_cli(backend, "ask", "q") == 1
    err = capsys.readouterr().err
    assert err.strip().endswith("error: Generation failed: no API key")
    assert "Traceback" not in err and "RuntimeError" not in err


def test_ask_without_done_exits_1(backend: FakeBackend, capsys):
    backend.chat_events = [{"type": "token", "data": "half"}]
    assert run_cli(backend, "ask", "q") == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == "half"
    assert captured.err.strip() == "error: Chat stream ended without a done event."


def test_ask_closes_the_stream_itself_and_leaves_the_loop_nothing_to_finalise(
    backend: FakeBackend, caplog
):
    """``ask`` runs under ``asyncio.run``; a generator chain left open there is
    closed by the loop's finaliser, which races and logs ``RuntimeError:
    aclose(): asynchronous generator is already running`` to stderr."""
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        assert run_cli(backend, "ask", "q") == 0
        assert backend.streams[-1].closed_by_reader is True
        backend.chat_events = [{"type": "error", "data": "boom"}]
        assert run_cli(backend, "ask", "q") == 1
        assert backend.streams[-1].closed_by_reader is True
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


def test_ingest_prints_progress_and_summary(backend: FakeBackend, pdf_path, capsys):
    assert run_cli(backend, "ingest", str(pdf_path), "--theme", "History") == 0
    captured = capsys.readouterr()
    assert "Uploaded history.pdf: 3 chunks (job job_000007)" in captured.err
    assert "[ingest] extracting 3/3" in captured.err
    assert "[communities] summarizing 2/2" in captured.err
    assert captured.out.strip() == "Ingested history.pdf: 3 chunks → 12 nodes, 18 relationships, 2 communities"


def test_ingest_missing_file_exits_1(backend: FakeBackend, tmp_path, capsys):
    assert run_cli(backend, "ingest", str(tmp_path / "missing.pdf")) == 1
    assert "error: File not found" in capsys.readouterr().err
    assert backend.calls == []


def test_communities_and_stats(backend: FakeBackend, capsys):
    assert run_cli(backend, "communities", "--limit", "3") == 0
    out = capsys.readouterr().out
    assert "• Victorian computing (3 entities)" in out and "Babbage and Lovelace." in out
    assert backend.calls[0].params == {"limit": "3"}

    assert run_cli(backend, "stats") == 0
    out = capsys.readouterr().out
    assert "Nodes: 4   Edges: 3   Isolated: 1" in out
    assert "PERSON" in out and "DESIGNED" in out and "Most connected:" in out


# ── failure contract ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("argv", [["status"], ["retrieve", "q"], ["stats"], ["communities"]])
def test_connection_error_exits_1_with_one_line(argv: list[str], capsys):
    backend = FakeBackend(unreachable=True)
    assert run_cli(backend, *argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: Cannot reach Synapse at http://synapse.test")
    assert "Traceback" not in captured.err


def test_backend_error_exits_1(backend: FakeBackend, capsys):
    backend.failures["/api/retrieve"] = (500, "Failed to query the knowledge graph.")
    assert run_cli(backend, "retrieve", "q") == 1
    assert capsys.readouterr().err.strip() == "error: HTTP 500: Failed to query the knowledge graph."


def test_mcp_subcommand_delegates_to_the_server(monkeypatch):
    forwarded: list[list[str]] = []
    monkeypatch.setattr("synapse_graphrag.mcp_server.main", lambda argv: forwarded.append(argv) or 0)
    assert main(["--url", "http://x:1", "mcp", "--transport", "streamable-http", "--port", "9000"]) == 0
    assert forwarded == [["--transport", "streamable-http", "--host", "127.0.0.1", "--port", "9000", "--url", "http://x:1"]]
