# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""``synapse-graphrag`` CLI through ``main([...])`` with a fake backend."""

from __future__ import annotations

import io
import json
import logging

import httpx
import pytest

from synapse_graphrag import __version__
from synapse_graphrag.cli import MCP_CLIENTS, evolution_estimate, evolution_minimum, main
from synapse_graphrag.client import SynapseClient
from tests.conftest import (
    BASE_URL,
    EVOLVE_REPORT,
    PROC_GRAPH,
    PROC_NAME,
    PROC_TEXT,
    RETRIEVAL,
    FakeBackend,
)


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
    assert "procedures" in out and "agent" in out and "evolve" in out


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


# ── procedures ───────────────────────────────────────────────────────────────
def test_procedures_lists_by_default(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures") == 0
    out = capsys.readouterr().out
    assert out.startswith(f"• {PROC_NAME}  v3 · score 0.556 · 5 nodes, 5 edges · updated 2026-09-24T12:00:00Z")
    assert "multi-hop question" in out
    assert run_cli(backend, "procedures", "list") == 0
    assert capsys.readouterr().out == out
    assert backend.paths() == ["/api/procedures", "/api/procedures"]


def test_procedures_list_when_empty(backend: FakeBackend, capsys):
    backend.procedures.clear()
    assert run_cli(backend, "procedures", "list") == 0
    out = capsys.readouterr().out
    assert "No procedural graphs yet" in out
    # Every bundled prior is seeded, including mcp-host, the MCP tools' default graph.
    assert "seeds its bundled priors (graphrag-navigator and mcp-host)" in out


def test_procedures_show_and_text(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures", "show", PROC_NAME) == 0
    out = capsys.readouterr().out
    assert f"{PROC_NAME}  v3 · score 0.556 · cycle policy forbid" in out
    assert "Tools: search_entities, neighbors, answer" in out
    assert "Nodes (5):" in out and "[Start] STATUS — A question has been received." in out
    assert "Transitions (5):" in out
    assert "[search_entities] —LEADS_TO→ [neighbors]  if the entity was found" in out
    assert "guidance: Read its relations to find the bridge entity." in out
    assert "pitfalls: Do not answer from the name alone." in out

    assert run_cli(backend, "procedures", "show", PROC_NAME, "--text") == 0
    assert capsys.readouterr().out.strip() == PROC_TEXT
    assert backend.calls[-1].params == {"format": "text"}


def test_procedures_unknown_graph_exits_1_with_one_line(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures", "show", "nope") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "error: HTTP 404: Procedural graph 'nope' not found."


def test_procedures_export_import_round_trip(backend: FakeBackend, capsys, tmp_path):
    assert run_cli(backend, "procedures", "export", PROC_NAME) == 0
    exported = json.loads(capsys.readouterr().out)  # stdout is pure JSON
    assert exported == {**PROC_GRAPH, "version": 3, "score": 0.5556}

    target = tmp_path / "nav.json"
    assert run_cli(backend, "procedures", "export", PROC_NAME, "-o", str(target)) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and f"Exported {PROC_NAME} v3 → {target}" in captured.err
    assert json.loads(target.read_text(encoding="utf-8")) == exported

    assert run_cli(backend, "procedures", "import", str(target)) == 0
    assert capsys.readouterr().out.strip() == f"Imported {PROC_NAME} as v4"
    put = backend.calls[-1]
    assert (put.method, put.path) == ("PUT", f"/api/procedures/{PROC_NAME}")
    assert put.json == PROC_GRAPH  # version/score are the backend's to assign

    assert run_cli(backend, "procedures", "import", str(target), "--name", "nav-copy") == 0
    assert capsys.readouterr().out.strip() == "Imported nav-copy as v1"
    assert backend.calls[-1].path == "/api/procedures/nav-copy"
    assert backend.calls[-1].json["name"] == "nav-copy"


def test_procedures_import_refusals(backend: FakeBackend, capsys, tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"name": "broken", "nodes": [], "edges": []}))
    assert run_cli(backend, "procedures", "import", str(invalid)) == 1
    assert capsys.readouterr().err.strip() == (
        "error: HTTP 422: Invalid procedural graph: missing Start node; no terminal node"
    )
    calls = len(backend.calls)

    nameless = tmp_path / "nameless.json"
    nameless.write_text(json.dumps({"nodes": [], "edges": []}))
    not_json = tmp_path / "graph.json"
    not_json.write_text("{nope")
    a_list = tmp_path / "list.json"
    a_list.write_text("[]")
    for path, message in [
        (nameless, 'has no "name"; pass --name'),
        (not_json, "is not valid JSON"),
        (a_list, "must hold one JSON object"),
        (tmp_path / "missing.json", "Cannot read"),
    ]:
        assert run_cli(backend, "procedures", "import", str(path)) == 1
        err = capsys.readouterr().err
        assert message in err and len(err.strip().splitlines()) == 1
    assert len(backend.calls) == calls


def test_procedures_versions_and_rollback(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures", "versions", PROC_NAME) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("v3   score 0.556 · accepted · 2026-09-24T12:00:00Z · evolution round 2")
    assert lines[0].endswith("(+1 node, +2 edges, ~1 edge)")
    assert lines[1].endswith("(no structural change)")
    assert lines[2].startswith("v1   score — · accepted") and lines[2].endswith("seeded from the expert prior")

    assert run_cli(backend, "procedures", "rollback", PROC_NAME, "2") == 0
    assert capsys.readouterr().out.strip() == (
        f"Rolled {PROC_NAME} back to v2 — saved as v4 (history is kept)"
    )
    assert backend.calls[-1].json == {"version": 2}
    with pytest.raises(SystemExit):
        run_cli(backend, "procedures", "rollback", PROC_NAME, "0")


def test_procedures_guide(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures", "guide", PROC_NAME, "--query", "Who?") == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Active Cognitive Node: [Start] (Type: ACTION)"
    assert f"{PROC_NAME} v3 · node Start (start) · scope local" in captured.err
    assert "0 LLM calls" in captured.err and "Next: search_entities" in captured.err
    # raw is sent explicitly: the CLI never lets a backend default spend an LLM call
    assert backend.calls[0].json == {"query": "Who?", "trajectory": [], "mode": "raw"}

    assert run_cli(
        backend, "procedures", "guide", PROC_NAME, "--query", "Who?",
        "--last-action", "neighbors", "--observation", "Ada Lovelace", "--mode", "generative",
    ) == 0
    captured = capsys.readouterr()
    assert "Guidance:\nRead the entity's relations next." in captured.out
    assert "node neighbors (exact)" in captured.err and "1 LLM calls" in captured.err
    assert backend.calls[1].json["trajectory"] == [{"action": "neighbors", "observation": "Ada Lovelace"}]
    assert backend.calls[1].json["mode"] == "generative"

    assert run_cli(backend, "procedures", "guide", PROC_NAME, "--query", "Who?", "--json") == 0
    assert json.loads(capsys.readouterr().out)["active_node"] == "Start"


def test_procedures_guide_observation_needs_an_action(backend: FakeBackend, capsys):
    assert run_cli(backend, "procedures", "guide", PROC_NAME, "--query", "q", "--observation", "x") == 1
    assert "pass --last-action too" in capsys.readouterr().err
    assert backend.calls == []


# ── agent ────────────────────────────────────────────────────────────────────
def test_agent_prints_the_trace_then_the_answer(backend: FakeBackend, capsys):
    assert run_cli(backend, "agent", "Who wrote programs for the Analytical Engine?") == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Ada Lovelace"  # stdout carries the answer alone
    err = captured.err
    assert f"Navigator · procedural graph {PROC_NAME} v3" in err
    assert "Step 1  [PG: start]" in err and "Step 2  [PG: exact]" in err
    assert "Thought: Find the engine first." in err
    assert 'Action: search_entities(query="Analytical Engine")' in err
    assert 'Action: answer(text="Ada Lovelace")' in err
    first_observation = next(line for line in err.splitlines() if "Observation: Analytical Engine" in line)
    assert first_observation.endswith("…") and len(first_observation) <= len("  Observation: ") + 240
    assert "Usage: 3 LLM calls (0 for guidance) · 2412 in / 188 out tokens · 4.2s" in err
    assert "(estimated)" not in err
    assert backend.calls[0].json == {"query": "Who wrote programs for the Analytical Engine?", "record": False}


def test_agent_flags_reach_the_body(backend: FakeBackend, capsys):
    assert run_cli(backend, "agent", "q", "--no-graph", "--guidance", "none", "--max-steps", "4", "--record") == 0
    err = capsys.readouterr().err
    assert "Navigator · no procedural graph" in err and "[PG:" not in err
    # Nothing to record a graph-less run against: say so rather than claim it was stored.
    assert "Trajectory not recorded: the run used no procedural graph" in err
    assert backend.calls[0].json == {"query": "q", "record": True, "graph": None, "guidance": "none", "max_steps": 4}

    assert run_cli(backend, "agent", "q", "--graph", PROC_NAME, "--guidance", "generative", "--record") == 0
    assert "Trajectory recorded (unscored)." in capsys.readouterr().err
    assert backend.calls[1].json == {"query": "q", "record": True, "graph": PROC_NAME, "guidance": "generative"}

    with pytest.raises(SystemExit):
        run_cli(backend, "agent", "q", "--graph", PROC_NAME, "--no-graph")
    with pytest.raises(SystemExit):
        run_cli(backend, "agent", "q", "--max-steps", "21")
    assert len(backend.calls) == 2


def test_agent_json_and_no_answer(backend: FakeBackend, capsys, monkeypatch):
    assert run_cli(backend, "agent", "q", "--json") == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "Ada Lovelace"

    import tests.conftest as fixtures

    stalled = {**fixtures.AGENT_RESULT, "answer": None, "stopped": "max_steps", "parse_failures": 2}
    stalled["usage"] = {**stalled["usage"], "estimated": True}
    monkeypatch.setattr(fixtures, "AGENT_RESULT", stalled)
    assert run_cli(backend, "agent", "q") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No answer: the navigator stopped (max_steps) after 3 steps." in captured.err
    assert "(estimated)" in captured.err and "2 parse failures" in captured.err
    assert run_cli(backend, "agent", "q", "--json") == 1
    assert json.loads(capsys.readouterr().out)["stopped"] == "max_steps"


def test_agent_backend_without_llm_key_is_one_line(backend: FakeBackend, capsys):
    backend.failures["/api/agent/ask"] = (503, "No LLM provider configured: set OPENAI_API_KEY.")
    assert run_cli(backend, "agent", "q") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "error: HTTP 503: No LLM provider configured: set OPENAI_API_KEY."


# ── evolve ───────────────────────────────────────────────────────────────────
@pytest.fixture
def qa_files(tmp_path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps([{"question": f"train q{i}?", "answer": f"a{i}"} for i in range(4)]))
    val = tmp_path / "val.jsonl"
    val.write_text("\n".join(json.dumps({"question": f"val q{i}?", "answer": f"b{i}"}) for i in range(3)) + "\n\n")
    return str(train), str(val)


def test_evolution_estimate_is_an_upper_bound_formula():
    # (rounds × (batch + val) + val) rollouts × max_steps calls, + one refiner call per round.
    assert evolution_estimate(3, 10, 9, "raw") == {"rollouts": 66, "per_rollout": 8, "refiner_calls": 3, "llm_calls": 531}
    assert evolution_estimate(3, 10, 9, "generative")["llm_calls"] == 66 * 16 + 3
    assert evolution_estimate(1, 1, 1, "raw", max_steps=2)["llm_calls"] == 3 * 2 + 1


def test_evolve_refuses_without_consent_on_a_non_interactive_stdin(backend: FakeBackend, capsys, monkeypatch, qa_files):
    train, val = qa_files
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # piped "y" is not consent
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val) == 1
    captured = capsys.readouterr()
    assert backend.calls == []
    assert captured.out == ""
    assert "Upper bound ≈ 42 agent runs (3 rounds × (10 train + 3 val) + 3 baseline) × 8 calls + 3 refinements = 339 calls" in captured.err
    assert "Hard cap: the backend stops cleanly at 339 calls, the bound above" in captured.err
    assert "steps of 0.333" in captured.err and "search trace" in captured.err
    assert captured.err.strip().splitlines()[-1] == (
        "error: refusing to start a paid evolution run without consent: re-run with --yes, "
        "or from an interactive terminal."
    )


def test_evolve_interactive_prompt(backend: FakeBackend, capsys, monkeypatch, qa_files):
    train, val = qa_files
    monkeypatch.setattr("synapse_graphrag.cli._stdin_is_interactive", lambda: True)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val) == 1
    err = capsys.readouterr().err
    assert "Start it? [y/N] " in err and "Cancelled; nothing was sent to the backend." in err
    assert backend.calls == []

    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val) == 0
    assert backend.paths() == [f"/api/procedures/{PROC_NAME}/evolve", "/api/procedures/evolve/job_000009/events"]


def test_evolve_with_yes_follows_progress_and_prints_the_report(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    assert run_cli(
        backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--rounds", "2",
        "--batch-size", "4", "--mode", "scratch", "--replace", "--metric", "em",
        "--guidance", "generative", "--max-llm-calls", "250", "--yes",
    ) == 0
    captured = capsys.readouterr()
    # Scratch mode first reads the stored graph (free), then starts the job.
    assert backend.paths() == [
        f"/api/procedures/{PROC_NAME}",
        f"/api/procedures/{PROC_NAME}/evolve",
        "/api/procedures/evolve/job_000009/events",
    ]
    body = backend.calls[1].json
    assert body == {
        "train": [{"question": f"train q{i}?", "answer": f"a{i}"} for i in range(4)],
        "val": [{"question": f"val q{i}?", "answer": f"b{i}"} for i in range(3)],
        "mode": "scratch",
        "metric": "em",
        "guidance": "generative",
        "rounds": 2,
        "batch_size": 4,
        "max_llm_calls": 250,
        "replace": True,
    }
    err = captured.err
    # The stored graph's one evaluation is in the upper bound: (2×(4+3) + 3 + 3) × 16 + 2.
    assert "Upper bound ≈ 20 agent runs (2 rounds × (4 train + 3 val) + 3 baseline + 3 stored graph) × 16 calls + 2 refinements = 322 calls" in err
    assert "stops cleanly at 250 calls" in err
    assert "mode scratch --replace: starts from an empty Start→End skeleton; the stored v3 (score 0.556) is scored once" in err
    assert "Evolution job job_000009 started" in err
    assert "[evolve] round 0 baseline processed=3 total=3 llm_calls=12" in err
    assert "[evolve] round 0 baseline score=0.444" in err
    assert "[evolve] round 1 rollout processed=10 total=10 llm_calls=52" in err
    assert "[evolve] round 1 refine llm_calls=52" in err
    assert "[evolve] round 1 accepted score=0.556 previous_score=0.444 version=4 change=+1 node, +1 edge" in err
    assert "[evolve] round 2 rejected reason=structural diagnostics=1\n" in err  # the edits dict is not dumped
    out = captured.out
    assert f"Evolved {PROC_NAME} (static): 2 rounds run, stopped: completed · 212 LLM calls" in out
    assert "baseline 0.444 → final 0.556 (v4)" in out
    assert "round 1: train 0.500 · candidate 0.556 · accepted [+1 node, +1 edge]" in out
    assert "round 2: train 0.450 · candidate — · rejected — structural (node 'x' cannot reach a terminal)" in out
    assert "Effect floor: 0.111" in out


def test_evolve_defaults_are_sent_explicitly(backend: FakeBackend, qa_files):
    train, val = qa_files
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--yes") == 0
    body = backend.calls[0].json
    assert (body["rounds"], body["batch_size"], body["mode"], body["metric"], body["guidance"]) == (3, 10, "static", "f1", "raw")
    # The consented upper bound is sent as the cap, so a backend configured with a
    # bigger AGENT_MAX_STEPS or EVOLUTION_MAX_LLM_CALLS cannot spend past it.
    expected = evolution_estimate(3, 10, 3, "raw")["llm_calls"]
    assert body["max_llm_calls"] == expected == (3 * (10 + 3) + 3) * 8 + 3


def test_evolve_prints_nothing_saved_when_no_version_was_kept(backend: FakeBackend, capsys, qa_files):
    backend.evolve_events = [
        {"type": "done", "data": {**EVOLVE_REPORT, "mode": "scratch", "final_version": None}}
    ]
    train, val = qa_files
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--mode", "scratch", "--replace", "--yes") == 0
    out = capsys.readouterr().out
    assert "(nothing saved)" in out and "vNone" not in out


def test_evolve_report_names_the_stored_graphs_score(backend: FakeBackend, capsys, qa_files):
    report = {**EVOLVE_REPORT, "mode": "scratch", "final_version": None, "stored_score": 0.8889, "stored_version": 3}
    backend.evolve_events = [{"type": "done", "data": report}]
    train, val = qa_files
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--mode", "scratch", "--replace", "--yes") == 0
    assert "stored v3 scored 0.889: the score a candidate had to reach to replace it" in capsys.readouterr().out


# ── evolve: refusals before consent ─────────────────────────────────────────
def test_evolution_minimum_is_the_backends_one_round_formula():
    # (|val| baseline [+ |val| stored] + min(batch, |train|) + |val|) × worst case + 1 refiner,
    # the numbers of the backend's own budget tests (test_procedural_evolution.py).
    assert evolution_minimum(2, 4, 3, "raw", max_steps=2) == 17
    assert evolution_minimum(2, 4, 3, "generative", max_steps=2) == 33
    assert evolution_minimum(2, 4, 3, "raw", max_steps=2, stored_eval=True) == 23
    assert evolution_minimum(99, 4, 3, "raw", max_steps=2) == 21  # the batch is capped by train
    # The default cap (the consented upper bound) always runs at least one round.
    for stored_eval in (False, True):
        for guidance in ("raw", "generative"):
            bound = evolution_estimate(1, 10, 3, guidance, stored_eval=stored_eval)["llm_calls"]
            assert bound >= evolution_minimum(10, 4, 3, guidance, stored_eval=stored_eval)


def test_evolve_refuses_a_cap_that_cannot_run_one_round(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    # The baseline (3 × 8 = 24) fits in 30; a baseline plus one round does not.
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--max-llm-calls", "30", "--yes") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip().splitlines() == [
        "error: --max-llm-calls 30 cannot pay for the baseline and one full round, whose worst "
        "case is (3 baseline + 4 train + 3 val) agent runs × 8 calls + 1 refinement; the smallest "
        "--max-llm-calls that runs one round is 81 (assuming AGENT_MAX_STEPS=8)."
    ]
    assert backend.calls == []  # nothing sent, and no consent asked for

    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--max-llm-calls", "81", "--yes") == 0
    assert backend.calls[0].json["max_llm_calls"] == 81


def test_evolve_refuses_a_cap_above_the_backend_limit_before_consent(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    argv = ["evolve", PROC_NAME, "--train", train, "--val", val, "--yes"]
    # The backend's EvolveRequest.max_llm_calls is le=100_000: a larger cap would be
    # promised to the user, then answered with a 422.
    with pytest.raises(SystemExit) as info:
        run_cli(backend, *argv, "--max-llm-calls", "100001")
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert "must be 1-100000, got 100001" in captured.err
    assert "Hard cap" not in captured.err  # no cap was promised
    assert backend.calls == []
    assert run_cli(backend, *argv, "--max-llm-calls", "100000") == 0
    assert backend.calls[0].json["max_llm_calls"] == 100_000


def test_evolve_scratch_refuses_to_overwrite_a_stored_graph_without_replace(backend: FakeBackend, capsys, qa_files, monkeypatch):
    train, val = qa_files
    monkeypatch.setattr("synapse_graphrag.cli._stdin_is_interactive", lambda: True)
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--mode", "scratch") == 1
    captured = capsys.readouterr()
    assert captured.err.strip().splitlines() == [
        f"error: procedural graph '{PROC_NAME}' already exists (v3, score 0.556); --mode scratch "
        "would replace it. Re-run with --replace to score it once on the validation set and "
        "overwrite it only with a candidate at least as good, or choose another NAME."
    ]
    assert "Start it?" not in captured.err  # refused before consent was asked for
    assert backend.paths() == [f"/api/procedures/{PROC_NAME}"]  # a read, never the POST


def test_evolve_replace_counts_the_stored_graph_in_the_minimum(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    argv = ["evolve", PROC_NAME, "--train", train, "--val", val, "--mode", "scratch", "--replace", "--yes"]
    assert run_cli(backend, *argv, "--max-llm-calls", "104") == 1
    err = capsys.readouterr().err
    assert "(3 baseline + 3 stored graph + 4 train + 3 val) agent runs" in err
    assert "the smallest --max-llm-calls that runs one round is 105" in err
    assert backend.paths("POST") == []
    assert run_cli(backend, *argv, "--max-llm-calls", "105") == 0
    assert backend.paths("POST") == [f"/api/procedures/{PROC_NAME}/evolve"]


def test_evolve_replace_needs_scratch_mode(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--replace", "--yes") == 1
    assert capsys.readouterr().err.strip() == (
        "error: --replace only applies to --mode scratch (static mode builds on the stored graph)."
    )
    assert backend.calls == []


def _with_fresh_graph(backend: FakeBackend, name: str):
    """A client factory whose backend also starts evolution jobs for a graph it has not stored
    (the router allows that in scratch mode; the shared fake only knows stored graphs)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == f"/api/procedures/{name}/evolve":
            response = await backend.handle(request)  # records the call (and any failure)
            if response.status_code == 404:
                return httpx.Response(200, json={"job_id": "job_000009", "status": "processing"})
            return response
        return await backend.handle(request)

    def factory(**kwargs):
        kwargs["base_url"] = kwargs.get("base_url") or BASE_URL
        return SynapseClient(transport=httpx.MockTransport(handler), **kwargs)

    return factory


def test_evolve_scratch_on_a_new_name_needs_no_replace(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files
    code = main(["evolve", "fresh", "--train", train, "--val", val, "--mode", "scratch", "--yes"], client_factory=_with_fresh_graph(backend, "fresh"))
    assert code == 0
    err = capsys.readouterr().err
    assert "nothing is saved as 'fresh' unless a candidate is accepted" in err
    assert "stored graph" not in err
    assert backend.paths() == ["/api/procedures/fresh", "/api/procedures/fresh/evolve", "/api/procedures/evolve/job_000009/events"]
    assert "replace" not in backend.calls[1].json


def test_evolve_backend_refusals_are_one_line(backend: FakeBackend, capsys, qa_files):
    """The backend has the last word (another AGENT_MAX_STEPS, a graph saved meanwhile)."""
    train, val = qa_files
    factory = _with_fresh_graph(backend, "fresh")
    argv = ["evolve", "fresh", "--train", train, "--val", val, "--mode", "scratch", "--yes"]
    backend.failures["/api/procedures/fresh/evolve"] = (409, "Procedural graph 'fresh' already exists (v1); mode=scratch would replace it.")
    assert main(argv, client_factory=factory) == 1
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        "error: HTTP 409: Procedural graph 'fresh' already exists (v1); mode=scratch would replace it."
    )
    backend.failures["/api/procedures/fresh/evolve"] = (
        422,
        {
            "message": "max_llm_calls cannot pay for the baseline and one full round",
            "diagnostics": ["max_llm_calls=105 cannot pay …; set max_llm_calls to at least 161"],
            "minimum_max_llm_calls": 161,
        },
    )
    assert main(argv, client_factory=factory) == 1
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith("error: HTTP 422: max_llm_calls cannot pay for the baseline and one full round: ")
    assert last.endswith("set max_llm_calls to at least 161")
    assert backend.paths("GET").count("/api/procedures/evolve/job_000009/events") == 0


# ── long-running requests ───────────────────────────────────────────────────
def test_agent_and_evolve_wait_at_least_ten_minutes(backend: FakeBackend, monkeypatch, qa_files):
    timeouts: list[float] = []

    def factory(**kwargs):
        client = backend.client(**kwargs)
        timeouts.append(client.timeout)
        return client

    train, val = qa_files
    assert main(["agent", "q"], client_factory=factory) == 0
    assert main(["evolve", PROC_NAME, "--train", train, "--val", val, "--yes"], client_factory=factory) == 0
    assert main(["retrieve", "q"], client_factory=factory) == 0
    # SYNAPSE_TIMEOUT unset (120 s): agent runs are raised to 600 s; other commands keep
    # the fake's default (5 s), i.e. they pass no timeout of their own.
    assert timeouts == [600.0, 600.0, 5.0]

    timeouts.clear()
    monkeypatch.setenv("SYNAPSE_TIMEOUT", "900")
    assert main(["agent", "q"], client_factory=factory) == 0
    assert main(["evolve", PROC_NAME, "--train", train, "--val", val, "--mode", "scratch", "--replace", "--yes"], client_factory=factory) == 0
    assert timeouts == [900.0, 900.0, 900.0]  # scratch's existence check included


@pytest.mark.parametrize("command", ["agent", "evolve"])
def test_agent_and_evolve_help_warn_that_the_backend_keeps_spending(command: str, capsys):
    with pytest.raises(SystemExit) as info:
        main([command, "--help"])
    assert info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())  # argparse wraps at the terminal width
    assert "max($SYNAPSE_TIMEOUT, 600) seconds" in text
    assert "the backend keeps working (and spending" in text


def test_evolve_split_selection_and_leak_warning(backend: FakeBackend, capsys, tmp_path):
    mixed = tmp_path / "demo_qa.json"
    mixed.write_text(json.dumps([
        {"question": "q1?", "answer": "a1", "split": "train"},
        {"question": "q2?", "answer": "a2", "split": "val"},
        {"question": "q3?", "answer": "a3", "split": "test"},
    ]))
    assert run_cli(backend, "evolve", PROC_NAME, "--train", str(mixed), "--val", str(mixed), "--yes") == 1
    assert "mixes splits (test, train, val); choose one with --train-split" in capsys.readouterr().err
    assert run_cli(backend, "evolve", PROC_NAME, "--train", str(mixed), "--train-split", "dev", "--val", str(mixed), "--val-split", "val", "--yes") == 1
    assert "has no items in split 'dev' (splits: test, train, val)" in capsys.readouterr().err
    assert backend.calls == []

    assert run_cli(backend, "evolve", PROC_NAME, "--train", str(mixed), "--train-split", "train", "--val", str(mixed), "--val-split", "val", "--yes") == 0
    assert backend.calls[0].json["train"] == [{"question": "q1?", "answer": "a1"}]
    assert backend.calls[0].json["val"] == [{"question": "q2?", "answer": "a2"}]
    assert "also appear in train" not in capsys.readouterr().err

    assert run_cli(backend, "evolve", PROC_NAME, "--train", str(mixed), "--train-split", "val", "--val", str(mixed), "--val-split", "val", "--yes") == 0
    assert "warning: 1 validation question(s) also appear in train" in capsys.readouterr().err


def test_evolve_rejects_malformed_files_before_anything_is_sent(backend: FakeBackend, capsys, tmp_path, qa_files):
    train, _ = qa_files
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"question": "q", "answer": "a"}\n{oops}\n')
    empty_answer = tmp_path / "empty.json"
    empty_answer.write_text(json.dumps([{"question": "q", "answer": ""}]))
    for path, message in [
        (broken, f"error: {broken}:2 is not valid JSON"),
        (empty_answer, f"error: {empty_answer}[0] needs a non-empty 'question' and 'answer'"),
        (tmp_path / "missing.json", "error: Cannot read"),
    ]:
        assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", str(path), "--yes") == 1
        assert capsys.readouterr().err.strip().startswith(message)
    with pytest.raises(SystemExit):
        run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", train, "--rounds", "21")
    assert backend.calls == []


def test_evolve_error_event_exits_1(backend: FakeBackend, capsys, qa_files):
    backend.evolve_events = [{"type": "error", "data": "LLM provider not configured"}]
    train, val = qa_files
    assert run_cli(backend, "evolve", PROC_NAME, "--train", train, "--val", val, "--yes") == 1
    assert capsys.readouterr().err.strip().endswith("error: LLM provider not configured")


def test_evolve_interrupt_says_the_job_keeps_running(backend: FakeBackend, capsys, qa_files):
    train, val = qa_files

    class Interrupting:
        def __init__(self, **kwargs):
            self.client = backend.client(**kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            await self.client.aclose()

        async def evolve_procedure(self, name, train, val, *, on_progress, **kwargs):
            on_progress({"type": "accepted", "job_id": "job_000009", "status": "processing"})
            raise KeyboardInterrupt

    assert main(["evolve", PROC_NAME, "--train", train, "--val", val, "--yes"], client_factory=Interrupting) == 130
    assert "interrupted: job job_000009 keeps running on the backend" in capsys.readouterr().err
