# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The Synapse Lab through the client and ``synapse-graphrag lab`` against a fake backend.

The fake speaks routers/lab.py's wire format (see ``conftest.py``). What matters
most here is money: a paid run needs ``--max-usd``, the estimate is printed
before anything starts, an over-cap estimate is refused before the run is even
requested, and nothing starts without ``--yes`` or an interactive "y".
"""

from __future__ import annotations

import io
import json
import math

import pytest

from synapse_graphrag.cli import main
from synapse_graphrag.client import SynapseError, lab_request, parse_budget, parse_budgets
from tests.conftest import (
    LAB_ABORTED_ID,
    LAB_ARMS,
    LAB_BATCH_ID,
    LAB_MODELS,
    LAB_RUN_ID,
    FakeBackend,
)

NEW_RUN = "20260925-120000-d4e5f6"


def run_cli(backend: FakeBackend, *argv: str) -> int:
    return main(list(argv), client_factory=backend.client)


def posted(backend: FakeBackend, path: str) -> list[dict]:
    return [c.json for c in backend.calls if c.method == "POST" and c.path == path]


# ── Request bodies ───────────────────────────────────────────────────────────
class TestLabRequest:
    def test_defaults_are_left_to_the_backend(self):
        assert lab_request() == {"dataset": "demo", "mode": "retrieve"}

    def test_every_field_is_passed_through(self):
        body = lab_request(
            dataset="qa-file:mine.jsonl", arms=["bm25", "bm25", "ppr"], budgets=["500", "2k", "default", 500],
            mode="batch", max_usd=2.5, reader_model="gpt-4o-mini", split="test", n=50, k=6,
            passage_k=4, seed=7, order="ascending", reasoning_allowance=128, max_concurrency=2,
            extra_cells=[("synapse_d", "default"), ("ppr", 2000)], ingest={"paragraphs": 10},
            ingest_usd=0.3, measured_run_id="r1", run_id="mine", offset=0, sample_seed=1,
        )
        assert body["arms"] == ["bm25", "ppr"]
        assert body["budgets"] == [500, 2000, None]
        assert body["extra_cells"] == [{"arm": "synapse_d", "budget": None}, {"arm": "ppr", "budget": 2000}]
        assert body["max_usd"] == 2.5 and body["ingest_usd"] == 0.3 and body["offset"] == 0
        assert body["ingest"] == {"paragraphs": 10} and body["measured_run_id"] == "r1"

    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"mode": "free"}, "mode must be one of"),
            ({"dataset": " "}, "dataset is required"),
            ({"arms": [" "]}, "at least one arm"),
            ({"budgets": []}, "at least one budget"),
            ({"budgets": ["lots"]}, "invalid budget"),
            ({"budgets": [0]}, "must be positive"),
            ({"max_usd": -1}, ">= 0"),
            ({"max_usd": math.nan}, ">= 0"),
            ({"max_usd": "ten"}, "must be a number"),
            ({"order": "random"}, "order must be one of"),
        ],
    )
    def test_bad_values_never_reach_the_backend(self, kwargs, needle):
        with pytest.raises(SynapseError, match=needle):
            lab_request(**kwargs)

    def test_budget_parsing(self):
        assert parse_budget(None) is None and parse_budget("default") is None
        assert parse_budget("1.5k") == 1500 and parse_budget(" 8K ") == 8000 and parse_budget(700) == 700
        assert parse_budgets("500, 1k,default") == [500, 1000, None]
        with pytest.raises(SynapseError):
            parse_budget(True)
        with pytest.raises(SynapseError, match="at least one budget"):
            parse_budgets(" , ")


# ── Client ───────────────────────────────────────────────────────────────────
class TestClient:
    async def test_catalogue_calls(self, backend: FakeBackend):
        async with backend.client() as client:
            assert await client.lab_arms() == LAB_ARMS
            assert await client.lab_models() == LAB_MODELS
            datasets = await client.lab_datasets()
        assert datasets["datasets"][1]["id"] == "qa-file:mine.jsonl"
        assert backend.paths() == ["/api/lab/arms", "/api/lab/models", "/api/lab/datasets"]

    async def test_estimate_posts_the_body_as_is(self, backend: FakeBackend):
        body = lab_request(arms=["bm25"], budgets=[500], mode="realtime", max_usd=1)
        async with backend.client() as client:
            est = await client.lab_estimate(body)
        assert posted(backend, "/api/lab/estimate") == [body]
        assert est["total_upper_usd"] == 0.002 and est["refuse"] is False

    async def test_run_starts_then_follows_to_the_summary(self, backend: FakeBackend):
        seen: list[dict] = []
        async with backend.client() as client:
            result = await client.lab_run(lab_request(arms=["bm25"]), on_progress=seen.append)
        assert result["run_id"] == NEW_RUN and result["job_id"] == "job_000010"
        assert result["status"] == "done"
        assert seen[0]["type"] == "accepted" and seen[0]["run_id"] == NEW_RUN
        assert {e["type"] for e in seen[1:]} == {"phase", "progress"}
        assert backend.paths() == ["/api/lab/runs", f"/api/lab/runs/{NEW_RUN}/events"]

    async def test_an_over_cap_run_is_a_422_carrying_the_estimate(self, backend: FakeBackend):
        body = lab_request(arms=["bm25"], budgets=[500], mode="realtime", max_usd=0.0001)
        async with backend.client() as client:
            with pytest.raises(SynapseError) as info:
                await client.lab_run(body)
        assert info.value.status == 422
        assert "breaks the spend cap" in str(info.value) and "exceeds the cap" in str(info.value)
        assert info.value.payload["estimate"]["refuse"] is True
        assert backend.paths() == ["/api/lab/runs"]  # never followed

    async def test_an_error_event_is_a_synapse_error(self, backend: FakeBackend):
        backend.lab_events = [{"type": "error", "data": "RuntimeError: disk full"}]
        async with backend.client() as client:
            with pytest.raises(SynapseError, match="disk full"):
                await client.lab_events(LAB_RUN_ID)

    async def test_show_and_resume(self, backend: FakeBackend):
        async with backend.client() as client:
            shown = await client.lab_show(LAB_RUN_ID, offset=1, limit=2, arm="bm25", budget=500)
            resumed = await client.lab_resume(LAB_ABORTED_ID, max_usd=1, retry_failed=True)
            with pytest.raises(SynapseError, match="Lab run id is required"):
                await client.lab_show(" ")
            with pytest.raises(SynapseError, match=">= 0"):
                await client.lab_resume(LAB_ABORTED_ID, max_usd=-2)
        get = backend.calls[0]
        assert get.params == {"offset": "1", "limit": "2", "arm": "bm25", "budget": "500"}
        assert shown["rows"]["total"] == 3 and len(shown["rows"]["rows"]) == 2
        assert posted(backend, f"/api/lab/runs/{LAB_ABORTED_ID}/resume") == [
            {"retry_failed": True, "max_usd": 1.0}
        ]
        assert resumed["status"] == "done" and resumed["job_id"] == "job_000011"

    async def test_show_budget_is_spelled_like_run_budgets(self, backend: FakeBackend):
        async with backend.client() as client:
            await client.lab_show(LAB_RUN_ID, budget="2k")
            await client.lab_show(LAB_RUN_ID, budget="default")
            await client.lab_show(LAB_RUN_ID, budget=" 8K ")
            with pytest.raises(SynapseError, match="invalid budget"):
                await client.lab_show(LAB_RUN_ID, budget="lots")
        assert [c.params["budget"] for c in backend.calls] == ["2000", "default", "8000"]

    async def test_run_ids_are_one_path_segment(self, backend: FakeBackend):
        async with backend.client() as client:
            with pytest.raises(SynapseError, match="404"):
                await client.lab_show("a/../b")
        assert backend.calls[0].raw_path == "/api/lab/runs/a%2F..%2Fb"

    async def test_upload(self, backend: FakeBackend, tmp_path):
        path = tmp_path / "mine.jsonl"
        path.write_text('{"question": "q", "answer": "a"}\n')
        async with backend.client() as client:
            stored = await client.lab_upload_qa_file(path, name="renamed.jsonl", replace=True)
            with pytest.raises(SynapseError, match="File not found"):
                await client.lab_upload_qa_file(tmp_path / "ghost.jsonl")
            (tmp_path / "x.csv").write_text("q,a")
            with pytest.raises(SynapseError, match=".json or .jsonl"):
                await client.lab_upload_qa_file(tmp_path / "x.csv")
        assert stored["id"] == "qa-file:mine.jsonl"
        (body,) = backend.lab_uploads
        assert b'filename="mine.jsonl"' in body and b'{"question": "q", "answer": "a"}' in body
        assert b'name="name"' in body and b"renamed.jsonl" in body and b'name="replace"' in body


# ── CLI: catalogue ───────────────────────────────────────────────────────────
def test_help_lists_lab(capsys):
    assert main([]) == 0
    assert "lab" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["lab", "--help"])
    out = capsys.readouterr().out
    for action in ("arms", "models", "datasets", "upload", "estimate", "run", "runs", "show", "resume"):
        assert action in out
    assert "free" in out and "--yes" in out


def test_lab_defaults_to_the_arms_grouped_by_family(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab") == 0
    out = capsys.readouterr().out
    floors, passages, graphs = (out.index(t) for t in ("Evidence floors", "Passage baselines", "Graph arms"))
    assert floors < passages < graphs
    assert "0 retrieval LLM calls" in out and "needs the extracted graph" in out
    assert "https://arxiv.org/abs/2502.14902" in out
    assert "retrieve (free, the default)" in out


def test_lab_arms_json(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "arms", "--json") == 0
    assert json.loads(capsys.readouterr().out) == LAB_ARMS


def test_lab_models_and_datasets(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "models") == 0
    captured = capsys.readouterr()
    nano = next(line for line in captured.out.splitlines() if line.startswith("gpt-5-nano"))
    assert "yes" in nano and "0.050" in nano and "2026-09-24" in nano
    assert "hand-recorded" in captured.err
    assert run_cli(backend, "lab", "datasets") == 0
    out = capsys.readouterr().out
    assert "• demo" in out and "splits test 9, train 12, val 9" in out
    assert "• qa-file:mine.jsonl" in out
    assert "✗ hotpotqa: no HotpotQA paragraph is ingested" in out


def test_lab_upload(backend: FakeBackend, capsys, tmp_path):
    path = tmp_path / "mine.jsonl"
    path.write_text('{"question": "q", "answer": "a"}\n')
    assert run_cli(backend, "lab", "upload", str(path)) == 0
    captured = capsys.readouterr()
    assert "Stored: mine.jsonl — 3 questions · splits test 2, train 1" in captured.out
    assert "--dataset qa-file:mine.jsonl" in captured.err
    taken = tmp_path / "taken.jsonl"
    taken.write_text('{"question": "q", "answer": "a"}\n')
    assert run_cli(backend, "lab", "upload", str(taken)) == 1
    assert capsys.readouterr().err.startswith("error: HTTP 409: A different QA file")
    assert run_cli(backend, "lab", "upload", str(tmp_path / "ghost.json")) == 1
    assert "File not found" in capsys.readouterr().err


# ── CLI: estimate ────────────────────────────────────────────────────────────
def test_estimate_retrieve_is_free(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "estimate", "--arms", "bm25,synapse_lean", "--budgets", "500,2k") == 0
    out = capsys.readouterr().out
    assert "Retrieve-only: 4 (arm, budget) cells × 9 questions; no model is called — $0." in out
    assert "Total: $0 point · $0 upper bound · cap none" in out
    (body,) = posted(backend, "/api/lab/estimate")
    assert body == {"dataset": "demo", "mode": "retrieve", "arms": ["bm25", "synapse_lean"], "budgets": [500, 2000]}


def test_estimate_paid_prints_the_table_and_a_refusal_exits_1(backend: FakeBackend, capsys):
    argv = ["lab", "estimate", "--arms", "bm25", "--budgets", "500,default", "--mode", "realtime"]
    assert run_cli(backend, *argv, "--max-usd", "1") == 0
    out = capsys.readouterr().out
    assert "ARM" in out and "UPPER" in out and "reasoning model, 512 reasoning tokens allowed" in out
    assert "Total: $0.0020 point · $0.0040 upper bound · cap $1.00" in out
    assert run_cli(backend, *argv, "--max-usd", "0.001") == 1
    assert "REFUSED: upper bound" in capsys.readouterr().out


def test_an_uploaded_dataset_is_named_by_its_file(backend: FakeBackend, capsys, monkeypatch):
    from tests import conftest

    real = conftest.lab_estimate_payload

    def estimate(body):
        return {**real(body), "dataset": {"name": "qa-file", "split": None, "n": 3, "source": "mine.jsonl"}}

    monkeypatch.setattr(conftest, "lab_estimate_payload", estimate)
    assert run_cli(backend, "lab", "estimate", "--dataset", "qa-file:mine.jsonl") == 0
    assert "Estimate · qa-file:mine.jsonl (all, n=3)" in capsys.readouterr().out


def test_estimate_json_and_flags_reach_the_body(backend: FakeBackend, capsys):
    assert run_cli(
        backend, "lab", "estimate", "--json", "--dataset", "hotpotqa", "--n", "100", "--offset", "100",
        "--sample-seed", "20260924", "--extra-cell", "synapse_lean@default", "--ingest",
        "--ingest-realtime", "--measured-from", "r1", "--order", "ascending", "--k", "6",
    ) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["measured_from"]["run_id"] == "r1"
    (body,) = posted(backend, "/api/lab/estimate")
    assert body["dataset"] == "hotpotqa" and body["n"] == 100 and body["offset"] == 100
    assert body["sample_seed"] == 20260924 and body["order"] == "ascending" and body["k"] == 6
    assert body["extra_cells"] == [{"arm": "synapse_lean", "budget": None}]
    assert body["ingest"] == {"model": "gpt-4o-mini", "batch": False}


@pytest.mark.parametrize(
    "argv",
    [
        ["--budgets", "lots"],
        ["--budgets", "0"],
        ["--max-usd", "-1"],
        ["--mode", "free"],
        ["--extra-cell", "@500"],
        ["--n", "0"],
    ],
)
def test_bad_flags_are_argparse_errors(backend: FakeBackend, argv, capsys):
    with pytest.raises(SystemExit) as info:
        run_cli(backend, "lab", "estimate", *argv)
    assert info.value.code == 2
    assert backend.calls == []


# ── CLI: run (consent and caps) ──────────────────────────────────────────────
def test_run_refuses_without_consent_on_a_non_interactive_stdin(backend: FakeBackend, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # piped "y" is not consent
    assert run_cli(backend, "lab", "run") == 1
    captured = capsys.readouterr()
    assert "Retrieve-only: 3 (arm, budget) cells" in captured.err  # the estimate came first
    assert "refusing to start a Lab run without consent" in captured.err
    assert backend.paths("POST") == ["/api/lab/estimate"]


def test_run_interactive_prompt(backend: FakeBackend, capsys, monkeypatch):
    monkeypatch.setattr("synapse_graphrag.cli._stdin_is_interactive", lambda: True)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert run_cli(backend, "lab", "run") == 1
    assert "Cancelled; no run was started." in capsys.readouterr().err
    assert posted(backend, "/api/lab/runs") == []
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert run_cli(backend, "lab", "run") == 0
    assert "Start it? [y/N]" in capsys.readouterr().err
    assert len(posted(backend, "/api/lab/runs")) == 1


def test_a_paid_run_needs_max_usd_before_anything_is_sent(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "run", "--mode", "realtime", "--yes") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: --mode realtime calls the backend's reader model: pass --max-usd")
    assert backend.calls == []


def test_an_over_cap_run_is_refused_before_it_is_requested(backend: FakeBackend, capsys):
    argv = ["lab", "run", "--mode", "realtime", "--arms", "bm25", "--budgets", "8k", "--max-usd", "0.001", "--yes"]
    assert run_cli(backend, *argv) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and "error: refused before anything was started" in err
    assert backend.paths("POST") == ["/api/lab/estimate"]


def test_a_free_run_follows_progress_and_prints_the_retrieval_leaderboard(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "run", "--yes", "--run-id", "mine") == 0
    captured = capsys.readouterr()
    assert "no model is called and nothing is spent" in captured.err
    assert "Lab run mine started (job job_000010)" in captured.err
    progress = [line for line in captured.err.splitlines() if line.startswith("[retrieve] ") and "/" in line]
    assert progress[-1] == "[retrieve] 27/27" and len(progress) <= 11  # throttled, not one per item
    assert "Lab run mine · done · mode retrieve" in captured.out
    assert "CONTEXT TOK" in captured.out and "null_closed_book (floor)" in captured.out
    assert "vocabulary null (N1)" in captured.out
    assert posted(backend, "/api/lab/runs")[0]["run_id"] == "mine"


def test_a_realtime_run_says_it_spends_and_prints_the_leaderboard(backend: FakeBackend, capsys):
    argv = ["lab", "run", "--mode", "realtime", "--arms", "bm25,synapse_lean,null_closed_book", "--budgets", "500", "--max-usd", "1", "--yes"]
    assert run_cli(backend, *argv) == 0
    captured = capsys.readouterr()
    assert "This run SPENDS: up to $0.0060 (upper bound), hard cap $1.00" in captured.err
    assert "[read] 18/18 · spent $0.0021" in captured.err
    out = captured.out
    assert "spend $0.0021 actual" in out and "cap $1.00" in out
    assert "$/100 CORRECT" in out and "null_closed_book (floor)" in out
    lean = next(line for line in out.splitlines() if line.startswith("synapse_lean"))
    assert "+25.6" in lean and "+3.4 (n.s.)" in lean
    assert "Effect floor: 11.11 points" in out
    assert "Pareto frontier (F1 vs $): bm25@500, synapse_lean@500" in out
    assert "Note: ingest cost unknown" in out
    assert posted(backend, "/api/lab/runs")[0]["max_usd"] == 1.0


def test_a_batch_run_parks_and_says_how_to_collect(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "run", "--mode", "batch", "--max-usd", "1", "--yes") == 0
    captured = capsys.readouterr()
    assert "Batch mode" in captured.err and "[read] batch submitted: 18 requests" in captured.err
    assert f"lab resume {NEW_RUN}" in captured.out


def test_run_json_and_a_failed_status_exit_1(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "run", "--yes", "--json") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "done"
    backend.lab_events = [
        {"type": "aborted", "reason": "spend cap: 3 request(s) not sent"},
        {"type": "done", "data": {"run_id": NEW_RUN, "status": "aborted", "reason": "spend cap: 3 request(s) not sent"}},
    ]
    assert run_cli(backend, "lab", "run", "--mode", "realtime", "--max-usd", "1", "--yes") == 1
    captured = capsys.readouterr()
    assert "[read] aborted: spend cap" in captured.err
    assert f"Run {NEW_RUN}: aborted — spend cap" in captured.out
    assert "--max-usd X" in captured.out


def test_run_error_event_exits_1_with_one_line(backend: FakeBackend, capsys):
    backend.lab_events = [{"type": "error", "data": "GraphNotIngested: the knowledge graph is empty"}]
    assert run_cli(backend, "lab", "run", "--yes") == 1
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        "error: GraphNotIngested: the knowledge graph is empty"
    )


def test_run_interrupt_says_the_run_keeps_going(backend: FakeBackend, capsys, monkeypatch):
    def interrupt(event, started):
        if event.get("type") == "accepted":
            started["run_id"] = event["run_id"]
        if event.get("type") == "progress":
            raise KeyboardInterrupt

    monkeypatch.setattr("synapse_graphrag.cli._print_lab_event", interrupt)
    assert run_cli(backend, "lab", "run", "--yes") == 130
    assert f"run {NEW_RUN} keeps going on the backend" in capsys.readouterr().err


# ── CLI: runs, show, resume ──────────────────────────────────────────────────
def test_runs_lists_every_run(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "runs") == 0
    out = capsys.readouterr().out
    assert f"• {LAB_RUN_ID}  done · realtime · demo n=9 · 3 arms × [500, default] · reader gpt-5-nano" in out
    assert f"• {LAB_BATCH_ID}  batch_submitted · batch" in out
    backend.lab_runs.clear()
    assert run_cli(backend, "lab", "runs") == 0
    assert "No Lab runs yet" in capsys.readouterr().out


def test_show_with_rows_and_json(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "show", LAB_RUN_ID, "--rows", "2", "--arm", "bm25") == 0
    out = capsys.readouterr().out
    assert f"Lab run {LAB_RUN_ID} · done · mode realtime" in out
    assert "code c6a6e35f00 (dirty) · Synapse 0.4.0" in out
    assert "Rows 1–2 of 3:" in out and "demo-01" in out and "demo-03" not in out
    assert backend.calls[-1].params == {"offset": "0", "limit": "2", "arm": "bm25"}
    assert run_cli(backend, "lab", "show", LAB_BATCH_ID, "--json") == 0
    assert json.loads(capsys.readouterr().out)["leaderboard"] is None
    assert run_cli(backend, "lab", "show", LAB_BATCH_ID) == 0
    assert "Not scored yet." in capsys.readouterr().out
    assert run_cli(backend, "lab", "show", "ghost") == 1
    assert capsys.readouterr().err.strip() == "error: HTTP 404: Unknown Lab run 'ghost'."


def test_show_says_unknown_when_the_commit_was_not_recorded(backend: FakeBackend, capsys):
    backend.lab_runs[LAB_RUN_ID]["manifest"]["code"] = {
        "git_sha": None, "git_dirty": None, "synapse_version": None,
        "git_unavailable": "no git checkout or git binary here",
    }
    assert run_cli(backend, "lab", "show", LAB_RUN_ID) == 0
    out = capsys.readouterr().out
    assert "code unknown · Synapse unknown" in out and "None" not in out


def test_resuming_a_paid_run_asks_first(backend: FakeBackend, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert run_cli(backend, "lab", "resume", LAB_ABORTED_ID, "--max-usd", "1") == 1
    err = capsys.readouterr().err
    assert "this SPENDS" in err and "5 pending requests, upper bound $0.0040" in err
    assert "$0.50 already spent; cap $1.00" in err
    assert "refusing to resume a paid Lab run without consent" in err
    assert posted(backend, f"/api/lab/runs/{LAB_ABORTED_ID}/resume") == []

    assert run_cli(backend, "lab", "resume", LAB_ABORTED_ID, "--max-usd", "1", "--yes") == 0
    assert posted(backend, f"/api/lab/runs/{LAB_ABORTED_ID}/resume") == [
        {"retry_failed": False, "max_usd": 1.0}
    ]
    assert f"Lab run {LAB_ABORTED_ID} · done" in capsys.readouterr().out


def test_resuming_a_parked_batch_needs_no_consent(backend: FakeBackend, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert run_cli(backend, "lab", "resume", LAB_BATCH_ID) == 0
    captured = capsys.readouterr()
    assert "Polling the batch" in captured.err and "no new spend" in captured.err
    assert posted(backend, f"/api/lab/runs/{LAB_BATCH_ID}/resume") == [{"retry_failed": False}]


def test_a_batch_that_is_still_running_exits_0_and_says_so(backend: FakeBackend, capsys):
    backend.lab_resume_status = "batch_submitted"
    backend.lab_runs[LAB_BATCH_ID]["status"] = "batch_submitted"
    backend.lab_events = [
        {"type": "batch_status", "status": {"done": False, "status": "in_progress"}},
        {"type": "done", "data": {"run_id": LAB_BATCH_ID, "status": "batch_submitted"}},
    ]
    assert run_cli(backend, "lab", "resume", LAB_BATCH_ID) == 0
    captured = capsys.readouterr()
    assert "[batch] in_progress" in captured.err
    assert "not finished yet" in captured.out


def test_rescoring_a_done_run_is_free(backend: FakeBackend, capsys):
    assert run_cli(backend, "lab", "resume", LAB_RUN_ID) == 0
    assert "Re-scoring" in capsys.readouterr().err
    assert run_cli(backend, "lab", "resume", LAB_RUN_ID, "--retry-failed") == 1  # spends → asks
    assert "refusing to resume a paid Lab run" in capsys.readouterr().err
