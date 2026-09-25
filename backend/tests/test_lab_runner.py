# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The Lab runner end to end: retrieve → read → score, resumable, capped, batch.

Hermetic: fake arms (the real registry's names, so floors and premiums
compute), ``fake_neo4j`` for the read-only graph checks, a fake OpenAI client
and a fake Batch backend. A word-count tokenizer makes every token exact.
Every Cypher statement the runner sends is asserted read-only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.lab import arms as lab_arms
from app.lab import packer, reader, runner
from app.lab.arms import BaseArm, NullClosedBook
from app.lab.evidence import Evidence, EvidenceUnit
from app.lab.runner import DatasetSpec, LabDataset, LabError, LabItem, LabRun

DEMO = {r["question"]: r["answer"] for r in json.loads(runner.DEMO_QA_PATH.read_text())}
WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH)\b")


def counts(q, entities=10, chunks=5, relations=4, mentions=6):
    """Answer a fingerprint count statement (``None`` for any other query)."""
    values = {"entities": entities, "chunks": chunks, "relations": relations,
              "mentions": mentions}
    for key, query in lab_arms.FINGERPRINT_QUERIES.items():
        if q == query:
            return [{"n": values[key]}]
    return None


def words(text: str, model: str | None = None) -> tuple[int, bool]:
    return len(text.split()), False


class FakeArm(BaseArm):
    def __init__(self, name, family, make, *, needs_graph=False):
        self.name, self.family, self.make, self.needs_graph = name, family, make, needs_graph
        self.title = name
        self.description = f"fake {name}"
        self.source = {"citation": "test", "url": "https://example.org"}
        self.calls: list[str] = []

    async def retrieve(self, question, *, k, seed):
        self.calls.append(question)
        return Evidence(self.make(question), self.name)


def answer_unit(question):
    answer = DEMO.get(question, "nothing")
    answer = answer[0] if isinstance(answer, list) else answer
    return [EvidenceUnit(f"The answer is {answer}", "prose", "doc-a", 1.0)]


def lean_units(question):
    filler = " ".join(["filler"] * 200)
    return [EvidenceUnit(f"Entity: X (Type: T)\n  Description: {question}", "entity", "X", 1.0),
            EvidenceUnit(filler, "prose", "doc-f", 0.5)]


def noise_units(question):
    return [EvidenceUnit("random passage about nothing", "prose", "doc-r", 1.0)]


@pytest.fixture
def fake_arms(monkeypatch):
    fakes = {
        "null_closed_book": NullClosedBook(),
        "null_random": FakeArm("null_random", "null", noise_units),
        "bm25": FakeArm("bm25", "passage", answer_unit),
        "synapse_lean": FakeArm("synapse_lean", "graph", lean_units, needs_graph=True),
    }
    monkeypatch.setattr(runner, "ARMS", fakes)
    return fakes


@pytest.fixture(autouse=True)
def _exact_tokens(monkeypatch):
    monkeypatch.setattr(packer, "count_tokens", words)
    monkeypatch.setattr(reader, "count_tokens", words)
    lab_arms.clear_caches()


@pytest.fixture
def graph(fake_neo4j):
    """A non-empty graph; fails the test on any write statement."""

    def handler(q, p):
        assert not WRITE.search(q), f"the Lab must never write to the graph: {q}"
        return counts(q) or []

    return fake_neo4j(handler)


class FakeClient:
    def __init__(self, prompt_tokens=100):
        self.bodies: list[dict] = []
        self.prompt_tokens = prompt_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **body):
        self.bodies.append(body)
        user = body["messages"][1]["content"]
        match = re.search(r"The answer is ([^\n]+)", user)
        answer = match.group(1) if match else "unknown"
        return SimpleNamespace(
            model="gpt-5-nano-2025-08-07",
            system_fingerprint="fp",
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer),
                                     finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens, completion_tokens=20,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


def make_run(tmp_path, **kw):
    defaults = {
        "dataset": DatasetSpec("demo"),
        "arms": ["null_closed_book", "null_random", "bm25", "synapse_lean"],
        "budgets": [30, 500],
        "mode": "realtime",
        "max_usd": 1.0,
        "run_dir": tmp_path / "run",
        "reader_model": "gpt-5-nano",
    }
    defaults.update(kw)
    return LabRun(**defaults)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ── Retrieve-only ($0) ───────────────────────────────────────────────────────
class TestRetrieveOnly:
    async def test_writes_every_file_and_calls_no_model(self, tmp_path, fake_arms, graph):
        run = make_run(tmp_path, mode="retrieve", max_usd=None, budgets=[30, None])
        manifest = await runner.run_lab(run)
        root = tmp_path / "run"
        assert manifest["status"] == "done"
        for name in ("manifest.json", "contexts.jsonl", "rows.jsonl", "leaderboard.json",
                     "report.md"):
            assert (root / name).exists(), name
        assert not (root / "requests.jsonl").exists()
        assert manifest["phases"]["read"]["status"] == "skipped"
        assert manifest["estimate"]["pre_run"]["total_upper_usd"] == 0.0

        board = json.loads((root / "leaderboard.json").read_text())
        assert board["mode"] == "retrieve"
        bm25 = next(r for r in board["rows"] if r["arm"] == "bm25" and r["budget"] == 30)
        assert bm25["containment"] == 100.0 and "f1" not in bm25
        noise = next(r for r in board["rows"] if r["arm"] == "null_random" and r["budget"] == 30)
        assert noise["containment"] == 0.0
        # Floors are listed first at each budget.
        assert board["rows"][0]["is_null"] is True
        assert "Retrieve-only" in (root / "report.md").read_text()

    async def test_manifest_records_what_reproduces_the_run(self, tmp_path, fake_arms, graph):
        manifest = await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        ds = manifest["dataset"]
        assert ds["name"] == "demo" and ds["split"] == "test" and ds["n"] == 9
        assert ds["sha256"] == runner._sha256_file(runner.DEMO_QA_PATH)
        assert len(ds["question_ids"]) == 9
        assert set(manifest["code"]) == {
            "git_sha", "git_dirty", "git_source", "git_unavailable", "synapse_version"
        }
        assert manifest["code"]["synapse_version"]
        assert manifest["packer"]["order"] == "score" and "selection" in manifest["packer"]
        assert manifest["arms"]["bm25"]["config_hash"]
        assert manifest["models"]["reader"]["prompt_version"] == reader.PROMPT_VERSION
        assert manifest["models"]["embedding"]["provider"] == "fake"
        assert manifest["prices"]["reader_checked_on"]
        assert manifest["graph"]["entities"] == 10
        assert manifest["phases"]["retrieve"]["cells"] == 4 * 2 * 9

    async def test_contexts_are_stored_once_with_hashes(self, tmp_path, fake_arms, graph):
        await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        raw = lines(tmp_path / "run" / "contexts.jsonl")
        assert len(raw) == 4 * 2 * 9
        n0 = [r for r in raw if r["arm"] == "null_closed_book"]
        assert sum(1 for r in n0 if r["text"] is not None) == 1  # identical texts stored once
        restored = runner.load_contexts(tmp_path / "run" / "contexts.jsonl")
        for r in restored:
            assert runner._sha(r["text"]) == r["sha256"]

    async def test_the_budget_binds(self, tmp_path, fake_arms, graph):
        await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        lean = [r for r in runner.load_contexts(tmp_path / "run" / "contexts.jsonl")
                if r["arm"] == "synapse_lean"]
        small = [r for r in lean if r["budget"] == 30]
        big = [r for r in lean if r["budget"] == 500]
        assert all(r["tokens"] <= 30 and r["truncated"] for r in small)
        assert all(r["by_kind"] == {"entity": 1, "prose": 1} for r in big)

    async def test_extra_cells_are_retrieved_scored_and_estimated(
        self, tmp_path, fake_arms, graph
    ):
        run = make_run(tmp_path, mode="retrieve", budgets=[30],
                       extra_cells=[("synapse_lean", None)])
        manifest = await runner.run_lab(run)
        cells = {(r["arm"], r["budget"]) for r in manifest["estimate"]["pre_run"]["cells"]}
        assert ("synapse_lean", None) in cells and ("bm25", None) not in cells
        board = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
        keys = {(r["arm"], r["budget"]) for r in board["rows"]}
        assert keys == {("null_closed_book", 30), ("null_random", 30), ("bm25", 30),
                        ("synapse_lean", 30), ("synapse_lean", None)}
        assert "extra cells: synapse_lean@default" in (tmp_path / "run" / "report.md").read_text()

    async def test_a_free_retrieve_run_prices_the_paid_one_tightly(
        self, tmp_path, fake_arms, graph
    ):
        await runner.run_lab(make_run(tmp_path, mode="retrieve", budgets=[500]))
        measured = runner.measured_context_tokens(tmp_path / "run")
        assert set(measured) == {("null_closed_book", 500), ("null_random", 500),
                                 ("bm25", 500), ("synapse_lean", 500)}
        paid = make_run(tmp_path, mode="batch", budgets=[500], run_dir=tmp_path / "paid")
        dataset = runner.load_dataset(paid.dataset)
        loose = runner.pre_run_estimate(paid, dataset)
        tight = runner.pre_run_estimate(paid, dataset, measured)
        assert tight["total_upper_usd"] < loose["total_upper_usd"]
        assert all(c["measured"] for c in tight["cells"])

    async def test_an_empty_graph_is_refused(self, tmp_path, fake_arms, fake_neo4j):
        fake_neo4j(lambda q, p: counts(q, 0, 0, 0, 0) or [])
        with pytest.raises(runner.GraphNotIngested):
            await runner.run_lab(make_run(tmp_path, mode="retrieve"))


# ── Realtime ─────────────────────────────────────────────────────────────────
class TestRealtime:
    async def test_reads_scores_and_dedupes_requests(self, tmp_path, fake_arms, graph):
        client = FakeClient()
        manifest = await runner.run_lab(make_run(tmp_path), client=client)
        root = tmp_path / "run"
        assert manifest["status"] == "done"
        # N0, N2 and bm25 read the same text at both budgets → one request each per question.
        assert len(client.bodies) == 9 + 9 + 9 + 18
        requests = lines(root / "requests.jsonl")
        assert len(requests) == 45
        first = requests[0]
        assert first["method"] == "POST" and first["url"] == "/v1/chat/completions"
        parts = runner.parse_custom_id(first["custom_id"])
        assert parts["run_id"] == "run" and parts["arm"] == "null_closed_book"
        assert parts["budget"] == 30

        board = json.loads((root / "leaderboard.json").read_text())
        by = {(r["arm"], r["budget"]): r for r in board["rows"]}
        assert by[("bm25", 30)]["f1"] == 100.0 and by[("null_closed_book", 30)]["f1"] == 0.0
        gain = by[("bm25", 30)]["comparisons"]["gain_above_n0"]
        assert gain["diff"] == 100.0 and gain["reportable"] is True
        assert by[("synapse_lean", 30)]["comparisons"]["graph_premium"]["against"] == "bm25"
        assert by[("bm25", 30)]["reasoning_tokens"] == 9 * 12
        rows = lines(root / "rows.jsonl")
        assert len(rows) == 4 * 2 * 9
        spent = sum(r["usd"] for r in lines(root / "responses.jsonl"))
        assert manifest["actual"]["reader"]["usd"] == pytest.approx(spent)
        assert manifest["estimate_vs_actual"]["reader_actual_over_upper"] < 1.0
        assert "| bm25 |" in (root / "report.md").read_text()

    async def test_a_failed_retrieval_is_not_read_and_not_scored(
        self, tmp_path, fake_arms, graph
    ):
        bm25 = fake_arms["bm25"]
        first_question = None

        def flaky(question):
            nonlocal first_question
            first_question = first_question or question
            if question == first_question:
                raise RuntimeError("index chunk_fulltext not found")
            return answer_unit(question)

        bm25.make = flaky
        run = make_run(tmp_path, arms=["null_closed_book", "bm25"], budgets=[500])
        client = FakeClient()
        manifest = await runner.run_lab(run, client=client)
        root = tmp_path / "run"
        assert manifest["status"] == "done"
        requests = lines(root / "requests.jsonl")
        assert len(requests) == 9 + 8  # the failed cell is never sent (nor shares N0's answer)
        rows = [r for r in lines(root / "rows.jsonl") if r["arm"] == "bm25"]
        failed = [r for r in rows if r.get("retrieval_error")]
        assert len(failed) == 1 and failed[0]["read_error"].startswith("retrieval failed")
        assert failed[0].get("request_id") is None and failed[0]["answer"] is None
        board = json.loads((root / "leaderboard.json").read_text())
        cell = next(r for r in board["rows"] if r["arm"] == "bm25")
        assert cell["answered"] == 8 and cell["retrieval_errors"] == 1 and cell["f1"] == 100.0
        assert cell["comparisons"]["gain_above_n0"]["n"] == 8

    async def test_a_rerun_spends_nothing_more(self, tmp_path, fake_arms, graph):
        await runner.run_lab(make_run(tmp_path), client=FakeClient())
        fakes = [a for a in fake_arms.values() if isinstance(a, FakeArm)]
        for arm in fakes:
            arm.calls.clear()
        again = FakeClient()
        manifest = await runner.run_lab(make_run(tmp_path), client=again)
        assert again.bodies == [] and manifest["status"] == "done"
        assert all(not arm.calls for arm in fakes)  # the retrieve phase is not repeated
        assert len(lines(tmp_path / "run" / "contexts.jsonl")) == 4 * 2 * 9

    async def test_a_changed_dataset_is_never_resumed_or_rescored(
        self, tmp_path, fake_arms, graph
    ):
        qa = tmp_path / "q.jsonl"
        qa.write_text("\n".join(json.dumps({"id": i, "question": f"{i}?", "answer": i})
                                for i in ("a", "b", "c")))
        spec = DatasetSpec("qa-file", path=str(qa))
        await runner.run_lab(make_run(tmp_path, dataset=spec, mode="retrieve"))
        board = (tmp_path / "run" / "leaderboard.json").read_text()

        qa.write_text(json.dumps({"id": "a", "question": "a?", "answer": "changed"}))
        with pytest.raises(runner.DatasetChanged, match="changed since"):
            await runner.resume(tmp_path / "run")
        with pytest.raises(runner.DatasetChanged, match="1 questions, not the 3 recorded"):
            await runner.run_lab(make_run(tmp_path, dataset=spec, mode="retrieve"))
        # Same ids, other golds: the content hash catches it.
        qa.write_text("\n".join(json.dumps({"id": i, "question": f"{i}?", "answer": "x"})
                                for i in ("a", "b", "c")))
        with pytest.raises(runner.DatasetChanged, match="content hash"):
            await runner.resume(tmp_path / "run")
        assert (tmp_path / "run" / "leaderboard.json").read_text() == board

    async def test_retrieval_is_resumed_not_repeated(self, tmp_path, fake_arms, graph):
        run = make_run(tmp_path, mode="retrieve")
        await runner.run_lab(run)
        # Simulate a crash mid-retrieval: drop the last lines, tear the final one.
        path = tmp_path / "run" / "contexts.jsonl"
        kept = path.read_text().splitlines()[:-5]
        path.write_text("\n".join(kept) + '\n{"arm": "bm2')
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        manifest["phases"]["retrieve"]["status"] = "running"
        (tmp_path / "run" / "manifest.json").write_text(json.dumps(manifest))
        for arm in fake_arms.values():
            if isinstance(arm, FakeArm):
                arm.calls.clear()
        await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        repeated = sum(len(a.calls) for a in fake_arms.values() if isinstance(a, FakeArm))
        assert 1 <= repeated <= 3  # only the (arm, question) pairs whose lines were lost
        assert len(runner.read_jsonl(path)) == 4 * 2 * 9

    async def test_the_metered_cap_aborts_cleanly_and_resume_finishes(
        self, tmp_path, fake_arms, graph
    ):
        # The provider bills more than the up-front bound assumed (a stale price, a
        # tokenizer fallback): the pre-read gate passes, so the METER must stop the run.
        # 999,840 prompt + 20 completion tokens = exactly $0.05 per call at gpt-5-nano.
        greedy = FakeClient(prompt_tokens=999_840)
        run = make_run(tmp_path, max_usd=0.10, max_concurrency=1)
        manifest = await runner.run_lab(run, client=greedy)
        assert manifest["status"] == "aborted" and "spend cap" in manifest["abort_reason"]
        # $0.05 spent, $0.05 + the next call's worst case still fits; at $0.10 it does not.
        assert len(greedy.bodies) == 2
        responses = lines(tmp_path / "run" / "responses.jsonl")
        assert len(responses) == len(greedy.bodies)
        assert sum(r["usd"] for r in responses) == pytest.approx(0.10)
        board = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
        assert any(r.get("read_errors") for r in board["rows"])

        finisher = FakeClient()
        manifest = await runner.resume(tmp_path / "run", client=finisher, max_usd=5.0)
        assert manifest["status"] == "done"
        assert len(finisher.bodies) == 45 - len(greedy.bodies)

    async def test_refused_before_spending_when_the_measured_bound_breaks_the_cap(
        self, tmp_path, fake_arms, graph
    ):
        client = FakeClient()
        manifest = await runner.run_lab(make_run(tmp_path, max_usd=0.000001), client=client)
        assert manifest["status"] == "refused" and client.bodies == []
        assert "exceeds the remaining cap" in manifest["refuse_reason"]
        assert manifest["estimate"]["read"]["refuse"] is True

    async def test_a_foreign_run_dir_is_refused(self, tmp_path, fake_arms, graph):
        await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        with pytest.raises(LabError, match="different run"):
            await runner.run_lab(make_run(tmp_path, mode="retrieve", arms=["bm25"]))

    async def test_failed_requests_are_retried_only_on_request(self, tmp_path, fake_arms, graph):
        class Flaky(FakeClient):
            def __init__(self):
                super().__init__()
                self.fail_first = True

            async def create(self, **body):
                if self.fail_first:
                    self.fail_first = False
                    self.bodies.append(body)
                    raise RuntimeError("upstream 500")
                return await super().create(**body)

        flaky = Flaky()
        manifest = await runner.run_lab(make_run(tmp_path, max_concurrency=1), client=flaky)
        assert manifest["status"] == "done"
        assert manifest["phases"]["read"]["status"] == "done_with_errors"
        again = FakeClient()
        await runner.resume(tmp_path / "run", client=again)
        assert again.bodies == []  # a plain resume never re-sends
        manifest = await runner.resume(tmp_path / "run", client=again, retry_failed=True)
        assert len(again.bodies) == 1 and manifest["phases"]["read"]["status"] == "done"
        board = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
        assert not any(r.get("read_errors") for r in board["rows"])

    async def test_events_are_emitted(self, tmp_path, fake_arms, graph):
        events = []
        await runner.run_lab(make_run(tmp_path), client=FakeClient(), on_event=events.append)
        types = [e["type"] for e in events]
        assert types[0] == "phase" and types[-1] == "done"
        assert any(e.get("phase") == "read" and e["type"] == "progress" for e in events)


# ── Batch ────────────────────────────────────────────────────────────────────
class FakeBackend:
    def __init__(self):
        self.submitted: list[dict] = []
        self.polls = 0
        self.ready_after = 2

    async def submit(self, run_dir, requests, *, run_id):
        self.submitted = list(requests)
        return {"batch_ids": ["batch_1"], "requests": len(requests)}

    async def poll(self, run_dir):
        self.polls += 1
        done = self.polls >= self.ready_after
        return {"status": "completed" if done else "in_progress", "done": done}

    async def collect(self, run_dir):
        out = []
        for i, line in enumerate(self.submitted):
            if i == 0:  # one request failed upstream
                out.append({"custom_id": line["custom_id"], "response": None,
                            "error": {"code": "server_error", "message": "boom"}})
                continue
            user = line["body"]["messages"][1]["content"]
            match = re.search(r"The answer is ([^\n]+)", user)
            body = {
                "model": "gpt-5-nano",
                "choices": [{"message": {"content": match.group(1) if match else "?"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "completion_tokens_details": {"reasoning_tokens": 12}},
            }
            out.append({"custom_id": line["custom_id"],
                        "response": {"status_code": 200, "body": body}, "error": None})
        return out


class TestBatch:
    async def test_submit_then_resume_until_collected(self, tmp_path, fake_arms, graph):
        backend = FakeBackend()
        run = make_run(tmp_path, mode="batch")
        manifest = await runner.run_lab(run, batch_backend=backend)
        assert manifest["status"] == "batch_submitted"
        assert len(backend.submitted) == 45
        assert all(set(x) == {"custom_id", "method", "url", "body"} for x in backend.submitted)

        manifest = await runner.resume(tmp_path / "run", batch_backend=backend)
        assert manifest["status"] == "batch_submitted" and backend.polls == 1

        manifest = await runner.resume(tmp_path / "run", batch_backend=backend)
        assert manifest["status"] == "done"
        responses = lines(tmp_path / "run" / "responses.jsonl")
        ok = [r for r in responses if not r["error"]]
        assert len(ok) == 44 and all(r["batch"] for r in responses)
        # Priced at half the realtime rate.
        realtime = runner._usd(reader.ReaderResult(prompt_tokens=100, completion_tokens=20),
                               "gpt-5-nano", False)
        assert ok[0]["usd"] == pytest.approx(realtime / 2)
        board = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
        assert sum(r.get("read_errors") or 0 for r in board["rows"]) >= 1
        assert manifest["phases"]["read"]["status"] == "done_with_errors"

        # The failed request — and only it — is re-submittable as a new batch.
        retry = FakeBackend()
        retry.ready_after = 1
        manifest = await runner.resume(tmp_path / "run", batch_backend=retry,
                                       retry_failed=True)
        assert manifest["status"] == "batch_submitted" and len(retry.submitted) == 1
        assert retry.submitted[0]["custom_id"] == backend.submitted[0]["custom_id"]

    async def test_the_pending_batch_must_fit_the_cap(self, tmp_path, fake_arms, graph):
        backend = FakeBackend()
        manifest = await runner.run_lab(make_run(tmp_path, mode="batch", max_usd=0.000001),
                                        batch_backend=backend)
        assert manifest["status"] == "refused" and backend.submitted == []

    def test_record_normalisation(self):
        ok = runner.result_from_record({"custom_id": "a", "answer": "x",
                                        "usage": {"prompt_tokens": 3}})
        assert ok.answer == "x" and ok.prompt_tokens == 3
        bad = runner.result_from_record({"custom_id": "a", "response": {"status_code": 500,
                                                                        "body": {"error": "e"}}})
        assert bad.error


# ── HotpotQA ─────────────────────────────────────────────────────────────────
def hotpot_dataset():
    items = [LabItem("h1", "Who is Ada?", ["Ada Lovelace"], {"gold_titles": ["P1", "P2"]})]
    return LabDataset("hotpotqa", "dev-distractor", items, "sha", "test",
                      notes={"seed": 1}, titles=["P1", "P2", "P3"])


class TestHotpot:
    async def test_refused_unless_the_corpus_is_ingested(self, tmp_path, fake_arms, fake_neo4j):
        from benchmarks.public import run_hotpotqa

        def handler(q, p):
            assert not WRITE.search(q)
            if (fp := counts(q, 1, 1, 0, 1)) is not None:
                return fp
            if q == run_hotpotqa.CHUNK_DOCUMENTS_QUERY:
                return [{"document": "P1"}]
            return []

        fake_neo4j(handler)
        run = make_run(tmp_path, dataset=DatasetSpec("hotpotqa", n=1), mode="retrieve")
        with pytest.raises(runner.GraphNotIngested, match="2 of 3"):
            await runner.run_lab(run, dataset=hotpot_dataset())

    async def test_strict_and_permissive_recall_per_arm(self, tmp_path, monkeypatch, fake_neo4j):
        from benchmarks.public import run_hotpotqa

        fakes = {
            "null_vocabulary": FakeArm(
                "null_vocabulary", "null",
                lambda q: [EvidenceUnit("Ada Lovelace\nZed Zulu", "name_list", None, 1.0)],
                needs_graph=True),
            "bm25": FakeArm("bm25", "passage",
                            lambda q: [EvidenceUnit("P2 text", "prose", "P2", 1.0)]),
        }
        monkeypatch.setattr(runner, "ARMS", fakes)

        def handler(q, p):
            assert not WRITE.search(q)
            if (fp := counts(q, 2, 3, 0, 2)) is not None:
                return fp
            if q == run_hotpotqa.CHUNK_DOCUMENTS_QUERY:
                return [{"document": t} for t in ("P1", "P2", "P3")]
            if q == run_hotpotqa.PROVENANCE_QUERY:
                return [{"name": "Ada Lovelace", "documents": ["P1"]},
                        {"name": "Zed Zulu", "documents": ["P3"]}]
            return []

        fake_neo4j(handler)
        run = make_run(tmp_path, dataset=DatasetSpec("hotpotqa", n=1), mode="retrieve",
                       arms=["null_vocabulary", "bm25"], budgets=[None])
        await runner.run_lab(run, dataset=hotpot_dataset())
        board = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
        by = {r["arm"]: r for r in board["rows"]}
        # N1 reaches P1 only through a NAME: permissive credit, zero strict credit.
        assert by["null_vocabulary"]["recall_permissive"] == 50.0
        assert by["null_vocabulary"]["recall_strict"] == 0.0
        assert by["bm25"]["recall_permissive"] == 50.0 and by["bm25"]["recall_strict"] == 50.0
        assert by["null_vocabulary"]["containment"] == 100.0


class TestCrediting:
    def test_packed_crediting_matches_the_harness_on_a_shipped_context(self):
        from app.services import chat_engine
        from benchmarks.public.run_hotpotqa import credit_paragraphs

        context = (
            "Entity: Ada Lovelace (Type: Person)\n  Description: wrote notes\n"
            "  Relationships:\n  → KNEW → Charles Babbage (Person)\n\n"
            "Entity: Analytical Engine (Type: Machine)"
            f"\n\n{chat_engine.PATHS_HEADING}\n  - Ada Lovelace -[KNEW]-> Charles Babbage"
            f"\n\n{chat_engine.SOURCES_HEADING}\n\n[S1] P9 (chunk 0)\nSome prose."
        )
        sources = [{"id": "c9", "text": "Some prose.", "document": "P9", "index": 0}]
        citations = [{"name": "Ada Lovelace", "kind": "entity"},
                     {"name": "Analytical Engine", "kind": "entity"}]
        provenance = {"Ada Lovelace": ["P1"], "Charles Babbage": ["P2"],
                      "Analytical Engine": ["P3"], "Nobody": ["P4"]}
        retrieval = chat_engine.Retrieval(context, citations, [], "local", sources)
        units, _ = lab_arms.explode_retrieval(retrieval)
        packed = packer.pack(units, None, "gpt-5-nano")
        ours = runner.credit_packed(packed, provenance, runner.NameMatcher(provenance))
        assert ours == credit_paragraphs(context, sources, provenance)
        assert ours == {"P9": "prose", "P1": "entity", "P3": "entity", "P2": "edge"}

    @pytest.mark.parametrize(
        "text",
        [
            "Ada Lovelace met Charles Babbage in London.",
            "ada lovelace (1815) and O'Brien; Zed-Zulu, St. Louis",
            "Nothing relevant here at all",
            "",
            "Ada Lovelaces is not Ada Lovelace's twin; Babbage",
        ],
    )
    def test_name_matcher_equals_names_present(self, text):
        from benchmarks.public.run_hotpotqa import names_present

        names = ["Ada Lovelace", "Charles Babbage", "O'Brien", "Zed-Zulu", "St. Louis",
                 "Babbage", "AB", "(1815)", "London"]
        assert runner.NameMatcher(names).present(text) == set(names_present(text, names))


# ── Datasets & files ─────────────────────────────────────────────────────────
class TestDatasets:
    def test_demo_default_split_is_test(self):
        ds = runner.load_dataset(DatasetSpec("demo"))
        assert ds.split == "test" and len(ds.items) == 9
        assert len(runner.load_dataset(DatasetSpec("demo", split="all")).items) == 30
        assert len(runner.load_dataset(DatasetSpec("demo", split="train", n=2)).items) == 2

    def test_qa_file_json_jsonl_and_aliases(self, tmp_path):
        a = tmp_path / "a.json"
        a.write_text(json.dumps([{"question": "Q1?", "answer": ["A", "alias"]},
                                 {"question": "Q2?", "answer": "B", "id": "x2"}]))
        ds = runner.load_dataset(DatasetSpec("qa-file", path=str(a)))
        assert [i.id for i in ds.items] == ["q0001", "x2"]
        assert ds.items[0].gold == ["A", "alias"]
        b = tmp_path / "b.jsonl"
        b.write_text('{"question": "Q?", "answer": "A"}\n\n{"question": "R?", "answer": "B"}\n')
        assert len(runner.load_dataset(DatasetSpec("qa-file", path=str(b))).items) == 2
        c = tmp_path / "c.json"
        c.write_text(json.dumps({"items": [{"question": "Q?", "answer": "A"}]}))
        assert len(runner.load_dataset(DatasetSpec("qa-file", path=str(c))).items) == 1

    @pytest.mark.parametrize(
        "records, message",
        [
            ([{"question": "Q?"}], "non-empty question and answer"),
            ([{"question": "Q?", "answer": "A", "id": "1"},
              {"question": "R?", "answer": "B", "id": "1"}], "duplicate id"),
            (["not an object"], "not an object"),
            ([], "no questions"),
        ],
    )
    def test_qa_file_validation(self, records, message):
        with pytest.raises(LabError, match=message):
            runner.parse_qa_records(records)

    def test_uploaded_names_cannot_escape_the_datasets_dir(self, tmp_path):
        assert runner.qa_file_path("my-set_1.jsonl", tmp_path) == tmp_path / "my-set_1.jsonl"
        for bad in ("../x.json", "/etc/passwd", "a/b.json", ".hidden.json", "x.txt", ""):
            with pytest.raises(LabError):
                runner.qa_file_path(bad, tmp_path)

    def test_list_qa_files(self, tmp_path):
        (tmp_path / "ok.json").write_text(json.dumps([{"question": "Q", "answer": "A"}]))
        (tmp_path / "bad.jsonl").write_text("{not json")
        (tmp_path / "notes.txt").write_text("ignored")
        listed = {e["name"]: e for e in runner.list_qa_files(tmp_path)}
        assert listed["ok.json"]["n"] == 1 and "error" in listed["bad.jsonl"]
        assert "notes.txt" not in listed


class TestFiles:
    def test_torn_jsonl_is_repaired_and_tolerated(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
        assert runner.read_jsonl(path) == [{"a": 1}, {"a": 2}]
        runner._repair_jsonl(path)
        assert path.read_text() == '{"a": 1}\n{"a": 2}\n'

    def test_custom_id_round_trip_with_a_pipe_in_the_qid(self):
        parts = runner.parse_custom_id("r1|bm25|default|q|with|pipes")
        assert parts == {"run_id": "r1", "arm": "bm25", "budget": None, "qid": "q|with|pipes"}

    def test_run_ids_cannot_escape_the_runs_dir(self, tmp_path):
        assert runner.run_dir_for("20260925-abc", tmp_path) == tmp_path / "20260925-abc"
        for bad in ("../etc", "a/b", "..", ""):
            with pytest.raises(LabError):
                runner.run_dir_for(bad, tmp_path)

    def test_reserved_and_hidden_run_ids_are_refused(self, tmp_path):
        for bad in ("datasets", "DataSets", "calibration.json", ".hidden", ".x.y"):
            with pytest.raises(LabError):
                runner.run_dir_for(bad, tmp_path)
        assert runner.check_run_id("datasets-2") == "datasets-2"

    def test_the_qa_file_store_is_never_listed_as_a_run(self, tmp_path):
        for name in ("datasets", ".trash"):
            (tmp_path / name).mkdir()
            (tmp_path / name / runner.MANIFEST).write_text(json.dumps({"run_id": name}))
        assert runner.list_runs(tmp_path) == []

    async def test_listing_and_paging(self, tmp_path, fake_arms, graph):
        await runner.run_lab(make_run(tmp_path, mode="retrieve", run_dir=tmp_path / "r1"))
        runs = runner.list_runs(tmp_path)
        assert [r["run_id"] for r in runs] == ["r1"] and runs[0]["status"] == "done"
        page = runner.read_rows(tmp_path / "r1", offset=2, limit=5, arm="bm25")
        assert page["total"] == 18 and len(page["rows"]) == 5
        assert runner.read_rows(tmp_path / "r1", budget="30", arm="bm25")["total"] == 9
        loaded = runner.load_run(tmp_path / "r1")
        assert loaded["leaderboard"]["mode"] == "retrieve"


class PoolArm(BaseArm):
    """A passage-ranking arm over a corpus of ``size`` 20-word passages; honours ``k``."""

    k_role = "passages"
    family = "passage"

    def __init__(self, name="bm25", size=100):
        self.name, self.title, self.description = name, name, "pool"
        self.source = {"citation": "test", "url": "https://example.org"}
        self.size = size
        self.ks: list[int] = []

    async def retrieve(self, question, *, k, seed):
        self.ks.append(k)
        units = [EvidenceUnit(f"p{i} " + " ".join(["w"] * 19), "prose", f"c{i}", 100.0 - i)
                 for i in range(min(k, self.size))]
        return Evidence(units, self.name)


class TestPassageFill:
    """Passage-ranking arms fill every capped budget (the floors are budget-matched)."""

    async def test_capped_budgets_are_filled_and_the_default_keeps_k(
        self, tmp_path, monkeypatch, graph
    ):
        arm = PoolArm()
        monkeypatch.setattr(runner, "ARMS", {"bm25": arm})
        run = make_run(tmp_path, arms=["bm25"], budgets=[100, 1000, None], mode="retrieve",
                       dataset=DatasetSpec("demo", n=1))
        manifest = await runner.run_lab(run)
        by_budget = {r["budget"]: r for r in lines(tmp_path / "run" / "contexts.jsonl")}
        for budget in (100, 1000):
            ctx = by_budget[budget]
            assert budget - 25 <= ctx["tokens"] <= budget, ctx["tokens"]  # full, never over
            assert ctx["truncated"] is True
        assert by_budget[None]["units_used"] == 8  # the arm's own k at the default context
        assert max(arm.ks) >= 1000 // 20
        assert manifest["packer"]["passages"] == {
            "mode": "fill", "default_k": 8, "max": runner.PASSAGE_FILL_MAX}

    async def test_a_resumed_default_cell_is_the_uninterrupted_one(
        self, tmp_path, monkeypatch, graph
    ):
        arm = PoolArm()
        monkeypatch.setattr(runner, "ARMS", {"bm25": arm})
        kw = {"arms": ["bm25"], "budgets": [1000, None], "mode": "retrieve",
              "dataset": DatasetSpec("demo", n=1)}
        await runner.run_lab(make_run(tmp_path, **kw))
        path = tmp_path / "run" / "contexts.jsonl"
        full = {r["budget"]: r["sha256"] for r in lines(path)}
        # Lose the default cell only, as a crash would; resume retrieves just that cell.
        path.write_text("".join(json.dumps(r) + "\n" for r in lines(path) if r["budget"]))
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        manifest["phases"]["retrieve"]["status"] = "running"
        (tmp_path / "run" / "manifest.json").write_text(json.dumps(manifest))
        arm.ks.clear()
        await runner.run_lab(make_run(tmp_path, **kw))
        latest = {r["budget"]: r["sha256"] for r in lines(path)}
        assert latest == full and max(arm.ks) > 8  # same pool, same top-k

    async def test_a_small_corpus_stops_the_fill(self, tmp_path, monkeypatch, graph):
        arm = PoolArm(size=5)
        monkeypatch.setattr(runner, "ARMS", {"bm25": arm})
        run = make_run(tmp_path, arms=["bm25"], budgets=[4000], mode="retrieve",
                       dataset=DatasetSpec("demo", n=1))
        await runner.run_lab(run)
        (ctx,) = lines(tmp_path / "run" / "contexts.jsonl")
        assert ctx["units_used"] == 5 and ctx["truncated"] is False
        assert len(arm.ks) == 1  # exhausted on the first retrieval: no pointless retries

    async def test_an_explicit_passage_k_is_exact(self, tmp_path, monkeypatch, graph):
        arm = PoolArm()
        monkeypatch.setattr(runner, "ARMS", {"bm25": arm})
        run = make_run(tmp_path, arms=["bm25"], budgets=[1000], mode="retrieve",
                       dataset=DatasetSpec("demo", n=1), passage_k=8)
        manifest = await runner.run_lab(run)
        (ctx,) = lines(tmp_path / "run" / "contexts.jsonl")
        assert arm.ks == [8] and ctx["units_used"] == 8 and ctx["tokens"] < 1000
        assert manifest["packer"]["passages"] == {"mode": "fixed", "k": 8}


class TestProvenance:
    def test_git_when_there_is_a_checkout(self, monkeypatch):
        answers = {"rev-parse": "abc123", "status": ""}
        monkeypatch.setattr(runner, "_git", lambda *args: answers[args[0]])
        code = runner.code_provenance()
        assert (code["git_sha"], code["git_dirty"], code["git_source"]) == ("abc123", False, "git")
        assert code["git_unavailable"] is None

    def test_the_environment_when_git_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(runner, "_git", lambda *args: None)
        monkeypatch.setenv("SYNAPSE_GIT_SHA", " deadbeef ")
        monkeypatch.setenv("SYNAPSE_GIT_DIRTY", "true")
        code = runner.code_provenance()
        assert code["git_sha"] == "deadbeef" and code["git_dirty"] is True
        assert code["git_source"] == "env:SYNAPSE_GIT_SHA"

    async def test_unknown_is_explained_never_a_bare_null(
        self, monkeypatch, tmp_path, fake_arms, graph
    ):
        monkeypatch.setattr(runner, "_git", lambda *args: None)
        monkeypatch.delenv("SYNAPSE_GIT_SHA", raising=False)
        code = runner.code_provenance()
        assert code["git_sha"] is None and "SYNAPSE_GIT_SHA" in code["git_unavailable"]
        await runner.run_lab(make_run(tmp_path, mode="retrieve"))
        report = (tmp_path / "run" / "report.md").read_text()
        assert "- Code: unknown (no git checkout" in report and "`None`" not in report


class TestConfig:
    def test_budgets_normalise(self):
        run = LabRun(dataset="demo", arms=["bm25", "bm25"], budgets=[500, "default", "500", None])
        assert run.budgets == [500, None] and run.arms == ["bm25"]
        assert run.mode == "retrieve"  # the free tier is the default

    @pytest.mark.parametrize(
        "kw",
        [{"mode": "free"}, {"arms": []}, {"arms": ["nope"]}, {"order": "random"}, {"k": 0},
         {"max_usd": -1.0}, {"dataset": {"name": "mystery"}}],
    )
    def test_invalid_configs(self, kw):
        base = {"dataset": "demo", "arms": ["bm25"], "budgets": [500]}
        base.update(kw)
        with pytest.raises(LabError):
            LabRun(**base).validate()

    def test_negative_budget(self):
        with pytest.raises(LabError):
            LabRun(dataset="demo", arms=["bm25"], budgets=[-5])

    def test_round_trip_and_hash(self, tmp_path):
        run = make_run(tmp_path)
        again = LabRun.from_dict(run.to_dict())
        assert again.config_hash() == run.config_hash()
        assert again.dataset == run.dataset

    def test_extra_cells_extend_the_grid(self):
        run = LabRun(dataset="demo", arms=["bm25", "synapse_lean"], budgets=[500],
                     extra_cells=[["synapse_lean", None], {"arm": "synapse_lean", "budget": 500}])
        assert run.cells() == [("bm25", 500), ("synapse_lean", 500), ("synapse_lean", None)]
        assert run.budgets_for("synapse_lean") == [500, None]
        with pytest.raises(LabError, match="not in the run"):
            LabRun(dataset="demo", arms=["bm25"], budgets=[500],
                   extra_cells=[("ppr", None)]).validate()

    def test_passage_k(self):
        run = LabRun(dataset="demo", arms=["bm25"], budgets=[500], k=8, passage_k=40)
        assert run.k_for(lab_arms.ARMS["bm25"]) == 40
        assert run.k_for(lab_arms.ARMS["synapse_lean"]) == 8
