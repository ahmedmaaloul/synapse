# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""API tests for ``/api/lab/*``: the HTTP contract the client, CLI and UI code against.

Hermetic by construction. The TestClient never runs the lifespan; runs and QA
files live under ``tmp_path``; retrieval goes through fake arms (the real
registry's names, so floors and premiums compute) over ``fake_neo4j`` with a
handler that FAILS on any write statement; realtime reads go to a fake OpenAI
client and batch reads to a fake Batch backend. A word-count tokenizer makes
every token count exact. Nothing here can reach a model, the network or a DB.
"""

from __future__ import annotations

import io
import json
import re
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.lab import arms as lab_arms
from app.lab import packer, reader, runner
from app.lab.arms import BaseArm, NullClosedBook
from app.lab.evidence import Evidence, EvidenceUnit
from app.main import app
from app.routers import lab as lab_router

WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH)\b")
DEMO = {r["question"]: r["answer"] for r in json.loads(runner.DEMO_QA_PATH.read_text())}
DEMO_TEST_N = sum(1 for r in json.loads(runner.DEMO_QA_PATH.read_text()) if r["split"] == "test")


# ── Fakes ────────────────────────────────────────────
def words(text: str, model: str | None = None) -> tuple[int, bool]:
    return len(text.split()), False


class FakeArm(BaseArm):
    def __init__(self, name, family, make, *, needs_graph=False):
        self.name, self.family, self.make, self.needs_graph = name, family, make, needs_graph
        self.title = name
        self.description = f"fake {name}"
        self.source = {"citation": "test", "url": "https://example.org"}

    async def retrieve(self, question, *, k, seed):
        return Evidence(self.make(question), self.name)


def answer_units(question):
    answer = DEMO.get(question, "nothing")
    answer = answer[0] if isinstance(answer, list) else answer
    return [EvidenceUnit(f"The answer is {answer}", "prose", "doc-a", 1.0)]


def graph_units(question):
    return [
        EvidenceUnit(f"Entity: X (Type: T)\n  Description: {question}", "entity", "X", 1.0),
        *answer_units(question),
    ]


def noise_units(question):
    return [EvidenceUnit("random passage about nothing", "prose", "doc-r", 1.0)]


def completion(content: str, prompt_tokens: int = 100) -> dict:
    return {
        "model": "gpt-5-nano-2025-08-07",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def fake_answer(body: dict) -> str:
    match = re.search(r"The answer is ([^\n]+)", body["messages"][1]["content"])
    return match.group(1) if match else "unknown"


class FakeClient:
    """The ``AsyncOpenAI`` surface the reader uses: ``chat.completions.create``."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **body):
        self.bodies.append(body)
        return completion(fake_answer(body))


class FakeBatch:
    """The runner's ``BatchBackend``: submit, poll, collect — answers in Batch OUTPUT format."""

    def __init__(self, done: bool = True):
        self.submitted: list[dict] = []
        self.done = done
        self.polls = 0

    async def submit(self, run_dir, requests, *, run_id):
        self.submitted.extend(requests)
        return {"batches": ["batch_fake_1"], "requests": len(requests)}

    async def poll(self, run_dir):
        self.polls += 1
        return {"done": self.done, "status": "completed" if self.done else "in_progress"}

    async def collect(self, run_dir):
        return [
            {
                "custom_id": line["custom_id"],
                "response": {"status_code": 200, "body": completion(fake_answer(line["body"]))},
                "error": None,
            }
            for line in self.submitted
        ]


# ── Fixtures ─────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    """Runs and datasets under tmp_path; exact tokens; no job state between tests."""
    monkeypatch.setattr(runner, "LAB_RUNS_DIR", tmp_path / "lab_runs")
    monkeypatch.setattr(runner, "DATASETS_DIR", tmp_path / "lab_runs" / "datasets")
    monkeypatch.setattr(packer, "count_tokens", words)
    monkeypatch.setattr(reader, "count_tokens", words)
    lab_arms.clear_caches()
    lab_router._active_runs.clear()
    lab_router._jobs_by_run.clear()
    lab_router._hotpot_titles.clear()
    yield
    lab_router._active_runs.clear()
    lab_router._jobs_by_run.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_arms(monkeypatch):
    fakes = {
        "null_closed_book": NullClosedBook(),
        "null_random": FakeArm("null_random", "null", noise_units),
        "bm25": FakeArm("bm25", "passage", answer_units),
        "synapse_lean": FakeArm("synapse_lean", "graph", graph_units, needs_graph=True),
    }
    monkeypatch.setattr(runner, "ARMS", fakes)
    return fakes


def fingerprint(query: str, *, empty: bool = False):
    for key, statement in lab_arms.FINGERPRINT_QUERIES.items():
        if query == statement:
            return [{"n": 0 if empty else {"entities": 10, "chunks": 5}.get(key, 4)}]
    return None


@pytest.fixture
def graph(fake_neo4j):
    """A small non-empty graph; any write statement fails the test."""

    def handler(query, params):
        assert not WRITE.search(query), f"the Lab must never write to the graph: {query}"
        return fingerprint(query) or []

    return fake_neo4j(handler)


@pytest.fixture
def fake_client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(lab_router, "_reader_client", lambda: fake)
    monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key="sk-test"))
    return fake


@pytest.fixture
def fake_batch(monkeypatch):
    fake = FakeBatch()
    monkeypatch.setattr(lab_router, "_batch_backend", lambda: fake)
    monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key="sk-test"))
    return fake


@pytest.fixture
def no_batch_module(monkeypatch):
    """Make ``from app.lab import batch`` fail, whether or not the module exists yet."""
    import app.lab

    monkeypatch.delattr(app.lab, "batch", raising=False)
    monkeypatch.setitem(sys.modules, "app.lab.batch", None)


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def upload(client, name: str, content: str | bytes, **form):
    data = content.encode() if isinstance(content, str) else content
    return client.post("/api/lab/qa-files", files={"file": (name, io.BytesIO(data))}, data=form)


QA_JSONL = "\n".join(
    json.dumps(r)
    for r in [
        {"id": "a", "question": "Who wrote programs for the Analytical Engine?", "answer": "Ada Lovelace", "split": "test"},
        {"id": "b", "question": "Who designed it?", "answer": ["Charles Babbage", "Babbage"], "split": "test"},
        {"id": "c", "question": "When?", "answer": "1837", "split": "train"},
    ]
)


def start(client, **body) -> dict:
    r = client.post("/api/lab/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── Catalogue ────────────────────────────────────────
class TestCatalogue:
    def test_arms_are_the_registry_grouped_floors_first(self, client, no_batch_module):
        body = client.get("/api/lab/arms").json()
        names = [a["name"] for a in body["arms"]]
        assert set(names) == set(lab_arms.ARMS) and len(names) == 8
        families = [a["family"] for a in body["arms"]]
        assert families == sorted(families, key=["null", "passage", "graph"].index)
        for arm in body["arms"]:
            assert set(arm) >= {
                "name", "family", "title", "description", "source", "retrieval_llm_calls",
                "needs_graph", "is_null",
            }
            assert arm["source"]["url"].startswith("https://")
            assert arm["retrieval_llm_calls"] == 0
            assert arm["is_null"] is (arm["family"] == "null")
        assert [f["family"] for f in body["families"]] == ["null", "passage", "graph"]
        assert body["modes"] == ["realtime", "batch", "retrieve"]
        assert body["default_mode"] == "retrieve"
        assert body["budgets"] == [500, 1000, 2000, 4000, 8000, None]
        assert body["batch_available"] is False

    def test_batch_is_reported_available_once_its_backend_exists(self, client, fake_batch):
        assert client.get("/api/lab/arms").json()["batch_available"] is True

    def test_models_are_priced_readers_only(self, client):
        body = client.get("/api/lab/models").json()
        models = {m["name"]: m for m in body["models"]}
        assert "text-embedding-3-small" not in models
        nano = models["gpt-5-nano"]
        assert nano["reasoning"] is True and models["gpt-4o-mini"]["reasoning"] is False
        assert nano["batch_input_usd_per_1m"] == pytest.approx(nano["input_usd_per_1m"] * 0.5)
        assert nano["batch_output_usd_per_1m"] == pytest.approx(nano["output_usd_per_1m"] * 0.5)
        assert nano["max_output_tokens"] == reader.MAX_ANSWER_TOKENS + reader.REASONING_ALLOWANCE
        assert models["gpt-4o-mini"]["max_output_tokens"] == reader.MAX_ANSWER_TOKENS
        assert nano["price_checked_on"] and body["default"] == "gpt-5-nano"
        assert body["batch_multiplier"] == 0.5 and body["pricing_url"].startswith("https://")


# ── Datasets ─────────────────────────────────────────
class TestDatasets:
    def test_demo_is_always_listed_with_its_splits(self, client, monkeypatch, tmp_path):
        from benchmarks.public import hotpotqa

        monkeypatch.setattr(hotpotqa, "CACHE_PATH", tmp_path / "missing.json")
        body = client.get("/api/lab/datasets").json()
        demo = body["datasets"][0]
        assert demo["id"] == "demo" and demo["default_split"] == "test"
        assert demo["splits"]["test"] == DEMO_TEST_N == demo["n"]
        assert set(demo["splits"]) == {"train", "val", "test"}
        (hot,) = body["unavailable"]
        assert hot["id"] == "hotpotqa" and "never downloads" in hot["reason"]

    def test_uploads_are_listed_and_broken_files_explained(self, client, monkeypatch, tmp_path):
        from benchmarks.public import hotpotqa

        monkeypatch.setattr(hotpotqa, "CACHE_PATH", tmp_path / "missing.json")
        assert upload(client, "mine.jsonl", QA_JSONL).status_code == 200
        runner.DATASETS_DIR.joinpath("broken.json").write_text('[{"question": "q"}]')
        runner.DATASETS_DIR.joinpath("notes.txt").write_text("ignored")
        body = client.get("/api/lab/datasets").json()
        mine = next(d for d in body["datasets"] if d.get("file") == "mine.jsonl")
        assert mine["id"] == "qa-file:mine.jsonl" and mine["n"] == 3
        assert mine["splits"] == {"test": 2, "train": 1}
        broken = next(d for d in body["unavailable"] if d.get("file") == "broken.json")
        assert "non-empty question and answer" in broken["reason"]
        assert not any(d.get("file") == "notes.txt" for d in body["datasets"] + body["unavailable"])

    def _hotpot_cache(self, monkeypatch, tmp_path):
        from benchmarks.public import hotpotqa

        cache = tmp_path / "hotpot.json"
        cache.write_text(json.dumps([
            {"_id": "1", "context": [["Ada Lovelace", ["s."]], ["Analytical Engine", ["s."]]]},
            {"_id": "2", "context": {"title": ["Charles Babbage"], "sentences": [["s."]]}},
        ]))
        monkeypatch.setattr(hotpotqa, "CACHE_PATH", cache)

    @pytest.mark.parametrize(
        ("documents", "available"),
        [(["Ada Lovelace", "my-report.pdf"], True), (["my-report.pdf"], False), ([], False)],
    )
    def test_hotpotqa_is_listed_only_when_its_paragraphs_are_in_the_graph(
        self, client, fake_neo4j, monkeypatch, tmp_path, documents, available
    ):
        self._hotpot_cache(monkeypatch, tmp_path)
        calls = fake_neo4j(
            lambda q, p: [{"document": d} for d in documents] if "c.document" in q else []
        )
        body = client.get("/api/lab/datasets").json()
        listed = [d for d in body["datasets"] if d["id"] == "hotpotqa"]
        missing = [d for d in body["unavailable"] if d["id"] == "hotpotqa"]
        if available:
            assert listed[0]["graph_paragraphs"] == 1 and listed[0]["graph_documents"] == 2
            assert "refuses otherwise" in listed[0]["note"] and not missing
        else:
            assert not listed and missing[0]["reason"]
        assert all(not WRITE.search(q) for q, _ in calls)

    def test_a_neo4j_outage_does_not_break_the_listing(self, client, fake_neo4j, monkeypatch, tmp_path):
        self._hotpot_cache(monkeypatch, tmp_path)

        def down(q, p):
            raise RuntimeError("connection refused")

        fake_neo4j(down)
        body = client.get("/api/lab/datasets").json()
        assert body["datasets"][0]["id"] == "demo"
        assert "connection refused" in body["unavailable"][0]["reason"]


# ── QA files ─────────────────────────────────────────
class TestQAFiles:
    def test_a_valid_file_is_stored_with_its_counts(self, client):
        r = upload(client, "my set.jsonl", QA_JSONL)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "created" and body["file"] == "my-set.jsonl"
        assert body["id"] == "qa-file:my-set.jsonl" and body["n"] == 3
        assert body["splits"] == {"test": 2, "train": 1} and len(body["sha256"]) == 64
        assert (runner.DATASETS_DIR / "my-set.jsonl").read_text() == QA_JSONL

    def test_a_json_array_works_and_a_path_cannot_escape(self, client):
        r = upload(client, "../../outside.json", json.dumps([{"question": "q", "answer": "a"}]))
        assert r.status_code == 200, r.text
        assert r.json()["file"] == "outside.json"
        assert (runner.DATASETS_DIR / "outside.json").is_file()
        assert not (runner.LAB_RUNS_DIR.parent / "outside.json").exists()

    def test_same_content_is_idempotent_and_different_content_needs_replace(self, client):
        assert upload(client, "s.jsonl", QA_JSONL).json()["status"] == "created"
        assert upload(client, "s.jsonl", QA_JSONL).json()["status"] == "unchanged"
        other = QA_JSONL.replace("1837", "1838")
        r = upload(client, "s.jsonl", other)
        assert r.status_code == 409 and "replace" in r.json()["detail"]
        assert (runner.DATASETS_DIR / "s.jsonl").read_text() == QA_JSONL
        assert upload(client, "s.jsonl", other, replace="true").json()["status"] == "replaced"
        assert (runner.DATASETS_DIR / "s.jsonl").read_text() == other

    @pytest.mark.parametrize(
        ("name", "content", "needle"),
        [
            ("bad.jsonl", '{"question": "q"}', "non-empty question and answer"),
            ("dup.jsonl", '{"id": "x", "question": "q", "answer": "a"}\n' * 2, "duplicate id"),
            ("bad.json", "[not json", "not valid JSON"),
            ("bad.csv", "question,answer", "*.json or *.jsonl"),
            ("empty.json", "", "empty"),
        ],
    )
    def test_invalid_files_are_422_and_nothing_is_stored(self, client, name, content, needle):
        r = upload(client, name, content)
        assert r.status_code == 422, r.text
        assert needle in json.dumps(r.json()["detail"])
        stored = [p for p in runner.DATASETS_DIR.rglob("*") if p.is_file()] if runner.DATASETS_DIR.exists() else []
        assert stored == []

    @pytest.mark.parametrize(
        ("name", "content", "needle"),
        [
            ("bad.json", "[not json", "bad.json is not valid JSON"),
            ("bad.jsonl", '{"question": "q", "answer": "a"}\n{oops', "bad.jsonl:2 is not valid JSON"),
            ("lines.json", '{"question": "q", "answer": "a"}\n{oops', "lines.json is not valid JSON"),
            ("empty.jsonl", "  \n", "empty.jsonl is empty"),
        ],
    )
    def test_errors_name_the_uploaded_file_not_the_staging_copy(self, client, name, content, needle):
        r = upload(client, name, content)
        assert r.status_code == 422, r.text
        detail = json.dumps(r.json()["detail"])
        assert needle in detail
        assert ".incoming" not in detail and str(runner.DATASETS_DIR) not in detail

    @pytest.mark.parametrize("name", ["bom.json", "bom.jsonl"])
    def test_a_utf8_byte_order_mark_is_accepted(self, client, name):
        records = [{"question": "Who?", "answer": "Ada"}, {"question": "When?", "answer": "1843"}]
        if name.endswith(".jsonl"):
            text = "\n".join(json.dumps(r) for r in records)
        else:
            text = json.dumps(records)
        r = upload(client, name, "\ufeff" + text)
        assert r.status_code == 200, r.text
        assert r.json()["n"] == 2
        body = client.post("/api/lab/estimate", json={"dataset": f"qa-file:{name}", "arms": ["bm25"]})
        assert body.status_code == 200, body.text

    def test_non_utf8_and_oversized_files_are_refused(self, client, monkeypatch):
        assert upload(client, "latin.jsonl", "é".encode("latin-1")).status_code == 422
        monkeypatch.setattr(lab_router, "MAX_QA_FILE_BYTES", 10)
        assert upload(client, "big.jsonl", QA_JSONL).status_code == 413


# ── Estimate ─────────────────────────────────────────
class TestEstimate:
    def test_retrieve_mode_is_free_and_touches_nothing(self, client, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        r = client.post("/api/lab/estimate", json={"arms": ["bm25", "dense"], "budgets": [500, "default"]})
        assert r.status_code == 200, r.text
        est = r.json()
        assert est["mode"] == "retrieve" and est["total_upper_usd"] == 0.0 and est["refuse"] is False
        assert est["dataset"]["name"] == "demo" and est["dataset"]["n"] == DEMO_TEST_N
        assert [(c["arm"], c["budget"]) for c in est["cells"]] == [
            ("bm25", 500), ("bm25", None), ("dense", 500), ("dense", None)
        ]
        assert est["cells_count"] == 4 and est["config_hash"]
        assert calls == []  # the estimate never reads the graph

    def test_every_arm_by_default(self, client):
        est = client.post("/api/lab/estimate", json={}).json()
        assert {c["arm"] for c in est["cells"]} == set(lab_arms.ARMS)

    def test_a_paid_estimate_has_point_and_upper_bounds(self, client):
        est = client.post(
            "/api/lab/estimate",
            json={"arms": ["null_closed_book", "bm25"], "budgets": [500, 2000], "mode": "batch",
                  "max_usd": 5, "reader_model": "gpt-5-nano"},
        ).json()
        assert est["batch"] is True and est["refuse"] is False
        assert 0 < est["total_point_usd"] < est["total_upper_usd"] <= 5
        n0 = [c for c in est["cells"] if c["arm"] == "null_closed_book"]
        assert [c["calls"] for c in n0] == [DEMO_TEST_N, 0]  # identical at every budget: once
        assert est["max_output_tokens"] == reader.MAX_ANSWER_TOKENS + reader.REASONING_ALLOWANCE
        assert any("REASONING" in a for a in est["assumptions"])

    def test_the_cap_refuses_and_an_unpriced_model_cannot_be_capped(self, client):
        body = {"arms": ["bm25"], "budgets": [8000], "mode": "realtime", "max_usd": 0.000001}
        est = client.post("/api/lab/estimate", json=body).json()
        assert est["refuse"] is True and "exceeds the cap" in est["refuse_reason"]
        est = client.post("/api/lab/estimate", json={**body, "reader_model": "mystery-1", "max_usd": 10}).json()
        assert est["refuse"] is True and "no price on file" in est["refuse_reason"]

    def test_qa_file_by_id_or_by_name(self, client):
        upload(client, "mine.jsonl", QA_JSONL)
        by_id = client.post("/api/lab/estimate", json={"dataset": "qa-file:mine.jsonl", "arms": ["bm25"]})
        by_name = client.post(
            "/api/lab/estimate", json={"dataset": "qa-file", "file": "mine.jsonl", "arms": ["bm25"], "split": "test"}
        )
        assert by_id.json()["dataset"]["n"] == 3 and by_name.json()["dataset"]["n"] == 2

    @pytest.mark.parametrize(
        ("body", "status", "needle"),
        [
            ({"arms": ["nope"]}, 422, "unknown arm"),
            ({"budgets": [0]}, 422, "positive token count"),
            ({"budgets": ["lots"]}, 422, "positive token count"),
            ({"budgets": [5_000_000]}, 422, "not a budget"),
            ({"dataset": "wiki"}, 422, "unknown dataset"),
            ({"dataset": "qa-file"}, 422, "needs 'file'"),
            ({"dataset": "qa-file:"}, 422, "after the colon"),
            ({"dataset": "qa-file:ghost.jsonl"}, 404, "No uploaded QA file"),
            ({"dataset": "qa-file:../etc/passwd"}, 422, "bare .json/.jsonl"),
            ({"file": "x.jsonl"}, 422, "qa-file dataset only"),
            ({"offset": 5}, 422, "hotpotqa dataset only"),
            ({"mode": "free"}, 422, "retrieve"),
            ({"max_usd": -1}, 422, "greater than or equal"),
            ({"split": "nosuch"}, 422, "no questions in split"),
            ({"extra_cells": [{"arm": "ppr", "budget": None}], "arms": ["bm25"]}, 422, "not in the run"),
            ({"ingest": {}}, 422, "only hotpotqa knows its corpus size"),
            ({"measured_run_id": "../x"}, 422, "letters, digits"),
            ({"measured_run_id": "ghost"}, 404, "Unknown Lab run"),
        ],
    )
    def test_invalid_requests(self, client, body, status, needle):
        r = client.post("/api/lab/estimate", json=body)
        assert r.status_code == status, r.text
        assert needle in json.dumps(r.json()["detail"])

    def test_an_ingest_plan_is_priced_next_to_the_run(self, client):
        est = client.post(
            "/api/lab/estimate",
            json={"arms": ["bm25"], "mode": "batch", "ingest": {"paragraphs": 1000}},
        ).json()
        phase = next(p for p in est["phases"] if p["phase"] == "ingest")
        assert phase["calls"] == 1000 and phase["model"] == "gpt-4o-mini" and phase["batch"] is True
        assert phase["prompt_tokens"] == 434_000 and phase["completion_tokens"] == 464_000

    def test_measured_contexts_from_a_retrieve_run_tighten_the_estimate(
        self, client, fake_arms, graph
    ):
        base = {"arms": ["bm25", "synapse_lean"], "budgets": [None]}
        start(client, **base, run_id="measure")
        paid = {**base, "mode": "realtime", "max_usd": 1}
        loose = client.post("/api/lab/estimate", json=paid).json()
        tight = client.post("/api/lab/estimate", json={**paid, "measured_run_id": "measure"}).json()
        assert tight["measured_from"]["run_id"] == "measure"
        assert len(tight["measured_from"]["cells"]) == 2 and tight["measured_from"]["unmeasured"] == []
        assert all(c["measured"] for c in tight["cells"])
        assert tight["total_upper_usd"] < loose["total_upper_usd"]
        r = client.post("/api/lab/estimate", json={**paid, "measured_run_id": "measure", "k": 3})
        assert r.status_code == 422 and "its k is 8, not 3" in json.dumps(r.json()["detail"])
        # A run measured before passage arms filled their budgets cannot price this one.
        path = runner.LAB_RUNS_DIR / "measure" / runner.MANIFEST
        manifest = json.loads(path.read_text())
        del manifest["packer"]["passages"]
        path.write_text(json.dumps(manifest))
        r = client.post("/api/lab/estimate", json={**paid, "measured_run_id": "measure"})
        assert r.status_code == 422 and "passage counts" in json.dumps(r.json()["detail"])

    def test_a_measured_run_with_its_own_seed_prices_an_estimate_that_sends_it(
        self, client, fake_arms, graph
    ):
        # The UI sends the seed with /estimate as well as /runs (it is part of the request).
        base = {"arms": ["bm25"], "budgets": [None], "seed": 7}
        start(client, **base, run_id="m7")
        paid = {**base, "mode": "realtime", "max_usd": 1, "measured_run_id": "m7"}
        assert client.post("/api/lab/estimate", json=paid).status_code == 200
        unseeded = {k: v for k, v in paid.items() if k != "seed"}
        r = client.post("/api/lab/estimate", json=unseeded)
        assert r.status_code == 422 and "its seed is 7" in json.dumps(r.json()["detail"])


# ── Runs ─────────────────────────────────────────────
class TestRetrieveRuns:
    def test_a_free_run_end_to_end(self, client, fake_arms, graph):
        accepted = start(client, budgets=[30, "default"])
        run_id = accepted["run_id"]
        assert accepted["status"] == "running" and accepted["mode"] == "retrieve"
        assert accepted["estimate"]["total_upper_usd"] == 0.0
        assert accepted["events"] == f"/api/lab/runs/{run_id}/events"

        events = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        kinds = [e["type"] for e in events]
        assert kinds[0] == "phase" and "progress" in kinds and kinds[-1] == "done"
        assert all(e["run_id"] == run_id for e in events[:-1])
        done = events[-1]["data"]
        assert done["status"] == "done" and done["mode"] == "retrieve" and done["reason"] is None
        assert done["phases"] == {"retrieve": "done", "read": "skipped", "score": "done"}

        # Drained: the stream now reports the stored state once.
        replay = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        assert replay == [{"type": "done", "data": done}]

        runs = client.get("/api/lab/runs").json()["runs"]
        assert [r["run_id"] for r in runs] == [run_id] and runs[0]["active"] is False

        shown = client.get(f"/api/lab/runs/{run_id}", params={"limit": 2}).json()
        assert shown["status"] == "done" and shown["active"] is False
        assert shown["leaderboard"]["mode"] == "retrieve"
        assert shown["frontiers"] == {"f1_vs_usd": [], "f1_vs_tokens": []}
        assert shown["rows"]["total"] == 4 * 2 * DEMO_TEST_N and len(shown["rows"]["rows"]) == 2
        assert "Retrieve-only" in shown["report"]
        bm25 = next(r for r in shown["leaderboard"]["rows"] if r["arm"] == "bm25" and r["budget"] == 30)
        assert bm25["containment"] == 100.0
        page = client.get(
            f"/api/lab/runs/{run_id}", params={"arm": "bm25", "budget": "default", "limit": 100}
        ).json()["rows"]
        assert page["total"] == DEMO_TEST_N
        assert {(r["arm"], r["budget"]) for r in page["rows"]} == {("bm25", None)}
        assert client.get(f"/api/lab/runs/{run_id}", params={"budget": "x"}).status_code == 422
        assert all(not WRITE.search(q) for q, _ in graph)

    def test_unknown_runs(self, client):
        assert client.get("/api/lab/runs/ghost").status_code == 404
        assert client.get("/api/lab/runs/..%2F..").status_code == 404
        assert client.post("/api/lab/runs/ghost/resume").status_code == 404
        events = sse_events(client.get("/api/lab/runs/ghost/events").text)
        assert events == [{"type": "error", "data": "Unknown Lab run."}]
        assert client.get("/api/lab/runs").json() == {"runs": []}

    def test_an_empty_graph_is_refused_before_the_job(self, client, fake_arms, fake_neo4j):
        fake_neo4j(lambda q, p: fingerprint(q, empty=True) or [])
        r = client.post("/api/lab/runs", json={"arms": ["bm25"]})
        assert r.status_code == 409 and "graph is empty" in r.json()["detail"]
        assert not runner.LAB_RUNS_DIR.exists() or not any(runner.LAB_RUNS_DIR.iterdir())

    def test_a_neo4j_outage_is_a_503(self, client, fake_arms, fake_neo4j):
        def down(q, p):
            raise RuntimeError("connection refused")

        fake_neo4j(down)
        r = client.post("/api/lab/runs", json={"arms": ["bm25"]})
        assert r.status_code == 503 and "Neo4j" in r.json()["detail"]

    def test_a_run_id_is_reused_only_for_the_same_configuration(self, client, fake_arms, graph):
        first = start(client, arms=["bm25"], run_id="mine")
        assert first["run_id"] == "mine"
        client.get("/api/lab/runs/mine/events")
        again = start(client, arms=["bm25"], run_id="mine")  # same config: a resume
        assert sse_events(client.get("/api/lab/runs/mine/events").text)[-1]["data"]["status"] == "done"
        assert again["run_id"] == "mine"
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "k": 3, "run_id": "mine"})
        assert r.status_code == 409 and "another configuration" in r.json()["detail"]

    def test_reserved_run_ids_are_refused(self, client, fake_arms, graph):
        # "datasets" is where uploaded QA files live; a run there would pose as QA files.
        upload(client, "mine.jsonl", QA_JSONL)
        for run_id in ("datasets", "Datasets", ".hidden", "calibration.json"):
            r = client.post("/api/lab/runs", json={"arms": ["bm25"], "run_id": run_id})
            assert r.status_code == 422, (run_id, r.text)
            assert client.get(f"/api/lab/runs/{run_id}").status_code == 404
            r = client.post("/api/lab/estimate", json={"arms": ["bm25"], "measured_run_id": run_id})
            assert r.status_code == 422, (run_id, r.text)
        assert client.get("/api/lab/runs").json()["runs"] == []
        body = client.get("/api/lab/datasets").json()
        files = [d.get("file") for d in body["datasets"] + body["unavailable"] if d.get("file")]
        assert files == ["mine.jsonl"]

    def test_a_replaced_qa_file_is_never_resumed_or_rescored(self, client, fake_arms, graph):
        upload(client, "q.jsonl", QA_JSONL)
        body = {"dataset": "qa-file:q.jsonl", "arms": ["bm25"], "budgets": [500], "run_id": "qa1"}
        start(client, **body)
        client.get("/api/lab/runs/qa1/events")
        before = client.get("/api/lab/runs/qa1").json()
        other = "\n".join(json.dumps(r) for r in [
            {"id": "a", "question": "Who?", "answer": "Totally different"},
            {"id": "z", "question": "New?", "answer": "New"},
        ])
        assert upload(client, "q.jsonl", other, replace="true").json()["status"] == "replaced"

        r = client.post("/api/lab/runs/qa1/resume")
        assert r.status_code == 409, r.text
        assert "changed since" in r.json()["detail"]
        r = client.post("/api/lab/runs", json=body)  # same run id, same config: a resume
        assert r.status_code == 409 and "changed since" in r.json()["detail"]
        after = client.get("/api/lab/runs/qa1").json()
        assert after["leaderboard"] == before["leaderboard"]  # never re-scored

        # Restoring the original file makes the run resumable again.
        upload(client, "q.jsonl", QA_JSONL, replace="true")
        assert client.post("/api/lab/runs/qa1/resume").status_code == 200
        done = sse_events(client.get("/api/lab/runs/qa1/events").text)[-1]["data"]
        assert done["status"] == "done"

    def test_one_job_per_run_at_a_time(self, client, fake_arms, graph):
        lab_router._active_runs.add("busy")
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "run_id": "busy"})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
        start(client, arms=["bm25"], run_id="free")
        client.get("/api/lab/runs/free/events")
        lab_router._active_runs.add("free")
        assert client.post("/api/lab/runs/free/resume").status_code == 409

    def test_a_failing_job_reports_an_error_event(self, client, fake_arms, graph, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(runner, "run_lab", explode)
        run_id = start(client, arms=["bm25"])["run_id"]
        events = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        assert events == [{"type": "error", "data": "RuntimeError: disk full"}]
        assert run_id not in lab_router._active_runs


class TestPaidRuns:
    def test_a_paid_mode_needs_a_cap(self, client, fake_arms, graph, fake_client):
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "mode": "realtime"})
        assert r.status_code == 422 and "needs max_usd" in r.json()["detail"]["message"]
        assert fake_client.bodies == []

    def test_realtime_needs_the_openai_key(self, client, fake_arms, graph, monkeypatch):
        monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key=""))
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "mode": "realtime", "max_usd": 1})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]

    def test_an_estimate_over_the_cap_is_refused_with_the_estimate(
        self, client, fake_arms, graph, fake_client
    ):
        r = client.post(
            "/api/lab/runs",
            json={"arms": ["bm25"], "budgets": [8000], "mode": "realtime", "max_usd": 0.000001},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "breaks the spend cap" in detail["message"]
        assert detail["estimate"]["refuse"] is True and "exceeds the cap" in detail["diagnostics"][0]
        assert fake_client.bodies == [] and not runner.LAB_RUNS_DIR.exists()

    def test_a_realtime_run_reads_scores_and_reports_its_spend(
        self, client, fake_arms, graph, fake_client
    ):
        run_id = start(
            client, budgets=[30, 500], mode="realtime", max_usd=1, reader_model="gpt-5-nano"
        )["run_id"]
        events = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        assert any(e["type"] == "progress" and e["phase"] == "read" for e in events)
        done = events[-1]["data"]
        assert done["status"] == "done" and done["spent_usd"] > 0 and done["cap_usd"] == 1
        assert fake_client.bodies  # the fake, never a real client
        assert all(b["model"] == "gpt-5-nano" and "max_completion_tokens" in b for b in fake_client.bodies)
        shown = client.get(f"/api/lab/runs/{run_id}").json()
        board = shown["leaderboard"]
        assert board["mode"] == "read" and board["reader_model"] == "gpt-5-nano"
        bm25 = next(r for r in board["rows"] if r["arm"] == "bm25" and r["budget"] == 500)
        assert bm25["em"] == 100.0 and "gain_above_n0" in bm25["comparisons"]
        lean = next(r for r in board["rows"] if r["arm"] == "synapse_lean" and r["budget"] == 500)
        assert "graph_premium" in lean["comparisons"]
        assert shown["frontiers"]["f1_vs_usd"]
        assert shown["manifest"]["actual"]["reader"]["usd"] == pytest.approx(done["spent_usd"])

    def test_batch_without_its_module_is_a_501(self, client, fake_arms, graph, no_batch_module):
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "mode": "batch", "max_usd": 1})
        assert r.status_code == 501 and "Batch mode is not available" in r.json()["detail"]
        assert not runner.LAB_RUNS_DIR.exists()

    def test_batch_needs_the_openai_key_too(self, client, fake_arms, graph, fake_batch, monkeypatch):
        monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key=""))
        r = client.post("/api/lab/runs", json={"arms": ["bm25"], "mode": "batch", "max_usd": 1})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
        assert fake_batch.submitted == []

    def test_the_real_batch_module_fits_the_runner_adapter(self, client):
        """Section B's ``app.lab.batch`` exposes what the router hands the runner (no call made)."""
        module = pytest.importorskip("app.lab.batch")
        backend = lab_router._batch_backend()
        assert isinstance(backend, runner.ModuleBatchBackend) and backend.module is module
        assert client.get("/api/lab/arms").json()["batch_available"] is True

    def test_a_batch_run_parks_then_resume_collects_and_scores(
        self, client, fake_arms, graph, fake_batch
    ):
        run_id = start(client, budgets=[500], mode="batch", max_usd=1)["run_id"]
        done = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)[-1]["data"]
        assert done["status"] == "batch_submitted" and done["phases"]["read"] == "batch_submitted"
        assert fake_batch.submitted and all(
            line["custom_id"].startswith(f"{run_id}|") and line["url"] == "/v1/chat/completions"
            for line in fake_batch.submitted
        )
        assert client.get(f"/api/lab/runs/{run_id}").json()["leaderboard"] is None

        fake_batch.done = False
        accepted = client.post(f"/api/lab/runs/{run_id}/resume").json()
        assert accepted["previous_status"] == "batch_submitted"
        events = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        assert any(e["type"] == "batch_status" for e in events)
        assert events[-1]["data"]["status"] == "batch_submitted"

        fake_batch.done = True
        client.post(f"/api/lab/runs/{run_id}/resume")
        done = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)[-1]["data"]
        assert done["status"] == "done"
        board = client.get(f"/api/lab/runs/{run_id}").json()["leaderboard"]
        bm25 = next(r for r in board["rows"] if r["arm"] == "bm25")
        assert bm25["em"] == 100.0 and bm25["usd"] > 0

    def test_resuming_a_parked_batch_without_the_module_is_a_501(
        self, client, fake_arms, graph, fake_batch, monkeypatch
    ):
        run_id = start(client, budgets=[500], mode="batch", max_usd=1)["run_id"]
        client.get(f"/api/lab/runs/{run_id}/events")
        monkeypatch.setattr(lab_router, "_batch_backend", runner.ModuleBatchBackend)
        import app.lab

        monkeypatch.delattr(app.lab, "batch", raising=False)
        monkeypatch.setitem(sys.modules, "app.lab.batch", None)
        r = client.post(f"/api/lab/runs/{run_id}/resume")
        assert r.status_code == 501

    def test_an_aborted_run_resumes_under_a_new_cap(self, client, fake_arms, graph, fake_client):
        run_id = start(client, budgets=[500], mode="realtime", max_usd=1)["run_id"]
        client.get(f"/api/lab/runs/{run_id}/events")
        # Pretend the metered cap stopped it: the manifest says aborted.
        path = runner.LAB_RUNS_DIR / run_id / runner.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["status"] = "aborted"
        manifest["abort_reason"] = "spend cap"
        path.write_text(json.dumps(manifest))
        replay = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)
        assert replay[0]["data"]["status"] == "aborted" and replay[0]["data"]["reason"] == "spend cap"
        sent = len(fake_client.bodies)
        accepted = client.post(f"/api/lab/runs/{run_id}/resume", json={"max_usd": 2}).json()
        assert accepted["previous_status"] == "aborted"
        done = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)[-1]["data"]
        assert done["status"] == "done" and done["cap_usd"] == 2
        assert len(fake_client.bodies) == sent  # everything already had an answer: nothing re-sent

    def test_resuming_a_parked_batch_needs_the_key(self, client, fake_arms, graph, fake_batch, monkeypatch):
        run_id = start(client, budgets=[500], mode="batch", max_usd=1)["run_id"]
        client.get(f"/api/lab/runs/{run_id}/events")
        monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key=""))
        assert client.post(f"/api/lab/runs/{run_id}/resume").status_code == 503
        assert fake_batch.polls == 0

    def test_rescoring_a_finished_run_needs_no_key_and_no_backend(
        self, client, fake_arms, graph, fake_client, monkeypatch, no_batch_module
    ):
        run_id = start(client, budgets=[500], mode="realtime", max_usd=1)["run_id"]
        client.get(f"/api/lab/runs/{run_id}/events")
        monkeypatch.setattr(lab_router, "get_settings", lambda: SimpleNamespace(openai_api_key=""))
        sent = len(fake_client.bodies)
        assert client.post(f"/api/lab/runs/{run_id}/resume").json()["previous_status"] == "done"
        done = sse_events(client.get(f"/api/lab/runs/{run_id}/events").text)[-1]["data"]
        assert done["status"] == "done" and len(fake_client.bodies) == sent

    def test_resume_body_is_validated(self, client, fake_arms, graph):
        run_id = start(client, arms=["bm25"])["run_id"]
        client.get(f"/api/lab/runs/{run_id}/events")
        assert client.post(f"/api/lab/runs/{run_id}/resume", json={"max_usd": -1}).status_code == 422
