# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The Lab's OpenAI Batch runner, against a fake Batch service.

Hermetic: ``FakeBatchService`` stands in for ``AsyncOpenAI`` (files.create /
files.content / batches.create / retrieve / list / cancel) and runs the
uploaded JSONL itself, so every byte the module would send is inspected and no
request ever leaves the process. The last class drives the real runner through
``runner.ModuleBatchBackend`` end to end.
"""

from __future__ import annotations

import itertools
import json
import re
from types import SimpleNamespace

import pytest

from app.lab import arms as lab_arms
from app.lab import batch, packer, reader, runner
from app.lab.arms import BaseArm, NullClosedBook
from app.lab.evidence import Evidence, EvidenceUnit
from app.lab.runner import DatasetSpec, LabRun
from benchmarks.public import cost


# ── The fake Batch service ───────────────────────────────────────────────────
def completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "gpt-5-nano-2025-08-07",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "completion_tokens_details": {"reasoning_tokens": 12},
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }


class FakeBatchService:
    """A tiny, deterministic OpenAI Batch API.

    ``answer(body) -> str`` produces each completion. ``fail`` holds custom_ids
    that come back in the error file (a 400 from the model endpoint);
    ``expire_after`` makes a batch expire after answering that many lines.
    ``polls_to_finish`` is how many ``retrieve`` calls a batch takes to complete.
    """

    def __init__(self, answer=None, *, fail=(), expire_after=None, polls_to_finish=1):
        self.answer = answer or (lambda body: "ok")
        self.fail = set(fail)
        self.expire_after = expire_after
        self.polls_to_finish = polls_to_finish
        self.files: dict[str, bytes] = {}
        self.uploads: list[dict] = []
        self.batches: dict[str, dict] = {}
        self.created: list[dict] = []
        self.retrieves = 0
        self.fail_next_upload = False
        self.fail_after_create = False
        self._ids = itertools.count(1)
        self.files_api = SimpleNamespace(create=self.files_create, content=self.files_content)
        self.batches_api = SimpleNamespace(create=self.batches_create,
                                           retrieve=self.batches_retrieve,
                                           list=self.batches_list, cancel=self.batches_cancel)

    @property
    def client(self):
        return SimpleNamespace(files=self.files_api, batches=self.batches_api)

    # files
    async def files_create(self, *, file, purpose):
        if self.fail_next_upload:
            self.fail_next_upload = False
            raise ConnectionError("upload dropped")
        name, data = file
        file_id = f"file-{next(self._ids)}"
        self.files[file_id] = bytes(data)
        self.uploads.append({"id": file_id, "name": name, "purpose": purpose,
                             "bytes": bytes(data)})
        return SimpleNamespace(id=file_id)

    async def files_content(self, file_id):
        return SimpleNamespace(content=self.files[file_id], text=self.files[file_id].decode())

    # batches
    def _view(self, b):
        return SimpleNamespace(
            id=b["id"], status=b["status"], input_file_id=b["input_file_id"],
            output_file_id=b.get("output_file_id"), error_file_id=b.get("error_file_id"),
            request_counts=SimpleNamespace(**b["counts"]),
            errors=SimpleNamespace(data=[SimpleNamespace(**e) for e in b.get("errors", [])])
            if b.get("errors") else None,
            created_at=1, completed_at=None, expires_at=86_401, metadata=b["metadata"],
        )

    async def batches_create(self, *, input_file_id, endpoint, completion_window, metadata):
        batch_id = f"batch_{next(self._ids)}"
        lines = self.files[input_file_id].decode().strip("\n").split("\n")
        b = {"id": batch_id, "status": "validating", "input_file_id": input_file_id,
             "endpoint": endpoint, "window": completion_window, "metadata": metadata,
             "counts": {"total": len(lines), "completed": 0, "failed": 0}, "polls": 0}
        self.batches[batch_id] = b
        self.created.append(b)
        if self.fail_after_create:  # the server created it; the reply was lost
            self.fail_after_create = False
            raise ConnectionError("reply lost after create")
        return self._view(b)

    async def batches_list(self, limit=20):
        return SimpleNamespace(data=[self._view(b) for b in reversed(self.created)][:limit])

    async def batches_cancel(self, batch_id):
        b = self.batches[batch_id]
        b["status"] = "cancelled"
        return self._view(b)

    async def batches_retrieve(self, batch_id):
        self.retrieves += 1
        b = self.batches[batch_id]
        if b["status"] in ("completed", "expired", "cancelled", "failed"):
            return self._view(b)
        b["polls"] += 1
        if b["polls"] < self.polls_to_finish:
            b["status"] = "in_progress"
            return self._view(b)
        self._run(b)
        return self._view(b)

    def _run(self, b):
        out, err = [], []
        lines = [json.loads(x) for x in
                 self.files[b["input_file_id"]].decode().strip("\n").split("\n")]
        for i, line in enumerate(lines):
            if self.expire_after is not None and i >= self.expire_after:
                break
            cid = line["custom_id"]
            if cid in self.fail:
                err.append({"id": f"req_{i}", "custom_id": cid, "error": None,
                            "response": {"status_code": 400, "request_id": "r",
                                         "body": {"error": {"message": "bad request",
                                                            "code": "invalid"}}}})
            else:
                out.append({"id": f"req_{i}", "custom_id": cid, "error": None,
                            "response": {"status_code": 200, "request_id": "r",
                                         "body": completion(self.answer(line["body"]))}})
        for records, key in ((out, "output_file_id"), (err, "error_file_id")):
            if records:
                file_id = f"file-{next(self._ids)}"
                self.files[file_id] = "".join(json.dumps(r) + "\n" for r in records).encode()
                b[key] = file_id
        b["counts"] = {"total": len(lines), "completed": len(out), "failed": len(err)}
        b["status"] = "expired" if self.expire_after is not None else "completed"


def line(cid: str, model: str = "gpt-5-nano", content: str = "Q?") -> dict:
    return {"custom_id": cid, "method": "POST", "url": batch.ENDPOINT,
            "body": {"model": model, "messages": [{"role": "user", "content": content}],
                     "max_completion_tokens": 544}}


def lines(n: int, prefix: str = "run|arm|500|q") -> list[dict]:
    return [line(f"{prefix}{i}") for i in range(n)]


def state(run_dir) -> dict:
    return json.loads((run_dir / batch.STATE_FILE).read_text())


# ── Pure pieces ──────────────────────────────────────────────────────────────
class TestValidateAndSplit:
    def test_rejects_malformed_lines(self):
        with pytest.raises(batch.BatchError, match="duplicate custom_id"):
            batch.validate_requests([line("a"), line("a")])
        with pytest.raises(batch.BatchError, match="url"):
            batch.validate_requests([{**line("a"), "url": "/v1/embeddings"}])
        with pytest.raises(batch.BatchError, match="POST"):
            batch.validate_requests([{**line("a"), "method": "GET"}])
        with pytest.raises(batch.BatchError, match="model"):
            batch.validate_requests([{**line("a"), "body": {"messages": []}}])
        with pytest.raises(batch.BatchError, match="custom_id"):
            batch.validate_requests([{**line("a"), "custom_id": ""}])

    def test_documented_limits(self):
        # What the installed openai SDK documents for batches.create.
        assert batch.MAX_REQUESTS_PER_BATCH == 50_000
        assert batch.MAX_BATCH_FILE_BYTES == 200_000_000

    def test_splits_on_request_count_bytes_and_model(self):
        reqs = lines(5) + [line("other", model="gpt-4o-mini")]
        parts = batch.plan_parts(reqs, max_requests=2)
        assert [len(p) for p in parts] == [2, 2, 1, 1]
        assert {p[0]["body"]["model"] for p in parts[-1:]} == {"gpt-4o-mini"}
        assert [x["custom_id"] for p in parts[:3] for x in p] == [f"run|arm|500|q{i}"
                                                                  for i in range(5)]
        size = len(batch.encode_line(reqs[0]))
        assert [len(p) for p in batch.plan_parts(reqs[:5], max_bytes=size * 2)] == [2, 2, 1]

    def test_a_request_above_the_file_limit_is_refused(self):
        with pytest.raises(batch.BatchError, match="byte batch file limit"):
            batch.plan_parts([line("a", content="x" * 500)], max_bytes=100)

    def test_token_gate_sizes_parts_only_when_set(self):
        counted = []

        def tokens_of(x):
            counted.append(x["custom_id"])
            return 10

        assert len(batch.plan_parts(lines(4), tokens_of=tokens_of)) == 1 and not counted
        parts = batch.plan_parts(lines(4), max_tokens=25, tokens_of=tokens_of)
        assert [len(p) for p in parts] == [2, 2]


# ── Submit ───────────────────────────────────────────────────────────────────
class TestSubmit:
    async def test_uploads_jsonl_and_creates_a_24h_chat_batch(self, tmp_path):
        svc = FakeBatchService()
        reqs = lines(3)
        info = await batch.submit(tmp_path, reqs, run_id="run-1", client=svc.client)

        assert len(svc.uploads) == 1 and svc.uploads[0]["purpose"] == "batch"
        assert svc.uploads[0]["bytes"] == b"".join(batch.encode_line(r) for r in reqs)
        created = svc.created[0]
        assert created["endpoint"] == "/v1/chat/completions" and created["window"] == "24h"
        assert created["metadata"]["synapse_run"] == "run-1"
        assert info["batch_ids"] == [created["id"]] and info["requests"] == 3
        st = state(tmp_path)
        assert st["parts"][0]["batch_id"] == created["id"]
        assert st["parts"][0]["input_file_id"] == svc.uploads[0]["id"]
        # The exact uploaded bytes are kept locally.
        part_file = tmp_path / "batches" / "part-0000.input.jsonl"
        assert part_file.read_bytes() == svc.uploads[0]["bytes"]

    async def test_resubmitting_open_lines_is_idempotent(self, tmp_path):
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(3), run_id="r", client=svc.client)
        info = await batch.submit(tmp_path, lines(3), run_id="r", client=svc.client)
        assert len(svc.uploads) == 1 and len(svc.created) == 1
        assert info["reused"] == 1 and len(state(tmp_path)["parts"]) == 1

    async def test_a_request_in_flight_is_never_sent_twice(self, tmp_path):
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(3), run_id="r", client=svc.client)
        with pytest.raises(batch.BatchError, match="already in flight"):
            await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        assert len(svc.created) == 1

    async def test_a_lost_create_reply_is_recovered_not_duplicated(self, tmp_path):
        svc = FakeBatchService()
        svc.fail_after_create = True
        with pytest.raises(ConnectionError):
            await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        assert state(tmp_path)["parts"][0].get("batch_id") is None
        info = await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        assert len(svc.created) == 1 and info["batch_ids"] == [svc.created[0]["id"]]

    async def test_a_failed_upload_is_retried_on_the_next_submit(self, tmp_path):
        svc = FakeBatchService()
        svc.fail_next_upload = True
        with pytest.raises(ConnectionError):
            await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        assert state(tmp_path)["parts"][0]["status"] == batch.LOCAL
        await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        assert len(svc.uploads) == 1 and len(svc.created) == 1

    async def test_large_submissions_are_split(self, tmp_path):
        svc = FakeBatchService()
        info = await batch.submit(tmp_path, lines(5), run_id="r", client=svc.client,
                                  max_requests=2)
        assert len(svc.created) == 3 and len(info["batch_ids"]) == 3
        assert [p["requests"] for p in info["parts"]] == [2, 2, 1]

    async def test_line_separators_inside_text_survive_the_round_trip(self, tmp_path):
        tricky = "a\u2028b\u2029c\x85d"  # raw in the JSONL: json keeps them unescaped
        reqs = [line("x", content=tricky), line("y")]
        svc = FakeBatchService(answer=lambda body: body["messages"][0]["content"])
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        assert "\u2028".encode() in svc.uploads[0]["bytes"]
        await batch.poll(tmp_path, client=svc.client)
        records = await batch.collect(tmp_path, client=svc.client)
        assert [r["custom_id"] for r in records] == ["x", "y"]
        assert batch.parse_results(records)["x"].answer == tricky
        assert batch.failed_requests(tmp_path) == []

    async def test_nothing_to_send_makes_no_call(self, tmp_path):
        info = await batch.submit(tmp_path, [], run_id="r", client=None)
        assert info["requests"] == 0 and not (tmp_path / batch.STATE_FILE).exists()

    async def test_without_a_client_it_refuses_under_pytest(self, tmp_path):
        with pytest.raises(RuntimeError, match="refusing to create a real OpenAI client"):
            await batch.submit(tmp_path, lines(1), run_id="r")


# ── Poll / collect ───────────────────────────────────────────────────────────
class TestPollAndCollect:
    async def test_poll_reports_status_and_counts_until_done(self, tmp_path):
        svc = FakeBatchService(polls_to_finish=2)
        await batch.submit(tmp_path, lines(3), run_id="r", client=svc.client)
        first = await batch.poll(tmp_path, client=svc.client)
        assert first["done"] is False and first["status"] == "in_progress"
        second = await batch.poll(tmp_path, client=svc.client)
        assert second["done"] is True and second["status"] == "completed"
        assert second["request_counts"] == {"total": 3, "completed": 3, "failed": 0}
        # Terminal batches are not polled again.
        before = svc.retrieves
        await batch.poll(tmp_path, client=svc.client)
        assert svc.retrieves == before

    async def test_collect_refuses_while_running(self, tmp_path):
        svc = FakeBatchService(polls_to_finish=3)
        await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        with pytest.raises(batch.BatchError, match="not finished"):
            await batch.collect(tmp_path, client=svc.client)

    async def test_collect_returns_every_request_in_order_with_batch_pricing(self, tmp_path):
        reqs = lines(4)
        svc = FakeBatchService(answer=lambda body: "Paris", fail={reqs[1]["custom_id"]})
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)
        records = await batch.collect(tmp_path, client=svc.client)

        assert [r["custom_id"] for r in records] == [r["custom_id"] for r in reqs]
        assert [batch.record_ok(r) for r in records] == [True, False, True, True]
        assert "bad request" in batch.record_error(records[1])
        assert runner.result_from_record(records[0]).answer == "Paris"
        part = state(tmp_path)["parts"][0]
        assert part["collected_at"] and part["succeeded"] == 3 and part["failed"] == 1
        realtime = cost.usd(cost.Usage(prompt_tokens=300, completion_tokens=60),
                            cost.resolve_price("gpt-5-nano"))
        assert part["usd"] == pytest.approx(realtime * cost.BATCH_PRICE_MULTIPLIER)
        assert batch.status(tmp_path)["usd"] == pytest.approx(part["usd"])
        # Downloaded once, then re-readable offline.
        assert (tmp_path / "batches" / "part-0000.output.jsonl").exists()
        assert (tmp_path / "batches" / "part-0000.errors.jsonl").exists()
        assert await batch.collect(tmp_path, client=svc.client) == []
        again = await batch.collect(tmp_path, client=None, include_collected=True)
        assert again == records

    async def test_requests_an_expired_batch_never_ran_come_back_as_errors(self, tmp_path):
        svc = FakeBatchService(expire_after=2)
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client)
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["done"] and status["status"] == "expired"
        records = await batch.collect(tmp_path, client=svc.client)
        assert len(records) == 4
        assert [batch.record_ok(r) for r in records] == [True, True, False, False]
        assert records[2]["synthesized"] and "expired" in batch.record_error(records[2])
        assert runner.result_from_record(records[3]).error

    async def test_a_batch_that_fails_validation_returns_every_request_as_an_error(
        self, tmp_path
    ):
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        b = svc.created[0]
        b["status"] = "failed"
        b["errors"] = [{"code": "token_limit_exceeded", "message": "Enqueued token limit "
                        "reached", "line": None}]
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["done"] and status["status"] == "failed"
        assert status["parts"][0]["batch_errors"][0]["code"] == "token_limit_exceeded"
        records = await batch.collect(tmp_path, client=svc.client)
        assert all("token_limit_exceeded" in batch.record_error(r) for r in records)
        assert len(batch.failed_requests(tmp_path)) == 2

    def test_parse_results_gives_answers_and_usage(self):
        ok = {"custom_id": "a", "error": None,
              "response": {"status_code": 200, "body": completion("Paris")}}
        bad = {"custom_id": "b", "error": {"code": "server_error", "message": "boom"},
               "response": None}
        parsed = batch.parse_results([ok, bad])
        assert parsed["a"].answer == "Paris" and parsed["a"].prompt_tokens == 100
        assert parsed["a"].reasoning_tokens == 12 and parsed["a"].ok
        assert parsed["b"].error == "server_error: boom" and not parsed["b"].ok

    async def test_failed_requests_are_listed_and_resubmitted_alone(self, tmp_path):
        reqs = lines(3)
        svc = FakeBatchService(fail={reqs[2]["custom_id"]})
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)
        await batch.collect(tmp_path, client=svc.client)
        assert [x["custom_id"] for x in batch.failed_requests(tmp_path)] == [reqs[2]["custom_id"]]

        svc.fail.clear()
        info = await batch.resubmit_failed(tmp_path, client=svc.client)
        assert info["requests"] == 1 and info["submission"] == 2
        assert svc.uploads[-1]["bytes"] == batch.encode_line(reqs[2])
        assert batch.failed_requests(tmp_path) == []  # in flight, not listed twice
        await batch.poll(tmp_path, client=svc.client)
        retried = await batch.collect(tmp_path, client=svc.client)
        assert len(retried) == 1 and batch.record_ok(retried[0])
        latest = batch.latest_records(tmp_path)
        assert all(batch.record_ok(r) for r in latest.values()) and len(latest) == 3
        assert batch.failed_requests(tmp_path) == []


class TestEnqueuedTokenGate:
    async def test_parts_wait_for_the_gate_and_poll_sends_them(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "request_prompt_tokens", lambda x: 10)
        svc = FakeBatchService()
        info = await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client,
                                  max_enqueued_tokens=25)
        assert len(svc.created) == 1 and info["queued"] == 1
        st = state(tmp_path)
        assert st["max_enqueued_tokens"] == 25 and "waiting" in st["parts"][1]["queued_reason"]

        status = await batch.poll(tmp_path, client=svc.client)  # part 0 completes → part 1 sent
        assert len(svc.created) == 2 and status["done"] is False
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["done"] is True
        assert len(await batch.collect(tmp_path, client=svc.client)) == 4

    async def test_the_gate_can_come_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(batch.ENQUEUED_TOKENS_ENV, "25")
        monkeypatch.setattr(batch, "request_prompt_tokens", lambda x: 10)
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client)
        assert len(svc.created) == 1 and len(state(tmp_path)["parts"]) == 2


class TestCancel:
    async def test_cancels_in_flight_batches_and_unsent_parts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "request_prompt_tokens", lambda x: 10)
        svc = FakeBatchService(polls_to_finish=5)
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client,
                           max_enqueued_tokens=25)
        result = await batch.cancel(tmp_path, client=svc.client)
        assert result["cancelled"] == ["part-0000", "part-0001"]
        assert len(svc.uploads) == 1  # the queued part was never sent
        assert (await batch.poll(tmp_path, client=svc.client))["done"] is True
        records = await batch.collect(tmp_path, client=svc.client)
        assert len(records) == 4 and not any(batch.record_ok(r) for r in records)


# ── Through the real runner ──────────────────────────────────────────────────
DEMO = {r["question"]: r["answer"] for r in json.loads(runner.DEMO_QA_PATH.read_text())}
WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH)\b")


class AnswerArm(BaseArm):
    name, family, title = "bm25", "passage", "bm25"
    description = "fake"
    source = {"citation": "test", "url": "https://example.org"}

    async def retrieve(self, question, *, k, seed):
        answer = DEMO.get(question, "nothing")
        answer = answer[0] if isinstance(answer, list) else answer
        return Evidence([EvidenceUnit(f"The answer is {answer}", "prose", "d", 1.0)], self.name)


def answer_from_body(body):
    match = re.search(r"The answer is ([^\n]+)", body["messages"][1]["content"])
    return match.group(1) if match else "unknown"


@pytest.fixture
def lab(monkeypatch, fake_neo4j):
    monkeypatch.setattr(runner, "ARMS", {"null_closed_book": NullClosedBook(),
                                         "bm25": AnswerArm()})
    words = lambda text, model=None: (len(text.split()), False)  # noqa: E731
    monkeypatch.setattr(packer, "count_tokens", words)
    monkeypatch.setattr(reader, "count_tokens", words)
    lab_arms.clear_caches()

    def handler(q, p):
        assert not WRITE.search(q), f"the Lab must never write to the graph: {q}"
        for _key, query in lab_arms.FINGERPRINT_QUERIES.items():
            if q == query:
                return [{"n": 5}]
        return []

    return fake_neo4j(handler)


class TestRunnerIntegration:
    async def test_a_batch_run_submits_resumes_and_scores(self, tmp_path, lab):
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=2)
        backend = runner.ModuleBatchBackend(svc.client)
        run = LabRun(dataset=DatasetSpec("demo"), arms=["null_closed_book", "bm25"],
                     budgets=[500], mode="batch", max_usd=1.0, run_dir=tmp_path / "run",
                     reader_model="gpt-5-nano")
        manifest = await runner.run_lab(run, batch_backend=backend)
        assert manifest["status"] == "batch_submitted"
        assert len(svc.created) == 1
        requests = runner.read_jsonl(tmp_path / "run" / runner.REQUESTS)
        assert svc.uploads[0]["bytes"] == b"".join(batch.encode_line(r) for r in requests)

        manifest = await runner.resume(tmp_path / "run", batch_backend=backend)
        assert manifest["status"] == "batch_submitted"  # still in progress
        manifest = await runner.resume(tmp_path / "run", batch_backend=backend)
        assert manifest["status"] == "done"
        assert manifest["phases"]["read"]["status"] == "done"

        responses = runner.read_jsonl(tmp_path / "run" / runner.RESPONSES)
        assert len(responses) == len(requests) and all(r["batch"] for r in responses)
        board = json.loads((tmp_path / "run" / runner.LEADERBOARD).read_text())
        bm25 = next(r for r in board["rows"] if r["arm"] == "bm25")
        assert bm25["em"] == pytest.approx(100.0)  # EM in points
        # The reader spend recorded by the runner is the batch rate.
        expected = cost.usd(cost.Usage(prompt_tokens=100, completion_tokens=20),
                            cost.resolve_price("gpt-5-nano"), batch=True)
        assert responses[0]["usd"] == pytest.approx(expected)

    @staticmethod
    def _batch_run(tmp_path):
        return LabRun(dataset=DatasetSpec("demo"), arms=["null_closed_book", "bm25"],
                      budgets=[500], mode="batch", max_usd=1.0, run_dir=tmp_path / "run",
                      reader_model="gpt-5-nano")

    async def test_answers_survive_a_crash_right_after_collect(
        self, tmp_path, lab, monkeypatch
    ):
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = runner.ModuleBatchBackend(svc.client)
        await runner.run_lab(self._batch_run(tmp_path), batch_backend=backend)
        root = tmp_path / "run"
        requests = runner.read_jsonl(root / runner.REQUESTS)

        real_append = runner._append_jsonl

        def crash_once(path, records):
            if path.name == runner.RESPONSES:
                monkeypatch.setattr(runner, "_append_jsonl", real_append)
                raise KeyboardInterrupt("killed after batch.collect marked the part collected")
            return real_append(path, records)

        monkeypatch.setattr(runner, "_append_jsonl", crash_once)
        with pytest.raises(KeyboardInterrupt):
            await runner.resume(root, batch_backend=backend)
        assert batch.status(root)["status"] == "collected"  # batch.py already handed it over

        manifest = await runner.resume(root, batch_backend=backend)
        responses = runner.read_jsonl(root / runner.RESPONSES)
        assert manifest["status"] == "done" and manifest["phases"]["read"]["status"] == "done"
        assert sorted(r["custom_id"] for r in responses) == sorted(
            r["custom_id"] for r in requests)
        assert manifest["actual"]["reader"]["usd"] == pytest.approx(batch.status(root)["usd"])
        # Re-scoring never records an answer twice (nor bills it twice).
        again = await runner.resume(root, batch_backend=backend)
        assert len(runner.read_jsonl(root / runner.RESPONSES)) == len(requests)
        assert again["actual"]["reader"]["usd"] == pytest.approx(batch.status(root)["usd"])

    async def test_a_failed_download_of_a_later_part_loses_no_earlier_part(
        self, tmp_path, lab, monkeypatch
    ):
        real_plan = batch.plan_parts
        monkeypatch.setattr(batch, "plan_parts",
                            lambda lines, **kw: real_plan(lines, **{**kw, "max_requests": 10}))
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = runner.ModuleBatchBackend(svc.client)
        await runner.run_lab(self._batch_run(tmp_path), batch_backend=backend)
        assert len(svc.created) == 2
        root = tmp_path / "run"
        requests = runner.read_jsonl(root / runner.REQUESTS)

        real_content = svc.files_api.content
        downloads = {"n": 0}

        async def flaky(file_id):
            downloads["n"] += 1
            if downloads["n"] == 2:
                raise ConnectionError("download dropped")
            return await real_content(file_id)

        svc.files_api.content = flaky
        with pytest.raises(ConnectionError):
            await runner.resume(root, batch_backend=backend)
        manifest = await runner.resume(root, batch_backend=backend)
        responses = runner.read_jsonl(root / runner.RESPONSES)
        assert manifest["status"] == "done" and len(responses) == len(requests) == 18
        assert manifest["actual"]["reader"]["usd"] == pytest.approx(batch.status(root)["usd"])

    async def test_read_status_counts_every_request_not_just_the_last_collect(
        self, tmp_path, lab, monkeypatch
    ):
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = runner.ModuleBatchBackend(svc.client)
        await runner.run_lab(self._batch_run(tmp_path), batch_backend=backend)
        root = tmp_path / "run"
        # The part is collected by someone else (e.g. `lab resume` killed mid-way):
        # the runner's own collect() then returns nothing new.
        await batch.poll(root, client=svc.client)
        await batch.collect(root, client=svc.client)
        monkeypatch.setattr(backend, "collected", None, raising=False)
        manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["phases"]["read"]["status"] == "done_with_errors"
        assert manifest["phases"]["read"]["failed"] == 18

