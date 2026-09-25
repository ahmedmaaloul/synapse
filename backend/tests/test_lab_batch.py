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


def token_limit_message(model: str, limit: int) -> str:
    """OpenAI's batch-level error, verbatim in shape (2026-09-25 incident)."""
    return (f"Enqueued token limit reached for {model} in organization org-AbC123xyz. "
            f"Limit: {limit:,} enqueued tokens. Please try again once some in_progress "
            "batches have been completed.")


class FakeBatchService:
    """A tiny, deterministic OpenAI Batch API.

    ``answer(body) -> str`` produces each completion. ``fail`` holds custom_ids
    that come back in the error file (a 400 from the model endpoint);
    ``expire_after`` makes a batch expire after answering that many lines.
    ``polls_to_finish`` is how many ``retrieve`` calls a batch takes to complete.
    ``enqueued_limit`` is the organisation's enqueued-token limit: a batch whose
    prompt tokens (``tokens_of`` per line), plus those of the batches still in
    flight, exceed it is created and then FAILS VALIDATION on its first
    retrieve with OpenAI's ``token_limit_exceeded`` error — nothing runs.
    """

    def __init__(self, answer=None, *, fail=(), expire_after=None, polls_to_finish=1,
                 enqueued_limit=None, tokens_of=None):
        self.answer = answer or (lambda body: "ok")
        self.fail = set(fail)
        self.expire_after = expire_after
        self.polls_to_finish = polls_to_finish
        self.enqueued_limit = enqueued_limit
        self.tokens_of = tokens_of or (lambda request: batch.request_prompt_tokens(request))
        self.max_enqueued_seen = 0
        self.rejected: list[str] = []
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
        if self.enqueued_limit is not None:
            b["tokens"] = sum(self.tokens_of(json.loads(x)) for x in lines)
            enqueued = sum(o.get("tokens", 0) for o in self.batches.values()
                           if o["status"] in ("validating", "in_progress", "finalizing")
                           and not o.get("reject"))
            if enqueued + b["tokens"] > self.enqueued_limit:
                b["reject"] = True  # refused when OpenAI validates it
            else:
                self.max_enqueued_seen = max(self.max_enqueued_seen, enqueued + b["tokens"])
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
        if b.get("reject"):
            model = json.loads(self.files[b["input_file_id"]].decode().split("\n")[0])
            b.update(status="failed", counts={"total": 0, "completed": 0, "failed": 0},
                     errors=[{"code": "token_limit_exceeded", "line": None,
                              "message": token_limit_message(model["body"]["model"],
                                                             self.enqueued_limit)}])
            self.rejected.append(batch_id)
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


class SubmitPollCollectOnly:
    """A Batch backend with ONLY the three required methods (no ``collected``).

    What a thin wrapper around ``ModuleBatchBackend`` looks like when it forwards
    just the protocol (the iso-token harness's guard did, until 2026-09-25).
    """

    def __init__(self, inner):
        self.inner = inner

    async def submit(self, run_dir, requests, *, run_id):
        return await self.inner.submit(run_dir, requests, run_id=run_id)

    async def poll(self, run_dir):
        return await self.inner.poll(run_dir)

    async def collect(self, run_dir):
        return await self.inner.collect(run_dir)


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

    async def test_a_failed_batch_with_no_batch_error_is_collected_as_errors(self, tmp_path):
        # Not a documented rejection (no batch-level error): nothing is guessed,
        # every request comes back as a synthesized error, as for an expired batch.
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        svc.created[0]["status"] = "failed"
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["done"] and status["status"] == "failed" and status["rejected"] == []
        records = await batch.collect(tmp_path, client=svc.client)
        assert all("batch_failed" in batch.record_error(r) for r in records)
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


class TestRejectedAtValidation:
    """A batch OpenAI refuses at validation (the 2026-09-25 pilot incident)."""

    @pytest.fixture(autouse=True)
    def ten_tokens(self, monkeypatch):
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        monkeypatch.setattr(batch, "request_prompt_tokens", lambda x: 10)

    @pytest.mark.parametrize(("message", "limit"), [
        (token_limit_message("gpt-5-nano", 2_000_000), 2_000_000),
        ("Enqueued token limit reached for gpt-5-nano in organization org-x. "
         "Limit: 90000 enqueued tokens.", 90_000),
        ("Limit: 1_500_000 tokens", 1_500_000),
        ("Enqueued token limit reached.", None),
        (None, None),
    ])
    def test_parse_enqueued_limit(self, message, limit):
        assert batch.parse_enqueued_limit(message) == limit

    async def test_a_rejected_part_is_never_collected_costs_nothing_and_sets_the_gate(
        self, tmp_path
    ):
        svc = FakeBatchService(enqueued_limit=25)
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client)  # 40 > 25
        status = await batch.poll(tmp_path, client=svc.client)

        assert status["done"] is True and status["status"] == "rejected"
        assert status["parts"] == []  # not an open part, not per-request failures
        (rej,) = status["rejected"]
        assert rej["code"] == "token_limit_exceeded" and rej["limit"] == 25
        assert rej["retryable"] and rej["requests"] == 4
        assert "org-AbC123xyz" not in json.dumps(state(tmp_path))  # org id redacted
        part = state(tmp_path)["parts"][0]
        assert part["status"] == "rejected" and part["openai_status"] == "failed"
        assert part["usd"] == 0.0 and not part.get("collected_at")
        # The gate: floor(0.9 × 25), persisted and explained in batches.json.
        st = state(tmp_path)
        assert st["max_enqueued_tokens"] == 22 and st["max_enqueued_tokens_source"] == "auto"
        assert st["enqueued_gate"]["limit_reported"] == 25 and st["enqueued_gate"]["applied"]
        # Nothing to collect, no result, no spend — and the requests are resendable.
        assert await batch.collect(tmp_path, client=svc.client) == []
        assert batch.latest_records(tmp_path) == {}
        summary = batch.status(tmp_path)
        assert summary["status"] == "rejected" and summary["usd"] == 0
        assert summary["request_counts"]["total"] == 0
        assert len(batch.failed_requests(tmp_path)) == 4
        assert len(batch.rejected_requests(tmp_path)) == 4

    async def test_rejected_requests_go_again_in_sequential_parts_under_the_gate(self, tmp_path):
        svc = FakeBatchService(enqueued_limit=25)
        await batch.submit(tmp_path, lines(5), run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)

        info = await batch.resubmit_failed(tmp_path, client=svc.client)
        assert info["submission"] == 2 and info["max_enqueued_tokens"] == 22
        assert [p["requests"] for p in info["parts"]] == [2, 2, 1]  # 20 + 20 + 10 tokens
        assert len(svc.created) == 2 and info["queued"] == 2  # one part in flight at a time
        st = state(tmp_path)
        assert st["parts"][0]["resubmitted_in"] == ["part-0001", "part-0002", "part-0003"]
        assert batch.failed_requests(tmp_path) == []  # in flight, never listed twice

        for _ in range(10):
            status = await batch.poll(tmp_path, client=svc.client)
            assert status["rejected"] == []  # the rejection is resolved
            if status["done"]:
                break
        records = await batch.collect(tmp_path, client=svc.client)
        assert len(records) == 5 and all(batch.record_ok(r) for r in records)
        assert svc.max_enqueued_seen <= 25 and svc.rejected == [svc.created[0]["id"]]
        sent = [json.loads(x)["custom_id"] for u in svc.uploads[1:]
                for x in u["bytes"].decode().strip().split("\n")]
        assert sorted(sent) == sorted(r["custom_id"] for r in lines(5))  # each once
        assert batch.failed_requests(tmp_path) == []
        assert batch.status(tmp_path)["status"] == "collected"

    async def test_a_part_the_gate_let_through_but_still_refused_tightens_the_gate(
        self, tmp_path
    ):
        # OpenAI counts twice what we do: the first auto gate (22) is still too big.
        svc = FakeBatchService(enqueued_limit=25, tokens_of=lambda request: 20)
        await batch.submit(tmp_path, lines(3), run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)
        await batch.resubmit_failed(tmp_path, client=svc.client)  # parts of 20 (ours)
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["rejected"][0]["part"] == "part-0001"
        assert state(tmp_path)["max_enqueued_tokens"] == 18  # floor(0.9 × 20)
        # The part queued behind it (sized under 22) is sent once nothing is in flight.
        for _ in range(5):
            if (await batch.poll(tmp_path, client=svc.client))["done"]:
                break
        await batch.collect(tmp_path, client=svc.client)
        await batch.resubmit_failed(tmp_path, client=svc.client)  # parts of 10 now
        for _ in range(10):
            if (await batch.poll(tmp_path, client=svc.client))["done"]:
                break
        await batch.collect(tmp_path, client=svc.client)
        assert len(batch.latest_records(tmp_path)) == 3
        assert all(batch.record_ok(r) for r in batch.latest_records(tmp_path).values())
        assert svc.max_enqueued_seen <= 25

    async def test_an_explicit_gate_wins_over_the_reported_limit(self, tmp_path):
        svc = FakeBatchService(enqueued_limit=25)
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client,
                           max_enqueued_tokens=30)  # parts of 30 + 10
        status = await batch.poll(tmp_path, client=svc.client)
        assert status["rejected"][0]["part"] == "part-0000"
        st = state(tmp_path)
        assert st["max_enqueued_tokens"] == 30 and st["max_enqueued_tokens_source"] == "explicit"
        assert st["enqueued_gate"]["applied"] is False and "above" in st["enqueued_gate"][
            "warning"]

    async def test_the_environment_gate_wins_over_the_reported_limit(self, tmp_path, monkeypatch):
        svc = FakeBatchService(enqueued_limit=25)
        await batch.submit(tmp_path, lines(4), run_id="r", client=svc.client)
        monkeypatch.setenv(batch.ENQUEUED_TOKENS_ENV, "15")
        await batch.poll(tmp_path, client=svc.client)
        st = state(tmp_path)
        assert st["max_enqueued_tokens"] == 15 and st["max_enqueued_tokens_source"] == "env"
        assert st["enqueued_gate"]["applied"] is False

    async def test_a_malformed_file_is_rejected_but_sets_no_gate(self, tmp_path):
        svc = FakeBatchService()
        await batch.submit(tmp_path, lines(2), run_id="r", client=svc.client)
        svc.created[0].update(status="failed", errors=[
            {"code": "invalid_json_line", "message": "line 2 is not valid JSON", "line": 2}])
        status = await batch.poll(tmp_path, client=svc.client)
        (rej,) = status["rejected"]
        assert rej["code"] == "invalid_json_line" and rej["retryable"] is False
        assert "max_enqueued_tokens" not in state(tmp_path)
        assert await batch.collect(tmp_path, client=svc.client) == []

    async def test_an_answered_request_is_never_sent_again(self, tmp_path):
        svc = FakeBatchService()
        reqs = lines(3)
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)
        await batch.collect(tmp_path, client=svc.client)
        info = await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        assert info["requests"] == 0 and info["already_answered"] == 3
        assert len(svc.created) == 1

    async def test_a_refused_resend_of_failed_requests_stays_owed_until_it_is_sent(
        self, tmp_path
    ):
        # Counterexample (review, 2026-09-25): 2 of 5 extractions failed, their resend
        # was itself refused at validation — and rejected_requests() then listed
        # NOTHING (the ids had an error line), so the rejection never cleared and the
        # harness looped on "re-submitted 0 request(s)" forever.
        reqs = lines(5)
        failing = {reqs[1]["custom_id"], reqs[3]["custom_id"]}
        svc = FakeBatchService(fail=failing)
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        await batch.poll(tmp_path, client=svc.client)
        await batch.collect(tmp_path, client=svc.client)
        svc.fail.clear()
        svc.enqueued_limit = 15
        assert (await batch.resubmit_failed(tmp_path, client=svc.client))["requests"] == 2
        status = await batch.poll(tmp_path, client=svc.client)  # 20 tokens > 15: refused
        assert [r["part"] for r in status["rejected"]] == ["part-0001"]

        owed = batch.rejected_requests(tmp_path)
        assert sorted(x["custom_id"] for x in owed) == sorted(failing)  # still owed
        info = await batch.submit(tmp_path, owed, run_id="r", client=svc.client)
        assert info["requests"] == 2 and info["max_enqueued_tokens"] == 13  # one per part
        assert batch.status(tmp_path)["rejected"] == []  # in flight again: resolved
        for _ in range(10):
            if (await batch.poll(tmp_path, client=svc.client))["done"]:
                break
        await batch.collect(tmp_path, client=svc.client)
        latest = batch.latest_records(tmp_path)
        assert len(latest) == 5 and all(batch.record_ok(r) for r in latest.values())
        assert batch.status(tmp_path)["status"] == "collected"
        assert batch.rejected_requests(tmp_path) == [] and batch.failed_requests(tmp_path) == []

    async def test_a_rejection_whose_requests_are_answered_is_resolved_by_poll(self, tmp_path):
        svc = FakeBatchService(enqueued_limit=25)
        reqs = lines(3)
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)  # 30 > 25
        await batch.poll(tmp_path, client=svc.client)
        await batch.resubmit_failed(tmp_path, client=svc.client)
        st = state(tmp_path)
        st["parts"][0].pop("resubmitted_at")  # a state written before the resend resolved it
        batch._save_state(tmp_path, st)
        assert batch.status(tmp_path)["rejected"]
        for _ in range(10):
            if (await batch.poll(tmp_path, client=svc.client))["done"]:
                break
        assert batch.status(tmp_path)["rejected"] == []
        await batch.collect(tmp_path, client=svc.client)
        info = await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)
        assert info["requests"] == 0 and info["already_answered"] == 3

    async def test_queued_parts_are_replanned_when_the_gate_tightens(self, tmp_path):
        # Counterexample (review, 2026-09-25): OpenAI counts 30% more than we do. The
        # first part the auto gate let through is refused and the gate tightens — but
        # the parts already queued under the OLD gate were still sent, each above the
        # new gate, each predictably refused (one owner re-run apiece).
        svc = FakeBatchService(enqueued_limit=100, tokens_of=lambda request: 13)
        sent: list[tuple[str, int, int | None, int | None]] = []
        real_create = svc.batches_api.create

        async def create(**kw):
            st = state(tmp_path)
            part = next(p for p in st["parts"] if p.get("input_file_id") == kw["input_file_id"])
            sent.append((part["name"], part["requests"], part.get("prompt_tokens"),
                         st.get("max_enqueued_tokens")))
            return await real_create(**kw)

        svc.batches_api.create = create
        reqs = lines(30)
        await batch.submit(tmp_path, reqs, run_id="r", client=svc.client)  # 390 > 100
        for _ in range(60):
            status = await batch.poll(tmp_path, client=svc.client)
            if not status["done"]:
                continue
            await batch.collect(tmp_path, client=svc.client)
            if not batch.failed_requests(tmp_path):
                break
            await batch.resubmit_failed(tmp_path, client=svc.client)
        latest = batch.latest_records(tmp_path)
        assert len(latest) == 30 and all(batch.record_ok(r) for r in latest.values())
        # Never a multi-request part above the gate in force when it was sent.
        over = [s for s in sent[1:] if s[1] > 1 and s[2] is not None and s[3] and s[2] > s[3]]
        assert over == []
        st = state(tmp_path)
        replanned = [p for p in st["parts"] if p["status"] == batch.REPLANNED]
        assert replanned and all(p["replanned_into"] and not p.get("batch_id")
                                 for p in replanned)
        # Refused: the first file, then only parts that fit the gate of their time.
        assert len(svc.rejected) == 3 and st["max_enqueued_tokens"] == 72
        for part in st["parts"]:  # a re-sent rejection never points at a re-planned part
            assert not set(part.get("resubmitted_in") or []) & {p["name"] for p in replanned}
        received = [json.loads(x)["custom_id"] for b in svc.created if not b.get("reject")
                    for x in svc.files[b["input_file_id"]].decode().strip().split("\n")]
        assert sorted(received) == sorted(r["custom_id"] for r in reqs)  # each ran once
        summary = batch.status(tmp_path)
        assert summary["status"] == "collected"
        assert summary["request_counts"]["total"] == 30  # a re-planned part counts nothing


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
        # The part is collected by someone else (e.g. `lab resume` killed mid-way)
        # and its output file is then lost: the runner's own collect() returns
        # nothing new and nothing can be recovered.
        await batch.poll(root, client=svc.client)
        await batch.collect(root, client=svc.client)
        (root / batch.PARTS_DIR / "part-0000.output.jsonl").unlink()
        monkeypatch.setattr(backend, "collected", None, raising=False)
        manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["phases"]["read"]["status"] == "done_with_errors"
        assert manifest["phases"]["read"]["failed"] == 18

    @pytest.mark.parametrize("interruption", ["download", "append"])
    async def test_a_backend_without_collected_loses_no_billed_answer(
        self, tmp_path, lab, monkeypatch, interruption
    ):
        # Counterexample (review, 2026-09-25): the harness wraps the module backend in
        # one with only submit/poll/collect. The final collect handed part 1 over, then
        # a download (or the responses append) failed: its answers were never recorded,
        # submit() then dropped them as "already answered", and no re-run could ever
        # record them (0/18 answered, done_with_errors, for good).
        real_plan = batch.plan_parts
        monkeypatch.setattr(batch, "plan_parts",
                            lambda lines, **kw: real_plan(lines, **{**kw, "max_requests": 10}))
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = SubmitPollCollectOnly(runner.ModuleBatchBackend(svc.client))
        await runner.run_lab(self._batch_run(tmp_path), batch_backend=backend)
        root = tmp_path / "run"
        requests = runner.read_jsonl(root / runner.REQUESTS)
        assert len(svc.created) == 2
        if interruption == "download":
            real_content, downloads = svc.files_api.content, {"n": 0}

            async def flaky(file_id):
                downloads["n"] += 1
                if downloads["n"] == 2:
                    raise ConnectionError("download dropped")
                return await real_content(file_id)

            svc.files_api.content = flaky
            error: type[BaseException] = ConnectionError
        else:
            real_append = runner._append_jsonl

            def crash_once(path, records):
                if path.name == runner.RESPONSES:
                    monkeypatch.setattr(runner, "_append_jsonl", real_append)
                    raise KeyboardInterrupt("killed after batch.collect handed parts over")
                return real_append(path, records)

            monkeypatch.setattr(runner, "_append_jsonl", crash_once)
            error = KeyboardInterrupt
        with pytest.raises(error):
            await runner.resume(root, batch_backend=backend)
        assert batch.load_state(root)["parts"][0].get("collected_at")  # handed over once

        manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["status"] == "done" and manifest["phases"]["read"]["status"] == "done"
        latest = runner._latest_responses(runner.RunPaths(root))
        assert sorted(latest) == sorted(r["custom_id"] for r in requests)
        assert all(not r.get("error") for r in latest.values())
        assert len(runner.read_jsonl(root / runner.RESPONSES)) == len(requests)  # once each
        assert len(svc.created) == 2  # nothing sent (nor billed) twice
        assert manifest["actual"]["reader"]["usd"] == pytest.approx(batch.status(root)["usd"])

    async def test_answers_the_responses_file_lost_are_reconciled_before_a_retry(
        self, tmp_path, lab
    ):
        # A run already scored on missing answers (the dead end the review found): a
        # --retry-failed re-plan records what the batch state already holds, and sends
        # nothing that has an answer.
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = SubmitPollCollectOnly(runner.ModuleBatchBackend(svc.client))
        await runner.run_lab(self._batch_run(tmp_path), batch_backend=backend)
        root = tmp_path / "run"
        await runner.resume(root, batch_backend=backend)
        responses = runner.read_jsonl(root / runner.RESPONSES)
        (root / runner.RESPONSES).write_text("".join(
            json.dumps({**r, "answer": None, "error": "lost"}) + "\n" for r in responses[:5]))
        manifest = json.loads((root / runner.MANIFEST).read_text())
        manifest["phases"]["read"].update(status="done_with_errors", failed=18)
        (root / runner.MANIFEST).write_text(json.dumps(manifest))

        manifest = await runner.resume(root, batch_backend=backend, retry_failed=True)
        assert manifest["status"] == "done" and manifest["phases"]["read"]["status"] == "done"
        assert len(svc.created) == 1
        latest = runner._latest_responses(runner.RunPaths(root))
        assert len(latest) == 18 and all(not r.get("error") for r in latest.values())



class TestRunnerRecoversFromARejectedBatch:
    """The reader batch refused for the organisation's enqueued-token limit.

    Before: every request came back as a synthesized error, the run was scored
    (``done_with_errors``) on ZERO answers. Now: nothing is scored, the gate is
    parsed from OpenAI's message, and the unanswered requests go again in
    sequential parts under it — each resume advancing one step.
    """

    @staticmethod
    def _run(tmp_path, max_usd=1.0):
        return LabRun(dataset=DatasetSpec("demo"), arms=["null_closed_book", "bm25"],
                      budgets=[500], mode="batch", max_usd=max_usd,
                      run_dir=tmp_path / "run", reader_model="gpt-5-nano")

    @staticmethod
    def _limit_for(tmp_path, parts):
        """An enqueued-token limit whose 90% gate splits the run into ``parts`` or more."""
        requests = runner.read_jsonl(tmp_path / "run" / runner.REQUESTS)
        total = sum(batch.request_prompt_tokens(r) for r in requests)
        return requests, total // parts

    async def test_a_rejected_reader_batch_is_resubmitted_in_parts_not_scored(
        self, tmp_path, lab, monkeypatch
    ):
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=2)
        backend = runner.ModuleBatchBackend(svc.client)
        root = tmp_path / "run"
        manifest = await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        assert manifest["status"] == "batch_submitted"
        requests, limit = self._limit_for(tmp_path, 3)
        svc.enqueued_limit = limit  # the first batch holds every request: refused
        svc.created[0]["tokens"] = limit * 3
        svc.created[0]["reject"] = True

        manifest = await runner.resume(root, batch_backend=backend)
        gate = batch.load_state(root)["max_enqueued_tokens"]
        assert gate == int(0.9 * limit)
        assert manifest["status"] == "batch_submitted"  # re-submitted, not scored
        read = manifest["phases"]["read"]
        assert read["message"].startswith("batch rejected at validation: Enqueued token limit")
        assert f"re-submitting in parts under {gate:,} enqueued tokens" in read["message"]
        assert "only_unanswered" not in read and read["rejections"][0]["gate"] == gate
        assert read["pending"] == len(requests)
        st = batch.load_state(root)
        new_parts = [p for p in st["parts"] if p["submission"] == 2]
        assert len(new_parts) >= 3 and all(p["prompt_tokens"] <= gate for p in new_parts)
        assert len(svc.created) == 2  # one part in flight, the rest queued behind the gate
        assert not (root / runner.RESPONSES).exists() or not runner.read_jsonl(
            root / runner.RESPONSES)
        assert not (root / runner.LEADERBOARD).exists()
        assert manifest["phases"]["score"]["status"] == "pending"

        for _ in range(30):
            manifest = await runner.resume(root, batch_backend=backend)
            if manifest["status"] == "done":
                break
            assert not (root / runner.LEADERBOARD).exists()  # never scored mid-way
        assert manifest["status"] == "done" and manifest["phases"]["read"]["status"] == "done"
        assert "message" not in manifest
        responses = runner.read_jsonl(root / runner.RESPONSES)
        assert sorted(r["custom_id"] for r in responses) == sorted(
            r["custom_id"] for r in requests)  # every request answered exactly once
        assert all(not r.get("error") for r in responses)
        assert svc.rejected == [svc.created[0]["id"]]  # the gate kept every part in
        assert svc.max_enqueued_seen <= limit and len(svc.created) - 1 == len(new_parts)
        assert manifest["actual"]["reader"]["usd"] == pytest.approx(batch.status(root)["usd"])

    async def test_a_rejection_waits_for_parts_still_running_and_resends_only_what_never_ran(
        self, tmp_path, lab, monkeypatch
    ):
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        real_plan = batch.plan_parts
        monkeypatch.setattr(batch, "plan_parts",
                            lambda lines, **kw: real_plan(lines, **{**kw, "max_requests": 10}))
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=2)
        backend = runner.ModuleBatchBackend(svc.client)
        root = tmp_path / "run"
        await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        requests = runner.read_jsonl(root / runner.REQUESTS)
        first, second = svc.created
        failing = json.loads(svc.files[second["input_file_id"]].decode().split("\n")[0])
        svc.fail = {failing["custom_id"]}  # one per-request failure in the part that runs
        first.update(reject=True)
        svc.enqueued_limit = 10**9

        manifest = await runner.resume(root, batch_backend=backend)  # part 2 still running
        assert manifest["status"] == "batch_submitted"
        assert "once the parts still running finish" in manifest["message"]
        assert len(svc.created) == 2
        manifest = await runner.resume(root, batch_backend=backend)  # part 2 done → re-plan
        assert manifest["status"] == "batch_submitted" and len(svc.created) == 3
        resent = [json.loads(x)["custom_id"]
                  for x in svc.files[svc.created[2]["input_file_id"]].decode().strip().split("\n")]
        first_ids = [json.loads(x)["custom_id"]
                     for x in svc.files[first["input_file_id"]].decode().strip().split("\n")]
        assert resent == first_ids  # what never ran — not the per-request failure
        for _ in range(10):
            manifest = await runner.resume(root, batch_backend=backend)
            if manifest["status"] == "done":
                break
        assert manifest["phases"]["read"]["status"] == "done_with_errors"
        assert manifest["phases"]["read"]["failed"] == 1
        # The per-request failure goes again only when asked to.
        svc.fail = set()
        manifest = await runner.resume(root, batch_backend=backend, retry_failed=True)
        for _ in range(10):
            if manifest["status"] == "done":
                break
            manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["phases"]["read"]["status"] == "done"
        latest = runner._latest_responses(runner.RunPaths(root))
        assert len(latest) == len(requests) and all(not r.get("error") for r in latest.values())

    @pytest.mark.parametrize("wrapped", [False, True])
    async def test_a_refused_retry_batch_is_sent_again_on_a_plain_resume(
        self, tmp_path, lab, monkeypatch, wrapped
    ):
        # Counterexample (review, 2026-09-25): 3 per-request failures → --retry-failed
        # → that batch refused at validation → a plain resume re-planned only ids with
        # NO response line, so the 3 retried ids (old error lines) were silently
        # dropped: the run ended done_with_errors, the manifest still said they were
        # being re-submitted, and batch.status() stayed "rejected" for good.
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = runner.ModuleBatchBackend(svc.client)
        if wrapped:
            backend = SubmitPollCollectOnly(backend)
        root = tmp_path / "run"
        await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        requests = runner.read_jsonl(root / runner.REQUESTS)
        retried = [r["custom_id"] for r in requests[:3]]
        svc.fail = set(retried)
        manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["phases"]["read"]["status"] == "done_with_errors"
        assert manifest["phases"]["read"]["failed"] == 3

        svc.fail = set()
        manifest = await runner.resume(root, batch_backend=backend, retry_failed=True)
        assert manifest["status"] == "batch_submitted" and len(svc.created) == 2
        tokens = sum(batch.request_prompt_tokens(r) for r in requests[:3])
        svc.enqueued_limit = tokens  # the 3 retried requests together: refused
        svc.created[1].update(reject=True, tokens=tokens + 1)

        manifest = await runner.resume(root, batch_backend=backend)  # WITHOUT --retry-failed
        assert manifest["status"] == "batch_submitted"
        assert len(svc.created) == 3  # the retried requests go again, one part at a time
        resent = [json.loads(x)["custom_id"] for p in batch.load_state(root)["parts"]
                  if p["submission"] == 3
                  for x in (root / batch.PARTS_DIR / f"{p['name']}.input.jsonl").read_text()
                  .strip().split("\n")]
        assert sorted(resent) == sorted(retried)
        for _ in range(10):
            manifest = await runner.resume(root, batch_backend=backend)
            if manifest["status"] == "done":
                break
        read = manifest["phases"]["read"]
        assert manifest["status"] == "done" and read["status"] == "done"
        assert "message" not in manifest and "rejected" not in read
        assert "only_unanswered" not in read
        st = batch.status(root)
        assert st["status"] == "collected" and st["rejected"] == []
        latest = runner._latest_responses(runner.RunPaths(root))
        assert len(latest) == len(requests) and all(not r.get("error") for r in latest.values())

    async def test_a_read_that_ends_without_resending_clears_the_rejection_message(
        self, tmp_path, lab, monkeypatch
    ):
        # A rejected part whose requests all have an answer by the time of the re-plan:
        # the read phase ends, and nothing may still claim a re-submission.
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        svc = FakeBatchService(answer=answer_from_body, polls_to_finish=1)
        backend = runner.ModuleBatchBackend(svc.client)
        root = tmp_path / "run"
        await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        await runner.resume(root, batch_backend=backend)
        manifest = json.loads((root / runner.MANIFEST).read_text())
        manifest["phases"]["read"].update(status="pending", only_unanswered=True,
                                          rejected=[{"part": "part-0000"}])
        manifest.update(status="reading", message="batch rejected at validation: x; "
                        "re-submitting the unanswered requests")
        (root / runner.MANIFEST).write_text(json.dumps(manifest))
        manifest = await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        read = manifest["phases"]["read"]
        assert manifest["status"] == "done" and read["status"] == "done"
        assert "message" not in manifest and "rejected" not in read
        assert "only_unanswered" not in read and len(svc.created) == 1

    async def test_a_refused_file_parks_the_run_until_the_next_resume(
        self, tmp_path, lab, monkeypatch
    ):
        monkeypatch.delenv(batch.ENQUEUED_TOKENS_ENV, raising=False)
        svc = FakeBatchService(answer=answer_from_body)
        backend = runner.ModuleBatchBackend(svc.client)
        root = tmp_path / "run"
        await runner.run_lab(self._run(tmp_path), batch_backend=backend)
        svc.created[0].update(status="failed", errors=[
            {"code": "invalid_request", "message": "bad line", "line": 3}])
        manifest = await runner.resume(root, batch_backend=backend)
        assert manifest["status"] == "reading" and len(svc.created) == 1
        assert manifest["phases"]["read"]["status"] == "pending"
        assert "Not re-submitted automatically" in manifest["message"]
        assert not (root / runner.LEADERBOARD).exists()
        manifest = await runner.resume(root, batch_backend=backend)  # the owner re-runs
        assert manifest["status"] == "batch_submitted" and len(svc.created) == 2
