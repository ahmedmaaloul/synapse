# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The Lab reader: one prompt, the request body per model family, the realtime path.

No model is ever called: every test hands ``answer_realtime`` a fake client,
and one test proves the real-client factory refuses to run under pytest.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.lab import reader
from app.lab.reader import SpendGuard, answer_realtime, build_request


class TestRequests:
    def test_reasoning_model_gets_effort_and_a_completion_cap_not_temperature(self):
        body = build_request("Who?", "Facts:\nAda", "gpt-5-nano")
        assert "temperature" not in body and "max_tokens" not in body and "seed" not in body
        assert body["reasoning_effort"] == get_settings().openai_reasoning_effort == "minimal"
        assert body["max_completion_tokens"] == reader.MAX_ANSWER_TOKENS + 512

    def test_o_series_gets_low_instead_of_minimal(self):
        body = build_request("Who?", "", "o3-mini")
        assert body["reasoning_effort"] == "low"

    def test_reasoning_allowance_moves_the_cap(self):
        body = build_request("Who?", "", "gpt-5-nano", reasoning_allowance=64)
        assert body["max_completion_tokens"] == 96

    def test_classic_model_is_deterministic_and_short(self):
        body = build_request("Who?", "ctx", "gpt-4o-mini", seed=11)
        assert body["temperature"] == 0 and body["max_tokens"] == 32 and body["seed"] == 11
        assert "reasoning_effort" not in body and "max_completion_tokens" not in body

    def test_one_prompt_for_every_arm(self):
        a = build_request("Q?", "Excerpts:\nsome text", "gpt-5-nano")
        b = build_request("Q?", "Names:\nAda\nBob", "gpt-5-nano")
        assert a["messages"][0] == b["messages"][0]
        assert a["messages"][0]["content"] == reader.SYSTEM_PROMPT
        assert "own knowledge" in reader.SYSTEM_PROMPT  # what makes N0 closed-book

    def test_empty_evidence_is_said_explicitly(self):
        body = build_request("When?", "   ", "gpt-5-nano")
        assert body["messages"][1]["content"] == "Evidence:\n(none)\n\nQuestion: When?\nAnswer:"

    def test_identical_bodies_hash_identically(self):
        a = build_request("Q", "", "gpt-5-nano")
        assert reader.request_hash(a) == reader.request_hash(build_request("Q", "", "gpt-5-nano"))
        assert reader.request_hash(a) != reader.request_hash(build_request("Q", "x", "gpt-5-nano"))

    def test_prompt_tokens_include_framing(self, monkeypatch):
        monkeypatch.setattr(reader, "count_tokens", lambda text, model: (len(text.split()), False))
        body = {"model": "m", "messages": [{"role": "system", "content": "a b"},
                                            {"role": "user", "content": "c"}]}
        assert reader.request_prompt_tokens(body) == (3 + 2 * 3 + 3, False)
        assert reader.request_output_cap(build_request("q", "", "gpt-5-nano")) == 544


def completion(answer="Ada", prompt=100, completion_tokens=40, reasoning=30, cached=0,
               model="gpt-5-nano-2025-08-07"):
    return SimpleNamespace(
        model=model,
        system_fingerprint="fp_x",
        choices=[SimpleNamespace(message=SimpleNamespace(content=f" {answer} "),
                                 finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


class FakeClient:
    """``client.chat.completions.create(**body)`` — records bodies, tracks concurrency."""

    def __init__(self, answers=None, fail_on=None, delay=0.0):
        self.bodies: list[dict] = []
        self.answers = answers or {}
        self.fail_on = fail_on or set()
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **body):
        self.bodies.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            question = body["messages"][1]["content"].split("Question: ")[1].split("\n")[0]
            if question in self.fail_on:
                raise RuntimeError("upstream 500")
            return completion(self.answers.get(question, "unknown"))
        finally:
            self.in_flight -= 1


class TestParse:
    def test_sdk_object(self):
        r = reader.parse_completion(completion(cached=12))
        assert (r.answer, r.prompt_tokens, r.completion_tokens, r.reasoning_tokens,
                r.cached_tokens) == ("Ada", 100, 40, 30, 12)
        assert r.system_fingerprint == "fp_x" and r.finish_reason == "stop"

    def test_plain_json_as_in_a_batch_output_file(self):
        body = {
            "model": "gpt-5-nano",
            "choices": [{"message": {"content": "1843"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                      "completion_tokens_details": {"reasoning_tokens": 1}},
        }
        r = reader.parse_completion(body)
        assert (r.answer, r.prompt_tokens, r.reasoning_tokens, r.cached_tokens) == (
            "1843", 7, 1, 0)

    def test_round_trip(self):
        r = reader.parse_completion(completion())
        assert reader.ReaderResult.from_dict(r.to_dict()) == r


class TestRealtime:
    async def test_answers_in_input_order_with_usage(self):
        client = FakeClient(answers={"q1": "one", "q2": "two", "q3": "three"}, delay=0.001)
        bodies = [build_request(q, "", "gpt-5-nano") for q in ("q1", "q2", "q3")]
        seen = []
        results = await answer_realtime(
            bodies, client=client, max_concurrency=2,
            on_result=lambda i, r: seen.append(i),
        )
        assert [r.answer for r in results] == ["one", "two", "three"]
        assert all(r.reasoning_tokens == 30 for r in results)
        assert sorted(seen) == [0, 1, 2]
        assert client.max_in_flight <= 2

    async def test_a_failed_call_is_recorded_not_raised(self):
        client = FakeClient(fail_on={"bad"})
        bodies = [build_request(q, "", "gpt-4o-mini") for q in ("good", "bad")]
        good, bad = await answer_realtime(bodies, client=client)
        assert good.ok and bad.error and "upstream 500" in bad.error and not bad.skipped

    async def test_spend_guard_stops_before_the_cap(self):
        client = FakeClient()
        bodies = [build_request(f"q{i}", "", "gpt-5-nano") for i in range(10)]
        guard = SpendGuard(
            0.35,
            upper_usd=lambda body: 0.1,  # worst case per call
            actual_usd=lambda result: 0.05,  # what each call really cost
        )
        results = await answer_realtime(bodies, client=client, max_concurrency=1, guard=guard)
        sent = [r for r in results if not r.skipped]
        # 0.05 spent per call, 0.1 reserved for the next: stops once spent + 0.1 > 0.35.
        assert len(sent) == 6 and len(client.bodies) == 6
        assert guard.spent == pytest.approx(0.30) and guard.spent <= 0.35
        assert all(r.skipped and r.error == "spend cap reached" for r in results[6:])

    async def test_spend_guard_counts_in_flight_reservations(self):
        client = FakeClient(delay=0.01)
        bodies = [build_request(f"q{i}", "", "gpt-5-nano") for i in range(8)]
        guard = SpendGuard(0.25, upper_usd=lambda b: 0.1, actual_usd=lambda r: 0.1)
        results = await answer_realtime(bodies, client=client, max_concurrency=4, guard=guard)
        assert sum(1 for r in results if not r.skipped) == 2
        assert guard.spent <= 0.25

    async def test_an_unpriced_model_cannot_be_capped(self):
        guard = SpendGuard(1.0, upper_usd=lambda b: None, actual_usd=lambda r: None)
        client = FakeClient()
        (result,) = await answer_realtime([build_request("q", "", "mystery")], client=client,
                                          guard=guard)
        assert result.skipped and client.bodies == []

    async def test_no_requests_no_client(self):
        assert await answer_realtime([]) == []


def test_the_real_client_is_refused_under_pytest():
    with pytest.raises(RuntimeError, match="pytest"):
        reader.default_client()


def test_prompt_version_is_pinned_to_the_prompt_text():
    assert len(reader.PROMPT_VERSION) == 12
