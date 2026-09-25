# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The Lab dry-run estimator and the Batch price multiplier. Free by construction.

A deterministic word-count tokenizer replaces tiktoken so every token figure is
exact; prices are the real (dated) table in ``benchmarks/public/cost.py``.
"""

from __future__ import annotations

import json

import pytest

from app.lab import estimate as est
from app.lab import reader
from app.lab.estimate import IngestPlan, estimate_ingest, estimate_run
from benchmarks.public import cost

QUESTIONS = ["Who designed the Analytical Engine?", "When was Ada born?"]
ARMS_ = ["null_closed_book", "bm25", "synapse_lean"]
BUDGETS = [500, 2000]


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch):
    monkeypatch.setattr(reader, "count_tokens", lambda text, model: (len(text.split()), False))


def overhead(question: str, model: str) -> int:
    body = reader.build_request(question, "", model)
    return reader.request_prompt_tokens(body, model)[0]


class TestBatchMultiplier:
    def test_batch_halves_both_sides_and_keeps_the_date(self):
        price = cost.resolve_price("gpt-5-nano")
        half = cost.batch_price(price)
        assert half.input_usd_per_1m == pytest.approx(price.input_usd_per_1m * 0.5)
        assert half.output_usd_per_1m == pytest.approx(price.output_usd_per_1m * 0.5)
        assert half.checked_on == price.checked_on
        usage = cost.Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost.usd(usage, price, batch=True) == pytest.approx(cost.usd(usage, price) / 2)

    def test_unpriced_stays_unpriced(self):
        assert cost.batch_price(None) is None
        assert cost.usd(cost.Usage(prompt_tokens=5), None, batch=True) is None


class TestRetrieveMode:
    def test_is_always_zero(self):
        e = estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS,
                         reader_model="gpt-5-nano", mode="retrieve", max_usd=0.0,
                         ingest=IngestPlan(paragraphs=10_000))
        assert (e.total_point_usd, e.total_upper_usd, e.refuse) == (0.0, 0.0, False)
        assert all(c.calls == 0 for c in e.cells)
        assert [p.phase for p in e.phases] == ["reader"]


class TestReader:
    def test_point_and_upper_arithmetic(self):
        model = "gpt-4o-mini"
        e = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[500], reader_model=model)
        (cell,) = e.cells
        base = sum(overhead(q, model) for q in QUESTIONS)
        assert cell.calls == 2
        assert cell.prompt_tokens == base + 2 * 500
        assert cell.upper_prompt_tokens == pytest.approx((base + 1000) * 1.1, abs=1)
        assert cell.completion_tokens == 2 * est.POINT_ANSWER_TOKENS
        assert cell.upper_completion_tokens == 2 * 32
        price = cost.resolve_price(model)
        expected = (cell.upper_prompt_tokens * price.input_usd_per_1m
                    + cell.upper_completion_tokens * price.output_usd_per_1m) / 1e6
        assert cell.upper_usd == pytest.approx(expected)
        assert e.total_upper_usd >= e.total_point_usd > 0

    def test_closed_book_is_charged_once_across_budgets(self):
        e = estimate_run(questions=QUESTIONS, arms=["null_closed_book"], budgets=BUDGETS,
                         reader_model="gpt-5-nano")
        calls = [c.calls for c in e.cells]
        assert calls == [2, 0]
        assert e.cells[0].context_tokens == 0

    def test_batch_is_half_of_realtime(self):
        kw = {"questions": QUESTIONS, "arms": ARMS_, "budgets": BUDGETS,
              "reader_model": "gpt-5-nano"}
        realtime = estimate_run(mode="realtime", **kw)
        batch = estimate_run(mode="batch", **kw)
        assert batch.batch is True
        assert batch.total_point_usd == pytest.approx(realtime.total_point_usd / 2)
        assert batch.total_upper_usd == pytest.approx(realtime.total_upper_usd / 2)

    def test_reasoning_upper_bound_uses_the_request_cap(self):
        e = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[500],
                         reader_model="gpt-5-nano")
        assert e.max_output_tokens == 32 + reader.REASONING_ALLOWANCE
        assert e.cells[0].upper_completion_tokens == 2 * (32 + 512)
        assert any("REASONING" in a for a in e.assumptions)
        lower = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[500],
                             reader_model="gpt-5-nano", reasoning_allowance=64)
        assert lower.cells[0].upper_completion_tokens == 2 * 96
        assert lower.total_upper_usd < e.total_upper_usd

    def test_measured_contexts_replace_the_budget(self):
        e = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[2000],
                         reader_model="gpt-4o-mini",
                         context_tokens={("bm25", 2000): [100, 300]})
        (cell,) = e.cells
        base = sum(overhead(q, "gpt-4o-mini") for q in QUESTIONS)
        assert cell.measured and cell.context_tokens == 200
        assert cell.prompt_tokens == base + 400

    def test_uncapped_cells_are_assumed_and_said_so(self):
        e = estimate_run(questions=QUESTIONS, arms=["synapse_lean"], budgets=[None],
                         reader_model="gpt-4o-mini")
        (cell,) = e.cells
        assert cell.context_tokens == est.DEFAULT_CONTEXT_TOKENS and not cell.measured
        assert any("not bounded before retrieval" in a for a in e.assumptions)

    def test_explicit_cells_replace_the_grid(self):
        e = estimate_run(questions=QUESTIONS, arms=["bm25", "synapse_lean"], budgets=[500],
                         cells=[("bm25", 500), ("synapse_lean", 500), ("synapse_lean", None)],
                         reader_model="gpt-4o-mini")
        assert [(c.arm, c.budget) for c in e.cells] == [
            ("bm25", 500), ("synapse_lean", 500), ("synapse_lean", None)]

    def test_unique_requests_override_calls(self):
        e = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[500, 2000],
                         reader_model="gpt-4o-mini",
                         unique_requests={("bm25", 500): 2, ("bm25", 2000): 0})
        assert [c.calls for c in e.cells] == [2, 0]


class TestRefusal:
    def test_refuses_when_the_upper_bound_breaks_the_cap(self):
        e = estimate_run(questions=QUESTIONS * 50, arms=ARMS_, budgets=BUDGETS,
                         reader_model="gpt-5-nano", max_usd=0.0001)
        assert e.refuse and "exceeds the cap" in e.refuse_reason

    def test_fits(self):
        e = estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS,
                         reader_model="gpt-5-nano", max_usd=5.0)
        assert not e.refuse and e.refuse_reason is None

    def test_an_unpriced_model_is_refused_under_a_cap(self):
        e = estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS,
                         reader_model="mystery-model", max_usd=100.0)
        assert e.total_upper_usd is None and e.refuse and "no price" in e.refuse_reason

    def test_no_cap_no_refusal(self):
        e = estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS,
                         reader_model="mystery-model")
        assert e.refuse is False


class TestIngest:
    def test_measured_defaults(self):
        phase = estimate_ingest(IngestPlan(paragraphs=500, batch=False))
        assert phase.calls == 500
        assert (phase.prompt_tokens, phase.completion_tokens) == (217_000, 232_000)
        # 500 paragraphs cost ≈ $0.1717 measured realtime; the estimate lands on it.
        assert phase.point_usd == pytest.approx(0.1717, abs=0.001)
        assert phase.upper_usd > phase.point_usd

    def test_batch_ingest_is_half(self):
        realtime = estimate_ingest(IngestPlan(paragraphs=1000, batch=False))
        batch = estimate_ingest(IngestPlan(paragraphs=1000, batch=True))
        assert batch.point_usd == pytest.approx(realtime.point_usd / 2)

    def test_community_summaries_are_off_by_default_and_countable(self):
        off = estimate_ingest(IngestPlan(paragraphs=10))
        on = estimate_ingest(IngestPlan(paragraphs=10, community_summaries=4))
        assert on.calls == off.calls + 4 and on.point_usd > off.point_usd

    def test_calibration_file_wins_for_the_same_model(self, tmp_path):
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps({"ingest": {
            "model": "gpt-4o-mini", "prompt_tokens_per_paragraph": 100,
            "completion_tokens_per_paragraph": 50, "measured_on": "2026-09-30"}}))
        calibration = est.load_calibration(path)
        e = estimate_run(questions=QUESTIONS, arms=["bm25"], budgets=[500],
                         reader_model="gpt-5-nano", ingest=IngestPlan(paragraphs=10),
                         calibration=calibration)
        ingest = e.phase("ingest")
        assert (ingest.prompt_tokens, ingest.completion_tokens) == (1000, 500)
        assert "calibration" in ingest.notes[0]

    def test_calibration_for_another_model_is_ignored(self):
        plan = IngestPlan(paragraphs=10, model="gpt-4.1-mini")
        out = est.calibrated_ingest(plan, {"ingest": {"model": "gpt-4o-mini",
                                                      "prompt_tokens_per_paragraph": 1,
                                                      "completion_tokens_per_paragraph": 1}})
        assert out is plan

    def test_missing_calibration_file_is_empty(self, tmp_path):
        assert est.load_calibration(tmp_path / "nope.json") == {}


def test_the_estimate_serialises():
    e = estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS, reader_model="gpt-5-nano",
                     mode="batch", max_usd=1.0, ingest=IngestPlan(paragraphs=20))
    data = json.loads(json.dumps(e.to_dict()))
    assert data["batch_multiplier"] == 0.5 and data["pricing_url"] == cost.PRICING_URL
    assert {p["phase"] for p in data["phases"]} == {"ingest", "retrieval_llm", "reader"}
    assert len(data["cells"]) == len(ARMS_) * len(BUDGETS)
    assert "reader:gpt-5-nano" in data["price_dates"]


def test_unknown_mode():
    with pytest.raises(ValueError):
        estimate_run(questions=QUESTIONS, arms=ARMS_, budgets=BUDGETS,
                     reader_model="gpt-5-nano", mode="free")
