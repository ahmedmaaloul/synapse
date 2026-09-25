# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The shared packer: budget never exceeded, skip-and-continue, layout, orders.

A deterministic word-count tokenizer stands in for tiktoken so every assertion
is exact and hermetic.
"""

from __future__ import annotations

import pytest

from app.lab import packer
from app.lab.evidence import Evidence, EvidenceUnit
from app.lab.packer import SECTION_HEADERS, pack


def words(text: str) -> int:
    """1 token per whitespace-separated word, +1 per blank-line separator."""
    return len(text.split()) + text.count("\n\n")


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch):
    monkeypatch.setattr(packer, "count_tokens", lambda text, model: (words(text), False))


def U(text, kind="prose", source=None, score=0.0):  # noqa: N802 - terse test factory
    return EvidenceUnit(text=text, kind=kind, source_id=source, score=score)


def evidence():
    return Evidence(
        [
            U("alpha beta gamma", "prose", "d1", 0.9),
            U("Entity: Ada (Type: Person)", "entity", "Ada", 1.0),
            U(" ".join(["w"] * 50), "prose", "d2", 0.8),  # big: skipped at small budgets
            U("Ada -[KNEW]-> Babbage", "path", None, 0.7),
            U("small tail", "prose", "d3", 0.1),
        ],
        "test",
    )


class TestBudget:
    @pytest.mark.parametrize("budget", range(0, 80))
    def test_never_exceeds_the_budget(self, budget):
        packed = pack(evidence(), budget, "gpt-5-nano")
        assert packed.tokens <= budget
        assert words(packed.text) == packed.tokens

    def test_a_unit_that_does_not_fit_is_skipped_and_a_smaller_one_still_fits(self):
        packed = pack(evidence(), 20, "gpt-5-nano")
        kept = {u.source_id for u, _ in packed.unit_texts() if u.kind == "prose"}
        assert "d2" not in kept  # 50 words never fits in 20
        assert "d3" in kept  # the tail after it still does
        assert packed.truncated and packed.skipped >= 1

    def test_none_is_uncapped_and_keeps_everything(self):
        packed = pack(evidence(), None, "gpt-5-nano")
        assert packed.units_used == 5
        assert packed.truncated is False and packed.budget is None
        assert packed.by_kind == {"entity": 1, "prose": 3, "path": 1}

    def test_zero_budget_is_empty(self):
        packed = pack(evidence(), 0, "gpt-5-nano")
        assert (packed.text, packed.tokens, packed.units_used) == ("", 0, 0)

    def test_exact_check_repairs_a_non_additive_tokenizer(self, monkeypatch):
        """Pieces fit on their own count, the joined text does not: drop until it fits."""

        def greedy_joiner(text, model):
            # Every blank-line separator costs 5 tokens once rendered, 1 on its own.
            return (len(text.split()) + (5 * text.count("\n\n") if text != "\n\n" else 1), False)

        monkeypatch.setattr(packer, "count_tokens", greedy_joiner)
        for budget in range(0, 60):
            packed = pack(evidence(), budget, "gpt-5-nano")
            assert packed.tokens <= budget


class TestLayout:
    def test_sections_have_headers_and_blank_line_separators(self):
        packed = pack(evidence(), None, "gpt-5-nano")
        assert packed.text.startswith(SECTION_HEADERS["entity"] + "\n")
        assert "\n\n" + SECTION_HEADERS["prose"] + "\n" in packed.text
        assert "\n\n" + SECTION_HEADERS["path"] + "\n" in packed.text
        assert "alpha beta gamma\n\n" in packed.text

    def test_spans_point_at_the_verbatim_unit_text(self):
        packed = pack(evidence(), None, "gpt-5-nano")
        for unit, text in packed.unit_texts():
            assert text == unit.text

    def test_score_order_best_first(self):
        packed = pack(evidence(), None, "gpt-5-nano")
        placed = [p.unit.source_id or p.unit.kind for p in packed.placed]
        # entity section first (best unit 1.0), then prose (0.9 best), then path
        assert placed == ["Ada", "d1", "d2", "d3", "path"]

    def test_ascending_is_the_exact_mirror_with_the_best_unit_last(self):
        best_first = pack(evidence(), None, "gpt-5-nano")
        ascending = pack(evidence(), None, "gpt-5-nano", order="ascending")
        assert [p.unit for p in ascending.placed] == [p.unit for p in best_first.placed][::-1]
        assert ascending.placed[-1].unit.score == 1.0
        # Same selection, different placement.
        assert sorted(ascending.text.split()) == sorted(best_first.text.split())

    def test_order_does_not_change_what_is_selected(self):
        ev = evidence()
        for budget in (10, 20, 30):
            a = pack(ev, budget, "m")
            b = pack(ev, budget, "m", order="ascending")
            assert a.placed and {p.unit for p in a.placed} == {p.unit for p in b.placed}

    def test_unknown_order_is_refused(self):
        with pytest.raises(ValueError):
            pack(evidence(), 10, "m", order="random")

    def test_ties_keep_the_arms_order(self):
        units = [U("first", score=1.0), U("second", score=1.0), U("third", score=1.0)]
        packed = pack(units, None, "m")
        assert [p.unit.text for p in packed.placed] == ["first", "second", "third"]


class TestUnits:
    def test_duplicates_are_packed_once(self):
        units = [U("same text", score=1.0), U("same text", score=0.5), U("other", score=0.1)]
        packed = pack(units, None, "m")
        assert packed.units_used == 2

    def test_empty_units_are_ignored(self):
        packed = pack([U("   "), U("", "entity"), U("real")], None, "m")
        assert packed.units_used == 1

    def test_name_list_is_cut_to_an_alphabetical_prefix_not_dropped(self):
        names = "\n".join(f"Name{i:03d}" for i in range(100))
        packed = pack([U(names, "name_list", score=1.0)], 11, "m")
        # header "Names:" = 1 word; 10 names fit.
        assert packed.tokens <= 11
        kept = packed.text.split("\n")[1:]
        assert kept == [f"Name{i:03d}" for i in range(10)]
        assert packed.truncated is True
        assert packed.by_kind == {"name_list": 1}

    def test_a_non_name_list_unit_is_never_cut(self):
        long_prose = " ".join(["w"] * 100)
        packed = pack([U(long_prose)], 50, "m")
        assert packed.text == "" and packed.skipped == 1

    def test_a_shared_token_cache_counts_each_unit_once(self, monkeypatch):
        seen: list[str] = []

        def counting(text, model):
            seen.append(text)
            return (words(text), False)

        monkeypatch.setattr(packer, "count_tokens", counting)
        cache: dict = {}
        ev = evidence()
        for budget in (10, 20, 30, None):
            pack(ev, budget, "m", token_cache=cache)
        assert seen.count("alpha beta gamma") == 1

    def test_estimated_flag_travels(self, monkeypatch):
        monkeypatch.setattr(packer, "count_tokens", lambda text, model: (words(text), True))
        assert pack(evidence(), 30, "m").estimated is True

    def test_to_dict_is_json_ready(self):
        import json

        data = pack(evidence(), 25, "m").to_dict()
        json.dumps(data)
        assert set(data) >= {"text", "tokens", "units_used", "truncated", "by_kind", "units"}
