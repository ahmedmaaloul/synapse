# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the SQuAD/HotpotQA answer metrics.

The expected values are worked by hand from the published definitions
(SQuAD ``normalize_answer`` and HotpotQA ``hotpot_evaluate_v1.f1_score``), so a
regression in normalization or in the yes/no rule shows up as a number that no
longer matches the literature's scorer.
"""

from __future__ import annotations

import pytest

from app.services.qa_metrics import METRICS, exact_match, f1, normalize_answer, score


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Eiffel Tower", "eiffel tower"),
            ("  An   apple, a day!  ", "apple day"),
            ("theatre", "theatre"),  # articles only as whole words
            ("Anne", "anne"),
            ("U.S.A.", "usa"),
            ("1,000", "1000"),  # punctuation is deleted, not split on
            ("rock-and-roll", "rockandroll"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_answer(raw) == expected


class TestExactMatch:
    def test_normalized_equality(self):
        assert exact_match("the Eiffel tower.", "Eiffel Tower") == 1.0
        assert exact_match("Eiffel", "Eiffel Tower") == 0.0

    def test_best_over_several_golds(self):
        assert exact_match("NYC", ["New York City", "NYC"]) == 1.0
        assert exact_match("Boston", ["New York City", "NYC"]) == 0.0

    def test_empty(self):
        assert exact_match("", "") == 1.0
        assert exact_match(None, "x") == 0.0
        assert exact_match("x", []) == 0.0


class TestF1:
    def test_perfect(self):
        assert f1("The Eiffel Tower", "eiffel tower") == 1.0

    def test_partial_overlap(self):
        # pred {eiffel, tower, paris} vs gold {eiffel, tower}: P = 2/3, R = 1 → F1 = 0.8
        assert f1("Eiffel Tower Paris", "Eiffel Tower") == pytest.approx(0.8)

    def test_repeated_tokens_count_once_per_occurrence(self):
        # pred [new, new, york] vs gold [new, york]: common = 2, P = 2/3, R = 1 → 0.8
        assert f1("new new york", "new york") == pytest.approx(0.8)

    def test_no_overlap(self):
        assert f1("London", "Paris") == 0.0

    @pytest.mark.parametrize(
        ("pred", "gold"),
        [("yes", "yes it was"), ("yes it was", "yes"), ("no", "yes"), ("noanswer", "no answer")],
    )
    def test_hotpotqa_yes_no_rule_gives_zero_not_partial_credit(self, pred, gold):
        assert f1(pred, gold) == 0.0

    def test_yes_no_exact(self):
        assert f1("Yes.", "yes") == 1.0

    def test_empty_prediction_scores_zero_like_the_reference(self):
        assert f1("", "Paris") == 0.0
        assert f1("", "") == 0.0
        assert f1(None, "Paris") == 0.0

    def test_best_over_several_golds(self):
        assert f1("Tower", ["Eiffel Tower", "Tower"]) == 1.0


class TestScore:
    def test_dispatch(self):
        assert score("Eiffel Tower Paris", "Eiffel Tower") == pytest.approx(0.8)
        assert score("Eiffel Tower Paris", "Eiffel Tower", metric="f1") == pytest.approx(0.8)
        assert score("Eiffel Tower Paris", "Eiffel Tower", metric="em") == 0.0

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="unknown metric 'bleu'"):
            score("a", "a", metric="bleu")

    def test_metric_names(self):
        assert METRICS == ("f1", "em")

    @pytest.mark.parametrize("metric", METRICS)
    def test_range(self, metric):
        for pred, gold in [("a b c", "b c d"), ("x", "x"), ("", "y"), ("yes", "no")]:
            assert 0.0 <= score(pred, gold, metric) <= 1.0
