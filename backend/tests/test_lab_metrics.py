# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Lab metrics: EM/F1, containment, the bootstrap rule, cost-of-pass, Pareto, leaderboard."""

from __future__ import annotations

import pytest

from app.lab import metrics
from app.lab.metrics import (
    amortized_cost_of_pass,
    answer_scores,
    contains_answer,
    effect_floor,
    leaderboard,
    paired_bootstrap,
    pareto_frontier,
)


class TestPerQuestion:
    def test_answer_scores_use_the_official_normalisation(self):
        assert answer_scores("The Eiffel Tower.", "eiffel tower") == (1.0, 1.0)
        em, f1 = answer_scores("Paris France", ["Paris"])
        assert em == 0.0 and f1 == pytest.approx(2 / 3)

    def test_containment_is_word_bounded_and_normalised(self):
        assert contains_answer("Built in 1889 by Eiffel.", ["1889"])
        assert not contains_answer("Built in 18890.", "1889")
        golds = ["Babbage, Charles", "Charles Babbage"]
        assert contains_answer("the Charles Babbage engine", golds)

    def test_yes_no_is_never_contained(self):
        assert not contains_answer("yes it was, no doubt", "yes")

    def test_effect_floor_is_one_question_in_points(self):
        assert effect_floor(100) == 1.0 and effect_floor(700) == pytest.approx(100 / 700)
        assert effect_floor(1) == 100.0 and effect_floor(0) == 0.0


class TestBootstrap:
    def test_a_clear_difference_is_reportable(self):
        a = [1.0] * 80 + [0.0] * 20
        b = [0.0] * 80 + [1.0] * 20
        c = paired_bootstrap(a, b, iterations=2000, seed=1)
        assert c.diff == pytest.approx(60.0)
        assert c.ci_low > 0 and c.reportable
        assert metrics.verdict(c) == "above"

    def test_a_difference_below_the_floor_is_not_called(self):
        a = [1.0] + [0.0] * 199
        b = [0.0] * 200
        c = paired_bootstrap(a, b, iterations=2000, seed=1)
        assert c.diff == pytest.approx(0.5)  # one question at n=200 is exactly the floor
        assert c.reportable is False and metrics.verdict(c) == "too close to call"

    def test_a_single_question_is_never_reportable(self):
        for a, b in (([1.0], [0.0]), ([0.0], [1.0])):
            c = paired_bootstrap(a, b, iterations=200, seed=1)
            assert abs(c.diff) == pytest.approx(100.0) and c.floor == 100.0
            assert c.reportable is False and metrics.verdict(c) == "too close to call"

    def test_seeded_and_deterministic(self):
        a = [0.2, 0.9, 0.4, 1.0, 0.0, 0.7]
        b = [0.1, 0.5, 0.4, 0.3, 0.2, 0.6]
        assert paired_bootstrap(a, b, seed=5) == paired_bootstrap(a, b, seed=5)

    def test_identical_arms_have_zero_width_ci(self):
        c = paired_bootstrap([0.5, 1.0], [0.5, 1.0], iterations=100)
        assert (c.diff, c.ci_low, c.ci_high, c.reportable) == (0.0, 0.0, 0.0, False)

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0])

    def test_empty(self):
        c = paired_bootstrap([], [])
        assert c.n == 0 and metrics.verdict(c) == "no data"

    def test_the_pure_python_percentile_matches_numpy(self):
        numpy = pytest.importorskip("numpy")
        values = sorted([0.3, 1.2, 5.0, -2.0, 0.0, 7.5, 3.3])
        for q in (2.5, 50, 97.5):
            assert metrics._percentile(values, q) == pytest.approx(numpy.percentile(values, q))


class TestCost:
    def test_amortized_cost_of_pass(self):
        # (10 / 1000 + 0.002) / 0.5 = 0.024 $ per correct answer
        assert amortized_cost_of_pass(10.0, 0.002, 0.5, 1000) == pytest.approx(0.024)
        assert amortized_cost_of_pass(None, 0.002, 0.5, 100) == pytest.approx(0.004)
        assert amortized_cost_of_pass(1.0, 0.002, 0.0, 100) is None

    def test_pareto_frontier(self):
        points = [
            {"arm": "cheap", "budget": 500, "x": 1.0, "y": 30.0},
            {"arm": "dominated", "budget": 500, "x": 2.0, "y": 25.0},
            {"arm": "better", "budget": 2000, "x": 3.0, "y": 45.0},
            {"arm": "best", "budget": 4000, "x": 9.0, "y": 50.0},
            {"arm": "worse_and_dearer", "budget": 4000, "x": 10.0, "y": 49.0},
            {"arm": "unpriced", "budget": None, "x": None, "y": 99.0},
        ]
        frontier = pareto_frontier(points, x="x", y="y")
        assert [p["arm"] for p in frontier] == ["cheap", "better", "best"]


def rows(arm, budget, f1s, *, usd=0.001, tokens=(100, 5)):
    return [
        {
            "arm": arm, "budget": budget, "qid": f"q{i}", "em": 1.0 if f == 1.0 else 0.0,
            "f1": f, "prompt_tokens": tokens[0], "completion_tokens": tokens[1],
            "reasoning_tokens": 2, "cached_tokens": 0, "usd": usd, "context_tokens": 50,
            "units_used": 3, "by_kind": {"prose": 3}, "truncated": False, "containment": f > 0,
        }
        for i, f in enumerate(f1s)
    ]


META = {
    "null_closed_book": {"family": "null", "is_null": True, "needs_graph": False},
    "null_random": {"family": "null", "is_null": True, "needs_graph": False},
    "bm25": {"family": "passage", "needs_graph": False},
    "dense": {"family": "passage", "needs_graph": False},
    "synapse_lean": {"family": "graph", "needs_graph": True},
}


class TestLeaderboard:
    def cells(self):
        n = 50
        return {
            ("null_closed_book", 500): rows("null_closed_book", 500, [0.0] * n),
            ("null_random", 500): rows("null_random", 500, [0.0] * 45 + [1.0] * 5),
            ("bm25", 500): rows("bm25", 500, [1.0] * 25 + [0.0] * 25),
            ("dense", 500): rows("dense", 500, [1.0] * 30 + [0.0] * 20),
            ("synapse_lean", 500): rows("synapse_lean", 500, [1.0] * 40 + [0.0] * 10,
                                        usd=0.002),
        }

    def test_rows_and_floor_comparisons(self):
        board = leaderboard(self.cells(), arm_meta=META, ingest_usd=5.0, iterations=2000)
        by_arm = {r["arm"]: r for r in board["rows"]}
        lean = by_arm["synapse_lean"]
        assert lean["f1"] == pytest.approx(80.0) and lean["em"] == pytest.approx(80.0)
        assert lean["tokens_per_correct"] == pytest.approx(50 * 105 / 40)
        assert lean["usd_per_100_correct"] == pytest.approx(0.1 / 40 * 100)
        assert lean["comparisons"]["gain_above_n0"]["diff"] == pytest.approx(80.0)
        assert lean["comparisons"]["gain_above_n2"]["diff"] == pytest.approx(70.0)
        premium = lean["comparisons"]["graph_premium"]
        assert premium["against"] == "dense" and premium["diff"] == pytest.approx(20.0)
        assert premium["reportable"] is True
        # Ingest is charged to graph arms only.
        assert lean["amortized_cost_of_pass"]["1000"] == pytest.approx((5.0 / 1000 + 0.002) / 0.8)
        assert by_arm["bm25"]["amortized_cost_of_pass"]["1000"] == pytest.approx(0.001 / 0.5)
        assert "graph_premium" not in by_arm["bm25"]["comparisons"]
        assert board["effect_floor_points"] == pytest.approx(2.0)
        assert board["floors"]["n0_f1"] == 0.0

    def test_unknown_ingest_leaves_graph_amortization_honest(self):
        board = leaderboard(self.cells(), arm_meta=META, ingest_usd=None, iterations=200)
        by_arm = {r["arm"]: r for r in board["rows"]}
        # The graph arm's ingest bill is unknown: no amortized figure, never a free ingest.
        assert by_arm["synapse_lean"]["amortized_cost_of_pass"]["100"] is None
        # Passage arms need no LLM-built index: their ingest is $0 and known.
        assert by_arm["bm25"]["amortized_cost_of_pass"]["100"] == pytest.approx(0.001 / 0.5)

    def test_ranked_by_cost_of_pass_and_floors_flagged(self):
        board = leaderboard(self.cells(), arm_meta=META, iterations=200)
        ranked = [r["arm"] for r in board["rows"] if r["rank"] is not None]
        assert ranked[0] == "dense"  # $0.05 for 30 correct is the cheapest per correct
        n0 = next(r for r in board["rows"] if r["arm"] == "null_closed_book")
        assert n0["rank"] is None and n0["is_null"] is True  # 0 correct: unranked

    def test_frontiers(self):
        board = leaderboard(self.cells(), arm_meta=META, iterations=200)
        usd_frontier = [p["arm"] for p in board["frontiers"]["f1_vs_usd"]]
        assert usd_frontier[-1] == "synapse_lean" and "bm25" not in usd_frontier

    def test_read_errors_are_counted_not_scored(self):
        cells = self.cells()
        broken = cells[("bm25", 500)]
        broken[0]["read_error"] = "spend cap"
        board = leaderboard(cells, arm_meta=META, iterations=200)
        bm25 = next(r for r in board["rows"] if r["arm"] == "bm25")
        assert bm25["read_errors"] == 1 and bm25["answered"] == 49
        assert bm25["f1"] == pytest.approx(24 / 49 * 100)

    def test_retrieve_only_mode(self):
        cells = {
            ("bm25", 500): rows("bm25", 500, [1.0, 0.0]),
            ("null_random", 500): rows("null_random", 500, [0.0, 0.0]),
            ("bm25", None): rows("bm25", None, [1.0, 1.0]),
        }
        for r in cells[("bm25", 500)]:
            r["recall_permissive"] = 1.0
            r["recall_strict"] = 0.5
        board = leaderboard(cells, arm_meta=META, read=False)
        assert board["mode"] == "retrieve"
        first = board["rows"][0]
        assert (first["arm"], first["budget"]) == ("null_random", 500)  # floors first
        bm25 = board["rows"][1]
        assert "f1" not in bm25 and bm25["containment"] == 50.0
        assert bm25["recall_permissive"] == 100.0 and bm25["recall_strict"] == 50.0
        assert board["rows"][-1]["budget"] is None
        assert board["frontiers"] == {"f1_vs_usd": [], "f1_vs_tokens": []}
