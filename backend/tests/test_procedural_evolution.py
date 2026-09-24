# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for offline self-evolution (Algorithm 1 of arXiv:2609.09153).

Hermetic: ``run`` is a fake agent whose answers depend on the graph it is
given, the refiner is a scripted fake LLM, and the procedural store is replaced
by a small in-memory recorder. What is under test is the loop itself: the
acceptance gate (ties accepted), rejection memory, no validation rollout after
a structural failure, tail truncation, the LLM-call budget, and scratch mode
leaving the stored graph alone until a candidate is accepted.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.services import procedural_evolution as evo
from app.services import procedural_store as store
from app.services.graph_agent import TOOL_NAMES
from app.services.procedural_graph import START, ProceduralGraph
from app.services.procedural_store import ProceduralGraphNotFound, load_prior

TRAIN = [{"question": f"train question {i}?", "answer": f"gold {i}"} for i in range(4)]
VAL = [{"question": f"val question {i}?", "answer": f"gold v{i}"} for i in range(3)]
GOLD = {item["question"]: item["answer"] for item in TRAIN + VAL}

GOOD_EDITS = {
    "add_edges": [
        {
            "source": "search_passages",
            "target": "find_path",
            "relation": "TRIGGERS",
            "condition": "When the passage names two entities",
            "guidance": "Link the two entities before comparing them.",
            "pitfalls": "Do not treat a missing path as a negative answer.",
        }
    ]
}
BAD_EDITS = {  # structurally valid, but the fake agent fails whenever it sees "Bad"
    "add_nodes": [{"id": "Bad", "type": "REASONING", "description": "A harmful detour."}],
    "add_edges": [{"source": "Bad", "target": "End", "guidance": "Give up."}],
}
BROKEN_EDITS = {"add_edges": [{"source": "Ghost", "target": "End", "guidance": "x"}]}


class FakeRun:
    """Answers with the gold unless the graph contains a node named ``Bad``."""

    def __init__(self, calls_per_run: int = 2) -> None:
        self.calls: list[dict] = []
        self.calls_per_run = calls_per_run

    async def __call__(self, question, **kwargs):
        graph = kwargs["graph"]
        self.calls.append({"question": question, "nodes": set(graph.nodes), **kwargs})
        answer = "wrong" if "Bad" in graph.nodes else GOLD[question]
        return {
            "answer": answer,
            "stopped": "answer",
            "parse_failures": 0,
            "steps": [
                {
                    "thought": "look",
                    "action": "search_entities",
                    "args": {"query": question},
                    "observation": "found it",
                },
                {
                    "thought": "done",
                    "action": "answer",
                    "args": {"text": answer},
                    "observation": "",
                },
            ],
            "usage": {
                "llm_calls": self.calls_per_run,
                "input_tokens": 100,
                "output_tokens": 10,
                "estimated": False,
            },
        }

    def questions(self, items):
        wanted = {item["question"] for item in items}
        return [c for c in self.calls if c["question"] in wanted]


class FakeRefiner:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
        )


class MemoryStore:
    """Records every store call the loop makes; holds one stored graph."""

    def __init__(self, graph: ProceduralGraph | None, version: int = 1) -> None:
        self.graph = graph
        self.version = version
        self.loads: list[str] = []
        self.saves: list[dict] = []
        self.rejections: list[dict] = []
        self.trajectories: list[dict] = []

    def install(self, monkeypatch) -> MemoryStore:
        async def load_graph_with_meta(name):
            self.loads.append(name)
            if self.graph is None:
                return None
            return self.graph, {"version": self.version, "score": None}

        async def save_graph(graph, *, score=None, note="", edits=None, previous=None):
            self.version += 1
            self.graph = graph
            self.saves.append(
                {
                    "graph": graph,
                    "score": score,
                    "note": note,
                    "edits": edits,
                    "previous": previous,
                    "version": self.version,
                }
            )
            return self.version

        async def record_rejection(
            name, *, round, reason, score=None, diagnostics=None, edits=None
        ):
            self.rejections.append(
                {
                    "name": name,
                    "round": round,
                    "reason": reason,
                    "score": score,
                    "diagnostics": diagnostics,
                    "edits": edits,
                }
            )

        async def record_trajectory(name, *, version, query, steps, score, source):
            self.trajectories.append(
                {
                    "name": name,
                    "version": version,
                    "query": query,
                    "score": score,
                    "source": source,
                    "steps": steps,
                }
            )

        for fn in (load_graph_with_meta, save_graph, record_rejection, record_trajectory):
            monkeypatch.setattr(store, fn.__name__, fn)
        return self


@pytest.fixture
def stored(monkeypatch):
    prior = load_prior("graphrag-navigator")
    return MemoryStore(prior).install(monkeypatch)


def _evolve(run, refiner, **overrides):
    params = {
        "train": TRAIN,
        "val": VAL,
        "rounds": 1,
        "batch_size": 2,
        "mode": "static",
        "max_llm_calls": 1000,
        "max_steps": 2,
        "run": run,
        "refiner_llm": refiner,
    }
    params.update(overrides)
    return evo.evolve("graphrag-navigator", **params)


# ── The acceptance gate ──────────────────────────────
class TestAcceptance:
    async def test_a_tie_is_accepted_and_saved_as_a_new_version(self, stored):
        run, refiner = FakeRun(), FakeRefiner(GOOD_EDITS)
        report = await _evolve(run, refiner)

        assert report["baseline_score"] == 1.0
        (entry,) = report["rounds"]
        assert entry["accepted"] is True and entry["reason"] == "accepted"
        assert entry["candidate_score"] == 1.0  # equal to S0: ties are accepted
        assert entry["version"] == 2
        assert entry["diff"]["added_edges"][0]["source"] == "search_passages"
        (save,) = stored.saves
        assert save["score"] == 1.0
        assert save["note"] == "evolution round 1"
        assert save["edits"] == GOOD_EDITS
        assert save["previous"] is not None and "find_path" in save["previous"].nodes
        assert report["final_version"] == 2 and report["final_score"] == 1.0
        assert report["accepted_rounds"] == 1 and report["rounds_run"] == 1
        assert report["stopped"] == "completed"

    async def test_a_lower_score_is_rejected_and_remembered(self, stored):
        run, refiner = FakeRun(), FakeRefiner(BAD_EDITS)
        report = await _evolve(run, refiner)

        (entry,) = report["rounds"]
        assert entry["accepted"] is False and entry["reason"] == "score"
        assert entry["candidate_score"] == 0.0
        assert stored.saves == []
        assert stored.rejections == [
            {
                "name": "graphrag-navigator",
                "round": 1,
                "reason": "score",
                "score": 0.0,
                "diagnostics": [],
                "edits": BAD_EDITS,
            }
        ]
        assert report["final_score"] == 1.0 and report["final_version"] == 1

    async def test_after_acceptance_rollouts_use_the_new_graph_and_version(self, stored):
        run, refiner = FakeRun(), FakeRefiner(GOOD_EDITS, {"add_nodes": []})
        await _evolve(run, refiner, rounds=2)
        round_one_train, round_two_train = run.questions(TRAIN)[:2], run.questions(TRAIN)[2:]
        assert {c["graph_version"] for c in round_one_train} == {1}
        assert {c["graph_version"] for c in round_two_train} == {2}
        # Round 2 rolls out the accepted candidate, which has the new edge.
        assert all(
            any(e.source == "search_passages" and e.target == "find_path" for e in c["graph"].edges)
            for c in round_two_train
        )
        assert [s["version"] for s in stored.saves] == [2]

    async def test_an_unchanged_candidate_is_not_validated(self, stored):
        run, refiner = FakeRun(), FakeRefiner({"add_nodes": []}, GOOD_EDITS)
        report = await _evolve(run, refiner, rounds=2)
        first, second = report["rounds"]
        assert first["reason"] == "no_change" and first["accepted"] is False
        assert first["candidate_score"] is None
        # Baseline + round 2's validation only: round 1 spent no validation rollout.
        assert len(run.questions(VAL)) == 2 * len(VAL)
        assert stored.saves[0]["note"] == "evolution round 2"
        # Reported as rejected, so persisted like every other rejected round.
        assert stored.rejections == [
            {
                "name": "graphrag-navigator",
                "round": 1,
                "reason": "no_change",
                "score": None,
                "diagnostics": [],
                "edits": {"add_nodes": []},
            }
        ]
        assert "Round 1: the edits left the graph unchanged" in refiner.prompts[1]
        assert report["rounds_run"] == 2

    async def test_the_metric_is_applied(self, stored):
        class PartialRun(FakeRun):
            async def __call__(self, question, **kwargs):
                result = await super().__call__(question, **kwargs)
                result["answer"] = f"{GOLD[question]} extra"
                return result

        f1_report = await _evolve(PartialRun(), FakeRefiner(GOOD_EDITS), metric="f1")
        em_report = await _evolve(PartialRun(), FakeRefiner(GOOD_EDITS), metric="em")
        assert f1_report["baseline_score"] == pytest.approx(0.8)
        assert em_report["baseline_score"] == 0.0


# ── Structural failures and rejection memory ─────────
class TestRejectionMemory:
    async def test_a_structural_failure_skips_the_validation_rollout(self, stored):
        run, refiner = FakeRun(), FakeRefiner(BROKEN_EDITS)
        report = await _evolve(run, refiner)

        (entry,) = report["rounds"]
        assert entry["reason"] == "structural"
        assert any("Ghost" in d for d in entry["diagnostics"])
        # Validation questions were rolled out for the baseline only.
        assert len(run.questions(VAL)) == len(VAL)
        assert stored.saves == []
        (rejection,) = stored.rejections
        assert rejection["reason"] == "structural" and rejection["score"] is None
        assert rejection["edits"] == BROKEN_EDITS

    async def test_rejections_are_serialized_into_the_next_refiner_prompt(self, stored):
        refiner = FakeRefiner(BROKEN_EDITS, BAD_EDITS, GOOD_EDITS)
        report = await _evolve(FakeRun(), refiner, rounds=3)

        assert [r["reason"] for r in report["rounds"]] == ["structural", "score", "accepted"]
        first, second, third = refiner.prompts
        assert "Previously rejected candidates (do not propose them again):\n(none)" in first
        assert "Rejected in round 1: structural failure" in second
        assert "Ghost" in second
        assert "Rejected in round 2: validation score 0.000 < retained 1.000" in third
        assert '"Bad"' in third
        # The paper serializes the rejected candidate GRAPH, not only its edits.
        assert "Candidate graph:" in third
        assert "Bad (REASONING)" in third and "Bad -LEADS_TO-> End" in third
        assert "search_entities (ACTION)" in third  # the whole candidate, not the delta
        # A structural failure has no candidate graph, only its diagnostics.
        structural = third.split("Rejected in round 1: structural failure", 1)[1]
        assert "Candidate graph:" not in structural.split("Rejected in round 2", 1)[0]

    async def test_a_rejected_candidate_graph_is_shown_even_after_a_later_acceptance(
        self, stored
    ):
        # Round 2 accepts GOOD_EDITS; round 1's rejected edits now describe a
        # change to a graph that is no longer current, so the refiner needs the
        # rejected graph itself to know what was tried.
        refiner = FakeRefiner(BAD_EDITS, GOOD_EDITS, {"add_nodes": []})
        await _evolve(FakeRun(), refiner, rounds=3)
        fourth_view = refiner.prompts[2]
        assert "Rejected in round 1: validation score" in fourth_view
        assert "Bad -LEADS_TO-> End" in fourth_view

    async def test_unparseable_refiner_output_is_a_structural_rejection(self, stored):
        run = FakeRun()
        report = await _evolve(run, FakeRefiner("I would add a verification step."))
        (entry,) = report["rounds"]
        assert entry["reason"] == "structural"
        assert "not a JSON object" in entry["diagnostics"][0]
        assert len(run.questions(VAL)) == len(VAL)

    async def test_a_refiner_failure_is_reported_and_the_loop_continues(self, stored):
        refiner = FakeRefiner(TimeoutError("refiner timed out"), GOOD_EDITS)
        report = await _evolve(FakeRun(), refiner, rounds=2)
        assert [r["reason"] for r in report["rounds"]] == ["refiner_error", "accepted"]
        assert report["rounds"][0]["diagnostics"] == ["TimeoutError: refiner timed out"]
        # Not a candidate, but a round reported as rejected: it has a durable row.
        (row,) = stored.rejections
        assert (row["round"], row["reason"], row["edits"]) == (1, "refiner_error", None)
        assert row["diagnostics"] == ["TimeoutError: refiner timed out"]
        # The failed call was attempted, so it counts: 6 + 2×(4 + 1) + 6.
        assert report["llm_calls"] == 6 + 4 + 1 + 4 + 1 + 6

    async def test_a_spend_cap_on_the_refiner_stops_the_run_as_budget(self, stored):
        # The bench's GuardedModel raises (a subclass of) SpendCapReached BEFORE
        # the call. That is the budget running out, not a refiner failure.
        refiner = FakeRefiner(evo.SpendCapReached("metered spend reached the --max-usd cap"))
        report = await _evolve(FakeRun(), refiner, rounds=2)
        assert report["stopped"] == "budget"
        (entry,) = report["rounds"]
        assert entry["reason"] == "budget" and entry["accepted"] is False
        assert report["rounds_run"] == 0
        assert report["llm_calls"] == 6 + 4  # the refused refiner call is not counted
        assert stored.rejections == [] and stored.saves == []


# ── Scratch mode ─────────────────────────────────────
#: A first graph grown from the skeleton: Start → search_entities → answer → End.
SCRATCH_EDITS = {
    "delete_edges": [{"source": "Start", "target": "End"}],
    "add_nodes": [
        {"id": "search_entities", "type": "ACTION", "description": "Find the entity."},
        {"id": "answer", "type": "ACTION", "description": "Answer."},
    ],
    "add_edges": [
        {"source": "Start", "target": "search_entities", "guidance": "Search."},
        {"source": "search_entities", "target": "answer", "guidance": "Answer."},
        {"source": "answer", "target": "End", "guidance": "Stop."},
    ],
}


class GraphScoredRun(FakeRun):
    """Scores by graph: the stored prior (it has ``find_path``) answers everything,
    a scratch candidate only the first validation question, the bare skeleton nothing."""

    async def __call__(self, question, **kwargs):
        result = await super().__call__(question, **kwargs)
        nodes = kwargs["graph"].nodes
        if "find_path" in nodes:
            right = True
        elif "search_entities" in nodes:
            right = question == VAL[0]["question"]
        else:
            right = False
        result["answer"] = GOLD[question] if right else "wrong"
        return result


@pytest.fixture
def empty(monkeypatch):
    return MemoryStore(None).install(monkeypatch)


class TestScratch:
    async def test_a_new_name_is_untouched_until_acceptance(self, empty):
        run = FakeRun()
        report = await _evolve(run, FakeRefiner(BROKEN_EDITS), mode="scratch")

        assert empty.loads == ["graphrag-navigator"]  # the existence check, nothing else
        assert empty.saves == []
        baseline = run.questions(VAL)[: len(VAL)]
        assert all(call["nodes"] == {START, "End"} for call in baseline)
        assert baseline[0]["graph"].tools == TOOL_NAMES
        assert report["final_version"] is None and report["stored_score"] is None
        assert report["rounds"][0]["reason"] == "structural"

    async def test_an_existing_graph_is_refused_without_replace_before_any_call(self, stored):
        run, refiner = FakeRun(), FakeRefiner(SCRATCH_EDITS)
        with pytest.raises(evo.ProceduralGraphExists, match="replace=true") as caught:
            await _evolve(run, refiner, mode="scratch")
        assert caught.value.version == 1
        assert "'graphrag-navigator' already exists (v1)" in str(caught.value)
        assert run.calls == [] and refiner.prompts == []
        assert stored.saves == [] and stored.trajectories == [] and stored.rejections == []

    async def test_the_refiner_is_told_it_starts_from_scratch(self, empty):
        refiner = FakeRefiner(BROKEN_EDITS)
        await _evolve(FakeRun(), refiner, mode="scratch")
        prompt = refiner.prompts[0]
        assert "Refinement mode: scratch_incremental" in prompt
        for tool in TOOL_NAMES:
            assert f"- {tool}: " in prompt

    async def test_replace_scores_the_stored_graph_once_and_accepts_a_candidate_as_good(
        self, stored
    ):
        run, events = FakeRun(), []
        report = await _evolve(
            run, FakeRefiner(SCRATCH_EDITS), mode="scratch", replace=True, on_progress=events.append
        )
        # The stored graph (v1, the prior) ran ONCE over val, between S0 and round 1.
        stored_runs = [c for c in run.calls if c["graph_version"] == 1]
        assert [c["question"] for c in stored_runs] == [item["question"] for item in VAL]
        assert all("find_path" in c["nodes"] for c in stored_runs)
        assert report["stored_score"] == 1.0 and report["stored_version"] == 1
        stages = list(dict.fromkeys(e["stage"] for e in events))
        assert stages == ["baseline", "stored", "rollout", "refine", "validate", "accepted"]
        # 1.0 ties the stored graph's 1.0: accepted, and saved over it.
        (entry,) = report["rounds"]
        assert entry["accepted"] is True and entry["floor"] == 1.0
        (save,) = stored.saves
        # previous=None: the store diffs against whatever is stored under the name.
        assert save["previous"] is None
        assert save["graph"].name == "graphrag-navigator"
        assert set(save["graph"].nodes) == {START, "End", "search_entities", "answer"}
        assert report["final_version"] == save["version"] == 2
        # Baseline 6 + stored 6 + batch 4 + refiner 1 + validation 6.
        assert report["llm_calls"] == 23 == report["min_llm_calls"]

    async def test_replace_never_overwrites_a_better_stored_graph(self, stored):
        # S_skeleton 0 < S_cand 1/3 < S_stored 1: the paper's gate alone would accept.
        refiner = FakeRefiner(SCRATCH_EDITS)
        report = await _evolve(GraphScoredRun(), refiner, mode="scratch", replace=True, rounds=2)

        assert report["baseline_score"] == 0.0 and report["stored_score"] == 1.0
        first, second = report["rounds"]
        assert first["candidate_score"] == pytest.approx(1 / 3)
        assert first["floor"] == 1.0 and second["floor"] == 1.0
        assert [r["reason"] for r in report["rounds"]] == ["score", "score"]
        assert stored.saves == [] and report["final_version"] is None
        assert [r["reason"] for r in stored.rejections] == ["score", "score"]
        # The refiner learns which bar the candidate missed.
        assert "validation score 0.333 < the stored graph's 1.000" in refiner.prompts[1]
        # The stored graph was scored once, not once per round.
        assert report["llm_calls"] == 6 + 6 + 2 * (4 + 1 + 6)

    async def test_without_a_stored_graph_the_same_candidate_is_accepted(self, empty):
        report = await _evolve(
            GraphScoredRun(), FakeRefiner(SCRATCH_EDITS), mode="scratch", replace=True
        )
        # replace is a no-op on a new name: no stored score, the paper's floor.
        assert report["stored_score"] is None
        (entry,) = report["rounds"]
        assert entry["floor"] == 0.0 and entry["accepted"] is True
        assert len(empty.saves) == 1

    async def test_the_stored_graphs_evaluation_counts_in_the_minimum(self, stored):
        run = FakeRun()
        with pytest.raises(evo.EvolutionBudgetTooSmall) as caught:
            await _evolve(
                run, FakeRefiner(SCRATCH_EDITS), mode="scratch", replace=True, max_llm_calls=22
            )
        assert caught.value.minimum == 23
        assert "(3 baseline + 3 stored-graph + 2 train + 3 val rollouts)" in str(caught.value)
        assert run.calls == []

    async def test_an_initial_graph_can_seed_a_separate_name(self, stored, monkeypatch):
        run = FakeRun()
        report = await evo.evolve(
            "nav-evolved-bench",
            train=TRAIN,
            val=VAL,
            rounds=1,
            batch_size=2,
            max_llm_calls=1000,
            max_steps=2,
            run=run,
            refiner_llm=FakeRefiner(GOOD_EDITS),
            initial=load_prior("graphrag-navigator"),
        )
        assert stored.loads == []
        (save,) = stored.saves
        assert save["graph"].name == "nav-evolved-bench"
        assert save["previous"] is None
        assert {c["graph_name"] for c in run.calls} == {"nav-evolved-bench"}
        assert report["graph"] == "nav-evolved-bench"

    async def test_static_mode_needs_a_stored_graph(self, monkeypatch):
        MemoryStore(None).install(monkeypatch)
        with pytest.raises(ProceduralGraphNotFound):
            await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS))


# ── The budget ───────────────────────────────────────
class TestBudget:
    async def test_a_cap_that_fits_the_baseline_but_not_one_round_is_refused_before_any_call(
        self, stored
    ):
        run, refiner = FakeRun(), FakeRefiner(GOOD_EDITS)
        # max_steps=2, raw → 2 calls per rollout. The baseline's 3×2 = 6 fits in
        # 16, but a baseline alone decides nothing: round 1's worst case is
        # (2 train + 3 val)×2 + 1 refiner = 11, and 6 + 11 = 17 > 16.
        with pytest.raises(evo.EvolutionBudgetTooSmall) as caught:
            await _evolve(run, refiner, max_llm_calls=16)
        error = caught.value
        assert (error.max_llm_calls, error.minimum) == (16, 17)
        assert isinstance(error, ValueError)
        assert str(error) == (
            "max_llm_calls=16 cannot pay for the baseline and one full round, whose worst "
            "case is (3 baseline + 2 train + 3 val rollouts) × 2 calls + 1 refiner call = 17 "
            "calls; set max_llm_calls to at least 17 (or use fewer validation questions or a "
            "smaller batch)"
        )
        assert run.calls == [] and refiner.prompts == []  # zero calls made
        assert stored.saves == [] and stored.trajectories == [] and stored.rejections == []

    async def test_a_round_whose_worst_case_fits_runs_to_its_decision(self, stored):
        report = await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS), max_llm_calls=17)
        assert report["stopped"] == "completed"
        assert report["rounds"][0]["reason"] == "accepted"
        assert report["llm_calls"] == 17 == report["min_llm_calls"]

    async def test_a_later_round_that_cannot_finish_is_not_started(self, stored):
        run, refiner = FakeRun(), FakeRefiner(GOOD_EDITS)
        # Round 1 runs to its decision in 17 calls; round 2's worst case (11)
        # would end at 28 > 27, so it is not started.
        report = await _evolve(run, refiner, rounds=2, max_llm_calls=27)
        assert report["stopped"] == "budget"
        assert [r["reason"] for r in report["rounds"]] == ["accepted", "budget"]
        assert report["rounds_run"] == 1
        assert len(run.questions(TRAIN)) == 2  # round 1's batch only
        assert len(refiner.prompts) == 1
        assert report["llm_calls"] == 17

    async def test_budget_exhausted_during_validation_neither_accepts_nor_rejects(self, stored):
        # Defensive path: a run that reports MORE calls than its max_steps bound
        # (3 > 2) can still run the budget dry mid-validation. Baseline 9; the
        # round gate sees 9 + 11 = 20 and lets it start; 2 rollouts (6) + the
        # refiner (1) + one validation rollout (3) = 19, and the next one could
        # pass 20.
        report = await _evolve(FakeRun(calls_per_run=3), FakeRefiner(GOOD_EDITS), max_llm_calls=20)
        (entry,) = report["rounds"]
        assert entry["reason"] == "budget" and entry["accepted"] is False
        assert entry["candidate_score"] is None
        assert stored.saves == [] and stored.rejections == []
        assert report["llm_calls"] == 19

    async def test_no_budget_for_the_baseline(self, stored):
        run = FakeRun()
        # 3 val × 2 = 6 > 5: refused, and the error still names the one-round minimum.
        with pytest.raises(evo.EvolutionBudgetTooSmall, match="at least 17") as caught:
            await _evolve(run, FakeRefiner(GOOD_EDITS), max_llm_calls=5)
        assert caught.value.minimum == 17
        assert run.calls == []

    async def test_generative_guidance_doubles_the_worst_case(self, stored):
        # 2 steps × (solver + guidance) = 4 per rollout: (3 + 2 + 3) × 4 + 1.
        run = FakeRun()
        with pytest.raises(evo.EvolutionBudgetTooSmall) as caught:
            await _evolve(run, FakeRefiner(GOOD_EDITS), max_llm_calls=32, guidance="generative")
        assert caught.value.minimum == 33 and run.calls == []
        report = await _evolve(
            run, FakeRefiner(GOOD_EDITS), max_llm_calls=33, guidance="generative"
        )
        assert report["stopped"] == "completed" and report["min_llm_calls"] == 33

    def test_the_minimum_is_one_round_with_the_batch_capped_by_train(self):
        def minimum(**overrides):
            params = {"train_size": 4, "val_size": 3, "batch_size": 2, "per_rollout": 2}
            params.update(overrides)
            return evo.minimum_llm_calls(**params)

        assert minimum() == (3 + 2 + 3) * 2 + 1
        assert minimum(batch_size=99) == (3 + 4 + 3) * 2 + 1  # |B_1| ≤ |train|
        assert minimum(stored_eval=True) == (3 + 3 + 2 + 3) * 2 + 1
        assert minimum(per_rollout=evo.rollout_worst_case(8, "generative")) == 8 * 16 + 1
        assert evo.rollout_worst_case(8, "raw") == 8
        assert evo.check_budget(17, train_size=4, val_size=3, batch_size=2, per_rollout=2) == 17

    async def test_calls_are_counted_across_rollouts_and_the_refiner(self, stored):
        report = await _evolve(FakeRun(calls_per_run=1), FakeRefiner(GOOD_EDITS))
        # baseline 3 + rollout 2 + refiner 1 + validation 3
        assert report["llm_calls"] == 9
        assert report["usage"]["input_tokens"] == 8 * 100 + 1000
        assert report["usage"]["output_tokens"] == 8 * 10 + 200


# ── Recording, progress, the report ──────────────────
class TestReporting:
    async def test_every_rollout_records_a_trajectory(self, stored):
        await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS))
        sources = [t["source"] for t in stored.trajectories]
        assert sources == ["evolution-val"] * 3 + ["evolution"] * 2 + ["evolution-val"] * 3
        assert [t["version"] for t in stored.trajectories] == [1] * 5 + [None] * 3
        assert stored.trajectories[0]["score"] == 1.0
        assert stored.trajectories[0]["steps"][0]["action"] == "search_entities"

    async def test_a_failing_trajectory_write_does_not_stop_the_run(self, stored, monkeypatch):
        async def broken(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(store, "record_trajectory", broken)
        report = await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS))
        assert report["rounds"][0]["accepted"] is True

    async def test_progress_events(self, stored):
        events = []
        await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS), on_progress=events.append)
        assert all(e["type"] == "progress" for e in events)
        stages = [e["stage"] for e in events]
        assert stages[0] == "baseline" and stages[-1] == "accepted"
        order = list(dict.fromkeys(stages))
        assert order == ["baseline", "rollout", "refine", "validate", "accepted"]
        assert events[-1]["version"] == 2 and events[-1]["round"] == 1
        assert {e["round"] for e in events if e["stage"] == "baseline"} == {0}

    async def test_an_async_or_failing_listener_is_fine(self, stored):
        seen = []

        async def listener(event):
            seen.append(event)
            raise RuntimeError("listener bug")

        report = await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS), on_progress=listener)
        assert seen and report["rounds"][0]["accepted"] is True

    async def test_the_report_states_its_effect_floor(self, stored):
        report = await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS))
        assert report["effect_floor"] == pytest.approx(1 / len(VAL))
        assert "search trace" in report["note"]
        assert set(report) >= {
            "graph",
            "mode",
            "rounds_run",
            "stopped",
            "baseline_score",
            "final_score",
            "final_version",
            "rounds",
            "llm_calls",
            "effect_floor",
        }
        assert set(report["rounds"][0]) >= {
            "round",
            "train_mean",
            "candidate_score",
            "accepted",
            "reason",
            "diagnostics",
            "diff",
        }
        json.dumps(report)

    async def test_the_default_refiner_is_temperature_zero_json(self, stored, monkeypatch):
        seen = {}
        refiner = FakeRefiner(GOOD_EDITS)

        def fake_get_chat_llm(**kwargs):
            seen.update(kwargs)
            return refiner

        monkeypatch.setattr(evo, "get_chat_llm", fake_get_chat_llm)
        await _evolve(FakeRun(), None)
        assert seen == {"temperature": 0, "json_mode": True}
        assert len(refiner.prompts) == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train": []},
            {"val": []},
            {"mode": "online"},
            {"metric": "bleu"},
            {"guidance": "none"},
            {"rounds": 0},
            {"train": [{"question": "", "answer": "x"}]},
            {"val": [{"question": "q?"}]},
            {"val": [{"question": "q?", "answer": ""}]},
            {"train": [{"question": "q?", "answer": []}]},
            {"train": [{"question": "q?", "answer": ["", "  "]}]},
        ],
    )
    async def test_invalid_inputs(self, stored, overrides):
        with pytest.raises(ValueError):
            await _evolve(FakeRun(), FakeRefiner(GOOD_EDITS), **overrides)


# ── The pure helpers ─────────────────────────────────
class TestHelpers:
    def test_tail_truncation_keeps_the_end(self):
        assert evo.tail_truncate("abcdef", 10) == "abcdef"
        cut = evo.tail_truncate("abcdef", 3)
        assert cut.endswith("\ndef") and "abc" not in cut
        assert cut.startswith("…[earlier trajectory text truncated]")

    @staticmethod
    def _rollout(question, value, answer="a"):
        return {
            "question": question,
            "gold": "g",
            "score": value,
            "result": {
                "answer": answer,
                "stopped": "answer",
                "steps": [
                    {
                        "action": "search_entities",
                        "args": {"query": "q"},
                        "observation": "o" * 300,
                    }
                ],
            },
        }

    def test_the_trace_budget_is_split_and_unused_halves_flow_over(self):
        assert evo.split_trace_budget(5000, 5000, 1000) == (500, 500)
        assert evo.split_trace_budget(5000, 5000, 1001) == (500, 501)
        assert evo.split_trace_budget(200, 5000, 1000) == (200, 800)  # high needs less
        assert evo.split_trace_budget(5000, 300, 1000) == (700, 300)  # low needs less
        assert evo.split_trace_budget(0, 5000, 1000) == (0, 1000)  # no successes at all
        assert evo.split_trace_budget(100, 100, 1000) == (100, 900)  # both fit

    def test_oversized_partitions_are_each_cut_to_their_half_keeping_their_end(self):
        high = [self._rollout(f"HIGH-{i}", 1.0, answer=f"high-answer-{i}") for i in range(20)]
        low = [self._rollout(f"LOW-{i}", 0.0, answer=f"low-answer-{i}") for i in range(20)]
        full = evo.attempts_block(high + low, 0)  # 0: no limit
        assert full.index("## High-scoring") < full.index("## Low-scoring")
        assert "HIGH-0" in full and "LOW-0" in full

        cut = evo.attempts_block(high + low, 2000)
        high_part, low_part = cut.split("## Low-scoring trajectories\n\n")
        high_part = high_part.removeprefix("## High-scoring trajectories\n\n").rstrip("\n")
        # Successes survive a flood of failures; each partition keeps its END.
        assert high_part.startswith(evo.TRUNCATION_MARKER)
        assert low_part.startswith(evo.TRUNCATION_MARKER)
        assert "high-answer-19" in high_part and "HIGH-0\n" not in high_part
        assert "low-answer-19" in low_part and "LOW-0\n" not in low_part
        assert len(high_part) == len(evo.TRUNCATION_MARKER) + 1000
        assert len(low_part) == len(evo.TRUNCATION_MARKER) + 1000

    def test_a_short_partition_gives_its_unused_half_to_the_other(self):
        high = [self._rollout("HIGH-ONLY", 1.0)]
        low = [self._rollout(f"LOW-{i}", 0.0, answer=f"low-answer-{i}") for i in range(20)]
        cut = evo.attempts_block(high + low, 3000)
        high_part, low_part = cut.split("## Low-scoring trajectories\n\n")
        assert "HIGH-ONLY" in high_part and evo.TRUNCATION_MARKER not in high_part
        high_chars = len(high_part.removeprefix("## High-scoring trajectories\n\n").rstrip("\n"))
        assert len(low_part) == len(evo.TRUNCATION_MARKER) + 3000 - high_chars
        assert "low-answer-19" in low_part

    def test_a_partition_with_no_room_says_it_was_cut(self):
        cut = evo.attempts_block([self._rollout("H", 1.0), self._rollout("L", 0.0)], 1)
        assert "Trajectory 1" not in cut and "…[earlier trajectory text truncated]" in cut
        assert evo.attempts_block([], 100) == "(no trajectories)"

    def test_the_refiner_prompt_carries_the_truncated_traces(self, monkeypatch):
        rollouts = [
            {
                "question": f"Q{i}",
                "gold": "g",
                "score": 0.0,
                "result": {
                    "answer": None,
                    "stopped": "max_steps",
                    "steps": [
                        {"action": None, "raw_action": f"marker-{i}", "observation": "Invalid"}
                    ],
                },
            }
            for i in range(50)
        ]
        prompt = evo.build_refiner_prompt(
            load_prior("graphrag-navigator"),
            mode="static",
            metric="f1",
            rollouts=rollouts,
            rejected=[],
            max_chars=600,
        )
        assert "marker-49" in prompt and "marker-0\n" not in prompt
        assert '"name": "graphrag-navigator"' in prompt
        assert "Training batch scores (f1, 0 to 1): mean 0.000 over 50 tasks" in prompt
        assert "Refinement mode: static_incremental" in prompt
        assert "cycles are NOT allowed" in prompt

    def test_training_batches_are_wrapping_strides(self):
        train = [{"question": str(i)} for i in range(5)]

        def ids(k, size):
            return [item["question"] for item in evo.training_batch(train, k, size)]

        assert ids(1, 2) == ["0", "1"]
        assert ids(2, 2) == ["2", "3"]
        assert ids(3, 2) == ["4", "0"]
        assert ids(1, 99) == ["0", "1", "2", "3", "4"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('```json\n{"add_nodes": []}\n```', {"add_nodes": []}),
            (
                'Here are my edits: {"delete_edges": [{"source": "a", "target": "b"}]} Thanks.',
                {"delete_edges": [{"source": "a", "target": "b"}]},
            ),
            ('{"add_edges": [], "delete_nodes": ["x"]}', {"add_edges": [], "delete_nodes": ["x"]}),
            ('I consider {x} then {"add_nodes": [{"id": "n"}]}', {"add_nodes": [{"id": "n"}]}),
        ],
    )
    def test_extract_edits(self, text, expected):
        assert evo.extract_edits(text) == (expected, None)

    def test_extract_edits_failure(self):
        edits, error = evo.extract_edits("no json here")
        assert edits is None and "not a JSON object" in error

    def test_trajectory_limit_default(self):
        assert get_settings().evolution_trajectory_max_chars == 24000
