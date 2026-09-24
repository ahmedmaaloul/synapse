# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Tests for the Procedural Graphs benchmark harness (``benchmarks/procedural``).

Hermetic: the solver, guidance and refiner models are scripted fakes that go
through the harness's own metering, the Navigator's tools are fakes, Neo4j is
``fake_neo4j`` or a monkeypatched readiness check, and the procedural store
is an in-memory recorder. Nothing here can reach a model provider, and the
dry-run tests make every paid route explode to prove it.

Five jobs:

  1. **The demo QA set is what it claims.** The shipped ``demo_qa.json``
     verifies against the demo graph fixture, and the verifier catches each
     kind of defect it is responsible for (a fact not in the fixture, a wrong
     comparison, an ungrounded or ambiguous answer, bad split sizes).
  2. **System wiring.** Each system reaches the REAL Navigator with the right
     graph, guidance mode and scope, and only the test split is scored.
  3. **Metric aggregation and the effect floor.** EM/F1 against gold and
     aliases, per-question means, the estimated-token flag, and verdicts that
     refuse to call a gap smaller than one question.
  4. **Spend.** The call and USD caps stop a run before the call that would
     cross them, evolution cannot starve the evolved rows, and ``--dry-run``
     is an upper bound that spends nothing.
  5. **The refusals.** HotpotQA without ``--reuse-graph``, an unseeded demo
     graph, an un-ingested HotpotQA sample, an unreachable Neo4j, a missing
     model and a projected cost over the cap all stop the run with their exit
     code instead of producing a number.
"""

from __future__ import annotations

import copy
import json
import re

import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.services import procedural_guidance as pgd
from app.services import procedural_store as store
from app.services.graph_agent import TOOL_NAMES
from app.services.llm_provider import ProviderConfigError
from app.services.procedural_graph import serialize_full
from app.services.procedural_store import load_prior
from benchmarks.procedural import run_procedural as rp
from benchmarks.public import cost, run_hotpotqa
from benchmarks.run_benchmark import DatasetError, IntegrityError

SOLVER_USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
GUIDE_USAGE = {"input_tokens": 300, "output_tokens": 40, "total_tokens": 340}
EVOLVED = "graphrag-navigator-evolved-bench"

# Structurally valid refiner edits (the same shape as the evolution tests use).
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


@pytest.fixture(autouse=True)
def _fresh_guidance_caches():
    pgd.clear_caches()
    yield
    pgd.clear_caches()


# ── Fakes ────────────────────────────────────────────────────────────────────
class FakeSolver:
    """The Navigator's model: ``searches`` lookups, then the answer for the question.

    Reads the question and the number of turns taken from the prompt it is
    sent, so it behaves the same whichever system is driving it.
    """

    def __init__(self, answers: dict[str, str], *, searches: int = 1, usage=SOLVER_USAGE):
        self.answers = dict(answers)
        self.searches = searches
        self.usage = usage
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        question = prompt.rsplit("\nQuestion: ", 1)[1].split("\n", 1)[0]
        trajectory = prompt.rsplit("Current Trajectory:", 1)[1]
        turns = len(re.findall(r"^Action: ", trajectory, re.MULTILINE))
        if turns < self.searches:
            text = (
                "Thought: Find the entity first.\n"
                f'Action: search_entities(query="{question[:30]}")'
            )
        else:
            answer = self.answers.get(question, "I do not know")
            text = f'Thought: The evidence answers it.\nAction: answer(text="{answer}")'
        if self.usage is None:
            return AIMessage(content=text)
        return AIMessage(content=text, usage_metadata=self.usage)


class FailingSolver:
    """A provider that is up but errors on every call."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise RuntimeError("provider exploded")


class FakeGuide:
    """The generative-guidance model; numbers its replies."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage(
            content=f"Guidance #{len(self.prompts)}: search the entity, then answer.",
            usage_metadata=GUIDE_USAGE,
        )


class FakeRefiner:
    def __init__(self, edits: dict) -> None:
        self.edits = edits
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage(
            content=json.dumps(self.edits),
            usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
        )


def fake_tools(observation: str = "nothing more to see"):
    """Recording stand-ins for the five retrieval tools."""
    calls: list[tuple[str, dict]] = []

    def make(name):
        async def tool(**kwargs):
            calls.append((name, kwargs))
            return f"{name}: {observation}"

        return tool

    return {name: make(name) for name in TOOL_NAMES if name != "answer"}, calls


class MemoryStore:
    """In-memory stand-in for the procedural store calls evolution makes."""

    def __init__(self) -> None:
        self.graphs: dict[str, tuple] = {}
        self.loads: list[str] = []
        self.saves: list[dict] = []
        self.trajectories: list[dict] = []
        self.rejections: list[dict] = []

    def install(self, monkeypatch) -> MemoryStore:
        async def load_graph_with_meta(name):
            self.loads.append(name)
            return self.graphs.get(name)

        async def save_graph(graph, *, score=None, note="", edits=None, previous=None):
            version = (self.graphs.get(graph.name, (None, {"version": 0}))[1]["version"]) + 1
            self.graphs[graph.name] = (graph, {"version": version, "score": score})
            self.saves.append({"name": graph.name, "version": version, "note": note})
            return version

        async def record_rejection(name, **kwargs):
            self.rejections.append({"name": name, **kwargs})

        async def record_trajectory(name, **kwargs):
            self.trajectories.append({"name": name, **kwargs})

        for fn in (load_graph_with_meta, save_graph, record_rejection, record_trajectory):
            monkeypatch.setattr(store, fn.__name__, fn)
        return self


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def demo() -> rp.Dataset:
    return rp.load_demo_dataset()


@pytest.fixture
def prior():
    return load_prior("graphrag-navigator")


@pytest.fixture
def qa_payload() -> list:
    return rp.load_json(rp.DEMO_QA_PATH)


@pytest.fixture
def fixture_graph() -> dict:
    return rp.load_json(rp.DEMO_GRAPH_PATH)


def gold_answers(dataset: rp.Dataset) -> dict[str, str]:
    return {item.question: item.answer for item in dataset.items}


def guard(*, max_calls=None, max_usd=None, model="gpt-4o-mini") -> rp.SpendGuard:
    return rp.SpendGuard(model, max_usd=max_usd, max_calls=max_calls)


async def bench(dataset, prior, systems, *, solver=None, spend=None, max_steps=4, **kwargs):
    """``rp.execute`` with fake models and tools; returns (result, solver, guide, guard)."""
    tools = kwargs.pop("tools", None) or fake_tools()[0]
    solver = solver or FakeSolver(gold_answers(dataset))
    guide = FakeGuide()
    spend = spend or guard()
    refiner = kwargs.pop("refiner", None) or FakeRefiner(GOOD_EDITS)
    result = await rp.execute(
        dataset,
        systems=systems,
        prior=prior,
        guard=spend,
        models=rp.Models(solver=solver, guidance=guide, refiner=refiner),
        tools=tools,
        max_steps=max_steps,
        echo=False,
        **kwargs,
    )
    return result, solver, guide, spend


def hotpot_record(i: int) -> dict:
    """One HotpotQA record in the dataset's own on-disk shape."""
    titles = (f"Gold{i}A", f"Gold{i}B", f"Other{i}C", f"Other{i}D")
    return {
        "_id": f"hq{i:02d}",
        "question": f"Who founded Gold{i}A?",
        "answer": f"Founder {i}",
        "type": "bridge",
        "level": "hard",
        "context": [
            [title, [f"{title} is a thing. ", f"{title} was founded in 19{i}0. "]]
            for title in titles
        ],
        "supporting_facts": [[titles[0], 0], [titles[1], 1]],
    }


@pytest.fixture
def hotpot_file(tmp_path):
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps([hotpot_record(i) for i in range(7)]), encoding="utf-8")
    return path


@pytest.fixture
def no_spending(monkeypatch):
    """Make every route to a paid model, an embedder, Neo4j or the network explode."""
    calls: list[str] = []

    def explode(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was called during a run that must not spend")

        return _raise

    from app import neo4j_driver
    from app.services import (
        chat_engine,
        chunk_store,
        graph_agent,
        llm_provider,
        procedural_evolution,
    )
    from benchmarks.public import hotpotqa

    for module, attribute in (
        (llm_provider, "get_chat_llm"),
        (llm_provider, "get_embeddings"),
        (graph_agent, "get_chat_llm"),
        (pgd, "get_chat_llm"),
        (pgd, "get_embeddings"),
        (procedural_evolution, "get_chat_llm"),
        (chat_engine, "get_embeddings"),
        (chunk_store, "get_embeddings"),
        (neo4j_driver, "get_driver"),
        (neo4j_driver, "verify_connectivity"),
        (hotpotqa, "ensure_dataset"),
        (rp, "make_models"),
    ):
        monkeypatch.setattr(module, attribute, explode(f"{module.__name__}.{attribute}"))
    return calls


@pytest.fixture
def reachable_neo4j(monkeypatch):
    """Neo4j 'up' for main(): connectivity true, schema and close are no-ops."""
    from app import neo4j_driver
    from app.services import graph_schema

    async def yes():
        return True

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(neo4j_driver, "verify_connectivity", yes)
    monkeypatch.setattr(neo4j_driver, "close_driver", nothing)
    monkeypatch.setattr(graph_schema, "ensure_schema", nothing)


def seeded_demo_handler(query: str, params: dict) -> list:
    if query == rp.DEMO_ENTITIES_QUERY:
        return [{"name": name} for name in params["names"]]
    if query == rp.GRAPH_COUNTS_QUERY:
        return [{"entities": 53, "chunks": 0}]
    return []


def mixed_demo_handler(query: str, params: dict) -> list:
    """The fixture is all there, plus another document (106 entities, 5 chunks)."""
    if query == rp.GRAPH_COUNTS_QUERY:
        return [{"entities": 106, "chunks": 5}]
    return seeded_demo_handler(query, params)


# ── 1. The demo QA set ───────────────────────────────────────────────────────
class TestDemoQA:
    def test_the_shipped_set_verifies_against_the_demo_graph(self, qa_payload, fixture_graph):
        assert rp.verify_demo_qa(qa_payload, fixture_graph) == []

    def test_it_has_the_promised_shape(self, demo):
        assert len(demo.items) == 30
        assert demo.sizes() == {"train": 12, "val": 9, "test": 9}
        multi_hop = [item for item in demo.items if item.hops >= 2]
        assert len(multi_hop) / len(demo.items) >= 0.6
        assert all(len(item.answer.split()) <= rp.MAX_ANSWER_WORDS for item in demo.items)
        assert len({item.id for item in demo.items}) == 30

    def test_every_evidence_entity_is_in_the_fixture(self, qa_payload, fixture_graph):
        names = {entity["name"] for entity in fixture_graph["entities"]}
        for item in qa_payload:
            relations, descriptions = rp.evidence_facts(item)
            used = {e for r in relations for e in (r[0], r[2])} | {d[0] for d in descriptions}
            assert used <= names, item["id"]

    def _broken(self, payload, item_id, **changes):
        broken = copy.deepcopy(payload)
        for item in broken:
            if item["id"] == item_id:
                item.update(changes)
        return broken

    def test_a_relation_not_in_the_fixture_is_caught(self, qa_payload, fixture_graph):
        broken = self._broken(
            qa_payload,
            "demo-02",
            evidence={"relations": [["Frank Rosenblatt", "DESIGNED", "Perceptron"]]},
        )
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any("is not in the fixture" in p for p in problems)

    def test_a_description_snippet_not_in_the_fixture_is_caught(self, qa_payload, fixture_graph):
        broken = self._broken(
            qa_payload,
            "demo-12",
            evidence={"descriptions": [["Anthropic", "founded in 2019"]]},
            answer="2019",
        )
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any("not in the fixture description" in p for p in problems)

    def test_a_comparison_answer_is_recomputed_from_the_fixture_years(
        self, qa_payload, fixture_graph
    ):
        broken = self._broken(qa_payload, "demo-07", answer="Perceptron")
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any("give 'Turing Test'" in p for p in problems)

    def test_an_ungrounded_answer_is_caught(self, qa_payload, fixture_graph):
        broken = self._broken(qa_payload, "demo-03", answer="Alan Turing")
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any("not grounded in its evidence" in p for p in problems)

    def test_an_ambiguous_final_hop_is_caught(self, qa_payload, fixture_graph):
        # Two relations of the same kind end at the answer's other endpoint.
        extra = copy.deepcopy(fixture_graph)
        extra["relationships"].append(
            {"source": "Ada Lovelace", "target": "Perceptron", "type": "INVENTED"}
        )
        problems = rp.verify_demo_qa(qa_payload, extra)
        assert any("is ambiguous" in p and "demo-02" in p for p in problems)

    def test_split_sizes_and_hop_counts_are_enforced(self, qa_payload, fixture_graph):
        broken = self._broken(qa_payload, "demo-30", split="train", hops=2)
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any(p.startswith("splits must be") for p in problems)
        assert any("hops is 2 but the evidence has 1" in p for p in problems)

    def test_a_broken_file_is_refused_before_any_run(self, tmp_path, qa_payload):
        broken = self._broken(qa_payload, "demo-09", answer="Google DeepMind")
        path = tmp_path / "qa.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(DatasetError, match="does not verify"):
            rp.load_demo_dataset(path)

    def test_the_file_is_what_the_evolve_cli_reads(self, qa_payload):
        """A JSON array of {question, answer, split}: ``synapse-graphrag evolve
        --train demo_qa.json --train-split train`` consumes it unchanged."""
        assert isinstance(qa_payload, list)
        for item in qa_payload:
            assert isinstance(item["question"], str) and item["question"].strip()
            answer = item["answer"]
            forms = answer if isinstance(answer, list) else [answer]
            assert forms and all(isinstance(f, str) and f.strip() for f in forms)
            assert item["split"] in rp.SPLITS

    def test_aliases_count_as_gold(self, demo):
        item = next(i for i in demo.items if i.id == "demo-21")
        assert item.answer == "Convolutional Neural Network"
        assert item.gold == ["Convolutional Neural Network", "CNN"]
        assert item.as_evolution_item() == {"question": item.question, "answer": item.gold}

    def test_repeated_aliases_are_caught(self, qa_payload, fixture_graph):
        broken = self._broken(qa_payload, "demo-16", answer=["Google DeepMind", "google deepmind"])
        problems = rp.verify_demo_qa(broken, fixture_graph)
        assert any("aliases repeat" in p for p in problems)

    def test_a_file_that_is_not_an_array_is_refused(self, fixture_graph):
        assert rp.verify_demo_qa({"items": []}, fixture_graph) == [
            "the QA file must be a JSON array of items"
        ]

    def test_the_splits_export_as_evolve_cli_input(self, demo, tmp_path):
        paths = rp.export_splits(demo, tmp_path / "splits")
        assert [p.name for p in paths] == ["train.json", "val.json", "test.json"]
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        assert [len(r) for r in rows] == [12, 9, 9]
        assert set(rows[0][0]) == {"question", "answer"}
        assert rows[0][0]["answer"] == "1843"
        assert rows[1][3]["answer"] == ["Google DeepMind", "DeepMind"]  # aliases survive


class TestHotpotQADataset:
    def test_the_sample_is_split_40_30_30_by_position(self, hotpot_file):
        dataset = rp.load_hotpotqa_dataset(7, seed=1, data_path=hotpot_file, allow_download=False)
        assert dataset.sizes() == {"train": 3, "val": 2, "test": 2}
        assert len(dataset.notes["titles"]) == 28  # 7 questions x 4 paragraphs

    def test_fewer_questions_than_splits_is_refused(self):
        with pytest.raises(DatasetError, match="at least 3"):
            rp.split_sizes(2)


# ── 2. System wiring (through the real Navigator) ────────────────────────────
class TestSystems:
    def test_the_default_systems_are_the_four_base_rows(self):
        assert rp.resolve_systems(None, evolve=False) == list(rp.BASE_SYSTEMS)
        assert rp.resolve_systems(None, evolve=True) == [*rp.BASE_SYSTEMS, *rp.EVOLVED_SYSTEMS]

    def test_a_subset_comes_back_in_canonical_order(self):
        assert rp.resolve_systems("pg_gen_full, no_pg", evolve=False) == ["no_pg", "pg_gen_full"]

    def test_unknown_and_premature_evolved_systems_are_refused(self):
        with pytest.raises(ValueError, match="unknown system"):
            rp.resolve_systems("pg_magic", evolve=False)
        with pytest.raises(ValueError, match="need --evolve"):
            rp.resolve_systems("pg_evolved_raw", evolve=False)

    async def test_only_the_test_split_is_scored(self, demo, prior):
        result, solver, _, _ = await bench(demo, prior, ["no_pg"])
        run = result.runs["no_pg"]
        assert [o.item.id for o in run.outcomes] == [i.id for i in demo.split("test")]
        asked = {p.rsplit("\nQuestion: ", 1)[1].split("\n", 1)[0] for p in solver.prompts}
        assert asked == {i.question for i in demo.split("test")}

    async def test_no_pg_runs_without_any_procedural_graph(self, demo, prior):
        result, solver, guide, _ = await bench(demo, prior, ["no_pg"])
        assert not any("Procedural Graph Guidance:" in p for p in solver.prompts)
        assert guide.prompts == []
        outcome = result.runs["no_pg"].outcomes[0]
        assert outcome.graph is None and outcome.guidance_llm_calls == 0

    async def test_pg_raw_local_injects_the_local_subgraph_with_no_guidance_call(
        self, demo, prior
    ):
        result, solver, guide, _ = await bench(demo, prior, ["pg_raw_local"])
        assert all("Procedural Graph Guidance:" in p for p in solver.prompts)
        assert "Active Cognitive Node: [Start]" in solver.prompts[0]
        assert "Active Cognitive Node: [search_entities]" in solver.prompts[1]
        assert guide.prompts == []  # raw: zero guidance LLM calls
        agg = result.runs["pg_raw_local"].agg
        assert agg["guidance_llm_calls"] == 0
        assert agg["localization"] == {"start": 9, "exact": 9}
        assert result.runs["pg_raw_local"].outcomes[0].graph == {
            "name": "graphrag-navigator",
            "version": None,
        }

    async def test_pg_gen_local_makes_one_guidance_call_per_step_on_the_local_scope(
        self, demo, prior
    ):
        result, solver, guide, _ = await bench(demo, prior, ["pg_gen_local"])
        agg = result.runs["pg_gen_local"].agg
        assert agg["steps"] == 2
        assert agg["guidance_llm_calls"] == 2
        assert agg["llm_calls"] == 4
        assert len(guide.prompts) == 18
        assert "Active Cognitive Node: [Start]" in guide.prompts[0]
        assert "Procedural Graph: [graphrag-navigator]" not in guide.prompts[0]
        assert "Guidance #1: search the entity" in solver.prompts[0]

    async def test_pg_gen_full_hands_the_whole_graph_to_the_guidance_model(self, demo, prior):
        result, _, guide, _ = await bench(demo, prior, ["pg_gen_full"])
        full = serialize_full(prior)
        assert guide.prompts and all(full in prompt for prompt in guide.prompts)
        assert result.runs["pg_gen_full"].agg["guidance_llm_calls"] == 2

    async def test_every_system_is_metered_on_its_own_ledger(self, demo, prior):
        result, _, _, spend = await bench(demo, prior, list(rp.BASE_SYSTEMS))
        by_label = {ledger.label: ledger.usage for ledger in spend.ledgers}
        assert by_label["no_pg (Navigator, no procedural graph)"].calls == 18
        assert by_label["pg_gen_local (Expert prior, generative local guidance)"].calls == 36
        assert spend.calls == 18 + 18 + 36 + 36
        # The ledger and the Navigator's own usage agree.
        navigator_calls = sum(r.agg["llm_calls_total"] for r in result.runs.values())
        assert navigator_calls == spend.calls

    async def test_a_system_whose_every_run_errored_is_refused(self, demo, prior):
        with pytest.raises(IntegrityError, match="model error"):
            await bench(demo, prior, ["no_pg"], solver=FailingSolver())

    async def test_an_empty_test_split_is_refused(self, demo, prior):
        empty = rp.Dataset("demo", [i for i in demo.items if i.split != "test"], "x")
        with pytest.raises(IntegrityError, match="test split is empty"):
            await bench(empty, prior, ["no_pg"])


# ── 3. Metrics and the effect floor ──────────────────────────────────────────
class TestMetrics:
    async def test_em_and_f1_are_means_over_the_test_questions(self, demo, prior):
        answers = gold_answers(demo)
        tests = {i.id: i for i in demo.split("test")}
        answers[tests["demo-23"].question] = "Hassabis"  # partial: F1 2/3, EM 0
        answers[tests["demo-28"].question] = "AMD"  # wrong
        result, _, _, _ = await bench(demo, prior, ["no_pg"], solver=FakeSolver(answers))
        agg = result.runs["no_pg"].agg
        assert agg["n"] == 9
        assert agg["em"] == pytest.approx(7 / 9)
        assert agg["f1"] == pytest.approx((7 + 2 / 3) / 9)
        assert agg["steps"] == 2 and agg["llm_calls"] == 2
        assert agg["input_tokens"] == 200 and agg["output_tokens"] == 20
        assert agg["answered"] == 1.0 and agg["errors"] == 0
        assert agg["estimated"] is False

    async def test_missing_usage_metadata_is_flagged_as_estimated(self, demo, prior):
        solver = FakeSolver(gold_answers(demo), usage=None)
        result, _, _, spend = await bench(demo, prior, ["no_pg"], solver=solver)
        agg = result.runs["no_pg"].agg
        assert agg["estimated"] is True and agg["estimated_questions"] == 9
        assert "≈" in "\n".join(rp.table_lines(result))
        assert spend.usage.estimated

    def test_an_alias_scores_as_the_answer(self, demo):
        item = next(i for i in demo.items if i.id == "demo-16")
        outcome = rp.outcome_from_result(item, {"answer": "DeepMind", "steps": []})
        assert outcome.em == 1.0 and outcome.f1 == 1.0

    def test_a_result_is_read_from_the_agent_result_shape(self, demo):
        item = demo.split("test")[0]
        outcome = rp.outcome_from_result(
            item,
            {
                "answer": None,
                "stopped": "max_steps",
                "parse_failures": 2,
                "steps": [{"localization": "start"}, {"localization": "none"}, {}],
                "usage": {"llm_calls": 3, "guidance_llm_calls": 1, "estimated": True},
                "latency_s": 1.5,
            },
        )
        assert (outcome.em, outcome.f1, outcome.steps) == (0.0, 0.0, 3)
        assert outcome.localizations == {"start": 1, "none": 1}
        agg = rp.aggregate([outcome])
        assert agg["max_steps_hits"] == 1 and agg["parse_failures_total"] == 2
        assert agg["answered"] == 0.0 and agg["estimated"] is True

    def test_a_gap_below_one_question_is_too_close_to_call(self):
        line = rp.verdict(
            "F1",
            "f1",
            {"n": 9, "f1": 0.45},
            {"n": 9, "f1": 0.50},
            baseline_label="no_pg",
            system_label="pg_raw_local",
        )
        assert "TOO CLOSE TO CALL" in line and "11.1 pp" in line

    def test_a_gap_of_exactly_one_question_is_called(self):
        # 5/9 − 4/9 is a hair under 1/9 in floating point; it must still count.
        line = rp.verdict(
            "EM",
            "em",
            {"n": 9, "em": 4 / 9},
            {"n": 9, "em": 5 / 9},
            baseline_label="no_pg",
            system_label="pg_raw_local",
        )
        assert "pg_raw_local ahead of no_pg" in line
        assert "not a significance result" in line

    def test_ties_and_single_questions_are_never_called(self):
        tie = rp.verdict("EM", "em", {"n": 9, "em": 0.5}, {"n": 9, "em": 0.5},
                         baseline_label="a", system_label="b")
        one = rp.verdict("EM", "em", {"n": 1, "em": 0.0}, {"n": 1, "em": 1.0},
                         baseline_label="a", system_label="b")
        assert "TIE" in tie
        assert "NOT SCOREABLE" in one

    async def test_verdicts_cover_only_pairs_that_ran(self, demo, prior):
        result, _, _, _ = await bench(demo, prior, ["no_pg", "pg_raw_local"])
        text = "\n".join(rp.verdict_lines(result))
        assert "pg_raw_local vs no_pg" in text
        assert "pg_gen_local" not in text
        assert "TIE" in text  # the fake answers identically with or without guidance


# ── 4. Spend: caps, evolution budget, dry run ────────────────────────────────
class TestSpendGuard:
    async def test_measured_usage_is_metered_per_call(self):
        spend = guard()
        ledger = spend.ledger("x")
        model = spend.wrap(FakeSolver({}), ledger)
        await model.ainvoke([_message("\nQuestion: q?\nCurrent Trajectory:\n")])
        assert ledger.usage.prompt_tokens == 100 and ledger.usage.completion_tokens == 10
        assert ledger.usage.estimated is False

    async def test_the_call_cap_stops_before_the_call_that_would_cross_it(self, demo, prior):
        solver = FakeSolver(gold_answers(demo))
        with pytest.raises(rp.BudgetExceeded, match="LLM-call cap") as caught:
            await bench(
                demo,
                prior,
                ["no_pg", "pg_raw_local"],
                solver=solver,
                spend=guard(max_calls=19),
            )
        assert len(solver.prompts) == 19  # never a 20th call
        partial = caught.value.partial
        assert list(partial.runs) == ["no_pg"]  # only whole systems are reported

    async def test_the_usd_cap_stops_before_the_call_that_would_cross_it(self, demo, prior):
        # Each solver call is $0.000021 on gpt-4o-mini: three fit under $0.00005.
        solver = FakeSolver(gold_answers(demo))
        with pytest.raises(rp.BudgetExceeded, match="--max-usd"):
            await bench(demo, prior, ["no_pg"], solver=solver, spend=guard(max_usd=0.00005))
        assert len(solver.prompts) == 3

    async def test_an_unpriced_model_is_capped_by_calls_alone(self, demo, prior):
        spend = guard(max_usd=0.0, model="my-local-model")
        result, _, _, _ = await bench(demo, prior, ["no_pg"], spend=spend)
        assert spend.usd() is None
        assert result.runs["no_pg"].agg["n"] == 9

    def test_a_real_run_is_priced_as_the_model_it_calls(self):
        # gpt-4o metered as gpt-4o-mini is ~17x under: --model must not weaken the cap.
        model, note = rp.billing_model("gpt-4o-mini", "gpt-4o")
        assert model == "gpt-4o" and "estimates only" in note
        assert rp.billing_model("", "gpt-4o") == ("gpt-4o", None)
        assert rp.billing_model("gpt-4o", "gpt-4o") == ("gpt-4o", None)
        # An unpriced id (a deployment alias, a local model) may be mapped explicitly.
        model, note = rp.billing_model("gpt-4o", "my-azure-deployment")
        assert model == "gpt-4o" and "only as accurate as that mapping" in note

    def test_the_cap_is_a_spend_cap_that_evolution_understands(self):
        from app.services.procedural_evolution import SpendCapReached

        assert issubclass(rp.BudgetExceeded, SpendCapReached)
        assert issubclass(rp.BudgetExceeded, RuntimeError)

    async def test_a_usd_cap_refusing_the_refiner_stops_evolution_as_budget(
        self, demo, prior, monkeypatch
    ):
        # no_pg (18 calls) + evolution baseline (18) + batch (6) = 42 solver calls of
        # $0.000021: the cap binds exactly at the refiner call, which is refused.
        MemoryStore().install(monkeypatch)
        refiner = FakeRefiner(GOOD_EDITS)
        with pytest.raises(rp.BudgetExceeded, match="--max-usd") as caught:
            await bench(
                demo,
                prior,
                ["no_pg"],
                spend=guard(max_usd=0.00087, max_calls=1000),
                max_steps=3,
                evolve_rounds=1,
                evolve_batch_size=3,
                refiner=refiner,
            )
        report = caught.value.partial.evolution
        assert refiner.prompts == []  # refused before it was made
        assert report["stopped"] == "budget"  # not "completed" after a "refiner_error"
        assert [r["reason"] for r in report["rounds"]] == ["budget"]
        assert report["llm_calls"] == 18 + 6  # the refused call is not counted

    async def test_an_evolution_budget_below_one_round_stops_as_a_cap_and_keeps_the_paid_rows(
        self, demo, prior, monkeypatch
    ):
        # 60 − 18 spent by no_pg leaves 42; one round needs (9 + 3 + 9) × 3 + 1 = 64.
        # evolve() refuses that before its first call: the harness must report it as a
        # cap (exit 5, INCOMPLETE with no_pg) instead of a traceback that loses no_pg.
        from app.services.procedural_evolution import EvolutionBudgetTooSmall

        MemoryStore().install(monkeypatch)
        solver, refiner = FakeSolver(gold_answers(demo)), FakeRefiner(GOOD_EDITS)
        with pytest.raises(rp.BudgetExceeded, match="evolution refused to start") as caught:
            await bench(
                demo,
                prior,
                ["no_pg"],
                solver=solver,
                spend=guard(max_calls=60),
                max_steps=3,
                evolve_rounds=1,
                evolve_batch_size=3,
                refiner=refiner,
            )
        assert isinstance(caught.value.__cause__, EvolutionBudgetTooSmall)
        assert "needs 64 LLM calls" in str(caught.value) and "left it 42" in str(caught.value)
        assert len(solver.prompts) == 18 and refiner.prompts == []  # evolution spent nothing
        partial = caught.value.partial
        assert list(partial.runs) == ["no_pg"] and partial.runs["no_pg"].agg["n"] == 9

    @pytest.mark.parametrize("slack", [0, -1])
    async def test_the_evolution_floor_is_exactly_what_a_worst_case_run_needs(
        self, demo, prior, monkeypatch, slack
    ):
        # A solver that never answers makes every episode take all its steps.
        MemoryStore().install(monkeypatch)
        systems = ["no_pg", "pg_evolved_raw"]
        floor = rp.evolution_floor(demo, systems=systems, max_steps=3, evolve_batch_size=3)
        assert (floor.before, floor.evolution, floor.reserve) == (27, 64, 27)
        spend = guard(max_calls=floor.total + slack)
        kwargs = {
            "solver": FakeSolver({}, searches=100),
            "spend": spend,
            "max_steps": 3,
            "evolve_rounds": 1,
            "evolve_batch_size": 3,
        }
        if slack == 0:
            result, _, _, _ = await bench(demo, prior, systems, **kwargs)
            assert result.evolution["rounds_run"] == 1
            assert list(result.runs) == systems and spend.calls == floor.total
        else:
            with pytest.raises(rp.BudgetExceeded, match="evolution refused to start"):
                await bench(demo, prior, systems, **kwargs)
            assert floor.refusal(floor.total + slack) is not None
            assert spend.calls == floor.before  # refused before evolution's first call


def _message(content: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)


class TestEvolution:
    def _report(self, final_version):
        return {
            "graph": EVOLVED,
            "mode": "static",
            "metric": "f1",
            "guidance": "raw",
            "rounds_requested": 2,
            "rounds_run": 2,
            "accepted_rounds": 1 if final_version else 0,
            "stopped": "completed",
            "baseline_score": 0.5,
            "final_score": 0.6,
            "final_version": final_version,
            "rounds": [],
            "llm_calls": 10,
            "effect_floor": 1 / 9,
        }

    async def test_evolution_trains_on_train_gates_on_val_and_saves_under_a_new_name(
        self, demo, prior
    ):
        seen: dict = {}
        evolved = prior.copy()
        evolved.name = EVOLVED

        async def fake_evolve(name, **kwargs):
            seen.update(kwargs, name=name)
            return self._report(3)

        async def fake_load(name):
            seen["loaded"] = name
            return evolved, {"version": 3}

        result, _, _, spend = await bench(
            demo,
            prior,
            ["no_pg", "pg_evolved_raw"],
            spend=guard(max_calls=500),
            evolve_rounds=2,
            evolve_batch_size=5,
            evolve_fn=fake_evolve,
            load_evolved=fake_load,
        )
        assert seen["name"] == EVOLVED and seen["loaded"] == EVOLVED
        assert seen["mode"] == "static" and seen["initial"] is prior
        assert (seen["rounds"], seen["batch_size"]) == (2, 5)
        assert [i["question"] for i in seen["train"]] == [i.question for i in demo.split("train")]
        assert [i["question"] for i in seen["val"]] == [i.question for i in demo.split("val")]
        assert seen["val"][3]["answer"] == ["Google DeepMind", "DeepMind"]  # aliases kept
        # 500 − 18 spent by no_pg − 9 test x 4 steps reserved for pg_evolved_raw.
        assert seen["max_llm_calls"] == 500 - 18 - 36
        outcome = result.runs["pg_evolved_raw"].outcomes[0]
        assert outcome.graph == {"name": EVOLVED, "version": 3}
        assert "Evolution (Algorithm 1)" in rp.results_markdown(result)

    async def test_when_nothing_is_accepted_the_evolved_rows_run_the_prior(self, demo, prior):
        loads: list[str] = []

        async def fake_evolve(name, **kwargs):
            return self._report(None)

        async def fake_load(name):
            loads.append(name)
            return None

        result, _, _, _ = await bench(
            demo,
            prior,
            ["pg_evolved_raw"],
            evolve_rounds=1,
            evolve_fn=fake_evolve,
            load_evolved=fake_load,
        )
        assert loads == []
        assert result.evolved_graph.name == EVOLVED
        assert result.evolved_graph.to_dict()["edges"] == prior.to_dict()["edges"]
        assert result.runs["pg_evolved_raw"].outcomes[0].graph == {
            "name": EVOLVED,
            "version": None,
        }

    async def test_evolution_never_falls_back_to_an_unmetered_refiner(self, demo, prior):
        solver = FakeSolver(gold_answers(demo))
        with pytest.raises(ValueError, match="needs Models.refiner"):
            await rp.execute(
                demo,
                systems=["pg_evolved_raw"],
                prior=prior,
                guard=guard(),
                models=rp.Models(solver=solver, guidance=FakeGuide(), refiner=None),
                tools=fake_tools()[0],
                max_steps=2,
                evolve_rounds=1,
                echo=False,
            )
        assert solver.prompts == []

    async def test_an_accepted_graph_that_cannot_be_read_back_is_refused(self, demo, prior):
        async def fake_evolve(name, **kwargs):
            return self._report(2)

        async def fake_load(name):
            return None

        with pytest.raises(IntegrityError, match="cannot be read back"):
            await bench(
                demo,
                prior,
                ["pg_evolved_raw"],
                evolve_rounds=1,
                evolve_fn=fake_evolve,
                load_evolved=fake_load,
            )

    async def test_the_real_loop_runs_end_to_end_and_never_touches_the_default_graph(
        self, demo, prior, monkeypatch
    ):
        memory = MemoryStore().install(monkeypatch)
        refiner = FakeRefiner(GOOD_EDITS)
        result, _, _, spend = await bench(
            demo,
            prior,
            ["pg_raw_local", "pg_evolved_raw"],
            spend=guard(max_calls=1000),
            max_steps=3,
            evolve_rounds=1,
            evolve_batch_size=3,
            refiner=refiner,
        )
        report = result.evolution
        # The fake answers every question correctly with any graph: a tie, accepted.
        assert report["rounds"][0]["reason"] == "accepted"
        assert report["final_version"] == 1
        assert [save["name"] for save in memory.saves] == [EVOLVED]
        assert memory.loads == [EVOLVED]
        assert {t["name"] for t in memory.trajectories} == {EVOLVED}
        assert len(refiner.prompts) == 1
        # Baseline (9 val) + batch (3 train) + validation (9 val) rollouts of 2 calls, + 1.
        evolution = next(ledger for ledger in spend.ledgers if "evolution" in ledger.label)
        assert evolution.usage.calls == 2 * (9 + 3 + 9) + 1
        assert ("search_passages", "find_path") in {
            (e.source, e.target) for e in result.evolved_graph.edges
        }
        assert result.runs["pg_evolved_raw"].outcomes[0].graph == {
            "name": EVOLVED,
            "version": 1,
        }


class TestEstimate:
    def test_generative_guidance_doubles_the_call_bound(self, demo, prior):
        plan = rp.estimate_plan(
            demo, systems=["no_pg", "pg_raw_local", "pg_gen_local"], prior=prior, max_steps=5
        )
        calls = [phase.usage.calls for phase in plan.phases]
        assert calls == [9 * 5, 9 * 5, 9 * 5 * 2]
        assert all(phase.usage.estimated for phase in plan.phases)

    def test_the_evolution_phase_follows_algorithm_1(self, demo, prior):
        plan = rp.estimate_plan(
            demo,
            systems=["no_pg"],
            prior=prior,
            max_steps=4,
            evolve_rounds=3,
            evolve_batch_size=6,
        )
        evolution = plan.phases[-1]
        # S0 on val, then per round: a train batch, one refiner call, val again.
        assert evolution.usage.calls == (9 + 3 * (6 + 9)) * 4 + 3

    @pytest.mark.parametrize("evolve_guidance", ["raw", "generative"])
    def test_the_default_call_cap_always_covers_the_evolution_floor(
        self, demo, prior, evolve_guidance
    ):
        # The default --max-llm-calls is the plan's bound: it must never be refused.
        systems = rp.resolve_systems(None, evolve=True)
        kwargs = {"systems": systems, "max_steps": 4, "evolve_guidance": evolve_guidance}
        floor = rp.evolution_floor(demo, **kwargs)
        plan = rp.estimate_plan(demo, prior=prior, evolve_rounds=1, **kwargs)
        assert plan.calls == floor.total  # one round: the plan IS the floor
        assert floor.refusal(plan.calls) is None
        more = rp.estimate_plan(demo, prior=prior, evolve_rounds=3, **kwargs)
        assert floor.refusal(more.calls) is None

    def test_guidance_makes_the_prompt_bound_larger(self, demo, prior):
        plan = rp.estimate_plan(demo, systems=["no_pg", "pg_raw_local"], prior=prior, max_steps=3)
        unguided, raw = (phase.usage.prompt_tokens for phase in plan.phases)
        assert raw > unguided

    @pytest.mark.parametrize("system", list(rp.BASE_SYSTEMS))
    async def test_the_estimate_bounds_a_worst_case_run(self, demo, prior, system):
        """Every step taken, every observation at the cap: still inside the estimate."""
        solver = FakeSolver({}, searches=100, usage=None)  # never answers
        tools, _ = fake_tools(observation="x" * 5000)  # truncated to the cap
        max_steps = 4
        plan = rp.estimate_plan(demo, systems=[system], prior=prior, max_steps=max_steps)
        _, _, _, spend = await bench(
            demo, prior, [system], solver=solver, tools=tools, max_steps=max_steps
        )
        (phase,) = plan.phases
        (ledger,) = spend.ledgers
        assert ledger.usage.calls == phase.usage.calls
        assert ledger.usage.prompt_tokens <= phase.usage.prompt_tokens
        assert ledger.usage.completion_tokens <= phase.usage.completion_tokens

    async def test_the_estimate_covers_the_full_graph_fallback(self, demo, prior, monkeypatch):
        """A step that matches no node is guided with the WHOLE graph; the bound knows."""

        def no_embeddings():
            raise RuntimeError("no embedder")  # semantic localization is skipped

        class Rambler:
            """Never writes a parseable action, so every later step localizes to none."""

            async def ainvoke(self, messages):
                return AIMessage(content="Thought: hmm, let me think about it for a while.")

        monkeypatch.setattr(pgd, "get_embeddings", no_embeddings)
        plan = rp.estimate_plan(demo, systems=["pg_raw_local"], prior=prior, max_steps=3)
        result, _, _, spend = await bench(
            demo, prior, ["pg_raw_local"], solver=Rambler(), max_steps=3
        )
        assert result.runs["pg_raw_local"].agg["localization"]["none"] == 9 * 2
        (phase,) = plan.phases
        (ledger,) = spend.ledgers
        assert ledger.usage.prompt_tokens <= phase.usage.prompt_tokens

    def test_the_dry_run_flags_a_plan_over_the_cap(self, demo, prior):
        plan = rp.estimate_plan(demo, systems=["no_pg"], prior=prior, max_steps=8)
        lines = rp.dry_run_lines(
            plan, demo, systems=["no_pg"], model="gpt-4o-mini", max_steps=8, max_usd=0.0
        )
        text = "\n".join(lines)
        assert "UPPER BOUND" in text and "exceeds --max-usd" in text

    def test_an_unpriced_model_says_so_instead_of_inventing_a_number(self, demo, prior):
        plan = rp.estimate_plan(demo, systems=["no_pg"], prior=prior, max_steps=8)
        lines = rp.dry_run_lines(
            plan, demo, systems=["no_pg"], model="my-local-model", max_steps=8, max_usd=1.0
        )
        assert any("no price on file" in line for line in lines)


class TestDryRun:
    async def test_it_calls_no_model_touches_no_database_and_exits_zero(
        self, no_spending, capsys
    ):
        code = await rp.main(["--dry-run", "--model", "gpt-4o-mini"])
        assert code == 0
        assert no_spending == []
        out = capsys.readouterr().out
        assert "DRY RUN" in out and "no model was called" in out
        assert "TOTAL (upper bound)" in out and "$" in out
        assert "python -m benchmarks.procedural.run_procedural --dataset demo" in out

    async def test_the_evolution_plan_is_free_and_deterministic(self, no_spending, capsys):
        args = ["--dry-run", "--model", "gpt-4o-mini", "--evolve", "2", "--max-steps", "3"]
        assert await rp.main(args) == 0
        first = capsys.readouterr().out
        assert "evolution (2 round(s), raw guidance)" in first
        assert "pg_evolved_gen" in first
        assert await rp.main(args) == 0
        assert capsys.readouterr().out == first
        assert no_spending == []
        assert "⛔" not in first  # the default call cap always covers evolution

    async def test_a_call_cap_too_small_for_evolution_is_flagged(self, no_spending, capsys):
        # no_pg: 9 × 3 = 27; evolution: (9 + 3 + 9) × 3 + 1 = 64; nothing reserved.
        code = await rp.main(
            [
                "--dry-run", "--model", "gpt-4o-mini", "--systems", "no_pg", "--evolve", "1",
                "--evolve-batch-size", "3", "--max-steps", "3", "--max-llm-calls", "60",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "⛔ --max-llm-calls 60 is below the 91 calls this --evolve run can need" in out
        assert "at least 91" in out and "The real run would refuse to start" in out
        assert no_spending == []

    async def test_a_hotpotqa_dry_run_is_free_too(self, no_spending, hotpot_file, capsys):
        code = await rp.main(
            [
                "--dry-run", "--dataset", "hotpotqa", "--n", "7", "--reuse-graph",
                "--data", str(hotpot_file), "--no-download", "--model", "gpt-4o-mini",
            ]
        )
        assert code == 0
        assert no_spending == []
        assert "test 2" in capsys.readouterr().out

    async def test_exporting_the_splits_is_free(self, no_spending, tmp_path):
        code = await rp.main(["--export-splits", str(tmp_path / "splits")])
        assert code == 0
        assert (tmp_path / "splits" / "val.json").is_file()
        assert no_spending == []


# ── 5. Refusals (main) ───────────────────────────────────────────────────────
class TestRefusals:
    async def test_hotpotqa_without_reuse_graph_is_refused_before_anything_runs(
        self, no_spending, hotpot_file, capsys
    ):
        code = await rp.main(
            ["--dataset", "hotpotqa", "--n", "7", "--data", str(hotpot_file), "--no-download"]
        )
        assert code == 4
        assert "never ingests" in capsys.readouterr().err
        assert no_spending == []

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--n", "5"], "--n applies to --dataset hotpotqa only"),
            (["--systems", "pg_evolved_raw"], "need --evolve"),
            (["--systems", "pg_magic"], "unknown system"),
            (["--max-steps", "0"], "--max-steps must be between 1 and 20"),
            (["--evolve", "21"], "--evolve must be between 0 and 20"),
            (["--max-llm-calls", "0"], "--max-llm-calls must be at least 1"),
        ],
    )
    async def test_bad_arguments_exit_1(self, no_spending, capsys, argv, message):
        assert await rp.main(argv) == 1
        assert message in capsys.readouterr().err
        assert no_spending == []

    async def test_a_qa_file_that_does_not_verify_exits_1(
        self, no_spending, tmp_path, qa_payload, capsys
    ):
        qa_payload[0]["answer"] = "1842"
        path = tmp_path / "qa.json"
        path.write_text(json.dumps(qa_payload), encoding="utf-8")
        assert await rp.main(["--qa", str(path), "--dry-run"]) == 1
        assert "does not verify" in capsys.readouterr().err

    async def test_an_unknown_default_prior_exits_1(self, no_spending, monkeypatch, capsys):
        monkeypatch.setenv("PROCEDURAL_DEFAULT_GRAPH", "no-such-prior")
        assert await rp.main(["--dry-run"]) == 1
        assert "bundled expert prior" in capsys.readouterr().err

    async def test_a_projected_cost_over_the_cap_stops_before_neo4j(
        self, no_spending, capsys
    ):
        code = await rp.main(["--model", "gpt-4o-mini", "--max-usd", "0.0001"])
        assert code == 5
        assert "exceeds --max-usd" in capsys.readouterr().err
        assert no_spending == []  # not even a connectivity check

    async def test_an_evolve_call_cap_below_one_round_stops_before_anything_is_spent(
        self, no_spending, capsys
    ):
        # Before the fix this ran no_pg (18 paid calls), then evolve() raised a
        # traceback: nothing reported, nothing written, the spend lost.
        code = await rp.main(
            [
                "--systems", "no_pg", "--evolve", "1", "--evolve-batch-size", "3",
                "--max-steps", "3", "--max-llm-calls", "60", "--model", "gpt-4o-mini",
            ]
        )
        assert code == 5
        err = capsys.readouterr().err
        assert "--max-llm-calls 60 is below the 91 calls" in err
        assert "Refusing before any call" in err
        assert no_spending == []  # not even a connectivity check

    async def test_an_unreachable_neo4j_exits_2(self, monkeypatch, capsys):
        from app import neo4j_driver

        async def no():
            return False

        monkeypatch.setattr(neo4j_driver, "verify_connectivity", no)
        assert await rp.main(["--model", "gpt-4o-mini"]) == 2
        assert "Cannot reach Neo4j" in capsys.readouterr().err

    async def test_an_unseeded_demo_graph_exits_4_before_any_model_exists(
        self, reachable_neo4j, fake_neo4j, monkeypatch, capsys
    ):
        calls = fake_neo4j(lambda query, params: [])

        def no_models(*args, **kwargs):
            raise AssertionError("models were built for a graph that is not there")

        monkeypatch.setattr(rp, "make_models", no_models)
        assert await rp.main(["--model", "gpt-4o-mini"]) == 4
        assert "python -m scripts.seed_demo --clear" in capsys.readouterr().err
        assert calls and calls[0][0] == rp.DEMO_ENTITIES_QUERY

    async def test_a_demo_graph_mixed_with_other_documents_exits_4(
        self, reachable_neo4j, fake_neo4j, monkeypatch, capsys
    ):
        fake_neo4j(mixed_demo_handler)

        def no_models(*args, **kwargs):
            raise AssertionError("models were built for a graph that is not the demo's")

        monkeypatch.setattr(rp, "make_models", no_models)
        assert await rp.main(["--model", "gpt-4o-mini"]) == 4
        err = capsys.readouterr().err
        assert "106 entities and 5 source chunks" in err
        assert "--allow-mixed-graph" in err and "seed_demo --clear" in err

    async def test_the_readiness_check_reports_a_mixed_graph(self, fake_neo4j, demo):
        fake_neo4j(seeded_demo_handler)
        assert await rp.check_graph_ready(demo) == {"entities": 53, "chunks": 0, "mixed": False}
        fake_neo4j(mixed_demo_handler)
        with pytest.raises(rp.GraphNotReady, match="Other documents share this database"):
            await rp.check_graph_ready(demo)
        status = await rp.check_graph_ready(demo, allow_mixed=True)
        assert status == {"entities": 106, "chunks": 5, "mixed": True}

    async def test_a_cheaper_model_flag_cannot_weaken_the_usd_cap(
        self, no_spending, monkeypatch, demo, prior, capsys
    ):
        # The run would call gpt-4o. Its upper bound is over the cap even though
        # gpt-4o-mini's is under it: the pre-flight must price gpt-4o.
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
        get_settings.cache_clear()
        plan = rp.estimate_plan(
            demo, systems=list(rp.BASE_SYSTEMS), prior=prior,
            max_steps=get_settings().agent_max_steps,
        )
        cheap, dear = plan.usd("gpt-4o-mini"), plan.usd("gpt-4o")
        assert cheap < dear
        cap = (cheap + dear) / 2
        assert await rp.main(["--model", "gpt-4o-mini", "--max-usd", str(cap)]) == 5
        captured = capsys.readouterr()
        assert "prices --dry-run estimates only" in captured.out
        assert "exceeds --max-usd" in captured.err
        assert no_spending == []

    async def test_an_uningested_hotpotqa_sample_exits_4(
        self, reachable_neo4j, hotpot_file, monkeypatch, capsys
    ):
        from benchmarks.public import run_hotpotqa

        async def only_one():
            return {"Gold0A"}

        def no_models(*args, **kwargs):
            raise AssertionError("models were built for a corpus that is not there")

        monkeypatch.setattr(run_hotpotqa, "graph_documents", only_one)
        monkeypatch.setattr(rp, "make_models", no_models)
        code = await rp.main(
            [
                "--dataset", "hotpotqa", "--n", "7", "--reuse-graph",
                "--data", str(hotpot_file), "--no-download", "--model", "gpt-4o-mini",
            ]
        )
        assert code == 4
        assert "never ingests" in capsys.readouterr().err

    async def test_no_configured_model_exits_6(
        self, reachable_neo4j, fake_neo4j, monkeypatch, capsys
    ):
        fake_neo4j(seeded_demo_handler)

        def no_key(*args, **kwargs):
            raise ProviderConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

        monkeypatch.setattr(rp, "make_models", no_key)
        assert await rp.main(["--model", "gpt-4o-mini"]) == 6
        assert "OPENAI_API_KEY is empty" in capsys.readouterr().err


# ── The whole run, end to end, with fakes ────────────────────────────────────
class TestReport:
    async def test_a_full_run_writes_a_report_with_provenance(
        self, reachable_neo4j, fake_neo4j, monkeypatch, demo, tmp_path, capsys
    ):
        fake_neo4j(seeded_demo_handler)
        configured = run_hotpotqa.chat_model_name(get_settings())
        solver = FakeSolver(gold_answers(demo))
        tools, tool_calls = fake_tools()
        monkeypatch.setattr(
            rp, "make_models", lambda settings, evolve: rp.Models(solver, FakeGuide())
        )
        monkeypatch.setattr(rp, "navigator_tools", lambda: tools)
        monkeypatch.setattr(rp, "git_commit", lambda: "abc1234")
        out = tmp_path / "results.md"
        code = await rp.main(
            [
                "--systems", "no_pg,pg_raw_local", "--max-steps", "3",
                "--model", "gpt-4o-mini", "--out", str(out),
            ]
        )
        assert code == 0
        assert tool_calls  # the injected tools, not the database, answered
        text = out.read_text(encoding="utf-8")
        for expected in (
            "## Provenance",
            "**Commit:** abc1234",
            # The model the run calls; --model only priced it (it has no price).
            f"**Model:** `{configured}` (provider `",
            "priced as `gpt-4o-mini`",
            "seed 20260721",
            "train 12 / val 9 / test 9",
            "53 entities, 0 chunks",
            "Effect floor: one question = 11.1 points",
            "| `no_pg` |",
            "| `pg_raw_local` |",
            "SELF-AUTHORED QUESTIONS",
            "NO SOURCE PASSAGES ON THE DEMO GRAPH",
            "--systems no_pg,pg_raw_local --max-steps 3",
        ):
            assert expected in text, expected
        assert f"📄 Wrote {out}" in capsys.readouterr().out

    async def test_an_allowed_mixed_graph_runs_and_is_flagged(
        self, reachable_neo4j, fake_neo4j, monkeypatch, demo, tmp_path, capsys
    ):
        fake_neo4j(mixed_demo_handler)
        monkeypatch.setattr(
            rp,
            "make_models",
            lambda settings, evolve: rp.Models(FakeSolver(gold_answers(demo)), FakeGuide()),
        )
        monkeypatch.setattr(rp, "navigator_tools", lambda: fake_tools()[0])
        out = tmp_path / "results.md"
        code = await rp.main(
            [
                "--systems", "no_pg", "--model", "gpt-4o-mini", "--allow-mixed-graph",
                "--out", str(out),
            ]
        )
        assert code == 0
        assert "MIXED GRAPH" in capsys.readouterr().out
        text = out.read_text(encoding="utf-8")
        assert "106 entities, 5 chunks — MIXED" in text
        assert "--allow-mixed-graph" in text  # in the reproduce command too
        assert "EXTRA ENTITIES" in text

    async def test_no_write_leaves_no_file(
        self, reachable_neo4j, fake_neo4j, monkeypatch, demo, tmp_path
    ):
        fake_neo4j(seeded_demo_handler)
        monkeypatch.setattr(
            rp,
            "make_models",
            lambda settings, evolve: rp.Models(FakeSolver(gold_answers(demo)), FakeGuide()),
        )
        monkeypatch.setattr(rp, "navigator_tools", lambda: fake_tools()[0])
        out = tmp_path / "results.md"
        code = await rp.main(
            ["--systems", "no_pg", "--model", "gpt-4o-mini", "--out", str(out), "--no-write"]
        )
        assert code == 0
        assert not out.exists()

    async def test_a_cap_hit_mid_run_exits_5_and_writes_nothing(
        self, reachable_neo4j, fake_neo4j, monkeypatch, demo, tmp_path, capsys
    ):
        fake_neo4j(seeded_demo_handler)
        monkeypatch.setattr(
            rp,
            "make_models",
            lambda settings, evolve: rp.Models(FakeSolver(gold_answers(demo)), FakeGuide()),
        )
        monkeypatch.setattr(rp, "navigator_tools", lambda: fake_tools()[0])
        out = tmp_path / "results.md"
        code = await rp.main(
            [
                "--systems", "no_pg,pg_raw_local", "--max-llm-calls", "20",
                "--model", "gpt-4o-mini", "--out", str(out),
            ]
        )
        assert code == 5
        captured = capsys.readouterr()
        assert "LLM-call cap reached" in captured.err
        assert "INCOMPLETE" in captured.out
        assert not out.exists()

    async def test_an_evolution_refusal_mid_run_exits_5_with_the_paid_rows(
        self, reachable_neo4j, fake_neo4j, monkeypatch, demo, tmp_path, capsys
    ):
        # The backstop behind the pre-flight: with the pre-flight out of the way,
        # evolve() refuses the 42 calls left and main() still reports no_pg as
        # INCOMPLETE and exits 5, instead of a traceback.
        fake_neo4j(seeded_demo_handler)
        MemoryStore().install(monkeypatch)
        monkeypatch.setattr(rp, "evolution_floor", lambda *a, **k: rp.EvolutionFloor(0, 0, 0, 3))
        solver = FakeSolver(gold_answers(demo))
        monkeypatch.setattr(
            rp,
            "make_models",
            lambda settings, evolve: rp.Models(solver, FakeGuide(), FakeRefiner(GOOD_EDITS)),
        )
        monkeypatch.setattr(rp, "navigator_tools", lambda: fake_tools()[0])
        out = tmp_path / "results.md"
        code = await rp.main(
            [
                "--systems", "no_pg", "--evolve", "1", "--evolve-batch-size", "3",
                "--max-steps", "3", "--max-llm-calls", "60", "--model", "gpt-4o-mini",
                "--out", str(out),
            ]
        )
        assert code == 5
        captured = capsys.readouterr()
        assert "evolution refused to start: it needs 64 LLM calls" in captured.err
        assert "INCOMPLETE" in captured.out
        assert "| `no_pg` |" in captured.out  # the paid row is reported, not lost
        assert len(solver.prompts) == 18 and not out.exists()

    def test_the_markdown_states_the_floor_and_the_threats(self, demo, prior):
        result = rp.BenchResult(dataset=demo, prior=prior, max_steps=8)
        text = rp.results_markdown(result)
        assert "one question = 11.1 points" in text
        assert "NOT A REPRODUCTION" in text
        assert "never edited by hand" in text

    def test_the_price_table_is_the_shared_one(self):
        # One ledger implementation and one price table for every harness.
        assert rp.cost is cost
        assert cost.resolve_price("gpt-4o-mini") is not None


class TestReasoningAllowanceOverride:
    """``--reasoning-allowance`` moves only the up-front bound, for reasoning models only."""

    def test_default_is_the_conservative_constant(self):
        assert rp.reasoning_allowance("gpt-5-nano") == rp.REASONING_TOKENS_ALLOWANCE

    def test_a_calibrated_value_replaces_it(self):
        assert rp.reasoning_allowance("gpt-5-nano", override=64) == 64
        assert rp.reasoning_allowance("gpt-5-nano", override=-3) == 0

    def test_non_reasoning_models_never_get_an_allowance(self):
        assert rp.reasoning_allowance("gpt-4o-mini", override=64) == 0

    def test_the_flag_is_parsed(self):
        args = rp.build_parser().parse_args(["--reasoning-allowance", "64", "--dry-run"])
        assert args.reasoning_allowance == 64
