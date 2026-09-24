# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the GraphRAG Navigator: the action parser, the ReAct loop, token
accounting, procedural-guidance injection and the deterministic tools.

Hermetic: the solver LLM is a scripted fake, tools are injected fakes (or the
real tools over monkeypatched retrieval primitives and ``fake_neo4j``), and no
test can reach a model provider.
"""

from __future__ import annotations

import json
import math

import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.services import chat_engine, chunk_store
from app.services import graph_agent as agent
from app.services import procedural_guidance as pgd
from app.services import procedural_store as store
from app.services.graph_agent import INVALID_ACTION_FORMAT, parse_action, run_agent
from app.services.llm_provider import ProviderConfigError
from app.services.procedural_store import ProceduralGraphNotFound, load_prior


@pytest.fixture(autouse=True)
def _fresh_guidance_caches():
    pgd.clear_caches()
    yield
    pgd.clear_caches()


class ScriptedLLM:
    """Replies with the next scripted turn; records every prompt it was sent."""

    def __init__(self, turns, usage=None) -> None:
        self.turns = list(turns)
        self.usage = usage
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        turn = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        if isinstance(turn, Exception):
            raise turn
        if self.usage is None:
            return AIMessage(content=turn)
        return AIMessage(content=turn, usage_metadata=self.usage)


class PlainReply:
    """A provider response with no usage metadata at all."""

    def __init__(self, content: str) -> None:
        self.content = content


def fake_tools(outputs=None):
    """Recording stand-ins for the five retrieval tools."""
    outputs = outputs or {}
    calls: list[tuple[str, dict]] = []

    def make(name):
        async def tool(**kwargs):
            calls.append((name, kwargs))
            value = outputs.get(name, f"{name} result")
            if isinstance(value, Exception):
                raise value
            return value

        return tool

    tools = {
        name: make(name)
        for name in ("search_entities", "neighbors", "read_sources", "search_passages", "find_path")
    }
    return tools, calls


SEARCH = 'Thought: I should find Ada first.\nAction: search_entities(query="Ada Lovelace")'
ANSWER = 'Thought: The evidence names her.\nAction: answer(text="Ada Lovelace")'


# ── Parser ───────────────────────────────────────────
class TestParseAction:
    @pytest.mark.parametrize(
        ("text", "tool", "args"),
        [
            (SEARCH, "search_entities", {"query": "Ada Lovelace"}),
            (
                "Action: search_entities(query=Ada Lovelace)",
                "search_entities",
                {"query": "Ada Lovelace"},
            ),
            ('Action: neighbors("Ada Lovelace")', "neighbors", {"entity": "Ada Lovelace"}),
            ('Action:   search_entities ( query = "Ada" )  ', "search_entities", {"query": "Ada"}),
            ("Action: read_sources(entity='Ada')", "read_sources", {"entity": "Ada"}),
            ('Action: find_path("A", "B")', "find_path", {"source": "A", "target": "B"}),
            (
                'Action: find_path(target="B", source="A")',
                "find_path",
                {"source": "A", "target": "B"},
            ),
            ("Action: answer(Paris, France)", "answer", {"text": "Paris, France"}),
            ("Action: answer(text=Paris, France)", "answer", {"text": "Paris, France"}),
            ('Action: answer(text="1,000")', "answer", {"text": "1,000"}),
            ('Action: answer(text="the \\"Engine\\"")', "answer", {"text": 'the "Engine"'}),
            ('Action: SearchEntities(query="x")', "search_entities", {"query": "x"}),
            (
                '**Thought:** a\n**Action:** `search_entities(query="x")`',
                "search_entities",
                {"query": "x"},
            ),
            ('Action: search_entities(name="Ada")', "search_entities", {"query": "Ada"}),
            ('Action: search_entities(query="Ada", k=5)', "search_entities", {"query": "Ada"}),
        ],
    )
    def test_tolerated_forms(self, text, tool, args):
        parsed = parse_action(text)
        assert parsed.error is None, parsed.error
        assert (parsed.tool, parsed.args) == (tool, args)

    def test_thought_is_extracted_without_its_label(self):
        assert parse_action(SEARCH).thought == "I should find Ada first."
        assert (
            parse_action('I will search.\nAction: search_entities(query="x")').thought
            == "I will search."
        )

    def test_a_simulated_observation_and_later_steps_are_ignored(self):
        text = (
            'Thought: a\nAction: search_entities(query="x")\nObservation: made up\n'
            'Thought: b\nAction: answer(text="y")'
        )
        parsed = parse_action(text)
        assert (parsed.tool, parsed.args) == ("search_entities", {"query": "x"})

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is Paris.",
            "Action: search_entities Ada",
            "Action:",
            "Thought: only a thought",
        ],
    )
    def test_unparseable_turns_are_format_errors(self, text):
        parsed = parse_action(text)
        assert parsed.tool is None
        assert parsed.error == INVALID_ACTION_FORMAT

    def test_unknown_tool(self):
        parsed = parse_action('Action: web_search(query="x")')
        assert parsed.tool is None
        assert parsed.error.startswith("Unknown tool 'web_search'")
        assert "search_entities" in parsed.error

    @pytest.mark.parametrize(
        ("text", "detail"),
        [
            ('Action: find_path(source="A")', "missing target"),
            ('Action: answer(text="")', "missing text"),
            ("Action: answer()", "missing text"),
            ('Action: find_path("A", "B", "C")', "too many arguments"),
        ],
    )
    def test_bad_arguments(self, text, detail):
        parsed = parse_action(text)
        assert parsed.tool is None
        assert detail in parsed.error
        assert "Use: Action: " in parsed.error

    def test_raw_action_is_kept_for_the_trace(self):
        assert parse_action("Action: search_entities Ada").raw_action == "search_entities Ada"


# ── The loop ─────────────────────────────────────────
class TestRunAgent:
    async def test_happy_path_to_an_answer(self):
        tools, calls = fake_tools({"search_entities": "1. Ada Lovelace (PERSON): mathematician"})
        llm = ScriptedLLM([SEARCH, ANSWER])
        result = await run_agent(
            "Who wrote the first program?", guidance="none", llm=llm, tools=tools
        )

        assert result.answer == "Ada Lovelace"
        assert result.stopped == "answer"
        assert result.parse_failures == 0
        assert result.graph is None
        assert calls == [("search_entities", {"query": "Ada Lovelace"})]
        assert [s["action"] for s in result.steps] == ["search_entities", "answer"]
        assert result.steps[0]["observation"] == "1. Ada Lovelace (PERSON): mathematician"
        assert result.steps[0]["thought"] == "I should find Ada first."
        assert result.steps[1]["args"] == {"text": "Ada Lovelace"}
        assert result.usage["llm_calls"] == 2
        assert result.latency_s >= 0
        # The second turn sees the first turn's trajectory, observation included.
        assert 'Action: search_entities(query="Ada Lovelace")' in llm.prompts[1]
        assert "Observation: 1. Ada Lovelace (PERSON): mathematician" in llm.prompts[1]
        assert "Question: Who wrote the first program?" in llm.prompts[0]
        assert "(no steps yet)" in llm.prompts[0]

    async def test_a_parse_failure_is_corrected_and_counted(self):
        tools, _ = fake_tools()
        llm = ScriptedLLM(["I think it is Ada.", ANSWER])
        result = await run_agent("Q?", guidance="none", llm=llm, tools=tools)

        assert result.answer == "Ada Lovelace"
        assert result.parse_failures == 1
        failed = result.steps[0]
        assert failed["action"] is None
        assert failed["observation"] == INVALID_ACTION_FORMAT
        assert failed["raw_action"] == "I think it is Ada."
        assert f"Observation: {INVALID_ACTION_FORMAT}" in llm.prompts[1]

    async def test_max_steps_stops_the_run(self):
        tools, calls = fake_tools()
        llm = ScriptedLLM([SEARCH])
        result = await run_agent("Q?", guidance="none", llm=llm, tools=tools, max_steps=3)
        assert result.stopped == "max_steps"
        assert result.answer is None
        assert len(result.steps) == 3 and len(calls) == 3
        assert result.usage["llm_calls"] == 3

    async def test_the_default_step_cap_comes_from_settings(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "agent_max_steps", 2)
        tools, _ = fake_tools()
        result = await run_agent("Q?", guidance="none", llm=ScriptedLLM([SEARCH]), tools=tools)
        assert len(result.steps) == 2

    async def test_observations_are_truncated(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "agent_observation_max_chars", 100)
        tools, _ = fake_tools({"search_entities": "x" * 5000})
        result = await run_agent(
            "Q?", guidance="none", llm=ScriptedLLM([SEARCH, ANSWER]), tools=tools
        )
        observation = result.steps[0]["observation"]
        assert len(observation) == 100
        assert observation.endswith("…[observation truncated]")

    async def test_usage_comes_from_usage_metadata(self):
        tools, _ = fake_tools()
        usage = {"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}
        result = await run_agent(
            "Q?", guidance="none", llm=ScriptedLLM([SEARCH, ANSWER], usage=usage), tools=tools
        )
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 14
        assert result.usage["estimated"] is False

    async def test_usage_is_estimated_when_the_provider_reports_none(self):
        tools, _ = fake_tools()

        class Plain(ScriptedLLM):
            async def ainvoke(self, messages):
                self.prompts.append(messages[0].content)
                return PlainReply(ANSWER)

        llm = Plain([ANSWER])
        result = await run_agent("Q?", guidance="none", llm=llm, tools=tools)
        assert result.usage["estimated"] is True
        assert result.usage["input_tokens"] == math.ceil(len(llm.prompts[0]) / 4)
        assert result.usage["output_tokens"] == math.ceil(len(ANSWER) / 4)

    async def test_a_tool_failure_becomes_an_observation(self):
        tools, _ = fake_tools({"search_entities": RuntimeError("neo4j down")})
        result = await run_agent(
            "Q?", guidance="none", llm=ScriptedLLM([SEARCH, ANSWER]), tools=tools
        )
        assert (
            result.steps[0]["observation"]
            == "Tool error (search_entities): RuntimeError: neo4j down"
        )
        assert result.stopped == "answer"

    async def test_an_llm_failure_ends_the_run_with_an_error(self):
        tools, _ = fake_tools()
        result = await run_agent(
            "Q?", guidance="none", llm=ScriptedLLM([SEARCH, TimeoutError("slow")]), tools=tools
        )
        assert result.stopped == "error"
        assert result.error == "TimeoutError: slow"
        assert len(result.steps) == 1

    async def test_a_missing_provider_is_raised_not_swallowed(self, monkeypatch):
        def no_key(**kwargs):
            raise ProviderConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

        monkeypatch.setattr(agent, "get_chat_llm", no_key)
        with pytest.raises(ProviderConfigError):
            await run_agent("Q?", guidance="none", tools=fake_tools()[0])

    async def test_the_default_solver_model_uses_the_agent_temperature(self, monkeypatch):
        seen = {}
        llm = ScriptedLLM([ANSWER])

        def fake_get_chat_llm(**kwargs):
            seen.update(kwargs)
            return llm

        monkeypatch.setattr(agent, "get_chat_llm", fake_get_chat_llm)
        await run_agent("Q?", guidance="none", tools=fake_tools()[0])
        assert seen == {"temperature": get_settings().agent_temperature}

    async def test_injected_tools_replace_the_registry(self):
        tools = {"search_entities": fake_tools()[0]["search_entities"]}
        llm = ScriptedLLM(['Action: neighbors(entity="Ada")', ANSWER])
        result = await run_agent("Q?", guidance="none", llm=llm, tools=tools)
        assert "- neighbors(" not in llm.prompts[0]
        assert '- search_entities(query="...")' in llm.prompts[0]
        assert '- answer(text="...")' in llm.prompts[0]
        assert result.parse_failures == 1
        assert result.steps[0]["observation"].startswith("Unknown tool 'neighbors'")

    async def test_the_result_is_json_serializable(self):
        tools, _ = fake_tools()
        result = await run_agent(
            "Q?", guidance="none", llm=ScriptedLLM([SEARCH, ANSWER]), tools=tools
        )
        payload = json.loads(json.dumps(result.to_dict()))
        assert set(payload) >= {
            "answer",
            "steps",
            "stopped",
            "parse_failures",
            "usage",
            "graph",
            "latency_s",
        }
        assert set(payload["usage"]) == {
            "llm_calls",
            "guidance_llm_calls",
            "input_tokens",
            "output_tokens",
            "estimated",
            "context_chars",
        }
        assert set(payload["steps"][0]) >= {
            "thought",
            "action",
            "args",
            "observation",
            "guidance_context_chars",
            "localization",
        }


# ── Procedural guidance inside the loop ──────────────
class TestGuidanceInjection:
    async def test_raw_guidance_is_injected_and_follows_the_last_action(self):
        prior = load_prior("graphrag-navigator")
        tools, _ = fake_tools()
        llm = ScriptedLLM([SEARCH, ANSWER])
        result = await run_agent("Q?", guidance="raw", graph=prior, llm=llm, tools=tools)

        assert "Procedural Graph Guidance:" in llm.prompts[0]
        assert "Active Cognitive Node: [Start] (Type: STATUS)" in llm.prompts[0]
        assert "Active Cognitive Node: [search_entities] (Type: ACTION)" in llm.prompts[1]
        assert [s["localization"] for s in result.steps] == ["start", "exact"]
        assert [s["active_node"] for s in result.steps] == ["Start", "search_entities"]
        assert result.graph == {"name": "graphrag-navigator", "version": None}
        chars = [s["guidance_context_chars"] for s in result.steps]
        assert all(c > 0 for c in chars)
        assert result.usage["context_chars"] == sum(chars)
        assert result.usage["guidance_llm_calls"] == 0
        assert result.usage["llm_calls"] == 2

    async def test_no_guidance_means_no_guidance_block(self):
        prior = load_prior("graphrag-navigator")
        llm = ScriptedLLM([ANSWER])
        result = await run_agent("Q?", guidance="none", graph=prior, llm=llm, tools=fake_tools()[0])
        assert "Procedural Graph Guidance" not in llm.prompts[0]
        assert result.graph is None
        assert result.steps[0]["localization"] is None

    async def test_generative_guidance_is_injected_and_counted(self):
        prior = load_prior("graphrag-navigator")
        guidance_llm = ScriptedLLM(
            ["Next: call search_entities with the name."],
            usage={"input_tokens": 300, "output_tokens": 20, "total_tokens": 320},
        )
        llm = ScriptedLLM(
            [SEARCH, ANSWER], usage={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        )
        result = await run_agent(
            "Q?",
            guidance="generative",
            graph=prior,
            llm=llm,
            tools=fake_tools()[0],
            guidance_llm=guidance_llm,
        )
        assert (
            "Procedural Graph Guidance:\nNext: call search_entities with the name."
            in llm.prompts[0]
        )
        assert (
            "Active Cognitive Node" not in llm.prompts[0]
        )  # the context went to the guidance model
        assert "Active Cognitive Node: [Start]" in guidance_llm.prompts[0]
        assert result.usage["guidance_llm_calls"] == 2
        assert result.usage["llm_calls"] == 4
        assert result.usage["input_tokens"] == 2 * 300 + 2 * 100

    async def test_after_a_parse_failure_the_full_graph_is_shown(self, monkeypatch):
        # The failed step's observation is the parser's error, which lists every
        # tool. It must not be embedded and matched to a node the agent never
        # reached: the paper's Match finds nothing and uses the full graph.
        class AnythingMatches:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                self.queries.append(text)
                return [1.0, 0.0]

        embedder = AnythingMatches()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        prior = load_prior("graphrag-navigator")
        llm = ScriptedLLM([SEARCH, 'Thought: try another.\nAction: lookup(query="Ada")', ANSWER])
        result = await run_agent("Q?", guidance="raw", graph=prior, llm=llm, tools=fake_tools()[0])

        assert result.steps[1]["action"] is None and "Unknown tool" in result.steps[1]["observation"]
        assert [s["localization"] for s in result.steps] == ["start", "exact", "none"]
        assert "Procedural Graph: [graphrag-navigator]" in llm.prompts[2]
        assert embedder.queries == []

    async def test_full_graph_scope_can_be_forced(self):
        prior = load_prior("graphrag-navigator")
        llm = ScriptedLLM([ANSWER])
        await run_agent(
            "Q?", guidance="raw", graph=prior, llm=llm, tools=fake_tools()[0], full_graph=True
        )
        assert "Procedural Graph: [graphrag-navigator]" in llm.prompts[0]
        assert "Active Cognitive Node" not in llm.prompts[0]

    async def test_a_named_graph_is_loaded_once_per_run(self, monkeypatch):
        reads = []

        async def fake_load(name):
            reads.append(name)
            return load_prior("graphrag-navigator"), {"version": 4, "score": None}

        monkeypatch.setattr(store, "load_graph_with_meta", fake_load)
        result = await run_agent(
            "Q?",
            graph_name="graphrag-navigator",
            guidance="raw",
            llm=ScriptedLLM([SEARCH, ANSWER]),
            tools=fake_tools()[0],
        )
        assert reads == ["graphrag-navigator"]
        assert result.graph == {"name": "graphrag-navigator", "version": 4}

    async def test_an_unknown_graph_raises(self, monkeypatch):
        async def absent(name):
            return None

        monkeypatch.setattr(store, "load_graph_with_meta", absent)
        with pytest.raises(ProceduralGraphNotFound):
            await run_agent(
                "Q?",
                graph_name="ghost",
                guidance="raw",
                llm=ScriptedLLM([ANSWER]),
                tools=fake_tools()[0],
            )

    async def test_a_guidance_failure_does_not_end_the_run(self, monkeypatch):
        async def broken_guide(*args, **kwargs):
            raise RuntimeError("guidance model timed out")

        monkeypatch.setattr(agent, "guide", broken_guide)
        prior = load_prior("graphrag-navigator")
        llm = ScriptedLLM([ANSWER])
        result = await run_agent(
            "Q?", guidance="generative", graph=prior, llm=llm, tools=fake_tools()[0]
        )
        assert result.stopped == "answer"
        assert result.steps[0]["guidance_error"] == "RuntimeError: guidance model timed out"
        assert "Procedural Graph Guidance" not in llm.prompts[0]


# ── Tool catalog ─────────────────────────────────────
class TestToolCatalog:
    def test_tool_names_are_the_priors_action_nodes(self):
        prior = load_prior("graphrag-navigator")
        assert agent.TOOL_NAMES == prior.tools
        actions = {node.id for node in prior.nodes.values() if node.type == "ACTION"}
        assert actions == set(agent.TOOL_NAMES)

    def test_every_non_terminal_tool_has_an_implementation(self):
        assert set(agent.DEFAULT_TOOLS) == set(agent.TOOL_NAMES) - {"answer"}


# ── The deterministic tools ──────────────────────────
ADA = {
    "name": "Ada Lovelace",
    "type": "PERSON",
    "description": "Wrote the first published algorithm.",
}
ENGINE = {"name": "Analytical Engine", "type": "MACHINE", "description": "Babbage's design."}


def _resolver(entities):
    by_lower = {e["name"].lower(): e for e in entities}

    def handler(query, params):
        if query == agent.RESOLVE_ENTITY_QUERY:
            found = by_lower.get(params["name"].lower())
            return [found] if found else []
        raise AssertionError(f"unexpected query: {query}")

    return handler


class TestTools:
    async def test_fake_neo4j_covers_the_agent_module(self, fake_neo4j):
        real = agent.execute_query
        calls = fake_neo4j(lambda q, p: [{"ok": 1}])
        assert agent.execute_query is not real
        assert await agent.execute_query("RETURN 1") == [{"ok": 1}]
        assert calls == [("RETURN 1", {})]

    async def test_search_entities_interleaves_and_dedupes(self, monkeypatch):
        async def vector(query, k):
            return [ADA, {"name": "Charles Babbage", "type": "PERSON", "description": "x" * 400}]

        async def keyword(query, k):
            return [ADA, ENGINE]

        monkeypatch.setattr(chat_engine, "_seeds_by_vector", vector)
        monkeypatch.setattr(chat_engine, "_seeds_by_keyword", keyword)
        text = await agent.search_entities('"Ada"')
        lines = text.splitlines()
        assert lines[0] == 'Entities matching "Ada":'
        assert lines[1] == "1. Ada Lovelace (PERSON): Wrote the first published algorithm."
        assert lines[2].startswith("2. Charles Babbage (PERSON): ")
        assert len(lines[2]) < 260  # description clipped
        assert lines[3].startswith("3. Analytical Engine (MACHINE)")

    async def test_search_entities_with_no_match(self, monkeypatch):
        async def nothing(query, k):
            return []

        monkeypatch.setattr(chat_engine, "_seeds_by_vector", nothing)
        monkeypatch.setattr(chat_engine, "_seeds_by_keyword", nothing)
        assert (await agent.search_entities("zzz")).startswith('No entities match "zzz"')

    async def test_neighbors_renders_directed_relations(self, monkeypatch, fake_neo4j):
        fake_neo4j(_resolver([ADA]))
        seen = {}

        async def edges(names, hops):
            seen.update(names=names, hops=hops)
            return [
                {
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine",
                    "rel": "WROTE_PROGRAM_FOR",
                },
                {"source": "Charles Babbage", "target": "Ada Lovelace", "rel": "CORRESPONDED_WITH"},
            ]

        monkeypatch.setattr(chat_engine, "_neighborhood_edges", edges)
        text = await agent.neighbors("ada lovelace")
        assert seen == {"names": ["Ada Lovelace"], "hops": 1}
        assert text.splitlines()[0] == "Entity: Ada Lovelace (PERSON)"
        assert "- Ada Lovelace -[WROTE_PROGRAM_FOR]-> Analytical Engine" in text
        assert "- Charles Babbage -[CORRESPONDED_WITH]-> Ada Lovelace" in text

    async def test_an_unknown_entity_points_back_to_search(self, monkeypatch, fake_neo4j):
        fake_neo4j(_resolver([]))

        async def never(*args):  # pragma: no cover - must not run
            raise AssertionError("no expansion for an unknown entity")

        monkeypatch.setattr(chat_engine, "_neighborhood_edges", never)
        text = await agent.neighbors("Nobody")
        assert text.startswith('No entity named "Nobody"')
        assert "search_entities" in text

    async def test_read_sources_lists_the_linked_passages(self, monkeypatch, fake_neo4j):
        fake_neo4j(_resolver([ADA]))
        seen = {}

        async def chunks(names, limit):
            seen.update(names=names, limit=limit)
            return [{"text": "Ada wrote notes. " * 60, "document": "ai.pdf", "index": 2}]

        monkeypatch.setattr(chunk_store, "chunks_for_entities", chunks)
        text = await agent.read_sources("Ada Lovelace")
        assert seen == {"names": ["Ada Lovelace"], "limit": agent.READ_SOURCES_K}
        assert text.startswith("Sources for Ada Lovelace:\n[1] ai.pdf (chunk 2): Ada wrote notes.")
        assert len(text.splitlines()[1]) <= agent.PASSAGE_CHARS + 30

    async def test_search_passages(self, monkeypatch):
        async def search(query, k):
            assert (query, k) == ("first algorithm", agent.PASSAGES_K)
            return [{"text": "The first algorithm...", "document": "ai.pdf", "index": 0}]

        monkeypatch.setattr(chunk_store, "search_chunks", search)
        text = await agent.search_passages("first algorithm")
        assert (
            text
            == 'Passages matching "first algorithm":\n[1] ai.pdf (chunk 0): The first algorithm...'
        )

    async def test_find_path_uses_the_chat_engines_reasoning_paths(self, monkeypatch, fake_neo4j):
        fake_neo4j(_resolver([ADA, ENGINE]))

        async def edges(names, hops):
            assert names == ["Ada Lovelace", "Analytical Engine"]
            return [
                {
                    "source": "Ada Lovelace",
                    "target": "Analytical Engine",
                    "rel": "WROTE_PROGRAM_FOR",
                }
            ]

        monkeypatch.setattr(chat_engine, "_neighborhood_edges", edges)
        text = await agent.find_path("Ada Lovelace", "analytical engine")
        assert text == (
            "Paths between Ada Lovelace and Analytical Engine:\n"
            "- Ada Lovelace -[WROTE_PROGRAM_FOR]-> Analytical Engine"
        )

    async def test_find_path_says_when_there_is_none(self, monkeypatch, fake_neo4j):
        fake_neo4j(_resolver([ADA, ENGINE]))

        async def no_edges(names, hops):
            return []

        monkeypatch.setattr(chat_engine, "_neighborhood_edges", no_edges)
        text = await agent.find_path("Ada Lovelace", "Analytical Engine")
        assert text.startswith("No path found between Ada Lovelace and Analytical Engine")
        assert "does not mean they are unrelated" in text


class TestDefaultToolWiring:
    async def test_the_default_registry_runs_over_the_real_retrieval_primitives(
        self, monkeypatch, fake_neo4j
    ):
        """No injected tools: the loop drives the real tools over ``fake_neo4j``."""

        async def no_vector_seeds(query, k):  # keeps the test off any embedding backend
            return []

        monkeypatch.setattr(chat_engine, "_seeds_by_vector", no_vector_seeds)

        def handler(query, params):
            if query == agent.RESOLVE_ENTITY_QUERY:
                return [ADA]
            if "db.index.fulltext.queryNodes" in query:
                return [{**ADA, "score": 3.2}]
            if query == chat_engine.NEIGHBOR_EDGES_QUERY:
                return [
                    {
                        "source": "Ada Lovelace",
                        "target": "Analytical Engine",
                        "rel": "WROTE_PROGRAM_FOR",
                    }
                ]
            raise AssertionError(f"unexpected query: {query}")

        calls = fake_neo4j(handler)
        llm = ScriptedLLM(
            [
                'Thought: find her\nAction: search_entities(query="Ada Lovelace")',
                'Thought: relations\nAction: neighbors(entity="Ada Lovelace")',
                'Thought: done\nAction: answer(text="Analytical Engine")',
            ]
        )
        result = await run_agent("What did Ada program?", guidance="none", llm=llm)

        assert result.answer == "Analytical Engine"
        first, second, _ = result.steps
        assert first["observation"].startswith('Entities matching "Ada Lovelace":\n1. Ada Lovelace')
        assert "- Ada Lovelace -[WROTE_PROGRAM_FOR]-> Analytical Engine" in second["observation"]
        assert calls  # every read went through the patched driver
