# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for online procedural guidance: the localization cascade, the three
modes, the guidance cache and its token accounting.

Hermetic: the graph is passed in memory (or the store read is monkeypatched),
embeddings are a tiny keyword embedder, and the guidance LLM is a scripted fake.
Nothing here can reach a database or a model provider.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.services import procedural_guidance as pgd
from app.services import procedural_store as store
from app.services.llm_provider import ProviderConfigError
from app.services.procedural_graph import (
    ProceduralGraph,
    serialize_full,
    serialize_local,
    skeleton,
)
from app.services.procedural_store import ProceduralGraphNotFound


@pytest.fixture(autouse=True)
def _fresh_caches():
    pgd.clear_caches()
    yield
    pgd.clear_caches()


def _graph(name: str = "g") -> ProceduralGraph:
    return ProceduralGraph.from_dict(
        {
            "name": name,
            "description": "Answer questions from documents.",
            "tools": ["search", "read", "answer"],
            "nodes": [
                {"id": "Start", "type": "STATUS", "description": "Nothing done yet."},
                {"id": "search", "type": "ACTION", "description": "Search for the entity."},
                {"id": "read", "type": "ACTION", "description": "Read the passage text."},
                {"id": "Check", "type": "REASONING", "description": "Verify the evidence."},
                {"id": "answer", "type": "ACTION", "description": "Submit the reply."},
                {"id": "End", "type": "STATUS", "description": "Done."},
            ],
            "edges": [
                {
                    "source": "Start",
                    "target": "search",
                    "guidance": "Search first.",
                    "pitfalls": "Not the whole question.",
                },
                {
                    "source": "search",
                    "target": "read",
                    "relation": "PROVIDES_INPUT_FOR",
                    "condition": "When an entity matched",
                    "guidance": "Read its passages.",
                },
                {"source": "read", "target": "Check", "guidance": "Check the evidence."},
                {"source": "Check", "target": "answer", "guidance": "Answer briefly."},
                {"source": "answer", "target": "End", "guidance": "Stop."},
            ],
        }
    )


class KeywordEmbedder:
    """Counts a few keywords: cosine similarity is then easy to reason about."""

    VOCAB = ("passage", "entity", "evidence", "reply", "done")

    def __init__(self) -> None:
        self.documents_calls = 0
        self.query_calls = 0

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in self.VOCAB]

    def embed_documents(self, texts):
        self.documents_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply="Call read next; do not guess.", usage=None) -> None:
        self.reply = reply
        self.usage = usage
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        if self.usage is None:
            return AIMessage(content=self.reply)
        return AIMessage(content=self.reply, usage_metadata=self.usage)


class ExplodingLLM:
    async def ainvoke(self, messages):  # pragma: no cover - must never run
        raise AssertionError("the guidance LLM must not be called")


def _no_embeddings(monkeypatch):
    def boom():
        raise AssertionError("no embedding call expected")

    monkeypatch.setattr(pgd, "get_embeddings", boom)


USAGE = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}


# ── Localization cascade ─────────────────────────────
class TestLocalization:
    async def test_empty_trajectory_starts_at_start(self, monkeypatch):
        _no_embeddings(monkeypatch)
        graph = _graph()
        result = await pgd.guide("g", query="Q?", trajectory=[], graph=graph, version=3)
        assert result["active_node"] == "Start"
        assert result["localization"] == "start"
        assert result["scope"] == "local"
        assert result["version"] == 3 and result["graph"] == "g"
        assert result["context"] == serialize_local(graph, "Start", 2)
        assert result["next_actions"] == ["search"]

    async def test_exact_match_on_the_action_name(self, monkeypatch):
        _no_embeddings(monkeypatch)
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[{"action": "search", "observation": "3 hits"}],
            graph=_graph(),
        )
        assert (result["active_node"], result["localization"]) == ("search", "exact")
        assert result["next_actions"] == ["read"]
        assert result["context"].startswith("Active Cognitive Node: [search] (Type: ACTION)")

    async def test_normalized_match_strips_arguments_and_case(self, monkeypatch):
        _no_embeddings(monkeypatch)
        result = await pgd.guide(
            "g", query="Q?", trajectory=[{"action": 'Search(query="Ada")'}], graph=_graph()
        )
        assert (result["active_node"], result["localization"]) == ("search", "normalized")

    async def test_semantic_fallback_matches_the_closest_description(self, monkeypatch):
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[{"action": "open_document", "observation": "The passage says it opened."}],
            graph=_graph(),
        )
        assert (result["active_node"], result["localization"]) == ("read", "semantic")
        assert result["localization_score"] >= get_settings().procedural_semantic_threshold
        assert result["scope"] == "local"

    async def test_semantic_below_threshold_falls_back_to_the_full_graph(self, monkeypatch):
        monkeypatch.setattr(pgd, "get_embeddings", lambda: KeywordEmbedder())
        monkeypatch.setattr(get_settings(), "procedural_semantic_threshold", 0.999)
        graph = _graph()
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[{"action": "open_document", "observation": "passage and entity"}],
            graph=graph,
        )
        assert result["active_node"] is None
        assert result["localization"] == "none"
        assert result["scope"] == "full"
        assert result["context"] == serialize_full(graph)
        assert result["next_actions"] == []

    async def test_an_embedding_failure_is_tolerated(self, monkeypatch):
        def broken():
            raise RuntimeError("embedding backend down")

        monkeypatch.setattr(pgd, "get_embeddings", broken)
        result = await pgd.guide(
            "g", query="Q?", trajectory=[{"action": "mystery", "observation": "x"}], graph=_graph()
        )
        assert result["localization"] == "none"
        assert result["scope"] == "full"

    async def test_node_embeddings_are_computed_once_per_graph_version(self, monkeypatch):
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        graph = _graph()
        for observation in ("a passage", "another passage"):
            await pgd.guide(
                "g",
                query="Q?",
                trajectory=[{"action": "open_document", "observation": observation}],
                graph=graph,
                version=1,
            )
        assert embedder.documents_calls == 1
        assert embedder.query_calls == 2

    async def test_changed_graph_content_is_re_embedded_even_without_a_version(self, monkeypatch):
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        step = [{"action": "open_document", "observation": "a passage"}]
        await pgd.guide("g", query="Q?", trajectory=step, graph=_graph())
        changed = _graph()
        changed.nodes["read"] = type(changed.nodes["read"])("read", "ACTION", "Read the evidence.")
        await pgd.guide("g", query="Q?", trajectory=step, graph=changed)
        assert embedder.documents_calls == 2

    async def test_a_failed_step_is_not_mistaken_for_the_start(self, monkeypatch):
        monkeypatch.setattr(pgd, "get_embeddings", lambda: KeywordEmbedder())
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[{"action": "", "observation": "Invalid action format."}],
            graph=_graph(),
        )
        assert result["localization"] == "none"

    @pytest.mark.parametrize("action", [None, "", "   "])
    async def test_a_parse_failure_uses_the_full_graph_without_embedding_the_error(
        self, monkeypatch, action
    ):
        # The Navigator records a failed parse with action None and its own error
        # text as the observation. That text names every tool ("read the passage"
        # here would match `read`), yet the agent never reached any of them: the
        # paper's Match finds nothing and uses the full graph, and so must we.
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        graph = _graph()
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[
                {"action": "search", "observation": "1 entity"},
                {
                    "action": action,
                    "observation": "Unknown tool 'lookup'. Available tools: search, read "
                    "the passage, answer.",
                },
            ],
            graph=graph,
        )
        assert (result["active_node"], result["localization"]) == (None, "none")
        assert result["scope"] == "full"
        assert result["context"] == serialize_full(graph)
        assert embedder.query_calls == 0  # the error text is never embedded

    async def test_semantic_never_lands_on_a_terminal(self, monkeypatch):
        # "done" matches End's description exactly, but End is terminal: its
        # context would tell the solver the procedure is over.
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        result = await pgd.guide(
            "g", query="Q?", trajectory=[{"action": "wrap_up", "observation": "done"}],
            graph=_graph(),
        )
        assert result["localization"] == "none" and result["scope"] == "full"
        assert "End" not in pgd._semantic_candidates(_graph())
        assert "Start" not in pgd._semantic_candidates(_graph())

    async def test_a_scratch_skeleton_has_no_semantic_candidates(self, monkeypatch):
        # Start -> End: End is the only non-Start node and it is terminal, so any
        # semantic match would have put the agent on "the procedure ends here".
        embedder = KeywordEmbedder()
        monkeypatch.setattr(pgd, "get_embeddings", lambda: embedder)
        graph = skeleton("scratchy", tools=("search_entities", "answer"))
        graph.nodes["End"] = type(graph.nodes["End"])("End", "STATUS", "Done.")
        result = await pgd.guide(
            "scratchy",
            query="Q?",
            trajectory=[{"action": "search_entities", "observation": "1. Ada Lovelace. done"}],
            graph=graph,
        )
        assert (result["active_node"], result["localization"]) == (None, "none")
        assert result["scope"] == "full"
        assert "procedure ends here" not in result["context"]
        assert embedder.documents_calls == embedder.query_calls == 0  # nothing to compare

    async def test_full_graph_scope_can_be_forced(self, monkeypatch):
        _no_embeddings(monkeypatch)
        graph = _graph()
        result = await pgd.guide(
            "g", query="Q?", trajectory=[{"action": "search"}], graph=graph, full_graph=True
        )
        assert result["scope"] == "full"
        assert result["active_node"] == "search"  # still reported
        assert result["context"] == serialize_full(graph)

    async def test_hops_bound_the_horizon(self, monkeypatch):
        _no_embeddings(monkeypatch)
        one = await pgd.guide("g", query="Q?", trajectory=[], graph=_graph(), hops=1)
        two = await pgd.guide("g", query="Q?", trajectory=[], graph=_graph(), hops=2)
        assert "Subsequent Horizon" not in one["context"]
        assert "Subsequent Horizon (Hop 2):" in two["context"]


# ── Modes ────────────────────────────────────────────
class TestModes:
    async def test_none_returns_no_context_and_calls_nothing(self, monkeypatch):
        _no_embeddings(monkeypatch)
        result = await pgd.guide(
            "g",
            query="Q?",
            trajectory=[{"action": "unknown_tool", "observation": "passage"}],
            graph=_graph(),
            mode="none",
            llm=ExplodingLLM(),
        )
        assert result["context"] == ""
        assert result["guidance"] is None
        assert result["localization"] == "none"
        assert result["usage"]["llm_calls"] == 0

    async def test_raw_is_the_serialized_subgraph_and_zero_llm_calls(self, monkeypatch):
        _no_embeddings(monkeypatch)
        result = await pgd.guide(
            "g", query="Q?", trajectory=[], graph=_graph(), mode="raw", llm=ExplodingLLM()
        )
        assert result["guidance"] is None
        assert result["context"].startswith("Active Cognitive Node: [Start]")
        usage = result["usage"]
        assert usage["llm_calls"] == 0 and usage["cached"] is False
        assert usage["context_chars"] == len(result["context"])
        assert usage["context_tokens_est"] == math.ceil(len(result["context"]) / 4)

    async def test_the_default_mode_comes_from_settings(self, monkeypatch):
        _no_embeddings(monkeypatch)
        assert get_settings().procedural_guidance_mode == "raw"
        result = await pgd.guide("g", query="Q?", trajectory=[], graph=_graph())
        assert result["guidance"] is None and result["context"]

    async def test_generative_makes_one_call_with_the_adapted_prompt(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM(usage=USAGE)
        trajectory = [
            {
                "thought": "look it up",
                "action": "search",
                "args": {"query": "Ada"},
                "observation": "Ada Lovelace (PERSON)",
            }
        ]
        result = await pgd.guide(
            "g",
            query="Who was Ada?",
            trajectory=trajectory,
            graph=_graph(),
            mode="generative",
            llm=llm,
        )
        assert result["guidance"] == "Call read next; do not guess."
        assert result["usage"]["llm_calls"] == 1
        assert result["usage"]["input_tokens"] == 120 and result["usage"]["output_tokens"] == 30
        assert result["usage"]["estimated"] is False
        (prompt,) = llm.prompts
        assert "solving the task: Answer questions from documents." in prompt
        assert result["context"] in prompt
        assert "Here is the current active query / observation:\nWho was Ada?" in prompt
        assert 'Action: search(query="Ada")' in prompt
        assert "Observation: Ada Lovelace (PERSON)" in prompt
        assert "condition, guidance and pitfalls" in prompt
        assert "localized subgraph" in prompt

    async def test_generative_full_scope_uses_the_full_graph_wording(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM()
        await pgd.guide(
            "g",
            query="Q?",
            trajectory=[],
            graph=_graph(),
            mode="generative",
            llm=llm,
            full_graph=True,
        )
        assert "the complete Procedural Graph governing the task structure" in llm.prompts[0]

    async def test_missing_usage_metadata_is_estimated_and_flagged(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM(reply="x" * 40)
        result = await pgd.guide(
            "g", query="Q?", trajectory=[], graph=_graph(), mode="generative", llm=llm
        )
        assert result["usage"]["estimated"] is True
        assert result["usage"]["input_tokens"] == math.ceil(len(llm.prompts[0]) / 4)
        assert result["usage"]["output_tokens"] == 10

    async def test_only_the_last_w_steps_reach_the_guidance_model(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM()
        steps = [{"action": "search", "observation": f"obs {i}"} for i in range(1, 6)]
        await pgd.guide(
            "g", query="Q?", trajectory=steps, graph=_graph(), mode="generative", llm=llm, window=2
        )
        prompt = llm.prompts[0]
        assert "Step 4:" in prompt and "Step 5:" in prompt
        assert "Step 3:" not in prompt and "obs 3" not in prompt

    async def test_default_guidance_model_is_temperature_zero(self, monkeypatch):
        _no_embeddings(monkeypatch)
        seen = {}
        llm = FakeLLM()

        def fake_get_chat_llm(**kwargs):
            seen.update(kwargs)
            return llm

        monkeypatch.setattr(pgd, "get_chat_llm", fake_get_chat_llm)
        await pgd.guide("g", query="Q?", trajectory=[], graph=_graph(), mode="generative")
        assert seen == {"temperature": 0}
        assert len(llm.prompts) == 1

    async def test_a_missing_provider_surfaces_as_provider_config_error(self, monkeypatch):
        _no_embeddings(monkeypatch)

        def no_key(**kwargs):
            raise ProviderConfigError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

        monkeypatch.setattr(pgd, "get_chat_llm", no_key)
        with pytest.raises(ProviderConfigError):
            await pgd.guide("g", query="Q?", trajectory=[], graph=_graph(), mode="generative")

    async def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            await pgd.guide("g", query="Q?", trajectory=[], graph=_graph(), mode="loud")


# ── Cache ────────────────────────────────────────────
class TestGuidanceCache:
    async def test_the_same_situation_is_answered_from_the_cache(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM(usage=USAGE)
        kwargs = {
            "query": "Q?",
            "trajectory": [{"action": "search"}],
            "graph": _graph(),
            "version": 2,
            "mode": "generative",
            "llm": llm,
        }
        first = await pgd.guide("g", **kwargs)
        second = await pgd.guide("g", **kwargs)
        assert len(llm.prompts) == 1
        assert second["guidance"] == first["guidance"]
        assert second["usage"]["cached"] is True
        assert second["usage"]["llm_calls"] == 0
        assert second["usage"]["input_tokens"] == 0

    async def test_a_different_query_or_window_misses(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM()
        base = {"graph": _graph(), "version": 2, "mode": "generative", "llm": llm}
        await pgd.guide("g", query="Q1?", trajectory=[{"action": "search"}], **base)
        await pgd.guide("g", query="Q2?", trajectory=[{"action": "search"}], **base)
        await pgd.guide(
            "g", query="Q1?", trajectory=[{"action": "search", "observation": "new"}], **base
        )
        assert len(llm.prompts) == 3

    async def test_a_new_version_misses(self, monkeypatch):
        _no_embeddings(monkeypatch)
        llm = FakeLLM()
        for version in (1, 2):
            await pgd.guide(
                "g",
                query="Q?",
                trajectory=[],
                graph=_graph(),
                version=version,
                mode="generative",
                llm=llm,
            )
        assert len(llm.prompts) == 2

    async def test_size_zero_disables_the_cache(self, monkeypatch):
        _no_embeddings(monkeypatch)
        monkeypatch.setattr(get_settings(), "procedural_guidance_cache_size", 0)
        llm = FakeLLM()
        for _ in range(2):
            await pgd.guide(
                "g", query="Q?", trajectory=[], graph=_graph(), mode="generative", llm=llm
            )
        assert len(llm.prompts) == 2

    async def test_the_cache_is_a_bounded_lru(self, monkeypatch):
        _no_embeddings(monkeypatch)
        monkeypatch.setattr(get_settings(), "procedural_guidance_cache_size", 2)
        llm = FakeLLM()
        for query in ("a", "b", "c", "a"):
            await pgd.guide(
                "g", query=query, trajectory=[], graph=_graph(), mode="generative", llm=llm
            )
        assert len(llm.prompts) == 4  # "a" was evicted by "c"
        assert len(pgd._guidance_cache) == 2


# ── Loading from the store ───────────────────────────
class TestStoreBackedGuidance:
    async def test_loads_the_graph_and_its_version(self, monkeypatch):
        _no_embeddings(monkeypatch)
        reads = []

        async def fake_load(name):
            reads.append(name)
            return _graph(name), {"version": 7, "score": 0.5}

        monkeypatch.setattr(store, "load_graph_with_meta", fake_load)
        result = await pgd.guide("nav", query="Q?", trajectory=[])
        assert reads == ["nav"]
        assert result["graph"] == "nav" and result["version"] == 7

    async def test_an_unknown_graph_raises_not_found(self, monkeypatch):
        async def absent(name):
            return None

        monkeypatch.setattr(store, "load_graph_with_meta", absent)
        with pytest.raises(ProceduralGraphNotFound):
            await pgd.guide("ghost", query="Q?", trajectory=[])


# ── Helpers ──────────────────────────────────────────
class TestHelpers:
    def test_llm_usage_reads_the_raw_provider_block(self):
        class Response:
            content = "hi"
            usage_metadata = {"input_tokens": 0, "output_tokens": 0}
            response_metadata = {"token_usage": {"prompt_tokens": 11, "completion_tokens": 4}}

        assert pgd.llm_usage(Response(), "prompt") == {
            "input_tokens": 11,
            "output_tokens": 4,
            "estimated": False,
        }

    def test_response_text_joins_content_blocks(self):
        response = AIMessage(content=[{"type": "text", "text": "a"}, "b"])
        assert pgd.response_text(response) == "ab"

    def test_format_window_numbers_steps_by_run_position(self):
        steps = [
            {"action": "search", "observation": "x" * 2000},
            {"action": "read", "args": {"entity": 'A "B"'}},
        ]
        text = pgd.format_window(steps, 5)
        assert text.startswith("Step 1:\nAction: search")
        assert 'Action: read(entity="A \\"B\\"")' in text
        assert len(text) < 2000  # observations are clipped
        assert pgd.format_window(steps, 0) == "(no steps taken yet)"
