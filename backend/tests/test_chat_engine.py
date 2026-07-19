"""Tests for GraphRAG routing, retrieval, source evidence, and streaming."""

import json

import pytest

from app.config import get_settings
from app.services import chat_engine
from app.services.chat_engine import (
    SOURCES_HEADING,
    _keyword_query,
    _to_lc_history,
    build_reasoning_paths,
    classify_query,
    generate_rag_response,
    retrieve_subgraph,
)


@pytest.fixture(autouse=True)
def _quiet_chunk_store(monkeypatch):
    """Chunk retrieval is on by default — keep it hermetic unless a test opts in.

    Without this, every retrieval test would reach for the chunk vector index
    (and therefore a real database). Tests that care about excerpts re-patch
    these two functions themselves.
    """

    async def nothing(*_args, **_kwargs):
        return []

    monkeypatch.setattr(chat_engine.chunk_store, "search_chunks", nothing)
    monkeypatch.setattr(chat_engine.chunk_store, "chunks_for_entities", nothing)


class TestKeywordQuery:
    def test_strips_stopwords_and_punctuation(self):
        q = _keyword_query("Who is Ahmed? Tell me about his skills!")
        assert "who" not in q and "his" not in q
        assert "ahmed" in q and "skills" in q

    def test_dedupes_terms(self):
        q = _keyword_query("python python PYTHON")
        assert q == "python"

    def test_all_stopwords_yields_empty(self):
        assert _keyword_query("who is the a an of to") == ""

    def test_escapes_lucene_specials(self):
        # Must not raise and must not leak Lucene operators.
        q = _keyword_query("C++ (react) AND docker!")
        assert "(" not in q and ")" not in q


class TestHistory:
    def test_converts_roles(self):
        msgs = _to_lc_history(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        )
        assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]

    def test_skips_empty_and_bounds_to_six(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        history.append({"role": "user", "content": ""})
        msgs = _to_lc_history(history)
        assert len(msgs) == 6  # last 6 non-empty

    def test_none_history(self):
        assert _to_lc_history(None) == []


# ── Query routing ────────────────────────────────────────────────────────────
GLOBAL_QUESTIONS = [
    "What are the main themes in my documents?",
    "Give me an overview of the knowledge base",
    "Summarize everything you know",
    "Can you summarise the corpus?",
    "What is the overall picture here?",
    "What topics are covered?",
    "Which topics come up the most?",
    "What are the key ideas across all documents?",
    "Describe the big picture",
    "Give me a high-level view",
    "In general, what does this collection say?",
    "What are the recurring patterns throughout these papers?",
    "What are these documents about?",
    "Tell me about the corpus as a whole",
    "TLDR of the graph please",
]

LOCAL_QUESTIONS = [
    "Who is Ahmed Maaloul?",
    "Where did Ahmed work in 2024?",
    "What tools does Crédit Mutuel use?",
    "Which projects use Argo Workflows?",
    "How is Ada Lovelace connected to the Analytical Engine?",
    "Does Neo4j support vector indexes?",
    "List the skills mentioned on the resume",
    "When was the Python library released?",
    "Compare Kafka and RabbitMQ",
    "What did Grace Hopper invent?",
]


class TestQueryRouting:
    @pytest.mark.parametrize("question", GLOBAL_QUESTIONS)
    def test_global_questions(self, question):
        assert classify_query(question) == "global"

    @pytest.mark.parametrize("question", LOCAL_QUESTIONS)
    def test_local_questions(self, question):
        assert classify_query(question) == "local"

    def test_empty_question_is_local(self):
        assert classify_query("") == "local"

    def test_disabled_flag_forces_local(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "query_routing_enabled", False, raising=False)
        assert classify_query("What are the main themes overall?") == "local"

    def test_enabled_flag_allows_global(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "query_routing_enabled", True, raising=False)
        assert classify_query("What are the main themes overall?") == "global"

    def test_broad_vs_focused_global_questions(self):
        assert chat_engine._is_broad_question("What are the main themes overall?")
        assert not chat_engine._is_broad_question(
            "Summarize what Ahmed worked on across all documents"
        )


# ── Reasoning paths ──────────────────────────────────────────────────────────
EDGES = [
    {"source": "Ahmed", "target": "Credit Mutuel", "rel": "WORKED_AT"},
    {"source": "Credit Mutuel", "target": "Argo Workflows", "rel": "USES_TOOL"},
    {"source": "Ahmed", "target": "Python", "rel": "KNOWS"},
    {"source": "Argo Workflows", "target": "Kubernetes", "rel": "RUNS_ON"},
]


class TestReasoningPaths:
    def test_renders_multi_hop_path_with_directions(self):
        paths = build_reasoning_paths(["Ahmed", "Argo Workflows"], EDGES)
        assert len(paths) == 1
        assert paths[0]["text"] == (
            "Ahmed -[WORKED_AT]-> Credit Mutuel -[USES_TOOL]-> Argo Workflows"
        )
        assert paths[0]["nodes"] == ["Ahmed", "Credit Mutuel", "Argo Workflows"]
        assert paths[0]["rels"] == ["WORKED_AT", "USES_TOOL"]
        assert paths[0]["dirs"] == [True, True]

    def test_reverse_direction_is_rendered_backwards(self):
        paths = build_reasoning_paths(["Argo Workflows", "Ahmed"], EDGES)
        assert paths[0]["text"] == (
            "Argo Workflows <-[USES_TOOL]- Credit Mutuel <-[WORKED_AT]- Ahmed"
        )
        # The adjacency is undirected, so this chain is walked against both
        # edges — the UI needs `dirs` to draw the arrows the right way round.
        assert paths[0]["dirs"] == [False, False]

    def test_dirs_are_emitted_and_agree_with_the_rendered_text(self):
        """Every path carries one boolean per hop, matching its own rendering."""
        edges = [
            {"source": "A", "target": "B", "rel": "R1"},  # A → B (forward)
            {"source": "C", "target": "B", "rel": "R2"},  # C → B, walked backwards
        ]
        paths = build_reasoning_paths(["A", "C"], edges)
        assert len(paths) == 1
        path = paths[0]
        assert len(path["dirs"]) == len(path["rels"]) == len(path["nodes"]) - 1
        assert all(isinstance(d, bool) for d in path["dirs"])
        assert path["dirs"] == [True, False]
        assert path["text"] == "A -[R1]-> B <-[R2]- C"

    def test_every_path_carries_dirs(self):
        paths = build_reasoning_paths(
            ["Ahmed", "Credit Mutuel", "Argo Workflows", "Kubernetes", "Python"], EDGES
        )
        assert paths
        for path in paths:
            assert len(path["dirs"]) == len(path["rels"])

    def test_prefers_shorter_paths(self):
        paths = build_reasoning_paths(
            ["Ahmed", "Kubernetes", "Credit Mutuel"], EDGES
        )
        hops = [len(p["rels"]) for p in paths]
        assert hops == sorted(hops)
        assert paths[0]["rels"] == ["WORKED_AT"]  # Ahmed → Credit Mutuel is shortest

    def test_prefers_paths_between_top_ranked_seeds(self):
        # Same length (2 hops) for both pairs; the higher-ranked pair wins.
        edges = [
            {"source": "A", "target": "X", "rel": "R1"},
            {"source": "X", "target": "B", "rel": "R2"},
            {"source": "C", "target": "Y", "rel": "R3"},
            {"source": "Y", "target": "D", "rel": "R4"},
        ]
        paths = build_reasoning_paths(["A", "B", "C", "D"], edges)
        assert paths[0]["nodes"] == ["A", "X", "B"]

    def test_respects_max_reasoning_paths(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "max_reasoning_paths", 2, raising=False)
        paths = build_reasoning_paths(
            ["Ahmed", "Credit Mutuel", "Argo Workflows", "Kubernetes", "Python"], EDGES
        )
        assert len(paths) == 2

    def test_zero_limit_returns_nothing(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "max_reasoning_paths", 0, raising=False)
        assert build_reasoning_paths(["Ahmed", "Argo Workflows"], EDGES) == []

    def test_needs_two_seeds_and_edges(self):
        assert build_reasoning_paths(["Ahmed"], EDGES) == []
        assert build_reasoning_paths(["Ahmed", "Ahmed"], EDGES) == []
        assert build_reasoning_paths(["Ahmed", "Argo Workflows"], []) == []

    def test_disconnected_seeds_yield_no_path(self):
        edges = [{"source": "A", "target": "B", "rel": "R"}]
        assert build_reasoning_paths(["A", "Z"], edges) == []

    def test_paths_longer_than_hop_budget_are_dropped(self, monkeypatch):
        settings = get_settings()
        # hops=1 → max path length 2; A..E is 4 hops away.
        monkeypatch.setattr(settings, "retrieval_max_hops", 1, raising=False)
        edges = [
            {"source": "A", "target": "B", "rel": "R"},
            {"source": "B", "target": "C", "rel": "R"},
            {"source": "C", "target": "D", "rel": "R"},
            {"source": "D", "target": "E", "rel": "R"},
        ]
        assert build_reasoning_paths(["A", "E"], edges) == []
        assert build_reasoning_paths(["A", "C"], edges)  # 2 hops still fits


class TestNeighborhoodExpansion:
    async def test_expands_hop_by_hop(self, fake_neo4j, monkeypatch):
        rows = {
            "Ahmed": [{"source": "Ahmed", "target": "Credit Mutuel", "rel": "WORKED_AT"}],
            "Credit Mutuel": [
                {"source": "Credit Mutuel", "target": "Argo Workflows", "rel": "USES_TOOL"}
            ],
        }

        def handler(query, params):
            out = []
            for name in params.get("names", []):
                out.extend(rows.get(name, []))
            return out

        calls = fake_neo4j(handler)
        edges = await chat_engine._neighborhood_edges(["Ahmed"], hops=2)
        assert len(calls) == 2  # one query per hop
        assert calls[1][1]["names"] == ["Credit Mutuel"]  # frontier = new nodes only
        assert {e["target"] for e in edges} == {"Credit Mutuel", "Argo Workflows"}

    async def test_db_failure_is_non_fatal(self, monkeypatch):
        async def boom(query, params=None):
            raise RuntimeError("db down")

        monkeypatch.setattr(chat_engine, "execute_query", boom)
        assert await chat_engine._neighborhood_edges(["Ahmed"], hops=2) == []


class TestRetrieval:
    async def test_expand_and_format_builds_citations(self, fake_neo4j):
        def handler(query, params):
            if "n.name IN" in query:
                return [
                    {
                        "entity": "Ada",
                        "type": "PERSON",
                        "description": "mathematician",
                        "connections": [
                            {
                                "related_entity": "Analytical Engine",
                                "related_type": "PROJECT",
                                "relationship": "WORKED_ON",
                            }
                        ],
                    }
                ]
            return []

        fake_neo4j(handler)
        context, citations = await chat_engine._expand_and_format(["Ada"])
        assert "Ada" in context
        assert "WORKED_ON" in context
        assert citations == [{"name": "Ada", "type": "PERSON", "kind": "entity"}]

    async def test_empty_seeds(self):
        context, citations = await chat_engine._expand_and_format([])
        assert citations == []
        assert "No relevant information" in context

    async def test_retrieve_merges_and_dedupes_seeds(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])

        async def fake_vec(q, k):
            return [{"name": "Ada"}, {"name": "Grace"}]

        async def fake_kw(q, k):
            return [{"name": "Grace"}, {"name": "Alan"}]  # Grace duplicated

        captured = {}

        async def fake_expand(names):
            captured["names"] = names
            return "ctx", []

        monkeypatch.setattr(chat_engine, "_seeds_by_vector", fake_vec)
        monkeypatch.setattr(chat_engine, "_seeds_by_keyword", fake_kw)
        monkeypatch.setattr(chat_engine, "_expand_and_format", fake_expand)

        await retrieve_subgraph("question", k=8)
        assert captured["names"] == ["Ada", "Grace", "Alan"]  # deduped, ordered

    async def test_local_retrieval_appends_reasoning_paths(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])

        async def fake_seeds(q, k):
            return ["Ahmed", "Argo Workflows"]

        async def fake_expand(names):
            return "CTX", [
                {"name": n, "type": "T", "kind": "entity"} for n in names
            ]

        async def fake_edges(names, hops):
            assert hops == get_settings().retrieval_max_hops
            return EDGES

        monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)
        monkeypatch.setattr(chat_engine, "_expand_and_format", fake_expand)
        monkeypatch.setattr(chat_engine, "_neighborhood_edges", fake_edges)

        result = await retrieve_subgraph("How is Ahmed linked to Argo Workflows?")
        assert result.mode == "local"
        assert "Reasoning paths:" in result.context
        assert "Ahmed -[WORKED_AT]-> Credit Mutuel" in result.context
        assert result.paths[0]["nodes"][0] == "Ahmed"

    async def test_returns_backward_compatible_two_tuple(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])

        async def fake_seeds(q, k):
            return ["Ada"]

        monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)
        context, citations = await retrieve_subgraph("Who is Ada?")
        assert isinstance(context, str)
        assert citations == []


# ── Global search ────────────────────────────────────────────────────────────
COMMUNITIES = [
    {
        "id": "c1",
        "title": "MLOps Platform",
        "summary": "Pipelines, orchestration and deployment.",
        "size": 7,
        "members": ["Argo Workflows", "Kubernetes"],
    },
    {
        "id": "c2",
        "title": "Career History",
        "summary": "Roles and employers.",
        "size": 4,
        "members": ["Ahmed", "Credit Mutuel"],
    },
]


class TestGlobalSearch:
    async def test_broad_question_lists_largest_communities(self, monkeypatch):
        used = {}

        async def fake_list(limit=10):
            used["list"] = limit
            return COMMUNITIES

        async def fake_search(query, k=5):
            used["search"] = query
            return COMMUNITIES

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", fake_list)
        monkeypatch.setattr(chat_engine.communities, "search_communities", fake_search)

        result = await retrieve_subgraph("What are the main themes overall?")
        assert "list" in used and "search" not in used
        assert result.mode == "global"
        assert "MLOps Platform" in result.context
        assert "Pipelines, orchestration" in result.context
        assert result.paths == []

    async def test_focused_question_vector_searches_communities(self, monkeypatch):
        used = {}

        async def fake_list(limit=10):
            used["list"] = limit
            return COMMUNITIES

        async def fake_search(query, k=5):
            used["search"] = query
            return COMMUNITIES

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", fake_list)
        monkeypatch.setattr(chat_engine.communities, "search_communities", fake_search)

        result = await retrieve_subgraph(
            "Summarize what Ahmed worked on across all documents"
        )
        assert used.get("search") is not None and "list" not in used
        assert result.mode == "global"

    async def test_community_citations_are_marked_and_backward_compatible(
        self, monkeypatch
    ):
        async def fake_list(limit=10):
            return COMMUNITIES

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", fake_list)
        _, citations = await retrieve_subgraph("Give me an overview")
        assert citations[0]["kind"] == "community"
        assert citations[0]["name"] == "MLOps Platform"
        assert citations[0]["type"]  # name/type still present for the existing UI
        assert citations[0]["members"] == ["Argo Workflows", "Kubernetes"]

    async def test_falls_back_to_local_when_no_communities(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])

        async def empty(limit=10):
            return []

        async def fake_seeds(q, k):
            return ["Ada"]

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", empty)
        monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)

        result = await retrieve_subgraph("What are the main themes?")
        assert result.mode == "local"

    async def test_community_failure_falls_back_to_local(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])

        async def boom(limit=10):
            raise RuntimeError("no community index")

        async def fake_seeds(q, k):
            return ["Ada"]

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", boom)
        monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)

        result = await retrieve_subgraph("Give me an overview")
        assert result.mode == "local"

    async def test_routing_disabled_never_hits_communities(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        settings = get_settings()
        monkeypatch.setattr(settings, "query_routing_enabled", False, raising=False)

        async def boom(limit=10):
            raise AssertionError("global search must not run when routing is disabled")

        async def fake_seeds(q, k):
            return ["Ada"]

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", boom)
        monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)

        result = await retrieve_subgraph("What are the main themes?")
        assert result.mode == "local"


# ── Source evidence (text units) ─────────────────────────────────────────────
def _excerpt(chunk_id, text, document="resume.pdf", index=0, score=0.9):
    return {
        "id": chunk_id,
        "text": text,
        "document": document,
        "index": index,
        "score": score,
    }


def _install_chunk_store(monkeypatch, *, semantic=(), structural=(), calls=None):
    """Stub both excerpt channels; record their arguments in ``calls``."""
    log = {} if calls is None else calls

    async def fake_search(query, k):
        log["semantic"] = {"query": query, "k": k}
        return list(semantic)

    async def fake_for_entities(names, limit):
        log["structural"] = {"names": list(names), "limit": limit}
        return list(structural)

    monkeypatch.setattr(chat_engine.chunk_store, "search_chunks", fake_search)
    monkeypatch.setattr(chat_engine.chunk_store, "chunks_for_entities", fake_for_entities)
    return log


def _install_local_retrieval(monkeypatch, seeds=("Ada",)):
    """Deterministic local pipeline: fixed seeds, fixed graph context, no edges."""

    async def fake_seeds(_question, _k):
        return list(seeds)

    async def fake_expand(names):
        return "GRAPH CONTEXT", [
            {"name": n, "type": "PERSON", "kind": "entity"} for n in names
        ]

    async def fake_edges(_names, _hops):
        return []

    monkeypatch.setattr(chat_engine, "_rank_seed_names", fake_seeds)
    monkeypatch.setattr(chat_engine, "_expand_and_format", fake_expand)
    monkeypatch.setattr(chat_engine, "_neighborhood_edges", fake_edges)


class TestSourceExcerpts:
    async def test_excerpts_are_appended_after_the_graph_context(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        _install_chunk_store(
            monkeypatch,
            structural=[_excerpt("c1", "Ada wrote the first algorithm in 1843.")],
        )

        result = await retrieve_subgraph("Who is Ada?")
        assert SOURCES_HEADING in result.context
        assert "Ada wrote the first algorithm in 1843." in result.context
        # Graph facts first, verbatim evidence after them.
        assert result.context.index("GRAPH CONTEXT") < result.context.index(SOURCES_HEADING)
        assert "resume.pdf" in result.context  # provenance is visible to the model
        assert [s["id"] for s in result.sources] == ["c1"]

    async def test_structural_lookup_uses_the_selected_entities(self, monkeypatch):
        _install_local_retrieval(monkeypatch, seeds=("Ada", "Analytical Engine"))
        settings = get_settings()
        monkeypatch.setattr(settings, "chunk_top_k", 3, raising=False)
        calls = _install_chunk_store(monkeypatch, structural=[_excerpt("c1", "prose")])

        await retrieve_subgraph("Who is Ada?")
        assert calls["structural"]["names"] == ["Ada", "Analytical Engine"]
        assert calls["structural"]["limit"] == 3
        assert calls["semantic"] == {"query": "Who is Ada?", "k": 3}

    async def test_merges_both_channels_deduped_by_chunk_id(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        _install_chunk_store(
            monkeypatch,
            structural=[_excerpt("c1", "structural one"), _excerpt("c2", "shared")],
            semantic=[_excerpt("c2", "shared"), _excerpt("c3", "semantic only")],
        )

        result = await retrieve_subgraph("Who is Ada?")
        assert [s["id"] for s in result.sources] == ["c1", "c2", "c3"]
        assert result.context.count("shared") == 1

    async def test_blank_excerpts_are_dropped(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        _install_chunk_store(
            monkeypatch,
            structural=[_excerpt("c1", "   "), _excerpt("c2", "real prose")],
        )

        result = await retrieve_subgraph("Who is Ada?")
        assert [s["id"] for s in result.sources] == ["c2"]

    async def test_char_cap_truncates_at_an_excerpt_boundary(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        chunks = [
            _excerpt("c1", "A" * 100, document="d.pdf", index=0),
            _excerpt("c2", "B" * 100, document="d.pdf", index=1),
            _excerpt("c3", "C" * 100, document="d.pdf", index=2),
        ]
        settings = get_settings()
        monkeypatch.setattr(settings, "chunk_top_k", 5, raising=False)
        # "[S1] d.pdf (chunk 0)\n" + 100 chars = 121 per block; 250 fits exactly two.
        monkeypatch.setattr(settings, "chunk_context_max_chars", 250, raising=False)
        _install_chunk_store(monkeypatch, structural=chunks)

        result = await retrieve_subgraph("Who is Ada?")
        assert "A" * 100 in result.context
        assert "B" * 100 in result.context
        assert "C" * 100 not in result.context  # never cut mid-excerpt
        assert "C" * 10 not in result.context
        assert "(1 further excerpt omitted for length.)" in result.context
        assert [s["id"] for s in result.sources] == ["c1", "c2"]

    def test_omitted_count_is_pluralized(self):
        chunks = [_excerpt(f"c{i}", "X" * 100, document="d.pdf", index=i) for i in range(4)]
        section, kept = chat_engine._format_excerpts(chunks, 250)
        assert len(kept) == 2
        assert "(2 further excerpts omitted for length.)" in section

    def test_budget_that_fits_nothing_yields_no_section(self):
        section, kept = chat_engine._format_excerpts([_excerpt("c1", "x" * 500)], 10)
        assert section == "" and kept == []

    async def test_zero_budget_degrades_to_graph_only(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        settings = get_settings()
        monkeypatch.setattr(settings, "chunk_context_max_chars", 0, raising=False)
        _install_chunk_store(monkeypatch, structural=[_excerpt("c1", "prose")])

        result = await retrieve_subgraph("Who is Ada?")
        assert SOURCES_HEADING not in result.context
        assert result.context.startswith("GRAPH CONTEXT")
        assert result.sources == []

    @pytest.mark.parametrize("channel", ["search_chunks", "chunks_for_entities"])
    async def test_chunk_failure_never_breaks_the_answer(self, monkeypatch, channel):
        _install_local_retrieval(monkeypatch)
        _install_chunk_store(monkeypatch, structural=[_excerpt("c1", "survivor")])

        async def boom(*_args, **_kwargs):
            raise RuntimeError("chunk index missing")

        monkeypatch.setattr(chat_engine.chunk_store, channel, boom)

        result = await retrieve_subgraph("Who is Ada?")
        assert result.mode == "local"
        assert "GRAPH CONTEXT" in result.context
        assert result.citations == [{"name": "Ada", "type": "PERSON", "kind": "entity"}]
        if channel == "search_chunks":
            # The surviving channel still contributes its evidence.
            assert [s["id"] for s in result.sources] == ["c1"]
        else:
            assert result.sources == []

    async def test_total_chunk_failure_degrades_to_todays_answer(self, monkeypatch):
        _install_local_retrieval(monkeypatch)

        async def boom(*_args, **_kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(chat_engine.chunk_store, "search_chunks", boom)
        monkeypatch.setattr(chat_engine.chunk_store, "chunks_for_entities", boom)

        result = await retrieve_subgraph("Who is Ada?")
        assert result.context == "GRAPH CONTEXT"
        assert result.sources == []

    async def test_disabled_flag_skips_chunk_retrieval_entirely(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        settings = get_settings()
        monkeypatch.setattr(settings, "chunk_retrieval_enabled", False, raising=False)
        calls = _install_chunk_store(monkeypatch, structural=[_excerpt("c1", "prose")])

        result = await retrieve_subgraph("Who is Ada?")
        assert calls == {}  # neither channel was even consulted
        assert SOURCES_HEADING not in result.context
        assert result.sources == []

    async def test_zero_top_k_skips_chunk_retrieval(self, monkeypatch):
        _install_local_retrieval(monkeypatch)
        settings = get_settings()
        monkeypatch.setattr(settings, "chunk_top_k", 0, raising=False)
        calls = _install_chunk_store(monkeypatch, structural=[_excerpt("c1", "prose")])

        result = await retrieve_subgraph("Who is Ada?")
        assert calls == {}
        assert result.sources == []

    async def test_global_search_attaches_member_excerpts(self, monkeypatch):
        async def fake_list(limit=10):
            return COMMUNITIES

        monkeypatch.setattr(chat_engine.communities, "get_community_summaries", fake_list)
        calls = _install_chunk_store(
            monkeypatch,
            structural=[_excerpt("c1", "Argo Workflows orchestrates the pipelines.")],
        )

        result = await retrieve_subgraph("What are the main themes overall?")
        assert result.mode == "global"
        assert calls["structural"]["names"] == [
            "Argo Workflows",
            "Kubernetes",
            "Ahmed",
            "Credit Mutuel",
        ]
        assert SOURCES_HEADING in result.context
        assert result.context.index(chat_engine.COMMUNITY_HEADING) < result.context.index(
            SOURCES_HEADING
        )
        assert [s["id"] for s in result.sources] == ["c1"]

    def test_community_member_names_are_deduped_and_capped(self):
        summaries = [
            {"members": [f"E{i}" for i in range(15)]},
            {"members": ["E0", *[f"F{i}" for i in range(15)]]},
        ]
        names = chat_engine._community_member_names(summaries)
        assert len(names) == chat_engine.MAX_COMMUNITY_MEMBERS_FOR_CHUNKS
        assert len(set(names)) == len(names)
        assert names[:3] == ["E0", "E1", "E2"]


# ── Streaming ────────────────────────────────────────────────────────────────
class _Chunk:
    def __init__(self, content):
        self.content = content


class _FakeChain:
    async def astream(self, _inputs):
        for tok in ["Hello", " ", "world"]:
            yield _Chunk(tok)


def _install_fake_llm(monkeypatch):
    monkeypatch.setattr(chat_engine, "get_chat_llm", lambda **kw: object())
    monkeypatch.setattr(chat_engine, "RAG_PROMPT", _PipeTo(_FakeChain()))


class TestStreaming:
    async def test_events_order_citations_tokens_done(self, monkeypatch):
        async def fake_retrieve(q, k=8):
            return "CONTEXT", [{"name": "Ada", "type": "PERSON"}]

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        events = [e async for e in generate_rag_response("hi", [])]
        types = [e["type"] for e in events]
        assert types[0] == "citations"
        assert "paths" not in types  # no paths → no empty event
        assert types[-1] == "done"
        assert "".join(e["data"] for e in events if e["type"] == "token") == "Hello world"

    async def test_events_order_citations_paths_tokens_done(self, monkeypatch):
        paths = [
            {
                "nodes": ["Ahmed", "Credit Mutuel", "Argo Workflows"],
                "rels": ["WORKED_AT", "USES_TOOL"],
                "text": "Ahmed -[WORKED_AT]-> Credit Mutuel -[USES_TOOL]-> Argo Workflows",
            }
        ]

        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval("CONTEXT", [{"name": "Ahmed"}], paths, "local")

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        events = [e async for e in generate_rag_response("hi", [])]
        types = [e["type"] for e in events]
        assert types[:2] == ["citations", "paths"]
        assert types[2] == "token"
        assert types[-1] == "done"
        assert events[1]["data"] == paths

    async def test_paths_event_carries_hop_directions(self, monkeypatch):
        """The SSE payload must keep `dirs` — the UI draws arrows from it."""
        paths = build_reasoning_paths(["Argo Workflows", "Ahmed"], EDGES)
        assert paths[0]["dirs"] == [False, False]

        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval("CONTEXT", [{"name": "Ahmed"}], paths, "local")

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        events = [e async for e in generate_rag_response("hi", [])]
        payload = next(e for e in events if e["type"] == "paths")["data"]
        assert payload[0]["dirs"] == [False, False]
        # The event is JSON-serialized verbatim by the /chat SSE route.
        assert json.loads(json.dumps(payload))[0]["dirs"] == [False, False]

    async def test_full_event_order_citations_paths_sources_tokens_done(
        self, monkeypatch
    ):
        paths = build_reasoning_paths(["Ahmed", "Argo Workflows"], EDGES)
        sources = [_excerpt("c1", "Ahmed joined Credit Mutuel in 2023.", index=2)]

        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval(
                "CONTEXT", [{"name": "Ahmed"}], paths, "local", sources
            )

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        events = [e async for e in generate_rag_response("hi", [])]
        types = [e["type"] for e in events]
        assert types[:3] == ["citations", "paths", "sources"]
        assert types[3] == "token"
        assert types[-1] == "done"

        payload = events[2]["data"]
        assert payload == [
            {
                "id": "c1",
                "document": "resume.pdf",
                "index": 2,
                "text": "Ahmed joined Credit Mutuel in 2023.",
            }
        ]
        # The event is JSON-serialized verbatim by the /chat SSE route.
        assert json.loads(json.dumps(payload)) == payload

    async def test_sources_event_without_paths_still_precedes_tokens(self, monkeypatch):
        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval(
                "CONTEXT", [{"name": "Ada"}], [], "local", [_excerpt("c1", "prose")]
            )

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        types = [e["type"] async for e in generate_rag_response("hi", [])]
        assert types[:3] == ["citations", "sources", "token"]

    async def test_no_sources_event_when_there_is_no_evidence(self, monkeypatch):
        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval("CONTEXT", [{"name": "Ada"}], [], "local", [])

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        types = [e["type"] async for e in generate_rag_response("hi", [])]
        assert "sources" not in types

    async def test_source_text_is_truncated_for_transport(self, monkeypatch):
        long_text = "x" * 1000

        async def fake_retrieve(q, k=8):
            return chat_engine.Retrieval(
                "CONTEXT", [], [], "local", [_excerpt("c1", long_text)]
            )

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        _install_fake_llm(monkeypatch)

        events = [e async for e in generate_rag_response("hi", [])]
        text = next(e for e in events if e["type"] == "sources")["data"][0]["text"]
        assert len(text) == chat_engine.SOURCE_SNIPPET_CHARS
        assert text.endswith("…")

    async def test_history_reaches_the_chain(self, monkeypatch):
        captured = {}

        class Recorder(_FakeChain):
            async def astream(self, inputs):
                captured.update(inputs)
                for chunk in []:  # pragma: no cover - no tokens needed
                    yield chunk

        async def fake_retrieve(q, k=8):
            return "CONTEXT", []

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        monkeypatch.setattr(chat_engine, "get_chat_llm", lambda **kw: object())
        monkeypatch.setattr(chat_engine, "RAG_PROMPT", _PipeTo(Recorder()))

        history = [{"role": "user", "content": "who is ada?"}]
        [e async for e in generate_rag_response("and grace?", history)]
        assert [type(m).__name__ for m in captured["history"]] == ["HumanMessage"]
        assert captured["question"] == "and grace?"

    async def test_retrieval_error_yields_error_event(self, monkeypatch):
        async def boom(q, k=8):
            raise RuntimeError("db down")

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", boom)
        events = [e async for e in generate_rag_response("hi", [])]
        assert events == [{"type": "error", "data": "Failed to query the knowledge graph."}]

    async def test_generation_error_yields_error_after_citations(self, monkeypatch):
        async def fake_retrieve(q, k=8):
            return "CONTEXT", [{"name": "Ada", "type": "PERSON"}]

        def explode(**kw):
            raise RuntimeError("model offline")

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)
        monkeypatch.setattr(chat_engine, "get_chat_llm", explode)

        events = [e async for e in generate_rag_response("hi", [])]
        assert [e["type"] for e in events] == ["citations", "error"]
        assert "model offline" in events[-1]["data"]


class _PipeTo:
    """Object whose ``| x`` returns a preset chain (mimics RAG_PROMPT | llm)."""

    def __init__(self, chain):
        self._chain = chain

    def __or__(self, _other):
        return self._chain
