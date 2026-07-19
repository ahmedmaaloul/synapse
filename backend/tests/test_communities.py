"""Tests for Louvain community detection, LLM summarization, and persistence.

Fully hermetic: the graph is hand-built in memory, Neo4j is the ``fake_neo4j``
fixture, and the LLM is a stub chain — no network, no DB, no model.
"""

from __future__ import annotations

import threading

import networkx as nx
import pytest

from app.config import get_settings
from app.services import communities
from app.services.communities import (
    _parse_summary_json,
    build_graph_from_rows,
    community_id,
    detect_communities,
)


@pytest.fixture
def fake_db(fake_neo4j, monkeypatch):
    """``fake_neo4j`` + the patch for this module's own ``execute_query`` import."""

    def install(handler):
        calls = fake_neo4j(handler)
        from app import neo4j_driver  # now holds the fake

        monkeypatch.setattr(communities, "execute_query", neo4j_driver.execute_query)
        return calls

    return install


def two_cluster_graph() -> nx.Graph:
    """Two 4-node cliques joined by a single bridge — an unambiguous 2-community graph."""
    graph = nx.Graph()
    a = ["A1", "A2", "A3", "A4"]
    b = ["B1", "B2", "B3", "B4"]
    for group in (a, b):
        for i, u in enumerate(group):
            for v in group[i + 1 :]:
                graph.add_edge(u, v, weight=1.0)
    graph.add_edge("A1", "B1", weight=1.0)
    return graph


class _PipeStub:
    """Stands in for the ChatPromptTemplate so ``prompt | llm`` yields a fake chain."""

    def __init__(self, content: str):
        self.content = content

    def __or__(self, _other):
        content = self.content

        class FakeResponse:
            def __init__(self):
                self.content = content

        class FakeChain:
            async def ainvoke(self, _inputs):
                return FakeResponse()

        return FakeChain()


def install_fake_llm(monkeypatch, content: str) -> None:
    monkeypatch.setattr(communities, "get_chat_llm", lambda **kw: object())
    monkeypatch.setattr(communities, "SUMMARY_PROMPT", _PipeStub(content))


GOOD_LLM_JSON = '{"title": "Ada and the Engine", "summary": "A tight cluster. It matters."}'


# ── Graph construction ───────────────────────────────
class TestBuildGraphFromRows:
    def test_builds_undirected_graph_with_metadata(self):
        rows = [
            {
                "source": "Ada",
                "target": "Engine",
                "rel_type": "WORKED_ON",
                "source_type": "PERSON",
                "source_description": "mathematician",
                "target_type": "PROJECT",
                "target_description": "a machine",
            }
        ]
        graph, meta, edges = build_graph_from_rows(rows)
        assert graph.has_edge("Engine", "Ada")  # undirected
        assert meta["Ada"] == {"type": "PERSON", "description": "mathematician"}
        assert edges == [{"source": "Ada", "target": "Engine", "type": "WORKED_ON"}]

    def test_parallel_edges_increase_weight(self):
        rows = [
            {"source": "A", "target": "B", "rel_type": "X"},
            {"source": "A", "target": "B", "rel_type": "Y"},
        ]
        graph, _meta, edges = build_graph_from_rows(rows)
        assert graph["A"]["B"]["weight"] == 2.0
        assert len(edges) == 2  # both relationship types kept for the prompt

    def test_duplicate_identical_edges_are_deduped_in_descriptions(self):
        rows = [{"source": "A", "target": "B", "rel_type": "X"}] * 3
        _graph, _meta, edges = build_graph_from_rows(rows)
        assert len(edges) == 1

    def test_skips_self_loops_and_blanks(self):
        rows = [
            {"source": "A", "target": "A", "rel_type": "X"},
            {"source": "", "target": "B", "rel_type": "X"},
            {"source": "C", "target": None, "rel_type": "X"},
        ]
        graph, _meta, edges = build_graph_from_rows(rows)
        assert graph.number_of_edges() == 0
        assert edges == []

    def test_empty_rows(self):
        graph, meta, edges = build_graph_from_rows([])
        assert graph.number_of_nodes() == 0
        assert (meta, edges) == ({}, [])

    def test_missing_rel_type_defaults(self):
        _graph, _meta, edges = build_graph_from_rows(
            [{"source": "A", "target": "B", "rel_type": None}]
        )
        assert edges[0]["type"] == "RELATED_TO"


# ── Detection ────────────────────────────────────────
class TestDetectCommunities:
    def test_finds_exactly_two_obvious_clusters(self):
        found, modularity = detect_communities(two_cluster_graph(), get_settings())
        assert len(found) == 2
        # each clique stays intact and un-mixed
        prefixes = [{name[0] for name in group} for group in found]
        assert prefixes == [{"A"}, {"B"}] or prefixes == [{"B"}, {"A"}]
        assert all(len(group) == 4 for group in found)
        assert 0.0 < modularity < 1.0

    def test_is_reproducible(self):
        first, m1 = detect_communities(two_cluster_graph(), get_settings())
        second, m2 = detect_communities(two_cluster_graph(), get_settings())
        assert first == second
        assert m1 == m2

    def test_partition_and_modularity_are_pinned(self):
        """Exact expected output for the fixture — LOUVAIN_SEED + sorted rows.

        Pinned rather than merely self-consistent: any change to the seed, the row
        ordering, or the weighting silently reshuffles every community id and
        summary downstream, and this is what catches it.
        """
        found, modularity = detect_communities(two_cluster_graph(), get_settings())
        assert found == [["A1", "A2", "A3", "A4"], ["B1", "B2", "B3", "B4"]]
        assert modularity == 0.4231

    def test_min_size_filters_small_clusters(self, monkeypatch):
        settings = get_settings()
        graph = two_cluster_graph()
        # A detached pair — a real cluster, but below the noise floor.
        graph.add_edge("Z1", "Z2", weight=1.0)

        monkeypatch.setattr(settings, "community_min_size", 3)
        kept, _ = detect_communities(graph, settings)
        assert all(len(group) >= 3 for group in kept)
        assert not any("Z1" in group for group in kept)

        monkeypatch.setattr(settings, "community_min_size", 2)
        kept_small, _ = detect_communities(graph, settings)
        assert any("Z1" in group for group in kept_small)

    def test_min_size_above_everything_returns_nothing(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "community_min_size", 99)
        kept, modularity = detect_communities(two_cluster_graph(), settings)
        assert kept == []
        # modularity is still reported: it describes the *full* partition
        assert modularity > 0

    def test_members_ordered_by_centrality(self):
        graph = nx.Graph()
        for leaf in ("l1", "l2", "l3", "l4"):
            graph.add_edge("hub", leaf, weight=1.0)
        kept, _ = detect_communities(graph, get_settings())
        assert kept[0][0] == "hub"

    def test_empty_graph_returns_zeros(self):
        assert detect_communities(nx.Graph(), get_settings()) == ([], 0.0)

    def test_graph_with_no_edges_returns_zeros(self):
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        assert detect_communities(graph, get_settings()) == ([], 0.0)


class TestCommunityId:
    def test_stable_and_order_independent(self):
        assert community_id(["b", "a"]) == community_id(["a", "b"])

    def test_differs_by_membership(self):
        assert community_id(["a", "b"]) != community_id(["a", "c"])


# ── Summary parsing ──────────────────────────────────
class TestParseSummaryJson:
    def test_plain_json(self):
        out = _parse_summary_json('{"title": "Graph Theory", "summary": "About graphs."}')
        assert out == {"title": "Graph Theory", "summary": "About graphs."}

    def test_markdown_fenced(self):
        raw = '```json\n{"title": "T", "summary": "S"}\n```'
        assert _parse_summary_json(raw)["title"] == "T"

    def test_prose_around_json(self):
        raw = 'Sure!\n{"title": "T", "summary": "S"}\nHope that helps.'
        assert _parse_summary_json(raw)["summary"] == "S"

    def test_garbage_returns_empty_shape(self):
        assert _parse_summary_json("not json at all {oops") == {"title": "", "summary": ""}

    def test_empty_input(self):
        assert _parse_summary_json("") == {"title": "", "summary": ""}

    def test_non_dict_json(self):
        assert _parse_summary_json("[1, 2, 3]") == {"title": "", "summary": ""}

    def test_missing_keys(self):
        assert _parse_summary_json('{"other": 1}') == {"title": "", "summary": ""}


class TestSummarizeCommunity:
    async def test_uses_llm_output(self, monkeypatch):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        settings = get_settings()
        chain = communities.SUMMARY_PROMPT | object()
        result = await communities._summarize_community(
            chain, ["A", "B", "C"], {}, [], settings=settings
        )
        assert result["title"] == "Ada and the Engine"
        assert result["size"] == 3
        assert result["llm_ok"] is True

    async def test_garbage_output_falls_back_without_raising(self, monkeypatch):
        install_fake_llm(monkeypatch, "I am a helpful assistant, not JSON.")
        chain = communities.SUMMARY_PROMPT | object()
        result = await communities._summarize_community(
            chain, ["A", "B", "C"], {}, [], settings=get_settings()
        )
        assert result["title"]  # derived from members
        assert result["summary"]
        assert result["llm_ok"] is False

    async def test_llm_exception_is_contained(self, monkeypatch):
        class BoomChain:
            async def ainvoke(self, _inputs):
                raise RuntimeError("provider exploded")

        result = await communities._summarize_community(
            BoomChain(), ["A", "B"], {}, [], settings=get_settings()
        )
        assert result["llm_ok"] is False
        assert result["size"] == 2

    async def test_members_are_capped_in_the_prompt(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "community_max_members_in_summary", 2)
        seen: dict = {}

        class CapturingChain:
            async def ainvoke(self, inputs):
                seen.update(inputs)

                class R:
                    content = '{"title": "T", "summary": "S"}'

                return R()

        await communities._summarize_community(
            CapturingChain(),
            ["A", "B", "C", "D"],
            {"A": {"type": "PERSON", "description": "d"}},
            [{"source": "A", "target": "B", "type": "KNOWS"}],
            settings=settings,
        )
        assert "A" in seen["members"] and "B" in seen["members"]
        assert "C" not in seen["members"]  # capped
        assert "KNOWS" in seen["relationships"]


# ── Orchestration + persistence ──────────────────────
GRAPH_ROWS = [
    {"source": u, "target": v, "rel_type": "RELATED_TO", "source_type": "CONCEPT",
     "source_description": "", "target_type": "CONCEPT", "target_description": ""}
    for u, v in two_cluster_graph().edges()
]


class TestDetectAndSummarize:
    async def test_end_to_end_persists_expected_cypher(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        calls = fake_db(handler)

        events: list[dict] = []

        async def on_progress(event):
            events.append(event)

        result = await communities.detect_and_summarize(on_progress=on_progress)

        assert result["communities"] == 2
        assert result["summarized"] == 2
        assert 0.0 < result["modularity"] < 1.0

        queries = [q for q, _ in calls]
        # prior communities wiped first, so re-running is idempotent
        assert any("DETACH DELETE c" in q for q in queries)
        wipe_idx = next(i for i, q in enumerate(queries) if "DETACH DELETE c" in q)
        write_idx = next(i for i, q in enumerate(queries) if "CREATE (c:Community" in q)
        assert wipe_idx < write_idx

        writes = [
            (q, p) for q, p in calls if "CREATE (c:Community" in q
        ]
        assert len(writes) == 2
        query, params = writes[0]
        assert "IN_COMMUNITY" in query
        assert "db.create.setNodeVectorProperty" in query
        assert set(params) == {"id", "title", "summary", "size", "members", "embedding"}
        assert params["title"] == "Ada and the Engine"
        assert params["size"] == 4
        assert len(params["members"]) == 4
        assert len(params["embedding"]) == get_settings().embedding_dim

        stages = {e.get("stage") for e in events}
        assert {"loading_graph", "summarizing", "writing_communities"} <= stages
        assert any(e.get("processed") == 2 for e in events)

    async def test_falls_back_when_vector_property_unsupported(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            if "setNodeVectorProperty" in query:
                raise RuntimeError("Unknown procedure")
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        calls = fake_db(handler)
        result = await communities.detect_and_summarize()

        assert result["communities"] == 2
        fallbacks = [
            q for q, _ in calls
            if "CREATE (c:Community" in q and "setNodeVectorProperty" not in q
        ]
        assert len(fallbacks) == 2

    async def test_empty_graph_returns_zeros(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        calls = fake_db(lambda q, p: [])

        result = await communities.detect_and_summarize()

        assert result == {"communities": 0, "summarized": 0, "modularity": 0.0}
        assert not any("CREATE (c:Community" in q for q, _ in calls)

    async def test_unreachable_db_returns_zeros(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            raise RuntimeError("Neo4j is down")

        calls = fake_db(handler)
        result = await communities.detect_and_summarize()
        assert result == {"communities": 0, "summarized": 0, "modularity": 0.0}
        # Zeros, but nothing was written or deleted — see TestUnreadableGraphIsNotAnEmptyGraph.
        assert not any("DETACH DELETE c" in q for q, _ in calls)

    async def test_no_llm_still_persists_clusters(self, monkeypatch, fake_db):
        def boom(**_kw):
            raise RuntimeError("no provider configured")

        monkeypatch.setattr(communities, "get_chat_llm", boom)

        def handler(query, params):
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        fake_db(handler)
        result = await communities.detect_and_summarize()
        assert result["communities"] == 2
        assert result["summarized"] == 0  # titles are derived, not LLM-written


class TestStaleCommunitiesAreWiped:
    """A rebuild that finds nothing must still clear the previous run's nodes.

    Regression: the wipe used to live inside ``_persist_communities``, which the
    zero-community early return never reaches — so an emptied graph left every
    stale :Community node behind and the UI kept showing dead themes.
    """

    async def test_empty_graph_wipes_previous_communities(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        calls = fake_db(lambda q, p: [])

        result = await communities.detect_and_summarize()

        assert result == {"communities": 0, "summarized": 0, "modularity": 0.0}
        queries = [q for q, _ in calls]
        assert any("DETACH DELETE c" in q for q in queries)
        assert not any("CREATE (c:Community" in q for q in queries)

    async def test_min_size_filtering_everything_out_still_wipes(
        self, monkeypatch, fake_db
    ):
        """Graph is healthy, but nothing clears the noise floor: still a wipe."""
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        monkeypatch.setattr(get_settings(), "community_min_size", 99)

        def handler(query, params):
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        calls = fake_db(handler)
        result = await communities.detect_and_summarize()

        assert result["communities"] == 0
        assert result["modularity"] > 0  # the partition itself was fine
        queries = [q for q, _ in calls]
        assert any("DETACH DELETE c" in q for q in queries)
        assert not any("CREATE (c:Community" in q for q in queries)

    async def test_wipe_failure_does_not_raise(self, monkeypatch, fake_db):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            if "DETACH DELETE c" in query:
                raise RuntimeError("write conflict")
            return []

        fake_db(handler)
        assert await communities.detect_and_summarize() == {
            "communities": 0,
            "summarized": 0,
            "modularity": 0.0,
        }

    async def test_populated_rebuild_wipes_exactly_once(self, monkeypatch, fake_db):
        """Hoisting the wipe must not leave a second copy on the happy path."""
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        calls = fake_db(handler)
        await communities.detect_and_summarize()

        assert sum(1 for q, _ in calls if "DETACH DELETE c" in q) == 1


class TestUnreadableGraphIsNotAnEmptyGraph:
    """The wipe above is safe *only* if "empty" and "unreadable" stay distinct.

    ``_fetch_graph_rows`` used to swallow every exception and return ``[]``, so a
    transient Neo4j failure looked exactly like an empty graph — and took the
    zero-community branch, which now deletes every :Community node. One hiccup
    would have destroyed the whole corpus's themes with no way to tell it had
    happened. A failed read must abort without touching the database.
    """

    async def test_read_failure_raises_instead_of_returning_empty(self, fake_db):
        def handler(query, params):
            raise RuntimeError("connection reset by peer")

        fake_db(handler)
        with pytest.raises(communities.GraphLoadError):
            await communities._fetch_graph_rows()

    async def test_genuinely_empty_graph_still_returns_an_empty_list(self, fake_db):
        fake_db(lambda q, p: [])
        assert await communities._fetch_graph_rows() == []

    async def test_read_failure_aborts_without_wiping_communities(
        self, monkeypatch, fake_db
    ):
        """Only the *load* fails; every other query would succeed. Nothing is written."""
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
                raise RuntimeError("Neo4j is down")
            return []

        calls = fake_db(handler)
        result = await communities.detect_and_summarize()

        assert result == {"communities": 0, "summarized": 0, "modularity": 0.0}
        queries = [q for q, _ in calls]
        assert not any("DETACH DELETE c" in q for q in queries)
        assert not any("CREATE (c:Community" in q for q in queries)

    async def test_read_failure_still_closes_out_the_progress_stream(
        self, monkeypatch, fake_db
    ):
        """The UI must not sit on a spinner because the load failed."""
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)

        def handler(query, params):
            if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
                raise RuntimeError("Neo4j is down")
            return []

        fake_db(handler)
        events: list[dict] = []

        async def on_progress(event):
            events.append(event)

        await communities.detect_and_summarize(on_progress=on_progress)

        assert {"loading_graph", "summarizing"} <= {e.get("stage") for e in events}
        assert events[-1]["total"] == 0

    async def test_empty_graph_still_wipes_after_the_fix(self, monkeypatch, fake_db):
        """The other half of the contract: a real empty graph *does* clear themes."""
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        calls = fake_db(lambda q, p: [])

        await communities.detect_and_summarize()

        assert any("DETACH DELETE c" in q for q, _ in calls)


class TestEventLoopIsNotBlocked:
    """Louvain is CPU-bound; running it inline stalls every other request."""

    async def test_graph_build_and_louvain_run_off_the_event_loop(
        self, monkeypatch, fake_db
    ):
        install_fake_llm(monkeypatch, GOOD_LLM_JSON)
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}

        real_build = communities.build_graph_from_rows
        real_detect = communities.detect_communities

        def spying_build(rows):
            seen["build"] = threading.get_ident()
            return real_build(rows)

        def spying_detect(graph, settings=None):
            seen["detect"] = threading.get_ident()
            return real_detect(graph, settings)

        monkeypatch.setattr(communities, "build_graph_from_rows", spying_build)
        monkeypatch.setattr(communities, "detect_communities", spying_detect)

        def handler(query, params):
            return GRAPH_ROWS if "MATCH (a:Entity)-[r]->(b:Entity)" in query else []

        fake_db(handler)
        result = await communities.detect_and_summarize()

        # offloaded, not skipped: the real work still produced the real answer
        assert result["communities"] == 2
        assert result["modularity"] == 0.4231
        assert seen["detect"] != loop_thread, "Louvain ran on the event loop thread"
        assert seen["build"] != loop_thread, "graph build ran on the event loop thread"


# ── Read paths ───────────────────────────────────────
COMMUNITY_ROW = {
    "id": "com-abc123",
    "title": "Graph Theory",
    "summary": "All about graphs.",
    "size": 4,
    "members": ["A1", "A2", "A3", "A4"],
}


class TestGetCommunitySummaries:
    async def test_returns_contracted_shape(self, fake_db):
        fake_db(lambda q, p: [COMMUNITY_ROW])
        out = await communities.get_community_summaries(limit=5)
        assert len(out) == 1
        assert set(out[0]) == {"id", "title", "summary", "size", "members"}
        assert out[0]["size"] == 4
        assert out[0]["members"] == ["A1", "A2", "A3", "A4"]

    async def test_limit_is_passed_through(self, fake_db):
        calls = fake_db(lambda q, p: [])
        await communities.get_community_summaries(limit=3)
        assert calls[0][1] == {"limit": 3}

    async def test_missing_size_is_derived_and_nulls_dropped(self, fake_db):
        fake_db(lambda q, p: [{"id": "x", "title": None, "summary": None,
                               "size": None, "members": ["A", None]}])
        out = await communities.get_community_summaries()
        assert out[0] == {"id": "x", "title": "", "summary": "", "size": 1,
                          "members": ["A"]}

    async def test_db_error_returns_empty_list(self, fake_db):
        def handler(q, p):
            raise RuntimeError("no such index")

        fake_db(handler)
        assert await communities.get_community_summaries() == []


class TestSearchCommunities:
    async def test_vector_search_shape(self, fake_db):
        row = dict(COMMUNITY_ROW, score=0.91)
        calls = fake_db(lambda q, p: [row] if "queryNodes" in q else [])

        out = await communities.search_communities("main themes?", k=5)

        assert len(out) == 1
        assert {"id", "title", "summary", "size", "members"} <= set(out[0])
        assert out[0]["score"] == pytest.approx(0.91)
        # a real query vector of the configured dimension was sent
        _query, params = calls[0]
        assert params["k"] == 5
        assert len(params["vec"]) == get_settings().embedding_dim

    async def test_respects_k(self, fake_db):
        rows = [dict(COMMUNITY_ROW, id=f"c{i}", score=1.0 - i / 10) for i in range(5)]
        fake_db(lambda q, p: rows if "queryNodes" in q else [])
        out = await communities.search_communities("themes", k=2)
        assert len(out) == 2

    async def test_falls_back_to_listing_when_index_missing(self, fake_db):
        def handler(query, params):
            if "queryNodes" in query:
                raise RuntimeError("no such vector schema index")
            return [COMMUNITY_ROW]

        fake_db(handler)
        out = await communities.search_communities("themes", k=5)
        assert out[0]["id"] == "com-abc123"
        assert "score" not in out[0]

    async def test_falls_back_when_embedding_fails(self, monkeypatch, fake_db):
        def boom(*_a, **_kw):
            raise RuntimeError("embedding provider down")

        monkeypatch.setattr(communities, "get_embeddings", boom)
        fake_db(lambda q, p: [COMMUNITY_ROW])
        out = await communities.search_communities("themes")
        assert len(out) == 1
