# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Lab arms: the registry, the three floors, the passage baselines, the graph arms.

Hermetic: ``fake_neo4j`` answers every Cypher statement, the hybrid seed
ranker / chunk store are monkeypatched, and the pure scoring cores
(``score_paths``, ``lean_units``, ``ppr_rank_chunks``) run on hand-built graphs
whose expected numbers are computed in the test.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from app.lab import arms
from app.lab.arms import ARMS, Arm, arm_catalog, get_arm
from app.lab.packer import pack
from app.services import chat_engine, chunk_store
from app.services.graph_schema import CHUNK_FULLTEXT_INDEX


@pytest.fixture(autouse=True)
def _clean_caches():
    arms.clear_caches()
    yield
    arms.clear_caches()


def fingerprint_answer(q, entities=3, chunks=3, relations=2, mentions=2):
    """The count row for a fingerprint statement, or ``None`` when ``q`` is not one."""
    values = {"entities": entities, "chunks": chunks, "relations": relations,
              "mentions": mentions}
    for key, query in arms.FINGERPRINT_QUERIES.items():
        if q == query:
            return [{"n": values[key]}]
    return None


# ── Registry ─────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_the_eight_arms_in_family_order(self):
        assert list(ARMS) == [
            "null_closed_book", "null_vocabulary", "null_random",
            "bm25", "dense", "synapse_d", "synapse_lean", "ppr",
        ]
        families = [entry["family"] for entry in arm_catalog()]
        assert families == sorted(families, key=["null", "passage", "graph"].index)

    def test_every_arm_satisfies_the_protocol_and_cites_a_source(self):
        for arm in ARMS.values():
            assert isinstance(arm, Arm)
            assert arm.source["citation"] and arm.source["url"].startswith("https://")
            assert arm.retrieval_llm_calls == 0
            assert arm.title and arm.description

    def test_catalog_shape(self):
        entry = next(e for e in arm_catalog() if e["name"] == "null_random")
        assert set(entry) >= {
            "name", "family", "title", "description", "source", "retrieval_llm_calls",
            "needs_graph", "is_null",
        }
        assert entry["is_null"] is True and entry["family_title"] == "Evidence floors"

    def test_needs_graph(self):
        needs = {name for name, arm in ARMS.items() if arm.needs_graph}
        assert needs == {"null_vocabulary", "synapse_d", "synapse_lean", "ppr"}

    def test_style_arms_say_what_they_are(self):
        assert "without the LLM filter" in ARMS["ppr"].description
        assert "2502.14802" in ARMS["ppr"].source["citation"]
        lean = ARMS["synapse_lean"].source["citation"]
        assert "2502.14902" in lean and "2609.10239" in lean

    def test_unknown_arm(self):
        with pytest.raises(KeyError):
            get_arm("nope")

    def test_config_hash_is_stable_and_knob_sensitive(self, monkeypatch):
        lean = ARMS["synapse_lean"]
        before = lean.config_hash()
        assert before == lean.config_hash()
        monkeypatch.setattr(arms, "LEAN_THETA", 0.5)
        assert lean.config_hash() != before


# ── Floors ───────────────────────────────────────────────────────────────────
class TestFloors:
    async def test_closed_book_is_empty(self):
        ev = await ARMS["null_closed_book"].retrieve("anything", k=8, seed=1)
        assert ev.units == [] and ev.arm == "null_closed_book"

    async def test_vocabulary_null_ignores_the_question_and_is_cached(self, fake_neo4j):
        def handler(q, p):
            if (fp := fingerprint_answer(q)) is not None:
                return fp
            if q == arms.ENTITY_NAMES_QUERY:
                return [{"name": "Babbage"}, {"name": "Ada"}, {"name": "Turing"}, {"name": "Ada"}]
            return []

        calls = fake_neo4j(handler)
        one = await ARMS["null_vocabulary"].retrieve("who?", k=8, seed=1)
        two = await ARMS["null_vocabulary"].retrieve("something else entirely", k=2, seed=9)
        assert one.units == two.units
        (unit,) = one.units
        assert unit.kind == "name_list" and unit.text == "Ada\nBabbage\nTuring"
        assert sum(1 for q, _ in calls if q == arms.ENTITY_NAMES_QUERY) == 1

    async def test_vocabulary_cache_follows_the_graph(self, fake_neo4j):
        state = {"n": 1}

        def handler(q, p):
            if (fp := fingerprint_answer(q, entities=state["n"])) is not None:
                return fp
            if q == arms.ENTITY_NAMES_QUERY:
                return [{"name": f"E{i}"} for i in range(state["n"])]
            return []

        fake_neo4j(handler)
        first = await ARMS["null_vocabulary"].retrieve("q", k=1, seed=1)
        state["n"] = 2
        second = await ARMS["null_vocabulary"].retrieve("q", k=1, seed=1)
        assert first.units[0].text == "E0" and second.units[0].text == "E0\nE1"

    def _chunk_handler(self, n=20):
        ids = [f"c{i:02d}" for i in range(n)]

        def handler(q, p):
            if (fp := fingerprint_answer(q, chunks=n)) is not None:
                return fp
            if q == arms.CHUNK_IDS_QUERY:
                return [{"id": i} for i in reversed(ids)]
            if q == arms.CHUNKS_BY_ID_QUERY:
                return [{"id": i, "text": f"text of {i}", "document": f"doc-{i}", "index": 0}
                        for i in p["ids"]]
            return []

        return handler

    async def test_random_context_is_seeded_by_seed_and_question(self, fake_neo4j):
        fake_neo4j(self._chunk_handler())
        arm = ARMS["null_random"]
        a = await arm.retrieve("question one", k=5, seed=7)
        b = await arm.retrieve("question one", k=5, seed=7)
        c = await arm.retrieve("question two", k=5, seed=7)
        d = await arm.retrieve("question one", k=5, seed=8)
        assert [u.text for u in a.units] == [u.text for u in b.units]
        assert len(a.units) == 5 and all(u.kind == "prose" for u in a.units)
        assert [u.text for u in a.units] != [u.text for u in c.units]
        assert [u.text for u in a.units] != [u.text for u in d.units]
        # Ranked by draw order, provenance is the document.
        assert [u.score for u in a.units] == sorted((u.score for u in a.units), reverse=True)
        assert all(u.source_id.startswith("doc-") for u in a.units)

    async def test_random_context_matches_its_documented_rng(self, fake_neo4j):
        fake_neo4j(self._chunk_handler())
        ev = await ARMS["null_random"].retrieve("q", k=3, seed=1)
        ids = sorted(f"c{i:02d}" for i in range(20))
        expected = random.Random(arms.question_seed(1, "q")).sample(ids, 3)
        assert [u.text for u in ev.units] == [f"text of {i}" for i in expected]

    async def test_random_context_caps_at_the_pool(self, fake_neo4j):
        fake_neo4j(self._chunk_handler(n=2))
        ev = await ARMS["null_random"].retrieve("q", k=10, seed=1)
        assert len(ev.units) == 2


# ── Passage baselines ────────────────────────────────────────────────────────
class TestPassages:
    async def test_bm25_queries_the_chunk_fulltext_index_with_escaped_terms(self, fake_neo4j):
        rows = [
            {"id": "b", "text": "second", "document": "D2", "index": 0, "score": 1.0},
            {"id": "a", "text": "first", "document": "D1", "index": 0, "score": 2.0},
            {"id": "c", "text": "tie", "document": "D3", "index": 0, "score": 1.0},
        ]
        calls = fake_neo4j(lambda q, p: rows if q == arms.BM25_QUERY else [])
        question = "Who wrote C++ (the language)?"
        ev = await ARMS["bm25"].retrieve(question, k=2, seed=0)
        (query, params), = [(q, p) for q, p in calls if q == arms.BM25_QUERY]
        assert params["index"] == CHUNK_FULLTEXT_INDEX
        assert params["q"] == chat_engine._keyword_query(question)
        assert "(" not in params["q"] and "+" not in params["q"]
        assert [u.text for u in ev.units] == ["first", "second"]  # score desc, id tie-break
        assert ev.units[0].source_id == "D1" and ev.units[0].kind == "prose"

    async def test_bm25_with_only_stopwords_does_not_query(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        ev = await ARMS["bm25"].retrieve("who is the", k=5, seed=0)
        assert ev.units == [] and calls == []

    async def test_dense_ranks_the_vector_hits(self, monkeypatch):
        async def search(question, k):
            assert k == 3
            return [
                {"id": "x", "text": "low", "document": "D", "index": 0, "score": 0.1},
                {"id": "y", "text": "high", "document": "E", "index": 1, "score": 0.9},
            ]

        monkeypatch.setattr(chunk_store, "search_chunks", search)
        ev = await ARMS["dense"].retrieve("q", k=3, seed=0)
        assert [(u.text, u.source_id) for u in ev.units] == [("high", "E"), ("low", "D")]


# ── Synapse-D (the shipped path) ─────────────────────────────────────────────
LOCAL_CONTEXT = (
    "Entity: Ada (Type: Person)\n  Description: mathematician\n  Relationships:\n"
    "  → KNEW → Babbage (Person)\n\n"
    "Entity: Babbage (Type: Person)\n  Description: engineer"
    f"\n\n{chat_engine.PATHS_HEADING}\n  - Ada -[KNEW]-> Babbage"
    f"\n\n{chat_engine.SOURCES_HEADING}\n\n[S1] doc1 (chunk 0)\nAda wrote notes.\n\n"
    "[S2] doc2 (chunk 3)\nBabbage built engines."
)
LOCAL_CITATIONS = [
    {"name": "Ada", "type": "Person", "kind": "entity"},
    {"name": "Babbage", "type": "Person", "kind": "entity"},
]
LOCAL_SOURCES = [
    {"id": "c1", "text": "Ada wrote notes.", "document": "doc1", "index": 0},
    {"id": "c2", "text": "Babbage built engines.", "document": "doc2", "index": 3},
]


class TestSynapseD:
    def test_explodes_a_local_retrieval_faithfully(self):
        retrieval = chat_engine.Retrieval(
            LOCAL_CONTEXT, LOCAL_CITATIONS, [{"text": "Ada -[KNEW]-> Babbage"}], "local",
            LOCAL_SOURCES,
        )
        units, meta = arms.explode_retrieval(retrieval)
        assert [(u.kind, u.source_id) for u in units] == [
            ("entity", "Ada"), ("entity", "Babbage"), ("path", None),
            ("prose", "doc1"), ("prose", "doc2"),
        ]
        assert units[0].text.startswith("Entity: Ada") and "→ KNEW → Babbage" in units[0].text
        assert units[2].text == "Ada -[KNEW]-> Babbage"
        scores = [u.score for u in units]
        assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores)
        assert meta == {"mode": "local", "seeds": ["Ada", "Babbage"], "paths": 1,
                        "excerpts": 2, "shipped_context_chars": len(LOCAL_CONTEXT)}

    def test_unbounded_pack_carries_every_piece_of_the_shipped_context(self):
        retrieval = chat_engine.Retrieval(LOCAL_CONTEXT, LOCAL_CITATIONS, [], "local",
                                          LOCAL_SOURCES)
        units, _ = arms.explode_retrieval(retrieval)
        packed = pack(units, None, "gpt-5-nano")
        for piece in ("Description: mathematician", "→ KNEW → Babbage (Person)",
                      "Ada -[KNEW]-> Babbage", "Ada wrote notes.", "Babbage built engines."):
            assert piece in packed.text

    def test_explodes_a_global_retrieval_into_community_units(self):
        context = (
            f"{chat_engine.COMMUNITY_HEADING}\n\nCommunity: Computing (3 entities)\n  "
            "Summary: early computing\n  Key members: Ada, Babbage"
        )
        citations = [{"name": "Computing", "kind": "community", "id": "comm-1"}]
        retrieval = chat_engine.Retrieval(context, citations, [], "global", [])
        units, meta = arms.explode_retrieval(retrieval)
        assert [(u.kind, u.source_id) for u in units] == [("community", "comm-1")]
        assert meta["mode"] == "global"

    def test_the_no_seed_sentinel_is_not_evidence(self):
        retrieval = chat_engine.Retrieval(
            "No relevant information found in the knowledge graph.", [], [], "local", []
        )
        assert arms.explode_retrieval(retrieval)[0] == []

    async def test_calls_the_shipped_path_unmodified(self, monkeypatch):
        seen = {}

        async def shipped(question, k=8):
            seen["args"] = (question, k)
            return chat_engine.Retrieval(LOCAL_CONTEXT, LOCAL_CITATIONS, [], "local",
                                         LOCAL_SOURCES)

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", shipped)
        ev = await ARMS["synapse_d"].retrieve("who knew Babbage?", k=5, seed=0)
        assert seen["args"] == ("who knew Babbage?", 5)
        assert len(ev.units) == 5 and ev.arm == "synapse_d"


# ── Synapse-Lean ─────────────────────────────────────────────────────────────
def edge(a, b, rel="R"):
    return {"source": a, "target": b, "rel": rel}


DIAMOND = [edge("A", "H"), edge("H", "C"), edge("A", "X"), edge("X", "C")]


class TestLeanScoring:
    def test_flow_reliability_and_hub_penalty_are_exact(self):
        paths = arms.score_paths(["A", "C"], DIAMOND, {"H": 500, "X": 2}, theta=0.0)
        via_x = next(p for p in paths if p.nodes == ("A", "X", "C"))
        # A has 2 neighbours: 0.7 / 2 = 0.35 reaches X; X has 2: 0.7 * 0.35 / 2 reaches C.
        assert via_x.reliability == pytest.approx((0.35 + 0.1225) / 2)
        assert via_x.hub == pytest.approx(1 / (1 + math.log(3)))
        assert via_x.score == pytest.approx(via_x.reliability * via_x.hub)
        assert via_x.text == "A -[R]-> X -[R]-> C"

    def test_the_hub_path_ranks_below_the_equal_flow_non_hub_path(self):
        paths = arms.score_paths(["A", "C"], DIAMOND, {"H": 500, "X": 2}, theta=0.0)
        assert [p.nodes for p in paths] == [("A", "X", "C"), ("A", "H", "C")]
        assert paths[0].reliability == pytest.approx(paths[1].reliability)

    def test_theta_prunes_weak_branches(self):
        # 0.1225 at the second hop is below theta 0.2, so no two-hop path survives.
        assert arms.score_paths(["A", "C"], DIAMOND, {}, theta=0.2) == []

    def test_a_direct_edge_has_no_hub_penalty(self):
        (path,) = arms.score_paths(["A", "B"], [edge("A", "B", "KNEW")], {}, theta=0.0)
        assert path.hub == 1.0 and path.reliability == pytest.approx(0.7)
        assert path.text == "A -[KNEW]-> B"

    def test_each_seed_pair_is_scored_once_from_the_better_seed(self):
        paths = arms.score_paths(["C", "A"], DIAMOND, {}, theta=0.0)
        assert all(p.nodes[0] == "C" for p in paths)

    def test_top_k_and_determinism(self):
        edges = [edge("S", f"M{i}") for i in range(6)] + [edge(f"M{i}", "T") for i in range(6)]
        shuffled = list(edges)
        random.Random(3).shuffle(shuffled)
        a = arms.score_paths(["S", "T"], edges, {}, theta=0.0, top_k=4)
        b = arms.score_paths(["S", "T"], shuffled, {}, theta=0.0, top_k=4)
        assert a == b and len(a) == 4

    def test_path_length_is_bounded(self):
        chain = [edge("A", "B"), edge("B", "C"), edge("C", "D"), edge("D", "E"), edge("E", "F")]
        paths = arms.score_paths(["A", "F"], chain, {}, theta=0.0, max_edges=4)
        assert paths == []
        assert arms.score_paths(["A", "F"], chain, {}, theta=0.0, max_edges=5)

    def test_parallel_relations_collapse_to_one_edge(self):
        edges = [edge("A", "B", "Z_REL"), edge("A", "B", "A_REL")]
        (path,) = arms.score_paths(["A", "B"], edges, {}, theta=0.0)
        assert path.rels == ("A_REL",)

    def test_units_share_one_score_scale(self):
        paths = arms.score_paths(["A", "C"], DIAMOND, {"H": 500, "X": 2}, theta=0.0, top_k=1)
        names = arms.excerpt_anchors(["A", "C"], paths)
        # Every node on the best path inherits its score (1.0): ties sort by name.
        assert names == ["A", "C", "X"]
        info = {"A": {"type": "Person", "description": "a"}, "C": {"type": "Org"}}
        chunks = [{"text": "about X", "document": "dx", "best_seed_rank": 2}]
        units = arms.lean_units(["A", "C"], info, paths, chunks, names)
        by_kind = {(u.kind, u.source_id or u.text): u.score for u in units}
        assert by_kind[("entity", "A")] == 1.0 and by_kind[("entity", "C")] == 0.5
        assert by_kind[("path", "A -[R]-> X -[R]-> C")] == pytest.approx(1.0)
        assert by_kind[("prose", "dx")] == pytest.approx(0.9)

    def test_no_path_means_seed_excerpts_only(self):
        assert arms.excerpt_anchors(["A", "B"], []) == ["A", "B"]


class TestLeanArm:
    async def test_retrieve_wires_the_pieces(self, monkeypatch, fake_neo4j):
        async def seeds(question, k):
            return ["A", "C"]

        async def neighbourhood(names, hops):
            assert hops == arms.LEAN_HOPS
            return list(DIAMOND)

        pulled = {}

        async def chunks_for(names, limit):
            pulled["names"], pulled["limit"] = list(names), limit
            return [{"id": "c1", "text": "X founded C.", "document": "dx", "index": 0,
                     "seed_mentions": 1, "best_seed_rank": 1}]

        def handler(q, p):
            if q == arms.ENTITY_INFO_QUERY:
                degrees = {"A": 2, "C": 2, "H": 500, "X": 2}
                return [{"name": n, "type": "T", "description": f"about {n}",
                         "degree": degrees[n]} for n in p["names"]]
            return []

        monkeypatch.setattr(arms, "_seed_names", seeds)
        monkeypatch.setattr(chat_engine, "_neighborhood_edges", neighbourhood)
        monkeypatch.setattr(chunk_store, "chunks_for_entities", chunks_for)
        fake_neo4j(handler)
        ev = await ARMS["synapse_lean"].retrieve("q", k=4, seed=0)
        kinds = [u.kind for u in ev.units]
        assert kinds == ["entity", "entity", "path", "path", "prose"]
        assert pulled["limit"] == 4 and pulled["names"] == ["A", "C", "X", "H"]
        assert ev.meta["paths"] == 2 and ev.meta["seeds"] == ["A", "C"]

    async def test_no_seed_no_evidence(self, monkeypatch):
        async def none(question, k):
            return []

        monkeypatch.setattr(arms, "_seed_names", none)
        ev = await ARMS["synapse_lean"].retrieve("q", k=4, seed=0)
        assert ev.units == []


# ── PPR ──────────────────────────────────────────────────────────────────────
RELATIONS = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"},
             {"source": "D", "target": "E"}]
MENTIONS = [{"entity": "A", "chunk": "c1"}, {"entity": "C", "chunk": "c2"},
            {"entity": "D", "chunk": "c3"}]
CHUNKS = [{"id": f"c{i}", "text": f"text {i}", "document": f"doc{i}", "index": 0}
          for i in (1, 2, 3)]


class TestPPR:
    def test_graph_shape(self):
        ppr = arms.build_ppr_graph(RELATIONS, MENTIONS, CHUNKS)
        g = ppr.graph
        assert g.has_edge("e:A", "c:c1") and g.has_edge("e:A", "e:B")
        assert len(ppr.members) == 2  # {A,B,C,c1,c2} and {D,E,c3}

    def test_passages_ranked_by_mass_within_the_seed_component(self):
        ppr = arms.build_ppr_graph(RELATIONS, MENTIONS, CHUNKS)
        ranked = arms.ppr_rank_chunks(ppr, ["A"], k=5)
        assert [cid for cid, _ in ranked] == ["c1", "c2"]  # c3 is unreachable
        assert ranked[0][1] > ranked[1][1] > 0

    def test_matches_pagerank_on_the_full_graph(self):
        ppr = arms.build_ppr_graph(RELATIONS, MENTIONS, CHUNKS)
        # The pure-Python reference, not nx.pagerank: networkx 3's pagerank needs SciPy,
        # which is not in requirements.txt (CI and the Docker image run without it).
        from networkx.algorithms.link_analysis import pagerank_alg

        full = pagerank_alg._pagerank_python(
            ppr.graph, alpha=0.5, personalization={"e:A": 1.0, "e:C": 0.5}, weight="weight"
        )
        expected = sorted(((n[2:], m) for n, m in full.items() if n.startswith("c:") and m > 1e-6),
                          key=lambda x: (-x[1], x[0]))
        ranked = arms.ppr_rank_chunks(ppr, ["A", "C"], k=5)
        assert [c for c, _ in ranked] == [c for c, _ in expected]

    def test_pure_python_fallback_when_scipy_is_missing(self, monkeypatch):
        ppr = arms.build_ppr_graph(RELATIONS, MENTIONS, CHUNKS)
        reference = arms.ppr_rank_chunks(ppr, ["A"], k=5)

        def no_scipy(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'scipy'")

        monkeypatch.setattr(nx, "pagerank", no_scipy)
        fallback = arms.ppr_rank_chunks(ppr, ["A"], k=5)
        assert [c for c, _ in fallback] == [c for c, _ in reference]
        assert fallback[0][1] == pytest.approx(reference[0][1], rel=1e-3)

    def test_unknown_seeds_yield_nothing(self):
        ppr = arms.build_ppr_graph(RELATIONS, MENTIONS, CHUNKS)
        assert arms.ppr_rank_chunks(ppr, ["Nobody"], k=5) == []

    async def test_arm_builds_the_graph_once_per_fingerprint(self, monkeypatch, fake_neo4j):
        async def seeds(question, k):
            return ["C"]

        def handler(q, p):
            if (fp := fingerprint_answer(q)) is not None:
                return fp
            return {
                arms.PPR_RELATIONS_QUERY: RELATIONS,
                arms.PPR_MENTIONS_QUERY: MENTIONS,
                arms.PPR_CHUNKS_QUERY: CHUNKS,
            }.get(q, [])

        monkeypatch.setattr(arms, "_seed_names", seeds)
        calls = fake_neo4j(handler)
        first = await ARMS["ppr"].retrieve("q1", k=1, seed=0)
        await ARMS["ppr"].retrieve("q2", k=1, seed=0)
        assert sum(1 for q, _ in calls if q == arms.PPR_RELATIONS_QUERY) == 1
        (unit,) = first.units
        assert (unit.text, unit.source_id, unit.kind) == ("text 2", "doc2", "prose")


async def test_seed_names_use_the_shipped_hybrid_ranker(monkeypatch):
    async def ranker(question, k):
        return ["a", "b", "c", "d"]

    monkeypatch.setattr(chat_engine, "_rank_seed_names", ranker)
    assert await arms._seed_names("q", 2) == ["a", "b"]
