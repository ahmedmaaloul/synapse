# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for entity resolution: clustering, canonical choice, Cypher merge, ingest wiring.

Fully hermetic — no network, no LLM, no Neo4j. Embeddings are hand-written unit
vectors so every cosine similarity in these tests is exact and obvious.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from app.config import Settings
from app.services import entity_resolution as er
from app.services import graph_builder

# Orthogonal unit vectors: SAME·SAME = 1.0, SAME·OTHER = 0.0, SAME·NEAR ≈ 0.995.
SAME = [1.0, 0.0, 0.0, 0.0]
NEAR = [0.995, 0.0998, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0, 0.0]


def _settings(**overrides) -> Settings:
    """Settings with the shipped resolution defaults unless a test overrides them."""
    base = {
        "entity_resolution_enabled": True,
        "entity_resolution_threshold": 0.93,
        "entity_resolution_name_threshold": 0.87,
    }
    return Settings(**{**base, **overrides})


class TestNormalization:
    def test_strips_case_punctuation_and_whitespace(self):
        assert er._normalize("  Ahmed-Maaloul!  ") == "ahmedmaaloul"
        assert er._normalize("PostgreSQL") == "postgresql"

    def test_strips_accents(self):
        assert er._normalize("Café") == er._normalize("Cafe")

    def test_empty(self):
        assert er._normalize("") == ""
        assert er._normalize("!!!") == ""

    def test_tokens(self):
        assert er._tokens("Ahmed  Maaloul, PhD") == ["ahmed", "maaloul", "phd"]


class TestNameSimilarity:
    def test_identical_after_normalization(self):
        assert er.name_similarity("PostgreSQL", "postgre sql") == 1.0

    def test_token_prefix_is_a_strong_match(self):
        # "Ahmed" is a token subsequence of "Ahmed Maaloul".
        assert er.name_similarity("Ahmed", "Ahmed Maaloul") >= er.CONTAINMENT_SCORE

    def test_string_prefix_is_a_strong_match(self):
        assert er.name_similarity("Postgres", "PostgreSQL") >= er.CONTAINMENT_SCORE

    def test_symmetric(self):
        assert er.name_similarity("Postgres", "PostgreSQL") == er.name_similarity(
            "PostgreSQL", "Postgres"
        )

    def test_short_prefix_does_not_count_as_containment(self):
        # "Java" covers only 40% of "JavaScript" — the classic false positive.
        assert er.name_similarity("Java", "JavaScript") < 0.87

    def test_tiny_names_never_win_on_containment(self):
        assert er.name_similarity("AI", "AI Platform Engineering") < 0.87

    def test_unrelated_names_score_low(self):
        assert er.name_similarity("Ahmed Maaloul", "Neo4j") < 0.5

    def test_empty_name(self):
        assert er.name_similarity("", "Ahmed") == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        assert er.cosine_similarity(SAME, SAME) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert er.cosine_similarity(SAME, OTHER) == pytest.approx(0.0)

    def test_length_mismatch_or_empty_is_zero(self):
        assert er.cosine_similarity([1.0, 0.0], SAME) == 0.0
        assert er.cosine_similarity([], SAME) == 0.0
        assert er.cosine_similarity([0.0, 0.0, 0.0, 0.0], SAME) == 0.0


class TestFindDuplicateClusters:
    def test_ahmed_and_ahmed_maaloul_merge(self):
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "PERSON"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Ahmed": SAME, "Ahmed Maaloul": NEAR}, settings=_settings()
        )
        assert clusters == [["Ahmed Maaloul", "Ahmed"]]

    def test_postgres_and_postgresql_merge(self):
        entities = [
            {"name": "Postgres", "type": "TOOL"},
            {"name": "PostgreSQL", "type": "TOOL"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Postgres": SAME, "PostgreSQL": NEAR}, settings=_settings()
        )
        assert clusters == [["PostgreSQL", "Postgres"]]

    def test_different_types_never_merge(self):
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "TOOL"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Ahmed": SAME, "Ahmed Maaloul": SAME}, settings=_settings()
        )
        assert clusters == []

    def test_missing_type_is_its_own_bucket(self):
        entities = [{"name": "Ahmed"}, {"name": "Ahmed Maaloul", "type": "PERSON"}]
        clusters = er.find_duplicate_clusters(
            entities, {"Ahmed": SAME, "Ahmed Maaloul": SAME}, settings=_settings()
        )
        assert clusters == []

    def test_low_vector_similarity_does_not_merge(self):
        """Names look alike but the embeddings disagree — one signal is not a quorum."""
        entities = [
            {"name": "Postgres", "type": "TOOL"},
            {"name": "PostgreSQL", "type": "TOOL"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Postgres": SAME, "PostgreSQL": OTHER}, settings=_settings()
        )
        assert clusters == []

    def test_low_name_similarity_does_not_merge(self):
        """Embeddings agree perfectly but the names do not — still no merge."""
        entities = [
            {"name": "Kubernetes", "type": "TOOL"},
            {"name": "Kafka", "type": "TOOL"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Kubernetes": SAME, "Kafka": SAME}, settings=_settings()
        )
        assert clusters == []

    def test_entity_without_embedding_never_merges(self):
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "PERSON"},
        ]
        clusters = er.find_duplicate_clusters(entities, {"Ahmed": SAME}, settings=_settings())
        assert clusters == []

    def test_transitive_clustering(self):
        """A~B and B~C must land in ONE cluster, even though A and C are not compared."""
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "PERSON"},
            {"name": "Ahmed Maaloul PhD", "type": "PERSON"},
        ]
        embeddings = {"Ahmed": SAME, "Ahmed Maaloul": SAME, "Ahmed Maaloul PhD": SAME}
        clusters = er.find_duplicate_clusters(entities, embeddings, settings=_settings())
        assert clusters == [["Ahmed Maaloul PhD", "Ahmed", "Ahmed Maaloul"]]

    def test_two_independent_clusters(self):
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "PERSON"},
            {"name": "Postgres", "type": "TOOL"},
            {"name": "PostgreSQL", "type": "TOOL"},
            {"name": "Neo4j", "type": "TOOL"},
        ]
        embeddings = {
            "Ahmed": SAME,
            "Ahmed Maaloul": SAME,
            "Postgres": SAME,
            "PostgreSQL": SAME,
            "Neo4j": OTHER,
        }
        clusters = er.find_duplicate_clusters(entities, embeddings, settings=_settings())
        assert clusters == [["Ahmed Maaloul", "Ahmed"], ["PostgreSQL", "Postgres"]]

    def test_deterministic_under_input_reordering(self):
        entities = [
            {"name": "Ahmed", "type": "PERSON"},
            {"name": "Ahmed Maaloul", "type": "PERSON"},
            {"name": "Postgres", "type": "TOOL"},
            {"name": "PostgreSQL", "type": "TOOL"},
        ]
        embeddings = dict.fromkeys([e["name"] for e in entities], SAME)
        first = er.find_duplicate_clusters(entities, embeddings, settings=_settings())
        second = er.find_duplicate_clusters(
            list(reversed(entities)), embeddings, settings=_settings()
        )
        assert first == second

    def test_reordered_tokens_still_block_together(self):
        entities = [
            {"name": "Ahmed Maaloul", "type": "PERSON"},
            {"name": "Maaloul", "type": "PERSON"},
        ]
        clusters = er.find_duplicate_clusters(
            entities, {"Ahmed Maaloul": SAME, "Maaloul": SAME}, settings=_settings()
        )
        assert clusters == [["Ahmed Maaloul", "Maaloul"]]

    def test_threshold_is_honoured(self):
        entities = [
            {"name": "Postgres", "type": "TOOL"},
            {"name": "PostgreSQL", "type": "TOOL"},
        ]
        embeddings = {"Postgres": SAME, "PostgreSQL": NEAR}  # cosine ≈ 0.995
        assert (
            er.find_duplicate_clusters(entities, embeddings, settings=_settings(entity_resolution_threshold=0.999))
            == []
        )
        assert er.find_duplicate_clusters(
            entities, embeddings, settings=_settings(entity_resolution_threshold=0.99)
        ) == [["PostgreSQL", "Postgres"]]

    def test_empty_inputs(self):
        assert er.find_duplicate_clusters([], {}, settings=_settings()) == []
        assert er.find_duplicate_clusters([{"name": ""}], {}, settings=_settings()) == []


class TestChooseCanonical:
    def test_prefers_the_longer_name(self):
        assert er.choose_canonical(["Ahmed", "Ahmed Maaloul"]) == "Ahmed Maaloul"
        assert er.choose_canonical(["Postgres", "PostgreSQL"]) == "PostgreSQL"

    def test_order_independent(self):
        assert er.choose_canonical(["Ahmed Maaloul", "Ahmed"]) == "Ahmed Maaloul"

    def test_ties_break_alphabetically(self):
        assert er.choose_canonical(["Bravo", "Alpha"]) == "Alpha"
        assert er.choose_canonical(["Alpha", "Bravo"]) == "Alpha"

    def test_ignores_blank_names(self):
        assert er.choose_canonical(["", "   ", "Ada"]) == "Ada"

    def test_empty_cluster(self):
        assert er.choose_canonical([]) == ""


class TestMergeEntityClusters:
    async def test_apoc_path_merges_and_counts(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [{"name": "Ahmed Maaloul"}])
        result = await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        assert result == {"clusters": 1, "merged": 1}
        queries = [q for q, _ in calls]
        assert any("apoc.refactor.mergeNodes" in q for q in queries)
        # The canonical keeps its identity and records the absorbed name.
        absorb = next(p for q, p in calls if "c.aliases" in q)
        assert absorb == {"canonical": "Ahmed Maaloul", "dups": ["Ahmed"]}

    async def test_apoc_call_disables_self_relationships(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        await er.merge_entity_clusters([["PostgreSQL", "Postgres"]])
        apoc = next(q for q, _ in calls if "apoc.refactor.mergeNodes" in q)
        assert "produceSelfRel: false" in apoc
        assert "mergeRels: true" in apoc

    async def test_plain_cypher_fallback_when_apoc_is_absent(self, fake_neo4j):
        def handler(query, params):
            if "apoc" in query:
                raise RuntimeError("There is no procedure with the name apoc.refactor.mergeNodes")
            if "RETURN DISTINCT type(r)" in query:
                return [{"rel_type": "USES_TOOL"}, {"rel_type": "WORKED_AT"}]
            return []

        calls = fake_neo4j(handler)
        result = await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        assert result == {"clusters": 1, "merged": 1}
        queries = [q for q, _ in calls]
        # One re-point query per direction per relationship type, then the delete.
        assert sum("MERGE (c)-[nr:USES_TOOL]->" in q for q in queries) == 1
        assert sum("MERGE (other)-[nr:USES_TOOL]->(c)" in q for q in queries) == 1
        assert sum("MERGE (c)-[nr:WORKED_AT]->" in q for q in queries) == 1
        assert sum("MERGE (other)-[nr:WORKED_AT]->(c)" in q for q in queries) == 1
        assert any("DETACH DELETE d" in q for q in queries)

    async def test_fallback_rewiring_avoids_self_loops(self, fake_neo4j):
        def handler(query, params):
            if "apoc" in query:
                raise RuntimeError("no apoc")
            if "RETURN DISTINCT type(r)" in query:
                return [{"rel_type": "RELATED_TO"}]
            return []

        calls = fake_neo4j(handler)
        await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])
        rewire = [q for q, _ in calls if "MERGE (c)-[nr:RELATED_TO]->" in q][0]
        # Other cluster members are excluded by name, the canonical by identity.
        assert "coalesce(other.name, '') IN $dups" in rewire
        assert "other <> c" in rewire

    async def test_fallback_rejects_unsafe_relationship_types(self, fake_neo4j):
        """A relationship type is interpolated into Cypher, so it must be validated."""

        def handler(query, params):
            if "apoc" in query:
                raise RuntimeError("no apoc")
            if "RETURN DISTINCT type(r)" in query:
                return [{"rel_type": "EVIL] -> () DETACH DELETE n //"}, {"rel_type": "OK_TYPE"}]
            return []

        calls = fake_neo4j(handler)
        await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])
        queries = [q for q, _ in calls]
        assert not any("EVIL" in q for q in queries)
        assert any("MERGE (c)-[nr:OK_TYPE]->" in q for q in queries)

    async def test_singleton_and_empty_clusters_are_ignored(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        result = await er.merge_entity_clusters([["Ahmed"], [], ["  ", ""]])
        assert result == {"clusters": 0, "merged": 0}
        assert calls == []

    async def test_failing_cluster_is_skipped_not_fatal(self, fake_neo4j):
        def handler(query, params):
            if params.get("canonical") == "Broken Entity":
                raise RuntimeError("Neo4j is on fire")
            return []

        fake_neo4j(handler)
        result = await er.merge_entity_clusters(
            [["Broken Entity", "Broken"], ["PostgreSQL", "Postgres"]]
        )
        assert result == {"clusters": 1, "merged": 1}

    async def test_counts_every_duplicate_in_a_cluster(self, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        result = await er.merge_entity_clusters([["Ahmed Maaloul PhD", "Ahmed", "Ahmed Maaloul"]])
        assert result == {"clusters": 1, "merged": 2}


def _no_apoc_handler(rel_types: list[str]):
    """Fake driver that forces the plain-Cypher path and reports ``rel_types``.

    It also mimics the real driver's refusal to run a query that references a
    parameter it was not given, so a query/param mismatch fails the test instead
    of passing silently.
    """

    def handler(query, params):
        if "apoc" in query:
            raise RuntimeError("There is no procedure with the name apoc.refactor.mergeNodes")
        for name in ("canonical", "dups"):
            if f"${name}" in query and name not in params:
                raise RuntimeError(f"Expected parameter(s): {name}")
        if "RETURN DISTINCT type(r)" in query:
            return [{"rel_type": t} for t in rel_types]
        return []

    return handler


class TestPlainMergeIsLossless:
    """Regressions for two silent data-loss bugs in the APOC-free fallback."""

    def test_far_endpoint_filter_is_null_safe(self):
        """An endpoint with no ``name`` property must survive the rewire.

        Cypher is three-valued: ``null IN [...]`` is ``null``, ``NOT null`` is
        ``null``, and ``WHERE`` keeps only rows that are *true*. The original
        ``NOT other.name IN $dups AND other.name <> $canonical`` therefore
        skipped **every** relationship whose far endpoint has no ``name`` — a
        ``:Community`` node, for instance — and the ``DETACH DELETE`` that
        follows then destroyed those edges for good.
        """
        for query in er._rewire_queries("IN_COMMUNITY"):
            assert "NOT other.name IN $dups" not in query
            assert "other.name <> $canonical" not in query
            # Nameless nodes coalesce to '' — never a cluster member, so kept.
            assert "coalesce(other.name, '') IN $dups" in query
            # The canonical is excluded by identity, which cannot go null.
            assert "other <> c" in query

    def test_far_endpoint_filter_is_label_aware(self):
        """Only an ``:Entity`` can be a member of the cluster being merged."""
        for query in er._rewire_queries("USES_TOOL"):
            assert "other:Entity AND coalesce(other.name, '') IN $dups" in query

    async def test_edge_to_a_nameless_node_is_rewired_before_the_delete(self, fake_neo4j):
        """``IN_COMMUNITY`` edges follow the duplicate onto the canonical node.

        They point at ``(:Community)`` nodes, which carry no ``name``. Both
        directions must be re-pointed, and both must happen *before* the
        duplicates are detached.
        """
        calls = fake_neo4j(_no_apoc_handler(["IN_COMMUNITY"]))
        await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        queries = [q for q, _ in calls]
        outgoing = [i for i, q in enumerate(queries) if "MERGE (c)-[nr:IN_COMMUNITY]->" in q]
        incoming = [i for i, q in enumerate(queries) if "MERGE (other)-[nr:IN_COMMUNITY]->(c)" in q]
        delete = [i for i, q in enumerate(queries) if "DETACH DELETE d" in q]
        assert len(outgoing) == len(incoming) == len(delete) == 1
        assert max(outgoing[0], incoming[0]) < delete[0]

    async def test_delete_is_gated_on_the_canonical_existing(self, fake_neo4j):
        """Duplicates must not be deleted when there is nothing to merge into.

        Every rewire query starts with ``MATCH (c:Entity {name: $canonical})``,
        which matches zero rows *without raising* when the canonical node is
        absent. An unguarded ``DETACH DELETE`` then wiped the duplicates and all
        their relationships after nothing at all had been re-pointed.
        """
        calls = fake_neo4j(_no_apoc_handler(["USES_TOOL"]))
        await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        delete_query, delete_params = next((q, p) for q, p in calls if "DETACH DELETE d" in q)
        assert delete_query.strip().startswith("MATCH (c:Entity {name: $canonical})")
        # ...and the canonical is actually supplied, or the query would error.
        assert delete_params == {"canonical": "Ahmed Maaloul", "dups": ["Ahmed"]}

    async def test_rewires_and_delete_share_the_same_canonical_gate(self, fake_neo4j):
        """No write in the fallback may run unless the canonical node exists."""
        calls = fake_neo4j(_no_apoc_handler(["USES_TOOL", "WORKED_AT"]))
        await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        writes = [q for q, _ in calls if "MERGE (" in q or "DETACH DELETE" in q]
        assert writes  # the fallback really did run
        assert all("MATCH (c:Entity {name: $canonical})" in q for q in writes)


class TestUnsafeRelationshipTypeNeverCausesDeletion:
    """A type the safety guard rejects must block the delete, not be dropped.

    ``_SAFE_REL_TYPE`` exists because Neo4j < 5.26 cannot parameterize a
    relationship type, so the type is interpolated into the rewire query — that
    guard is correct and stays. The bug was what happened *next*: rejected types
    were silently skipped and ``DETACH DELETE`` ran anyway, so those edges were
    destroyed with nothing re-pointed. The fix keeps the duplicate instead.
    """

    UNSAFE = "EVIL] -> () DETACH DELETE n //"

    async def test_duplicate_is_not_deleted_and_its_edge_survives(self, fake_neo4j):
        calls = fake_neo4j(_no_apoc_handler(["USES_TOOL", self.UNSAFE]))
        result = await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        queries = [q for q, _ in calls]
        # The unsafe type is still never interpolated into Cypher...
        assert not any("EVIL" in q for q in queries)
        # ...the safe type is still re-pointed...
        assert any("MERGE (c)-[nr:USES_TOOL]->" in q for q in queries)
        # ...but the duplicate survives, so the un-rewired edge survives with it.
        assert not any("DETACH DELETE d" in q for q in queries)
        # Nothing was absorbed, so nothing may be reported as merged.
        assert result == {"clusters": 0, "merged": 0}

    async def test_warning_names_the_entity_and_the_offending_type(self, fake_neo4j, caplog):
        """Silent is the failure mode we are fixing — the log must be actionable."""
        fake_neo4j(_no_apoc_handler(["WEIRD-TYPE"]))
        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "Ahmed Maaloul" in logged
        assert "WEIRD-TYPE" in logged

    async def test_only_the_offending_cluster_is_held_back(self, fake_neo4j):
        """One un-mergeable cluster must not stop the clean ones."""

        def handler(query, params):
            if "apoc" in query:
                raise RuntimeError("no apoc")
            if "RETURN DISTINCT type(r)" in query:
                unsafe = "Ahmed" in params.get("dups", [])
                return [{"rel_type": self.UNSAFE if unsafe else "USES_TOOL"}]
            return []

        calls = fake_neo4j(handler)
        result = await er.merge_entity_clusters(
            [["Ahmed Maaloul", "Ahmed"], ["PostgreSQL", "Postgres"]]
        )

        assert result == {"clusters": 1, "merged": 1}
        deletes = [p for q, p in calls if "DETACH DELETE d" in q]
        assert [p["canonical"] for p in deletes] == ["PostgreSQL"]

    async def test_all_types_safe_still_deletes(self, fake_neo4j):
        """The guard must not make the ordinary path any more conservative."""
        calls = fake_neo4j(_no_apoc_handler(["USES_TOOL", "WORKED_AT"]))
        result = await er.merge_entity_clusters([["Ahmed Maaloul", "Ahmed"]])

        assert result == {"clusters": 1, "merged": 1}
        assert any("DETACH DELETE d" in q for q, _ in calls)


class TestFetchGraphEntities:
    async def test_reads_names_types_and_embeddings(self, fake_neo4j):
        fake_neo4j(
            lambda q, p: [
                {"name": "Postgres", "type": "TOOL", "embedding": [1, 0, 0, 0]},
                {"name": "Neo4j", "type": "TOOL", "embedding": None},
                {"name": "  ", "type": "TOOL", "embedding": None},
            ]
        )
        entities, embeddings = await er.fetch_graph_entities()
        assert [e["name"] for e in entities] == ["Postgres", "Neo4j"]
        assert embeddings == {"Postgres": [1.0, 0.0, 0.0, 0.0]}

    async def test_db_failure_is_not_fatal(self, fake_neo4j):
        def handler(query, params):
            raise RuntimeError("connection refused")

        fake_neo4j(handler)
        assert await er.fetch_graph_entities() == ([], {})


# ── Ingest pipeline integration ──────────────────────────
class _FakeEmbeddings:
    """Deterministic embedder: everything 'ahmed' shares a vector, so does 'postgre'."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if lowered.startswith("ahmed"):
                vectors.append(SAME)
            elif lowered.startswith("postgre"):
                vectors.append(OTHER)
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return vectors


_EXTRACTION = """{
  "entities": [
    {"name": "Ahmed", "type": "PERSON", "description": ""},
    {"name": "Ahmed Maaloul", "type": "PERSON", "description": "AI engineer"},
    {"name": "Postgres", "type": "TOOL", "description": "relational database"},
    {"name": "PostgreSQL", "type": "TOOL", "description": ""}
  ],
  "relationships": [
    {"source": "Ahmed", "target": "Postgres", "type": "USES_TOOL"},
    {"source": "Ahmed Maaloul", "target": "PostgreSQL", "type": "USES_TOOL"},
    {"source": "Ahmed", "target": "Ahmed Maaloul", "type": "RELATED_TO"}
  ]
}"""


class _PipeStub:
    """Stands in for a ChatPromptTemplate so ``prompt | llm`` yields a fake chain."""

    def __or__(self, _other):
        class FakeResponse:
            content = _EXTRACTION

        class FakeChain:
            async def ainvoke(self, _inputs):
                return FakeResponse()

        return FakeChain()


def _install_fake_pipeline(monkeypatch):
    monkeypatch.setattr(graph_builder, "get_chat_llm", lambda **kw: object())
    monkeypatch.setattr(graph_builder, "get_extraction_prompt", lambda theme: _PipeStub())
    monkeypatch.setattr(graph_builder, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())


class TestGraphBuilderIntegration:
    async def test_duplicates_collapse_before_anything_is_written(
        self, monkeypatch, fake_neo4j
    ):
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)

        events: list[dict] = []

        async def on_progress(event):
            events.append(event)

        result = await graph_builder.build_knowledge_graph(
            ["Ahmed Maaloul uses PostgreSQL."],
            "cv.pdf",
            theme="Personal CV / Resume",
            on_progress=on_progress,
        )

        assert result["entities_merged"] == 2
        assert result["unique_entities"] == 2
        assert result["nodes_created"] == 2

        written = [p["name"] for q, p in calls if "MERGE (n:Entity {name: $name})" in q]
        assert sorted(written) == ["Ahmed Maaloul", "PostgreSQL"]

        # The absorbed names are searchable as aliases on the surviving node.
        aliases = {p["name"]: p["aliases"] for q, p in calls if "MERGE (n:Entity" in q}
        assert aliases["Ahmed Maaloul"] == ["Ahmed"]
        assert aliases["PostgreSQL"] == ["Postgres"]

        # A progress event was emitted for the new stage.
        resolving = [e for e in events if e.get("stage") == "resolving_entities"]
        assert resolving and resolving[-1]["merged"] == 2
        assert all(e["type"] == "progress" for e in resolving)

    async def test_edges_follow_the_merged_nodes(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)

        await graph_builder.build_knowledge_graph(["Ahmed uses Postgres."], "cv.pdf")

        edges = {
            (p["source"], p["target"]) for q, p in calls if "apoc.merge.relationship" in q
        }
        # Both extracted edges now point at the canonical nodes, and the
        # Ahmed -> Ahmed Maaloul edge became a self-loop and was dropped.
        assert edges == {("Ahmed Maaloul", "PostgreSQL")}

    async def test_canonical_inherits_a_missing_description(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)

        await graph_builder.build_knowledge_graph(["Ahmed uses Postgres."], "cv.pdf")

        descriptions = {
            p["name"]: p["description"] for q, p in calls if "MERGE (n:Entity" in q
        }
        # "PostgreSQL" was extracted without one; it inherits "Postgres"'s.
        assert descriptions["PostgreSQL"] == "relational database"
        assert descriptions["Ahmed Maaloul"] == "AI engineer"

    async def test_disabled_flag_writes_every_duplicate(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)
        monkeypatch.setenv("ENTITY_RESOLUTION_ENABLED", "false")

        result = await graph_builder.build_knowledge_graph(["Ahmed uses Postgres."], "cv.pdf")

        assert result["entities_merged"] == 0
        assert result["unique_entities"] == 4
        written = [p["name"] for q, p in calls if "MERGE (n:Entity {name: $name})" in q]
        assert sorted(written) == ["Ahmed", "Ahmed Maaloul", "PostgreSQL", "Postgres"]

    async def test_merges_against_entities_from_earlier_documents(self, fake_neo4j):
        """A fresh "PostgreSQL" must absorb the "Postgres" written by a previous ingest."""

        def handler(query, params):
            if "RETURN n.name AS name, n.type AS type, n.embedding AS embedding" in query:
                return [
                    {"name": "Postgres", "type": "TOOL", "embedding": SAME},
                    {"name": "Kafka", "type": "TOOL", "embedding": OTHER},
                ]
            return []

        calls = fake_neo4j(handler)
        fresh = [{"name": "PostgreSQL", "type": "TOOL"}]

        merged = await graph_builder._resolve_against_graph(
            fresh, {"PostgreSQL": SAME}, _settings()
        )

        assert merged == 1
        apoc = [p for q, p in calls if "apoc.refactor.mergeNodes" in q]
        assert apoc == [{"canonical": "PostgreSQL", "dups": ["Postgres"]}]

    async def test_graph_pass_ignores_clusters_without_a_fresh_entity(self, fake_neo4j):
        """Two old duplicates unrelated to this document are left alone."""

        def handler(query, params):
            if "RETURN n.name AS name, n.type AS type, n.embedding AS embedding" in query:
                return [
                    {"name": "Postgres", "type": "TOOL", "embedding": SAME},
                    {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},
                ]
            return []

        calls = fake_neo4j(handler)
        merged = await graph_builder._resolve_against_graph(
            [{"name": "Kafka", "type": "TOOL"}], {"Kafka": OTHER}, _settings()
        )

        assert merged == 0
        assert not any("apoc.refactor.mergeNodes" in q for q, _ in calls)

    async def test_graph_pass_noop_on_empty_graph(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        merged = await graph_builder._resolve_against_graph(
            [{"name": "Kafka", "type": "TOOL"}], {"Kafka": SAME}, _settings()
        )
        assert merged == 0
        assert len(calls) == 1  # only the candidate fetch


def _graph_entities_handler(query, params):
    """Fake driver holding one pre-existing "Postgres" entity."""
    if "RETURN n.name AS name, n.type AS type, n.embedding AS embedding" in query:
        return [{"name": "Postgres", "type": "TOOL", "embedding": SAME}]
    return []


def _spy_on_to_thread(monkeypatch) -> list:
    """Record every callable handed to :func:`asyncio.to_thread`, then run it."""
    offloaded: list = []
    real_to_thread = asyncio.to_thread

    async def spy(func, /, *args, **kwargs):
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy)
    return offloaded


class TestClusteringNeverBlocksTheEventLoop:
    """Clustering is CPU-bound; run inline it would freeze the whole API server."""

    async def test_graph_pass_offloads_clustering_to_a_thread(self, fake_neo4j, monkeypatch):
        fake_neo4j(_graph_entities_handler)
        offloaded = _spy_on_to_thread(monkeypatch)

        merged = await graph_builder._resolve_against_graph(
            [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": SAME}, _settings()
        )

        assert merged == 1  # behaviour is unchanged by the offload
        assert er.find_duplicate_clusters in offloaded

    async def test_in_memory_collapse_is_offloaded_to_a_thread(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)
        offloaded = _spy_on_to_thread(monkeypatch)

        result = await graph_builder.build_knowledge_graph(["Ahmed uses Postgres."], "cv.pdf")

        assert result["entities_merged"] == 2  # behaviour is unchanged
        assert graph_builder._collapse_duplicate_entities in offloaded

    async def test_a_slow_cluster_pass_lets_other_coroutines_run(self, fake_neo4j, monkeypatch):
        """The real proof: a concurrent coroutine keeps ticking during clustering."""
        fake_neo4j(_graph_entities_handler)

        def slow_clusters(entities, embeddings, settings=None):
            time.sleep(0.3)
            return []

        monkeypatch.setattr(er, "find_duplicate_clusters", slow_clusters)

        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        async def resolve():
            await graph_builder._resolve_against_graph(
                [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": SAME}, _settings()
            )
            return ticks

        ticks_when_resolved, _ = await asyncio.gather(resolve(), ticker())
        # Called inline, the 0.3s sleep would pin the loop and leave this at 0.
        assert ticks_when_resolved >= 5
