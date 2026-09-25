# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for entity resolution: clustering, canonical choice, Cypher merge, ingest wiring.

Fully hermetic — no network, no LLM, no Neo4j. Embeddings are hand-written unit
vectors so every cosine similarity in these tests is exact and obvious.
"""

from __future__ import annotations

import asyncio
import functools
import heapq
import logging
import math
import operator
import random
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
        "entity_resolution_candidate_k": 25,
    }
    return Settings(**{**base, **overrides})


def _unit_or_none(vector: list[float] | None) -> list[float] | None:
    if not er.is_probe_vector(vector):
        return None
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class FakeEntityGraph:
    """The ``:Entity`` nodes of a hermetic Neo4j, with an EXACT cosine vector index.

    It answers the two reads cross-document resolution makes. The whole-graph
    scan returns the first ``$limit`` nodes. The candidate query is answered by
    brute force: each probe is scored against every indexed node the way
    Neo4j's cosine index scores, ``(1 + cos) / 2``, the top ``$k`` are kept,
    and then the query's own filters (``$min_score``, ``$known``, ``$names``)
    apply. Like the real query it answers one row: ``saturated``, the probes
    whose ``$k`` answers all cleared the floor (counted before ``$known``), and
    ``entities``. Because it is exact, a test built on it measures the
    resolution algorithm, not an index's recall.

    It records the rows each read returned and the candidate queries it
    served. It also records whether any probe had more nodes above the floor
    than ``$k`` could hold (``truncated``), which is what makes
    ``fetch_candidate_entities`` ask again with a larger k.
    """

    def __init__(self, nodes: list[dict]):
        self.nodes: dict[str, dict] = {n["name"]: n for n in nodes}
        # Like Neo4j, only usable vectors are indexed (see er.is_probe_vector).
        self._index = [
            (name, unit)
            for name, node in self.nodes.items()
            if (unit := _unit_or_none(node.get("embedding"))) is not None
        ]
        self.scan_rows = 0
        self.candidate_rows = 0
        self.candidate_queries: list[dict] = []
        self.truncated = False
        self.candidate_error: Exception | None = None

    @staticmethod
    def _row(node: dict) -> dict:
        return {"name": node["name"], "type": node.get("type"), "embedding": node.get("embedding")}

    def handler(self, query: str, params: dict) -> list[dict]:
        if query == er._FETCH_GRAPH_ENTITIES:
            rows = [self._row(n) for n in list(self.nodes.values())[: params["limit"]]]
            self.scan_rows += len(rows)
            return rows
        if query == er._FETCH_CANDIDATE_ENTITIES:
            self.candidate_queries.append(params)
            if self.candidate_error is not None:
                raise self.candidate_error
            return self._candidates(params)
        return []

    def _candidates(self, params: dict) -> list[dict]:
        assert params["index"] == er.ENTITY_VECTOR_INDEX
        known = set(params["known"])
        found: dict[str, dict] = {}
        saturated: list[int] = []
        for i, probe in enumerate(params["probes"]):
            unit_probe = _unit_or_none(probe)
            scored = [
                ((1.0 + sum(map(operator.mul, unit_probe, vector))) / 2.0, name)
                for name, vector in self._index
                if len(vector) == len(unit_probe)
            ]
            above = sum(1 for score, _ in scored if score >= params["min_score"])
            if above > params["k"]:
                self.truncated = True
            hits = [
                name
                for score, name in heapq.nlargest(params["k"], scored)
                if score >= params["min_score"]
            ]
            if len(hits) >= params["k"]:
                saturated.append(i)
            for name in hits:
                if name not in known:
                    found.setdefault(name, self.nodes[name])
        for name in params["names"]:
            if name in self.nodes:
                found.setdefault(name, self.nodes[name])
        rows = [self._row(node) for node in found.values()]
        self.candidate_rows += len(rows)
        return [{"saturated": saturated, "entities": rows}]


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

    def test_non_finite_vector_is_zero(self):
        nan, inf = float("nan"), float("inf")
        assert er.cosine_similarity([nan, 0.0, 0.0, 0.0], SAME) == 0.0
        assert er.cosine_similarity([inf, 1.0, 0.0, 0.0], SAME) == 0.0

    def test_a_nan_vector_cannot_merge_on_the_name_alone(self):
        """NaN compares false with everything, so it used to slip past ``cos < threshold``."""
        entities = [{"name": "Postgres", "type": "TOOL"}, {"name": "PostgreSQL", "type": "TOOL"}]
        broken = [float("nan"), 0.0, 0.0, 0.0]
        clusters = er.find_duplicate_clusters(
            entities, {"Postgres": broken, "PostgreSQL": SAME}, settings=_settings()
        )
        assert clusters == []


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
        graph = FakeEntityGraph(
            [
                {"name": "Postgres", "type": "TOOL", "embedding": SAME},
                {"name": "Kafka", "type": "TOOL", "embedding": OTHER},
                # Written by this ingest moments before the resolution pass.
                {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},
            ]
        )
        calls = fake_neo4j(graph.handler)
        fresh = [{"name": "PostgreSQL", "type": "TOOL"}]

        merged = await graph_builder._resolve_against_graph(
            fresh, {"PostgreSQL": SAME}, _settings()
        )

        assert merged == 1
        apoc = [p for q, p in calls if "apoc.refactor.mergeNodes" in q]
        assert apoc == [{"canonical": "PostgreSQL", "dups": ["Postgres"]}]

    async def test_graph_pass_ignores_clusters_without_a_fresh_entity(self, fake_neo4j):
        """Two old duplicates unrelated to this document are left alone."""
        graph = FakeEntityGraph(
            [
                {"name": "Postgres", "type": "TOOL", "embedding": SAME},
                {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},
                {"name": "Kafka", "type": "TOOL", "embedding": OTHER},
            ]
        )
        calls = fake_neo4j(graph.handler)
        merged = await graph_builder._resolve_against_graph(
            [{"name": "Kafka", "type": "TOOL"}], {"Kafka": OTHER}, _settings()
        )

        assert merged == 0
        assert not any("apoc.refactor.mergeNodes" in q for q, _ in calls)

    async def test_full_scan_fallback_also_ignores_clusters_without_a_fresh_entity(
        self, fake_neo4j
    ):
        """Same guarantee on the whole-graph path, which does read the old pair."""
        graph = FakeEntityGraph(
            [
                {"name": "Postgres", "type": "TOOL", "embedding": SAME},
                {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},
            ]
        )
        calls = fake_neo4j(graph.handler)
        merged = await graph_builder._resolve_against_graph(
            [{"name": "Kafka", "type": "TOOL"}],
            {"Kafka": OTHER},
            _settings(entity_resolution_candidate_k=0),
        )

        assert merged == 0
        assert graph.scan_rows == 2
        assert not any("apoc.refactor.mergeNodes" in q for q, _ in calls)

    async def test_graph_pass_noop_on_empty_graph(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        merged = await graph_builder._resolve_against_graph(
            [{"name": "Kafka", "type": "TOOL"}], {"Kafka": SAME}, _settings()
        )
        assert merged == 0
        assert len(calls) == 1  # only the candidate fetch


def _graph_entities_handler(query, params):
    """Fake driver holding one pre-existing "Postgres" and the fresh "PostgreSQL"."""
    graph = FakeEntityGraph(
        [
            {"name": "Postgres", "type": "TOOL", "embedding": SAME},
            {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},
        ]
    )
    return graph.handler(query, params)


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


# ── Cross-document candidates (vector index) ─────────────
class TestVectorIndexScoreFloor:
    """Neo4j's cosine index scores (1 + cos) / 2. The floor must be in *that* unit."""

    def test_default_threshold_maps_to_the_index_score(self):
        floor = er.vector_index_score_floor(0.93)
        assert floor == pytest.approx((1 + 0.93 - er.CANDIDATE_COSINE_MARGIN) / 2)
        assert floor == pytest.approx(0.965, abs=1e-4)

    def test_a_pair_on_the_threshold_clears_the_floor(self):
        at_threshold = (1 + 0.93) / 2
        assert at_threshold >= er.vector_index_score_floor(0.93)

    def test_a_pair_below_the_threshold_does_not(self):
        """The raw threshold would be the wrong floor: it admits cosine down to 0.86."""
        cos_090 = (1 + 0.90) / 2  # 0.95 — above a raw 0.93 floor
        assert cos_090 < er.vector_index_score_floor(0.93)

    def test_margin_absorbs_float32_rounding(self):
        """Lucene scores in float32; a threshold pair must not drop out by an ulp."""
        float32_low = (1 + 0.93) / 2 - 1e-7
        assert float32_low >= er.vector_index_score_floor(0.93)

    def test_is_clamped_to_the_score_range(self):
        assert er.vector_index_score_floor(1.5) == 1.0
        assert er.vector_index_score_floor(-2.0) == 0.0

    @pytest.mark.parametrize(
        "vector",
        [None, [], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], ["x", 1.0], "abc"],
    )
    def test_unusable_vectors_are_not_probes(self, vector):
        assert not er.is_probe_vector(vector)

    def test_usable_vector_is_a_probe(self):
        assert er.is_probe_vector(NEAR)
        assert er.is_probe_vector((0.0, 1e-9))


class TestFetchCandidateEntities:
    async def test_all_probes_go_in_one_batched_unwind_query(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        vectors = {"A": SAME, "B": NEAR, "C": OTHER, "Zero": [0.0] * 4}

        await er.fetch_candidate_entities(
            ["A", "B", "C", "Zero", "Missing"],
            vectors,
            25,
            settings=_settings(),
            pinned_names=["Alias", " ", "Alias"],
            known_names=["B", "A"],
        )

        assert len(calls) == 1  # one round trip, not one per entity
        query, params = calls[0]
        assert query == er._FETCH_CANDIDATE_ENTITIES
        assert query.strip().startswith("UNWIND range(0, size($probes) - 1) AS i")
        assert "CALL db.index.vector.queryNodes($index, $k, $probes[i])" in query
        assert "score >= $min_score" in query
        # Only usable vectors probe; the zero vector and the missing one cannot merge.
        assert params["probes"] == [SAME, NEAR, OTHER]
        assert params["index"] == er.ENTITY_VECTOR_INDEX == "entity_embedding"
        assert params["k"] == 26  # k others + the probe's own node
        assert params["min_score"] == er.vector_index_score_floor(0.93)
        assert params["names"] == ["Alias"]
        assert params["known"] == ["A", "B"]

    async def test_nothing_to_probe_or_pin_costs_no_query(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        result = await er.fetch_candidate_entities(
            ["A"], {"A": [0.0] * 4}, 25, settings=_settings()
        )
        assert result == ([], {})
        assert calls == []

    async def test_rows_have_the_full_scan_shape(self, fake_neo4j):
        fake_neo4j(
            lambda q, p: [
                {
                    "saturated": [],
                    "entities": [
                        {"name": "Postgres", "type": "TOOL", "embedding": [1, 0, 0, 0]},
                        {"name": "Neo4j", "type": "TOOL", "embedding": None},
                        {"name": " ", "type": "TOOL", "embedding": SAME},
                    ],
                }
            ]
        )
        entities, embeddings = await er.fetch_candidate_entities(
            ["PostgreSQL"], {"PostgreSQL": SAME}, 25, settings=_settings()
        )
        assert entities == [
            {"name": "Postgres", "type": "TOOL"},
            {"name": "Neo4j", "type": "TOOL"},
        ]
        assert embeddings == {"Postgres": [1.0, 0.0, 0.0, 0.0]}

    async def test_driver_failure_raises_for_the_caller_to_fall_back(self, fake_neo4j):
        def handler(query, params):
            raise RuntimeError("There is no such vector schema index: entity_embedding")

        fake_neo4j(handler)
        with pytest.raises(er.CandidateSearchUnavailable, match="entity_embedding") as info:
            await er.fetch_candidate_entities(["A"], {"A": SAME}, 25, settings=_settings())
        assert isinstance(info.value.__cause__, RuntimeError)

    def test_saturation_is_counted_before_known_names_are_dropped(self):
        """Known nodes take top-k places too, so they must count towards a full answer.

        This document's own entities and nodes an earlier round fetched are
        both known. Counting after the ``$known`` filter would miss the case
        where they alone fill the top k.
        """
        query = er._FETCH_CANDIDATE_ENTITIES
        assert "WHERE score >= $min_score\nWITH i, collect(node) AS hits" in query
        assert "size(hits) >= $k THEN i END) AS saturated" in query
        assert "RETURN saturated," in query

    async def test_a_saturated_probe_is_asked_again_with_twice_the_k(self, fake_neo4j):
        answers = iter(
            [
                [
                    {
                        "saturated": [1],
                        "entities": [{"name": "Postgres", "type": "TOOL", "embedding": SAME}],
                    }
                ],
                [
                    {
                        "saturated": [],
                        "entities": [{"name": "PG", "type": "TOOL", "embedding": NEAR}],
                    }
                ],
            ]
        )
        calls = fake_neo4j(lambda q, p: next(answers))

        entities, embeddings = await er.fetch_candidate_entities(
            ["A", "B"],
            {"A": SAME, "B": NEAR},
            25,
            settings=_settings(),
            pinned_names=["Alias"],
            known_names=["A", "B"],
        )

        assert [p["k"] for _, p in calls] == [26, 51]
        again = calls[1][1]
        assert again["probes"] == [NEAR]  # only the probe whose answer was full
        assert again["names"] == []  # names were fetched by the first query
        assert again["known"] == ["A", "B", "Postgres"]  # no row is sent twice
        assert [e["name"] for e in entities] == ["Postgres", "PG"]
        assert set(embeddings) == {"Postgres", "PG"}

    async def test_the_doubling_is_bounded_by_the_ceiling_then_warns(self, fake_neo4j, caplog):
        """A neighbourhood that never ends: log2 extra queries, then a warning, never a loop."""
        assert er.MAX_CANDIDATE_K == er.MAX_GRAPH_CANDIDATES
        calls = fake_neo4j(lambda q, p: [{"saturated": [0], "entities": []}])

        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            await er.fetch_candidate_entities(["A"], {"A": SAME}, 25, settings=_settings())

        ks = [p["k"] - 1 for _, p in calls]
        assert ks == [25, 50, 100, 200, 400, 800, 1600, 3200, er.MAX_CANDIDATE_K]
        warned = [r for r in caplog.records if "clear the merge threshold" in r.getMessage()]
        assert len(warned) == 1 and "'A'" in warned[0].getMessage()

    async def test_a_configured_k_above_the_ceiling_raises_it(self, fake_neo4j, monkeypatch):
        monkeypatch.setattr(er, "MAX_CANDIDATE_K", 10)
        calls = fake_neo4j(lambda q, p: [{"saturated": [0], "entities": []}])
        await er.fetch_candidate_entities(["A"], {"A": SAME}, 40, settings=_settings())
        assert [p["k"] for _, p in calls] == [41]


def _orthonormal(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _postgres_graph(**extra_nodes) -> FakeEntityGraph:
    nodes = [
        {"name": "Postgres", "type": "TOOL", "embedding": SAME},
        {"name": "Kafka", "type": "TOOL", "embedding": OTHER},
        {"name": "PostgreSQL", "type": "TOOL", "embedding": SAME},  # this ingest's node
    ]
    nodes += [{"name": n, **fields} for n, fields in extra_nodes.items()]
    return FakeEntityGraph(nodes)


class TestCandidateResolution:
    """The resolution pass reads neighbourhoods, never the whole graph."""

    async def test_a_document_that_merges_nothing_costs_one_query(self, fake_neo4j):
        names = ["Kafka", "Redis", "Spark", "Flink", "Druid"]
        vectors = dict(zip(names, _orthonormal(5), strict=True))
        graph = FakeEntityGraph(
            [{"name": n, "type": "TOOL", "embedding": v} for n, v in vectors.items()]
        )
        calls = fake_neo4j(graph.handler)

        merged = await graph_builder._resolve_against_graph(
            [{"name": n, "type": "TOOL"} for n in names], vectors, _settings()
        )

        assert merged == 0
        assert [q for q, _ in calls] == [er._FETCH_CANDIDATE_ENTITIES]
        assert len(calls[0][1]["probes"]) == 5
        assert graph.scan_rows == 0

    async def test_a_merge_costs_one_confirming_query_and_never_the_scan(self, fake_neo4j):
        graph = _postgres_graph()
        calls = fake_neo4j(graph.handler)

        merged = await graph_builder._resolve_against_graph(
            [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": SAME}, _settings()
        )

        assert merged == 1
        assert len(graph.candidate_queries) == 2
        # Round two probes the entity round one found, and nothing else.
        assert graph.candidate_queries[1]["probes"] == [SAME]
        assert graph.scan_rows == 0
        assert not any(q == er._FETCH_GRAPH_ENTITIES for q, _ in calls)

    async def test_chain_through_unresolved_old_duplicates_is_found_whole(self, fake_neo4j):
        """fresh ~ A ~ B, with fresh and B too far apart for the index to pair directly.

        A and B are old duplicates an earlier merge never resolved. The full
        scan clusters all three; so must the candidate path, via a second hop.
        """
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.95, math.sqrt(1 - 0.95**2), 0.0, 0.0]
        fresh_vec = [0.95, -math.sqrt(1 - 0.95**2), 0.0, 0.0]
        assert er.cosine_similarity(fresh_vec, b) < 0.93  # not a direct neighbour
        graph = FakeEntityGraph(
            [
                {"name": "Ahmed Maaloul", "type": "PERSON", "embedding": a},
                {"name": "Ahmed Maaloul PhD", "type": "PERSON", "embedding": b},
                {"name": "Ahmed", "type": "PERSON", "embedding": fresh_vec},
            ]
        )
        fake_neo4j(graph.handler)
        fresh = [{"name": "Ahmed", "type": "PERSON"}]
        vectors = {"Ahmed": fresh_vec}

        full = await graph_builder._full_scan_clusters(fresh, vectors, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, vectors, _settings(), 25)

        assert full == [["Ahmed Maaloul PhD", "Ahmed", "Ahmed Maaloul"]]
        assert candidates == full
        assert len(graph.candidate_queries) == 3  # fresh, then A, then B

    async def test_absorbed_alias_is_fetched_by_name(self, fake_neo4j):
        """An old "Ahmed" node takes this document's "Ahmed" vector in the full scan.

        Its stored vector is far away, so the index cannot find it: it must be
        fetched by exact name, or the paths would disagree.
        """
        graph = FakeEntityGraph(
            [
                {"name": "Ahmed", "type": "PERSON", "embedding": OTHER},
                {"name": "Ahmed Maaloul", "type": "PERSON", "embedding": SAME},
            ]
        )
        fake_neo4j(graph.handler)
        fresh = [{"name": "Ahmed Maaloul", "type": "PERSON"}]
        # "Ahmed" was folded into "Ahmed Maaloul" in memory; its vector stays in the map.
        vectors = {"Ahmed Maaloul": SAME, "Ahmed": NEAR}

        full = await graph_builder._full_scan_clusters(fresh, vectors, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, vectors, _settings(), 25)

        assert full == [["Ahmed Maaloul", "Ahmed"]]
        assert candidates == full
        assert graph.candidate_queries[0]["names"] == ["Ahmed"]

    async def test_fresh_entity_without_a_vector_keeps_its_stored_one(self, fake_neo4j):
        """No vector this time (embedding failed), but the node kept an older one."""
        graph = _postgres_graph()
        fake_neo4j(graph.handler)
        fresh = [{"name": "PostgreSQL", "type": "TOOL"}]

        full = await graph_builder._full_scan_clusters(fresh, {}, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, {}, _settings(), 25)

        assert full == [["PostgreSQL", "Postgres"]]
        assert candidates == full
        first = graph.candidate_queries[0]
        assert first["probes"] == [] and first["names"] == ["PostgreSQL"]

    async def test_entities_with_no_vector_anywhere_never_merge(self, fake_neo4j):
        """The invariant: no vector, no quorum, no merge. It costs one lookup, no scan."""
        graph = _postgres_graph(Postgre={"type": "TOOL", "embedding": None})
        fake_neo4j(graph.handler)
        fresh = [{"name": "Postgre", "type": "TOOL"}]

        merged = await graph_builder._resolve_against_graph(fresh, {}, _settings())

        assert merged == 0
        assert len(graph.candidate_queries) == 1
        assert graph.scan_rows == 0
        assert await graph_builder._full_scan_clusters(fresh, {}, _settings()) == []

    async def test_unusable_vectors_cost_no_query(self, fake_neo4j):
        graph = _postgres_graph()
        calls = fake_neo4j(graph.handler)
        merged = await graph_builder._resolve_against_graph(
            [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": [0.0] * 4}, _settings()
        )
        assert merged == 0
        assert calls == []


class TestCandidateFallback:
    async def test_index_failure_falls_back_to_the_full_scan(self, fake_neo4j, monkeypatch):
        monkeypatch.setattr(graph_builder, "_candidate_fallback_logged", False)
        graph = _postgres_graph()
        graph.candidate_error = RuntimeError("There is no procedure db.index.vector.queryNodes")
        calls = fake_neo4j(graph.handler)

        merged = await graph_builder._resolve_against_graph(
            [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": SAME}, _settings()
        )

        assert merged == 1
        assert graph.scan_rows == 3  # the whole graph, as before
        apoc = [p for q, p in calls if "apoc.refactor.mergeNodes" in q]
        assert apoc == [{"canonical": "PostgreSQL", "dups": ["Postgres"]}]

    async def test_failure_in_a_later_round_still_falls_back_whole(self, fake_neo4j, monkeypatch):
        monkeypatch.setattr(graph_builder, "_candidate_fallback_logged", False)
        graph = _postgres_graph()
        real_candidates = graph._candidates

        def second_round_fails(params):
            if graph.candidate_queries[1:]:
                raise RuntimeError("connection reset")
            return real_candidates(params)

        graph._candidates = second_round_fails
        fake_neo4j(graph.handler)
        fresh, vectors = [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": SAME}

        clusters = await graph_builder._graph_clusters(fresh, vectors, _settings())

        assert clusters == [["PostgreSQL", "Postgres"]]
        assert len(graph.candidate_queries) == 2
        assert graph.scan_rows == 3

    async def test_fallback_warns_once_then_logs_at_debug(self, fake_neo4j, monkeypatch, caplog):
        monkeypatch.setattr(graph_builder, "_candidate_fallback_logged", False)
        graph = _postgres_graph()
        graph.candidate_error = RuntimeError("no vector index")
        fake_neo4j(graph.handler)

        with caplog.at_level(logging.DEBUG, logger="app.services.graph_builder"):
            for _ in range(3):
                await graph_builder._graph_clusters(
                    [{"name": "Kafka", "type": "TOOL"}], {"Kafka": OTHER}, _settings()
                )

        fallback = [r for r in caplog.records if "vector index unavailable" in r.getMessage()]
        assert [r.levelno for r in fallback] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
        assert "entity_embedding" in fallback[0].getMessage()
        assert graph.scan_rows == 9  # every ingest still resolved, via the scan

    async def test_k_zero_restores_the_full_scan(self, fake_neo4j):
        graph = _postgres_graph()
        fake_neo4j(graph.handler)

        merged = await graph_builder._resolve_against_graph(
            [{"name": "PostgreSQL", "type": "TOOL"}],
            {"PostgreSQL": SAME},
            _settings(entity_resolution_candidate_k=0),
        )

        assert merged == 1
        assert graph.candidate_queries == []
        assert graph.scan_rows == 3

    def test_candidate_k_setting_default_and_env(self, monkeypatch):
        monkeypatch.delenv("ENTITY_RESOLUTION_CANDIDATE_K", raising=False)
        assert Settings(_env_file=None).entity_resolution_candidate_k == 25
        monkeypatch.setenv("ENTITY_RESOLUTION_CANDIDATE_K", "7")
        assert Settings(_env_file=None).entity_resolution_candidate_k == 7


# ── Property: candidates == full scan, on random graphs ──
_SYLLABLES = (
    "ba", "ce", "di", "fo", "gu", "ha", "je", "ki", "lo", "mu", "na", "pe", "qui",
    "ro", "su", "ta", "vo", "we", "xi", "yu", "za", "or", "el", "an", "im", "ub",
)  # fmt: skip
_TYPES = ("PERSON", "TOOL", "ORGANIZATION", "CONCEPT", "LOCATION", "EVENT", None)
_DIM = 6  # low on purpose: random pairs then clear 0.93 now and then, as noise
_SCENARIOS = (
    "match", "chain", "near_miss", "boundary", "semantic_only",
    "type_barrier", "stale", "alias", "alias_absent", "degenerate", "fresh_pair",
)  # fmt: skip


def _random_unit(rng: random.Random, dim: int) -> list[float]:
    while True:
        vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-9:
            return [v / norm for v in vector]


def _at_cosine(
    rng: random.Random, anchor: list[float], cos: float, away_from: list[float] | None = None
) -> list[float]:
    """A unit vector at exactly ``cos`` from unit ``anchor``.

    With ``away_from`` the vector leans away from that one, so a chain
    fresh ~ A ~ B can be built with fresh and B too far apart to pair directly.
    """
    direction = [-v for v in away_from] if away_from is not None else _random_unit(rng, len(anchor))
    along = sum(map(operator.mul, direction, anchor))
    ortho = [d - along * a for d, a in zip(direction, anchor, strict=True)]
    norm = math.sqrt(sum(v * v for v in ortho))
    if norm < 1e-9:
        return _at_cosine(rng, anchor, cos)
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    return [cos * a + sin * o / norm for a, o in zip(anchor, ortho, strict=True)]


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


def _name(rng: random.Random) -> str:
    return " ".join(_word(rng) for _ in range(rng.randint(1, 2)))


def _spellings(rng: random.Random, base: str) -> list[str]:
    """``base`` and deliberate near-duplicates of it that the name gate accepts."""
    out = [base, base.lower().replace(" ", "-"), f"{base} {_word(rng)}"]
    tokens = base.split()
    if len(tokens) > 1 and len(tokens[0]) >= er.MIN_CONTAINMENT_CHARS:
        out.append(tokens[0])  # "Ahmed" for "Ahmed Maaloul"
    if len(base.replace(" ", "")) >= 6:
        out.append(base[:-1])  # a typo
    return list(dict.fromkeys(out))


def _synthetic_graph(seed: int):
    """``(FakeEntityGraph, fresh_entities, vectors, planted)`` for one seed; see :func:`_synthetic_case`."""
    nodes, fresh, vectors, planted = _synthetic_case(seed)
    return FakeEntityGraph(list(nodes)), list(fresh), vectors, planted


@functools.cache
def _synthetic_case(seed: int):
    """A random graph, one fresh document written into it, and the planted cases.

    The graph holds 5-2,000 unrelated entities (log-uniform), with 5% stored
    without a vector, plus 1-6 planted groups of near-duplicate names with
    close vectors. Some groups hold old duplicates that an earlier merge never
    resolved, sometimes as chains. For each group, the document brings an
    entity picked from :data:`_SCENARIOS`: a clear match, a chain end, a near
    miss, a pair on the threshold, a semantic-only or type-barred look-alike,
    a fresh name without a vector of its own, an absorbed alias, a degenerate
    vector, or a fresh pair. As in the real pipeline, the fresh entities are
    already in the graph, stored with this document's vectors.
    """
    rng = random.Random(seed)
    nodes: dict[str, dict] = {}
    fresh: dict[str, dict] = {}
    vectors: dict[str, list[float]] = {}
    planted: dict[str, set[str]] = {"alias": set(), "stale": set()}

    def unused(name: str) -> bool:
        return name not in nodes and name not in fresh and name not in vectors

    def add_fresh(name: str, etype, vector) -> None:
        fresh[name] = {"name": name, "type": etype}
        if vector is not None:
            vectors[name] = vector
            nodes[name] = {"name": name, "type": etype, "embedding": vector}

    size = round(math.exp(rng.uniform(math.log(5), math.log(2000))))
    for _ in range(size):
        name = _name(rng)
        if unused(name):
            vector = None if rng.random() < 0.05 else _random_unit(rng, _DIM)
            nodes[name] = {"name": name, "type": rng.choice(_TYPES), "embedding": vector}

    for _ in range(rng.randint(1, 6)):
        spellings = [s for s in _spellings(rng, _name(rng)) if unused(s)]
        if len(spellings) < 3:
            continue
        rng.shuffle(spellings)
        etype = rng.choice(_TYPES)
        anchor = _random_unit(rng, _DIM)
        members: list[list[float]] = []
        for spelling in spellings[: min(rng.randint(1, 2), len(spellings) - 2)]:
            if members and rng.random() < 0.5:  # an unresolved chain link
                vector = _at_cosine(rng, members[-1], rng.uniform(0.94, 0.97), members[0])
            else:
                vector = _at_cosine(rng, anchor, rng.uniform(0.93, 0.995))
            nodes[spelling] = {"name": spelling, "type": etype, "embedding": vector}
            members.append(vector)
        spare = spellings[len(members) :]
        target = rng.choice(members)
        near = _at_cosine(rng, target, rng.uniform(0.935, 0.995))

        scenario = rng.choice(_SCENARIOS)
        if scenario == "match":
            add_fresh(spare[0], etype, near)
        elif scenario == "chain":
            add_fresh(spare[0], etype, _at_cosine(rng, target, 0.95, members[0]))
        elif scenario == "near_miss":
            add_fresh(spare[0], etype, _at_cosine(rng, target, rng.uniform(0.85, 0.929)))
        elif scenario == "boundary":
            offset = rng.choice((-2e-5, -1e-9, 0.0, 1e-9, 2e-5))
            add_fresh(spare[0], etype, _at_cosine(rng, target, 0.93 + offset))
        elif scenario == "semantic_only":
            other = _name(rng)
            if unused(other):
                add_fresh(other, etype, near)
        elif scenario == "type_barrier":
            add_fresh(spare[0], "TYPE_" + str(etype), near)
        elif scenario == "stale":
            # Fresh in this document, but no vector this time: the node keeps an old one.
            fresh[spare[0]] = {"name": spare[0], "type": etype}
            nodes[spare[0]] = {"name": spare[0], "type": etype, "embedding": near}
            planted["stale"].add(spare[0])
        elif scenario in ("alias", "alias_absent"):
            # spare[0] was folded into spare[1] in memory; its vector stays in the map.
            vectors[spare[0]] = _at_cosine(rng, target, rng.uniform(0.94, 0.99))
            if scenario == "alias":  # ...and an older document wrote a node by that name
                far = _random_unit(rng, _DIM)
                nodes[spare[0]] = {"name": spare[0], "type": etype, "embedding": far}
                planted["alias"].add(spare[0])
            add_fresh(spare[1], etype, near)
        elif scenario == "degenerate":
            add_fresh(spare[0], etype, rng.choice(([0.0] * _DIM, [float("nan")] * _DIM)))
        elif scenario == "fresh_pair":
            add_fresh(spare[0], etype, near)
            add_fresh(spare[1], etype, _at_cosine(rng, near, rng.uniform(0.95, 0.99)))

    for _ in range(rng.randint(0, 4)):  # unrelated new entities
        name = _name(rng)
        if unused(name):
            add_fresh(name, rng.choice(_TYPES), _random_unit(rng, _DIM))

    # Cached (generation is deterministic and read-only), so both property tests
    # pay for it once. Each test still builds its own FakeEntityGraph and counters.
    return tuple(nodes.values()), tuple(fresh.values()), vectors, planted


def _needed_a_second_hop(
    clusters: list[list[str]], fresh: list[dict], vectors: dict, graph: FakeEntityGraph
) -> bool:
    """True if a cluster holds an old entity no fresh vector is close enough to find."""
    fresh_names = {e["name"] for e in fresh}
    probes = [vectors[n] for n in fresh_names if er.is_probe_vector(vectors.get(n))]
    for member in (m for c in clusters for m in c):
        if member in fresh_names or member in vectors:
            continue
        stored = graph.nodes[member]["embedding"]
        floor = 0.93 - er.CANDIDATE_COSINE_MARGIN
        if all(er.cosine_similarity(p, stored) < floor for p in probes):
            return True
    return False


class TestCandidatesEqualTheFullScan:
    """Property tests over 200 seeded random graphs against an exact kNN index."""

    GRAPHS = 200

    @pytest.mark.parametrize("k", [25, 1])
    async def test_same_clusters_as_the_full_scan_on_200_random_graphs(self, fake_neo4j, k):
        """At k = 1 most probes overflow their first answer: the re-asking keeps it exact."""
        current: dict[str, FakeEntityGraph] = {}
        fake_neo4j(lambda q, p: current["graph"].handler(q, p))
        settings = _settings()
        seen = dict.fromkeys(
            ("clusters", "second_hop", "alias", "stale", "three_rounds", "big_graph", "crowded"),
            0,
        )

        for seed in range(self.GRAPHS):
            graph, fresh, vectors, planted = _synthetic_graph(seed)
            current["graph"] = graph

            full = await graph_builder._full_scan_clusters(fresh, vectors, settings)
            candidates = await graph_builder._candidate_clusters(fresh, vectors, settings, k)

            assert candidates == full, f"seed {seed}"

            members = {m for c in full for m in c}
            seen["clusters"] += bool(full)
            seen["second_hop"] += _needed_a_second_hop(full, fresh, vectors, graph)
            seen["alias"] += bool(planted["alias"] & members)
            seen["stale"] += bool(planted["stale"] & members)
            seen["three_rounds"] += len(graph.candidate_queries) >= 3
            seen["big_graph"] += len(graph.nodes) >= 1000
            seen["crowded"] += graph.truncated

        # Not vacuous: every mechanism of the proof was exercised, many times.
        assert seen["clusters"] >= 120, seen
        assert seen["second_hop"] >= 15, seen
        assert seen["alias"] >= 5, seen
        assert seen["stale"] >= 5, seen
        assert seen["three_rounds"] >= 5, seen
        assert seen["big_graph"] >= 20, seen
        if k == 1:  # more than k nodes above the floor, on most graphs
            assert seen["crowded"] >= 100, seen

    async def test_past_the_ceiling_it_only_ever_misses_never_invents(
        self, fake_neo4j, monkeypatch, caplog
    ):
        """The residual, shown: with no room to widen k, a member can be missed.

        Even then, every cluster is a subset of a full-scan cluster: the
        candidate path can drop an entity from a merge, never add a wrong one.
        And it says so in the log.
        """
        monkeypatch.setattr(er, "MAX_CANDIDATE_K", 1)
        current: dict[str, FakeEntityGraph] = {}
        fake_neo4j(lambda q, p: current["graph"].handler(q, p))
        settings = _settings()
        differed = 0

        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            for seed in range(self.GRAPHS):
                graph, fresh, vectors, _ = _synthetic_graph(seed)
                current["graph"] = graph
                full = await graph_builder._full_scan_clusters(fresh, vectors, settings)
                candidates = await graph_builder._candidate_clusters(fresh, vectors, settings, 1)

                full_sets = [set(c) for c in full]
                for cluster in candidates:
                    assert any(set(cluster) <= f for f in full_sets), f"seed {seed}: {cluster}"
                differed += candidates != full

        assert differed > 0  # with k = 1 and no widening the residual is real
        assert any("clear the merge threshold" in r.getMessage() for r in caplog.records)


# ── Review counterexamples, kept as regressions ──────────
def _gibberish(rng: random.Random) -> str:
    """Nine random consonants: never passes the name gate against a real name."""
    return "".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(9)).capitalize()


def _crowd(rng: random.Random, anchor: list[float], size: int, etype, cos: float) -> list[dict]:
    """``size`` entities at ``cos`` from ``anchor`` whose names match nothing."""
    names: set[str] = set()
    while len(names) < size:
        names.add(_gibberish(rng))
    return [
        {"name": n, "type": etype, "embedding": _at_cosine(rng, anchor, cos)} for n in sorted(names)
    ]


async def _merges(fake_neo4j, nodes: list[dict], fresh: list[dict], vectors: dict, k: int):
    """``(merged, apoc merge params, graph)`` for one resolution pass at this k."""
    graph = FakeEntityGraph(nodes)
    calls = fake_neo4j(graph.handler)
    merged = await graph_builder._resolve_against_graph(
        fresh, vectors, _settings(entity_resolution_candidate_k=k)
    )
    return merged, [p for q, p in calls if "apoc.refactor.mergeNodes" in q], graph


class TestCrowdedNeighbourhoods:
    """More nodes above the threshold around one probe than k lets through.

    Before the query reported full answers, each of these lost a merge the
    full scan makes.
    """

    async def test_this_documents_own_entities_count_towards_k(self, fake_neo4j):
        """25 fresh look-alikes fill PostgreSQL's top 26, so the old "Postgres" ranks 27th.

        They are this document's own entities, so the query drops them as
        known, but only after they have taken their places in the top k.
        """
        rng = random.Random(7)
        base = _random_unit(rng, 16)
        document = [{"name": "PostgreSQL", "type": "TOOL", "embedding": base}]
        document += _crowd(rng, base, 25, "TOOL", 0.995)
        postgres = {"name": "Postgres", "type": "TOOL", "embedding": _at_cosine(rng, base, 0.95)}
        fresh = [{"name": n["name"], "type": "TOOL"} for n in document]
        vectors = {n["name"]: n["embedding"] for n in document}

        *with_index, graph = await _merges(fake_neo4j, [*document, postgres], fresh, vectors, 25)
        *full_scan, _ = await _merges(fake_neo4j, [*document, postgres], fresh, vectors, 0)

        assert graph.truncated  # the first answer really was too short
        assert [q["k"] for q in graph.candidate_queries[:2]] == [26, 51]
        assert with_index == full_scan == [1, [{"canonical": "PostgreSQL", "dups": ["Postgres"]}]]

    async def test_a_crowd_cannot_split_a_chain(self, fake_neo4j):
        """Ahmed ~ Ahmed Maaloul ~ Ahmed Maaloul PhD, with 26 look-alikes around the middle.

        Cut short, the middle's probe never reached "Ahmed Maaloul PhD". The
        merge then kept a different name and left that node behind as a
        duplicate.
        """
        rng = random.Random(11)
        middle = _random_unit(rng, 16)
        ahmed = _at_cosine(rng, middle, 0.96)
        phd = _at_cosine(rng, middle, 0.95, ahmed)
        assert er.cosine_similarity(ahmed, phd) < 0.93  # only the middle links them
        nodes = [
            *_crowd(rng, middle, 26, "PERSON", 0.995),
            {"name": "Ahmed Maaloul", "type": "PERSON", "embedding": middle},
            {"name": "Ahmed Maaloul PhD", "type": "PERSON", "embedding": phd},
            {"name": "Ahmed", "type": "PERSON", "embedding": ahmed},
        ]
        fresh, vectors = [{"name": "Ahmed", "type": "PERSON"}], {"Ahmed": ahmed}

        *with_index, graph = await _merges(fake_neo4j, nodes, fresh, vectors, 25)
        *full_scan, _ = await _merges(fake_neo4j, nodes, fresh, vectors, 0)

        assert graph.truncated
        expected = [{"canonical": "Ahmed Maaloul PhD", "dups": ["Ahmed", "Ahmed Maaloul"]}]
        assert with_index == full_scan == [2, expected]

    @pytest.mark.parametrize("crowd", [24, 25, 26, 80])
    async def test_no_crowd_size_hides_the_duplicate(self, fake_neo4j, crowd):
        """The old boundary: 24 closer look-alikes were fine, 25 lost the merge."""
        rng = random.Random(crowd)
        base = _random_unit(rng, 16)
        graph = FakeEntityGraph(
            [
                *_crowd(rng, base, crowd, "TOOL", 0.99),
                {"name": "Postgres", "type": "TOOL", "embedding": _at_cosine(rng, base, 0.95)},
                {"name": "PostgreSQL", "type": "TOOL", "embedding": base},
            ]
        )
        fake_neo4j(graph.handler)
        fresh, vectors = [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": base}

        full = await graph_builder._full_scan_clusters(fresh, vectors, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, vectors, _settings(), 25)

        assert full == [["PostgreSQL", "Postgres"]]
        assert candidates == full


class TestFullScanCap:
    """The full scan stops at MAX_GRAPH_CANDIDATES rows, and now says so."""

    @staticmethod
    def _graph(size: int) -> FakeEntityGraph:
        return FakeEntityGraph(
            [{"name": f"Entity {i}", "type": "TOOL", "embedding": None} for i in range(size)]
        )

    async def test_a_graph_larger_than_the_limit_is_reported(self, fake_neo4j, caplog):
        calls = fake_neo4j(self._graph(5).handler)
        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            entities, _ = await er.fetch_graph_entities(limit=3)

        assert len(entities) == 3
        assert calls[0][1] == {"limit": 4}  # one more row than kept, to see the cut
        warned = [r for r in caplog.records if "read only the first 3" in r.getMessage()]
        assert len(warned) == 1

    async def test_a_graph_of_exactly_the_limit_is_not(self, fake_neo4j, caplog):
        fake_neo4j(self._graph(3).handler)
        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            entities, _ = await er.fetch_graph_entities(limit=3)

        assert len(entities) == 3
        assert not [r for r in caplog.records if "read only the first" in r.getMessage()]

    async def test_the_candidate_path_finds_a_duplicate_past_the_cap(self, fake_neo4j, caplog):
        """The proof's reference is the *uncapped* scan, so here the capped one differs.

        The duplicate sits at row 5,001. The capped scan (the fallback, and
        k = 0) never compares it and says so. The candidate path merges it.
        """
        rng = random.Random(3)
        nodes: dict[str, dict] = {}
        while len(nodes) < er.MAX_GRAPH_CANDIDATES:
            name = _name(rng)
            nodes.setdefault(
                name, {"name": name, "type": "CONCEPT", "embedding": _random_unit(rng, 16)}
            )
        base = _random_unit(rng, 16)
        graph = FakeEntityGraph(
            [
                *nodes.values(),
                {"name": "Postgres", "type": "TOOL", "embedding": _at_cosine(rng, base, 0.97)},
                {"name": "PostgreSQL", "type": "TOOL", "embedding": base},
            ]
        )
        fake_neo4j(graph.handler)
        fresh, vectors = [{"name": "PostgreSQL", "type": "TOOL"}], {"PostgreSQL": base}

        with caplog.at_level(logging.WARNING, logger="app.services.entity_resolution"):
            full = await graph_builder._full_scan_clusters(fresh, vectors, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, vectors, _settings(), 25)

        assert full == []
        assert any("read only the first 5000" in r.getMessage() for r in caplog.records)
        assert candidates == [["PostgreSQL", "Postgres"]]


class TestBothPathsAgree:
    async def test_a_nan_vector_merges_on_neither_path(self, fake_neo4j):
        """A deliberate change from the old scan, recorded in the CHANGELOG.

        A NaN vector used to pass ``cosine < threshold`` (NaN compares false),
        so the old full scan merged this pair on the name alone.
        """
        nodes = [
            {"name": "Postgres", "type": "TOOL", "embedding": SAME},
            {"name": "PostgreSQL", "type": "TOOL", "embedding": None},
        ]
        fresh = [{"name": "PostgreSQL", "type": "TOOL"}]
        vectors = {"PostgreSQL": [float("nan"), 0.0, 0.0, 0.0]}

        for k in (25, 0):
            merged, apoc, _ = await _merges(fake_neo4j, nodes, fresh, vectors, k)
            assert (merged, apoc) == (0, []), f"k={k}"

    async def test_an_empty_graph_read_still_clusters_the_fresh_entities(self, fake_neo4j):
        """The full scan used to return nothing here while the candidate path merged the pair.

        The pipeline cannot reach this (fresh nodes are written first, and a
        fresh pair was already collapsed in memory), but direct callers can.
        """
        fake_neo4j(FakeEntityGraph([]).handler)
        fresh = [{"name": "Postgres", "type": "TOOL"}, {"name": "PostgreSQL", "type": "TOOL"}]
        vectors = {"Postgres": SAME, "PostgreSQL": NEAR}

        full = await graph_builder._full_scan_clusters(fresh, vectors, _settings())
        candidates = await graph_builder._candidate_clusters(fresh, vectors, _settings(), 25)

        assert full == candidates
        assert [set(c) for c in full] == [{"Postgres", "PostgreSQL"}]


# ── Micro-benchmark: per-document work vs graph size ─────
def _benchmark_graph(background: int) -> tuple[FakeEntityGraph, list[dict], dict, int]:
    """One 8-entity document against ``background`` unrelated entities + its planted matches.

    The document and the 3 old entities it should merge with are identical at
    every size. Only the unrelated rest of the graph grows. Vectors are 16-d, so
    random pairs never clear 0.93.
    """
    dim = 16
    doc = random.Random(7)
    nodes: list[dict] = []
    fresh: list[dict] = []
    vectors: dict[str, list[float]] = {}
    for i in range(8):
        name = f"Synapse Topic {i}"
        vectors[name] = _random_unit(doc, dim)
        fresh.append({"name": name, "type": "CONCEPT"})
        nodes.append({"name": name, "type": "CONCEPT", "embedding": vectors[name]})
        if i < 3:  # an older spelling of it, from a previous document
            old = f"synapse-topic-{i}"
            nodes.append(
                {"name": old, "type": "CONCEPT", "embedding": _at_cosine(doc, vectors[name], 0.97)}
            )
    planted = len(nodes)

    rng = random.Random(background)
    taken = {n["name"] for n in nodes}
    while len(nodes) < planted + background:
        name = _name(rng)
        if name not in taken:
            taken.add(name)
            nodes.append(
                {"name": name, "type": rng.choice(_TYPES), "embedding": _random_unit(rng, dim)}
            )
    return FakeEntityGraph(nodes), fresh, vectors, planted


class TestPerDocumentCostIsIndependentOfGraphSize:
    SIZES = (250, 1_000, 4_000)

    async def test_rows_read_and_entities_clustered_do_not_grow_with_the_graph(
        self, fake_neo4j, monkeypatch
    ):
        real_find = er.find_duplicate_clusters
        clustered: list[int] = []

        def counting_find(entities, embeddings, settings=None):
            clustered.append(len(entities))
            return real_find(entities, embeddings, settings=settings)

        monkeypatch.setattr(er, "find_duplicate_clusters", counting_find)
        k = _settings().entity_resolution_candidate_k
        work: dict[int, tuple] = {}
        old_path_rows: dict[int, int] = {}

        for size in self.SIZES:
            graph, fresh, vectors, planted = _benchmark_graph(size)
            fake_neo4j(graph.handler)
            clustered.clear()

            clusters = await graph_builder._graph_clusters(fresh, vectors, _settings())

            probes = sum(len(q["probes"]) for q in graph.candidate_queries)
            assert graph.candidate_rows <= probes * (k + 1)  # O(fresh × k)
            assert graph.scan_rows == 0  # the whole graph is never read
            work[size] = (
                len(graph.candidate_queries),
                graph.candidate_rows,
                max(clustered),
                clusters,
            )
            # For contrast: what the old path read for this one document.
            old_path_rows[size] = len((await er.fetch_graph_entities())[0])
            assert old_path_rows[size] == planted + size

        # Identical work at every size, while the graph grew 16x.
        assert len(set(map(repr, work.values()))) == 1, work
        queries, rows, entities_clustered, clusters = work[self.SIZES[0]]
        assert queries == 2  # probe the document, then confirm its 3 matches end there
        assert rows == 3 and entities_clustered == 8 + 3
        assert len(clusters) == 3
        assert old_path_rows[self.SIZES[-1]] == 16 * old_path_rows[self.SIZES[0]] - 15 * planted
