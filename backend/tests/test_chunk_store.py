# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the source-chunk text units: identity, linking, retrieval, isolation.

Fully hermetic — no network, no LLM, no Neo4j. The ``fake_neo4j`` fixture stands
in for the driver, so every assertion is about the Cypher and parameters the
service *actually* emits.
"""

from __future__ import annotations

import pytest

from app.services import chunk_store, graph_builder
from app.services.graph_schema import CHUNK_VECTOR_INDEX

# Orthogonal unit vectors, so every cosine similarity below is exact.
SAME = [1.0, 0.0, 0.0, 0.0]
OTHER = [0.0, 1.0, 0.0, 0.0]


class _FakeEmbeddings:
    """Deterministic embedder: 'ahmed*' share a vector, 'postgre*' share another."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if lowered.startswith("ahmed"):
            return SAME
        if lowered.startswith("postgre"):
            return OTHER
        return [0.0, 0.0, 1.0, 0.0]


def _write_calls(calls) -> list[dict]:
    return [p for q, p in calls if "MERGE (c:Chunk {id: $id})" in q]


def _link_calls(calls) -> list[dict]:
    return [p for q, p in calls if "MERGE (e)-[:MENTIONED_IN]->(c)" in q]


# ── Identity ─────────────────────────────────────────
class TestChunkId:
    def test_is_deterministic(self):
        assert chunk_store.chunk_id("cv.pdf", 0, "hello") == chunk_store.chunk_id(
            "cv.pdf", 0, "hello"
        )

    def test_varies_with_every_component(self):
        base = chunk_store.chunk_id("cv.pdf", 0, "hello")
        assert base != chunk_store.chunk_id("other.pdf", 0, "hello")
        assert base != chunk_store.chunk_id("cv.pdf", 1, "hello")
        assert base != chunk_store.chunk_id("cv.pdf", 0, "hello!")

    def test_separator_prevents_field_smearing(self):
        """"cv.pdf" + 1 + "x" must not collide with "cv.pdf" + 11 + "" style joins."""
        assert chunk_store.chunk_id("cv.pdf", 1, "1x") != chunk_store.chunk_id("cv.pdf", 11, "x")

    def test_is_a_short_stable_hex_digest(self):
        cid = chunk_store.chunk_id("cv.pdf", 0, "hello")
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)


# ── Write path ───────────────────────────────────────
class TestStoreChunks:
    async def test_merges_on_a_content_addressed_id(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        result = await chunk_store.store_chunks("cv.pdf", ["Ahmed ships graphs."], [["Ahmed"]])

        assert result == {"chunks": 1, "links": 1}
        writes = _write_calls(calls)
        assert len(writes) == 1
        assert writes[0]["id"] == chunk_store.chunk_id("cv.pdf", 0, "Ahmed ships graphs.")
        assert writes[0]["text"] == "Ahmed ships graphs."
        assert writes[0]["document"] == "cv.pdf"
        assert writes[0]["index"] == 0
        assert writes[0]["embedding"] == SAME
        # MERGE, never CREATE — that is what makes re-ingestion idempotent.
        assert all("CREATE (c:Chunk" not in q for q, _ in calls)

    async def test_reingesting_the_same_document_reuses_the_same_ids(
        self, monkeypatch, fake_neo4j
    ):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        chunks = ["Ahmed ships graphs.", "Postgres stores them."]
        await chunk_store.store_chunks("cv.pdf", chunks, [["Ahmed"], ["Postgres"]])
        await chunk_store.store_chunks("cv.pdf", chunks, [["Ahmed"], ["Postgres"]])

        ids = [p["id"] for p in _write_calls(calls)]
        assert ids[:2] == ids[2:]  # second ingest targets exactly the same nodes
        assert len(set(ids)) == 2  # ...and only two distinct chunks exist

    async def test_embeds_every_chunk_in_a_single_batched_call(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        batches: list[list[str]] = []

        class _Recording(_FakeEmbeddings):
            def embed_documents(self, texts):
                batches.append(list(texts))
                return super().embed_documents(texts)

        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _Recording())

        await chunk_store.store_chunks("cv.pdf", ["a", "b", "c"], [[], [], []])

        assert batches == [["a", "b", "c"]]  # one call, not three

    async def test_links_only_named_entities_and_dedupes_them(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        await chunk_store.store_chunks(
            "cv.pdf", ["text"], [["Ahmed", " Ahmed ", "", "Postgres"]]
        )

        assert _link_calls(calls)[0]["names"] == ["Ahmed", "Postgres"]

    async def test_no_link_query_when_a_chunk_has_no_entities(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        result = await chunk_store.store_chunks("cv.pdf", ["text"], [[]])

        assert result == {"chunks": 1, "links": 0}
        assert _link_calls(calls) == []

    async def test_links_never_merge_a_missing_entity_into_existence(
        self, monkeypatch, fake_neo4j
    ):
        """A resolution-merged duplicate must stay dead; the link MATCHes, not MERGEs."""
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        await chunk_store.store_chunks("cv.pdf", ["text"], [["Ghost"]])

        link_query = [q for q, _ in calls if "MENTIONED_IN" in q][0]
        assert "MATCH (e:Entity {name: name})" in link_query
        assert "MERGE (e:Entity" not in link_query

    async def test_short_entity_name_list_is_tolerated(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        result = await chunk_store.store_chunks("cv.pdf", ["a", "b"], [["Ahmed"]])

        assert result["chunks"] == 2
        assert len(_link_calls(calls)) == 1

    async def test_empty_input_is_a_noop(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        assert await chunk_store.store_chunks("cv.pdf", [], []) == {"chunks": 0, "links": 0}
        assert calls == []

    async def test_reports_the_link_count_the_database_returns(self, monkeypatch, fake_neo4j):
        def handler(query, params):
            if "MENTIONED_IN" in query:
                return [{"links": 1}]  # only one of the two names exists
            return []

        fake_neo4j(handler)
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        result = await chunk_store.store_chunks("cv.pdf", ["t"], [["Ahmed", "Ghost"]])

        assert result == {"chunks": 1, "links": 1}


class TestStoreChunksDegradesGracefully:
    async def test_embedding_failure_still_stores_the_text(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])

        def boom(*a, **kw):
            raise RuntimeError("no embedding backend")

        monkeypatch.setattr(chunk_store, "get_embeddings", boom)

        result = await chunk_store.store_chunks("cv.pdf", ["a", "b"], [["Ahmed"], []])

        # The prose is the whole point — it is persisted with a null vector
        # rather than lost, so entity-linked retrieval still reaches it.
        assert result["chunks"] == 2
        assert [p["embedding"] for p in _write_calls(calls)] == [None, None]
        assert result["links"] == 1

    async def test_embedding_count_mismatch_falls_back_to_no_vectors(
        self, monkeypatch, fake_neo4j
    ):
        calls = fake_neo4j(lambda q, p: [])

        class _Short:
            def embed_documents(self, texts):
                return [SAME]  # one vector for two chunks

        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _Short())

        result = await chunk_store.store_chunks("cv.pdf", ["a", "b"], [[], []])

        assert result["chunks"] == 2
        assert [p["embedding"] for p in _write_calls(calls)] == [None, None]

    async def test_missing_vector_procedure_falls_back_to_a_plain_write(
        self, monkeypatch, fake_neo4j
    ):
        def handler(query, params):
            if "setNodeVectorProperty" in query:
                raise RuntimeError("Unknown procedure db.create.setNodeVectorProperty")
            return []

        calls = fake_neo4j(handler)
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        result = await chunk_store.store_chunks("cv.pdf", ["a"], [["Ahmed"]])

        assert result == {"chunks": 1, "links": 1}
        assert any(
            "MERGE (c:Chunk {id: $id})" in q and "setNodeVectorProperty" not in q
            for q, _ in calls
        )

    async def test_a_dead_database_does_not_abort_the_ingest(self, monkeypatch, fake_neo4j):
        def handler(query, params):
            raise RuntimeError("connection refused")

        fake_neo4j(handler)
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        assert await chunk_store.store_chunks("cv.pdf", ["a"], [["Ahmed"]]) == {
            "chunks": 0,
            "links": 0,
        }

    async def test_link_failure_still_counts_the_stored_chunk(self, monkeypatch, fake_neo4j):
        def handler(query, params):
            if "MENTIONED_IN" in query:
                raise RuntimeError("deadlock")
            return []

        fake_neo4j(handler)
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        assert await chunk_store.store_chunks("cv.pdf", ["a"], [["Ahmed"]]) == {
            "chunks": 1,
            "links": 0,
        }


# ── Vector search ────────────────────────────────────
class TestSearchChunks:
    async def test_queries_the_chunk_vector_index(self, monkeypatch, fake_neo4j):
        rows = [
            {"id": "c1", "text": "Ahmed ships graphs.", "document": "cv.pdf", "index": 0,
             "score": 0.91}
        ]
        calls = fake_neo4j(lambda q, p: rows if "queryNodes" in q else [])
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        out = await chunk_store.search_chunks("ahmed", k=3)

        assert out == [
            {"id": "c1", "text": "Ahmed ships graphs.", "document": "cv.pdf", "index": 0,
             "score": 0.91}
        ]
        params = [p for q, p in calls if "queryNodes" in q][0]
        assert params["index"] == CHUNK_VECTOR_INDEX
        assert params["k"] == 3
        assert params["vec"] == SAME

    async def test_returns_empty_when_embeddings_are_unavailable(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])

        def boom(*a, **kw):
            raise RuntimeError("no embedding backend")

        monkeypatch.setattr(chunk_store, "get_embeddings", boom)

        assert await chunk_store.search_chunks("ahmed", k=3) == []
        assert calls == []  # never even reached the database

    async def test_returns_empty_when_the_index_does_not_exist(self, monkeypatch, fake_neo4j):
        def handler(query, params):
            raise RuntimeError("There is no such vector schema index: chunk_embedding")

        fake_neo4j(handler)
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())

        assert await chunk_store.search_chunks("ahmed", k=3) == []

    @pytest.mark.parametrize(("query", "k"), [("", 3), ("   ", 3), ("ahmed", 0), ("ahmed", -1)])
    async def test_degenerate_arguments_short_circuit(self, query, k, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        assert await chunk_store.search_chunks(query, k=k) == []
        assert calls == []


# ── Entity-anchored lookup ───────────────────────────
class TestChunksForEntities:
    async def test_dedupes_by_id_and_orders_by_document_then_index(self, fake_neo4j):
        # Deliberately shuffled, with "c1" returned twice (two entities mention it).
        rows = [
            {"id": "c2", "text": "second", "document": "cv.pdf", "index": 1},
            {"id": "c1", "text": "first", "document": "cv.pdf", "index": 0},
            {"id": "c0", "text": "other doc", "document": "a.pdf", "index": 5},
            {"id": "c1", "text": "first", "document": "cv.pdf", "index": 0},
        ]
        fake_neo4j(lambda q, p: rows if "MENTIONED_IN" in q else [])

        out = await chunk_store.chunks_for_entities(["Ahmed", "Postgres"], limit=10)

        assert [c["id"] for c in out] == ["c0", "c1", "c2"]
        assert [c["document"] for c in out] == ["a.pdf", "cv.pdf", "cv.pdf"]
        assert all(c["score"] == 0.0 for c in out)

    async def test_ordering_is_independent_of_driver_row_order(self, fake_neo4j):
        rows = [
            {"id": "c1", "text": "first", "document": "cv.pdf", "index": 0},
            {"id": "c2", "text": "second", "document": "cv.pdf", "index": 1},
        ]

        fake_neo4j(lambda q, p: rows if "MENTIONED_IN" in q else [])
        forward = await chunk_store.chunks_for_entities(["Ahmed"], limit=10)

        rows.reverse()
        reversed_out = await chunk_store.chunks_for_entities(["Ahmed"], limit=10)

        assert forward == reversed_out

    async def test_respects_the_limit_after_dedupe(self, fake_neo4j):
        rows = [
            {"id": "c1", "text": "a", "document": "cv.pdf", "index": 0},
            {"id": "c1", "text": "a", "document": "cv.pdf", "index": 0},
            {"id": "c2", "text": "b", "document": "cv.pdf", "index": 1},
            {"id": "c3", "text": "c", "document": "cv.pdf", "index": 2},
        ]
        calls = fake_neo4j(lambda q, p: rows if "MENTIONED_IN" in q else [])

        out = await chunk_store.chunks_for_entities(["Ahmed"], limit=2)

        assert [c["id"] for c in out] == ["c1", "c2"]
        assert [p for q, p in calls if "MENTIONED_IN" in q][0]["limit"] == 2

    async def test_cleans_and_dedupes_the_requested_names(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        await chunk_store.chunks_for_entities([" Ahmed ", "Ahmed", "", "Postgres"], limit=5)
        assert [p for q, p in calls if "MENTIONED_IN" in q][0]["names"] == ["Ahmed", "Postgres"]

    @pytest.mark.parametrize(("names", "limit"), [([], 5), (["", "  "], 5), (["Ahmed"], 0)])
    async def test_degenerate_arguments_short_circuit(self, names, limit, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        assert await chunk_store.chunks_for_entities(names, limit=limit) == []
        assert calls == []

    async def test_database_failure_degrades_to_empty(self, fake_neo4j):
        def handler(query, params):
            raise RuntimeError("connection refused")

        fake_neo4j(handler)
        assert await chunk_store.chunks_for_entities(["Ahmed"], limit=5) == []


# ── Ingest wiring ────────────────────────────────────
_EXTRACTION = """{
  "entities": [
    {"name": "Ahmed", "type": "PERSON", "description": ""},
    {"name": "Ahmed Maaloul", "type": "PERSON", "description": "AI engineer"},
    {"name": "Postgres", "type": "TOOL", "description": "relational database"},
    {"name": "PostgreSQL", "type": "TOOL", "description": ""}
  ],
  "relationships": [
    {"source": "Ahmed", "target": "Postgres", "type": "USES_TOOL"}
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
    monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())


class TestBuildKnowledgeGraphStoresChunks:
    async def test_chunk_links_use_canonical_names_after_a_merge(self, monkeypatch, fake_neo4j):
        """The whole point: a merged duplicate's excerpts must follow the survivor."""
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)

        result = await graph_builder.build_knowledge_graph(
            ["Ahmed Maaloul uses PostgreSQL."], "cv.pdf", theme="Personal CV / Resume"
        )

        assert result["entities_merged"] == 2
        assert result["chunks_stored"] == 1
        # "Ahmed" and "Postgres" were absorbed; linking them would leave the
        # excerpt unreachable from the entities that survived.
        assert _link_calls(calls)[0]["names"] == ["Ahmed Maaloul", "PostgreSQL"]

    async def test_each_chunk_is_linked_to_its_own_entities(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        monkeypatch.setattr(graph_builder, "get_chat_llm", lambda **kw: object())
        monkeypatch.setattr(chunk_store, "get_embeddings", lambda *a, **kw: _FakeEmbeddings())
        monkeypatch.setenv("ENTITY_RESOLUTION_ENABLED", "false")

        per_chunk = {
            "Ada Lovelace wrote the first algorithm.": '{"entities": [{"name": "Ada Lovelace", "type": "PERSON"}], "relationships": []}',
            "Neo4j stores graphs.": '{"entities": [{"name": "Neo4j", "type": "DATABASE"}], "relationships": []}',
        }

        class _PerChunkPipe:
            def __or__(self, _other):
                class FakeChain:
                    async def ainvoke(self, inputs):
                        class R:
                            content = per_chunk[inputs["text"]]

                        return R()

                return FakeChain()

        monkeypatch.setattr(graph_builder, "get_extraction_prompt", lambda theme: _PerChunkPipe())

        await graph_builder.build_knowledge_graph(list(per_chunk), "history.pdf")

        writes = _write_calls(calls)
        links = _link_calls(calls)
        assert [w["index"] for w in writes] == [0, 1]
        assert [w["text"] for w in writes] == list(per_chunk)
        assert [link["names"] for link in links] == [["Ada Lovelace"], ["Neo4j"]]
        # Each link targets the chunk it came from, not some other passage.
        assert [link["id"] for link in links] == [w["id"] for w in writes]

    async def test_emits_a_storing_chunks_progress_event(self, monkeypatch, fake_neo4j):
        fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)

        events: list[dict] = []

        async def on_progress(event):
            events.append(event)

        await graph_builder.build_knowledge_graph(
            ["Ahmed Maaloul uses PostgreSQL."], "cv.pdf", on_progress=on_progress
        )

        storing = [e for e in events if e.get("stage") == "storing_chunks"]
        assert storing, "no storing_chunks progress event was emitted"
        assert all(e["type"] == "progress" for e in storing)
        assert all(e["total"] == 1 for e in storing)
        assert storing[-1]["processed"] == 1

    async def test_disabled_flag_skips_chunk_storage_entirely(self, monkeypatch, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        _install_fake_pipeline(monkeypatch)
        monkeypatch.setenv("STORE_SOURCE_CHUNKS", "false")

        events: list[dict] = []

        async def on_progress(event):
            events.append(event)

        result = await graph_builder.build_knowledge_graph(
            ["Ahmed Maaloul uses PostgreSQL."], "cv.pdf", on_progress=on_progress
        )

        assert result["chunks_stored"] == 0
        assert not any(":Chunk" in q for q, _ in calls)
        assert not any(e.get("stage") == "storing_chunks" for e in events)
        # The graph itself is unaffected by the flag.
        assert result["nodes_created"] == 2

    async def test_chunk_failure_never_breaks_the_ingest(self, monkeypatch, fake_neo4j):
        def handler(query, params):
            if ":Chunk" in query:
                raise RuntimeError("disk full")
            return []

        fake_neo4j(handler)
        _install_fake_pipeline(monkeypatch)

        result = await graph_builder.build_knowledge_graph(
            ["Ahmed Maaloul uses PostgreSQL."], "cv.pdf"
        )

        assert result["chunks_stored"] == 0
        assert result["nodes_created"] == 2  # the graph was still built


class TestCanonicalizeChunkNames:
    def test_applies_the_case_fold_then_the_alias_map(self):
        out = graph_builder._canonicalize_chunk_names(
            [["python", "Postgres"]],
            {"python": "Python", "postgres": "Postgres"},
            {"Postgres": "PostgreSQL"},
            {"Python", "PostgreSQL"},
        )
        assert out == [["Python", "PostgreSQL"]]

    def test_drops_names_that_were_never_written(self):
        out = graph_builder._canonicalize_chunk_names(
            [["Ghost", "Ahmed"]], {"ahmed": "Ahmed"}, {}, {"Ahmed"}
        )
        assert out == [["Ahmed"]]

    def test_dedupes_names_that_collapse_onto_the_same_node(self):
        out = graph_builder._canonicalize_chunk_names(
            [["Ahmed", "Ahmed Maaloul"]],
            {"ahmed": "Ahmed", "ahmed maaloul": "Ahmed Maaloul"},
            {"Ahmed": "Ahmed Maaloul"},
            {"Ahmed Maaloul"},
        )
        assert out == [["Ahmed Maaloul"]]

    def test_preserves_chunk_alignment(self):
        out = graph_builder._canonicalize_chunk_names(
            [["Ahmed"], [], ["Ghost"]], {"ahmed": "Ahmed"}, {}, {"Ahmed"}
        )
        assert out == [["Ahmed"], [], []]


# ── Isolation from the visualization ─────────────────
class TestChunksNeverLeakIntoTheGraphVisualization:
    """:Chunk nodes are storage. Rendered as graph nodes they are a wall of text."""

    async def test_graph_data_queries_are_scoped_to_entity(self, fake_neo4j):
        from app.routers import graph as graph_router

        def handler(query, params):
            # Any MATCH that is not label-scoped would sweep :Chunk nodes into
            # the force-graph payload.
            assert "MATCH (n:Entity)" in query or "MATCH (a:Entity)-[r]->(b:Entity)" in query, (
                f"unscoped graph-data query would pull in :Chunk nodes: {query}"
            )
            if "MATCH (n:Entity)" in query:
                return [
                    {
                        "id": "1",
                        "labels": ["Entity"],
                        "properties": {"name": "Ahmed", "type": "PERSON", "embedding": [0.1]},
                    }
                ]
            return []

        fake_neo4j(handler)

        data = await graph_router.get_graph_data()

        assert [n["label"] for n in data["nodes"]] == ["Ahmed"]
        assert data["links"] == []
        # And the vectors stay on the server.
        assert "embedding" not in data["nodes"][0]["properties"]

    def test_chunk_writes_never_use_the_entity_label(self):
        for query in (
            chunk_store.WRITE_CHUNK_QUERY,
            chunk_store.WRITE_CHUNK_FALLBACK_QUERY,
        ):
            assert ":Entity" not in query
            assert ":Chunk" in query
