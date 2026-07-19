# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for extraction parsing, dedup, and the ingest orchestration."""


from app.services import graph_builder
from app.services.graph_builder import (
    _dedupe_entities,
    _parse_llm_json,
    get_extraction_prompt,
)


class TestParseLLMJson:
    def test_plain_json(self):
        out = _parse_llm_json('{"entities": [{"name": "Ada"}], "relationships": []}')
        assert out["entities"] == [{"name": "Ada"}]
        assert out["relationships"] == []

    def test_markdown_fenced_json(self):
        raw = '```json\n{"entities": [{"name": "Ada"}], "relationships": []}\n```'
        assert _parse_llm_json(raw)["entities"][0]["name"] == "Ada"

    def test_json_with_prose_around_it(self):
        raw = 'Sure! Here is the JSON:\n{"entities": [], "relationships": [{"source":"A","target":"B","type":"X"}]}\nHope that helps.'
        out = _parse_llm_json(raw)
        assert out["relationships"][0]["source"] == "A"

    def test_malformed_returns_empty_shape(self):
        out = _parse_llm_json("not json at all {oops")
        assert out == {"entities": [], "relationships": []}

    def test_empty_input(self):
        assert _parse_llm_json("") == {"entities": [], "relationships": []}

    def test_non_dict_json(self):
        assert _parse_llm_json("[1, 2, 3]") == {"entities": [], "relationships": []}


class TestDedupe:
    def test_collapses_case_and_whitespace(self):
        entities = [
            {"name": "Python", "type": "TOOL"},
            {"name": " python ", "type": "SKILL"},
            {"name": "PYTHON", "type": "LANGUAGE"},
        ]
        out = _dedupe_entities(entities)
        assert len(out) == 1  # last one wins
        assert out[0]["type"] == "LANGUAGE"

    def test_drops_unnamed(self):
        assert _dedupe_entities([{"type": "X"}, {"name": ""}]) == []

    def test_keeps_distinct(self):
        out = _dedupe_entities([{"name": "A"}, {"name": "B"}])
        assert len(out) == 2


class TestPromptSchema:
    def test_theme_specific_types_present(self):
        prompt = get_extraction_prompt("Medical/Scientific")
        rendered = prompt.format(text="t", theme="Medical/Scientific", document_name="d")
        assert "DISEASE" in rendered
        assert "TREATS" in rendered

    def test_unknown_theme_falls_back_to_generic(self):
        prompt = get_extraction_prompt("Nonexistent Theme")
        rendered = prompt.format(text="t", theme="x", document_name="d")
        assert "CONCEPT" in rendered  # from the Generic schema


class TestBuildKnowledgeGraph:
    async def test_end_to_end_with_mocks(self, monkeypatch, fake_neo4j):
        """Extraction + write path with a fake LLM chain and fake Neo4j."""
        fake_neo4j(lambda q, p: [])

        # Fake the LLM chain: prompt | llm  ->  object with .ainvoke returning content.
        class FakeResponse:
            content = '{"entities": [{"name": "Ada", "type": "PERSON", "description": "mathematician"}], "relationships": [{"source": "Ada", "target": "Analytical Engine", "type": "WORKED_ON"}]}'

        class FakeChain:
            async def ainvoke(self, _inputs):
                return FakeResponse()

        monkeypatch.setattr(graph_builder, "get_chat_llm", lambda **kw: object())
        monkeypatch.setattr(
            graph_builder, "get_extraction_prompt", lambda theme: _PipeStub()
        )

        events = []

        async def on_progress(e):
            events.append(e)

        result = await graph_builder.build_knowledge_graph(
            ["Ada Lovelace worked on the Analytical Engine."],
            "history.pdf",
            theme="Generic",
            on_progress=on_progress,
        )

        assert result["nodes_created"] == 1
        assert result["unique_entities"] == 1
        # progress events for extracting + writing stages were emitted
        stages = {e.get("stage") for e in events}
        assert "extracting" in stages
        assert "writing_nodes" in stages


class _PipeStub:
    """Stands in for a ChatPromptTemplate so `prompt | llm` yields a fake chain."""

    def __or__(self, _other):
        class FakeResponse:
            content = '{"entities": [{"name": "Ada", "type": "PERSON", "description": "mathematician"}], "relationships": [{"source": "Ada", "target": "Analytical Engine", "type": "WORKED_ON"}]}'

        class FakeChain:
            async def ainvoke(self, _inputs):
                return FakeResponse()

        return FakeChain()
