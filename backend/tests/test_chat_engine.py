"""Tests for GraphRAG retrieval, formatting, history, and streaming."""

from app.services import chat_engine
from app.services.chat_engine import (
    _keyword_query,
    _to_lc_history,
    generate_rag_response,
    retrieve_subgraph,
)


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
        assert citations == [{"name": "Ada", "type": "PERSON"}]

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


class TestStreaming:
    async def test_events_order_citations_tokens_done(self, monkeypatch):
        async def fake_retrieve(q, k=8):
            return "CONTEXT", [{"name": "Ada", "type": "PERSON"}]

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", fake_retrieve)

        class Chunk:
            def __init__(self, c):
                self.content = c

        class FakeChain:
            async def astream(self, _inputs):
                for tok in ["Hello", " ", "world"]:
                    yield Chunk(tok)

        # RAG_PROMPT | llm  -> FakeChain
        monkeypatch.setattr(chat_engine, "get_chat_llm", lambda **kw: object())
        monkeypatch.setattr(chat_engine, "RAG_PROMPT", _PipeTo(FakeChain()))

        events = [e async for e in generate_rag_response("hi", [])]
        types = [e["type"] for e in events]
        assert types[0] == "citations"
        assert types[-1] == "done"
        assert "".join(e["data"] for e in events if e["type"] == "token") == "Hello world"

    async def test_retrieval_error_yields_error_event(self, monkeypatch):
        async def boom(q, k=8):
            raise RuntimeError("db down")

        monkeypatch.setattr(chat_engine, "retrieve_subgraph", boom)
        events = [e async for e in generate_rag_response("hi", [])]
        assert events == [{"type": "error", "data": "Failed to query the knowledge graph."}]


class _PipeTo:
    """Object whose ``| x`` returns a preset chain (mimics RAG_PROMPT | llm)."""

    def __init__(self, chain):
        self._chain = chain

    def __or__(self, _other):
        return self._chain
