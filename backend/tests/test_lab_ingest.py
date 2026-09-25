# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Batch ingest: the graph_builder split, and plan → submit → apply over fakes.

Hermetic: the fake Batch service from ``test_lab_batch`` answers the
extraction requests, ``fake_neo4j`` records every statement, embeddings are the
deterministic fake provider. Three properties matter most and are asserted
directly:

  • the split keeps ``build_knowledge_graph`` byte-for-byte: the same replies
    produce the same Neo4j statements (parameters included) and the same
    return dict through either entry point;
  • the Batch requests are the requests the realtime pipeline sends —
    compared against ``ChatOpenAI``'s own request payload, built offline;
  • a corpus applied this way passes the Lab's HotpotQA "graph is ingested"
    check.
"""

from __future__ import annotations

import json
import re

import pytest

from app.config import Settings, get_settings
from app.lab import arms as lab_arms
from app.lab import batch, ingest, runner, tokens
from app.lab.estimate import IngestPlan
from app.lab.runner import DatasetSpec, GraphNotIngested
from app.services import chunk_store, communities, graph_builder
from benchmarks.public import cost, hotpotqa, run_hotpotqa
from tests.test_lab_batch import FakeBatchService

REPLY = json.dumps({
    "entities": [
        {"name": "Ada Lovelace", "type": "PERSON", "description": "mathematician"},
        {"name": "Analytical Engine", "type": "THING", "description": "a machine"},
        {"name": "ada lovelace ", "type": "PERSON", "description": "duplicate by case"},
    ],
    "relationships": [
        {"source": "Ada Lovelace", "target": "Analytical Engine", "type": "RELATED_TO",
         "description": "worked on"},
    ],
})


# ── The graph_builder split ──────────────────────────────────────────────────
class _Chain:
    def __init__(self, reply):
        self.reply = reply

    async def ainvoke(self, _inputs):
        return type("R", (), {"content": self.reply})()


class _Prompt:
    def __init__(self, reply):
        self.reply = reply

    def __or__(self, _llm):
        return _Chain(self.reply)


class TestGraphBuilderSplit:
    async def test_same_replies_same_writes_same_result(self, monkeypatch, fake_neo4j):
        monkeypatch.setattr(graph_builder, "get_chat_llm", lambda **kw: object())
        monkeypatch.setattr(graph_builder, "get_extraction_prompt", lambda t: _Prompt(REPLY))
        chunks = ["Ada Lovelace worked on the Analytical Engine."]

        realtime_calls = fake_neo4j(lambda q, p: [])
        realtime_events: list[dict] = []

        async def on_rt(e):
            realtime_events.append(e)

        realtime = await graph_builder.build_knowledge_graph(chunks, "history.pdf", "Generic",
                                                             on_progress=on_rt)

        split_calls = fake_neo4j(lambda q, p: [])
        split_events: list[dict] = []

        async def on_split(e):
            split_events.append(e)

        split = await graph_builder.build_knowledge_graph_from_extractions(
            chunks, [REPLY], "history.pdf", "Generic", on_progress=on_split
        )

        assert split == realtime
        assert realtime["unique_entities"] == 2 and realtime["chunks_stored"] == 1
        assert list(split_calls) == list(realtime_calls) and len(split_calls) > 3
        # Identical events apart from the extraction stage the split skips.
        assert split_events == [e for e in realtime_events if e.get("stage") != "extracting"]

    async def test_a_failed_extraction_still_stores_the_chunk(self, fake_neo4j):
        calls = fake_neo4j(lambda q, p: [])
        result = await graph_builder.build_knowledge_graph_from_extractions(
            ["Some text."], [None], "doc", "Generic"
        )
        assert result["unique_entities"] == 0 and result["chunks_stored"] == 1
        assert any(q == chunk_store.WRITE_CHUNK_QUERY and p["document"] == "doc"
                   for q, p in calls)

    def test_parse_extraction_uses_the_pipeline_parser(self):
        fenced = f"```json\n{REPLY}\n```"
        assert graph_builder.parse_extraction(fenced) == graph_builder._parse_llm_json(fenced)
        assert graph_builder.parse_extraction({"entities": None}) == {
            "entities": [], "relationships": []}
        assert graph_builder.parse_extraction(None) == {"entities": [], "relationships": []}
        assert graph_builder.parse_extraction("not json") == {"entities": [], "relationships": []}
        with pytest.raises(TypeError):
            graph_builder.parse_extraction(42)

    async def test_one_extraction_per_chunk(self):
        with pytest.raises(ValueError, match="one extraction per chunk"):
            await graph_builder.build_knowledge_graph_from_extractions(["a", "b"], [REPLY], "d")


# ── Request parity with the realtime pipeline ────────────────────────────────
def realtime_payload(model: str, text: str, name: str, theme: str) -> dict:
    """What ``build_knowledge_graph``'s ``prompt | llm`` would POST — computed offline."""
    from app.services.llm_provider import get_chat_llm

    settings = Settings(llm_provider="openai", openai_api_key="sk-test-not-real",
                        openai_chat_model=model)
    llm = get_chat_llm(json_mode=True, temperature=settings.extraction_temperature,
                       settings=settings)
    prompt = graph_builder.get_extraction_prompt(theme).invoke(
        {"text": text, "theme": theme, "document_name": name}
    )
    payload = llm._get_request_payload(prompt)
    payload.pop("stream", None)  # transport detail; a Batch request is never streamed
    return payload, settings


class TestRequestParity:
    @pytest.mark.parametrize("theme", ["Generic", "AI Safety"])
    def test_messages_are_the_realtime_messages(self, theme):
        expected, _ = realtime_payload("gpt-4o-mini", "Ada\nAda did X.", "Ada", theme)
        rendered = graph_builder.render_extraction_request("Ada\nAda did X.", "Ada", theme)
        assert rendered == expected["messages"]

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5-nano"])
    def test_body_is_the_realtime_body(self, model):
        expected, settings = realtime_payload(model, "Ada\nAda did X.", "Ada", "Generic")
        body = ingest.extraction_request_body("Ada\nAda did X.", "Ada", model=model,
                                              theme="Generic", settings=settings)
        assert body == expected
        assert ("temperature" in body) != ("reasoning_effort" in body)

    def test_an_output_cap_is_opt_in(self):
        body = ingest.extraction_request_body("t", "d", max_output_tokens=900)
        assert body["max_tokens"] == 900
        assert "max_tokens" not in ingest.extraction_request_body("t", "d")


# ── Corpus ───────────────────────────────────────────────────────────────────
def hotpot_item(qid: str, gold: tuple[str, str], extra: tuple[str, ...]) -> dict:
    titles = (*gold, *extra)
    return {
        "_id": qid, "question": f"Who founded {gold[0]}?", "answer": "yes", "type": "bridge",
        "level": "hard",
        "context": [[t, [f"{t} is a thing. ", f"{t} was founded in 19{i}0. "]]
                    for i, t in enumerate(titles)],
        "supporting_facts": [[gold[0], 0], [gold[1], 1]],
    }


@pytest.fixture
def hotpot_file(tmp_path):
    path = tmp_path / "hotpot.json"
    items = [hotpot_item(f"q{i}", (f"A{i}", f"B{i}"), ("Shared", f"C{i}")) for i in range(6)]
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def spec(path, **kw) -> DatasetSpec:
    return DatasetSpec("hotpotqa", n=kw.pop("n", 2), path=str(path), seed=kw.pop("seed", 7),
                       allow_download=False, **kw)


class TestCorpus:
    def test_documents_follow_the_harness_order(self):
        corpus = run_hotpotqa.build_corpus(
            [hotpotqa.parse_record(hotpot_item("q", ("B", "A"), ("C",)), index=0)]
        )
        docs = ingest.corpus_documents(corpus)
        assert [d.name for d in docs] == corpus.titles == ["A", "B", "C"]
        assert docs[0].text == run_hotpotqa.paragraph_text(corpus.paragraphs["A"])

    def test_other_shapes_and_bad_input(self):
        assert [d.name for d in ingest.corpus_documents({"x": "1", "y": "2"})] == ["x", "y"]
        assert ingest.corpus_documents([("x", "1")]) == [ingest.Document("x", "1")]
        with pytest.raises(ingest.IngestError, match="duplicate"):
            ingest.corpus_documents([("x", "1"), ("x", "2")])
        with pytest.raises(ingest.IngestError, match="empty"):
            ingest.corpus_documents({})
        with pytest.raises(ingest.IngestError, match="non-empty"):
            ingest.corpus_documents({"x": "  "})

    def test_the_corpus_is_the_runners_sample(self, hotpot_file):
        for kw in ({}, {"offset": 2}):
            s = spec(hotpot_file, **kw)
            corpus = ingest.load_hotpotqa_corpus(s)
            assert corpus.titles == runner._load_hotpotqa(s).titles
        disjoint = ingest.load_hotpotqa_corpus(spec(hotpot_file, offset=2))
        first = ingest.load_hotpotqa_corpus(spec(hotpot_file))
        assert {q.id for q in disjoint.questions}.isdisjoint({q.id for q in first.questions})


# ── Plan ─────────────────────────────────────────────────────────────────────
class _WordEncoding:
    """A deterministic stand-in for a tiktoken ``Encoding``: one token per word.

    The counted path must not depend on tiktoken's cache being warm (a fresh CI
    runner has none, and tests never download it), so it runs on this.
    """

    def encode(self, text, disallowed_special=()):
        return text.split()


class TestPlan:
    def test_counts_real_prompt_tokens_and_prices_at_batch_rate(self, monkeypatch):
        monkeypatch.setattr(tokens, "get_encoding", lambda _model: _WordEncoding())
        corpus = {"A": "A\nAlpha is a thing.", "B": "B\nBeta was founded in 1910."}
        report = ingest.plan_ingest(corpus, calibration={})
        bodies = [ingest.extraction_request_body(t, n) for n, t in corpus.items()]
        from app.lab import reader

        assert report.documents == 2
        assert report.prompt_tokens == sum(reader.request_prompt_tokens(b, "gpt-4o-mini")[0]
                                           for b in bodies)
        assert report.plan.prompt_tokens_per_paragraph == pytest.approx(report.prompt_tokens / 2)
        assert report.plan.completion_tokens_per_paragraph == 464.0
        assert report.extraction.calls == 2 and report.plan.batch
        assert 0 < report.point_usd < report.upper_usd
        realtime = ingest.plan_ingest(corpus, batch=False, calibration={})
        assert report.point_usd == pytest.approx(realtime.point_usd * cost.BATCH_PRICE_MULTIPLIER)
        assert report.summaries is None and "OFF" in " ".join(report.notes)
        assert isinstance(report.plan, IngestPlan) and report.to_dict()["upper_usd"]

    def test_without_a_tokenizer_the_measured_mean_is_used_and_flagged(self, monkeypatch):
        monkeypatch.setattr(tokens, "get_encoding", lambda _model: None)
        report = ingest.plan_ingest({"A": "A\nAlpha.", "B": "B\nBeta."}, calibration={})
        assert report.prompt_tokens_estimated is True
        assert report.tokenizer.startswith("estimate:")
        assert report.plan.prompt_tokens_per_paragraph == 434.0
        assert any("tokenizer unavailable" in n for n in report.notes)

    def test_calibration_output_cap_and_summaries(self):
        corpus = {"A": "A\nAlpha."}
        calibrated = ingest.plan_ingest(corpus, calibration={"ingest": {
            "model": "gpt-4o-mini", "prompt_tokens_per_paragraph": 1,
            "completion_tokens_per_paragraph": 999, "measured_on": "pilot"}})
        assert calibrated.plan.completion_tokens_per_paragraph == 999
        capped = ingest.plan_ingest(corpus, calibration={}, max_output_tokens=100)
        assert capped.extraction.upper_completion_tokens == 100
        with_summaries = ingest.plan_ingest(corpus, calibration={}, community_summaries=True)
        assert with_summaries.summaries.calls >= 1 and not with_summaries.summaries.batch
        assert "ASSUMED" in with_summaries.summaries.notes[0]


# ── Submit / apply ───────────────────────────────────────────────────────────
def extraction_reply(body) -> str:
    """A deterministic 'model': two entities and one edge per paragraph title."""
    text = body["messages"][1]["content"].split("\n\n", 1)[1]
    title = text.split("\n", 1)[0]
    return json.dumps({
        "entities": [{"name": title, "type": "THING", "description": f"{title} itself"},
                     {"name": f"{title} founder", "type": "PERSON", "description": "founder"}],
        "relationships": [{"source": f"{title} founder", "target": title,
                           "type": "RELATED_TO", "description": "founded"}],
    })


WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH)\b")


class GraphRecorder:
    """A fake Neo4j that remembers the chunk documents written and serves them back.

    With ``allow_writes=False`` every write statement is recorded in
    ``violations`` (not raised: the pipeline swallows write errors by design, so
    a raise would be silently logged instead of failing the test).
    """

    def __init__(self, fake_neo4j, *, community_rows=None, allow_writes=True):
        self.documents: list[str] = []
        self.violations: list[str] = []
        self.allow_writes = allow_writes
        self.community_rows = community_rows or []
        self.calls = fake_neo4j(self.handle)

    def handle(self, q, p):
        if not self.allow_writes and WRITE.search(q):
            self.violations.append(q)
            return []
        if q in (chunk_store.WRITE_CHUNK_QUERY, chunk_store.WRITE_CHUNK_FALLBACK_QUERY):
            self.documents.append(p["document"])
        if q == run_hotpotqa.CHUNK_DOCUMENTS_QUERY:
            return [{"document": d} for d in dict.fromkeys(self.documents)]
        if q == communities.LOAD_GRAPH_QUERY:
            return self.community_rows
        for _key, query in lab_arms.FINGERPRINT_QUERIES.items():
            if q == query:
                return [{"n": len(self.documents)}]
        return []


@pytest.fixture(autouse=True)
def _no_calibration_file(monkeypatch):
    monkeypatch.setattr(ingest, "load_calibration", lambda path=None: {})


class TestSubmit:
    async def test_one_request_per_document_with_the_pipeline_prompt(self, tmp_path,
                                                                     hotpot_file):
        corpus = ingest.load_hotpotqa_corpus(spec(hotpot_file))
        svc = FakeBatchService(answer=extraction_reply)
        out = await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path,
                                               client=svc.client, max_usd=1.0)
        sent = [json.loads(x) for x in svc.uploads[0]["bytes"].decode().splitlines()]
        assert out["requests"] == len(corpus.titles) == len(sent)
        run_id = out["run_id"]
        for i, (title, request) in enumerate(zip(corpus.titles, sent, strict=True)):
            assert request["custom_id"] == f"{run_id}|ingest|extract|{i:06d}"
            assert request["body"] == ingest.extraction_request_body(corpus.text(title), title)
        assert svc.created[0]["metadata"]["synapse_phase"] == "ingest"
        manifest = ingest.load_manifest(tmp_path)
        assert manifest["status"] == "submitted" and manifest["model"] == "gpt-4o-mini"
        assert [d.name for d in ingest.load_documents(tmp_path)] == corpus.titles
        # The reader's batch state (if the Lab run shares this dir) is untouched.
        assert not (tmp_path / batch.STATE_FILE).exists()
        assert (tmp_path / "ingest" / batch.STATE_FILE).exists()

    async def test_refused_above_the_cap_before_anything_is_written(self, tmp_path):
        svc = FakeBatchService()
        with pytest.raises(ingest.IngestRefused, match="exceeds the cap") as err:
            await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "gpt-4o-mini", tmp_path,
                                             client=svc.client, max_usd=1e-9)
        assert err.value.plan.upper_usd > 1e-9
        assert svc.uploads == [] and not (tmp_path / "ingest").exists()
        with pytest.raises(ingest.IngestRefused, match="no price"):
            await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "mystery-model", tmp_path,
                                             client=svc.client, max_usd=1.0)

    async def test_resubmission_is_idempotent_and_a_foreign_corpus_refused(self, tmp_path):
        svc = FakeBatchService()
        corpus = {"A": "A\nAlpha.", "B": "B\nBeta."}
        await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path, client=svc.client)
        await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path, client=svc.client)
        assert len(svc.created) == 1
        with pytest.raises(ingest.IngestError, match="different ingest"):
            await ingest.submit_ingest_batch({"A": "A\nOther."}, "gpt-4o-mini", tmp_path,
                                             client=svc.client)
        # A new run id would re-key every request and pay for the batch twice.
        with pytest.raises(ingest.IngestError, match="resume it under that run id"):
            await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path,
                                             client=svc.client, run_id="another")
        assert len(svc.created) == 1

    async def test_never_builds_a_real_client_under_pytest(self, tmp_path):
        with pytest.raises(RuntimeError, match="refusing to create a real OpenAI client"):
            await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "gpt-4o-mini", tmp_path)


class TestApply:
    async def test_pending_batch_writes_nothing(self, tmp_path, fake_neo4j):
        svc = FakeBatchService(answer=extraction_reply, polls_to_finish=3)
        await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "gpt-4o-mini", tmp_path,
                                         client=svc.client)
        graph = GraphRecorder(fake_neo4j, allow_writes=False)
        out = await ingest.apply_ingest_batch(tmp_path, client=svc.client)
        assert out["status"] == "pending" and graph.violations == [] and graph.calls == []

    async def test_applied_corpus_passes_the_hotpotqa_graph_check(self, tmp_path, fake_neo4j,
                                                                  hotpot_file, monkeypatch):
        s = spec(hotpot_file)
        corpus = ingest.load_hotpotqa_corpus(s)
        svc = FakeBatchService(answer=extraction_reply)
        await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path, client=svc.client)
        graph = GraphRecorder(fake_neo4j)
        monkeypatch.setattr(communities, "get_chat_llm", _explode("community summaries"))

        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client)

        assert manifest["status"] == "applied"
        # One document per paragraph, written one at a time in corpus order.
        assert graph.documents == corpus.titles
        assert manifest["stats"]["documents"] == len(corpus.titles)
        assert manifest["stats"]["failed_extractions"] == 0
        assert manifest["verified"]["missing_count"] == 0
        assert manifest["communities"]["summaries"] is False
        expected = cost.usd(cost.Usage(prompt_tokens=100 * len(corpus.titles),
                                       completion_tokens=20 * len(corpus.titles)),
                            cost.resolve_price("gpt-4o-mini"), batch=True)
        assert manifest["actual"]["usd"] == pytest.approx(expected)
        assert ingest.ingest_usd(tmp_path) == pytest.approx(expected)

        # The Lab's own gate accepts this graph for this sample …
        dataset = runner._load_hotpotqa(s)
        await runner.check_graph_ready(dataset)
        # … and still refuses a sample it does not hold.
        with pytest.raises(GraphNotIngested):
            await runner.check_graph_ready(runner._load_hotpotqa(spec(hotpot_file, offset=2)))

        # Re-applying an applied ingest is a no-op.
        before = len(graph.calls)
        again = await ingest.apply_ingest_batch(tmp_path, client=svc.client)
        assert again["status"] == "applied" and len(graph.calls) == before

    async def test_writes_through_the_shared_post_extraction_code(self, tmp_path, fake_neo4j,
                                                                  monkeypatch):
        seen = []
        original = graph_builder.build_knowledge_graph_from_extractions

        async def spy(chunks, extractions, filename, theme="Generic", on_progress=None):
            seen.append((chunks, extractions, filename, theme))
            return await original(chunks, extractions, filename, theme, on_progress)

        monkeypatch.setattr(graph_builder, "build_knowledge_graph_from_extractions", spy)
        svc = FakeBatchService(answer=extraction_reply)
        corpus = {"A": "A\nAlpha.", "B": "B\nBeta."}
        await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path, client=svc.client)
        GraphRecorder(fake_neo4j)
        await ingest.apply_ingest_batch(tmp_path, client=svc.client, detect_communities=False)
        assert [(c, f, t) for c, _e, f, t in seen] == [(["A\nAlpha."], "A", "Generic"),
                                                       (["B\nBeta."], "B", "Generic")]
        assert [e[0] for _c, e, _f, _t in seen] == [
            extraction_reply({"messages": [{}, {"content": "x\n\nA\nAlpha."}]}),
            extraction_reply({"messages": [{}, {"content": "x\n\nB\nBeta."}]}),
        ]

    async def test_failed_extractions_block_apply_until_resent(self, tmp_path, fake_neo4j):
        corpus = {"A": "A\nAlpha.", "B": "B\nBeta.", "C": "C\nGamma."}
        svc = FakeBatchService(answer=extraction_reply)
        out = await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path,
                                               client=svc.client)
        svc.fail = {f"{out['run_id']}|ingest|extract|000001"}
        graph = GraphRecorder(fake_neo4j, allow_writes=False)
        with pytest.raises(ingest.IngestIncomplete, match="1 of 3") as err:
            await ingest.apply_ingest_batch(tmp_path, client=svc.client)
        assert err.value.failed == ["B"] and graph.violations == []

        svc.fail.clear()
        info = await ingest.resubmit_failed_ingest(tmp_path, client=svc.client)
        assert info["requests"] == 1
        graph.allow_writes = True
        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                                   detect_communities=False)
        assert manifest["status"] == "applied" and graph.documents == ["A", "B", "C"]
        assert manifest["stats"]["failed_extractions"] == 0

    async def test_allow_failed_stores_the_document_without_entities(self, tmp_path,
                                                                     fake_neo4j):
        svc = FakeBatchService(answer=extraction_reply, expire_after=1)
        await ingest.submit_ingest_batch({"A": "A\nAlpha.", "B": "B\nBeta."}, "gpt-4o-mini",
                                         tmp_path, client=svc.client)
        graph = GraphRecorder(fake_neo4j)
        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                                   allow_failed=True, detect_communities=False)
        assert graph.documents == ["A", "B"]
        assert manifest["stats"]["failed_extractions"] == 1
        applied = [json.loads(x) for x in
                   (tmp_path / "ingest" / ingest.APPLIED).read_text().splitlines()]
        assert [a["extraction"] for a in applied] == ["ok", "failed"]
        assert applied[1]["result"]["unique_entities"] == 0

    async def test_an_interrupted_apply_resumes_where_it_stopped(self, tmp_path, fake_neo4j,
                                                                 monkeypatch):
        svc = FakeBatchService(answer=extraction_reply)
        corpus = {"A": "A\nAlpha.", "B": "B\nBeta.", "C": "C\nGamma."}
        await ingest.submit_ingest_batch(corpus, "gpt-4o-mini", tmp_path, client=svc.client)
        graph = GraphRecorder(fake_neo4j)
        original = graph_builder.build_knowledge_graph_from_extractions
        written = []

        async def crash_on_c(chunks, extractions, filename, theme="Generic", on_progress=None):
            if filename == "C" and not written.count("crashed"):
                written.append("crashed")
                raise RuntimeError("database went away")
            written.append(filename)
            return await original(chunks, extractions, filename, theme, on_progress)

        monkeypatch.setattr(graph_builder, "build_knowledge_graph_from_extractions", crash_on_c)
        with pytest.raises(RuntimeError, match="went away"):
            await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                            detect_communities=False)
        assert ingest.load_manifest(tmp_path)["status"] == "applying"
        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                                   detect_communities=False)
        assert written == ["A", "B", "crashed", "C"]  # A and B were not written twice
        assert graph.documents == ["A", "B", "C"] and manifest["stats"]["documents"] == 3

    async def test_documents_with_line_separators_round_trip(self, tmp_path, fake_neo4j):
        text = "A\nAlpha\u2028beta\x85gamma."
        svc = FakeBatchService(answer=extraction_reply)
        await ingest.submit_ingest_batch({"A": text}, "gpt-4o-mini", tmp_path, client=svc.client)
        assert ingest.load_documents(tmp_path) == [ingest.Document("A", text)]
        graph = GraphRecorder(fake_neo4j)
        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                                   detect_communities=False)
        assert manifest["status"] == "applied" and graph.documents == ["A"]

    async def test_refuses_without_source_chunks(self, tmp_path, fake_neo4j, monkeypatch):
        svc = FakeBatchService(answer=extraction_reply)
        await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "gpt-4o-mini", tmp_path,
                                         client=svc.client)
        graph = GraphRecorder(fake_neo4j, allow_writes=False)
        monkeypatch.setenv("STORE_SOURCE_CHUNKS", "false")
        get_settings.cache_clear()
        with pytest.raises(ingest.IngestError, match="STORE_SOURCE_CHUNKS"):
            await ingest.apply_ingest_batch(tmp_path, client=svc.client)
        assert graph.violations == []


def _explode(what):
    def boom(*a, **kw):
        raise AssertionError(f"{what} must not call an LLM")

    return boom


class TestCommunities:
    ROWS = [
        {"source": s, "target": t, "rel_type": "RELATED_TO", "source_type": "THING",
         "source_description": "", "target_type": "THING", "target_description": ""}
        for s, t in [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z")]
    ]

    async def test_detection_without_summaries_makes_no_llm_call(self, fake_neo4j, monkeypatch):
        monkeypatch.setattr(communities, "get_chat_llm", _explode("detection"))
        calls = fake_neo4j(lambda q, p: self.ROWS if q == communities.LOAD_GRAPH_QUERY else [])
        result = await ingest.detect_communities_without_summaries()
        assert result["communities"] == 2 and result["summarized"] == 0
        writes = [p for q, p in calls if q == communities.WRITE_COMMUNITY_QUERY]
        assert {w["summary"] for w in writes} == {"A cluster of 3 closely related entities."}
        assert all(w["title"] == communities._fallback_title(w["members"]) for w in writes)

    async def test_summaries_flag_runs_the_product_path(self, tmp_path, fake_neo4j,
                                                        monkeypatch):
        called = []

        async def fake_detect_and_summarize(on_progress=None):
            called.append(True)
            return {"communities": 1, "summarized": 1, "modularity": 0.5}

        monkeypatch.setattr(communities, "detect_and_summarize", fake_detect_and_summarize)
        svc = FakeBatchService(answer=extraction_reply)
        await ingest.submit_ingest_batch({"A": "A\nAlpha."}, "gpt-4o-mini", tmp_path,
                                         client=svc.client)
        GraphRecorder(fake_neo4j)
        manifest = await ingest.apply_ingest_batch(tmp_path, client=svc.client,
                                                   community_summaries=True)
        assert called and manifest["communities"]["summaries"] is True
