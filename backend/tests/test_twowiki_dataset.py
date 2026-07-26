# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for 2WikiMultihopQA acquisition + the deterministic capped sampler.

Hermetic by construction: every test runs against a tiny inline fixture that
mirrors the real 2Wiki schema, and :class:`TestNoNetworkInTheHotPath` proves it by
booby-trapping ``urllib.request.urlopen`` — if any parsing/sampling/stats path
ever grows a download, that test fails rather than quietly costing bandwidth.

2Wiki is HotpotQA-shaped, so ``twowiki`` reuses ``hotpotqa``'s parser, sampler and
stats. What is genuinely new — and therefore what these tests concentrate on — is:

1. **Acquisition + fallback.** The datasets-server rows API is probed first and,
   being down for this dataset, the canonical ``dev.json`` is fetched instead.
   Both paths, and the choice between them, are covered.
2. **The four-value ``type`` field**, including that ``bridge_comparison`` carries
   FOUR gold paragraphs, not two — the one real difference from HotpotQA that a
   downstream "all-gold" metric has to survive.
3. **Determinism + the hard cap** — a fixed ``(seed, n)`` yields the same ids, and
   ``n`` bounds the output, which is the only thing between a later phase and an
   unbounded LLM bill.

The one test that needs the real ~56 MB file is skip-guarded on ``SYNAPSE_2WIKI=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.public import twowiki
from benchmarks.public.twowiki import (
    EXPECTED_DEV_RECORDS,
    HOP_TYPES,
    CorpusStats,
    Provenance,
    corpus_stats,
    format_report,
    load_raw_records,
    load_sample,
    main,
    parse_records,
    sample_questions,
)

# ── A tiny fixture mirroring the real 2Wiki schema ───────
# Two gold paragraphs for comparison/inference/compositional; FOUR for
# bridge_comparison. ``evidences`` is present (2Wiki carries it) and there is no
# ``level`` field (2Wiki has none) — both facts the parser must tolerate.


def _record(qid: str, *, hop_type: str = "compositional", extra_paragraphs: int = 1) -> dict:
    context = [
        ["Alpha Corp", ["Alpha Corp is a firm.", "It was founded in 1990."]],
        ["Beta Ltd", ["Beta Ltd makes widgets.", "Beta Ltd was founded in 1985."]],
    ]
    supporting = [["Alpha Corp", 1], ["Beta Ltd", 1]]
    evidences = [["Alpha Corp", "inception", "1990"], ["Beta Ltd", "inception", "1985"]]
    if hop_type == "bridge_comparison":
        context += [
            ["Gamma Inc", ["Gamma Inc sells gadgets.", "Gamma Inc was founded in 1970."]],
            ["Delta LLC", ["Delta LLC ships freight.", "Delta LLC was founded in 1960."]],
        ]
        supporting += [["Gamma Inc", 1], ["Delta LLC", 1]]
        evidences += [["Gamma Inc", "inception", "1970"], ["Delta LLC", "inception", "1960"]]
    for i in range(extra_paragraphs):
        context.append([f"Distractor {i}", [f"Irrelevant sentence {i}."]])
    return {
        "_id": qid,
        "type": hop_type,
        "question": f"Which firm is older? ({qid})",
        "context": context,
        "supporting_facts": supporting,
        "evidences": evidences,
        "answer": "Beta Ltd",
    }


# A representative mix of all four 2Wiki reasoning types.
_TYPE_CYCLE = ("comparison", "inference", "compositional", "bridge_comparison")


@pytest.fixture
def raw_records() -> list[dict]:
    """12 well-formed records spanning every 2Wiki type."""
    return [_record(f"{i:024x}", hop_type=_TYPE_CYCLE[i % 4]) for i in range(12)]


@pytest.fixture
def questions(raw_records):
    parsed, skipped = parse_records(raw_records)
    assert skipped == []
    return parsed


# ── 1. Parsing and the 2Wiki type field ──────────────────
class TestParsing:
    def test_the_fixture_parses_into_the_normalised_shape(self, questions):
        assert len(questions) == 12
        q = questions[0]
        assert q.answer == "Beta Ltd"
        assert q.hop_type in HOP_TYPES
        # 2Wiki has no difficulty level; the parser must default it, not crash.
        assert q.level == "unknown"

    def test_bridge_comparison_carries_four_gold_paragraphs(self):
        """The one real departure from HotpotQA's two-gold invariant."""
        q = parse_records([_record("bc", hop_type="bridge_comparison")])[0][0]
        assert q.hop_type == "bridge_comparison"
        assert [p.title for p in q.gold_paragraphs] == [
            "Alpha Corp", "Beta Ltd", "Gamma Inc", "Delta LLC",
        ]
        assert len(q.supporting_facts) == 4

    def test_the_other_three_types_carry_two_gold(self):
        for hop_type in ("comparison", "inference", "compositional"):
            q = parse_records([_record("x", hop_type=hop_type)])[0][0]
            assert len(q.gold_paragraphs) == 2, hop_type

    def test_gold_paragraphs_are_exactly_the_supporting_fact_titles(self, questions):
        q = questions[0]
        assert [p.title for p in q.gold_paragraphs] == ["Alpha Corp", "Beta Ltd"]
        assert all(not p.supporting_sentences for p in q.distractor_paragraphs)

    def test_supporting_facts_map_to_the_real_sentences(self, questions):
        q = questions[0]
        gold = {p.title: p for p in q.gold_paragraphs}
        assert gold["Alpha Corp"].supporting_text() == ("It was founded in 1990.",)
        assert gold["Beta Ltd"].supporting_text() == ("Beta Ltd was founded in 1985.",)

    def test_the_evidences_field_is_tolerated_not_required(self):
        """2Wiki carries gold triples the harness does not use; extra keys are fine."""
        raw = _record("no-ev")
        del raw["evidences"]
        assert len(parse_records([raw])[0]) == 1

    def test_the_rows_api_column_encoding_transcodes_and_parses_identically(self, raw_records):
        """If the datasets-server viewer is fixed, its column-oriented rows must
        normalise to exactly what the on-disk file produces."""
        official = raw_records[3]  # a bridge_comparison record
        hf_row = {
            "id": official["_id"],
            "type": official["type"],
            "question": official["question"],
            "answer": official["answer"],
            "supporting_facts": {
                "title": [t for t, _ in official["supporting_facts"]],
                "sent_id": [i for _, i in official["supporting_facts"]],
            },
            "context": {
                "title": [t for t, _ in official["context"]],
                "sentences": [s for _, s in official["context"]],
            },
        }
        transcoded = twowiki._to_official_schema(hf_row)
        assert transcoded["_id"] == official["_id"]
        assert transcoded["type"] == "bridge_comparison"
        assert parse_records([transcoded])[0][0] == parse_records([official])[0][0]


# ── 2. Malformed records are skipped, not fatal ──────────
class TestMalformedRecords:
    @pytest.mark.parametrize(
        "mangle",
        [
            lambda r: r.pop("_id"),
            lambda r: r.update(question="   "),
            lambda r: r.pop("answer"),
            lambda r: r.update(context="a string"),
            lambda r: r.update(supporting_facts=[["Absent Corp", 0]]),
            lambda r: r.update(supporting_facts=[]),
        ],
    )
    def test_parse_record_rejects_broken_records(self, mangle):
        raw = _record("broken")
        mangle(raw)
        parsed, skipped = parse_records([raw])
        assert parsed == [] and len(skipped) == 1
        assert skipped[0].reason

    def test_one_broken_record_does_not_take_the_run_down(self, raw_records):
        broken = _record("broken")
        del broken["_id"]
        mixed = [*raw_records[:5], broken, *raw_records[5:]]
        parsed, skipped = parse_records(mixed)
        assert len(parsed) == 12
        assert len(skipped) == 1 and skipped[0].index == 5


# ── 3. Deterministic, capped sampling ────────────────────
class TestSampling:
    def test_the_same_seed_and_n_always_yield_the_same_ids(self, questions):
        a = [q.id for q in sample_questions(questions, 5, seed=20260721)]
        b = [q.id for q in sample_questions(questions, 5, seed=20260721)]
        assert a == b

    def test_the_sample_is_independent_of_the_input_ordering(self, questions):
        shuffled = list(reversed(questions))
        assert [q.id for q in sample_questions(questions, 5, seed=99)] == [
            q.id for q in sample_questions(shuffled, 5, seed=99)
        ]

    def test_a_different_seed_selects_a_different_set(self, questions):
        a = {q.id for q in sample_questions(questions, 5, seed=1)}
        b = {q.id for q in sample_questions(questions, 5, seed=2)}
        assert a != b

    def test_a_larger_n_extends_the_smaller_sample(self, questions):
        small = [q.id for q in sample_questions(questions, 3, seed=42)]
        large = [q.id for q in sample_questions(questions, 8, seed=42)]
        assert large[: len(small)] == small

    @pytest.mark.parametrize("n", [0, 1, 3, 12])
    def test_the_cap_is_respected_exactly(self, questions, n):
        assert len(sample_questions(questions, n, seed=5)) == n

    def test_asking_for_more_than_the_pool_returns_the_pool(self, questions):
        picked = sample_questions(questions, 500, seed=5)
        assert len(picked) == 12

    def test_a_negative_n_is_a_hard_error(self, questions):
        with pytest.raises(ValueError):
            sample_questions(questions, -1, seed=5)

    def test_the_chosen_ids_are_recorded_for_reproduction(self, questions, tmp_path, raw_records):
        path = tmp_path / "dev.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        result = load_sample(4, seed=11, path=path)
        assert result.ids == [q.id for q in result.questions]
        assert result.to_dict()["question_ids"] == result.ids
        assert result.to_dict()["dataset"] == "2wikimultihopqa-dev"
        assert result.to_dict()["seed"] == 11


# ── 4. Corpus statistics ─────────────────────────────────
class TestCorpusStats:
    def test_counts_are_computed_not_assumed(self, questions):
        stats = corpus_stats(questions)
        assert stats.n_questions == 12
        # 3 records per type over 12; bridge_comparison records have 5 paragraphs
        # (4 gold + 1 distractor), the rest have 3 (2 gold + 1 distractor).
        assert stats.n_gold_paragraphs == 9 * 2 + 3 * 4  # 30
        assert stats.n_dangling_supporting_facts == 0

    def test_all_four_reasoning_types_are_visible(self, questions):
        stats = corpus_stats(questions)
        assert set(stats.hop_types) == set(HOP_TYPES)
        assert sum(stats.hop_types.values()) == stats.n_questions

    def test_the_report_prints_the_numbers_and_the_ids(self, tmp_path, raw_records):
        path = tmp_path / "dev.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        report = format_report(load_sample(4, seed=3, path=path))
        assert "bridge_comparison" in report and "compositional" in report
        assert "gold paragraphs" in report
        for qid in load_sample(4, seed=3, path=path).ids:
            assert qid in report


# ── 5. No network in the hot path ────────────────────────
class TestNoNetworkInTheHotPath:
    @pytest.fixture
    def no_network(self, monkeypatch):
        import urllib.request

        def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
            raise AssertionError("this code path must not touch the network")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

    def test_loading_from_an_explicit_path_never_downloads(self, no_network, raw_records, tmp_path):
        path = tmp_path / "dev.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        result = load_sample(2, seed=1, path=path)
        assert len(result.questions) == 2
        assert result.source == str(path)

    def test_the_cli_runs_offline_against_a_local_file(
        self, no_network, raw_records, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "2wikimultihopqa_dev_v1.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        monkeypatch.setattr(twowiki, "CACHE_PATH", path)
        monkeypatch.setattr(twowiki, "PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr(twowiki, "DATA_DIR", tmp_path)
        assert main(["--n", "3", "--seed", "4", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sampled_n"] == 3
        assert payload["dataset"] == "2wikimultihopqa-dev"

    def test_no_download_fails_loudly_instead_of_fetching(self, no_network, tmp_path, monkeypatch):
        monkeypatch.setattr(twowiki, "CACHE_PATH", tmp_path / "absent.json")
        monkeypatch.setattr(twowiki, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_sample(2, seed=1, allow_download=False)


# ── 6. Caching / provenance ──────────────────────────────
class TestCacheAndProvenance:
    def test_a_present_cache_is_reused_without_a_download(self, tmp_path, monkeypatch, raw_records):
        import urllib.request

        cache = tmp_path / "2wikimultihopqa_dev_v1.json"
        cache.write_text(json.dumps(raw_records), encoding="utf-8")
        monkeypatch.setattr(twowiki, "DATA_DIR", tmp_path)
        monkeypatch.setattr(twowiki, "CACHE_PATH", cache)
        monkeypatch.setattr(twowiki, "PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: pytest.fail("cache hit must not download")
        )
        provenance = twowiki.ensure_dataset()
        assert provenance.n_records == 12
        assert provenance.sha256 and provenance.n_bytes == cache.stat().st_size
        assert twowiki.ensure_dataset() == provenance  # the recorded file is read back

    def test_provenance_round_trips(self):
        p = Provenance(
            source_url="https://example.invalid/dev.json",
            retrieved_at="2026-07-26T00:00:00+00:00",
            n_records=12576,
            sha256="deadbeef",
            n_bytes=55_934_464,
            truncated=False,
        )
        assert Provenance.from_dict(p.to_dict()) == p
        assert p.to_dict()["dataset"] == "2wikimultihopqa-dev"

    def test_load_raw_records_rejects_a_non_array_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an array"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_raw_records(path)

    def test_the_data_directory_is_gitignored(self):
        """A 56 MB blob must never enter the repository."""
        gitignore = Path(__file__).resolve().parent.parent / "benchmarks" / "public" / ".gitignore"
        assert gitignore.is_file()
        entries = {
            line.strip()
            for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert "data/" in entries


# ── 7. Acquisition: probe the rows API, fall back to dev.json ─
class TestAcquisition:
    def _hf_row(self, qid: str) -> dict:
        official = _record(qid)
        return {
            "id": qid,
            "type": official["type"],
            "question": official["question"],
            "answer": official["answer"],
            "supporting_facts": {
                "title": [t for t, _ in official["supporting_facts"]],
                "sent_id": [i for _, i in official["supporting_facts"]],
            },
            "context": {
                "title": [t for t, _ in official["context"]],
                "sentences": [s for _, s in official["context"]],
            },
        }

    def _pager(self, total: int, *, fail_at: int | None = None):
        seen: list[int] = []

        def page(offset: int, length: int, timeout: float, **kwargs):
            seen.append(offset)
            if fail_at is not None and offset >= fail_at:
                raise RuntimeError("simulated rate limit")
            rows = [
                {"row_idx": i, "row": self._hf_row(f"{i:024x}")}
                for i in range(offset, min(offset + length, total))
            ]
            return {"rows": rows, "num_rows_total": total}

        page.seen = seen  # type: ignore[attr-defined]
        return page

    @pytest.fixture
    def tw(self, monkeypatch):
        monkeypatch.setattr(twowiki, "HF_PAGE_SIZE", 100)
        monkeypatch.setattr(twowiki, "HF_PAGE_PAUSE", 0.0)
        return twowiki

    def test_a_down_rows_api_falls_back_to_the_canonical_dev_json(self, tw, monkeypatch, tmp_path):
        cache = tmp_path / "2wikimultihopqa_dev_v1.json"
        monkeypatch.setattr(tw, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tw, "CACHE_PATH", cache)
        monkeypatch.setattr(tw, "PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr(tw, "probe_rows_api", lambda *a, **k: (False, "HTTP 500"))
        monkeypatch.setattr(tw, "_download_rows_api", lambda *a: pytest.fail("rows API is down"))

        def fake_resolve(dest, timeout):
            dest.write_text(json.dumps([_record("a"), _record("b")]), encoding="utf-8")
            return 2

        monkeypatch.setattr(tw, "_download_resolve_json", fake_resolve)
        provenance = tw.ensure_dataset()
        assert provenance.source_url == tw.RESOLVE_URL
        assert provenance.n_records == 2 and provenance.truncated is False

    def test_a_live_rows_api_is_preferred_over_the_file(self, tw, monkeypatch, tmp_path):
        cache = tmp_path / "2wikimultihopqa_dev_v1.json"
        monkeypatch.setattr(tw, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tw, "CACHE_PATH", cache)
        monkeypatch.setattr(tw, "PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr(tw, "probe_rows_api", lambda *a, **k: (True, "live"))
        monkeypatch.setattr(tw, "_hf_page", self._pager(120))
        monkeypatch.setattr(
            tw, "_download_resolve_json", lambda *a: pytest.fail("rows API is up — use it")
        )
        provenance = tw.ensure_dataset()
        assert "datasets-server.huggingface.co" in provenance.source_url
        assert provenance.n_records == 120

    def test_rows_pagination_assembles_the_whole_split(self, tw, monkeypatch, tmp_path):
        pager = self._pager(250)
        monkeypatch.setattr(tw, "_hf_page", pager)
        dest = tmp_path / "dev.json"
        assert tw._download_rows_api(dest, 10.0, None) == 250
        assert pager.seen == [0, 100, 200]
        record = json.loads(dest.read_text())[0]
        assert "_id" in record and isinstance(record["context"], list)

    def test_max_records_caps_the_rows_download_and_flags_truncation(self, tw, monkeypatch, tmp_path):
        cache = tmp_path / "2wikimultihopqa_dev_v1.json"
        monkeypatch.setattr(tw, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tw, "CACHE_PATH", cache)
        monkeypatch.setattr(tw, "PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr(tw, "probe_rows_api", lambda *a, **k: (True, "live"))
        pager = self._pager(7405)
        monkeypatch.setattr(tw, "_hf_page", pager)
        provenance = tw.ensure_dataset(max_records=150)
        assert provenance.truncated is True and provenance.n_records == 150
        assert pager.seen == [0, 100]

    def test_a_failed_page_keeps_completed_pages_and_resumes(self, tw, monkeypatch, tmp_path):
        dest = tmp_path / "dev.json"
        monkeypatch.setattr(tw, "_hf_page", self._pager(250, fail_at=200))
        with pytest.raises(RuntimeError):
            tw._download_rows_api(dest, 10.0, None)
        assert not dest.exists()
        shard = dest.with_suffix(".jsonl.part")
        assert shard.exists() and len(shard.read_text().splitlines()) == 200

        resumed = self._pager(250)
        monkeypatch.setattr(tw, "_hf_page", resumed)
        assert tw._download_rows_api(dest, 10.0, None) == 250
        assert resumed.seen == [200] and not shard.exists()

    def test_resolve_download_rejects_a_non_array_payload(self, tw, monkeypatch, tmp_path):
        dest = tmp_path / "dev.json"

        class _Resp:
            def __init__(self):
                self._chunks = [b'{"not": "an array"}']

            def read(self, _n):
                return self._chunks.pop(0) if self._chunks else b""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(tw, "_open", lambda *a, **k: _Resp())
        with pytest.raises(RuntimeError, match="JSON array"):
            tw._download_resolve_json(dest, 10.0)
        assert not dest.exists()


# ── 8. Against the REAL download (skip-guarded) ──────────
@pytest.mark.skipif(
    os.environ.get("SYNAPSE_2WIKI") != "1",
    reason="needs the real ~56MB 2Wiki download — set SYNAPSE_2WIKI=1 to run",
)
class TestRealDataset:
    """Run with: SYNAPSE_2WIKI=1 pytest tests/test_twowiki_dataset.py -k Real"""

    def test_the_real_split_parses_and_samples(self):
        result = load_sample(50, seed=20260721)
        assert abs(result.pool_size - EXPECTED_DEV_RECORDS) < 50
        assert len(result.questions) == 50

    def test_the_real_split_parses_with_no_skips(self):
        parsed, skipped = parse_records(load_raw_records())
        assert len(skipped) == 0
        assert sum(len(q.dangling_supporting_facts) for q in parsed) == 0

    def test_gold_count_is_two_or_four_and_bridge_comparison_is_always_four(self):
        parsed, _ = parse_records(load_raw_records())
        for q in parsed:
            assert len(q.gold_paragraphs) in (2, 4), q.id
            if q.hop_type == "bridge_comparison":
                assert len(q.gold_paragraphs) == 4, q.id
            else:
                assert len(q.gold_paragraphs) == 2, q.id

    def test_the_real_sample_contains_bridge_like_types(self):
        stats: CorpusStats = load_sample(50, seed=20260721).stats
        assert set(stats.hop_types) <= set(HOP_TYPES)
        assert any(stats.hop_types.get(t, 0) > 0 for t in ("compositional", "inference"))

    def test_sampling_the_real_split_is_reproducible(self):
        assert load_sample(20, seed=20260721).ids == load_sample(20, seed=20260721).ids
