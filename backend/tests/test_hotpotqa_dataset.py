# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for HotpotQA acquisition + the deterministic capped sampler.

Hermetic by construction: every test here runs against a tiny inline fixture that
mirrors the real schema, and :class:`TestNoNetworkInTheHotPath` proves it by
booby-trapping ``urllib.request.urlopen`` — if any parsing/sampling/stats path
ever grows a download, that test fails rather than quietly costing bandwidth (or,
in later phases, money).

The one test that needs the real 45 MB file is skip-guarded on ``SYNAPSE_HOTPOT=1``,
matching the ``SYNAPSE_IT=1`` pattern used by ``tests/integration``.

What these lock down
--------------------
1. **Determinism** — a fixed ``(seed, n)`` yields the same ids every time, across
   input orderings, and a larger ``n`` extends the smaller sample rather than
   replacing it (so growing a run never re-pays for extraction already done).
2. **Gold identification** — exactly the supporting-fact paragraphs are gold, and
   the sentence indices point at the sentences HotpotQA meant.
3. **The cap is real** — ``n`` bounds the output, which is the only thing standing
   between a later phase and an unbounded LLM bill.
4. **Malformed records are skipped, not fatal** — one broken record in a 7405-row
   file must not take the run down, and must be *reported* rather than hidden.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.public.hotpotqa import (
    EXPECTED_DEV_RECORDS,
    HOP_TYPES,
    CorpusStats,
    MalformedRecord,
    Provenance,
    corpus_stats,
    format_report,
    load_raw_records,
    load_sample,
    main,
    parse_record,
    parse_records,
    sample_questions,
)

# ── A tiny fixture mirroring the real schema ─────────────
# Two paragraphs per record instead of ten, so the file stays readable; the
# *shape* is exactly the official one: context is [[title, [sentences]]] and
# supporting_facts is [[title, sent_id]].


def _record(
    qid: str,
    *,
    hop_type: str = "bridge",
    level: str = "hard",
    extra_paragraphs: int = 1,
) -> dict:
    context = [
        ["Alpha Corp", ["Alpha Corp is a firm.", "It was founded in 1990."]],
        ["Beta Ltd", ["Beta Ltd makes widgets.", "Beta Ltd was founded in 1985."]],
    ]
    for i in range(extra_paragraphs):
        context.append([f"Distractor {i}", [f"Irrelevant sentence {i}."]])
    return {
        "_id": qid,
        "question": f"Which of Alpha Corp and Beta Ltd is older? ({qid})",
        "answer": "Beta Ltd",
        "type": hop_type,
        "level": level,
        "supporting_facts": [["Alpha Corp", 1], ["Beta Ltd", 1]],
        "context": context,
    }


@pytest.fixture
def raw_records() -> list[dict]:
    """12 well-formed records with a deliberate 4/8 comparison/bridge split."""
    return [
        _record(f"{i:024x}", hop_type="comparison" if i % 3 == 0 else "bridge")
        for i in range(12)
    ]


@pytest.fixture
def questions(raw_records):
    parsed, skipped = parse_records(raw_records)
    assert skipped == []
    return parsed


# ── 1. Parsing and gold identification ───────────────────
class TestParsing:
    def test_the_fixture_parses_into_the_normalised_shape(self, questions):
        assert len(questions) == 12
        q = questions[0]
        assert q.id == f"{0:024x}"
        assert q.answer == "Beta Ltd"
        assert q.hop_type in HOP_TYPES
        assert q.level == "hard"
        assert len(q.paragraphs) == 3

    def test_gold_paragraphs_are_exactly_the_supporting_fact_titles(self, questions):
        q = questions[0]
        assert [p.title for p in q.gold_paragraphs] == ["Alpha Corp", "Beta Ltd"]
        assert [p.title for p in q.distractor_paragraphs] == ["Distractor 0"]
        assert all(not p.supporting_sentences for p in q.distractor_paragraphs), (
            "a distractor must never carry supporting-sentence indices"
        )

    def test_supporting_facts_map_to_the_real_sentences(self, questions):
        """The index must select the sentence HotpotQA actually labelled."""
        q = questions[0]
        gold = {p.title: p for p in q.gold_paragraphs}
        assert gold["Alpha Corp"].supporting_sentences == (1,)
        assert gold["Alpha Corp"].supporting_text() == ("It was founded in 1990.",)
        assert gold["Beta Ltd"].supporting_text() == ("Beta Ltd was founded in 1985.",)
        assert q.supporting_facts == (("Alpha Corp", 1), ("Beta Ltd", 1))

    def test_every_supporting_index_is_in_range_for_its_paragraph(self, questions):
        for q in questions:
            for p in q.paragraphs:
                for i in p.supporting_sentences:
                    assert 0 <= i < len(p.sentences)

    def test_the_huggingface_column_encoding_parses_identically(self, raw_records):
        """The fallback source is column-oriented; both shapes must normalise the same."""
        official = raw_records[0]
        hf = {
            "id": official["_id"],
            "question": official["question"],
            "answer": official["answer"],
            "type": official["type"],
            "level": official["level"],
            "supporting_facts": {
                "title": [t for t, _ in official["supporting_facts"]],
                "sent_id": [i for _, i in official["supporting_facts"]],
            },
            "context": {
                "title": [t for t, _ in official["context"]],
                "sentences": [s for _, s in official["context"]],
            },
        }
        assert parse_record(hf) == parse_record(official)

    def test_a_duplicate_supporting_index_is_not_double_counted(self):
        raw = _record("dup")
        raw["supporting_facts"] = [["Alpha Corp", 1], ["Alpha Corp", 1], ["Beta Ltd", 1]]
        q = parse_record(raw)
        assert q.supporting_facts == (("Alpha Corp", 1), ("Beta Ltd", 1))

    def test_char_count_covers_title_and_every_sentence(self, questions):
        p = questions[0].paragraphs[0]
        assert p.char_count == len("Alpha Corp") + sum(len(s) for s in p.sentences)


# ── 2. Malformed records are skipped, not fatal ──────────
def _broken(reason_key: str) -> dict:
    raw = _record("broken")
    match reason_key:
        case "no_id":
            del raw["_id"]
        case "empty_id":
            raw["_id"] = ""
        case "no_question":
            raw["question"] = "   "
        case "no_answer":
            del raw["answer"]
        case "context_is_a_string":
            raw["context"] = "Alpha Corp is a firm."
        case "context_entry_not_a_pair":
            raw["context"] = [["Alpha Corp"]]
        case "context_empty":
            raw["context"] = []
        case "non_string_sentence":
            raw["context"] = [["Alpha Corp", ["ok", 42]]]
        case "supporting_facts_missing":
            del raw["supporting_facts"]
        case "supporting_facts_empty":
            raw["supporting_facts"] = []
        case "supporting_title_absent":
            raw["supporting_facts"] = [["Gamma Inc", 0]]
        case "supporting_sent_id_not_int":
            raw["supporting_facts"] = [["Alpha Corp", "1"]]
        case "all_indices_dangling":
            raw["supporting_facts"] = [["Alpha Corp", 99]]
        case _:  # pragma: no cover
            raise AssertionError(reason_key)
    return raw


BROKEN_KEYS = [
    "no_id",
    "empty_id",
    "no_question",
    "no_answer",
    "context_is_a_string",
    "context_entry_not_a_pair",
    "context_empty",
    "non_string_sentence",
    "supporting_facts_missing",
    "supporting_facts_empty",
    "supporting_title_absent",
    "supporting_sent_id_not_int",
    "all_indices_dangling",
]


class TestMalformedRecords:
    @pytest.mark.parametrize("key", BROKEN_KEYS)
    def test_parse_record_rejects_it_with_a_typed_error(self, key):
        with pytest.raises(MalformedRecord):
            parse_record(_broken(key))

    @pytest.mark.parametrize("key", BROKEN_KEYS)
    def test_one_broken_record_does_not_take_the_run_down(self, key, raw_records):
        """The failure mode this exists to prevent: 7404 good rows lost to 1 bad one."""
        mixed = [*raw_records[:5], _broken(key), *raw_records[5:]]
        parsed, skipped = parse_records(mixed)
        assert len(parsed) == 12, "every well-formed record must survive"
        assert len(skipped) == 1
        assert skipped[0].index == 5
        assert skipped[0].reason, "a skip with no reason is an unauditable skip"

    @pytest.mark.parametrize("garbage", [None, 42, "a string", [], ["not", "an", "object"]])
    def test_a_non_object_record_is_skipped_too(self, garbage, raw_records):
        parsed, skipped = parse_records([garbage, *raw_records])
        assert len(parsed) == 12
        assert len(skipped) == 1

    def test_a_dangling_sentence_index_is_dropped_and_reported_not_fatal(self):
        """A known upstream quirk: keep the question, drop the index, count it."""
        raw = _record("dangling")
        raw["supporting_facts"] = [["Alpha Corp", 1], ["Beta Ltd", 1], ["Beta Ltd", 99]]
        q = parse_record(raw)
        assert q.supporting_facts == (("Alpha Corp", 1), ("Beta Ltd", 1))
        assert q.dangling_supporting_facts == (("Beta Ltd", 99),)
        assert len(q.gold_paragraphs) == 2, "the paragraph is still gold"

    def test_skipped_records_are_surfaced_in_the_sample_output(self, raw_records, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps([*raw_records, _broken("no_id")]), encoding="utf-8")
        result = load_sample(3, seed=7, path=path)
        payload = result.to_dict()
        assert payload["pool_size"] == 12
        assert len(payload["skipped_records"]) == 1
        assert payload["skipped_records"][0]["reason"]


# ── 3. Deterministic, capped sampling ────────────────────
class TestSampling:
    def test_the_same_seed_and_n_always_yield_the_same_ids(self, questions):
        a = [q.id for q in sample_questions(questions, 5, seed=20260721)]
        b = [q.id for q in sample_questions(questions, 5, seed=20260721)]
        assert a == b

    def test_the_sample_is_independent_of_the_input_ordering(self, questions):
        """Canonicalising by id first is what makes this true."""
        shuffled = list(reversed(questions))
        assert [q.id for q in sample_questions(questions, 5, seed=99)] == [
            q.id for q in sample_questions(shuffled, 5, seed=99)
        ]

    def test_a_different_seed_selects_a_different_set(self, questions):
        a = {q.id for q in sample_questions(questions, 5, seed=1)}
        b = {q.id for q in sample_questions(questions, 5, seed=2)}
        assert a != b, "the seed must actually steer the selection"

    def test_a_larger_n_extends_the_smaller_sample(self, questions):
        """Prefix-nesting: growing a run re-uses every extraction already paid for."""
        small = [q.id for q in sample_questions(questions, 3, seed=42)]
        large = [q.id for q in sample_questions(questions, 8, seed=42)]
        assert large[: len(small)] == small

    @pytest.mark.parametrize("n", [0, 1, 3, 12])
    def test_the_cap_is_respected_exactly(self, questions, n):
        assert len(sample_questions(questions, n, seed=5)) == n

    def test_asking_for_more_than_the_pool_returns_the_pool(self, questions):
        picked = sample_questions(questions, 500, seed=5)
        assert len(picked) == 12
        assert {q.id for q in picked} == {q.id for q in questions}

    def test_the_sample_never_repeats_a_question(self, questions):
        ids = [q.id for q in sample_questions(questions, 12, seed=5)]
        assert len(set(ids)) == 12

    def test_a_negative_n_is_a_hard_error_not_a_silent_slice(self, questions):
        """``pool[:-1]`` would quietly return almost everything — an unbounded run."""
        with pytest.raises(ValueError):
            sample_questions(questions, -1, seed=5)

    def test_the_chosen_ids_are_recorded_for_reproduction(self, questions, tmp_path, raw_records):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        result = load_sample(4, seed=11, path=path)
        assert result.ids == [q.id for q in result.questions]
        assert len(result.ids) == 4
        assert result.to_dict()["question_ids"] == result.ids
        assert result.to_dict()["seed"] == 11
        assert result.to_dict()["requested_n"] == 4


# ── 4. Corpus statistics ─────────────────────────────────
class TestCorpusStats:
    def test_counts_are_computed_not_assumed(self, questions):
        stats = corpus_stats(questions)
        assert stats.n_questions == 12
        assert stats.n_paragraphs == 36  # 3 paragraphs x 12 questions
        assert stats.n_gold_paragraphs == 24
        assert stats.n_distractor_paragraphs == 12
        assert stats.n_supporting_facts == 24
        assert stats.n_dangling_supporting_facts == 0
        assert stats.mean_gold_paragraphs_per_question == 2.0

    def test_hop_type_distribution_separates_bridge_from_comparison(self, questions):
        """Bridge questions are the genuinely multi-hop ones; the split must be visible."""
        stats = corpus_stats(questions)
        assert stats.hop_types == {"bridge": 8, "comparison": 4}
        assert sum(stats.hop_types.values()) == stats.n_questions

    def test_character_totals_add_up(self, questions):
        stats = corpus_stats(questions)
        expected = sum(p.char_count for q in questions for p in q.paragraphs)
        assert stats.total_context_chars == expected
        assert stats.total_gold_chars < stats.total_context_chars
        assert stats.total_question_chars == sum(len(q.question) for q in questions)

    def test_an_empty_sample_does_not_divide_by_zero(self):
        stats = corpus_stats([])
        assert stats.n_questions == 0
        assert stats.mean_paragraphs_per_question == 0.0
        assert stats.to_dict()["hop_types"] == {}

    def test_stats_round_trip_to_plain_json(self, questions):
        payload = corpus_stats(questions).to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert set(payload) >= {
            "n_questions",
            "n_paragraphs",
            "n_gold_paragraphs",
            "total_context_chars",
            "hop_types",
        }

    def test_the_report_prints_the_numbers_rather_than_prose(self, questions, tmp_path,
                                                             raw_records):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        report = format_report(load_sample(4, seed=3, path=path))
        assert "bridge" in report and "comparison" in report
        assert "gold paragraphs" in report
        for qid in load_sample(4, seed=3, path=path).ids:
            assert qid in report, "the ids are the reproducibility contract — print them"


# ── 5. No network in the hot path ────────────────────────
class TestNoNetworkInTheHotPath:
    """Parsing, sampling and stats must never open a socket."""

    @pytest.fixture
    def no_network(self, monkeypatch):
        import urllib.request

        def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
            raise AssertionError("this code path must not touch the network")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

    def test_loading_from_an_explicit_path_never_downloads(
        self, no_network, raw_records, tmp_path
    ):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        result = load_sample(2, seed=1, path=path)
        assert len(result.questions) == 2
        assert result.source == str(path)

    def test_the_cli_runs_offline_against_a_local_file(
        self, no_network, raw_records, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "hotpot_dev_distractor_v1.json"
        path.write_text(json.dumps(raw_records), encoding="utf-8")
        monkeypatch.setattr("benchmarks.public.hotpotqa.CACHE_PATH", path)
        monkeypatch.setattr("benchmarks.public.hotpotqa.PROVENANCE_PATH", tmp_path / "p.json")
        monkeypatch.setattr("benchmarks.public.hotpotqa.DATA_DIR", tmp_path)
        assert main(["--n", "3", "--seed", "4", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sampled_n"] == 3
        assert len(payload["question_ids"]) == 3

    def test_no_download_fails_loudly_instead_of_fetching(self, no_network, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.public.hotpotqa.CACHE_PATH", tmp_path / "absent.json")
        monkeypatch.setattr("benchmarks.public.hotpotqa.DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_sample(2, seed=1, allow_download=False)


# ── 6. Caching / provenance ──────────────────────────────
class TestCacheAndProvenance:
    def test_a_present_cache_is_reused_without_a_download(self, tmp_path, monkeypatch,
                                                          raw_records):
        """Re-running must not re-download — the whole point of the cache."""
        import urllib.request

        from benchmarks.public import hotpotqa

        cache = tmp_path / "hotpot_dev_distractor_v1.json"
        cache.write_text(json.dumps(raw_records), encoding="utf-8")
        monkeypatch.setattr(hotpotqa, "DATA_DIR", tmp_path)
        monkeypatch.setattr(hotpotqa, "CACHE_PATH", cache)
        monkeypatch.setattr(hotpotqa, "PROVENANCE_PATH", tmp_path / "provenance.json")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("cache hit must not download"),
        )
        provenance = hotpotqa.ensure_dataset()
        assert provenance.n_records == 12
        assert provenance.sha256 and provenance.n_bytes == cache.stat().st_size
        assert (tmp_path / "provenance.json").is_file()

        # ...and the recorded provenance is what a second call reads back.
        assert hotpotqa.ensure_dataset() == provenance

    def test_provenance_round_trips(self):
        p = Provenance(
            source_url="https://example.invalid/x.json",
            retrieved_at="2026-07-21T00:00:00+00:00",
            n_records=7405,
            sha256="deadbeef",
            n_bytes=45_000_000,
            truncated=False,
        )
        assert Provenance.from_dict(p.to_dict()) == p

    def test_load_raw_records_rejects_a_non_array_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "an array"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_raw_records(path)

    def test_the_data_directory_is_gitignored(self):
        """A 45 MB blob must never enter the repository."""
        gitignore = Path(__file__).resolve().parent.parent / "benchmarks" / "public" / ".gitignore"
        assert gitignore.is_file()
        entries = {
            line.strip()
            for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert "data/" in entries


# ── 7. The HuggingFace fallback (stubbed, no network) ────
class TestHuggingFaceFallback:
    """The official CMU host was unreachable on 2026-07-21, so this path is the
    one that actually runs. It is paginated 100 rows at a time against an
    anonymous, rate-limited API — the resume shard is what stops a 429 on page 55
    from throwing away 5,500 rows of completed download."""

    @staticmethod
    def _hf_row(qid: str) -> dict:
        """A row in the column-oriented shape the rows API really returns."""
        official = _record(qid)
        return {
            "id": qid,
            "question": official["question"],
            "answer": official["answer"],
            "type": official["type"],
            "level": official["level"],
            "supporting_facts": {
                "title": [t for t, _ in official["supporting_facts"]],
                "sent_id": [i for _, i in official["supporting_facts"]],
            },
            "context": {
                "title": [t for t, _ in official["context"]],
                "sentences": [s for _, s in official["context"]],
            },
        }

    def _pager(self, total: int, *, fail_at: int | None = None, broken: set[int] = frozenset()):
        """A fake ``_hf_page`` recording every offset it was asked for."""
        seen: list[int] = []

        def page(offset: int, length: int, timeout: float, **kwargs):
            seen.append(offset)
            if fail_at is not None and offset >= fail_at:
                raise RuntimeError("simulated rate limit")
            rows = []
            for i in range(offset, min(offset + length, total)):
                row = {"broken": True} if i in broken else self._hf_row(f"{i:024x}")
                rows.append({"row_idx": i, "row": row})
            return {"rows": rows, "num_rows_total": total}

        page.seen = seen  # type: ignore[attr-defined]
        return page

    @pytest.fixture
    def hf(self, monkeypatch, tmp_path):
        from benchmarks.public import hotpotqa

        monkeypatch.setattr(hotpotqa, "HF_PAGE_SIZE", 100)
        monkeypatch.setattr(hotpotqa, "HF_PAGE_PAUSE", 0.0)
        return hotpotqa

    def test_pagination_assembles_the_whole_split(self, hf, monkeypatch, tmp_path):
        pager = self._pager(250)
        monkeypatch.setattr(hf, "_hf_page", pager)
        dest = tmp_path / "hotpot.json"
        assert hf._download_huggingface(dest, 10.0, None) == 250
        assert pager.seen == [0, 100, 200]
        assert len(json.loads(dest.read_text())) == 250

    def test_rows_are_transcoded_into_the_official_schema(self, hf, monkeypatch, tmp_path):
        monkeypatch.setattr(hf, "_hf_page", self._pager(3))
        dest = tmp_path / "hotpot.json"
        hf._download_huggingface(dest, 10.0, None)
        record = json.loads(dest.read_text())[0]
        assert "_id" in record, "the cache must use the official key, not HF's 'id'"
        assert isinstance(record["context"], list)
        assert isinstance(record["supporting_facts"][0], list)
        parse_record(record)  # the parser only ever sees one schema

    def test_max_records_caps_the_download(self, hf, monkeypatch, tmp_path):
        pager = self._pager(7405)
        monkeypatch.setattr(hf, "_hf_page", pager)
        dest = tmp_path / "hotpot.json"
        assert hf._download_huggingface(dest, 10.0, 150) == 150
        assert pager.seen == [0, 100], "a capped run must stop requesting pages"

    def test_a_failed_page_keeps_the_completed_pages_and_resumes(
        self, hf, monkeypatch, tmp_path
    ):
        """The 429-on-page-55 scenario: a retry must not re-download rows 0-5,499."""
        dest = tmp_path / "hotpot.json"
        monkeypatch.setattr(hf, "_hf_page", self._pager(250, fail_at=200))
        with pytest.raises(RuntimeError):
            hf._download_huggingface(dest, 10.0, None)
        assert not dest.exists(), "a partial download must never masquerade as complete"
        shard = dest.with_suffix(".jsonl.part")
        assert shard.exists() and len(shard.read_text().splitlines()) == 200

        resumed = self._pager(250)
        monkeypatch.setattr(hf, "_hf_page", resumed)
        assert hf._download_huggingface(dest, 10.0, None) == 250
        assert resumed.seen == [200], "resume must start where the shard left off"
        assert not shard.exists(), "a completed download clears its shard"

    def test_a_broken_row_keeps_the_resume_offset_exact(self, hf, monkeypatch, tmp_path):
        """A dropped row would desynchronise a naive len(records) resume offset."""
        dest = tmp_path / "hotpot.json"
        monkeypatch.setattr(hf, "_hf_page", self._pager(150, fail_at=100, broken={3, 7}))
        with pytest.raises(RuntimeError):
            hf._download_huggingface(dest, 10.0, None)
        shard = dest.with_suffix(".jsonl.part")
        records, rows = hf._read_shard(shard)
        assert rows == 100, "the shard counts rows, not surviving records"
        assert len(records) == 98

    def test_a_torn_final_shard_line_is_discarded(self, hf, tmp_path):
        shard = tmp_path / "s.jsonl.part"
        shard.write_text('{"_id": "a"}\n{"_id": "b"}\n{"_id": "c', encoding="utf-8")
        records, rows = hf._read_shard(shard)
        assert rows == 2 and [r["_id"] for r in records] == ["a", "b"]

    def test_ensure_dataset_falls_back_when_the_official_host_is_down(
        self, hf, monkeypatch, tmp_path
    ):
        cache = tmp_path / "hotpot_dev_distractor_v1.json"
        monkeypatch.setattr(hf, "DATA_DIR", tmp_path)
        monkeypatch.setattr(hf, "CACHE_PATH", cache)
        monkeypatch.setattr(hf, "PROVENANCE_PATH", tmp_path / "provenance.json")
        monkeypatch.setattr(hf, "probe_official", lambda *a, **k: (False, "unreachable"))
        monkeypatch.setattr(hf, "_hf_page", self._pager(120))
        monkeypatch.setattr(
            hf, "_download_official", lambda *a: pytest.fail("official host is down")
        )
        provenance = hf.ensure_dataset()
        assert "datasets-server.huggingface.co" in provenance.source_url
        assert provenance.n_records == 120
        assert provenance.truncated is False

    def test_a_truncated_download_is_flagged_as_truncated(self, hf, monkeypatch, tmp_path):
        """Sampling from a truncated pool is a different experiment — say so."""
        cache = tmp_path / "hotpot_dev_distractor_v1.json"
        monkeypatch.setattr(hf, "DATA_DIR", tmp_path)
        monkeypatch.setattr(hf, "CACHE_PATH", cache)
        monkeypatch.setattr(hf, "PROVENANCE_PATH", tmp_path / "provenance.json")
        monkeypatch.setattr(hf, "probe_official", lambda *a, **k: (False, "unreachable"))
        monkeypatch.setattr(hf, "_hf_page", self._pager(7405))
        provenance = hf.ensure_dataset(max_records=100)
        assert provenance.truncated is True
        assert provenance.n_records == 100


# ── 8. Against the REAL download (skip-guarded) ──────────
@pytest.mark.skipif(
    os.environ.get("SYNAPSE_HOTPOT") != "1",
    reason="needs the real ~45MB HotpotQA download — set SYNAPSE_HOTPOT=1 to run",
)
class TestRealDataset:
    """Run with: SYNAPSE_HOTPOT=1 pytest tests/test_hotpotqa_dataset.py -k Real"""

    def test_the_real_split_parses_and_samples(self):
        result = load_sample(20, seed=20260721)
        assert result.pool_size > 7000, "dev/distractor should hold ~7405 questions"
        assert abs(result.pool_size - EXPECTED_DEV_RECORDS) < 100
        assert len(result.questions) == 20
        assert len(set(result.ids)) == 20

    def test_every_real_question_has_two_gold_paragraphs_and_at_most_ten(self):
        """"10 paragraphs" is the documented design, not an invariant of the file.

        Measured over the whole 2026-07-21 dev/distractor split: 7,345 of 7,405
        questions carry 10 paragraphs; the other 60 carry between 2 and 9. Two
        gold paragraphs, by contrast, holds for all 7,405. Asserting the
        distractor count would make this test seed-dependent — a green run that
        proves nothing until the sample happens to include one of the 60.
        """
        result = load_sample(20, seed=20260721)
        for q in result.questions:
            assert 2 <= len(q.paragraphs) <= 10, f"{q.id} has {len(q.paragraphs)} paragraphs"
            assert len(q.gold_paragraphs) == 2, f"{q.id} has {len(q.gold_paragraphs)} gold"

    def test_the_two_gold_invariant_holds_across_the_entire_split(self):
        """The invariant every downstream retrieval metric will lean on."""
        from benchmarks.public.hotpotqa import load_raw_records, parse_records

        parsed, skipped = parse_records(load_raw_records())
        assert len(skipped) == 0, f"the real split parsed with {len(skipped)} skips"
        assert all(len(q.gold_paragraphs) == 2 for q in parsed)
        assert sum(len(q.dangling_supporting_facts) for q in parsed) <= 5, (
            "tolerating dangling sentence indices is meant to cover a handful of "
            "upstream typos; a large count would mean the parser is misreading them"
        )

    def test_the_real_sample_contains_both_hop_types(self):
        stats: CorpusStats = load_sample(20, seed=20260721).stats
        assert set(stats.hop_types) <= {*HOP_TYPES}
        assert stats.hop_types.get("bridge", 0) > 0, "no genuinely multi-hop questions sampled"

    def test_sampling_the_real_split_is_reproducible(self):
        assert load_sample(20, seed=20260721).ids == load_sample(20, seed=20260721).ids
