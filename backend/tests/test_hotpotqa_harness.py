# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Tests for the public (HotpotQA) benchmark harness.

Hermetic: no network, no Neo4j, no model call — which is not merely a
convenience here, because one of the things under test is precisely that the
harness *does not* call a model when it says it is not going to.

Five jobs:

  1. **Metric correctness.** Gold-paragraph recall, the both-gold rate (the
     multi-hop discriminator), the bridge/comparison split, sentence-level
     supporting-fact recall and the channel attribution that separates the
     permissive rule from the strict one are checked against hand-computed
     fixtures. If these are wrong every number in the report is wrong, and
     nothing else in the run would notice.

  2. **Cost accounting.** The arithmetic that turns tokens into dollars, the
     three ways usage arrives from a provider, and — most importantly — that an
     *estimated* count can never lose its label by being added to a measured one.

  3. **``--dry-run`` really is free.** The LLM factory is monkeypatched to raise
     on any call, the embedder likewise, the Neo4j driver too, and the dataset
     downloader on top. A dry run that touches any of them fails loudly here
     instead of quietly on the author's card.

  4. **The effect-size floor.** A gap smaller than one question out of N is
     reported as TOO CLOSE TO CALL, exactly as ``benchmarks/run_benchmark.py``
     does — and one test asserts the two harnesses agree on the same input, so
     the public benchmark cannot drift into a laxer standard than the internal
     one it was written to answer for.

  5. **The refusals.** ``--reuse-graph`` over a graph that is not this sample,
     and a pre-flight estimate over ``--max-usd``, both have to stop the run
     rather than produce a number about the wrong corpus or spend past the cap.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.public import cost, hotpotqa
from benchmarks.public.run_hotpotqa import (
    DEFAULT_THEME,
    EDGE,
    ENTITY,
    PROSE,
    Corpus,
    QuestionResult,
    Report,
    ReuseError,
    Run,
    aggregate,
    build_corpus,
    channel_split,
    chat_model_name,
    credit_paragraphs,
    dry_run_lines,
    effect_floor,
    estimate_ingestion,
    evaluate_passage_rag,
    headline,
    load_corpus,
    loud_banner,
    main,
    metric_verdict,
    names_present,
    pool_titles,
    rank_titles,
    results_markdown,
    split_by_type,
    supporting_found,
)


# ── Fixtures: hand-written HotpotQA items ────────────────────────────────────
def hotpot_item(
    qid: str,
    question: str,
    *,
    qtype: str = "bridge",
    gold: tuple[str, str] = ("Alpha", "Beta"),
    supporting: tuple[tuple[str, int], ...] | None = None,
    extra_titles: tuple[str, ...] = ("Gamma", "Delta"),
) -> dict:
    """One item in HotpotQA's own on-disk shape.

    Sentences are written so each is unique and long enough that the
    sentence-level matcher cannot hit by accident.
    """
    titles = (*gold, *extra_titles)
    context = [
        [title, [f"{title} is a thing. ", f"{title} was founded in 19{i}0. "]]
        for i, title in enumerate(titles)
    ]
    facts = supporting if supporting is not None else ((gold[0], 0), (gold[1], 1))
    return {
        "_id": qid,
        "question": question,
        "answer": "yes",
        "type": qtype,
        "level": "hard",
        "context": context,
        "supporting_facts": [list(f) for f in facts],
    }


def parsed(*items: dict) -> list[hotpotqa.HotpotQuestion]:
    """Raw items → the normalised records ``build_corpus`` consumes.

    Goes through the real parser in ``benchmarks.public.hotpotqa`` rather than
    hand-building dataclasses, so a fixture can never describe a record shape the
    harness will not actually be handed.
    """
    return [hotpotqa.parse_record(item, index=i) for i, item in enumerate(items)]


@pytest.fixture
def corpus() -> Corpus:
    """Two questions — one bridge, one comparison — over six paragraphs."""
    return build_corpus(
        parsed(
            hotpot_item("q1", "Who founded Alpha and Beta?", qtype="bridge"),
            hotpot_item(
                "q2",
                "Were Epsilon and Zeta founded in the same year?",
                qtype="comparison",
                gold=("Epsilon", "Zeta"),
                extra_titles=("Gamma", "Delta"),
            ),
        )
    )


@pytest.fixture
def dataset_file(tmp_path):
    """A HotpotQA file on disk, so nothing anywhere reaches the network."""
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                hotpot_item(f"q{i:02d}", f"Who founded Alpha{i}?", gold=(f"A{i}", f"B{i}"))
                for i in range(6)
            ]
            + [
                hotpot_item(
                    "qc", "Same year?", qtype="comparison", gold=("Iota", "Kappa")
                )
            ]
        ),
        encoding="utf-8",
    )
    return path


# ── 1. Corpus loading ────────────────────────────────────────────────────────
class TestCorpusLoading:
    def test_paragraphs_are_deduplicated_across_questions(self, corpus):
        # Gamma and Delta appear in both items; six distinct titles remain.
        assert corpus.titles == ["Alpha", "Beta", "Delta", "Epsilon", "Gamma", "Zeta"]
        assert len(corpus.questions) == 2

    def test_the_type_split_is_read_from_the_dataset(self, corpus):
        assert corpus.counts_by_type() == {"bridge": 1, "comparison": 1}

    def test_paragraph_text_is_title_then_verbatim_sentences(self, corpus):
        text = corpus.text("Alpha")
        assert text.startswith("Alpha\n")
        # Verbatim: the sentence-level metric depends on it being byte-identical.
        assert "Alpha was founded in 1900." in text

    def test_a_supporting_fact_past_the_end_of_its_paragraph_is_dropped(self):
        """No system could ever retrieve it, so it must not sit in the denominator."""
        built = build_corpus(
            parsed(hotpot_item("q1", "?", supporting=(("Alpha", 0), ("Beta", 99))))
        )
        assert built.notes["dangling_supporting_facts"] == 1
        assert built.questions[0].supporting_facts == (("Alpha", 0),)

    def test_a_question_whose_gold_lives_in_one_paragraph_is_counted_not_hidden(self):
        """'Both-gold' is really 'all-gold' for such an item, and the report says so."""
        built = build_corpus(
            parsed(hotpot_item("q1", "?", supporting=(("Alpha", 0), ("Alpha", 1))))
        )
        assert built.notes["questions_without_two_gold"] == ["q1"]

    def test_a_record_missing_its_gold_paragraph_is_refused_by_the_parser(self):
        item = hotpot_item("q1", "?")
        item["context"] = [c for c in item["context"] if c[0] != "Beta"]
        with pytest.raises(hotpotqa.MalformedRecord, match="absent from context"):
            hotpotqa.parse_record(item)

    def test_an_empty_sample_is_refused_rather_than_scored(self):
        with pytest.raises(Exception, match="nothing to score"):
            build_corpus([])


class TestSampleCap:
    """Cost discipline rule 1: every run is bounded by an explicit question count."""

    def test_the_question_count_is_a_hard_cap(self, dataset_file):
        loaded = load_corpus(questions=2, seed=1, data_path=dataset_file, allow_download=False)
        assert len(loaded.questions) == 2

    def test_the_same_seed_and_count_select_the_same_questions(self, dataset_file):
        first = load_corpus(questions=3, seed=7, data_path=dataset_file, allow_download=False)
        again = load_corpus(questions=3, seed=7, data_path=dataset_file, allow_download=False)
        assert [q.id for q in first.questions] == [q.id for q in again.questions]

    def test_an_unbounded_run_is_never_allowed(self, dataset_file):
        with pytest.raises(Exception, match="never allowed"):
            load_corpus(questions=0, seed=1, data_path=dataset_file, allow_download=False)


# ── 2. Metrics ───────────────────────────────────────────────────────────────
class TestGoldParagraphMetrics:
    def _result(self, credited: dict[str, str], gold=("Alpha", "Beta")) -> QuestionResult:
        return QuestionResult(
            qid="q",
            question="?",
            qtype="bridge",
            gold=gold,
            credited=credited,
            context_chars=100,
            prose_chars=50,
            supporting_hits=1,
            supporting_total=2,
        )

    def test_gold_recall_is_the_fraction_of_the_two_gold_paragraphs(self):
        assert self._result({"Alpha": PROSE}).gold_recall == 0.5
        assert self._result({"Alpha": PROSE, "Beta": ENTITY}).gold_recall == 1.0
        assert self._result({"Gamma": PROSE}).gold_recall == 0.0

    def test_both_gold_is_all_or_nothing(self):
        """The multi-hop discriminator: one of two is a wrong answer, not half."""
        assert self._result({"Alpha": PROSE}).both_gold is False
        assert self._result({"Alpha": PROSE, "Beta": EDGE}).both_gold is True

    def test_the_strict_rule_counts_only_prose(self):
        result = self._result({"Alpha": PROSE, "Beta": ENTITY})
        assert result.gold_recall == 1.0
        assert result.strict_gold_recall == 0.5
        assert result.strict_both_gold is False

    def test_a_graph_only_context_scores_zero_strictly_by_construction(self):
        result = self._result({"Alpha": ENTITY, "Beta": EDGE})
        assert result.gold_recall == 1.0
        assert result.strict_gold_recall == 0.0

    def test_aggregate_matches_hand_computed_values(self):
        results = [
            self._result({"Alpha": PROSE, "Beta": PROSE}),  # 1.0, both
            self._result({"Alpha": PROSE}),                 # 0.5, not both
            self._result({}),                               # 0.0
        ]
        agg = aggregate(results)
        assert agg["gold_recall"] == pytest.approx(1.5 / 3)
        assert agg["both_gold"] == pytest.approx(1 / 3)
        assert agg["support_recall"] == pytest.approx(0.5)
        assert agg["n"] == 3

    def test_aggregate_of_nothing_is_zeroes_not_a_crash(self):
        assert aggregate([])["n"] == 0
        assert aggregate([])["both_gold"] == 0.0

    def test_split_by_type_separates_bridge_from_comparison(self):
        bridge = self._result({"Alpha": PROSE})
        comparison = self._result({"Alpha": PROSE})
        comparison.qtype = "comparison"
        buckets = split_by_type([bridge, comparison, bridge])
        assert list(buckets) == ["bridge", "comparison"]
        assert len(buckets["bridge"]) == 2
        assert len(buckets["comparison"]) == 1

    def test_an_unexpected_question_type_is_named_not_silently_pooled(self):
        odd = self._result({"Alpha": PROSE})
        odd.qtype = "unknown"
        buckets = split_by_type([self._result({}), odd])
        assert list(buckets) == ["bridge", "unknown"]

    def test_channel_split_only_counts_gold_paragraphs(self):
        stats = channel_split([self._result({"Alpha": PROSE, "Beta": EDGE, "Gamma": ENTITY})])
        assert stats["total"] == 2  # Gamma is not gold
        assert stats["counts"] == {PROSE: 1, ENTITY: 0, EDGE: 1}
        assert stats["shares"][PROSE] == pytest.approx(0.5)


class TestSupportingFacts:
    def test_a_supporting_sentence_is_found_in_retrieved_prose(self, corpus):
        question = corpus.questions[0]
        assert supporting_found(
            corpus.text("Alpha"), question.supporting_facts, corpus.paragraphs
        ) == 1

    def test_no_prose_means_no_supporting_facts(self, corpus):
        question = corpus.questions[0]
        assert supporting_found("", question.supporting_facts, corpus.paragraphs) == 0

    def test_whitespace_differences_do_not_break_the_match(self, corpus):
        question = corpus.questions[0]  # supports Alpha[0] and Beta[1]
        prose = "Alpha   is a thing.\n\nBeta was\nfounded in 1910."
        assert supporting_found(prose, question.supporting_facts, corpus.paragraphs) == 2

    def test_a_sentence_that_never_came_back_is_not_credited(self, corpus):
        question = corpus.questions[0]
        assert supporting_found(
            "Gamma is a thing.", question.supporting_facts, corpus.paragraphs
        ) == 0


class TestChannelAttribution:
    """The permissive rule is uniform, not neutral — this is what sizes it."""

    provenance = {
        "Alpha Corp": ["Alpha"],
        "Beta Corp": ["Beta"],
        "Gamma Corp": ["Gamma"],
    }

    def test_an_excerpt_credits_its_paragraph_to_the_prose_channel(self):
        credited = credit_paragraphs(
            "Entity: Alpha Corp (Type: ORGANIZATION)\n\n"
            "Source excerpts:\n\n[S1] Alpha (chunk 0)\nAlpha is a thing.",
            [{"document": "Alpha", "text": "Alpha is a thing."}],
            self.provenance,
        )
        assert credited["Alpha"] == PROSE

    def test_an_entity_block_credits_its_source_paragraph(self):
        credited = credit_paragraphs(
            "Entity: Beta Corp (Type: ORGANIZATION)\n  Description: a company",
            [],
            self.provenance,
        )
        assert credited == {"Beta": ENTITY}

    def test_a_neighbour_line_alone_is_the_weakest_channel(self):
        context = (
            "Entity: Alpha Corp (Type: ORGANIZATION)\n"
            "  Relationships:\n"
            "  → FOUNDED_BY → Gamma Corp (PERSON)"
        )
        credited = credit_paragraphs(context, [], self.provenance)
        assert credited["Alpha"] == ENTITY
        assert credited["Gamma"] == EDGE

    def test_prose_beats_scaffolding_for_the_same_paragraph(self):
        """Attribution is conservative AGAINST the graph, by design."""
        context = (
            "Entity: Alpha Corp (Type: ORGANIZATION)\n\n"
            "Source excerpts:\n\n[S1] Alpha (chunk 0)\nAlpha is a thing."
        )
        credited = credit_paragraphs(context, [{"document": "Alpha"}], self.provenance)
        assert credited["Alpha"] == PROSE

    def test_an_entity_with_no_provenance_credits_nothing(self):
        """A graph the harness cannot map back to a paragraph must score 0, not guess."""
        assert credit_paragraphs("Entity: Zeta Corp (Type: ORGANIZATION)", [], {}) == {}

    def test_short_and_partial_names_never_credit_a_paragraph(self):
        """HotpotQA entity names are LLM-produced, so matching is word-bounded."""
        assert names_present("The US and the USA", ["US"]) == []  # too short
        assert names_present("Alphabetical order", ["Alpha Corp"]) == []  # not a word
        assert names_present("Alpha Corp filed", ["Alpha Corp"]) == ["Alpha Corp"]
        assert names_present("alpha corp filed", ["Alpha Corp"]) == ["Alpha Corp"]


class TestPassageBaseline:
    def test_it_credits_exactly_what_it_retrieved_as_prose(self, corpus):
        rankings = {"q1": ["Alpha", "Gamma", "Beta"], "q2": ["Gamma", "Delta"]}
        results = evaluate_passage_rag(corpus, rankings, 2)
        first = next(r for r in results if r.qid == "q1")
        assert first.credited == {"Alpha": PROSE, "Gamma": PROSE}
        assert first.gold_recall == 0.5
        # Every character it returns is prose, so both rules score it identically.
        assert first.strict_gold_recall == first.gold_recall

    def test_context_chars_are_the_paragraphs_it_actually_returned(self, corpus):
        results = evaluate_passage_rag(corpus, {"q1": ["Alpha"], "q2": []}, 1)
        first = next(r for r in results if r.qid == "q1")
        assert first.context_chars == len(corpus.text("Alpha"))
        assert first.prose_chars == first.context_chars

    def test_a_larger_k_can_only_add_paragraphs(self, corpus):
        rankings = {"q1": ["Alpha", "Beta"], "q2": ["Gamma"]}
        one = next(r for r in evaluate_passage_rag(corpus, rankings, 1) if r.qid == "q1")
        two = next(r for r in evaluate_passage_rag(corpus, rankings, 2) if r.qid == "q1")
        assert one.gold_recall == 0.5
        assert two.gold_recall == 1.0
        assert two.context_chars > one.context_chars

    def test_the_distractor_pool_ranking_never_leaves_the_questions_own_ten(self, corpus):
        vectors = {title: [float(i), 1.0] for i, title in enumerate(corpus.titles)}
        questions = {q.id: [1.0, 0.0] for q in corpus.questions}
        pooled = rank_titles(corpus, vectors, questions, pooled=True)
        per_question = rank_titles(corpus, vectors, questions, pooled=False)
        for question in corpus.questions:
            assert set(pooled[question.id]) == set(corpus.titles)
            assert set(per_question[question.id]) <= set(pool_titles(question))
            assert set(per_question[question.id]) < set(corpus.titles)


# ── 3. The effect-size floor ─────────────────────────────────────────────────
class TestEffectSizeFloor:
    base = {"gold_recall": 0.50, "both_gold": 0.40, "n": 20, "mean_context_chars": 100.0}

    def test_one_question_out_of_n(self):
        assert effect_floor(20) == pytest.approx(0.05)
        assert effect_floor(1) == 0.0  # a 100 pp floor would suppress every verdict
        assert effect_floor(0) == 0.0

    def test_a_sub_question_gap_is_refused(self):
        system = {**self.base, "gold_recall": 0.52}
        line = metric_verdict(
            "Gold-paragraph recall", "gold_recall", self.base, system,
            baseline_label="A", system_label="D",
        )
        assert "TOO CLOSE TO CALL" in line
        assert "smaller than one question out of 20" in line
        assert "No significance is claimed" in line

    def test_a_gap_wider_than_one_question_is_called(self):
        system = {**self.base, "gold_recall": 0.70}
        line = metric_verdict(
            "Gold-paragraph recall", "gold_recall", self.base, system,
            baseline_label="A", system_label="D",
        )
        assert "D ahead of A" in line
        assert "TOO CLOSE" not in line

    def test_the_baseline_winning_is_reported_just_as_plainly(self):
        system = {**self.base, "gold_recall": 0.20}
        line = metric_verdict(
            "Gold-paragraph recall", "gold_recall", self.base, system,
            baseline_label="A", system_label="D",
        )
        assert "A ahead of D" in line

    def test_an_exact_tie_is_a_tie(self):
        line = metric_verdict(
            "Both-gold rate", "both_gold", self.base, {**self.base},
            baseline_label="A", system_label="D",
        )
        assert line.startswith("Both-gold rate: TIE")

    def test_a_single_question_is_not_a_measurement(self):
        one = {"gold_recall": 0.0, "n": 1, "mean_context_chars": 10.0}
        line = metric_verdict(
            "Gold-paragraph recall", "gold_recall", one, {**one, "gold_recall": 1.0},
            baseline_label="A", system_label="D",
        )
        assert "NOT SCOREABLE" in line

    def test_it_agrees_with_the_internal_benchmarks_floor(self):
        """The public harness must not quietly adopt a laxer standard."""
        from benchmarks.run_benchmark import verdict as internal_verdict

        internal = internal_verdict(
            {"fact_recall": 0.50, "full_coverage": 0.40, "mean_context_chars": 100.0, "n": 20},
            {"fact_recall": 0.52, "full_coverage": 0.40, "mean_context_chars": 100.0, "n": 20},
            [(1, {"fact_recall": 0.0, "full_coverage": 0.0, "mean_context_chars": 10.0})],
        )
        assert any("TOO CLOSE TO CALL" in line for line in internal)
        public = metric_verdict(
            "Gold-paragraph recall", "gold_recall",
            self.base, {**self.base, "gold_recall": 0.52},
            baseline_label="A", system_label="D",
        )
        assert "TOO CLOSE TO CALL" in public

    def test_a_subgroup_is_judged_against_its_own_smaller_n(self):
        small = {"gold_recall": 0.50, "n": 4, "mean_context_chars": 100.0}
        line = metric_verdict(
            "Gold-paragraph recall", "gold_recall", small, {**small, "gold_recall": 0.70},
            baseline_label="A", system_label="D",
        )
        assert "TOO CLOSE TO CALL" in line  # 20 pp < one question out of 4 = 25 pp


# ── 4. Cost accounting ───────────────────────────────────────────────────────
class TestPrices:
    def test_a_pinned_model_id_resolves_to_its_family(self):
        assert (
            cost.resolve_price("gpt-4o-mini-2024-07-18")
            == cost.PRICES_USD_PER_1M_TOKENS["gpt-4o-mini"]
        )

    def test_the_longest_matching_prefix_wins(self):
        assert cost.resolve_price("gpt-4.1-mini") == cost.PRICES_USD_PER_1M_TOKENS["gpt-4.1-mini"]

    def test_an_unknown_model_is_never_guessed_at(self):
        assert cost.resolve_price("some-local-llama") is None
        assert cost.usd(cost.Usage(prompt_tokens=1000), None) is None

    def test_the_table_states_when_it_was_checked(self):
        assert cost.PRICES_CHECKED_ON
        assert cost.PRICING_URL.startswith("https://")


class TestCostArithmetic:
    def test_one_million_prompt_tokens_costs_the_input_price(self):
        price = cost.PRICES_USD_PER_1M_TOKENS["gpt-4o-mini"]
        assert cost.usd(cost.Usage(prompt_tokens=1_000_000), price) == pytest.approx(0.15)
        assert cost.usd(cost.Usage(completion_tokens=1_000_000), price) == pytest.approx(0.60)

    def test_prompt_and_completion_are_priced_separately(self):
        price = cost.PRICES_USD_PER_1M_TOKENS["gpt-4o-mini"]
        usage = cost.Usage(prompt_tokens=500_000, completion_tokens=100_000)
        assert cost.usd(usage, price) == pytest.approx(0.075 + 0.06)

    def test_usd_is_formatted_to_four_decimals(self):
        assert cost.format_usd(0.00012345) == "$0.0001"
        assert cost.format_usd(1.5) == "$1.5000"
        assert cost.format_usd(None) == "n/a"

    def test_token_estimation_rounds_up_and_handles_empty(self):
        assert cost.estimate_tokens("") == 0
        assert cost.estimate_tokens("abcde") == 2  # ceil(5/4)
        assert cost.tokens_from_chars(-10) == 0

    def test_usages_add_and_keep_their_estimated_flag(self):
        measured = cost.Usage(prompt_tokens=10, completion_tokens=2, calls=1)
        guessed = cost.Usage(prompt_tokens=4, completion_tokens=1, calls=1, estimated_calls=1)
        total = measured + guessed
        assert total.prompt_tokens == 14
        assert total.total_tokens == 17
        assert total.calls == 2
        # An estimate mixed into a measurement is still an estimate.
        assert total.estimated is True
        assert measured.estimated is False

    def test_scaling_a_measured_call_scales_everything_including_the_flag(self):
        one = cost.estimate_call(prompt_chars=400, completion_tokens=50)
        ten = one.scaled(10)
        assert ten.prompt_tokens == 1000  # 400/4 * 10
        assert ten.completion_tokens == 500
        assert ten.calls == 10
        assert ten.estimated is True


class TestUsageFromResponse:
    class _Response:
        def __init__(self, content="", usage_metadata=None, response_metadata=None):
            self.content = content
            if usage_metadata is not None:
                self.usage_metadata = usage_metadata
            if response_metadata is not None:
                self.response_metadata = response_metadata

    def test_langchain_usage_metadata_is_a_measurement(self):
        usage = cost.usage_from_response(
            self._Response("hi", usage_metadata={"input_tokens": 120, "output_tokens": 30})
        )
        assert (usage.prompt_tokens, usage.completion_tokens) == (120, 30)
        assert usage.estimated is False

    def test_the_raw_provider_block_is_also_a_measurement(self):
        usage = cost.usage_from_response(
            self._Response(
                "hi",
                response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            )
        )
        assert (usage.prompt_tokens, usage.completion_tokens) == (7, 3)
        assert usage.estimated is False

    def test_no_metadata_falls_back_to_characters_and_says_so(self):
        usage = cost.usage_from_response(self._Response("abcdefgh"), prompt_text="a" * 40)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 2
        assert usage.estimated is True
        assert usage.estimated_calls == 1

    def test_an_empty_metadata_block_is_treated_as_missing(self):
        """Some providers attach the key and fill it with zeroes."""
        usage = cost.usage_from_response(
            self._Response("abcd", usage_metadata={"input_tokens": 0, "output_tokens": 0}),
            prompt_text="abcd",
        )
        assert usage.estimated is True
        assert usage.total_tokens == 2


class TestCostLedger:
    def test_it_accumulates_and_prices(self):
        ledger = cost.CostLedger("gpt-4o-mini", label="Ingestion")
        ledger.record(cost.Usage(prompt_tokens=1_000_000, calls=1))
        ledger.record(cost.Usage(completion_tokens=1_000_000, calls=1))
        assert ledger.usage.calls == 2
        assert ledger.usd() == pytest.approx(0.75)
        assert "$0.7500" in "\n".join(ledger.lines())

    def test_every_dollar_figure_is_labelled_an_estimate(self):
        ledger = cost.CostLedger("gpt-4o-mini")
        ledger.record(cost.Usage(prompt_tokens=1000, calls=1))
        text = "\n".join(ledger.lines())
        assert "ESTIMATED" in text
        assert cost.PRICES_CHECKED_ON in text

    def test_an_unpriced_model_reports_tokens_and_no_dollars(self):
        ledger = cost.CostLedger("some-local-llama")
        ledger.record(cost.Usage(prompt_tokens=1000, calls=1))
        assert ledger.usd() is None
        assert "no price on file" in "\n".join(ledger.lines())

    def test_estimated_calls_are_disclosed_in_the_report(self):
        ledger = cost.CostLedger("gpt-4o-mini")
        ledger.record(cost.Usage(prompt_tokens=10, calls=1, estimated_calls=1))
        assert "reported no usage metadata" in "\n".join(ledger.lines())

    def test_it_works_as_a_context_manager_and_times_the_block(self):
        with cost.CostLedger("gpt-4o-mini") as ledger:
            ledger.record(cost.Usage(prompt_tokens=4, calls=1))
        assert ledger.elapsed >= 0.0
        assert ledger.summary()["prompt_tokens"] == 4

    def test_it_does_not_swallow_an_exception_a_failed_run_still_cost_money(self):
        with pytest.raises(ValueError), cost.CostLedger("gpt-4o-mini"):
            raise ValueError("boom")

    def test_merging_ledgers_for_different_models_is_refused(self):
        a = cost.CostLedger("gpt-4o-mini")
        b = cost.CostLedger("gpt-4o")
        with pytest.raises(ValueError, match="wrong rate"):
            a.merge(b)

    def test_the_summary_is_plain_data(self):
        ledger = cost.CostLedger("gpt-4o-mini", label="Ingestion")
        ledger.record(cost.Usage(prompt_tokens=1_000_000, calls=3, estimated_calls=1))
        summary = ledger.summary()
        assert summary["label"] == "Ingestion"
        assert summary["calls"] == 3
        assert summary["estimated"] is True
        assert summary["usd"] == pytest.approx(0.15)


class TestMeteredRunnable:
    """The ledger has to work wrapped *around* the untouched ingestion path."""

    class _Model:
        """The smallest thing that looks like a LangChain chat model."""

        def __init__(self):
            self.calls = 0

        def invoke(self, value, config=None):
            self.calls += 1
            return TestUsageFromResponse._Response(
                "{}", usage_metadata={"input_tokens": 900, "output_tokens": 100}
            )

        async def ainvoke(self, value, config=None):
            return self.invoke(value, config)

    async def test_it_records_the_providers_own_numbers_without_changing_the_result(self):
        model = self._Model()
        ledger = cost.CostLedger("gpt-4o-mini", label="Ingestion")
        response = await cost.metered(model, ledger).ainvoke("extract this")
        assert model.calls == 1
        assert response.usage_metadata["input_tokens"] == 900
        assert ledger.usage.prompt_tokens == 900
        assert ledger.usage.completion_tokens == 100
        assert ledger.usage.calls == 1
        assert ledger.usage.estimated is False  # measured, not guessed
        assert ledger.usd() == pytest.approx((900 * 0.15 + 100 * 0.60) / 1_000_000)


class TestRunEstimate:
    def test_the_estimate_counts_one_call_per_paragraph(self, corpus):
        usage = estimate_ingestion(corpus, theme=DEFAULT_THEME)
        assert usage.calls == len(corpus.paragraphs)
        assert usage.estimated is True

    def test_the_estimate_includes_the_prompt_template_not_only_the_paragraph(self, corpus):
        """On a short Wikipedia paragraph the template is most of the input."""
        usage = estimate_ingestion(corpus, theme=DEFAULT_THEME)
        paragraph_only = cost.tokens_from_chars(corpus.total_chars)
        assert usage.prompt_tokens > paragraph_only * 2

    def test_more_questions_cost_more(self):
        one = build_corpus(parsed(hotpot_item("q1", "?")))
        two = build_corpus(
            parsed(hotpot_item("q1", "?"), hotpot_item("q2", "?", gold=("Eta", "Theta")))
        )
        assert estimate_ingestion(two).prompt_tokens > estimate_ingestion(one).prompt_tokens


# ── 5. --dry-run spends nothing ──────────────────────────────────────────────
@pytest.fixture
def no_spending(monkeypatch):
    """Make every route to a paid API — or to Neo4j, or to the network — explode."""
    calls: list[str] = []

    def explode(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was called during a run that must not spend")

        return _raise

    from app import neo4j_driver
    from app.services import chat_engine, chunk_store, graph_builder, llm_provider

    monkeypatch.setattr(llm_provider, "get_chat_llm", explode("llm_provider.get_chat_llm"))
    monkeypatch.setattr(llm_provider, "get_embeddings", explode("llm_provider.get_embeddings"))
    monkeypatch.setattr(graph_builder, "get_chat_llm", explode("graph_builder.get_chat_llm"))
    monkeypatch.setattr(graph_builder, "get_embeddings", explode("graph_builder.get_embeddings"))
    monkeypatch.setattr(chunk_store, "get_embeddings", explode("chunk_store.get_embeddings"))
    monkeypatch.setattr(chat_engine, "get_embeddings", explode("chat_engine.get_embeddings"))
    monkeypatch.setattr(neo4j_driver, "get_driver", explode("neo4j_driver.get_driver"))
    # The dataset is on disk in these tests; reaching for the network would mean
    # the --data / --no-download contract is broken.
    monkeypatch.setattr(hotpotqa, "ensure_dataset", explode("hotpotqa.ensure_dataset"))
    monkeypatch.setattr(hotpotqa, "_open", explode("hotpotqa._open"))
    return calls


class TestDryRun:
    async def test_it_calls_no_model_touches_no_database_and_exits_zero(
        self, dataset_file, no_spending, capsys
    ):
        code = await main(
            [
                "--dry-run",
                "--questions", "2",
                "--seed", "1",
                "--data", str(dataset_file),
                "--no-download",
                "--model", "gpt-4o-mini",
            ]
        )
        assert code == 0
        assert no_spending == []  # nothing paid was even reached
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "no model was called" in out
        assert "$" in out  # an estimate was actually produced

    async def test_it_names_the_command_that_would_spend_money(
        self, dataset_file, no_spending, capsys
    ):
        await main(
            [
                "--dry-run", "--questions", "2", "--seed", "1",
                "--data", str(dataset_file), "--no-download",
            ]
        )
        out = capsys.readouterr().out
        assert "python -m benchmarks.public.run_hotpotqa --questions 2" in out
        assert "--seed 1" in out

    async def test_a_missing_dataset_is_an_error_not_a_download(self, tmp_path, no_spending):
        code = await main(
            [
                "--dry-run", "--questions", "1",
                "--data", str(tmp_path / "nope.json"),
                "--no-download",
            ]
        )
        assert code == 1
        assert no_spending == []

    async def test_the_estimate_is_the_same_every_time_for_the_same_flags(
        self, dataset_file, no_spending, capsys
    ):
        args = [
            "--dry-run", "--questions", "3", "--seed", "3",
            "--data", str(dataset_file), "--no-download", "--model", "gpt-4o-mini",
        ]
        assert await main(args) == 0
        first = capsys.readouterr().out
        assert await main(args) == 0
        assert capsys.readouterr().out == first

    def test_the_estimate_refuses_a_run_that_would_blow_the_cap(self, corpus):
        lines = dry_run_lines(corpus, model="gpt-4o-mini", theme=DEFAULT_THEME, max_usd=0.0)
        assert any("exceeds --max-usd" in line for line in lines)

    def test_an_unpriced_model_says_so_instead_of_inventing_a_number(self, corpus):
        lines = dry_run_lines(corpus, model="my-local-model", theme=DEFAULT_THEME, max_usd=1.0)
        assert any("no price on file" in line for line in lines)

    def test_the_plan_names_one_extraction_call_per_paragraph(self, corpus):
        text = "\n".join(
            dry_run_lines(corpus, model="gpt-4o-mini", theme=DEFAULT_THEME, max_usd=1.0)
        )
        assert f"{len(corpus.paragraphs)} extraction calls" in text
        assert "ESTIMATE" in text


class TestRefusals:
    """A run that cannot be honestly reported has to stop, not print a number."""

    async def test_reuse_graph_refuses_a_graph_that_is_not_this_sample(
        self, corpus, monkeypatch, no_spending
    ):
        from app.services import graph_schema
        from benchmarks.public import run_hotpotqa

        async def no_schema():
            return None

        async def present():
            return {"Alpha"}  # the sample needs six paragraphs, not one

        monkeypatch.setattr(graph_schema, "ensure_schema", no_schema)
        monkeypatch.setattr(run_hotpotqa, "graph_documents", present)
        args = SimpleNamespace(
            reuse_graph=True, model="gpt-4o-mini", theme=DEFAULT_THEME,
            baseline_k=2, graph_k=8, max_usd=1.0, ingest_concurrency=1,
        )
        with pytest.raises(ReuseError, match="not this sample"):
            await run_hotpotqa._run_systems(corpus, args, [])
        assert no_spending == []

    async def test_a_projected_cost_over_the_cap_stops_before_ingestion(
        self, dataset_file, monkeypatch, no_spending, capsys
    ):
        from app import neo4j_driver

        async def reachable():
            return True

        async def close():
            return None

        monkeypatch.setattr(neo4j_driver, "verify_connectivity", reachable)
        monkeypatch.setattr(neo4j_driver, "close_driver", close)
        code = await main(
            [
                "--questions", "3", "--seed", "1", "--data", str(dataset_file),
                "--no-download", "--model", "gpt-4o-mini", "--max-usd", "0.0",
            ]
        )
        assert code == 5
        assert no_spending == []
        assert "exceeds --max-usd" in capsys.readouterr().err


class TestLoudness:
    def test_a_modest_run_is_quiet(self):
        assert loud_banner(20) == []

    def test_a_big_run_shouts(self):
        banner = "\n".join(loud_banner(200))
        assert "🚨" in banner and "200 QUESTIONS" in banner

    def test_the_model_name_comes_from_the_configured_provider(self):
        from app.config import get_settings

        settings = get_settings()
        settings.llm_provider = "openai"
        settings.openai_chat_model = "gpt-4o-mini"
        assert chat_model_name(settings) == "gpt-4o-mini"


# ── 6. The report is generated, never typed ──────────────────────────────────
def _report() -> Report:
    """A tiny synthetic run — enough to exercise every reporting path."""

    def result(qid, qtype, credited, chars, prose_chars, hits):
        return QuestionResult(
            qid=qid, question=f"question {qid}", qtype=qtype,
            gold=("Alpha", "Beta"), credited=credited,
            context_chars=chars, prose_chars=prose_chars,
            supporting_hits=hits, supporting_total=2,
        )

    baseline = [
        result("q1", "bridge", {"Alpha": PROSE}, 200, 200, 1),
        result("q2", "comparison", {"Alpha": PROSE, "Beta": PROSE}, 200, 200, 2),
    ]
    chunked = [
        result("q1", "bridge", {"Alpha": PROSE, "Beta": ENTITY}, 900, 300, 1),
        result("q2", "comparison", {"Alpha": PROSE, "Beta": PROSE}, 900, 300, 2),
    ]
    graph = [
        result("q1", "bridge", {"Alpha": ENTITY, "Beta": EDGE}, 600, 0, 0),
        result("q2", "comparison", {"Alpha": ENTITY}, 600, 0, 0),
    ]
    corpus = build_corpus(
        parsed(hotpot_item("q1", "?"), hotpot_item("q2", "?", qtype="comparison"))
    )
    sweep = [(1, aggregate(baseline)), (2, aggregate(chunked))]
    return Report(
        corpus=corpus,
        baseline=Run("A", "Passage vector RAG", baseline),
        distractor=Run("A_d", "Passage vector RAG (distractor)", baseline),
        graph=Run("C", "GraphRAG", graph),
        chunked=Run("D", "GraphRAG + chunks", chunked),
        sweep=sweep,
        baseline_k=2,
        graph_k=8,
        ledgers=[cost.CostLedger("gpt-4o-mini", label="Ingestion")],
        settings_summary={"extraction_temperature": 0.1, "model": "gpt-4o-mini"},
    )


class TestReport:
    def test_the_markdown_carries_the_table_the_headline_and_the_threats(self):
        markdown = results_markdown(_report())
        assert "| System |" in markdown
        assert "HEADLINE (computed, not asserted)" in markdown
        assert "Threats to validity" in markdown
        assert "Bridge vs. comparison" in markdown
        assert "CC BY-SA 4.0" in markdown

    def test_the_strict_rule_is_explained_where_row_c_scores_zero(self):
        markdown = results_markdown(_report())
        assert "by construction" in markdown
        assert "0.0%" in markdown

    def test_the_threats_section_names_the_sample_size_and_the_floor(self):
        text = "\n".join(results_markdown(_report()).splitlines())
        assert "SAMPLE SIZE" in text
        assert "RETRIEVAL ONLY" in text
        assert "PRICES ARE HAND-RECORDED" in text

    def test_the_headline_is_computed_from_the_sweep_not_asserted(self):
        agg = {"gold_recall": 0.9, "both_gold": 0.9, "mean_context_chars": 1000.0, "n": 10}
        cheap = {"gold_recall": 0.95, "both_gold": 0.95, "mean_context_chars": 100.0, "n": 10}
        lines = headline(agg, [(1, cheap)])
        assert "matches it on EVERY metric while reading LESS text" in "\n".join(lines)

    def test_a_graph_that_cannot_be_matched_is_reported_as_such(self):
        agg = {"gold_recall": 0.9, "both_gold": 0.9, "mean_context_chars": 1000.0, "n": 10}
        weak = {"gold_recall": 0.1, "both_gold": 0.1, "mean_context_chars": 100.0, "n": 10}
        lines = headline(agg, [(1, weak)])
        assert "survives a budget match" in "\n".join(lines)

    def test_the_cost_section_is_present_even_when_nothing_was_spent(self):
        report = _report()
        report.reused_graph = True
        text = "\n".join(report.analysis())
        assert "Ingestion: SKIPPED (--reuse-graph)" in text
        assert "$0.0000" in text

    def test_an_oversized_run_is_disclosed_in_the_report_not_only_in_the_terminal(self):
        """The banner promises 'the report will say so'; this is the report saying so."""
        from benchmarks.public.run_hotpotqa import LOUD_QUESTION_THRESHOLD, threats_lines

        report = _report()
        big = list(report.chunked.results) * (LOUD_QUESTION_THRESHOLD + 1)
        report.chunked = Run("D", "GraphRAG + chunks", big)
        assert "OVERSIZED RUN" in "\n".join(threats_lines(report))
        # …and stays out of the way of a modest one.
        assert "OVERSIZED RUN" not in "\n".join(threats_lines(_report()))

    def test_the_bridge_and_comparison_rows_are_never_averaged_together(self):
        text = "\n".join(_report().analysis())
        assert "bridge (N=1" in text
        assert "comparison (N=1" in text
        assert "UNDERPOWERED" in text
