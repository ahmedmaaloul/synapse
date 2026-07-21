# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Tests for the GraphRAG vs. vector-RAG benchmark.

Four jobs, all hermetic (no network, no LLM, no Neo4j):

  1. **Dataset integrity** — a benchmark nobody can audit is worth nothing, so
     every claim ``dataset.json`` makes about itself is verified here: the
     extraction is internally consistent, the passages really do mention the
     entities they declare, no entity name can be credited by accident, and all
     three fairness rules hold —

       R1  a question names at most ONE of its required facts (its anchor), and
           shares no content-word *family* with the NAME or the hand-written
           DESCRIPTION of any other required fact. The description half is the
           one that matters most in practice: descriptions are what GraphRAG
           embeds and matches on, so a description echoing the question's
           distinctive words would hand the answer over for free without a
           single edge being walked. "Family" rather than "stem" because suffix
           stripping alone only ever collides words that differ at the end, and
           the leaks that actually occurred here differed in the middle
           ("dataset" / "database", "back" / "backpropagation");
       R2  no single passage contains all of a question's required facts, so
           every question genuinely needs more than one hop;
       R3  no entity description names another entity, so relationships are
           carried by the graph rather than smuggled into text that the
           edgeless systems read too.

     Plus the dataset's own prose: the count of distractors that mention a
     required fact in passing is pinned here, so the README cannot drift back
     into claiming distractors "carry none of" the required facts.

  2. **Metric correctness** — the scoring and reporting primitives are checked
     against hand-computed values, including the ones that decide whether the
     report is allowed to claim a win.

  3. **Channel attribution** — the permissive scoring rule ("the fact's name is
     somewhere in the context") is not neutral between these systems, because a
     GraphRAG context prints entity names as scaffolding while a passage context
     contains only prose. The harness therefore splits every context into
     ``prose`` / ``entity`` / ``edge`` channels and scores a strict rule as well
     as the permissive one. These tests pin the split, pin that the three
     channels cover every name the context contains — a name that fell through
     would be silently dropped from the strict column — and pin that the strict
     rule really is zero-by-construction for the prose-free systems.

  4. **README ↔ results consistency** — an audit found numbers in ``README.md``
     that a fresh run did not reproduce. The numeric core of the README is now
     *generated* into a marked region from the same function that writes
     ``results.md``; :func:`test_the_readme_generated_region_is_verbatim_in_the_results`
     and :func:`test_every_number_in_the_readme_appears_in_the_generated_results`
     fail if that ever stops being true, so the defect cannot recur silently.
"""

from __future__ import annotations

import re
from itertools import combinations

import pytest

from benchmarks.run_benchmark import (
    CHANNELS,
    CLEAR_QUERIES,
    DATASET_PATH,
    EDGE,
    ENTITY,
    METRICS,
    PATHS_HEADING,
    PREFIX_FAMILY_LEN,
    PROSE,
    README_BEGIN,
    README_END,
    README_PATH,
    RESULTS_PATH,
    SOURCES_HEADING,
    STRICT_METRICS,
    SWEEP_KS,
    Channels,
    DatasetError,
    IntegrityError,
    QuestionResult,
    ablation_note,
    aggregate,
    chunk_note,
    classify_hits,
    contaminated,
    content_words,
    context_matched,
    context_matched_note,
    cosine_similarity,
    edge_segment,
    edgeless_budget_lines,
    entity_context_block,
    entity_document,
    evaluate_baseline,
    evaluate_entity_rag,
    excerpt_overlap,
    fact_recall,
    facts_found,
    headline,
    hit_provenance,
    leaks,
    load_dataset,
    misses_note,
    prose_segment,
    provenance_note,
    rank_by_similarity,
    render_readme,
    same_family,
    scale_note,
    scoring_rule_note,
    smallest_k_reaching,
    strip_edges,
    strip_graph_structure,
    sweep_ks,
    verdict,
)

DATASET = load_dataset()
PASSAGES = DATASET["passages"]
ENTITIES = DATASET["entities"]
RELATIONSHIPS = DATASET["relationships"]
QUESTIONS = DATASET["questions"]
ENTITY_NAMES = [e["name"] for e in ENTITIES]
ENTITY_BY_NAME = {e["name"]: e for e in ENTITIES}


def _mentions(passage: dict) -> set[str]:
    """Entity names that literally occur in a passage's text."""
    return set(facts_found(passage["text"], ENTITY_NAMES))


def _min_passages_to_cover(facts: list[str], max_size: int = 4) -> int:
    """Smallest number of passages whose union mentions every fact.

    Exhaustive up to ``max_size`` (18 choose 4 = 3060 — trivial); returns
    ``max_size + 1`` when no cover of that size exists.
    """
    wanted = {f.lower() for f in facts}
    covers = [{m.lower() for m in _mentions(p)} for p in PASSAGES]
    for size in range(1, max_size + 1):
        for combo in combinations(covers, size):
            if wanted <= set().union(*combo):
                return size
    return max_size + 1


# ── Dataset shape ────────────────────────────────────────────────────────────
def test_dataset_has_the_advertised_shape():
    core = [p for p in PASSAGES if p["role"] == "core"]
    distractors = [p for p in PASSAGES if p["role"] == "distractor"]
    assert 12 <= len(core) <= 18
    assert 10 <= len(QUESTIONS) <= 14
    # Distractors exist so top-k has to compete for its slots: on a corpus of
    # only 18 passages, retrieving 4 of them is 22% of everything, which makes
    # any multi-hop question trivially coverable and the comparison meaningless.
    assert len(distractors) >= 10
    assert len(ENTITIES) >= 20
    assert len(RELATIONSHIPS) >= 20


def test_every_passage_declares_a_known_role():
    assert {p["role"] for p in PASSAGES} == {"core", "distractor"}


def test_distractor_passages_are_not_needed_by_any_question():
    """Distractors add competition for retrieval slots, never missing facts."""
    core_ids = {p["id"] for p in PASSAGES if p["role"] == "core"}
    core_mentions: set[str] = set()
    for passage in PASSAGES:
        if passage["id"] in core_ids:
            core_mentions |= set(passage["entities"])
    for question in QUESTIONS:
        assert set(question["required_facts"]) <= core_mentions, question["id"]


def test_the_documented_distractor_leakage_is_the_real_one():
    """The dataset used to claim distractors "carry none of" the required facts.

    Some of them do mention one in passing, which is realistic; the claim was
    false. The true list is written down in ``dataset.json`` and pinned here, so
    prose and corpus cannot drift apart again.

    An earlier version of this docstring added that "a leaked mention can only
    help the *passage* baseline, never GraphRAG". That is false too, and is
    deleted rather than softened: the shipped system's excerpt channel retrieves
    the same passages the baseline ranks, so a leaked mention helps every system
    that returns prose (A, A′ and D). It cannot help the rows that return no
    prose at all (B, B″, B′, C), and :func:`test_the_strict_rule_is_zero_for_the_prose_free_systems`
    is what makes "returns no prose" a checked property rather than a claim.
    """
    required = {f for q in QUESTIONS for f in q["required_facts"]}
    leaking = sorted(
        p["id"]
        for p in PASSAGES
        if p["role"] == "distractor" and required & set(p["entities"])
    )
    assert leaking == DATASET["distractors_leaking_required_facts"]


def test_no_entity_description_names_another_entity():
    """Fairness rule R3 — relationships live in the graph, not in the prose.

    If ``Charles Babbage``'s description said "designed the Analytical Engine",
    every system that retrieves Babbage would be credited with the Analytical
    Engine for free, and the edgeless entity baseline would be scoring the
    graph's job. Descriptions therefore never name another entity.
    """
    for entity in ENTITIES:
        others = [n for n in ENTITY_NAMES if n != entity["name"]]
        assert facts_found(entity["description"], others) == [], entity["name"]


def test_entity_names_never_match_inside_a_longer_word():
    """The scoring rule is a substring test, so substrings must not lie.

    ("CERN" hiding inside "concerned" is exactly how a benchmark quietly starts
    crediting facts nobody retrieved.)
    """
    texts = (
        [(p["id"], p["text"]) for p in PASSAGES]
        + [(f"description:{e['name']}", e["description"]) for e in ENTITIES]
        + [(q["id"], q["question"]) for q in QUESTIONS]
    )
    for label, text in texts:
        for name in ENTITY_NAMES:
            substring = name.lower() in text.lower()
            whole_word = re.search(
                rf"(?<![A-Za-z0-9]){re.escape(name.lower())}(?![A-Za-z0-9])", text.lower()
            )
            assert substring == bool(whole_word), f"{label}: {name!r} matches inside a word"


def test_ids_are_unique():
    assert len({p["id"] for p in PASSAGES}) == len(PASSAGES)
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    assert len(set(ENTITY_NAMES)) == len(ENTITY_NAMES)


def test_entities_are_well_formed():
    for entity in ENTITIES:
        assert entity["name"].strip(), entity
        assert entity["type"].strip(), entity
        assert entity["description"].strip(), entity


def test_load_dataset_rejects_a_broken_file(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"passages": []}', encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(broken)
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.json")


# ── The shipped extraction is internally consistent ──────────────────────────
def test_every_relationship_endpoint_is_a_declared_entity():
    names = set(ENTITY_NAMES)
    for rel in RELATIONSHIPS:
        assert rel["source"] in names, f"unknown source: {rel['source']}"
        assert rel["target"] in names, f"unknown target: {rel['target']}"
        assert rel["source"] != rel["target"], f"self-loop: {rel}"
        assert rel["type"].strip(), rel


def test_every_declared_passage_entity_is_a_declared_entity():
    names = set(ENTITY_NAMES)
    for passage in PASSAGES:
        for name in passage["entities"]:
            assert name in names, f"{passage['id']} mentions undeclared {name!r}"


def test_declared_passage_entities_match_what_the_text_actually_says():
    """No inflation and no omission: declared mentions == detected mentions.

    This is what makes crediting a system for "retrieving a fact" meaningful —
    the passages genuinely contain the entity names the dataset claims.
    """
    for passage in PASSAGES:
        assert _mentions(passage) == set(passage["entities"]), passage["id"]


def test_no_entity_name_is_a_substring_of_another():
    """Guards the scoring rule: a substring match can never credit two facts."""
    for a, b in combinations(ENTITY_NAMES, 2):
        assert a.lower() not in b.lower(), f"{a!r} is a substring of {b!r}"
        assert b.lower() not in a.lower(), f"{b!r} is a substring of {a!r}"


def test_every_entity_is_mentioned_by_at_least_one_passage():
    mentioned: set[str] = set()
    for passage in PASSAGES:
        mentioned |= set(passage["entities"])
    assert mentioned == set(ENTITY_NAMES)


def test_the_graph_is_connected():
    """A disconnected graph would make some questions unreachable by traversal."""
    adjacency: dict[str, set[str]] = {name: set() for name in ENTITY_NAMES}
    for rel in RELATIONSHIPS:
        adjacency[rel["source"]].add(rel["target"])
        adjacency[rel["target"]].add(rel["source"])

    seen = {ENTITY_NAMES[0]}
    frontier = [ENTITY_NAMES[0]]
    while frontier:
        node = frontier.pop()
        for neighbour in adjacency[node] - seen:
            seen.add(neighbour)
            frontier.append(neighbour)
    assert seen == set(ENTITY_NAMES), f"unreachable: {sorted(set(ENTITY_NAMES) - seen)}"


# ── The questions really are multi-hop ───────────────────────────────────────
def test_required_facts_are_declared_entities():
    names = set(ENTITY_NAMES)
    for question in QUESTIONS:
        assert question["required_facts"], question["id"]
        assert len(set(question["required_facts"])) == len(question["required_facts"])
        for fact in question["required_facts"]:
            assert fact in names, f"{question['id']} requires undeclared {fact!r}"


def test_a_question_names_at_most_one_of_its_required_facts():
    """Fairness rule R1 — the anti-leak guarantee.

    A question may name its anchor and nothing else. Without this rule a
    question like "which architecture underlies BERT and GPT?" hands three of
    its own answers to a lexical/semantic ranker, and the benchmark measures
    string overlap rather than multi-hop retrieval.
    """
    for question in QUESTIONS:
        named = facts_found(question["question"], question["required_facts"])
        assert len(named) <= 1, f"{question['id']} leaks {named}"


def test_a_question_shares_no_content_word_with_a_non_anchor_fact_description():
    """Fairness rule R1, the half that actually decides the benchmark.

    Checking R1 only against entity *names* is close to worthless: GraphRAG's
    seed search embeds and full-text-indexes ``name — description``, so it is
    the hand-written descriptions that a question can leak into. A description
    that echoes the question's distinctive words ("Which algorithm overcame …"
    ↔ "Gradient algorithm that trains …") hands over a required fact with no
    traversal at all, and the benchmark would be measuring the author's word
    choice.

    So: for every question, no NON-anchor required fact may share a
    stopword-filtered content-word *family* with the question — neither in its
    description nor in its own name. Family, not stem: an auditor pointed out
    that stem equality is blind to exactly the leak class R1 advertises catching
    (q05 asked about a "dataset" while a required fact's description said
    "database"; q04 asked what Hinton popularised "back" in 1986 while the
    answer was "Backpropagation"). Both are now caught, and both were fixed in
    the dataset rather than excused.
    """
    for question in QUESTIONS:
        for fact in question["required_facts"]:
            if fact == question["anchor"]:
                continue
            entity = ENTITY_BY_NAME[fact]
            leaked_by_description = leaks(question["question"], entity["description"])
            leaked_by_name = leaks(question["question"], fact)
            assert not leaked_by_description, (
                f"{question['id']} leaks {fact!r} through its description "
                f"{entity['description']!r}: {sorted(leaked_by_description)}"
            )
            assert not leaked_by_name, (
                f"{question['id']} leaks {fact!r} through its name: {sorted(leaked_by_name)}"
            )


def test_the_declared_anchor_is_the_named_required_fact():
    for question in QUESTIONS:
        anchor = question["anchor"]
        assert anchor in question["required_facts"], question["id"]
        assert facts_found(question["question"], question["required_facts"]) == [anchor], (
            question["id"]
        )


def test_no_single_passage_answers_any_question():
    """The defining property of the benchmark: every question spans passages."""
    for question in QUESTIONS:
        wanted = {f.lower() for f in question["required_facts"]}
        for passage in PASSAGES:
            covered = {m.lower() for m in _mentions(passage)}
            assert not wanted <= covered, (
                f"{question['id']} is answerable from {passage['id']} alone"
            )


def test_every_question_needs_at_least_two_passages():
    for question in QUESTIONS:
        needed = _min_passages_to_cover(question["required_facts"])
        assert needed >= 2, f"{question['id']} needs only {needed} passage(s)"
        assert needed <= 4, f"{question['id']} is not answerable from the corpus"


def test_required_facts_are_connected_in_the_graph():
    """Every question's facts sit in one connected component, so traversal can reach them."""
    adjacency: dict[str, set[str]] = {name: set() for name in ENTITY_NAMES}
    for rel in RELATIONSHIPS:
        adjacency[rel["source"]].add(rel["target"])
        adjacency[rel["target"]].add(rel["source"])

    for question in QUESTIONS:
        facts = list(question["required_facts"])
        seen = {facts[0]}
        frontier = [facts[0]]
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node] - seen:
                seen.add(neighbour)
                frontier.append(neighbour)
        assert set(facts) <= seen, question["id"]


# ── Metrics ──────────────────────────────────────────────────────────────────
def test_cosine_similarity_hand_computed():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # 3·1 + 4·0 = 3, |a| = 5, |b| = 1 -> 0.6
    assert cosine_similarity([3.0, 4.0], [1.0, 0.0]) == pytest.approx(0.6)
    # Degenerate inputs score 0 rather than exploding.
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0


def test_rank_by_similarity_orders_by_score_and_breaks_ties_by_index():
    docs = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    assert rank_by_similarity([1.0, 0.0], docs) == [1, 2, 0]


def test_facts_found_is_case_insensitive_and_order_preserving():
    context = "Entity: Ada Lovelace\n  → WROTE_ALGORITHM_FOR → Analytical Engine"
    assert facts_found(context, ["ANALYTICAL ENGINE", "Ada Lovelace", "IBM"]) == [
        "ANALYTICAL ENGINE",
        "Ada Lovelace",
    ]
    assert facts_found("", ["IBM"]) == []


def test_fact_recall_hand_computed():
    assert fact_recall(["A", "B"], ["A", "B", "C", "D"]) == pytest.approx(0.5)
    assert fact_recall(["a"], ["A", "B"]) == pytest.approx(0.5)  # case-insensitive
    assert fact_recall([], ["A"]) == 0.0
    assert fact_recall(["A", "B"], ["A", "B"]) == pytest.approx(1.0)
    # Facts retrieved that were not required cannot inflate the score.
    assert fact_recall(["A", "Z"], ["A", "B"]) == pytest.approx(0.5)
    assert fact_recall(["A"], []) == 0.0


def test_question_result_missing_and_complete():
    result = QuestionResult(
        qid="q01", question="?", required=["A", "B"], found=["a"], context_chars=10
    )
    assert result.missing == ["B"]
    assert result.recall == pytest.approx(0.5)
    assert result.complete is False

    full = QuestionResult(
        qid="q02", question="?", required=["A"], found=["A"], context_chars=4
    )
    assert full.missing == []
    assert full.complete is True


def test_aggregate_hand_computed():
    results = [
        QuestionResult("q1", "?", ["A", "B"], ["A", "B"], 100),  # recall 1.0, complete
        QuestionResult("q2", "?", ["A", "B"], ["A"], 200),  # recall 0.5
        QuestionResult("q3", "?", ["A", "B"], [], 300),  # recall 0.0
    ]
    agg = aggregate(results)
    assert agg["fact_recall"] == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3
    assert agg["full_coverage"] == pytest.approx(1 / 3)
    assert agg["mean_context_chars"] == pytest.approx(200.0)
    assert agg["n"] == 3
    # No channel attribution was supplied, so nothing is credited to prose and
    # the strict rule scores zero. Silence is not a pass under the strict rule.
    assert agg["strict_fact_recall"] == 0.0
    assert agg["strict_full_coverage"] == 0.0

    empty = aggregate([])
    assert empty == {
        "fact_recall": 0.0,
        "full_coverage": 0.0,
        "strict_fact_recall": 0.0,
        "strict_full_coverage": 0.0,
        "mean_context_chars": 0.0,
        "n": 0,
    }


def test_aggregate_scores_both_rules_from_the_channel_attribution():
    """The strict rule counts only the facts credited to retrieved prose."""
    results = [
        QuestionResult(
            "q1", "?", ["A", "B"], ["A", "B"], 100,
            hit_channels={"A": PROSE, "B": PROSE},
        ),
        QuestionResult(
            "q2", "?", ["A", "B"], ["A", "B"], 100,
            # Both credited, but B only because the graph named it as a
            # neighbour — the exact case the strict rule exists to expose.
            hit_channels={"A": PROSE, "B": EDGE},
        ),
        QuestionResult(
            "q3", "?", ["A", "B"], ["A", "B"], 100,
            hit_channels={"A": ENTITY, "B": ENTITY},
        ),
    ]
    agg = aggregate(results)
    assert agg["fact_recall"] == pytest.approx(1.0)
    assert agg["full_coverage"] == pytest.approx(1.0)
    assert agg["strict_fact_recall"] == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3
    assert agg["strict_full_coverage"] == pytest.approx(1 / 3)
    assert results[1].found_strict == ["A"]
    assert results[1].missing_strict == ["B"]
    assert results[1].strict_complete is False


def _sweep(*rows: tuple[int, float, float, float]) -> list[tuple[int, dict]]:
    return [
        (k, {"fact_recall": recall, "full_coverage": cov, "mean_context_chars": chars})
        for k, recall, cov, chars in rows
    ]


def test_verdict_reports_a_baseline_win_truthfully():
    baseline = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 100.0}
    graph = {"fact_recall": 0.5, "full_coverage": 0.8, "mean_context_chars": 200.0}
    sweep = _sweep((1, 0.9, 0.8, 100.0))
    lines = verdict(baseline, graph, sweep)
    assert "passage vector RAG ahead" in lines[0]
    assert "TIE" in lines[2]
    assert "2.00x" in lines[-1]


def test_verdict_never_claims_an_unqualified_win():
    """Every metric line must be followed by the parity k that qualifies it.

    This is the regression guard for the worst thing this file used to do:
    print "GraphRAG wins" while the sweep three rows lower showed a cheaper
    baseline matching it.
    """
    baseline = {"fact_recall": 0.5, "full_coverage": 0.2, "mean_context_chars": 100.0}
    graph = {"fact_recall": 0.95, "full_coverage": 0.85, "mean_context_chars": 400.0}
    sweep = _sweep((4, 0.5, 0.2, 100.0), (6, 0.95, 0.85, 200.0))
    lines = verdict(baseline, graph, sweep)

    assert "GraphRAG ahead at the configured k" in lines[0]
    assert "+45.0 pp" in lines[0]  # percentage points, not fractions
    assert "matches it at top-6" in lines[1]
    assert "50% of GraphRAG's context" in lines[1]
    assert "GraphRAG ahead at the configured k" in lines[2]
    assert "+65.0 pp" in lines[2]
    assert "matches it at top-6" in lines[3]
    # No line is allowed to end the story with a bare win.
    assert not any(line.startswith("Fact Recall: GraphRAG wins") for line in lines)


def test_verdict_will_not_call_a_gap_smaller_than_one_question():
    """A lead worth less than a single question is not a lead.

    With n questions, Full-Coverage moves in steps of 1/n, so on a 14-question
    set a +1.8 pp "win" is a rounding artefact of one item — the exact kind of
    claim a referee rejects. The floor engages only when the question count is
    known; see the sibling test.
    """
    baseline = {"fact_recall": 0.893, "full_coverage": 0.714,
                "mean_context_chars": 831.0, "n": 14}
    graph = {"fact_recall": 0.911, "full_coverage": 0.857,
             "mean_context_chars": 3516.0, "n": 14}
    sweep = _sweep((4, 0.893, 0.714, 831.0), (20, 1.0, 1.0, 4072.0))
    lines = verdict(baseline, graph, sweep)

    # +1.8 pp < one question (7.1 pp) -> refused.
    assert "TOO CLOSE TO CALL" in lines[0]
    assert "No significance is claimed." in lines[0]
    assert "GraphRAG ahead at the configured k" not in lines[0]
    # +14.3 pp = two questions -> reported, and qualified with the size of one.
    assert "GraphRAG ahead at the configured k" in lines[2]
    assert "one question = 7.1 pp" in lines[2]


def test_effect_size_floor_is_inert_when_the_question_count_is_unknown():
    """A missing ``n`` must NOT silently suppress every verdict.

    Defaulting the floor to "1 question = 100 pp" would swallow even a 40 pp
    difference, which is strictly worse than having no floor at all.
    """
    baseline = {"fact_recall": 0.9, "full_coverage": 0.9, "mean_context_chars": 100.0}
    graph = {"fact_recall": 0.5, "full_coverage": 0.5, "mean_context_chars": 400.0}
    lines = verdict(baseline, graph, _sweep((4, 0.9, 0.9, 100.0)))
    assert "TOO CLOSE TO CALL" not in lines[0]
    assert "passage vector RAG ahead" in lines[0]


def test_verdict_says_when_no_baseline_k_catches_up():
    baseline = {"fact_recall": 0.5, "full_coverage": 0.2, "mean_context_chars": 100.0}
    graph = {"fact_recall": 0.95, "full_coverage": 0.85, "mean_context_chars": 400.0}
    sweep = _sweep((4, 0.5, 0.2, 100.0), (12, 0.7, 0.4, 900.0))
    lines = verdict(baseline, graph, sweep)
    assert "No baseline k up to 12" in lines[1]
    assert "best is 70.0% at top-12" in lines[1]


def test_smallest_k_reaching_finds_the_cheapest_matching_setting():
    sweep = _sweep((1, 0.3, 0.0, 100.0), (4, 0.9, 0.5, 400.0), (8, 0.95, 0.9, 800.0))
    assert smallest_k_reaching(sweep, 0.9, "fact_recall")[0] == 4  # exact match counts
    assert smallest_k_reaching(sweep, 0.92, "fact_recall")[0] == 8
    assert smallest_k_reaching(sweep, 0.99, "fact_recall") is None
    assert smallest_k_reaching(sweep, 0.6, "full_coverage")[0] == 8


def test_headline_refuses_to_claim_a_win_the_sweep_contradicts():
    """The exact situation this benchmark was caught in: a cheaper baseline ties."""
    graph = {"fact_recall": 0.946, "full_coverage": 0.857, "mean_context_chars": 2073.0}
    sweep = _sweep((4, 0.893, 0.714, 836.0), (6, 0.946, 0.857, 1256.0))
    lines = headline(graph, sweep)
    assert "94.6%" in lines[0] and "2,073 chars" in lines[0]
    assert "EVERY metric while reading LESS text" in lines[1]
    assert "top-6" in lines[1]
    assert "61% of the context" in lines[1]
    assert "no context saving" in lines[1]


def test_headline_claims_the_win_when_the_sweep_supports_it():
    graph = {"fact_recall": 0.95, "full_coverage": 0.9, "mean_context_chars": 500.0}
    sweep = _sweep((4, 0.6, 0.3, 400.0), (12, 0.95, 0.9, 1500.0))
    lines = headline(graph, sweep)
    assert "survives a budget match" in lines[1]


def test_headline_splits_the_claim_when_only_some_metrics_hold_up():
    graph = {"fact_recall": 0.9, "full_coverage": 0.9, "mean_context_chars": 500.0}
    # Recall is matched cheaply at top-2; coverage only at top-12, on more text.
    sweep = _sweep((2, 0.9, 0.4, 200.0), (12, 0.95, 0.9, 1500.0))
    lines = headline(graph, sweep)
    assert "holds up under a budget match on: Full-Coverage Rate" in lines[1]
    assert "does NOT on: Fact Recall at top-2 using 40% of the context" in lines[1]


def test_ablation_note_isolates_the_graph():
    entity = {"fact_recall": 0.5, "full_coverage": 0.1, "mean_context_chars": 400.0}
    stripped = {"fact_recall": 0.6, "full_coverage": 0.3, "mean_context_chars": 500.0}
    graph = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 1000.0}
    lines = ablation_note(entity, stripped, graph, 8)
    # The header must name EVERYTHING the ablation removes. Saying "the
    # relationship lines" alone was false — the reasoning-path section goes too.
    assert "Relationships lines AND the rendered Reasoning paths section" in lines[0]
    assert "60.0% without the graph → 90.0% with it (+30.0 pp)" in lines[1]
    assert "30.0% without the graph → 80.0% with it (+50.0 pp)" in lines[2]
    assert "2.00x the context" in lines[3]
    # B′ is C minus text, so the two sides are NOT budget-matched. The README
    # used to claim the ablation "is now budget-matched"; the harness now states
    # the residual gap itself, immediately after the cost line it qualifies.
    assert "NOT budget-matched" in lines[4]
    assert "500 chars against C's 1,000" in lines[4]
    assert "controls the *seed set*, not the budget" in lines[4]
    # …and that the ablation only exists under the permissive rule at all.
    assert "Permissive rule only" in lines[5]
    assert "under the strict rule both score 0.0" in lines[5]
    # The standalone entity baseline is reported too, and never confused with B′.
    assert "50.0% / 10.0% on 400 chars" in lines[6]


def test_ablation_note_reports_a_negative_contribution_too():
    entity = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 500.0}
    stripped = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 500.0}
    graph = {"fact_recall": 0.7, "full_coverage": 0.8, "mean_context_chars": 1500.0}
    lines = ablation_note(entity, stripped, graph, 8)
    assert "(-20.0 pp)" in lines[1]
    assert "(+0.0 pp)" in lines[2]


def test_ablation_note_appends_the_edgeless_budget_match_when_a_sweep_is_given():
    """The ablation used to be the only comparison never budget-matched."""
    entity = {"fact_recall": 0.5, "full_coverage": 0.1, "mean_context_chars": 400.0}
    stripped = {"fact_recall": 0.6, "full_coverage": 0.3, "mean_context_chars": 500.0}
    graph = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 1000.0}
    sweep = _sweep((8, 0.6, 0.3, 500.0), (20, 0.7, 0.4, 1200.0))
    plain = ablation_note(entity, stripped, graph, 8)
    with_budget = ablation_note(entity, stripped, graph, 8, sweep)
    assert len(with_budget) > len(plain)
    assert any("Budget check (B″)" in line for line in with_budget)


# ── The ablation's edgeless side is budget-matched, never truncated ──────────
def test_edgeless_budget_lines_report_the_matching_k_and_the_widths():
    graph = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 1000.0}
    sweep = _sweep(
        (8, 0.5, 0.1, 500.0),
        (20, 0.6, 0.2, 1200.0),
        (48, 0.95, 0.85, 3000.0),
        (79, 1.0, 1.0, 4000.0),
    )
    lines = edgeless_budget_lines(sweep, graph, graph_k=8)
    assert "top-20 is the first edgeless entity setting" in lines[0]
    assert "1,200 vs 1,000 chars" in lines[0]
    assert "60.0% / 20.0%" in lines[0]
    assert "still behind on both metrics" in lines[1]
    # Per metric: how much wider the edgeless system has to read to reach C.
    assert "Fact Recall: the edgeless system needs top-48 — 48 of the 79 entities" in lines[2]
    assert "3.0x C's context" in lines[2]
    assert "Full-Coverage Rate: the edgeless system needs top-48" in lines[3]
    # And it must NEVER claim the edgeless system "never catches up": at
    # k = every entity it trivially scores 100%, which the last line states.
    assert "never" not in " ".join(lines)
    assert "trivial 100% any system reaches by retrieving everything" in lines[-1]
    assert "C gets there from 8 ranked seeds" in lines[-1]


def test_edgeless_budget_lines_admit_when_the_edgeless_system_catches_up():
    graph = {"fact_recall": 0.5, "full_coverage": 0.4, "mean_context_chars": 1000.0}
    sweep = _sweep((8, 0.2, 0.1, 500.0), (20, 0.6, 0.5, 1200.0))
    lines = edgeless_budget_lines(sweep, graph)
    assert "already matches it" in lines[1]
    assert "not about information the graph reaches" in " ".join(lines)


def test_edgeless_budget_lines_state_the_residual_gap_when_no_k_reaches_it():
    """If even reading every entity is less context, say so — do not compare."""
    graph = {"fact_recall": 0.9, "full_coverage": 0.8, "mean_context_chars": 9_000.0}
    sweep = _sweep((8, 0.5, 0.1, 500.0), (79, 0.6, 0.2, 4000.0))
    lines = edgeless_budget_lines(sweep, graph)
    assert "never reaches C's context" in lines[0]
    assert "even reading ALL 79 entities" in lines[0]
    assert "residual context gap is stated rather than hidden" in lines[0]
    # No k in this sweep reaches C's scores, and the report says exactly that
    # instead of quietly comparing unequal budgets.
    assert "no k reaches C's 90.0%; the edgeless best is 60.0% at top-79" in lines[1]
    assert edgeless_budget_lines([], graph) == []


def test_scale_note_says_how_much_of_the_corpus_each_side_reads():
    """A budget-matched baseline on a small corpus is reading nearly all of it."""
    dataset = {"passages": [{"text": "x" * 100} for _ in range(10)]}
    chunked = {"fact_recall": 0.9, "full_coverage": 0.9, "mean_context_chars": 500.0}
    lines = scale_note(dataset, chunked, (8, {"mean_context_chars": 800.0}))
    assert "entire corpus is 1,000 characters" in lines[0]
    assert "mean context is 50% of it" in lines[0]
    assert "A′ reads 80% of the corpus (8 of 10 passages)" in lines[1]
    assert "not as a deployable configuration" in lines[1]
    assert len(scale_note(dataset, chunked, None)) == 1


# ── B′: the exact ablation, by surgery on C's own context ───────────────────
GRAPH_CONTEXT = (
    "Entity: Ada Lovelace (Type: PERSON)\n"
    "  Description: British mathematician\n"
    "  Relationships:\n"
    "  → WROTE_ALGORITHM_FOR → Analytical Engine (MACHINE)\n"
    "  → COLLABORATED_WITH → Charles Babbage (PERSON)\n\n"
    "Entity: IBM (Type: ORGANIZATION)\n"
    "  Description: Computing company\n\n"
    f"{PATHS_HEADING}\n"
    "  - Ada Lovelace -[COLLABORATED_WITH]-> Charles Babbage\n"
)


def test_strip_graph_structure_removes_edges_and_paths_and_nothing_else():
    """The claim "only the relationship lines are removed" was false, so the
    surgery is now explicit and asserted line by line."""
    stripped = strip_graph_structure(GRAPH_CONTEXT)
    assert stripped == (
        "Entity: Ada Lovelace (Type: PERSON)\n"
        "  Description: British mathematician\n\n"
        "Entity: IBM (Type: ORGANIZATION)\n"
        "  Description: Computing company"
    )
    assert "Relationships" not in stripped
    assert "→" not in stripped
    assert PATHS_HEADING not in stripped
    # What survives is byte-identical to the edgeless block format system B uses.
    assert stripped.split("\n\n")[1] == entity_context_block(
        {"name": "IBM", "type": "ORGANIZATION", "description": "Computing company"}
    )


def test_strip_graph_structure_also_drops_a_source_excerpt_section():
    """Total by construction: the ablation must never keep prose C read."""
    context = f"Entity: IBM (Type: ORG)\n\n{SOURCES_HEADING}\n\n[S1] doc (chunk 0)\nprose"
    assert strip_graph_structure(context) == "Entity: IBM (Type: ORG)"
    assert strip_graph_structure("") == ""


def test_strip_edges_scores_the_stripped_context():
    """B′ must be C's retrieval minus the graph — same seeds, same order."""
    graph = [
        QuestionResult(
            qid="q01",
            question="?",
            required=["Ada Lovelace", "Analytical Engine"],
            found=["Ada Lovelace", "Analytical Engine"],
            context_chars=len(GRAPH_CONTEXT),
            extras={"seeds": ["Ada Lovelace", "IBM"], "context": GRAPH_CONTEXT},
        )
    ]
    (result,) = strip_edges(graph)
    assert result.detail == "Ada Lovelace, IBM"  # seed order preserved
    assert result.found == ["Ada Lovelace"]  # the edge-reached fact is gone
    assert result.context_chars == len(strip_graph_structure(GRAPH_CONTEXT))
    assert result.context_chars < len(GRAPH_CONTEXT)


def test_strip_edges_tolerates_a_retrieval_with_no_context():
    graph = [QuestionResult("q01", "?", ["IBM"], ["IBM"], 500, extras={})]
    (result,) = strip_edges(graph)
    assert result.found == []
    assert result.context_chars == 0


def test_context_matched_picks_the_first_k_reaching_the_budget():
    sweep = _sweep((1, 0.3, 0.0, 100.0), (2, 0.6, 0.2, 200.0), (4, 0.9, 0.7, 400.0))
    assert context_matched(sweep, 150.0)[0] == 2
    assert context_matched(sweep, 200.0)[0] == 2  # exactly on budget counts
    assert context_matched(sweep, 401.0) is None


def test_context_matched_note_admits_when_the_baseline_catches_up():
    """The equal-context row must be reported honestly in both directions."""
    sweep = _sweep((1, 0.3, 0.0, 100.0), (10, 0.99, 0.95, 500.0))
    graph = {"fact_recall": 0.9, "full_coverage": 0.85, "mean_context_chars": 500.0}
    lines = context_matched_note(sweep, graph)
    assert "top-10" in lines[0]
    assert "matches or beats GraphRAG" in lines[1]


def test_context_matched_note_reports_a_structural_win():
    sweep = _sweep((1, 0.3, 0.0, 100.0), (10, 0.5, 0.4, 500.0))
    graph = {"fact_recall": 0.9, "full_coverage": 0.85, "mean_context_chars": 500.0}
    assert "structural" in context_matched_note(sweep, graph)[1]


def test_context_matched_note_when_the_baseline_never_reaches_the_budget():
    sweep = _sweep((1, 0.3, 0.0, 100.0))
    graph = {"fact_recall": 0.9, "full_coverage": 0.85, "mean_context_chars": 9_000.0}
    lines = context_matched_note(sweep, graph)
    assert len(lines) == 1
    assert "never reaches" in lines[0]


# ── Baseline evaluation is deterministic and honest ──────────────────────────
def test_evaluate_baseline_scores_the_passages_it_retrieved():
    """A ranking that hands the baseline the right passages must score 100%."""
    question = QUESTIONS[0]
    ideal = sorted(
        range(len(PASSAGES)),
        key=lambda i: -len(_mentions(PASSAGES[i]) & set(question["required_facts"])),
    )
    rankings = {q["id"]: list(range(len(PASSAGES))) for q in QUESTIONS}
    rankings[question["id"]] = ideal

    results = {r.qid: r for r in evaluate_baseline(DATASET, rankings, k=4)}
    assert results[question["id"]].recall == pytest.approx(1.0)
    assert results[question["id"]].context_chars > 0


def test_evaluate_baseline_respects_k():
    rankings = {q["id"]: list(range(len(PASSAGES))) for q in QUESTIONS}
    small = evaluate_baseline(DATASET, rankings, k=1)
    large = evaluate_baseline(DATASET, rankings, k=6)
    assert all(s.context_chars < ll.context_chars for s, ll in zip(small, large, strict=True))
    assert all(len(r.detail.split(", ")) == 1 for r in small)


# ── The leak detector that enforces R1 ───────────────────────────────────────
def test_content_words_drops_function_words_and_short_tokens():
    assert content_words("Who was the man that helped break Enigma?") == {
        "man",
        "help",
        "break",
        "enigma",
    }
    assert content_words("") == set()
    assert content_words(None) == set()


def test_content_words_conflates_the_inflections_that_matter():
    """A leak must not be hidden behind an -s or an -ed."""
    for a, b in (
        ("walks", "walk"),
        ("designed", "design"),
        ("popularised", "popularise"),
        ("meeting", "meet"),
        ("trains", "trained"),
        ("networks", "network"),
        ("introduces", "introducing"),
    ):
        assert content_words(a) == content_words(b), (a, b)
    # …but unrelated words are not conflated into a false positive.
    assert content_words("machine") != content_words("mechanism")
    assert content_words("class") == {"class"}  # -ss is not a plural


def test_same_family_catches_the_prefix_overlap_stemming_cannot():
    """The leak class R1 advertises catching, which stem equality misses.

    Suffix stripping only ever collides words that differ at the *end*. The
    leaks actually found in this dataset differed in the middle, and a
    subword-tokenising embedder sees them perfectly well.
    """
    assert same_family("dataset", "databas")
    assert same_family("back", "backpropag")
    assert same_family("statistical", "stat")
    assert same_family("train", "train")
    # Shorter shared prefixes are not a family — otherwise the rule flags noise.
    assert not same_family("machin", "mechanis")
    assert not same_family("network", "neural")
    assert not same_family("gpu", "gradient")
    assert PREFIX_FAMILY_LEN == 4


def test_leaks_returns_the_question_side_words_to_rewrite():
    assert leaks("Where was the dataset assembled?", "directed an image database") == {
        "dataset"
    }
    assert leaks("What did he popularise in 1986?", "Gradient method") == set()
    assert leaks("", "anything") == set()


# ── Sweeps run to the whole corpus, so a budget match always exists ──────────
def test_sweep_ks_always_ends_at_the_whole_corpus():
    """The audit's finding: a sweep that stops early cannot budget-match.

    The old sweep stopped at k=12, where the entity system read 66% of
    GraphRAG's context — and the report compared them anyway.
    """
    assert sweep_ks(34)[-1] == 34
    assert sweep_ks(79)[-1] == 79
    assert sweep_ks(3) == (1, 2, 3)
    assert sweep_ks(1) == (1,)
    for total in (5, 34, 79):
        ks = sweep_ks(total)
        assert ks == tuple(sorted(set(ks))), ks  # strictly increasing, no repeats
        assert all(1 <= k <= total for k in ks)
    assert max(SWEEP_KS) >= 64  # deep enough to reach a big graph's ceiling


def test_a_sweep_that_reaches_the_whole_corpus_can_always_be_budget_matched():
    """Property that makes the honesty machinery total rather than best-effort."""
    sweep = _sweep((1, 0.3, 0.0, 100.0), (34, 1.0, 1.0, 7000.0))
    for target in (50.0, 800.0, 6999.0, 7000.0):
        assert context_matched(sweep, target) is not None
    assert smallest_k_reaching(sweep, 1.0, "full_coverage") == sweep[-1]


# ── System B: entity vector RAG, the graph ablation ──────────────────────────
def test_entity_document_matches_what_graph_builder_embeds():
    """Same string ``graph_builder._embed_entities`` builds, or the ablation lies."""
    from app.services.graph_builder import _embed_entities  # noqa: F401  (import check)

    entity = {"name": "Neo4j", "type": "DATABASE", "description": "Native graph store"}
    assert entity_document(entity) == "Neo4j — Native graph store"
    assert entity_document({"name": "Neo4j", "description": ""}) == "Neo4j"


def test_entity_context_block_is_the_graphrag_block_minus_the_edges():
    entity = {"name": "Neo4j", "type": "DATABASE", "description": "Native graph store"}
    assert entity_context_block(entity) == (
        "Entity: Neo4j (Type: DATABASE)\n  Description: Native graph store"
    )
    assert "→" not in entity_context_block(entity)  # no relationship lines
    assert entity_context_block({"name": "X", "type": "T"}) == "Entity: X (Type: T)"


def test_evaluate_entity_rag_scores_only_the_entities_it_retrieved():
    rankings = {q["id"]: list(range(len(ENTITIES))) for q in QUESTIONS}
    top1 = {r.qid: r for r in evaluate_entity_rag(DATASET, rankings, k=1)}
    first = ENTITIES[0]["name"]
    for question in QUESTIONS:
        expected = [f for f in question["required_facts"] if f == first]
        assert top1[question["id"]].found == expected, question["id"]
        assert top1[question["id"]].detail == first


def test_evaluate_entity_rag_respects_k_and_cannot_beat_a_perfect_ranking():
    """With every required fact ranked first, the ceiling is 100% — and only then."""
    question = QUESTIONS[0]
    required = list(question["required_facts"])
    ideal = sorted(
        range(len(ENTITIES)),
        key=lambda i: (ENTITIES[i]["name"] not in required, i),
    )
    rankings = {q["id"]: list(range(len(ENTITIES))) for q in QUESTIONS}
    rankings[question["id"]] = ideal

    perfect = {r.qid: r for r in evaluate_entity_rag(DATASET, rankings, k=len(required))}
    assert perfect[question["id"]].recall == pytest.approx(1.0)

    starved = {r.qid: r for r in evaluate_entity_rag(DATASET, rankings, k=len(required) - 1)}
    assert starved[question["id"]].recall < 1.0
    assert starved[question["id"]].context_chars < perfect[question["id"]].context_chars


def test_dataset_path_points_at_the_shipped_file():
    assert DATASET_PATH.name == "dataset.json"
    assert DATASET_PATH.exists()


# ── System D: source chunks, and the honesty machinery around them ──────────
def test_the_section_markers_match_the_engine_the_ablation_cuts_up():
    """B′ is text surgery on ``chat_engine``'s output, so the markers must match.

    They are duplicated in the harness (importing LangChain into a scoring
    module would be silly) — this test is what keeps the duplicate honest.
    """
    from app.services import chat_engine

    assert PATHS_HEADING == chat_engine.PATHS_HEADING
    assert SOURCES_HEADING == chat_engine.SOURCES_HEADING


def test_the_harness_clears_chunks_before_measuring_the_graph_only_row():
    """A leftover demo :Chunk would be retrieved by the production path.

    Row C claims to be "GraphRAG with no source chunks". That is only true if
    the store was empty, so the harness deletes them and then verifies it.
    """
    assert any("(c:Chunk)" in q and "DELETE" in q for q in CLEAR_QUERIES)


def test_contaminated_flags_a_graph_only_run_that_returned_excerpts():
    clean = [QuestionResult("q01", "?", ["IBM"], [], 10, extras={"context": "Entity: IBM"})]
    dirty = [
        QuestionResult("q01", "?", ["IBM"], [], 10, extras={"context": "Entity: IBM"}),
        QuestionResult(
            "q02", "?", ["IBM"], [], 10, extras={"context": f"Entity: IBM\n\n{SOURCES_HEADING}\n"}
        ),
    ]
    assert contaminated(clean) == []
    assert contaminated(dirty) == ["q02"]
    assert contaminated([QuestionResult("q03", "?", [], [], 0)]) == []


def test_chunk_note_states_the_gain_and_the_tautology_together():
    """D's semantic channel *is* system A; the report has to say so itself."""
    graph_only = {
        "fact_recall": 0.875,
        "full_coverage": 0.643,
        "strict_fact_recall": 0.0,
        "strict_full_coverage": 0.0,
        "mean_context_chars": 2081.0,
    }
    chunked = {
        "fact_recall": 0.95,
        "full_coverage": 0.85,
        "strict_fact_recall": 0.9,
        "strict_full_coverage": 0.8,
        "mean_context_chars": 4162.0,
    }
    overlap = {
        "baseline_k": 4,
        "containment": 0.75,
        "mean_excerpts": 6.0,
        "mean_beyond": 3.0,
    }
    lines = chunk_note(graph_only, chunked, overlap)
    assert "87.5% graph-only → 95.0% with source chunks (+7.5 pp)" in lines[1]
    assert "64.3% graph-only → 85.0% with source chunks (+20.7 pp)" in lines[2]
    assert "2.00x the context" in lines[3]
    # Under the strict rule the graph-only side scores nothing, so the whole of
    # the shipped system's strict score is attributable to the chunk channel.
    assert "C scores 0.0% / 0.0%" in lines[4]
    assert "D scores 90.0% / 80.0%" in lines[4]
    assert "75% of the passages system A retrieves at top-4" in lines[5]
    assert "D's semantic excerpt channel IS the passage baseline" in lines[5]


def test_excerpt_overlap_measures_how_much_of_d_is_just_the_baseline():
    baseline = [
        QuestionResult("q01", "?", [], [], 0, extras={"passages": [0, 1, 2, 3]}),
        QuestionResult("q02", "?", [], [], 0, extras={"passages": [4, 5, 6, 7]}),
    ]
    chunked = [
        # Two of A's four passages come back, plus one A never saw.
        QuestionResult("q01", "?", [], [], 0, extras={"chunks": [0, 1, 9]}),
        # Nothing A retrieved.
        QuestionResult("q02", "?", [], [], 0, extras={"chunks": [20]}),
    ]
    stats = excerpt_overlap(baseline, chunked)
    assert stats["baseline_k"] == 4
    assert stats["containment"] == pytest.approx(0.25)  # (2/4 + 0/4) / 2
    assert stats["mean_excerpts"] == pytest.approx(2.0)  # (3 + 1) / 2
    assert stats["mean_beyond"] == pytest.approx(1.0)  # {9} and {20}


def test_excerpt_overlap_handles_a_run_with_no_excerpts_at_all():
    baseline = [QuestionResult("q01", "?", [], [], 0, extras={"passages": [0, 1]})]
    chunked = [QuestionResult("q01", "?", [], [], 0, extras={})]
    stats = excerpt_overlap(baseline, chunked)
    assert stats["containment"] == 0.0
    assert stats["mean_excerpts"] == 0.0


def test_misses_note_is_generated_from_the_results_not_typed():
    """The README asserted three failing questions while the table showed five.

    The sentence is now derived from the same data as the table, so the two
    cannot disagree again.
    """
    results = [
        QuestionResult("q01", "?", ["A", "B"], ["A", "B"], 10),
        QuestionResult("q02", "?", ["A", "B"], ["A"], 10),
        QuestionResult("q03", "?", ["C"], [], 10),
    ]
    (line,) = misses_note("D", results)
    assert "D missed at least one required fact on 2 of 3 questions (q02, q03)" in line
    assert "The facts it never retrieved: B, C." in line

    (perfect,) = misses_note("D", [QuestionResult("q01", "?", ["A"], ["A"], 10)])
    assert perfect == "D retrieved every required fact on all 1 questions."


def test_misses_note_reports_the_strict_rule_separately():
    """A fact credited only to scaffolding is a miss under the strict rule."""
    results = [
        QuestionResult("q01", "?", ["A"], ["A"], 10, hit_channels={"A": PROSE}),
        QuestionResult("q02", "?", ["A"], ["A"], 10, hit_channels={"A": EDGE}),
    ]
    (permissive,) = misses_note("D", results)
    assert "retrieved every required fact" in permissive
    (strict,) = misses_note("D", results, strict=True)
    assert "missed at least one required fact on 1 of 2 questions (q02)" in strict
    assert "The facts it never retrieved: A." in strict


# ── Channel attribution: the permissive rule is uniform, not neutral ─────────
# A GraphRAG context prints entity names as scaffolding, so a required fact can
# be credited because the graph listed it as a neighbour rather than because the
# evidence to answer came back. These tests pin the machinery that measures it.
GRAPH_CONTEXT_WITH_SOURCES = (
    "Entity: Ada Lovelace (Type: PERSON)\n"
    "  Description: British mathematician\n"
    "  Relationships:\n"
    "  → WROTE_ALGORITHM_FOR → Analytical Engine (MACHINE)\n\n"
    "Entity: IBM (Type: ORGANIZATION)\n"
    "  Description: Computing company\n\n"
    f"{PATHS_HEADING}\n"
    "  - IBM -[FOUNDED_BY]-> Herman Hollerith\n\n"
    f"{SOURCES_HEADING}\n\n"
    "[S1] benchmark (chunk 7)\n"
    "Punched Cards fed the machine, and IBM sold them.\n\n"
    "(1 further excerpt omitted for length.)"
)


def test_prose_segment_is_corpus_text_and_nothing_else():
    """Provenance headers and the truncation note are the engine's words, not the corpus."""
    prose = prose_segment(GRAPH_CONTEXT_WITH_SOURCES)
    assert "Punched Cards fed the machine" in prose
    assert "[S1]" not in prose
    assert "chunk 7" not in prose
    assert "further excerpt omitted" not in prose
    assert "Ada Lovelace" not in prose  # the graph half is not prose
    # A context with no excerpt section has no prose at all — which is why the
    # graph-only and entity-granular rows score zero under the strict rule.
    assert prose_segment("Entity: IBM (Type: ORG)") == ""
    assert prose_segment("") == ""


def test_edge_segment_is_exactly_what_strip_graph_structure_removes():
    """The two functions must partition the graph half, or attribution leaks."""
    edges = edge_segment(GRAPH_CONTEXT_WITH_SOURCES)
    assert "→ WROTE_ALGORITHM_FOR → Analytical Engine (MACHINE)" in edges
    assert PATHS_HEADING in edges
    assert "Herman Hollerith" in edges
    # What survives the strip must be absent here, and vice versa.
    assert "Description: British mathematician" not in edges
    assert "Entity: IBM" not in edges
    assert "→" not in strip_graph_structure(GRAPH_CONTEXT_WITH_SOURCES)
    assert edge_segment("Entity: IBM (Type: ORG)") == ""


def test_the_three_channels_cover_every_name_the_context_contains():
    """A name in none of the channels would be dropped from the strict column.

    That is the quiet distortion this whole exercise exists to prevent, so the
    covering property is asserted rather than assumed — and the harness raises
    :class:`IntegrityError` at run time if a real context ever violates it.
    """
    channels = Channels.of_graph(GRAPH_CONTEXT_WITH_SOURCES)
    joined = "\n".join([channels.prose, channels.entities, channels.edges]).lower()
    for name in (
        "Ada Lovelace",
        "IBM",
        "Analytical Engine",
        "Herman Hollerith",
        "Punched Cards",
        "Neo4j",  # not present anywhere: must be absent from both sides
    ):
        assert (name.lower() in GRAPH_CONTEXT_WITH_SOURCES.lower()) == (
            name.lower() in joined
        ), name


def test_classify_hits_attributes_each_fact_to_its_channel():
    channels = Channels.of_graph(GRAPH_CONTEXT_WITH_SOURCES)
    attribution = classify_hits(
        channels,
        ["Punched Cards", "Ada Lovelace", "Analytical Engine", "Herman Hollerith", "Neo4j"],
    )
    assert attribution == {
        "Punched Cards": PROSE,  # verbatim corpus text
        "Ada Lovelace": ENTITY,  # a materialised entity block
        "Analytical Engine": EDGE,  # named ONLY by a relationship line
        "Herman Hollerith": EDGE,  # named ONLY by a reasoning path
    }
    assert "Neo4j" not in attribution


def test_classify_hits_credits_the_best_evidence_available():
    """Attribution is conservative *against* the criticism of the permissive rule.

    IBM occurs in an entity block, in a reasoning path AND in the prose. It is
    credited to the prose, so the measured "share of hits that are scaffolding
    only" is a lower bound on the advantage, never an inflated one.
    """
    channels = Channels.of_graph(GRAPH_CONTEXT_WITH_SOURCES)
    assert classify_hits(channels, ["IBM"]) == {"IBM": PROSE}
    # With the excerpt gone, the same name falls back to the entity block.
    graph_only = GRAPH_CONTEXT_WITH_SOURCES.partition(SOURCES_HEADING)[0]
    assert classify_hits(Channels.of_graph(graph_only), ["IBM"]) == {"IBM": ENTITY}


def test_the_channel_constructors_describe_the_systems_that_use_them():
    passage = Channels.of_passages("Punched Cards fed the machine.")
    assert classify_hits(passage, ["Punched Cards"]) == {"Punched Cards": PROSE}
    assert passage.entities == "" and passage.edges == ""

    entity = Channels.of_entities("Entity: IBM (Type: ORG)")
    assert classify_hits(entity, ["IBM"]) == {"IBM": ENTITY}
    assert entity.prose == ""  # nothing can ever be credited to prose here


def test_hit_provenance_counts_and_shares():
    results = [
        QuestionResult("q1", "?", [], [], 0, hit_channels={"A": PROSE, "B": EDGE}),
        QuestionResult("q2", "?", [], [], 0, hit_channels={"C": ENTITY, "D": EDGE}),
    ]
    stats = hit_provenance(results)
    assert stats["total"] == 4
    assert stats["counts"] == {PROSE: 1, ENTITY: 1, EDGE: 2}
    assert stats["shares"][EDGE] == pytest.approx(0.5)
    assert set(stats["shares"]) == set(CHANNELS)

    empty = hit_provenance([])
    assert empty["total"] == 0
    assert all(share == 0.0 for share in empty["shares"].values())


def test_provenance_note_sizes_the_scaffolding_advantage_per_system():
    rows = [
        ("A. passage", [QuestionResult("q1", "?", [], [], 0, hit_channels={"X": PROSE})]),
        ("C. graph", [QuestionResult("q1", "?", [], [], 0, hit_channels={"X": EDGE})]),
        ("Z. nothing", [QuestionResult("q1", "?", [], [], 0)]),
    ]
    lines = provenance_note(rows)
    assert "conservative against the graph" in lines[0]
    assert "100% of its 1 scored hits came from retrieved prose" in lines[1]
    assert "0% of its 1 scored hits came from retrieved prose, 100% from graph" in lines[2]
    assert "100% an edge or reasoning path and nothing else" in lines[2]
    assert "no scored hits" in lines[3]


def test_scoring_rule_note_says_which_rule_favours_which_system():
    rows = [
        (
            "A. passage",
            {
                "fact_recall": 0.9,
                "full_coverage": 0.7,
                "strict_fact_recall": 0.9,
                "strict_full_coverage": 0.7,
            },
        ),
        (
            "C. graph",
            {
                "fact_recall": 0.9,
                "full_coverage": 0.7,
                "strict_fact_recall": 0.0,
                "strict_full_coverage": 0.0,
            },
        ),
    ]
    lines = scoring_rule_note(rows)
    assert "Fact Recall −0.0 pp" in lines[1]
    assert "unaffected — it returns nothing but prose" in lines[1]
    assert "Fact Recall −90.0 pp, Full-Coverage Rate −70.0 pp" in lines[2]
    assert "favoured by the permissive rule" in lines[2]
    assert "upper bound" in lines[-1] and "lower bound" in lines[-1]


def test_headline_runs_the_same_scrutiny_under_the_strict_rule():
    """The strict headline is recomputed, not reworded — including its parity k."""
    graph = {
        "fact_recall": 0.95,
        "full_coverage": 0.9,
        "strict_fact_recall": 0.6,
        "strict_full_coverage": 0.5,
        "mean_context_chars": 1000.0,
    }
    sweep = [
        (4, {"fact_recall": 0.7, "full_coverage": 0.6,
             "strict_fact_recall": 0.7, "strict_full_coverage": 0.6,
             "mean_context_chars": 200.0}),
        (20, {"fact_recall": 0.95, "full_coverage": 0.9,
              "strict_fact_recall": 0.95, "strict_full_coverage": 0.9,
              "mean_context_chars": 5000.0}),
    ]
    permissive = headline(graph, sweep, METRICS)
    assert "survives a budget match" in permissive[1]
    strict = headline(graph, sweep, STRICT_METRICS, "strict rule")
    assert "strict rule" in strict[0]
    assert "Fact Recall 60.0%" in strict[0]
    # Under the strict rule the cheap baseline already beats it, and the
    # generated sentence says so rather than reusing the permissive verdict.
    assert "EVERY metric while reading LESS text" in strict[1]
    assert "top-4 using 20% of the context" in strict[1]


def test_verdict_and_context_matched_note_follow_the_rule_they_are_given():
    baseline = {
        "fact_recall": 0.9, "full_coverage": 0.7,
        "strict_fact_recall": 0.9, "strict_full_coverage": 0.7,
        "mean_context_chars": 100.0,
    }
    graph = {
        "fact_recall": 0.95, "full_coverage": 0.9,
        "strict_fact_recall": 0.5, "strict_full_coverage": 0.4,
        "mean_context_chars": 400.0,
    }
    sweep = [
        (4, dict(baseline)),
        (20, {"fact_recall": 1.0, "full_coverage": 1.0,
              "strict_fact_recall": 1.0, "strict_full_coverage": 1.0,
              "mean_context_chars": 500.0}),
    ]
    permissive = verdict(baseline, graph, sweep)
    assert "GraphRAG ahead at the configured k" in permissive[0]
    strict = verdict(baseline, graph, sweep, STRICT_METRICS)
    assert "passage vector RAG ahead at the configured k" in strict[0]
    assert "90.0% vs 50.0%" in strict[0]
    # The equal-context row must quote the strict scores when asked for them.
    assert "100.0% recall / 100.0% coverage" in context_matched_note(
        sweep, graph, STRICT_METRICS
    )[0]


def test_the_two_rules_agree_exactly_on_the_passage_baseline():
    """Half of the asymmetry, on the real dataset: prose systems lose nothing."""
    rankings = {q["id"]: list(range(len(PASSAGES))) for q in QUESTIONS}
    for k in (1, 4, len(PASSAGES)):
        agg = aggregate(evaluate_baseline(DATASET, rankings, k))
        assert agg["strict_fact_recall"] == pytest.approx(agg["fact_recall"]), k
        assert agg["strict_full_coverage"] == pytest.approx(agg["full_coverage"]), k


def test_the_strict_rule_is_zero_for_the_prose_free_systems():
    """The other half: entity-granular retrieval scores nothing under the strict rule.

    Not a failure — a property. These systems return a compressed representation
    of the corpus and never the corpus, so every point they score under the
    permissive rule is a point the permissive rule gave them. Even reading
    *every entity in the graph*, which is a permissive 100%, buys zero.
    """
    rankings = {q["id"]: list(range(len(ENTITIES))) for q in QUESTIONS}
    everything = aggregate(evaluate_entity_rag(DATASET, rankings, k=len(ENTITIES)))
    assert everything["fact_recall"] == pytest.approx(1.0)
    assert everything["full_coverage"] == pytest.approx(1.0)
    assert everything["strict_fact_recall"] == 0.0
    assert everything["strict_full_coverage"] == 0.0

    # Same for B′, whose context is C's entity blocks with the edges deleted.
    (stripped,) = strip_edges(
        [
            QuestionResult(
                "q01", "?", ["Ada Lovelace"], [], 0,
                extras={"seeds": ["Ada Lovelace"], "context": GRAPH_CONTEXT},
            )
        ]
    )
    assert stripped.found == ["Ada Lovelace"]
    assert stripped.found_strict == []


# ── R1's asymmetry, disclosed rather than left implicit ─────────────────────
def test_r1_is_not_enforced_against_passage_text_and_that_asymmetry_is_real():
    """R1 constrains the graph's text only. The README must not pretend otherwise.

    Questions may not share a content-word family with the *name* or
    *description* of a non-anchor required fact — those are what GraphRAG embeds
    and full-text-indexes. Nothing forbids a question from sharing words with the
    **passages** that carry its facts, which is what the passage baseline embeds.

    The asymmetry therefore cuts *against* GraphRAG, and it is real in this
    dataset rather than hypothetical: this test finds the actual overlaps. If a
    future rewrite removed them, the README's disclosure would become false — so
    this fails, and the disclosure has to be revisited rather than silently rot.
    """
    overlaps = [
        (question["id"], fact, passage["id"])
        for question in QUESTIONS
        for fact in question["required_facts"]
        if fact != question["anchor"]
        for passage in PASSAGES
        if fact in passage["entities"] and leaks(question["question"], passage["text"])
    ]
    assert overlaps, (
        "no question shares a content-word family with a passage carrying one of its "
        "non-anchor required facts — the README documents this asymmetry as real"
    )


# ── The README's numbers are generated, and cannot rot ──────────────────────
#: Any number, with optional thousands separators, decimals and a percent sign.
#: The lookbehind keeps identifiers ("R1", "Neo4j", "bge-small-en-v1.5") out.
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def test_the_readme_generated_region_is_verbatim_in_the_results():
    """One function writes the numeric core into both files, so it must match.

    An audit found README.md quoting a corpus character count and several
    figures that a fresh run did not reproduce, under a sentence promising every
    number was "copied from the generated report and reproduced by re-running
    the script". Copying was the defect. There is now nothing to copy, and this
    test is what keeps it that way.
    """
    readme = README_PATH.read_text(encoding="utf-8")
    results = RESULTS_PATH.read_text(encoding="utf-8")

    assert README_BEGIN in readme and README_END in readme, (
        "README.md has lost its generated region; run `make benchmark`"
    )
    block = readme.split(README_BEGIN)[1].split(README_END)[0].strip()
    assert block, "the generated region is empty; run `make benchmark`"
    assert "| System |" in block, "the generated region has lost the results table"
    assert block in results, (
        "README.md's generated region is not present verbatim in results.md — the two "
        "were written by different runs. Re-run `make benchmark`."
    )


def test_every_number_in_the_readme_appears_in_the_generated_results():
    """No figure may exist in README.md that the generated report does not carry.

    Containment is necessary, not sufficient: it catches a stale or invented
    figure, not a correct figure quoted about the wrong thing. That is the
    weaker claim, and it is the one this test actually supports.
    """
    readme = README_PATH.read_text(encoding="utf-8")
    results = RESULTS_PATH.read_text(encoding="utf-8")

    missing = sorted(
        {
            token
            for token in _NUMBER.findall(readme)
            # Matched as a whole token in results.md too, so "4" is not
            # satisfied by the "4" inside "48".
            if not re.search(rf"(?<![\w.]){re.escape(token)}(?![\d,.%])", results)
        }
    )
    assert not missing, (
        f"README.md quotes numbers that results.md does not contain: {missing}. "
        "Numbers belong in the generated report; re-run `make benchmark` or delete them."
    )


def test_render_readme_replaces_only_the_marked_region():
    class _Report:
        settings_line = ""

        def rows(self):
            return [
                (
                    "D. system",
                    {
                        "fact_recall": 1.0,
                        "full_coverage": 1.0,
                        "strict_fact_recall": 0.5,
                        "strict_full_coverage": 0.5,
                        "mean_context_chars": 10.0,
                    },
                )
            ]

        def headlines(self):
            return ["HEADLINE: computed"]

    original = f"before\n\n{README_BEGIN}\n\nstale numbers\n\n{README_END}\n\nafter\n"
    rendered = render_readme(original, _Report())
    assert rendered.startswith("before\n\n")
    assert rendered.endswith(f"{README_END}\n\nafter\n")
    assert "stale numbers" not in rendered
    # The shipped row is emphasised; the context column never is.
    assert (
        "| **D. system** | **100.0%** | **100.0%** | **50.0%** | **50.0%** | 10 |"
    ) in rendered
    assert "- HEADLINE: computed" in rendered


def test_render_readme_refuses_a_readme_with_no_generated_region():
    """Appending or guessing would hide exactly the state the rot test catches."""
    with pytest.raises(IntegrityError):
        render_readme("no markers here", object())
