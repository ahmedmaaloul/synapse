# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Calibrate ``PROCEDURAL_SEMANTIC_THRESHOLD`` for the configured embedder.

The semantic step of the procedural localization cascade (Synapse's addition
to Procedural Graphs, arXiv:2609.09153) places an agent on a node when the
cosine between its last action + observation and a node's ``"<id>
<description>"`` reaches the threshold. Below it, the full graph is used,
which is the paper's behaviour. The right floor depends on the embedder:
fastembed's ``BAAI/bge-small-en-v1.5`` puts even unrelated text at a high
cosine, so a floor tuned for another model can make garbage steps localize.

Why a script with two FIXED string sets: a calibration someone cannot rerun is
a claim, not a measurement. ``UNRELATED`` holds steps that belong to no node
of the navigator prior (shell commands, SQL, errands, text-game moves, filler
words). ``PARAPHRASES`` holds tool calls written the way another model or MCP
host might name the navigator's six tools, each with the node it means. Every
string misses the name-based steps (``exact`` / ``normalized``), so each one
exercises the semantic step alone; ``check_sets`` enforces that. The
integration test ``tests/integration/test_semantic_localization.py`` imports
the same sets, so the docs, this script and the test cannot drift apart.

It scores each string exactly as ``procedural_guidance`` does (same candidate
set: every node but ``Start`` and the terminals; same text; same cosine),
against the BUNDLED ``graphrag-navigator`` prior, then prints the score
distributions and the outcome at each threshold from 0.50 to 0.85.

No LLM and no database. With the default ``fastembed`` it runs locally and
costs nothing (the model is downloaded once if absent). With a cloud
``EMBEDDING_PROVIDER`` it makes one small embedding call per string, plus one
batch for the node descriptions.

Usage (from ``backend/``, or inside the backend container):

    python -m scripts.calibrate_semantic_threshold
    python -m scripts.calibrate_semantic_threshold --details   # every string's best node
    docker compose exec -T backend python -m scripts.calibrate_semantic_threshold

Exit codes: 0 success · 1 a string set is invalid for the prior · 2 the
embedder could not be loaded.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.services.llm_provider import get_embeddings
from app.services.procedural_graph import ProceduralGraph, localize
from app.services.procedural_guidance import (
    SEMANTIC_OBSERVATION_CHARS,
    _cosine,
    _node_text,
    _semantic_candidates,
)
from app.services.procedural_store import load_prior

PRIOR = "graphrag-navigator"

#: Thresholds reported, 0.50 to 0.85 by 0.05. The configured value is added.
THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(8))

#: (action, observation): steps that belong to no node of the navigator prior.
#: The right outcome is always ``none``, i.e. the full graph.
UNRELATED: tuple[tuple[str, str], ...] = (
    ("x", ""),
    ("weather_forecast", "It will rain in Paris tomorrow afternoon."),
    ("bake_cake", "Preheat the oven to 180 degrees and whisk three eggs."),
    ("ls -la", "total 0"),
    ("zzqx", "qwerty 12345"),
    ("!!!", "..."),
    ("hello world", ""),
    ("SELECT * FROM users", ""),
    ("lookup", ""),
    ("click_button", "Submit clicked"),
    ("done", ""),
    ("git push origin main", "Everything up-to-date"),
    ("send_email", "Message sent to 3 recipients."),
    ("book_flight", "Flight AF123 confirmed for 14 March."),
    ("play_music", "Now playing: track 4"),
    ("set_alarm", "Alarm set for 7:00 AM"),
    ("resize_image", "Saved thumbnail.png (128x128)"),
    ("npm install", "added 312 packages in 9s"),
    ("translate_text", "Bonjour le monde"),
    ("calculator", "2 + 2 = 4"),
    ("scroll_down", ""),
    ("open_browser", "about:blank"),
    ("create_calendar_event", "Team sync on Friday at 10:00"),
    ("water_plants", "The soil is moist."),
    ("stock_price", "AAPL 187.32 USD"),
    ("turn_on_lights", "Living room lights on"),
    ("order_pizza", "Your pizza will arrive in 30 minutes."),
    ("compile", "0 errors, 2 warnings"),
    ("docker ps", "CONTAINER ID   IMAGE   STATUS"),
    ("pip freeze", "numpy==2.1.0"),
    ("rm -rf /tmp/cache", ""),
    ("move_forward", "You walk north into a dark corridor."),
    ("pick_up_key", "You take the brass key."),
    ("sleep", ""),
    ("ok", ""),
    ("retry", ""),
    ("continue", ""),
    ("print('hi')", "hi"),
    ("{}", "null"),
    ("12345", ""),
    ("lorem ipsum dolor sit amet", ""),
    ("tweet", "Posted to your timeline."),
    ("upload_file", "report.pdf uploaded"),
    ("run_tests", "12 passed"),
    ("take_screenshot", "screenshot saved"),
    ("check_inventory", "3 widgets in stock"),
    ("convert_currency", "100 EUR = 108.40 USD"),
    ("pay_invoice", "Invoice #42 paid"),
)

#: (action, observation, intended node): the navigator's six tools under the
#: names another model or MCP host might give them. None equals a tool name
#: after normalization, so only the semantic step can place them.
PARAPHRASES: tuple[tuple[str, str, str], ...] = (
    ("entity_search", "Ada Lovelace", "search_entities"),
    ("search entities in the knowledge graph", "", "search_entities"),
    ("find_entity", "", "search_entities"),
    ("lookup_entity", "Charles Babbage", "search_entities"),
    ("search_graph_nodes", "", "search_entities"),
    ("kg_entity_search", "Analytical Engine", "search_entities"),
    ("query_entities", "", "search_entities"),
    ("find entities by name", "", "search_entities"),
    ("get_neighbors", "", "neighbors"),
    ("list_relations", "", "neighbors"),
    ("entity_relations", "Ada Lovelace", "neighbors"),
    ("get_relationships", "", "neighbors"),
    ("expand_node", "", "neighbors"),
    ("one_hop_neighbours", "", "neighbors"),
    ("related_entities", "", "neighbors"),
    ("list the direct relations of an entity", "", "neighbors"),
    ("read_source_passages", "", "read_sources"),
    ("get_sources", "", "read_sources"),
    ("fetch_source_text", "", "read_sources"),
    ("read_entity_sources", "Ada Lovelace", "read_sources"),
    ("source_excerpts", "", "read_sources"),
    ("get_evidence_passages", "", "read_sources"),
    ("show_provenance", "", "read_sources"),
    ("read the passages an entity was extracted from", "", "read_sources"),
    ("passage_search", "", "search_passages"),
    ("semantic_search", "", "search_passages"),
    ("search_chunks", "", "search_passages"),
    ("full_text_search", "", "search_passages"),
    ("search_documents", "", "search_passages"),
    ("find_passages", "", "search_passages"),
    ("vector_search", "", "search_passages"),
    ("search the source passages for a phrase", "", "search_passages"),
    ("shortest_path", "", "find_path"),
    ("find_connection", "", "find_path"),
    ("path_between_entities", "", "find_path"),
    ("relation_chain", "", "find_path"),
    ("connect_entities", "", "find_path"),
    ("trace_path", "", "find_path"),
    ("find_route", "", "find_path"),
    ("find the chain of relations between two entities", "", "find_path"),
    ("give_answer", "yes", "answer"),
    ("final_answer", "", "answer"),
    ("submit_answer", "Charles Babbage", "answer"),
    ("respond", "", "answer"),
    ("reply_to_user", "", "answer"),
    ("submit", "1843", "answer"),
    ("answer_question", "", "answer"),
    ("submit the final short answer", "", "answer"),
)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One string's best semantic candidate, as ``procedural_guidance`` computes it."""

    action: str
    observation: str
    intended: str | None  # None for an UNRELATED string
    node: str
    score: float

    @property
    def right_node(self) -> bool:
        return self.intended is not None and self.node == self.intended


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the semantic step does with both sets at one threshold."""

    threshold: float
    unrelated_full: int  # unrelated → full graph (right)
    right: int  # paraphrase → its intended node (right)
    wrong: int  # paraphrase → another node (the harmful case)
    missed: int  # paraphrase → full graph (safe: the paper's behaviour)
    total: int

    @property
    def accuracy(self) -> float:
        return (self.unrelated_full + self.right) / self.total if self.total else 0.0


def step_text(action: str, observation: str) -> str:
    """The text the semantic step embeds for a step (``procedural_guidance._semantic_match``)."""
    return f"{action} {observation[:SEMANTIC_OBSERVATION_CHARS]}".strip()


def check_sets(
    graph: ProceduralGraph,
    unrelated: Sequence[tuple[str, str]] = UNRELATED,
    paraphrases: Sequence[tuple[str, str, str]] = PARAPHRASES,
) -> list[str]:
    """Problems that would make a string measure something other than the semantic step."""
    problems: list[str] = []
    candidates = set(_semantic_candidates(graph))
    actions = [a for a, _ in unrelated] + [a for a, _, _ in paraphrases]
    seen: set[str] = set()
    for action in actions:
        if action in seen:
            problems.append(f"duplicate action {action!r}")
        seen.add(action)
        found = localize(graph, action)
        if found.node_id is not None:
            problems.append(
                f"{action!r} already localizes by name ({found.method} → {found.node_id})"
            )
    for action, _, node in paraphrases:
        if node not in candidates:
            problems.append(f"{action!r}: intended node {node!r} is not a semantic candidate")
    return problems


def measure(
    graph: ProceduralGraph,
    embeddings: Any,
    unrelated: Sequence[tuple[str, str]] = UNRELATED,
    paraphrases: Sequence[tuple[str, str, str]] = PARAPHRASES,
) -> tuple[list[Measurement], list[Measurement]]:
    """Best candidate node and cosine for every string of both sets."""
    ids = _semantic_candidates(graph)
    vectors = embeddings.embed_documents([_node_text(graph, node_id) for node_id in ids])

    def best(action: str, observation: str, intended: str | None) -> Measurement:
        query = embeddings.embed_query(step_text(action, observation))
        node, score = None, -1.0
        for node_id, vector in zip(ids, vectors, strict=True):
            similarity = _cosine(query, vector)
            if similarity > score:  # strict: the first maximum wins, as in guide()
                node, score = node_id, similarity
        return Measurement(action, observation, intended, str(node), score)

    return (
        [best(a, o, None) for a, o in unrelated],
        [best(a, o, n) for a, o, n in paraphrases],
    )


def outcome_at(
    threshold: float, unrelated: Sequence[Measurement], paraphrases: Sequence[Measurement]
) -> Outcome:
    """Apply the step's rule (localize iff score ≥ threshold) to both sets."""
    placed = [m for m in paraphrases if m.score >= threshold]
    right = sum(1 for m in placed if m.right_node)
    return Outcome(
        threshold=threshold,
        unrelated_full=sum(1 for m in unrelated if m.score < threshold),
        right=right,
        wrong=len(placed) - right,
        missed=len(paraphrases) - len(placed),
        total=len(unrelated) + len(paraphrases),
    )


def thresholds_with(configured: float, grid: Iterable[float] = THRESHOLDS) -> list[float]:
    return sorted({round(t, 2) for t in grid} | {round(configured, 2)})


def distribution(scores: Sequence[float]) -> str:
    """``n  min  p25  median  p75  max`` for one set of best cosines."""
    if not scores:
        return f"{0:>4}"
    ordered = sorted(scores)
    if len(ordered) > 1:
        p25, _, p75 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        p25 = p75 = ordered[0]
    cells = (ordered[0], p25, statistics.median(ordered), p75, ordered[-1])
    return f"{len(ordered):>4}  " + "  ".join(f"{v:.3f}" for v in cells)


def histogram_rows(
    unrelated: Sequence[Measurement], paraphrases: Sequence[Measurement], width: float = 0.05
) -> list[tuple[str, int, int, int]]:
    """(bin, unrelated, paraphrase whose best node is right, … is wrong) per ``width`` bin."""
    everything = [m.score for m in (*unrelated, *paraphrases)]
    if not everything:
        return []
    low = int(min(everything) / width)
    high = int(max(everything) / width)
    rows = []
    for index in range(low, high + 1):
        start, end = index * width, (index + 1) * width

        def inside(m: Measurement, start: float = start, end: float = end) -> bool:
            return start <= m.score < end

        rows.append(
            (
                f"{start:.2f}–{end:.2f}",
                sum(1 for m in unrelated if inside(m)),
                sum(1 for m in paraphrases if inside(m) and m.right_node),
                sum(1 for m in paraphrases if inside(m) and not m.right_node),
            )
        )
    return rows


def _shown(m: Measurement) -> str:
    text = step_text(m.action, m.observation)
    return text if len(text) <= 44 else text[:43] + "…"


def report(
    unrelated: Sequence[Measurement],
    paraphrases: Sequence[Measurement],
    *,
    configured: float,
    embedder: str,
    candidates: int,
    details: bool = False,
) -> list[str]:
    """The printed report, as lines."""
    right_top = [m.score for m in paraphrases if m.right_node]
    wrong_top = [m.score for m in paraphrases if not m.right_node]
    lines = [
        "Semantic localization calibration",
        f"  embedder   {embedder}",
        f"  graph      {PRIOR} (bundled prior): {candidates} candidate nodes "
        "(every node but Start and the terminals)",
        f"  strings    {len(unrelated)} unrelated · {len(paraphrases)} paraphrased tool calls",
        f"  configured PROCEDURAL_SEMANTIC_THRESHOLD = {configured:.2f}",
        "",
        "Best cosine per string (what the semantic step compares with the threshold)",
        "                            n    min    p25  median    p75    max",
        f"  unrelated              {distribution([m.score for m in unrelated])}",
        f"  paraphrases            {distribution([m.score for m in paraphrases])}",
        f"    best node right      {distribution(right_top)}",
        f"    best node wrong      {distribution(wrong_top)}",
        "",
        "Histogram of best cosines",
        "  bin          unrelated  paraphrase:right  paraphrase:wrong",
    ]
    for label, u, r, w in histogram_rows(unrelated, paraphrases):
        lines.append(f"  {label:<11}  {u:>9}  {r:>16}  {w:>16}")
    lines += [
        "",
        "Outcome at each threshold (a step localizes iff its best cosine >= threshold)",
        "  threshold  unrelated→full  paraphrase→right  paraphrase→wrong  paraphrase→full"
        "  accuracy",
    ]
    for t in thresholds_with(configured):
        o = outcome_at(t, unrelated, paraphrases)
        mark = "  ◀ configured" if round(t, 2) == round(configured, 2) else ""
        lines.append(
            f"  {t:>9.2f}  {o.unrelated_full:>9}/{len(unrelated):<4}  {o.right:>12}/{len(paraphrases):<3}"
            f"  {o.wrong:>16}  {o.missed:>15}  {o.accuracy:>7.1%}{mark}"
        )
    lines += [
        "",
        "  accuracy = (unrelated → full graph + paraphrase → intended node) / all strings.",
        "  paraphrase→wrong is the harmful case; paraphrase→full falls back to the paper's",
        "  behaviour (the whole graph).",
        "",
        "Highest-scoring unrelated strings",
    ]
    for m in sorted(unrelated, key=lambda m: m.score, reverse=True)[:5]:
        lines.append(f"  {m.score:.3f}  {_shown(m):<44}  → {m.node}")
    lines += ["", "Paraphrases whose best node is not the intended one"]
    wrong = sorted((m for m in paraphrases if not m.right_node), key=lambda m: -m.score)
    lines += [f"  {m.score:.3f}  {_shown(m):<44}  → {m.node} (meant {m.intended})" for m in wrong]
    if not wrong:
        lines.append("  (none)")
    if details:
        lines += ["", "Every string"]
        for m in sorted((*unrelated, *paraphrases), key=lambda m: -m.score):
            meant = f" (meant {m.intended})" if m.intended else " (unrelated)"
            lines.append(f"  {m.score:.3f}  {_shown(m):<44}  → {m.node}{meant}")
    return lines


def _embedder_label(settings: Any) -> str:
    provider = settings.embedding_provider.lower()
    field = "fastembed_model" if provider == "fastembed" else f"{provider}_embedding_model"
    model = getattr(settings, field, "")
    return f"{provider} · {model}" if model else provider


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.calibrate_semantic_threshold",
        description=(
            "Measure PROCEDURAL_SEMANTIC_THRESHOLD for the configured embedder on two fixed "
            "string sets (no LLM, no database)."
        ),
    )
    parser.add_argument(
        "--details", action="store_true", help="also list every string with its best node"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    graph = load_prior(PRIOR)
    problems = check_sets(graph)
    if problems:
        for problem in problems:
            print(f"❌ {problem}", file=sys.stderr)
        return 1
    try:
        embeddings = get_embeddings(settings)
        unrelated, paraphrases = measure(graph, embeddings)
    except Exception as e:  # noqa: BLE001 - a missing key or package is a user-facing message
        print(
            f"❌ Could not embed with EMBEDDING_PROVIDER={settings.embedding_provider}: {e}",
            file=sys.stderr,
        )
        return 2
    print(
        "\n".join(
            report(
                unrelated,
                paraphrases,
                configured=float(settings.procedural_semantic_threshold),
                embedder=_embedder_label(settings),
                candidates=len(_semantic_candidates(graph)),
                details=args.details,
            )
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
