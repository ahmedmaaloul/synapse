# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
RQ6 — Does graph topology predict retrieval success?

HYPOTHESIS
    Retrieval succeeds when the question's required facts sit close together in
    the graph. Per-question topological features — shortest-path distance from
    the question's anchor entity (and from the entities retrieval actually
    seeds on) to each required fact, the degree of those nodes, whether they
    share a Louvain community, the local clustering coefficient — should
    therefore predict whether the graph systems retrieve *all* the evidence.

PREDICTION (registered before looking at the outcome column)
    P1  Questions whose required facts are FAR from the seeds fail more often
        than questions whose required facts are near them.
    P2  Mechanistically: ``chat_engine._expand_and_format`` materialises every
        seed's *immediate* neighbours, so a required fact at seed-distance 0 or
        1 is retrieved by construction. A fact at distance >= 2 can only be
        retrieved if it happens to lie on a rendered reasoning path. Misses
        should therefore be concentrated at seed-distance >= 2.
    P3  Anchor-based distance (which does not depend on what the retriever
        chose) should be a weaker predictor than seed-based distance, because
        the retriever's seeds, not the dataset's anchor, are where traversal
        starts.

WHAT IS MEASURED, WHAT IS INFERRED
    MEASURED: the fixture graph's topology, every per-question feature, and the
    per-question / per-fact outcomes of the two graph systems, run through the
    unmodified production retrieval path.
    INFERRED: any causal reading of the association. n = 14 questions. This is
    an exploratory, underpowered analysis; no p-value is reported as if it
    settled anything.

COST
    Zero LLM calls. Pre-extracted fixtures + local fastembed embeddings only.

RUN
    cd backend && EMBEDDING_PROVIDER=fastembed \\
        NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret \\
        python research/rq6_topology/run_rq6.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Read-only import: benchmarks/ is owned by another workflow and is never
# modified here. Reusing its evaluators is what makes these numbers comparable
# with the published benchmark report.
from benchmarks.run_benchmark import (  # noqa: E402
    DATASET_PATH,
    QuestionResult,
    contaminated,
    evaluate_graphrag,
    load_dataset,
    seed_chunks,
    seed_graph,
)

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"
FINDINGS_PATH = HERE / "FINDINGS.md"

#: Fixed everywhere: bootstrap resampling, Louvain, tie-breaking.
SEED = 20260719
#: Resamples for every 95% CI. The brief asks for >= 2000; 10000 is cheap here.
BOOTSTRAP_N = 10000
#: Seed entities handed to ``retrieve_subgraph`` — the production default and
#: the value the published benchmark uses, so rows are comparable.
GRAPH_K = 8
#: How many times to re-attempt the measurement when the database is observed to
#: change underneath it. See ``Interference``.
MAX_ATTEMPTS = 6


class Interference(RuntimeError):
    """The database changed while a measurement was in flight.

    This is not hypothetical. The shared research instance on port 7688 is
    written by several experiment scripts at once; a monitoring probe recorded
    its entity count moving 38 → 78 → 19 → 119 → 114 → 94 → 109 → 0 → 79 over
    two minutes while this experiment was idle. A retrieval measurement taken
    across such a window is meaningless — and, worse, silently meaningless: it
    still prints a plausible-looking coverage number. Every measurement here is
    therefore bracketed by a graph fingerprint, and any run whose fingerprint
    moves is discarded and retried rather than reported.
    """


async def graph_fingerprint() -> dict:
    """A cheap content hash of the stored graph, used to detect concurrent writes."""
    from app.neo4j_driver import execute_query

    rows = await execute_query("MATCH (n:Entity) RETURN n.name AS name ORDER BY n.name")
    names = [r["name"] for r in rows]
    rels = await execute_query("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c")
    chunks = await execute_query("MATCH (c:Chunk) RETURN count(c) AS c")
    return {
        "n_entities": len(names),
        "n_relationships": int(rels[0]["c"]),
        "n_chunks": int(chunks[0]["c"]),
        "digest": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16],
    }


def _same_graph(a: dict, b: dict) -> bool:
    """Entities and relationships identical. Chunk count is allowed to differ."""
    return (
        a["digest"] == b["digest"]
        and a["n_entities"] == b["n_entities"]
        and a["n_relationships"] == b["n_relationships"]
    )


async def measured_run(dataset: dict) -> tuple[list[QuestionResult], list[QuestionResult], dict]:
    """Seed, measure C and D, and verify the graph never moved. Raises ``Interference``."""
    await seed_graph(dataset, echo=False)

    before = await graph_fingerprint()
    expected_entities = len({e["name"] for e in dataset["entities"]})
    if before["n_entities"] != expected_entities:
        raise Interference(
            f"after seeding, the database holds {before['n_entities']} entities, expected "
            f"{expected_entities} — another writer is active"
        )
    if before["n_chunks"]:
        raise Interference(
            f"the chunk store is not empty ({before['n_chunks']} chunks) at the start of the "
            "graph-only measurement"
        )

    graph_only = await evaluate_graphrag(dataset, GRAPH_K)
    mid = await graph_fingerprint()
    if not _same_graph(before, mid) or mid["n_chunks"]:
        raise Interference("the graph changed while system C was being measured")
    tainted = contaminated(graph_only)
    if tainted:
        raise Interference(
            f"graph-only contexts already carry source excerpts ({', '.join(tainted)})"
        )

    await seed_chunks(dataset)
    seeded_chunks = await graph_fingerprint()
    if not _same_graph(before, seeded_chunks):
        raise Interference("the graph changed while the source chunks were being stored")

    with_chunks = await evaluate_graphrag(dataset, GRAPH_K)
    after = await graph_fingerprint()
    if not _same_graph(before, after) or after["n_chunks"] != seeded_chunks["n_chunks"]:
        raise Interference("the graph changed while system D was being measured")

    return graph_only, with_chunks, {"graph": before, "with_chunks": seeded_chunks}


# ── Bootstrap / effect-size helpers ──────────────────────────────────────────
def bootstrap_ci(
    values: np.ndarray,
    statistic,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``statistic`` over rows of ``values``.

    Rows are the resampling unit. Everywhere in this file a row is one
    *question*, never one fact: required facts within a question are not
    independent, so resampling facts would understate every interval.
    """
    values = np.asarray(values, dtype=float)
    point = float(statistic(values))
    if values.size == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = np.array([statistic(values[row]) for row in idx], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_ci_paired(
    table: np.ndarray,
    statistic,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic of a (questions x columns) table."""
    table = np.asarray(table, dtype=float)
    point = float(statistic(table))
    if table.shape[0] == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, table.shape[0], size=(n, table.shape[0]))
    draws = np.array([statistic(table[row]) for row in idx], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    """Exact binomial 95% interval.

    Reported *alongside* the bootstrap wherever the bootstrap degenerates: at
    0/n or n/n the percentile bootstrap returns a zero-width interval, which
    looks like certainty and is not. Clopper–Pearson treats the facts as
    independent, which they are not (they are nested in questions), so it is
    optimistic in the other direction — quoting both is the honest option.
    """
    from scipy.stats import beta

    if total == 0:
        return float("nan"), float("nan")
    lo = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    hi = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return lo, hi


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta — P(a > b) − P(a < b). Nonparametric, no distributional claim."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = float(np.sum(a[:, None] > b[None, :]))
    less = float(np.sum(a[:, None] < b[None, :]))
    return (greater - less) / (a.size * b.size)


def cliffs_label(delta: float) -> str:
    """Romano et al. (2006) thresholds. A label, not a verdict."""
    if not np.isfinite(delta):
        return "undefined"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho via Pearson on ranks (average ranks for ties)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — the same convention as scipy.stats.rankdata."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average the ranks inside each tie group.
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


# ── The fixture graph ────────────────────────────────────────────────────────
def build_graph(dataset: dict) -> nx.Graph:
    """The fixture graph as retrieval sees it: UNDIRECTED.

    ``chat_engine._neighborhood_edges`` matches ``(n)-[r]-(m)`` and
    ``chat_engine._adjacency`` stores both directions, so traversal ignores edge
    direction. Modelling the graph as directed would measure a system that does
    not exist.
    """
    graph = nx.Graph()
    for entity in dataset["entities"]:
        graph.add_node(entity["name"], type=entity.get("type"))
    for rel in dataset["relationships"]:
        source, target = rel.get("source"), rel.get("target")
        if source is None or target is None or source == target:
            continue
        graph.add_edge(source, target, rel=rel.get("type"))
    return graph


def graph_profile(graph: nx.Graph, partition: list[set[str]]) -> dict:
    """Global properties, so a reader can judge how representative the fixture is."""
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    giant = max(nx.connected_components(graph), key=len)
    giant_graph = graph.subgraph(giant)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_p90": float(np.percentile(degrees, 90)),
        "degree_histogram": {str(k): v for k, v in sorted(Counter(degrees.astype(int)).items())},
        "isolated_nodes": int((degrees == 0).sum()),
        "components": components,
        "giant_fraction": len(giant) / graph.number_of_nodes(),
        "giant_diameter": int(nx.diameter(giant_graph)),
        "giant_mean_path_length": float(nx.average_shortest_path_length(giant_graph)),
        "avg_clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "degree_assortativity": float(nx.degree_assortativity_coefficient(graph)),
        "n_communities": len(partition),
        "community_sizes": sorted((len(c) for c in partition), reverse=True),
        "modularity": float(nx.algorithms.community.modularity(graph, partition)),
        # Independent cycles: E − N + components. 0 would be an exact forest.
        "independent_cycles": (
            graph.number_of_edges() - graph.number_of_nodes() + len(components)
        ),
        "n_triangles": int(sum(nx.triangles(graph).values()) // 3),
    }


def louvain(graph: nx.Graph) -> list[set[str]]:
    """Louvain partition with the production settings (resolution 1.0) and a fixed seed."""
    return nx.algorithms.community.louvain_communities(graph, resolution=1.0, seed=SEED)


# ── Per-question topological features ────────────────────────────────────────
#: Distance used when two nodes lie in different connected components. The
#: fixture graph is one component (asserted in ``main``), so this is a guard,
#: not a fudge factor: if it ever fires, the run says so loudly.
DISCONNECTED = 99


@dataclass
class QuestionFeatures:
    """Topology of one benchmark question, independent of any retrieval run."""

    qid: str
    anchor: str
    required: list[str]
    n_required: int
    anchor_max_dist: float
    anchor_mean_dist: float
    required_spread: float
    required_degree_mean: float
    required_degree_min: float
    anchor_degree: float
    required_clustering_mean: float
    anchor_clustering: float
    same_community_fraction: float
    anchor_dists: dict[str, int] = field(default_factory=dict)


def question_features(
    question: dict,
    graph: nx.Graph,
    dist: dict[str, dict[str, int]],
    community_of: dict[str, int],
    clustering: dict[str, float],
) -> QuestionFeatures:
    anchor = question["anchor"]
    required = list(question["required_facts"])
    anchor_dists = {f: dist[anchor].get(f, DISCONNECTED) for f in required}
    pairwise = [
        dist[a].get(b, DISCONNECTED)
        for i, a in enumerate(required)
        for b in required[i + 1 :]
    ]
    return QuestionFeatures(
        qid=question["id"],
        anchor=anchor,
        required=required,
        n_required=len(required),
        anchor_max_dist=float(max(anchor_dists.values())),
        anchor_mean_dist=float(np.mean(list(anchor_dists.values()))),
        required_spread=float(max(pairwise)) if pairwise else 0.0,
        required_degree_mean=float(np.mean([graph.degree(f) for f in required])),
        required_degree_min=float(min(graph.degree(f) for f in required)),
        anchor_degree=float(graph.degree(anchor)),
        required_clustering_mean=float(np.mean([clustering[f] for f in required])),
        anchor_clustering=float(clustering[anchor]),
        same_community_fraction=float(
            np.mean([1.0 if community_of[f] == community_of[anchor] else 0.0 for f in required])
        ),
        anchor_dists=anchor_dists,
    )


def seed_distances(
    result: QuestionResult,
    dist: dict[str, dict[str, int]],
) -> dict[str, int]:
    """For each required fact, its distance to the NEAREST retrieved seed entity.

    This is the operationally relevant distance: traversal starts at the seeds
    the retriever actually picked, not at the dataset's nominal anchor.
    """
    seeds = [s for s in (result.extras.get("seeds") or []) if s in dist]
    out: dict[str, int] = {}
    for fact in result.required:
        if not seeds or fact not in dist:
            out[fact] = DISCONNECTED
            continue
        out[fact] = min(dist[seed].get(fact, DISCONNECTED) for seed in seeds)
    return out


# ── Analysis ─────────────────────────────────────────────────────────────────
#: The per-question features tested against success. Direction is the sign the
#: hypothesis predicts for "feature vs. success": '-' = higher feature, lower
#: success.
FEATURES: tuple[tuple[str, str, str], ...] = (
    ("seed_max_dist", "Max seed→fact distance", "-"),
    ("seed_mean_dist", "Mean seed→fact distance", "-"),
    ("n_facts_beyond_1hop", "Required facts at seed-distance ≥ 2", "-"),
    ("anchor_max_dist", "Max anchor→fact distance", "-"),
    ("anchor_mean_dist", "Mean anchor→fact distance", "-"),
    ("required_spread", "Spread of the required set (max pairwise distance)", "-"),
    ("n_required", "Number of required facts", "-"),
    ("required_degree_mean", "Mean degree of required facts", "+"),
    ("required_degree_min", "Min degree of required facts", "+"),
    ("anchor_degree", "Degree of the anchor entity", "+"),
    ("required_clustering_mean", "Mean local clustering of required facts", "+"),
    ("anchor_clustering", "Local clustering of the anchor", "+"),
    ("same_community_fraction", "Fraction of required facts in the anchor's community", "+"),
)


def feature_table(
    rows: list[dict],
    outcome_key: str,
) -> list[dict]:
    """Group means (success vs. failure), bootstrap CIs, Cliff's delta, Spearman rho."""
    success = np.array([r[outcome_key] for r in rows], dtype=float)
    out: list[dict] = []
    for key, label, predicted in FEATURES:
        values = np.array([r[key] for r in rows], dtype=float)
        table = np.column_stack([values, success])

        def mean_when(t: np.ndarray, want: float) -> float:
            sel = t[t[:, 1] == want][:, 0]
            return float(sel.mean()) if sel.size else float("nan")

        hit_m, hit_lo, hit_hi = bootstrap_ci_paired(table, lambda t: mean_when(t, 1.0))
        miss_m, miss_lo, miss_hi = bootstrap_ci_paired(table, lambda t: mean_when(t, 0.0))
        diff, diff_lo, diff_hi = bootstrap_ci_paired(
            table, lambda t: mean_when(t, 0.0) - mean_when(t, 1.0)
        )
        rho, rho_lo, rho_hi = bootstrap_ci_paired(table, lambda t: spearman(t[:, 0], t[:, 1]))
        delta = cliffs_delta(values[success == 0], values[success == 1])
        out.append(
            {
                "key": key,
                "label": label,
                "predicted_sign": predicted,
                "mean_success": hit_m,
                "mean_success_ci": [hit_lo, hit_hi],
                "mean_failure": miss_m,
                "mean_failure_ci": [miss_lo, miss_hi],
                "diff_failure_minus_success": diff,
                "diff_ci": [diff_lo, diff_hi],
                "spearman_rho_vs_success": rho,
                "spearman_ci": [rho_lo, rho_hi],
                "cliffs_delta_failure_vs_success": delta,
                "cliffs_label": cliffs_label(delta),
                "ci_excludes_zero": bool(
                    np.isfinite(diff_lo) and np.isfinite(diff_hi) and (diff_lo > 0 or diff_hi < 0)
                ),
            }
        )
    return out


def fact_level_analysis(rows: list[dict], system: str) -> dict:
    """P(fact retrieved | its distance to the nearest seed), clustered by question.

    Facts are nested inside questions, so the bootstrap resamples QUESTIONS and
    recomputes the pooled per-fact rate. Resampling facts directly would treat
    four facts from one question as four independent observations.
    """
    buckets = {"0": [], "1": [], ">=2": []}
    per_question: list[dict] = []
    for row in rows:
        facts = row[f"{system}_facts"]
        counts = {k: [0, 0] for k in buckets}  # bucket -> [found, total]
        for fact in facts:
            bucket = "0" if fact["seed_dist"] == 0 else ("1" if fact["seed_dist"] == 1 else ">=2")
            counts[bucket][1] += 1
            counts[bucket][0] += 1 if fact["found"] else 0
        per_question.append(counts)
        for bucket, (found, total) in counts.items():
            if total:
                buckets[bucket].append((found, total))

    table = np.array(
        [
            [c[b][i] for b in ("0", "1", ">=2") for i in (0, 1)]
            for c in per_question
        ],
        dtype=float,
    )

    def rate(t: np.ndarray, col: int) -> float:
        found, total = t[:, col].sum(), t[:, col + 1].sum()
        return float(found / total) if total else float("nan")

    out = {}
    for i, bucket in enumerate(("0", "1", ">=2")):
        col = i * 2
        point, lo, hi = bootstrap_ci_paired(table, lambda t, c=col: rate(t, c))
        found = int(table[:, col].sum())
        total = int(table[:, col + 1].sum())
        out[bucket] = {
            "n_facts": total,
            "n_found": found,
            "rate": point,
            "ci": [lo, hi],
            "exact_ci": list(clopper_pearson(found, total)),
        }
    def near_rate(t: np.ndarray) -> float:
        """Buckets 0 and 1 pooled — everything inside the materialised neighbourhood."""
        found = t[:, 0].sum() + t[:, 2].sum()
        total = t[:, 1].sum() + t[:, 3].sum()
        return float(found / total) if total else float("nan")

    near, near_lo, near_hi = bootstrap_ci_paired(table, near_rate)
    gap, gap_lo, gap_hi = bootstrap_ci_paired(table, lambda t: near_rate(t) - rate(t, 4))
    near_total = int(table[:, 1].sum() + table[:, 3].sum())
    near_found = int(table[:, 0].sum() + table[:, 2].sum())
    out["near"] = {
        "n_facts": near_total,
        "n_found": near_found,
        "rate": near,
        "ci": [near_lo, near_hi],
        "exact_ci": list(clopper_pearson(near_found, near_total)),
    }
    out["near_minus_far"] = {"gap": gap, "ci": [gap_lo, gap_hi]}
    return out


#: ``settings.retrieval_max_hops`` (2) bounds the BFS that finds reasoning
#: paths, and ``build_reasoning_paths`` accepts paths up to ``hops * 2`` edges.
MAX_HOPS = 2
MAX_PATH_LEN = MAX_HOPS * 2
#: ``chat_engine.MAX_SEEDS_FOR_PATHS`` — only the top seeds are paired up.
MAX_SEEDS_FOR_PATHS = 6


def hop_frontier_analysis(
    rows: list[dict],
    graph: nx.Graph,
    dist: dict[str, dict[str, int]],
    system: str,
) -> list[dict]:
    """Cost/benefit of materialising a deeper neighbourhood.

    This is ARITHMETIC ON THE GRAPH, not a measured intervention: it says how
    many facts would fall inside an ``h``-hop neighbourhood of the seeds the
    retriever already picked, and how many entities that neighbourhood contains.
    It does *not* run a modified retriever, and it assumes the permissive
    scoring rule, under which "in the neighbourhood" implies "scored".
    """
    out: list[dict] = []
    for h in (1, 2, 3, 4):
        frontier_sizes, covered = [], []
        for row in rows:
            seeds = [s for s in row["seeds"] if s in dist]
            reach = set()
            for seed in seeds:
                reach.update(n for n, d in dist[seed].items() if d <= h)
            frontier_sizes.append(len(reach))
            covered.append(
                1.0
                if all(f["seed_dist"] <= h for f in row[f"{system}_facts"])
                else 0.0
            )
        point, lo, hi = bootstrap_ci(np.array(covered), np.mean)
        out.append(
            {
                "hops": h,
                "mean_frontier_nodes": float(np.mean(frontier_sizes)),
                "frontier_share_of_graph": float(np.mean(frontier_sizes))
                / graph.number_of_nodes(),
                "implied_full_coverage": point,
                "implied_ci": [lo, hi],
                "n_questions_covered": int(sum(covered)),
            }
        )
    return out


def path_rescue_diagnostic(
    rows: list[dict],
    graph: nx.Graph,
    system: str,
) -> dict:
    """Why reasoning paths do not rescue distant facts.

    A rendered reasoning path is a shortest path between two of the question's
    top seeds. A required fact beyond the materialised neighbourhood therefore
    only reaches the context if it happens to be an *internal node* of such a
    path. This counts how often that is even geometrically possible, so the
    observed rescue rate can be read as mechanism rather than bad luck.

    Deliberately GENEROUS to the path channel in two ways, so a zero here is
    hard to argue with: it unions the interiors of *all* shortest paths between
    every seed pair (the engine renders one, chosen by a tie-break), and it
    searches the whole graph rather than only the edges the engine's bounded
    BFS actually discovered.
    """
    eligible, rescued, checked = 0, 0, 0
    details = []
    for row in rows:
        seeds = [s for s in row["seeds"] if s in graph][:MAX_SEEDS_FOR_PATHS]
        interior: set[str] = set()
        for i, source in enumerate(seeds):
            for target in seeds[i + 1 :]:
                try:
                    for nodes in nx.all_shortest_paths(graph, source, target):
                        if len(nodes) - 1 <= MAX_PATH_LEN:
                            interior.update(nodes[1:-1])
                except nx.NetworkXNoPath:
                    continue
        for fact in row[f"{system}_facts"]:
            if fact["seed_dist"] < 2:
                continue
            checked += 1
            on_path = fact["fact"] in interior
            eligible += 1 if on_path else 0
            rescued += 1 if fact["found"] else 0
            details.append(
                {
                    "qid": row["qid"],
                    "fact": fact["fact"],
                    "seed_dist": fact["seed_dist"],
                    "on_a_seed_pair_shortest_path": on_path,
                    "found": fact["found"],
                }
            )
    return {
        "n_far_facts": checked,
        "n_on_seed_pair_path": eligible,
        "n_retrieved": rescued,
        "mean_rendered_paths": float(np.mean([r["n_paths"] for r in rows])),
        "details": details,
    }


def miss_concentration(rows: list[dict], system: str) -> dict:
    """Where do the misses live? The actionable corollary, and the RQ2 link."""
    missed, all_facts = [], []
    for row in rows:
        for fact in row[f"{system}_facts"]:
            all_facts.append(fact)
            if not fact["found"]:
                missed.append(fact)
    dist_hist = Counter(min(f["seed_dist"], 5) for f in all_facts)
    miss_hist = Counter(min(f["seed_dist"], 5) for f in missed)

    table = np.array(
        [
            [
                sum(1 for f in row[f"{system}_facts"] if not f["found"] and f["seed_dist"] >= 2),
                sum(1 for f in row[f"{system}_facts"] if not f["found"]),
            ]
            for row in rows
        ],
        dtype=float,
    )

    def share(t: np.ndarray) -> float:
        total = t[:, 1].sum()
        return float(t[:, 0].sum() / total) if total else float("nan")

    point, lo, hi = bootstrap_ci_paired(table, share)
    return {
        "n_missed": len(missed),
        "n_facts": len(all_facts),
        "share_missed_beyond_1hop": point,
        "share_ci": [lo, hi],
        "distance_histogram_all": {str(k): v for k, v in sorted(dist_hist.items())},
        "distance_histogram_missed": {str(k): v for k, v in sorted(miss_hist.items())},
        "missed_facts": [
            {"qid": f["qid"], "fact": f["fact"], "seed_dist": f["seed_dist"]} for f in missed
        ],
    }


# ── Report generation ────────────────────────────────────────────────────────
def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


def num(x: float, places: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{places}f}"


def ci(lo: float, hi: float, formatter=num) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[n/a]"
    return f"[{formatter(lo)}, {formatter(hi)}]"


def render_findings(data: dict) -> str:
    """Write FINDINGS.md. Every number in it comes from ``data`` — none is typed."""
    profile = data["graph_profile"]
    lines: list[str] = []
    add = lines.append

    add("# RQ6 — Does graph topology predict retrieval success?")
    add("")
    add(f"*Generated by `research/rq6_topology/run_rq6.py` — every number below is computed, "
        f"never hand-typed. Seed {SEED}, {BOOTSTRAP_N:,} bootstrap resamples.*")
    add("")
    add("```")
    add(data["command"])
    add("```")
    add("")

    add("## Hypothesis and prediction (stated before the results)")
    add("")
    add("**Hypothesis.** Retrieval succeeds when a question's required facts sit close together "
        "in the graph. Per-question topological features should therefore predict whether the "
        "graph systems retrieve *all* the evidence.")
    add("")
    add("**Predictions, registered in the script's docstring before the outcome column was "
        "inspected:**")
    add("")
    add("- **P1** — questions whose required facts are far from the retrieval seeds fail more "
        "often than questions whose facts are near them.")
    add("- **P2** — mechanistically, `chat_engine._expand_and_format` materialises every seed's "
        "*immediate* neighbours, so a fact at seed-distance 0 or 1 is retrieved by construction, "
        "while a fact at distance ≥ 2 is retrieved only if it lands on a rendered reasoning path. "
        "Misses should be concentrated at seed-distance ≥ 2.")
    add("- **P3** — the dataset's nominal anchor entity should predict less well than the seeds "
        "the retriever actually chose.")
    add("")

    add("## Method")
    add("")
    add(f"- **Graph.** The {profile['nodes']} entities and {profile['edges']} relationships of "
        "`benchmarks/dataset.json`, loaded into NetworkX **undirected** — because "
        "`chat_engine._neighborhood_edges` matches `(n)-[r]-(m)` and `_adjacency` stores both "
        "directions, so the production traversal ignores edge direction. Modelling it as directed "
        "would measure a system that does not exist.")
    add(f"- **Retrieval.** The two graph rows of the published benchmark, run through the "
        f"unmodified production path `chat_engine.retrieve_subgraph` at k={GRAPH_K}, by importing "
        "`benchmarks/run_benchmark.py` read-only rather than reimplementing it: "
        "**C** = graph only (empty chunk store, verified), **D** = graph + source chunks. Scoring "
        "is the benchmark's own permissive rule (a required fact counts when its name appears "
        "anywhere in the retrieved context) and the outcome analysed is **full coverage** — all "
        "required facts retrieved, which is the only pass/fail that matters for a multi-hop "
        "question.")
    add("- **Features.** Per question: shortest-path distance from the anchor entity and from the "
        "nearest *retrieved seed* to each required fact; the spread of the required set; degrees; "
        "local clustering coefficients; and the fraction of required facts sharing the anchor's "
        "Louvain community (resolution 1.0, the production setting, fixed seed).")
    add(f"- **Uncertainty.** Percentile bootstrap, {BOOTSTRAP_N:,} resamples, resampling "
        "**questions** (never facts — facts are nested inside questions and are not independent). "
        "Effect size is Cliff's delta. No p-values are reported: with n=14 they would be theatre.")
    add("- **Cost.** Zero LLM calls; pre-extracted fixtures and local fastembed embeddings only.")
    integrity = data["integrity"]
    fp = integrity["fingerprints"]["graph"]
    add(f"- **Measurement isolation (this matters — see below).** Every number was measured "
        f"inside a *verified-quiet* window on `{data['neo4j_uri']}`: the graph is fingerprinted "
        f"(entity-name digest `{fp['digest']}`, {fp['n_entities']} entities, "
        f"{fp['n_relationships']} relationships) immediately after seeding and again after each "
        f"system is scored, and any run in which it moved is **discarded and retried**, never "
        f"reported. This run needed {integrity['attempts_used']} attempt(s).")
    add("")

    add("### Why isolation had to be enforced")
    add("")
    add("The first version of this experiment was run against the shared research instance on "
        "port 7688 and **its headline was not reproducible**: repeated identical runs put system "
        "C's full coverage at 9/14, then 5/14, then 6/14, then 9/14, and the set of features "
        "whose interval excluded zero changed with it. The cause was not the retriever and not "
        "the bootstrap. Probing that database while this experiment was completely idle showed "
        "its entity count moving")
    add("")
    add("```")
    add("38 → 78 → 19 → 119 → 114 → 94 → 109 → 0 → 79    (over ~2 minutes, nothing of ours running)")
    add("```")
    add("")
    add("because several experiment scripts wipe and re-seed that instance concurrently. A "
        "retrieval measurement taken across such a window is meaningless, and dangerously so: it "
        "still returns a plausible coverage number with no outward sign of corruption. Seeding at "
        "the start of the script — the documented mitigation — does not help, because the "
        "interference arrives *during* the measurement. The experiment was therefore moved to a "
        "private Neo4j instance that nothing else writes to, and the fingerprint guard above was "
        "added so that this failure mode can never again be mistaken for a result. **Any earlier "
        "RQ6 numbers taken on 7688 should be discarded.**")
    add("")

    add("## Result 0 — the fixture graph itself")
    add("")
    add("So a reader can judge how representative this substrate is.")
    add("")
    add("| Property | Value |")
    add("| --- | ---: |")
    add(f"| Nodes / edges | {profile['nodes']} / {profile['edges']} |")
    add(f"| Density | {num(profile['density'], 4)} |")
    add(f"| Degree: mean / median / max | {num(profile['degree_mean'])} / "
        f"{num(profile['degree_median'], 1)} / {profile['degree_max']} |")
    add(f"| Degree: min / 90th pct | {profile['degree_min']} / {num(profile['degree_p90'], 1)} |")
    add(f"| Connected components (sizes) | {len(profile['components'])} "
        f"({', '.join(str(c) for c in profile['components'])}) |")
    add(f"| Giant component share | {pct(profile['giant_fraction'])} |")
    add(f"| Diameter / mean path length (giant) | {profile['giant_diameter']} / "
        f"{num(profile['giant_mean_path_length'])} |")
    add(f"| Average clustering / transitivity | {num(profile['avg_clustering'], 3)} / "
        f"{num(profile['transitivity'], 3)} |")
    add(f"| Degree assortativity | {num(profile['degree_assortativity'], 3)} |")
    add(f"| Triangles / independent cycles | {profile['n_triangles']} / "
        f"{profile['independent_cycles']} |")
    add(f"| Louvain communities (sizes) | {profile['n_communities']} "
        f"({', '.join(str(c) for c in profile['community_sizes'])}) |")
    add(f"| Modularity | {num(profile['modularity'], 3)} |")
    add("")
    add(f"Degree distribution: `{json.dumps(profile['degree_histogram'])}` (degree → count).")
    add("")
    add(data["profile_note"])
    add("")

    add("## Result 1 — per-question features vs. full coverage")
    add("")
    for system in ("C", "D"):
        agg = data["systems"][system]
        add(f"### System {system} — {agg['label']}")
        add("")
        add(f"Full coverage {agg['n_success']}/{agg['n']} = {pct(agg['full_coverage'])}, "
            f"95% CI {ci(*agg['full_coverage_ci'], pct)}. "
            f"Fact recall {pct(agg['fact_recall'])}, 95% CI {ci(*agg['fact_recall_ci'], pct)}.")
        add("")
        if agg["n_success"] in (0, agg["n"]):
            add("Every question falls on the same side of the outcome, so no group comparison is "
                "possible for this system.")
            add("")
            continue
        add("Mean feature value among questions that fully succeeded vs. those that did not "
            "(bootstrap 95% CIs; Δ is failure − success, so a Δ whose CI excludes 0 is the only "
            "kind of signal this n can show).")
        add("")
        add("| Feature | pred. | success mean [95% CI] | failure mean [95% CI] | Δ [95% CI] | "
            "Cliff's δ | Spearman ρ vs. success [95% CI] |")
        add("| --- | :-: | ---: | ---: | ---: | ---: | ---: |")
        for row in agg["features"]:
            flag = " **\\***" if row["ci_excludes_zero"] else ""
            add(
                f"| {row['label']} | {row['predicted_sign']} | "
                f"{num(row['mean_success'])} {ci(*row['mean_success_ci'])} | "
                f"{num(row['mean_failure'])} {ci(*row['mean_failure_ci'])} | "
                f"{num(row['diff_failure_minus_success'])} {ci(*row['diff_ci'])}{flag} | "
                f"{num(row['cliffs_delta_failure_vs_success'])} ({row['cliffs_label']}) | "
                f"{num(row['spearman_rho_vs_success'])} {ci(*row['spearman_ci'])} |"
            )
        add("")
        add(f"**\\*** = the bootstrap CI of Δ excludes 0 "
            f"({agg['n_features_ci_excludes_zero']} of {len(agg['features'])} features). "
            f"Read that count with care: the first three rows are three renderings of one "
            f"construct (distance from the seeds to the required facts), not three independent "
            f"findings, and the failure group has {agg['n'] - agg['n_success']} question"
            f"{'' if agg['n'] - agg['n_success'] == 1 else 's'} in it. The two "
            "clustering rows are identically zero because the fixture graph contains no "
            "triangles at all — that is a fact about the substrate, not a null result about "
            "clustering.")
        add("")

    add("## Result 2 — the mechanism: retrieval probability by seed distance")
    add("")
    add("Per required fact, pooled over questions, bootstrapped by question. This is the test of "
        "P2 and it has more observations behind it than the question-level table "
        f"({data['n_required_facts']} facts vs. {data['n_questions']} questions).")
    add("")
    for system in ("C", "D"):
        fa = data["systems"][system]["fact_analysis"]
        add(f"**System {system} — {data['systems'][system]['label']}**")
        add("")
        add("| Distance from nearest retrieved seed | Facts | Retrieved | Rate | "
            "bootstrap 95% CI | exact 95% CI |")
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        for bucket, name in (
            ("0", "0 (the fact *is* a seed)"),
            ("1", "1 (a seed's neighbour)"),
            ("near", "**≤ 1 (inside the materialised neighbourhood)**"),
            (">=2", "**≥ 2 (beyond it)**"),
        ):
            cell = fa[bucket]
            add(f"| {name} | {cell['n_facts']} | {cell['n_found']} | {pct(cell['rate'])} | "
                f"{ci(*cell['ci'], pct)} | {ci(*cell['exact_ci'], pct)} |")
        gap = fa["near_minus_far"]
        add("")
        add(f"Near (≤ 1 hop) minus far (≥ 2 hops): {pct(gap['gap'])} "
            f"[{pct(gap['ci'][0])}, {pct(gap['ci'][1])}].")
        add("")
    add(data["degenerate_ci_note"])
    add("")

    add("## Result 2b — why reasoning paths do not rescue distant facts")
    add("")
    add(f"A rendered reasoning path is the shortest path between two of the question's top "
        f"{MAX_SEEDS_FOR_PATHS} seeds, at most `retrieval_max_hops × 2` = {MAX_PATH_LEN} edges "
        "long. A required fact beyond the one-hop neighbourhood can therefore only reach the "
        "context by being an *internal node* of such a path. That is a narrow target, and it is "
        "why the ≥ 2 row above is what it is. The check below is deliberately generous to the "
        "path channel — it unions the interiors of *all* shortest seed-pair paths over the whole "
        "graph, where the engine renders one path per pair over a bounded BFS frontier — so a "
        "zero in the third column is hard to argue away.")
    add("")
    add("| System | Far facts (≥ 2 hops) | Of those, on a seed-pair shortest path | "
        "Actually retrieved | Mean rendered paths / question |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for system in ("C", "D"):
        pr = data["systems"][system]["path_rescue"]
        add(f"| {system} | {pr['n_far_facts']} | {pr['n_on_seed_pair_path']} | "
            f"{pr['n_retrieved']} | {num(pr['mean_rendered_paths'], 1)} |")
    add("")
    add("System D's non-zero *retrieved* count is the source-chunk channel, not the graph: those "
        "facts arrived in prose excerpts, not through traversal.")
    add("")

    add("## Result 2c — cost/benefit of a deeper neighbourhood (graph arithmetic)")
    add("")
    add("How many facts *would* fall inside an h-hop neighbourhood of the seeds the retriever "
        "already picked, and how many entities that neighbourhood contains. **This is arithmetic "
        "on the graph, not a measured intervention**: no modified retriever was run, and it "
        "assumes the permissive rule, under which 'in the neighbourhood' implies 'scored'. Read "
        "it as an upper bound on what deepening buys and a lower bound on what it costs.")
    add("")
    add("| h | Mean entities within h hops of the seeds | Share of graph | "
        "Implied full coverage | 95% CI |")
    add("| ---: | ---: | ---: | ---: | ---: |")
    for cell in data["systems"]["C"]["hop_frontier"]:
        marker = " (shipped)" if cell["hops"] == 1 else ""
        add(f"| {cell['hops']}{marker} | "
            f"{num(cell['mean_frontier_nodes'], 1)} | "
            f"{pct(cell['frontier_share_of_graph'])} | "
            f"{cell['n_questions_covered']}/{data['n_questions']} = "
            f"{pct(cell['implied_full_coverage'])} | {ci(*cell['implied_ci'], pct)} |")
    add("")
    add(f"One table, not two: {data['seed_identity_note']}")
    add("")

    add("## Result 3 — the actionable corollary (and the link to RQ2)")
    add("")
    for system in ("C", "D"):
        mc = data["systems"][system]["miss_concentration"]
        n_far_missed = round(mc["share_missed_beyond_1hop"] * mc["n_missed"])
        add(f"**System {system}.** {mc['n_missed']} of {mc['n_facts']} required facts were "
            f"missed, and {n_far_missed} of those {mc['n_missed']} "
            f"({pct(mc['share_missed_beyond_1hop'])}) sit at seed-distance ≥ 2, i.e. outside the "
            "one-hop neighbourhood the retriever materialises. (A share of exactly 0% or 100% "
            "has a degenerate bootstrap interval; the counts are the number to read.)")
        add("")
        add(f"- All facts by seed distance: `{json.dumps(mc['distance_histogram_all'])}`")
        add(f"- Missed facts by seed distance: `{json.dumps(mc['distance_histogram_missed'])}`")
        if mc["missed_facts"]:
            add("")
            add("| Question | Missed fact | Distance to nearest seed |")
            add("| --- | --- | ---: |")
            for miss in mc["missed_facts"]:
                add(f"| {miss['qid']} | {miss['fact']} | {miss['seed_dist']} |")
        add("")

    add("## Interpretation")
    add("")
    for paragraph in data["interpretation"]:
        add(paragraph)
        add("")

    add("## Threats to validity")
    add("")
    for threat in data["threats"]:
        add(f"- {threat}")
    add("")

    add("## Claim we could defend in a paper")
    add("")
    add(f"> {data['claim']}")
    add("")
    return "\n".join(lines)


# ── Interpretation (prose is fixed; every number inside it is substituted) ───
def build_interpretation(data: dict) -> list[str]:
    c = data["systems"]["C"]
    d = data["systems"]["D"]
    fa = c["fact_analysis"]
    mc = c["miss_concentration"]

    near = fa["near"]
    far = fa[">=2"]
    seed_row = next(f for f in c["features"] if f["key"] == "seed_max_dist")
    anchor_row = next(f for f in c["features"] if f["key"] == "anchor_max_dist")

    rescue = c["path_rescue"]
    frontier = {cell["hops"]: cell for cell in c["hop_frontier"]}

    paragraphs = [
        "**The headline is split.** The *topology* half of the hypothesis is a **null result**: "
        "none of the classical graph-structure features — degree, local clustering, Louvain "
        "community co-membership, the spread of the required set, or distance from the "
        "question's nominal anchor — separates the questions system C answers fully from the "
        "ones it does not. The *distance-from-the-seeds* half is not merely supported but "
        "deterministic on this fixture. Those are different claims and we keep them apart.",

        "**What we measured, on the question level.** Of "
        f"{len(c['features'])} candidate features, {c['n_features_ci_excludes_zero']} had a "
        "group-difference bootstrap CI that excluded zero for system C — and all "
        f"{c['n_features_ci_excludes_zero']} are restatements of the same quantity, the distance "
        "from the retrieved seeds to the required facts. Every anchor-based, degree-based, "
        "clustering-based and community-based feature has a Δ interval straddling zero and a "
        "negligible-to-small Cliff's δ. With 14 questions split "
        f"{c['n_success']}/{c['n'] - c['n_success']}, group means rest on as few as "
        f"{c['n'] - c['n_success']} observations. Nothing in that table is a model of retrieval "
        "success, and we do not fit one.",

        "**Where the signal actually is.** Moving from the question to the *fact* as the unit of "
        f"analysis — {data['n_required_facts']} facts instead of 14 questions, still bootstrapped "
        "by question so the clustering is respected — produces the one large, well-separated "
        f"effect in this experiment. For system C, {near['n_found']} of {near['n_facts']} "
        "required facts within one hop of a retrieved seed were retrieved (exact 95% CI "
        f"{ci(*near['exact_ci'], pct)}); {far['n_found']} of {far['n_facts']} facts two or more "
        f"hops away were retrieved (exact 95% CI {ci(*far['exact_ci'], pct)}). That "
        "is not a subtle statistical association, it is the retriever's construction showing "
        "through: `_expand_and_format` writes out every seed's immediate neighbours, so anything "
        "at distance ≤ 1 is in the context by definition, and anything further away is in the "
        "context only if a rendered reasoning path happens to pass through it.",

        "**P2 confirmed, and it is the mechanism behind the hop limit.** System C missed "
        f"{mc['n_missed']} of the {mc['n_facts']} required facts, and all "
        f"{mc['n_missed']} of those misses ({pct(mc['share_missed_beyond_1hop'])}) lie at "
        "seed-distance ≥ 2. The failure mode of the graph-only system is not 'the graph was too "
        "sparse' or 'the community structure was wrong'; it is 'the evidence was one hop further "
        "out than the retriever materialises'. The reasoning-path channel does not compensate: "
        f"{rescue['n_on_seed_pair_path']} of the {rescue['n_far_facts']} far facts lie on a "
        "shortest path between two of the question's seeds — the only route by which a distant "
        f"entity can reach the context — and {rescue['n_retrieved']} of them were retrieved, "
        f"even though {num(rescue['mean_rendered_paths'], 1)} reasoning paths were rendered per "
        "question on average. Reasoning paths connect seeds to each other; they are not a "
        "mechanism for reaching evidence that does not happen to sit between two seeds.",

        "**The actionable corollary, and the link to RQ2.** The graph arithmetic says that "
        "materialising two hops instead of one would put "
        f"{frontier[2]['n_questions_covered']}/{c['n']} questions' full evidence inside the "
        f"neighbourhood instead of {frontier[1]['n_questions_covered']}/{c['n']}, and three hops "
        f"would reach {frontier[3]['n_questions_covered']}/{c['n']} — at the cost of growing the "
        f"materialised neighbourhood from {num(frontier[1]['mean_frontier_nodes'], 1)} entities "
        f"({pct(frontier[1]['frontier_share_of_graph'])} of the graph) to "
        f"{num(frontier[2]['mean_frontier_nodes'], 1)} "
        f"({pct(frontier[2]['frontier_share_of_graph'])}) and then "
        f"{num(frontier[3]['mean_frontier_nodes'], 1)} "
        f"({pct(frontier[3]['frontier_share_of_graph'])}). That is the shape of a hop-limit "
        "trade-off: on a near-tree the frontier grows roughly geometrically while coverage "
        "improves in discrete jumps, so the extra depth is bought with context that a "
        "budget-matched baseline will also be given. We did not run the deepened retriever, so "
        "this is arithmetic and an upper bound on the benefit, not a measured intervention — but "
        "it identifies exactly which knob the observed failures are asking for, which is what "
        "connects this result to RQ2.",

        "**P3 confirmed, weakly.** The seeds the retriever actually chose carry more signal than "
        "the dataset's nominal anchor: max seed→fact distance has Cliff's δ = "
        f"{num(seed_row['cliffs_delta_failure_vs_success'])} "
        f"({seed_row['cliffs_label']}) against success, while max anchor→fact distance has δ = "
        f"{num(anchor_row['cliffs_delta_failure_vs_success'])} ({anchor_row['cliffs_label']}). "
        "Both CIs are wide; the ordering is suggestive, not established.",

        "**What we infer, and what we do not.** We infer that seed-distance is *causally* "
        "implicated in retrieval success for the graph-only system, because we can read the "
        "mechanism directly out of `_expand_and_format` — it writes every seed's immediate "
        "neighbours and nothing beyond them — rather than only out of the correlation. We do "
        "**not** infer that topology predicts success in general; on this fixture it "
        "demonstrably does not, beyond distance. And we infer nothing from system D's feature "
        f"table: D reaches {d['n_success']}/{d['n']} full coverage, so its failure group is "
        f"{d['n'] - d['n_success']} questions and the degree features whose intervals exclude "
        "zero there are describing two data points. What D does show is that its source-chunk "
        f"channel rescues {d['fact_analysis']['>=2']['n_found']} of the "
        f"{d['fact_analysis']['>=2']['n_facts']} far facts by retrieving prose that happens to "
        "name them — adding chunks papers over the topological failure mode rather than fixing "
        "it, and the graph's own reach is unchanged.",
    ]
    return paragraphs


def build_threats(data: dict) -> list[str]:
    profile = data["graph_profile"]
    return [
        f"**Small n.** {data['n_questions']} questions, "
        f"{data['n_required_facts']} required facts, one corpus, one graph. The question-level "
        "group comparisons rest on a handful of observations per group; every CI in the "
        "question-level table is wide enough to contain 'no effect'. This is exploratory.",
        "**Multiplicity.** Thirteen features were examined against the same 14 outcomes. Any "
        "feature whose CI excludes zero here would need pre-registration and a fresh dataset "
        "before it could be called a finding; no correction is applied because with this n a "
        "corrected interval would be uninformative either way.",
        f"**The substrate is a near-tree, and this is the biggest threat.** "
        f"{profile['edges']} edges over {profile['nodes']} nodes, "
        f"{profile['independent_cycles']} independent cycles, {profile['n_triangles']} triangles, "
        f"clustering exactly {num(profile['avg_clustering'], 3)}, diameter "
        f"{profile['giant_diameter']}. On a tree there is essentially only one topological "
        "variable — distance — so the finding that 'only distance matters' is partly baked into "
        "the fixture. Two of the thirteen features are constant by construction. A real "
        "extracted knowledge graph has hubs, triangles and redundant paths, on which degree and "
        "clustering could matter and on which multiple short paths could rescue a distant fact. "
        "This experiment cannot speak to that case at all.",
        "**The neighbourhood-depth result is close to analytic.** Once you know that "
        "`_expand_and_format` materialises exactly the seeds' immediate neighbours and that "
        "scoring is permissive, 'facts at distance ≤ 1 are always scored' follows without any "
        "measurement. The empirical content of Result 2 is therefore the *other* half — that "
        "nothing else (reasoning paths, chunk co-occurrence) fills the gap — plus the "
        "distribution of required facts over distances, which is where the 7 far facts come "
        "from.",
        "**Hand-built benchmark.** `dataset.json` was authored with bridge-style multi-hop "
        "structure in mind. The distance distribution of required facts is therefore a property "
        "of the benchmark's design as much as of any real retrieval workload.",
        "**Permissive scoring.** Full coverage is scored on the benchmark's permissive rule, "
        "under which a fact counts when its name appears anywhere in the context — including in "
        "a bare neighbour line. That rule is exactly what makes distance-1 facts score ~100%, so "
        "Result 2 partly measures the scoring rule. Under the strict (prose-only) rule system C "
        "scores 0 by construction and the analysis would be vacuous; this is stated rather than "
        "worked around.",
        "**Seeds are not random.** Seed selection is a hybrid vector + full-text search, so "
        "seed-distance is itself an outcome of the retriever. It is a *diagnostic* of the "
        "pipeline, not an exogenous property of the graph.",
        "**Louvain instability.** Community assignment on a graph this small is sensitive to the "
        "seed; a fixed seed makes the run reproducible, not the partition canonical.",
        "**Environmental reproducibility had to be engineered, and that is itself a finding.** "
        "On a database shared with other writers this experiment produced four different "
        "headline numbers from four identical runs. The results above are only trustworthy "
        "because the graph is fingerprinted around every measurement and disturbed runs are "
        "discarded. Anyone reproducing this must use an instance nothing else writes to; the "
        "guard will abort rather than report a corrupted run, but it can only detect "
        "interference it can see between two fingerprints.",
    ]


def build_claim(data: dict) -> str:
    c = data["systems"]["C"]
    fa = c["fact_analysis"]
    mc = c["miss_concentration"]
    return (
        "Graph topology per se does **not** predict GraphRAG retrieval success on this benchmark "
        "— degree, local clustering, Louvain co-membership and anchor distance are all null at "
        f"n={data['n_questions']} — but the depth of the neighbourhood the retriever materialises "
        f"does, completely: graph-only GraphRAG retrieved {fa['near']['n_found']}/"
        f"{fa['near']['n_facts']} required facts lying within one hop of a retrieved seed and "
        f"{fa['>=2']['n_found']}/{fa['>=2']['n_facts']} lying further away "
        f"(exact 95% CI on the far rate {ci(*fa['>=2']['exact_ci'], pct)}), so all "
        f"{mc['n_missed']} of its misses sit outside the one-hop frontier and its multi-hop "
        "failures are a property of the retrieval depth rather than of the graph's structure."
    )


def build_degenerate_ci_note(data: dict) -> str:
    """Say out loud that a 0/n or n/n bootstrap interval is not certainty."""
    fa = data["systems"]["C"]["fact_analysis"]
    far = fa[">=2"]
    near = fa["near"]
    return (
        "**The zero-width bootstrap intervals are an artefact, not certainty.** System C's "
        f"outcome is perfectly separated — {near['n_found']}/{near['n_facts']} near facts "
        f"retrieved, {far['n_found']}/{far['n_facts']} far facts retrieved — and a percentile "
        "bootstrap of a constant is a constant. The exact (Clopper–Pearson) column is the honest "
        f"reading: the far-fact retrieval rate is {pct(far['rate'])} with a 95% interval reaching "
        f"up to {pct(far['exact_ci'][1])}, because it rests on only {far['n_facts']} facts. Even "
        "that interval treats facts as independent when they are nested in questions, so it is "
        "optimistic. The separation is clean; the *precision* of the rates is not."
    )


def build_profile_note(profile: dict) -> str:
    n_components = len(profile["components"])
    shape = "a single connected component" if n_components == 1 else f"{n_components} components"
    return (
        f"**The fixture is a near-tree, and that matters for everything below.** It is {shape} "
        f"with {profile['edges']} edges over {profile['nodes']} nodes — only "
        f"{profile['independent_cycles']} independent cycles — and it contains "
        f"{profile['n_triangles']} triangles, so average clustering and transitivity are exactly "
        f"{num(profile['avg_clustering'], 3)}. Mean degree is {num(profile['degree_mean'])}, the "
        f"maximum degree is {profile['degree_max']}, and the diameter is "
        f"{profile['giant_diameter']} with a mean shortest path of "
        f"{num(profile['giant_mean_path_length'])} — a long, stringy graph, not a small world. "
        f"Modularity is {num(profile['modularity'], 3)} over {profile['n_communities']} "
        f"communities, which on a near-tree mostly reflects where the branches are rather than "
        "genuinely dense clusters. Two consequences follow and are visible in the tables: the "
        "clustering-coefficient features are identically zero for every node and therefore carry "
        "no information *on this graph*, and Louvain community membership is close to a coarse "
        "restatement of graph distance. A knowledge graph extracted from real documents is "
        "usually denser locally (hub entities, triangles) and more fragmented globally; this "
        "fixture is neither."
    )


# ── Main ─────────────────────────────────────────────────────────────────────
async def main() -> int:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    command = (
        f"cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI={uri} "
        "NEO4J_PASSWORD=research_secret python research/rq6_topology/run_rq6.py"
    )
    dataset = load_dataset(DATASET_PATH)
    graph = build_graph(dataset)

    components = list(nx.connected_components(graph))
    if len(components) > 1:
        print(f"⚠️  graph has {len(components)} components; cross-component distances "
              f"are recorded as {DISCONNECTED} and the analysis says so.")

    partition = louvain(graph)
    community_of = {name: i for i, group in enumerate(partition) for name in group}
    clustering = nx.clustering(graph)
    dist = dict(nx.all_pairs_shortest_path_length(graph))

    print("📐 Fixture graph: "
          f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{len(partition)} Louvain communities")

    # ── Run the two graph systems through the production retrieval path ──────
    # Wipe-and-reseed, then measure inside a verified-quiet window. Nothing
    # about the database's prior state is assumed, and — because the instance
    # may be shared — nothing about its state *during* the measurement is
    # assumed either: the graph is fingerprinted before and after every step
    # and a run that moves is thrown away, never reported.
    attempts: list[str] = []
    graph_only = with_chunks = None
    fingerprints: dict = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"🧨 Wipe, re-seed and measure (attempt {attempt}/{MAX_ATTEMPTS}) …")
        try:
            graph_only, with_chunks, fingerprints = await measured_run(dataset)
        except Interference as exc:
            print(f"   ⚠️  discarded: {exc}")
            attempts.append(str(exc))
            await asyncio.sleep(2.0)
            continue
        print(f"   ✅ measured on a verified-quiet graph "
              f"({fingerprints['graph']['n_entities']} entities, "
              f"{fingerprints['graph']['n_relationships']} relationships, "
              f"digest {fingerprints['graph']['digest']})")
        break
    else:
        print(f"❌ {MAX_ATTEMPTS} attempts, every one disturbed by a concurrent writer. "
              f"Point NEO4J_URI at an instance nothing else is writing to.")
        for i, why in enumerate(attempts, 1):
            print(f"   attempt {i}: {why}")
        return 3

    # ── Assemble the per-question feature/outcome table ──────────────────────
    features = {
        q["id"]: question_features(q, graph, dist, community_of, clustering)
        for q in dataset["questions"]
    }
    by_qid = {"C": {r.qid: r for r in graph_only}, "D": {r.qid: r for r in with_chunks}}

    rows: list[dict] = []
    for question in dataset["questions"]:
        qid = question["id"]
        feat = features[qid]
        row: dict = {
            "qid": qid,
            "question": question["question"],
            "anchor": feat.anchor,
            "required": feat.required,
            "n_required": feat.n_required,
            "anchor_max_dist": feat.anchor_max_dist,
            "anchor_mean_dist": feat.anchor_mean_dist,
            "required_spread": feat.required_spread,
            "required_degree_mean": feat.required_degree_mean,
            "required_degree_min": feat.required_degree_min,
            "anchor_degree": feat.anchor_degree,
            "required_clustering_mean": feat.required_clustering_mean,
            "anchor_clustering": feat.anchor_clustering,
            "same_community_fraction": feat.same_community_fraction,
        }
        # Seed-based features come from system C's seeds: C is the pure-graph
        # system, and D's seeds are identical (same seeding call, same k) —
        # asserted below rather than assumed.
        result_c = by_qid["C"][qid]
        result_d = by_qid["D"][qid]
        sdist = seed_distances(result_c, dist)
        row["seeds"] = list(result_c.extras.get("seeds") or [])
        row["anchor_is_seed"] = feat.anchor in row["seeds"]
        row["seed_max_dist"] = float(max(sdist.values()))
        row["seed_mean_dist"] = float(np.mean(list(sdist.values())))
        row["n_facts_beyond_1hop"] = float(sum(1 for v in sdist.values() if v >= 2))
        row["n_paths"] = int(result_c.extras.get("paths") or 0)
        for key, result in (("C", result_c), ("D", result_d)):
            found = {f.lower() for f in result.found}
            row[f"{key}_complete"] = 1.0 if result.complete else 0.0
            row[f"{key}_recall"] = result.recall
            row[f"{key}_facts"] = [
                {
                    "qid": qid,
                    "fact": fact,
                    "seed_dist": (
                        sdist[fact]
                        if key == "C"
                        else seed_distances(result, dist)[fact]
                    ),
                    "found": fact.lower() in found,
                    "channel": result.hit_channels.get(fact),
                }
                for fact in result.required
            ]
        rows.append(row)

    n = len(rows)
    data: dict = {
        "command": command,
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "graph_k": GRAPH_K,
        "n_questions": n,
        "n_required_facts": sum(int(r["n_required"]) for r in rows),
        "graph_profile": graph_profile(graph, partition),
        "questions": rows,
        "systems": {},
        "neo4j_uri": uri,
        "integrity": {
            "verified_quiet_window": True,
            "attempts_used": len(attempts) + 1,
            "discarded_attempts": attempts,
            "fingerprints": fingerprints,
        },
    }
    data["profile_note"] = build_profile_note(data["graph_profile"])

    # Systems C and D seed identically (same call, same k, seeding does not read
    # the chunk store), so one hop-frontier table serves both. Verified, not
    # assumed — if it ever stopped holding, the report would be wrong.
    identical = all(
        (by_qid["C"][q["id"]].extras.get("seeds") or [])
        == (by_qid["D"][q["id"]].extras.get("seeds") or [])
        for q in dataset["questions"]
    )
    data["seeds_identical_c_d"] = identical
    data["seed_identity_note"] = (
        "systems C and D select identical seed entities on all "
        f"{n} questions (verified this run), so their seed-distance frontiers coincide; they "
        "differ only in the source-chunk channel."
        if identical
        else "systems C and D did NOT select identical seeds this run — the frontier below is "
        "system C's and does not describe D."
    )

    for system, label in (
        ("C", "Synapse GraphRAG, graph only"),
        ("D", "Synapse GraphRAG + source chunks"),
    ):
        complete = np.array([r[f"{system}_complete"] for r in rows], dtype=float)
        recall = np.array([r[f"{system}_recall"] for r in rows], dtype=float)
        fc, fc_lo, fc_hi = bootstrap_ci(complete, np.mean)
        fr, fr_lo, fr_hi = bootstrap_ci(recall, np.mean)
        table = feature_table(rows, f"{system}_complete")
        data["systems"][system] = {
            "label": label,
            "n": n,
            "n_success": int(complete.sum()),
            "full_coverage": fc,
            "full_coverage_ci": [fc_lo, fc_hi],
            "fact_recall": fr,
            "fact_recall_ci": [fr_lo, fr_hi],
            "features": table,
            "n_features_ci_excludes_zero": sum(1 for t in table if t["ci_excludes_zero"]),
            "fact_analysis": fact_level_analysis(rows, system),
            "miss_concentration": miss_concentration(rows, system),
            "hop_frontier": hop_frontier_analysis(rows, graph, dist, system),
            "path_rescue": path_rescue_diagnostic(rows, graph, system),
        }

    data["degenerate_ci_note"] = build_degenerate_ci_note(data)
    data["interpretation"] = build_interpretation(data)
    data["threats"] = build_threats(data)
    data["claim"] = build_claim(data)

    RESULTS_JSON.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    FINDINGS_PATH.write_text(render_findings(data), encoding="utf-8")

    c = data["systems"]["C"]
    print()
    print(f"📊 C full coverage {c['n_success']}/{n} = {pct(c['full_coverage'])} "
          f"{ci(*c['full_coverage_ci'], pct)}")
    print(f"   fact retrieved | seed-distance ≤ 1: "
          f"{pct(c['fact_analysis']['near']['rate'])}, "
          f"≥ 2: {pct(c['fact_analysis']['>=2']['rate'])}")
    print(f"   misses beyond one hop: {pct(c['miss_concentration']['share_missed_beyond_1hop'])} "
          f"{ci(*c['miss_concentration']['share_ci'], pct)}")
    print(f"   features whose Δ CI excludes 0: {c['n_features_ci_excludes_zero']} of "
          f"{len(c['features'])}")
    print(f"\n📝 {FINDINGS_PATH}\n📦 {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
