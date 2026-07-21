# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Retrieval Evaluation Harness

Measures how well hybrid GraphRAG retrieval surfaces the *right* entities for a
question. Seeds a fixture graph into Neo4j, runs ``retrieve_subgraph`` for each
query, and reports Hit@1, Recall@k, Precision@k, and MRR.

Why it exists: shipping a RAG system without measuring retrieval quality is
flying blind. This harness turns "it feels grounded" into numbers you can track.

Run it (needs a live Neo4j; uses real embeddings by default):

    docker compose up -d neo4j
    EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7687 \\
        python -m eval.run_eval

Writes a Markdown summary to eval/results.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.neo4j_driver import close_driver, execute_query
from app.services.chat_engine import retrieve_subgraph
from app.services.graph_builder import (
    _embed_entities,
    _write_entities,
    _write_relationships,
)
from app.services.graph_schema import ensure_schema

DATA = Path(__file__).parent / "dataset.json"
K = 8


def _metrics(retrieved: list[str], expected: list[str]) -> dict:
    exp = set(expected)
    hits = [r for r in retrieved if r in exp]
    # Rank of first relevant result (1-indexed) for MRR.
    rank = next((i + 1 for i, r in enumerate(retrieved) if r in exp), None)
    return {
        "hit@1": 1.0 if retrieved[:1] and retrieved[0] in exp else 0.0,
        "recall@k": len(set(hits)) / len(exp) if exp else 0.0,
        "precision@k": len(hits) / len(retrieved) if retrieved else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
    }


async def main() -> None:
    dataset = json.loads(DATA.read_text())

    print("⏳ Seeding fixture graph …")
    await execute_query("MATCH (n) DETACH DELETE n")
    await ensure_schema()
    entities = dataset["entities"]
    embeddings = await _embed_entities(entities)
    await _write_entities(entities, "eval-fixture", embeddings)
    await _write_relationships(dataset["relationships"])

    rows = []
    agg = {"hit@1": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0}
    queries = dataset["queries"]

    print(f"🔎 Running {len(queries)} retrieval queries (k={K}) …\n")
    for q in queries:
        _, citations = await retrieve_subgraph(q["question"], k=K)
        retrieved = [c["name"] for c in citations]
        m = _metrics(retrieved, q["expected"])
        for key in agg:
            agg[key] += m[key]
        rows.append((q["question"], q["expected"], retrieved[:5], m))

    n = len(queries)
    for key in agg:
        agg[key] /= n

    _print_table(rows, agg)
    _write_results(rows, agg)
    await close_driver()


def _print_table(rows, agg) -> None:
    for question, expected, retrieved, m in rows:
        ok = "✅" if m["recall@k"] > 0 else "❌"
        print(f"{ok} {question}")
        print(f"    expected : {expected}")
        print(f"    top-5    : {retrieved}")
        print(f"    hit@1={m['hit@1']:.0f} recall={m['recall@k']:.2f} mrr={m['mrr']:.2f}\n")

    print("── Aggregate ─────────────────────────────")
    print(f"  Hit@1       : {agg['hit@1']:.2%}")
    print(f"  Recall@{K}    : {agg['recall@k']:.2%}")
    print(f"  Precision@{K} : {agg['precision@k']:.2%}")
    print(f"  MRR         : {agg['mrr']:.3f}")


def _write_results(rows, agg) -> None:
    lines = [
        "# Retrieval Evaluation Results",
        "",
        "Hybrid GraphRAG retrieval (vector + full-text seeds → graph expansion) on "
        "the fixture in `eval/dataset.json`. Queries are paraphrased to avoid lexical "
        "overlap with entity names, so keyword-only search would miss most of them.",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Hit@1 | {agg['hit@1']:.0%} |",
        f"| Recall@{K} | {agg['recall@k']:.0%} |",
        f"| Precision@{K} | {agg['precision@k']:.0%} |",
        f"| MRR | {agg['mrr']:.3f} |",
        "",
        "<details><summary>Per-query breakdown</summary>",
        "",
        "| Question | Expected | Retrieved (top-5) | Recall |",
        "| --- | --- | --- | --- |",
    ]
    for question, expected, retrieved, m in rows:
        lines.append(
            f"| {question} | {', '.join(expected)} | {', '.join(retrieved) or '—'} | {m['recall@k']:.0%} |"
        )
    lines += ["", "</details>", ""]
    (Path(__file__).parent / "results.md").write_text("\n".join(lines))
    print("\n📄 Wrote eval/results.md")


if __name__ == "__main__":
    asyncio.run(main())
