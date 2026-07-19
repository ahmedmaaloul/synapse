# GraphRAG vs. vector RAG — benchmark results

`Synapse Multi-Hop Benchmark v1` — 34 passages, 79 entities, 81 relationships, 14 multi-hop questions.

Every system below uses the same embedding model, and every system is scored with the same **two** rules. No LLM is involved — this measures **retrieval**, which is the part a knowledge graph actually changes. Regenerate with `make benchmark`; this file is written by the harness, never edited by hand.

**Why two rules.** Applying an identical rule to every system is not the same as applying a neutral one. The permissive rule credits a required fact whose name appears anywhere in the retrieved context — and a GraphRAG context contains entity names as structured scaffolding (`Entity:` headers, `Description:` lines, `→ REL → Neighbour` lines, rendered reasoning paths), so a fact can score there because the graph *listed it as a neighbour*, with none of the evidence needed to answer retrieved. A passage baseline has no such channel. The strict rule closes it: a fact counts only when its name appears in retrieved prose. Both are reported, and the size of the gap between them is measured per system below.

**D is the shipped system**: the graph *and* the source chunks the entities were extracted from. **C is what Synapse was before source chunks** — graph only. **B′ is the ablation**: C's own context with every edge-derived part deleted (the per-entity `Relationships:` lines and the rendered `Reasoning paths:` section), so C − B′ is what the relationships contribute with the seed set held fixed — it controls the *seed set*, not the character budget, and B′ therefore reads less text than C. **B″ and A′** are the graph-free systems given the same character budget as C and D respectively, because a comparison between unequal budgets is not a comparison.

One thing to notice before reading D's row: **D's semantic excerpt channel is system A**. It runs a top-k cosine search over exactly the passages A ranks, with the same embedder, inside `chunk_store.search_chunks`. D scoring above A is therefore close to arithmetic, not evidence. The measured overlap is reported below so the effect can be sized.

**Reproduce:** `docker compose up -d neo4j && make benchmark` — no API key, no network. Pinned configuration for the numbers below: embeddings `fastembed` · seeds k=8 · `retrieval_max_hops=2` · `chunk_retrieval_enabled=True` · `chunk_top_k=4` · `chunk_context_max_chars=4000`

| System | Fact Recall | Full-Coverage | Fact Recall (strict) | Full-Coverage (strict) | Mean context (chars) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Passage vector RAG (top-4) | 89.3% | 71.4% | 89.3% | 71.4% | 831 |
| A′. Passage vector RAG (top-20, budget-matched to D) | 100.0% | 100.0% | 100.0% | 100.0% | 4,072 |
| B. Entity vector RAG, no edges (top-8) | 48.1% | 14.3% | 0.0% | 0.0% | 916 |
| B″. Entity vector RAG (top-20, budget-matched to C) | 63.0% | 21.4% | 0.0% | 0.0% | 2,246 |
| B′. GraphRAG seeds, graph stripped (k=8) | 48.1% | 7.1% | 0.0% | 0.0% | 899 |
| C. Synapse GraphRAG, graph only (k=8) | 87.5% | 64.3% | 0.0% | 0.0% | 2,057 |
| **D. Synapse GraphRAG + source chunks (k=8)** | **94.6%** | **85.7%** | **91.1%** | **78.6%** | 3,516 |

*Permissive* credits a required fact whose name appears anywhere in the retrieved context; *strict* credits it only when the name appears in retrieved prose (verbatim corpus text). The entity-granular rows (B, B″, B′) and the graph-only row (C) score 0.0 under the strict rule **by construction** — they return a compressed representation of the corpus, never the corpus. Treat permissive as an upper bound and strict as a lower bound.

- HEADLINE (computed, not asserted) — permissive rule: the fact's name appears anywhere in the context: the shipped system scores Fact Recall 94.6%, Full-Coverage Rate 85.7% on 3,516 chars of retrieved context.
  - A plain passage baseline matches it on EVERY metric while reading LESS text (Fact Recall at top-8 using 48% of the context; Full-Coverage Rate at top-6 using 36% of the context). On this corpus the graph buys no context saving; what it buys has to be argued from the ablation row below, not from these numbers.

- HEADLINE (computed, not asserted) — strict rule: the fact's name appears in retrieved prose: the shipped system scores Fact Recall 91.1%, Full-Coverage Rate 78.6% on 3,516 chars of retrieved context.
  - A plain passage baseline matches it on EVERY metric while reading LESS text (Fact Recall at top-5 using 30% of the context; Full-Coverage Rate at top-5 using 30% of the context). On this corpus the graph buys no context saving; what it buys has to be argued from the ablation row below, not from these numbers.

## How much the permissive rule is worth

- Scoring-rule sensitivity — how many points each system loses when the rule tightens from 'the name is anywhere in the context' to 'the name is in retrieved prose'. This is the size of the advantage the permissive rule hands out, per system, and it is not the same for all of them:
  - A. Passage vector RAG (top-4): Fact Recall −0.0 pp, Full-Coverage Rate −0.0 pp (unaffected — it returns nothing but prose).
  - A′. Passage vector RAG (top-20, budget-matched to D): Fact Recall −0.0 pp, Full-Coverage Rate −0.0 pp (unaffected — it returns nothing but prose).
  - B. Entity vector RAG, no edges (top-8): Fact Recall −48.1 pp, Full-Coverage Rate −14.3 pp (favoured by the permissive rule).
  - B″. Entity vector RAG (top-20, budget-matched to C): Fact Recall −63.0 pp, Full-Coverage Rate −21.4 pp (favoured by the permissive rule).
  - B′. GraphRAG seeds, graph stripped (k=8): Fact Recall −48.1 pp, Full-Coverage Rate −7.1 pp (favoured by the permissive rule).
  - C. Synapse GraphRAG, graph only (k=8): Fact Recall −87.5 pp, Full-Coverage Rate −64.3 pp (favoured by the permissive rule).
  - D. Synapse GraphRAG + source chunks (k=8): Fact Recall −3.6 pp, Full-Coverage Rate −7.1 pp (favoured by the permissive rule).
  - → The passage rows lose nothing because every character they return is corpus prose. The entity-granular and graph-only rows lose everything they had, because they return a representation of the corpus and never the corpus. Read the permissive rule as an upper bound and the strict rule as a lower bound; the true answerability of a context is between them and this benchmark does not measure it.

- Where the scored hits come from — the permissive rule credits a fact whose name appears anywhere in the context, and the systems put different kinds of string in front of it. Attribution is conservative against the graph: a fact present in both prose and scaffolding is credited to the prose.
  - A. Passage vector RAG (top-4): 100% of its 48 scored hits came from retrieved prose, 0% from graph scaffolding (0% an entity block, 0% an edge or reasoning path and nothing else).
  - B. Entity vector RAG, no edges (top-8): 0% of its 25 scored hits came from retrieved prose, 100% from graph scaffolding (100% an entity block, 0% an edge or reasoning path and nothing else).
  - B′. GraphRAG seeds, graph stripped (k=8): 0% of its 25 scored hits came from retrieved prose, 100% from graph scaffolding (100% an entity block, 0% an edge or reasoning path and nothing else).
  - C. Synapse GraphRAG, graph only (k=8): 0% of its 47 scored hits came from retrieved prose, 100% from graph scaffolding (53% an entity block, 47% an edge or reasoning path and nothing else).
  - D. Synapse GraphRAG + source chunks (k=8): 96% of its 51 scored hits came from retrieved prose, 4% from graph scaffolding (0% an entity block, 4% an edge or reasoning path and nothing else).

A′ and B″ are read off the k sweeps as aggregates, so they carry no per-question attribution; A′ is pure corpus prose and B″ pure entity records, which is why neither appears in the table above.

## What the numbers support

**Per-metric detail, permissive rule (D vs. the passage baseline)**

- Fact Recall: TOO CLOSE TO CALL — 94.6% vs 89.3% (+5.4 pp for GraphRAG, nominally GraphRAG). The gap is smaller than one question out of 14 (7.1 pp), so it is not a meaningful lead. No significance is claimed.
  - → A plain passage baseline beats it at top-8 (96.4%), reading 1,672 chars = 48% of GraphRAG's context.
- Full-Coverage Rate: GraphRAG ahead at the configured k — 85.7% vs 71.4% (+14.3 pp, > one question = 7.1 pp).
  - → A plain passage baseline matches it at top-6 (85.7%), reading 1,260 chars = 36% of GraphRAG's context.
- Context size: GraphRAG hands the generator 4.23x the characters the baseline does (3,516 vs 831).
- Equal-context check: top-20 is the first baseline setting that reads at least as much as GraphRAG (4,072 vs 3,516 chars); it scores 100.0% recall / 100.0% coverage.
  - → At that budget the baseline matches or beats GraphRAG. Caveat in both directions: 20 passages is a large share of a corpus this small, so top-k saturates here in a way it would not on a real one — but that is a limitation of the benchmark, not a point in GraphRAG's favour.
- Scale caveat (computed): the entire corpus is 6,750 characters. The shipped system's mean context is 52% of it.
  - The budget-matched baseline A′ reads 60% of the corpus (20 of 34 passages), which is why it scores what it scores. On a corpus this size 'budget-matched' and 'reads nearly everything' are the same row; treat A′ as an upper bound on what top-k can do here, not as a deployable configuration.

**Per-metric detail, strict rule (D vs. the passage baseline)**

- Fact Recall: TOO CLOSE TO CALL — 91.1% vs 89.3% (+1.8 pp for GraphRAG, nominally GraphRAG). The gap is smaller than one question out of 14 (7.1 pp), so it is not a meaningful lead. No significance is claimed.
  - → A plain passage baseline matches it at top-5 (91.1%), reading 1,045 chars = 30% of GraphRAG's context.
- Full-Coverage Rate: TOO CLOSE TO CALL — 78.6% vs 71.4% (+7.1 pp for GraphRAG, nominally GraphRAG). The gap is smaller than one question out of 14 (7.1 pp), so it is not a meaningful lead. No significance is claimed.
  - → A plain passage baseline matches it at top-5 (78.6%), reading 1,045 chars = 30% of GraphRAG's context.
- Context size: GraphRAG hands the generator 4.23x the characters the baseline does (3,516 vs 831).
- Equal-context check: top-20 is the first baseline setting that reads at least as much as GraphRAG (4,072 vs 3,516 chars); it scores 100.0% recall / 100.0% coverage.
  - → At that budget the baseline matches or beats GraphRAG. Caveat in both directions: 20 passages is a large share of a corpus this small, so top-k saturates here in a way it would not on a real one — but that is a limitation of the benchmark, not a point in GraphRAG's favour.

**What the source chunks add (C → D)**

- Source chunks — what the prose adds to the graph (C → D):
  - Fact Recall: 87.5% graph-only → 94.6% with source chunks (+7.1 pp).
  - Full-Coverage Rate: 64.3% graph-only → 85.7% with source chunks (+21.4 pp).
  - Cost of the excerpts: 1.71x the context (3,516 vs 2,057 chars).
  - Under the strict rule the whole of D's score is the chunk channel: C scores 0.0% / 0.0% (zero by construction — no prose) and D scores 91.1% / 78.6%. Every fact D makes answerable from verbatim text, it makes answerable through the excerpts.
  - Overlap with the baseline, measured: 100% of the passages system A retrieves at top-4 also appear among D's excerpts, and D returns 5.7 excerpts per question of which 1.7 are passages A did not retrieve. D's semantic excerpt channel IS the passage baseline, run inside Synapse; only the structural channel (chunks reached through the entities the graph selected) is something plain vector RAG cannot do.

**Isolating the graph (B′ vs. C — seed-matched, not budget-matched; budget-matched by B″)**

- Graph ablation — B′ is GraphRAG's own k=8 context with every edge-derived part deleted (the per-entity Relationships lines AND the rendered Reasoning paths section), so the relationships are the only difference between B′ and C:
  - Fact Recall: 48.1% without the graph → 87.5% with it (+39.4 pp).
  - Full-Coverage Rate: 7.1% without the graph → 64.3% with it (+57.1 pp).
  - Cost of the graph: 2.29x the context (2,057 vs 899 chars).
  - NOT budget-matched: B′ is C minus text, so it reads 899 chars against C's 2,057. C − B′ controls the *seed set*, not the budget; the budget-matched edgeless row is B″, below. Any reading of this ablation as 'the graph wins at equal context' is unsupported.
  - Permissive rule only: B′ and C return no prose, so under the strict rule both score 0.0 and the difference between them is undefined. The graph's contribution is measurable here only because the permissive rule credits names in scaffolding.
  - For reference, a standalone entity vector RAG at top-8 (its own cosine ranking instead of Synapse's hybrid seeding) scores 48.1% / 14.3% on 916 chars.
  - Budget check (B″): top-20 is the first edgeless entity setting that reads at least as much as C (2,246 vs 2,057 chars); at that budget it scores 63.0% / 21.4%.
  - → At equal budget the edgeless system is still behind on both metrics.
  - → Fact Recall: the edgeless system needs top-48 — 48 of the 79 entities in the graph, 5,282 chars = 2.6x C's context — before it reaches C's 87.5%.
  - → Full-Coverage Rate: the edgeless system needs top-48 — 48 of the 79 entities in the graph, 5,282 chars = 2.6x C's context — before it reaches C's 64.3%.
  - → Read as a claim: the relationships buy retrieval *width*. C gets there from 8 ranked seeds; the same entity descriptions without edges have to be read several times wider to get there — and at k=79 the edgeless system has read every description in the graph (8,570 chars), which is the trivial 100% any system reaches by retrieving everything.

**Where retrieval still fails**

- D (GraphRAG + source chunks) missed at least one required fact on 2 of 14 questions (q02, q11). The facts it never retrieved: GraphRAG, Jacquard Loom, Knowledge Graph.
- C (graph only) missed at least one required fact on 5 of 14 questions (q02, q06, q08, q11, q12). The facts it never retrieved: AlexNet, Attention Is All You Need, Backpropagation, Geoffrey Hinton, GraphRAG, Jacquard Loom, Knowledge Graph.
- A (passage baseline, top-4) missed at least one required fact on 4 of 14 questions (q01, q02, q03, q11). The facts it never retrieved: GraphRAG, Herman Hollerith, Jacquard Loom, Knowledge Graph, Punched Cards.
- D under the strict rule missed at least one required fact on 3 of 14 questions (q01, q02, q11). The facts it never retrieved: GraphRAG, Herman Hollerith, Jacquard Loom, Knowledge Graph, Punched Cards.

## k sweeps

How the two graph-free systems behave as they are allowed to retrieve more. Both sweeps run to the point where the system has read **everything it has** — all 34 passages, all 79 entity records — so a budget match is always reachable and no comparison in this report is left un-matched. A corpus this small saturates, and the `Share` column is printed so that is visible rather than implied: once a row is reading a large fraction of everything that exists, its score says more about the corpus than about the retriever.

### A. Passage vector RAG

| top-k | Fact recall | Full coverage | Mean chars | Share of corpus |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 59.0% | 0.0% | 199 | 3% |
| 2 | 78.0% | 42.9% | 413 | 6% |
| 3 | 83.9% | 57.1% | 627 | 9% |
| 4 | 89.3% | 71.4% | 831 | 12% |
| 5 | 91.1% | 78.6% | 1,045 | 15% |
| 6 | 92.9% | 85.7% | 1,260 | 18% |
| 8 | 96.4% | 92.9% | 1,672 | 24% |
| 10 | 96.4% | 92.9% | 2,068 | 29% |
| 12 | 96.4% | 92.9% | 2,470 | 35% |
| 16 | 100.0% | 100.0% | 3,253 | 47% |
| 20 | 100.0% | 100.0% | 4,072 | 59% |
| 24 | 100.0% | 100.0% | 4,840 | 71% |
| 32 | 100.0% | 100.0% | 6,433 | 94% |
| 34 | 100.0% | 100.0% | 6,816 | 100% |

Every character this system returns is corpus prose, so its strict-rule scores are identical to the permissive ones printed above and are not repeated.

### B. Entity vector RAG (no edges)

| top-k | Fact recall | Full coverage | Mean chars | Share of entities |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.3% | 0.0% | 112 | 1% |
| 2 | 26.4% | 0.0% | 225 | 3% |
| 3 | 35.4% | 7.1% | 341 | 4% |
| 4 | 37.1% | 7.1% | 458 | 5% |
| 5 | 40.7% | 7.1% | 570 | 6% |
| 6 | 42.1% | 7.1% | 682 | 8% |
| 8 | 48.1% | 14.3% | 916 | 10% |
| 10 | 49.9% | 14.3% | 1,142 | 13% |
| 12 | 55.2% | 21.4% | 1,366 | 15% |
| 16 | 58.8% | 21.4% | 1,811 | 20% |
| 20 | 63.0% | 21.4% | 2,246 | 25% |
| 24 | 66.5% | 35.7% | 2,688 | 30% |
| 32 | 74.8% | 35.7% | 3,566 | 41% |
| 48 | 91.1% | 78.6% | 5,282 | 61% |
| 64 | 96.4% | 92.9% | 6,982 | 81% |
| 79 | 100.0% | 100.0% | 8,570 | 100% |

This system returns no prose at any k, so every row above scores 0.0% under the strict rule — including the k that reads every entity in the graph.

## Per-question breakdown

Permissive-rule recall per system, then D's recall under the strict rule.

| # | Question | Required facts | A. passage | B. entity | B′. no graph | C. graph | D. graph+chunks | D strict | D missed (permissive) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q01 | Who founded the business that eventually became the maker of Deep Blue? | Deep Blue, IBM, Tabulating Machine Company, Herman Hollerith | 75% | 75% | 75% | 100% | 100% | 75% | — |
| q02 | What fed instructions into the machine Ada Lovelace wrote her notes about, and where did that idea come from? | Ada Lovelace, Analytical Engine, Punched Cards, Jacquard Loom | 50% | 50% | 50% | 75% | 75% | 50% | Jacquard Loom |
| q03 | Whose statistical work borrowed the input medium of the machine Charles Babbage designed? | Charles Babbage, Analytical Engine, Punched Cards, Herman Hollerith | 75% | 50% | 50% | 100% | 100% | 100% | — |
| q04 | Who supervised the AlexNet team, and what did he popularise in 1986? | AlexNet, Geoffrey Hinton, Backpropagation | 100% | 33% | 67% | 100% | 100% | 100% | — |
| q05 | Where was the dataset that AlexNet won on assembled, and by whom? | AlexNet, ImageNet, Fei-Fei Li, Stanford University | 100% | 50% | 50% | 100% | 100% | 100% | — |
| q06 | Which vision network was co-built by the first chief scientist of the laboratory that released GPT? | GPT, OpenAI, Ilya Sutskever, AlexNet | 100% | 0% | 25% | 75% | 100% | 100% | — |
| q07 | What mechanism replaced recurrence inside the architecture BERT is built from? | BERT, Transformer, Self-Attention | 100% | 100% | 67% | 100% | 100% | 100% | — |
| q08 | Which algorithm overcame the limits of the machine built at the Cornell Aeronautical Laboratory, and who popularised it? | Cornell Aeronautical Laboratory, Perceptron, Backpropagation, Geoffrey Hinton | 100% | 25% | 25% | 50% | 100% | 100% | — |
| q09 | Which thought experiment was proposed by the man who helped break Enigma? | Enigma, Alan Turing, Turing Test | 100% | 100% | 100% | 100% | 100% | 100% | — |
| q10 | Which retrieval method walks the structure held in the database that Cypher queries? | Cypher, Neo4j, Knowledge Graph, GraphRAG | 100% | 75% | 50% | 100% | 100% | 100% | — |
| q11 | Which technique extends the method that reduces hallucination, and what does it walk over? | Hallucination, Retrieval-Augmented Generation, GraphRAG, Knowledge Graph | 50% | 25% | 25% | 50% | 50% | 50% | GraphRAG, Knowledge Graph |
| q12 | Which paper introduced the architecture behind BERT, and which company employed its authors? | BERT, Transformer, Attention Is All You Need, Google | 100% | 25% | 25% | 75% | 100% | 100% | — |
| q13 | Which network won the contest that measured progress in Computer Vision, and on what hardware was it trained? | Computer Vision, ImageNet, AlexNet, GPU | 100% | 25% | 25% | 100% | 100% | 100% | — |
| q14 | The creator of Lisp organised a meeting; which early neural network was attacked by another of its attendees? | Lisp, John McCarthy, Dartmouth Workshop, Marvin Minsky, Perceptron | 100% | 40% | 40% | 100% | 100% | 100% | — |

---

Generated by `backend/benchmarks/run_benchmark.py` — Synapse, © 2026 Ahmed Maaloul, AGPL-3.0-or-later.
