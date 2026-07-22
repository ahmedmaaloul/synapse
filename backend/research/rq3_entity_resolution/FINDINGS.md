# RQ3 — How does entity-resolution aggressiveness affect retrieval?

**Reproduce:** `NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret EMBEDDING_PROVIDER=fastembed python -m research.rq3_entity_resolution.run_rq3`

Corpus: `benchmarks/dataset.json` — 34 passages, 79 entities, 81 relationships, 14 multi-hop questions. Embeddings `fastembed`, seeds k=8, `retrieval_max_hops=2`, `chunk_top_k=4`. No LLM is called anywhere in this experiment. Every number below is generated into this file by `run_rq3.py`; none is typed by hand.

---

## 1. Hypothesis and prediction

**Hypothesis.** Entity resolution has a precision/recall sweet spot. Too conservative leaves duplicate nodes that fragment each entity's neighbourhood and split its evidence; too aggressive merges distinct entities, collapsing the graph and creating false paths. There should be an interior optimum and a catastrophic-collapse regime.

**Prediction, registered before the run.** Downstream fact recall over the (`entity_resolution_threshold` × `entity_resolution_name_threshold`) plane is an inverted U: rising as merging repairs fragmentation, peaking strictly inside the grid, then falling off a cliff.

## 2. Method

Two corpora are swept over the same 15 × 14 = 210 threshold grid.

**CLEAN** is `benchmarks/dataset.json` exactly as it ships. Its entity list was hand-curated, so it contains no duplicates and *no merge is ever correct*. This corpus measures one thing: how much head-room the shipped defaults have before they start doing damage.

**NOISY** injects the surface-form noise a real LLM extractor produces. 36 of the 79 entities (every entity of degree >= 2 that admits a variant) gain one duplicate node: 9 are a PERSON written by surname alone ("Geoffrey Hinton" -> "Hinton"), 27 are a one-character deletion inside the name ("Backpropagation" -> "Backproagation"). Every second incident edge of the original entity, and every second chunk link, is re-pointed onto the duplicate, so the entity's neighbourhood and its source evidence are genuinely split in two. Half the duplicates inherit the source entity's description (a well-described re-extraction), half carry none (a bare mention) — the single design choice that spreads the duplicate pairs across the embedding-similarity axis rather than parking them all at cosine ~1. On top of that, 11 **adversarial decoys** are added: genuinely distinct entities that share surface form with something already in the graph, four of them being different people who share a surname with a real person. Each decoy carries its own edge, so a wrong merge does not merely lose a node — it grafts a foreign edge onto a real entity.

Two properties are asserted at construction time, because without them every number downstream would be junk: every injected name is strictly shorter than the entity it duplicates (so `choose_canonical` always restores the original name on a correct merge), and no injected name or description contains a required fact as a substring (so the injection can never manufacture a scoring hit).

For each cell the script runs the **production** resolver (`entity_resolution.find_duplicate_clusters`), the **production** ingest collapse (`graph_builder._collapse_duplicate_entities`), the **production** write path, and the **production** retrieval path (`chat_engine.retrieve_subgraph`), scored by `benchmarks/run_benchmark.py`'s own rules so the numbers are comparable with the shipped benchmark report. Systems are named as in that report: **C** = graph only, **D** = graph + source chunks (the shipped system). The permissive rule credits a required fact whose name appears anywhere in the retrieved context; the strict rule credits it only in retrieved prose. C scores 0.0 under the strict rule by construction (it returns no prose), so C is reported permissive-only and D under both.

Cells whose resolver output is identical share one Neo4j evaluation — the downstream result is a deterministic function of the merge set. The 210 cells collapse to 62 distinct resolver outcomes on NOISY and 14 on CLEAN.

Uncertainty: n = 14 questions. Every headline number carries a percentile bootstrap 95% CI over questions (10,000 resamples, seed 20260719); every comparison between two cells is a *paired* bootstrap on the per-question differences, because both cells answered the same questions.

## 3. The decision surface, measured

Before any retrieval number: what do the two thresholds actually have to separate? Below, every injected duplicate pair (`T`) and every same-type pair that must **not** merge (`x`) is plotted at its true coordinates. Pairs of different types are omitted — the type barrier blocks those at every threshold.

```
  cosine
   1.00 ┤                                                  │     #     
   0.97 ┤                                                  │    T ###  
   0.94 ┤──────────────────────────────────────────────────┼────TT#────
   0.91 ┤                                                  │           
   0.88 ┤                                  x               │        #  
   0.85 ┤                           x                      │           
   0.82 ┤                      x    x    x                 │      T    
   0.79 ┤                   x   x  x  x  x x               │      xTT  
   0.76 ┤                        xx            x           │    T  T   
   0.73 ┤                 x                                │      #T   
   0.70 ┤                 x  x             x               │       #   
   0.67 ┤                 #           #  x                 │    T #    
   0.63 ┤                   xx    x              x         │    T #    
   0.60 ┤                 x  x    x x                      │           
   0.57 ┤                 x xx   x                         │           
   0.54 ┤                 x      x   x                     │           
   0.51 ┤                 x   x  x                         │    T      
   0.48 ┤                                                  │           
   0.45 ┤                                                  │           
   0.42 ┤                                                  │           
   0.39 ┤                                                  │           
   0.36 ┤                                                  │           
   0.33 ┤                                                  │           
   0.30 ┤                                                  │           
        └──────────────────────────────────────────────────────────────
         0.30                                            1.00  name similarity

   T = must merge (injected duplicate)   x = must NOT merge (same-type pair)
   # = both   ─│ = the shipped defaults (cosine 0.93, name 0.87)
```

The 36 duplicate pairs span cosine 0.506 to 0.996 but are squeezed into name similarity 0.933 to 0.982. The worst adversarial same-type pair reaches cosine 0.866 and name similarity 0.950.

**The two gates are not symmetric, and this is the mechanism behind everything below.** The lexical gate is very nearly a binary switch: `name_similarity` pins every prefix / token-subsequence pair at exactly `CONTAINMENT_SCORE` = 0.95, and a one-character typo in an *n*-character name scores 1 − 1/*n*, so realistic duplicates all land in a narrow band just above 0.93. The embedding gate is the only genuinely continuous knob.

**Perfect resolution is unreachable at any threshold.** No cell of the grid merges every true pair without merging a false one; the best fully-precise setting is 0.80 / 0.93, which recovers 20 of 36 duplicates (55.6% of them). That is not a defect of this resolver's constants: it is forced by the geometry above, where 4 decoy pairs sit at the *same* lexical score as the true surname duplicates and are separated only by cosine.

The four hardest adversarial pairs — different people, same surname, identical lexical score to a true duplicate:

| pair | name similarity | cosine |
| --- | ---: | ---: |
| `Krizhevsky` / `Tomas Krizhevsky` | 0.950 | 0.788 |
| `Peter Sutskever` / `Sutskever` | 0.950 | 0.714 |
| `Hinton` / `Simon Hinton` | 0.950 | 0.659 |
| `Minsky` / `Rachel Minsky` | 0.950 | 0.647 |

## 4. Results — CLEAN corpus: the knob does nothing until it does damage

On the shipped corpus the resolver merges **nothing** at 182 of the 210 grid cells, including the shipped defaults. Retrieval is therefore literally identical across all of them: graph-only fact recall 87.5%, full coverage 64.3%; D (graph + chunks) 94.6% / 85.7% permissive and 91.1% strict. Across the whole grid, graph-only fact recall takes exactly 12 distinct value(s): 46.9%, 47.7%, 62.6%, 66.4%, 70.0%, 74.2%, 75.0%, 77.1%, 81.0%, 83.7%, 85.1%, 87.5%.

**Nodes absorbed, CLEAN corpus** (rows = cosine threshold, columns = name threshold; ★ marks the shipped default):

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.97 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.95 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.93&nbsp;★ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.87 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| 0.75 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 5 | 8 |
| 0.70 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 11 | 17 |
| 0.60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 5 | 23 | 38 |
| 0.50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 28 | 45 |
| 0.30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 28 | 45 |
| 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 28 | 45 |

**Graph-only fact recall (C, permissive), CLEAN corpus:**

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.99 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.97 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.95 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.93&nbsp;★ | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.90 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.87 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.85 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| 0.80 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 85.1% | 85.1% | 85.1% |
| 0.75 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 85.1% | 83.7% | 77.1% |
| 0.70 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 85.1% | 74.2% | 70.0% |
| 0.60 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 81.0% | 66.4% | 62.6% |
| 0.50 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 75.0% | 47.7% | 46.9% |
| 0.30 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 75.0% | 47.7% | 46.9% |
| 0.00 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 85.1% | 75.0% | 47.7% | 46.9% |

## 5. Results — NOISY corpus

**Nodes absorbed** (ground truth: 36 duplicates should be absorbed, nothing else):

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.99 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 0.97 | 0 | 0 | 2 | 4 | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| 0.95 | 0 | 0 | 5 | 9 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| 0.93&nbsp;★ | 0 | 0 | 5 | 10 | 13 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 |
| 0.90 | 0 | 0 | 5 | 10 | 13 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 |
| 0.87 | 0 | 0 | 7 | 12 | 15 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |
| 0.85 | 0 | 1 | 8 | 13 | 16 | 18 | 18 | 18 | 18 | 18 | 19 | 19 | 19 | 19 |
| 0.80 | 0 | 1 | 9 | 15 | 18 | 20 | 20 | 20 | 20 | 20 | 23 | 24 | 24 | 24 |
| 0.75 | 0 | 1 | 11 | 18 | 21 | 24 | 24 | 24 | 24 | 24 | 29 | 32 | 35 | 39 |
| 0.70 | 0 | 1 | 13 | 23 | 26 | 29 | 29 | 29 | 29 | 30 | 35 | 39 | 48 | 57 |
| 0.60 | 0 | 1 | 14 | 31 | 34 | 39 | 39 | 39 | 39 | 40 | 45 | 50 | 70 | 86 |
| 0.50 | 0 | 1 | 14 | 31 | 34 | 40 | 40 | 40 | 40 | 41 | 47 | 53 | 79 | 96 |
| 0.30 | 0 | 1 | 14 | 31 | 34 | 40 | 40 | 40 | 40 | 41 | 47 | 53 | 81 | 96 |
| 0.00 | 0 | 1 | 14 | 31 | 34 | 40 | 40 | 40 | 40 | 41 | 47 | 53 | 81 | 96 |

**Pair-level merge F1** against the known ground truth:

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.99 | 0.0% | 0.0% | 0.0% | 0.0% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% |
| 0.97 | 0.0% | 0.0% | 10.5% | 20.0% | 28.6% | 32.6% | 32.6% | 32.6% | 32.6% | 32.6% | 32.6% | 32.6% | 32.6% | 32.6% |
| 0.95 | 0.0% | 0.0% | 24.4% | 40.0% | 46.8% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| 0.93&nbsp;★ | 0.0% | 0.0% | 24.4% | 43.5% | 53.1% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% |
| 0.90 | 0.0% | 0.0% | 24.4% | 43.5% | 53.1% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% | 58.8% |
| 0.87 | 0.0% | 0.0% | 32.6% | 50.0% | 58.8% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% | 64.2% |
| 0.85 | 0.0% | 5.4% | 36.4% | 53.1% | 61.5% | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% | 65.5% | 65.5% | 65.5% | 65.5% |
| 0.80 | 0.0% | 5.4% | 40.0% | 58.8% | 66.7% | 71.4% | 71.4% | 71.4% | 71.4% | 71.4% | 66.7% | 64.5% | 64.5% | 64.5% |
| 0.75 | 0.0% | 5.4% | 46.8% | 63.0% | 70.2% | 76.7% | 76.7% | 76.7% | 76.7% | 76.7% | 68.7% | 64.8% | 55.4% | 50.5% |
| 0.70 | 0.0% | 5.4% | 53.1% | 70.0% | 76.2% | 81.8% | 81.8% | 81.8% | 81.8% | 79.4% | 71.1% | 65.1% | 38.6% | 23.1% |
| 0.60 | 0.0% | 5.4% | 56.0% | 76.1% | 81.1% | 88.6% | 88.6% | 88.6% | 88.6% | 86.4% | 77.8% | 70.0% | 20.0% | 9.4% |
| 0.50 | 0.0% | 5.4% | 56.0% | 76.1% | 81.1% | 90.0% | 90.0% | 90.0% | 90.0% | 87.8% | 77.4% | 69.2% | 17.3% | 7.0% |
| 0.30 | 0.0% | 5.4% | 56.0% | 76.1% | 81.1% | 90.0% | 90.0% | 90.0% | 90.0% | 87.8% | 77.4% | 69.2% | 15.8% | 7.0% |
| 0.00 | 0.0% | 5.4% | 56.0% | 76.1% | 81.1% | 90.0% | 90.0% | 90.0% | 90.0% | 87.8% | 77.4% | 69.2% | 15.8% | 7.0% |

**Graph modularity** of the resulting graph:

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| 0.99 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| 0.97 | 0.95 | 0.95 | 0.94 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 |
| 0.95 | 0.95 | 0.95 | 0.93 | 0.91 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| 0.93&nbsp;★ | 0.95 | 0.95 | 0.93 | 0.90 | 0.89 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| 0.90 | 0.95 | 0.95 | 0.93 | 0.90 | 0.89 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| 0.87 | 0.95 | 0.95 | 0.92 | 0.88 | 0.88 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 |
| 0.85 | 0.95 | 0.95 | 0.91 | 0.88 | 0.87 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| 0.80 | 0.95 | 0.95 | 0.91 | 0.87 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.75 | 0.95 | 0.95 | 0.91 | 0.87 | 0.86 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 | 0.83 | 0.82 | 0.82 | 0.78 |
| 0.70 | 0.95 | 0.95 | 0.91 | 0.84 | 0.83 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.80 | 0.77 | 0.71 | 0.64 |
| 0.60 | 0.95 | 0.95 | 0.91 | 0.80 | 0.78 | 0.75 | 0.75 | 0.75 | 0.75 | 0.74 | 0.73 | 0.69 | 0.51 | 0.42 |
| 0.50 | 0.95 | 0.95 | 0.91 | 0.80 | 0.78 | 0.74 | 0.74 | 0.74 | 0.74 | 0.73 | 0.72 | 0.68 | 0.44 | 0.31 |
| 0.30 | 0.95 | 0.95 | 0.91 | 0.80 | 0.78 | 0.74 | 0.74 | 0.74 | 0.74 | 0.73 | 0.72 | 0.68 | 0.43 | 0.31 |
| 0.00 | 0.95 | 0.95 | 0.91 | 0.80 | 0.78 | 0.74 | 0.74 | 0.74 | 0.74 | 0.73 | 0.72 | 0.68 | 0.43 | 0.31 |

**Graph-only fact recall (C, permissive)** — the downstream metric the whole question is about:

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% |
| 0.99 | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% | 67.5% |
| 0.97 | 67.5% | 67.5% | 67.5% | 67.5% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% | 70.7% |
| 0.95 | 67.5% | 67.5% | 73.6% | 73.6% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% |
| 0.93&nbsp;★ | 67.5% | 67.5% | 73.6% | 73.6% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% |
| 0.90 | 67.5% | 67.5% | 73.6% | 73.6% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% | 76.8% |
| 0.87 | 67.5% | 67.5% | 75.4% | 75.4% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% | 78.6% |
| 0.85 | 67.5% | 69.3% | 77.1% | 77.1% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% |
| 0.80 | 67.5% | 69.3% | 77.1% | 77.1% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 76.2% | 74.4% | 74.4% | 74.4% |
| 0.75 | 67.5% | 69.3% | 77.1% | 77.1% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 79.8% | 78.0% | 76.5% | 70.0% |
| 0.70 | 67.5% | 69.3% | 77.1% | 75.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 80.4% | 78.0% | 76.2% | 61.1% | 57.3% |
| 0.60 | 67.5% | 69.3% | 77.1% | 79.3% | 82.1% | 83.9% | 83.9% | 83.9% | 83.9% | 83.9% | 81.5% | 75.6% | 62.9% | 60.2% |
| 0.50 | 67.5% | 69.3% | 77.1% | 79.3% | 82.1% | 83.9% | 83.9% | 83.9% | 83.9% | 83.9% | 79.8% | 69.6% | 46.0% | 39.2% |
| 0.30 | 67.5% | 69.3% | 77.1% | 79.3% | 82.1% | 83.9% | 83.9% | 83.9% | 83.9% | 83.9% | 79.8% | 69.6% | 46.0% | 39.2% |
| 0.00 | 67.5% | 69.3% | 77.1% | 79.3% | 82.1% | 83.9% | 83.9% | 83.9% | 83.9% | 83.9% | 79.8% | 69.6% | 46.0% | 39.2% |

**Shipped system fact recall (D, permissive):**

| v \ n | off | 0.98 | 0.96 | 0.95 | 0.94 | 0.93 | 0.90 | 0.87&nbsp;★ | 0.80 | 0.70 | 0.60 | 0.50 | 0.30 | 0.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.99 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.97 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.95 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.93&nbsp;★ | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.90 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.87 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.85 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% |
| 0.80 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 89.3% | 92.9% | 92.9% | 92.9% |
| 0.75 | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 91.1% | 92.9% | 91.1% |
| 0.70 | 91.1% | 91.1% | 91.1% | 91.1% | 92.9% | 92.9% | 92.9% | 92.9% | 92.9% | 92.9% | 92.9% | 91.1% | 91.1% | 91.1% |
| 0.60 | 91.1% | 91.1% | 91.1% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% |
| 0.50 | 91.1% | 91.1% | 91.1% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 92.9% |
| 0.30 | 91.1% | 91.1% | 91.1% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 92.9% |
| 0.00 | 91.1% | 91.1% | 91.1% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 94.6% | 92.9% |

### 5.1 The 1-D curve at the shipped lexical threshold (name = 0.87)

| cosine threshold | merges | pair prec. | pair recall | nodes | edges | comps | modularity | C recall | C coverage | D recall | D coverage | D recall (strict) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0 | 100.0% | 0.0% | 126 | 92 | 34 | 0.953 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 0.99 | 1 | 100.0% | 2.8% | 125 | 92 | 33 | 0.952 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 0.97 | 7 | 100.0% | 19.4% | 119 | 92 | 27 | 0.925 | 70.7% | 28.6% | 91.1% | 71.4% | 89.3% |
| 0.95 | 12 | 100.0% | 33.3% | 114 | 92 | 22 | 0.898 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.93 ★ | 15 | 100.0% | 41.7% | 111 | 92 | 19 | 0.879 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.90 | 15 | 100.0% | 41.7% | 111 | 92 | 19 | 0.879 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.87 | 17 | 100.0% | 47.2% | 109 | 92 | 17 | 0.866 | 78.6% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.85 | 18 | 100.0% | 50.0% | 108 | 92 | 16 | 0.863 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.80 | 20 | 100.0% | 55.6% | 106 | 92 | 15 | 0.856 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.75 | 24 | 95.8% | 63.9% | 102 | 92 | 13 | 0.843 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 0.70 | 29 | 90.0% | 75.0% | 97 | 92 | 8 | 0.822 | 80.4% | 42.9% | 92.9% | 78.6% | 91.1% |
| 0.60 | 39 | 81.4% | 97.2% | 87 | 92 | 1 | 0.749 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 0.50 | 40 | 81.8% | 100.0% | 86 | 92 | 1 | 0.742 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 0.30 | 40 | 81.8% | 100.0% | 86 | 92 | 1 | 0.742 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 0.00 | 40 | 81.8% | 100.0% | 86 | 92 | 1 | 0.742 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |

### 5.2 The aggressiveness curve — the shape the hypothesis is about

The grid is two-dimensional but the hypothesis is one-dimensional: it is about *how much merging happened*. Cells that produced the identical merge set are the identical graph, so each of the 62 distinct resolver outcomes below is one point, not many — otherwise a densely sampled corner of the grid would dominate the curve.

```
  graph-only fact recall (C, permissive)
   83.9% ┤                                       oo                     
   80.5% ┤                         o#oooo o ooo     oo                  
   77.0% ┤                o oo #ooo #   #    o # o    o                 
   73.6% ┤              o    oo         o                               
   70.2% ┤      o        oo                      o     o                
   66.7% ┤o     o  o  o                                                 
   63.3% ┤                                                    o         
   59.8% ┤                                           o              o   
   56.4% ┤                                               o              
   52.9% ┤                                                              
   49.5% ┤                                                              
   46.1% ┤                                                       oo     
   42.6% ┤                                                              
   39.2% ┤                                                             o
         └──────────────────────────────────────────────────────────────
          0              6              24                             96
          duplicate nodes absorbed (√ scale)

  shipped-system fact recall (D, permissive)
   94.6% ┤                                   oo  oo oooo      o  oo o   
   94.2% ┤                                                              
   93.8% ┤                                                              
   93.4% ┤                                                              
   93.0% ┤                              o o #  #                       o
   92.6% ┤                                                              
   92.2% ┤                                                              
   91.8% ┤                                                              
   91.3% ┤                                                              
   90.9% ┤o     #  o  o oo# o#o#oooo#ooo#   oo   #   o   o              
   90.5% ┤                                                              
   90.1% ┤                                                              
   89.7% ┤                                                              
   89.3% ┤                              o                               
         └──────────────────────────────────────────────────────────────
          0              6              24                             96
          duplicate nodes absorbed (√ scale)
```

| merges | pair prec. | pair recall | nodes | edges | comps | modularity | C recall | C coverage | D recall | D coverage | D strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100.0% | 0.0% | 126 | 92 | 34 | 0.953 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 1 | 100.0% | 2.8% | 125 | 92 | 33 | 0.952 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 1 | 100.0% | 2.8% | 125 | 92 | 33 | 0.950 | 69.3% | 28.6% | 91.1% | 71.4% | 89.3% |
| 2 | 100.0% | 5.6% | 124 | 92 | 32 | 0.944 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 4 | 100.0% | 11.1% | 122 | 92 | 30 | 0.932 | 67.5% | 28.6% | 91.1% | 71.4% | 89.3% |
| 5 | 100.0% | 13.9% | 121 | 92 | 29 | 0.930 | 73.6% | 28.6% | 91.1% | 71.4% | 89.3% |
| 6 | 100.0% | 16.7% | 120 | 92 | 28 | 0.926 | 70.7% | 28.6% | 91.1% | 71.4% | 89.3% |
| 7 | 100.0% | 19.4% | 119 | 92 | 27 | 0.925 | 70.7% | 28.6% | 91.1% | 71.4% | 89.3% |
| 7 | 100.0% | 19.4% | 119 | 92 | 27 | 0.918 | 75.4% | 28.6% | 91.1% | 71.4% | 89.3% |
| 8 | 100.0% | 22.2% | 118 | 92 | 26 | 0.914 | 77.1% | 28.6% | 91.1% | 71.4% | 89.3% |
| 9 | 100.0% | 25.0% | 117 | 92 | 25 | 0.906 | 73.6% | 21.4% | 91.1% | 71.4% | 89.3% |
| 9 | 100.0% | 25.0% | 117 | 92 | 25 | 0.912 | 77.1% | 28.6% | 91.1% | 71.4% | 89.3% |
| 10 | 100.0% | 27.8% | 116 | 92 | 24 | 0.896 | 73.6% | 21.4% | 91.1% | 71.4% | 89.3% |
| 11 | 100.0% | 30.6% | 115 | 92 | 23 | 0.900 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 11 | 100.0% | 30.6% | 115 | 92 | 24 | 0.911 | 77.1% | 28.6% | 91.1% | 71.4% | 89.3% |
| 12 | 100.0% | 33.3% | 114 | 92 | 22 | 0.898 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 12 | 100.0% | 33.3% | 114 | 92 | 22 | 0.881 | 75.4% | 21.4% | 91.1% | 71.4% | 89.3% |
| 13 | 100.0% | 36.1% | 113 | 92 | 21 | 0.887 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 13 | 100.0% | 36.1% | 113 | 92 | 21 | 0.878 | 77.1% | 21.4% | 91.1% | 71.4% | 89.3% |
| 13 | 100.0% | 36.1% | 113 | 92 | 22 | 0.907 | 77.1% | 28.6% | 91.1% | 71.4% | 89.3% |
| 14 | 100.0% | 38.9% | 112 | 92 | 21 | 0.907 | 77.1% | 28.6% | 91.1% | 71.4% | 89.3% |
| 15 | 100.0% | 41.7% | 111 | 92 | 19 | 0.879 | 76.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 15 | 100.0% | 41.7% | 111 | 92 | 19 | 0.876 | 78.6% | 35.7% | 91.1% | 71.4% | 89.3% |
| 15 | 100.0% | 41.7% | 111 | 92 | 19 | 0.868 | 77.1% | 21.4% | 91.1% | 71.4% | 89.3% |
| 16 | 100.0% | 44.4% | 110 | 92 | 18 | 0.871 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 17 | 100.0% | 47.2% | 109 | 92 | 17 | 0.866 | 78.6% | 35.7% | 91.1% | 71.4% | 89.3% |
| 18 | 100.0% | 50.0% | 108 | 92 | 16 | 0.863 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 18 | 100.0% | 50.0% | 108 | 92 | 16 | 0.862 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 18 | 94.4% | 47.2% | 108 | 92 | 17 | 0.865 | 77.1% | 21.4% | 91.1% | 71.4% | 89.3% |
| 19 | 94.7% | 50.0% | 107 | 92 | 15 | 0.861 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 20 | 100.0% | 55.6% | 106 | 92 | 15 | 0.856 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 21 | 95.2% | 55.6% | 105 | 92 | 14 | 0.857 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 23 | 83.3% | 55.6% | 103 | 91 | 13 | 0.850 | 76.2% | 28.6% | 89.3% | 71.4% | 89.3% |
| 23 | 87.5% | 58.3% | 103 | 92 | 12 | 0.844 | 75.4% | 21.4% | 91.1% | 71.4% | 89.3% |
| 24 | 76.9% | 55.6% | 102 | 91 | 12 | 0.845 | 74.4% | 21.4% | 92.9% | 78.6% | 91.1% |
| 24 | 95.8% | 63.9% | 102 | 92 | 13 | 0.843 | 80.4% | 35.7% | 91.1% | 71.4% | 89.3% |
| 26 | 88.9% | 66.7% | 100 | 92 | 9 | 0.834 | 80.4% | 42.9% | 92.9% | 78.6% | 91.1% |
| 29 | 74.2% | 63.9% | 97 | 90 | 11 | 0.835 | 79.8% | 35.7% | 91.1% | 71.4% | 89.3% |
| 29 | 90.0% | 75.0% | 97 | 92 | 8 | 0.822 | 80.4% | 42.9% | 92.9% | 78.6% | 91.1% |
| 30 | 84.4% | 75.0% | 96 | 92 | 7 | 0.816 | 80.4% | 42.9% | 92.9% | 78.6% | 91.1% |
| 31 | 77.1% | 75.0% | 95 | 92 | 6 | 0.799 | 79.3% | 42.9% | 94.6% | 85.7% | 91.1% |
| 32 | 65.7% | 63.9% | 94 | 90 | 8 | 0.817 | 78.0% | 35.7% | 91.1% | 71.4% | 89.3% |
| 34 | 78.9% | 83.3% | 92 | 92 | 3 | 0.782 | 82.1% | 50.0% | 94.6% | 85.7% | 91.1% |
| 35 | 48.9% | 63.9% | 91 | 89 | 7 | 0.820 | 76.5% | 28.6% | 92.9% | 78.6% | 91.1% |
| 35 | 67.5% | 75.0% | 91 | 90 | 6 | 0.798 | 78.0% | 35.7% | 92.9% | 78.6% | 91.1% |
| 39 | 41.8% | 63.9% | 87 | 88 | 6 | 0.778 | 70.0% | 14.3% | 91.1% | 71.4% | 89.3% |
| 39 | 57.4% | 75.0% | 87 | 90 | 4 | 0.770 | 76.2% | 35.7% | 91.1% | 71.4% | 89.3% |
| 39 | 81.4% | 97.2% | 87 | 92 | 1 | 0.749 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 40 | 77.8% | 97.2% | 86 | 92 | 1 | 0.739 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 40 | 81.8% | 100.0% | 86 | 92 | 1 | 0.742 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 41 | 78.3% | 100.0% | 85 | 92 | 1 | 0.731 | 83.9% | 57.1% | 94.6% | 85.7% | 91.1% |
| 45 | 64.8% | 97.2% | 81 | 89 | 1 | 0.727 | 81.5% | 50.0% | 94.6% | 85.7% | 91.1% |
| 47 | 63.2% | 100.0% | 79 | 89 | 1 | 0.719 | 79.8% | 42.9% | 94.6% | 85.7% | 89.3% |
| 48 | 26.0% | 75.0% | 78 | 87 | 3 | 0.709 | 61.1% | 7.1% | 91.1% | 71.4% | 89.3% |
| 50 | 54.7% | 97.2% | 76 | 89 | 1 | 0.694 | 75.6% | 35.7% | 94.6% | 85.7% | 91.1% |
| 53 | 52.9% | 100.0% | 73 | 89 | 1 | 0.683 | 69.6% | 21.4% | 94.6% | 85.7% | 91.1% |
| 57 | 13.6% | 75.0% | 69 | 83 | 3 | 0.637 | 57.3% | 7.1% | 91.1% | 71.4% | 89.3% |
| 70 | 11.1% | 97.2% | 56 | 81 | 1 | 0.511 | 62.9% | 14.3% | 94.6% | 85.7% | 91.1% |
| 79 | 9.4% | 100.0% | 47 | 78 | 1 | 0.440 | 46.0% | 0.0% | 94.6% | 85.7% | 89.3% |
| 81 | 8.6% | 100.0% | 45 | 74 | 1 | 0.433 | 46.0% | 0.0% | 94.6% | 85.7% | 89.3% |
| 86 | 4.9% | 97.2% | 40 | 65 | 1 | 0.418 | 60.2% | 7.1% | 94.6% | 85.7% | 89.3% |
| 96 | 3.6% | 100.0% | 30 | 57 | 1 | 0.306 | 39.2% | 0.0% | 92.9% | 78.6% | 92.9% |

**Measured shape: inverted-u.** The peak of graph-only fact recall sits at 39 absorbed nodes (83.9%); the least aggressive outcome (0 merges) scores 67.5% and the most aggressive (96 merges, graph down to 30 nodes) scores 39.2%. Rise to the peak +16.4 pp, fall from it -44.8 pp, total spread 44.8 pp. The effect-size floor — one question out of 14 — is 7.1 pp, and the peak is strictly interior, so the predicted inverted U is present.

### 5.3 What merges, and what breaks, at each regime

**(0.00 / off)** — 0 nodes absorbed, precision 100.0%, recall 0.0%, largest cluster 0; graph 126 nodes / 92 edges / 34 components, modularity 0.953; C recall 67.5%, D recall 91.1%.

  - no merges at all

**(0.90 / 0.96)** — 5 nodes absorbed, precision 100.0%, recall 13.9%, largest cluster 2; graph 121 nodes / 92 edges / 29 components, modularity 0.930; C recall 73.6%, D recall 91.1%.

  - correct merges: `Backproagation` = `Backpropagation`; `Dartmouth Workshop` = `Dartmouth orkshop`; `Large Langage Model` = `Large Language Model`; `Tabulating Machine Company` = `Tabulating Mahine Company`; `Turing Machine` = `Turing achine`

**(0.80 / 0.95)** — 15 nodes absorbed, precision 100.0%, recall 41.7%, largest cluster 2; graph 111 nodes / 92 edges / 19 components, modularity 0.868; C recall 77.1%, D recall 91.1%.

  - correct merges: `Alan Turing` = `Turing`; `Artificial Intelligence` = `Artificial ntelligence`; `Backproagation` = `Backpropagation`; `Berners-Lee` = `Tim Berners-Lee`; `Dartmouth Workshop` = `Dartmouth orkshop`; `Ilya Sutskever` = `Sutskever`

**(0.70 / 0.94)** — 26 nodes absorbed, precision 88.9%, recall 66.7%, largest cluster 3; graph 100 nodes / 92 edges / 9 components, modularity 0.834; C recall 80.4%, D recall 92.9%.

  - correct merges: `Alan Turing` = `Turing`; `Artificial Intelligence` = `Artificial ntelligence`; `Attention Is All You Need` = `Attention Is ll You Need`; `Backproagation` = `Backpropagation`; `Berners-Lee` = `Tim Berners-Lee`; `Bletchley Park` = `Bletchly Park`
  - **wrong merges**: `Ilya Sutskever` = `Peter Sutskever`; `Krizhevsky` = `Tomas Krizhevsky`; `Peter Sutskever` = `Sutskever`

**(0.70 / 0.60)** — 35 nodes absorbed, precision 67.5%, recall 75.0%, largest cluster 3; graph 91 nodes / 90 edges / 6 components, modularity 0.798; C recall 78.0%, D recall 92.9%.

  - correct merges: `Alan Turing` = `Turing`; `Artificial Intelligence` = `Artificial ntelligence`; `Attention Is All You Need` = `Attention Is ll You Need`; `Backproagation` = `Backpropagation`; `Berners-Lee` = `Tim Berners-Lee`; `Bletchley Park` = `Bletchly Park`
  - **wrong merges**: `Dartmouth Reunion` = `Dartmouth Workshop`; `Dartmouth Reunion` = `Dartmouth orkshop`; `EDVAC` = `ENIAC`; `Ilya Sutskever` = `Peter Sutskever`; `Knowlede Graph` = `Knowledge Base`; `Knowledge Base` = `Knowledge Graph`

**(0.60 / 0.50)** — 50 nodes absorbed, precision 54.7%, recall 97.2%, largest cluster 4; graph 76 nodes / 89 edges / 1 components, modularity 0.694; C recall 75.6%, D recall 94.6%.

  - correct merges: `AI Winter` = `AI Wnter`; `Alan Turing` = `Turing`; `Alex Krizhevsky` = `Krizhevsky`; `Analytical Engine` = `Analyticl Engine`; `Artificial Intelligence` = `Artificial ntelligence`; `Attention Is All You Need` = `Attention Is ll You Need`
  - **wrong merges**: `Alex Krizhevsky` = `Tomas Krizhevsky`; `Charles Babbage` = `Julian Babbage`; `Dartmouth Reunion` = `Dartmouth Workshop`; `Dartmouth Reunion` = `Dartmouth orkshop`; `EDVAC` = `ENIAC`; `Fine-Tuning` = `Turing Machine`

**(0.00 / 0.30)** — 81 nodes absorbed, precision 8.6%, recall 100.0%, largest cluster 22; graph 45 nodes / 74 edges / 1 components, modularity 0.433; C recall 46.0%, D recall 94.6%.

  - correct merges: `AI Winter` = `AI Wnter`; `Alan Turing` = `Turing`; `Alex Krizhevsky` = `Krizhevsky`; `Analytical Engine` = `Analyticl Engine`; `Artificial Intelligence` = `Artificial ntelligence`; `Attention Is All You Need` = `Attention Is ll You Need`
  - **wrong merges**: `ARPANET` = `AlexNet`; `ARPANET` = `AlphaGo`; `ARPANET` = `Analytical Engine`; `ARPANET` = `Analyticl Engine`; `ARPANET` = `EDVAC`; `ARPANET` = `ENIAC`

**(0.00 / 0.00)** — 96 nodes absorbed, precision 3.6%, recall 100.0%, largest cluster 31; graph 30 nodes / 57 edges / 1 components, modularity 0.306; C recall 39.2%, D recall 92.9%.

  - correct merges: `AI Winter` = `AI Wnter`; `Alan Turing` = `Turing`; `Alex Krizhevsky` = `Krizhevsky`; `Analytical Engine` = `Analyticl Engine`; `Artificial Intelligence` = `Artificial ntelligence`; `Attention Is All You Need` = `Attention Is ll You Need`
  - **wrong merges**: `AI Winter` = `Dartmouth Reunion`; `AI Winter` = `Dartmouth Workshop`; `AI Winter` = `Dartmouth orkshop`; `AI Wnter` = `Dartmouth Reunion`; `AI Wnter` = `Dartmouth Workshop`; `AI Wnter` = `Dartmouth orkshop`

### 5.4 The collapse point

Scanning the grid from conservative to aggressive, the first cell that makes any wrong merge is **(0.85 / 0.60)** (1 wrong pair(s), precision 94.7%). The first cell at which the *majority* of asserted merges are wrong is **(0.75 / 0.30)** (precision 48.9%, 35 nodes absorbed, graph down to 91 nodes, modularity 0.820).

At the most aggressive cell in the grid, (0.50 / 0.00), the graph is 30 nodes and 57 edges (from 126 / 92 unresolved), modularity 0.306, largest merge cluster 31 nodes, pair precision 3.6%. Graph-only fact recall there is 39.2% and the shipped system's is 92.9%.

## 6. Headline numbers, with uncertainty

| Configuration | Graph-only fact recall (C, permissive) | bootstrap 95% CI |
| --- | ---: | ---: |
| resolution OFF (off / off) | 67.5% | [53.6%, 81.8%] |
| shipped defaults (0.93 / 0.87) ★ | 76.8% | [64.3%, 87.5%] |
| best cell in the grid (0.60 / 0.93) | 83.9% | [73.2%, 94.6%] |
| worst cell in the grid (0.50 / 0.00) | 39.2% | [27.4%, 50.5%] |
| ORACLE — perfect resolution | 87.5% | [76.8%, 96.4%] |

| Paired comparison (per-question, n=14) | mean Δ | bootstrap 95% CI | crosses 0? |
| --- | ---: | ---: | :---: |
| shipped defaults − resolution OFF | +9.3 pp | [-1.8, +23.9] pp | yes |
| best cell − shipped defaults | +7.1 pp | [-3.6, +17.9] pp | yes |
| oracle − shipped defaults | +10.7 pp | [+3.6, +19.6] pp | **no** |
| best cell − worst cell (total grid sensitivity) | +44.8 pp | [+26.2, +63.2] pp | **no** |

## 7. Interpretation

**Measured.**

1. **On a corpus with no duplicates the knob is inert until it is harmful.** 182 of the 210 CLEAN cells — including the shipped defaults — merge nothing at all, so retrieval is bit-identical across them. The 28 cells that do merge are all merging entities that are not duplicates, because the CLEAN entity list is canonical by construction. There is no upside available on this corpus, only downside.

2. **Unresolved duplication is expensive, and resolution does recover it.** On NOISY, going from resolution OFF to *perfect* resolution (the oracle) is worth +20.0 pp [95% CI +5.4, +34.6 pp; excludes 0] of graph-only fact recall. The shipped defaults recover +9.3 pp [95% CI -1.8, +23.9 pp; includes 0] of that, absorbing 15 of the 36 injected duplicates at pair precision 100.0%.

3. **The head-room left on the table by the shipped defaults is real but bounded.** Oracle − defaults = +10.7 pp [95% CI +3.6, +19.6 pp; excludes 0]; best grid cell − defaults = +7.1 pp [95% CI -3.6, +17.9 pp; includes 0] (best cell (0.60 / 0.93)).

4. **Total sensitivity of the whole grid.** Best cell − worst cell = +44.8 pp [95% CI +26.2, +63.2 pp; excludes 0] on graph-only fact recall. Across the 62 distinct resolver outcomes, graph-only fact recall spans 39.2%–83.9% (44.8 pp), which is larger than the one-question floor of 7.1 pp.

5. **The shipped system is far less sensitive than the graph-only system.** D (graph + source chunks) spans 89.3%–94.6% (5.4 pp) over the same 62 outcomes, against C's 44.8 pp. The source-chunk channel does not care which entity node the evidence hangs off, so it absorbs most of the damage the entity layer takes.

6. **Structure moves much further than retrieval does.** Over the same outcomes the graph goes from 126 nodes down to 30 and modularity from 0.953 down to 0.306 — a 65-point swing in a quantity bounded by 1 — while graph-only fact recall moves 44.8 pp and the shipped system moves 5.4 pp. Graph statistics are a far more sensitive indicator of over-merging than the retrieval metric is.

7. **Perfect resolution is geometrically unreachable here, at any threshold.** No cell separates all 36 true duplicate pairs from the 45 same-type pairs that must not merge; the best fully-precise setting recovers 55.6% of the duplicates. The binding constraint is that different people who share a surname score *exactly* the same lexically as a person and their own surname.

8. **The collapse is gradual on retrieval and abrupt on structure.** The first wrong merge appears at (0.85 / 0.60) and the majority of asserted merges are wrong from (0.75 / 0.30); at the most aggressive cell (0.50 / 0.00) the graph is 30 nodes (from 126) yet graph-only fact recall is still 39.2% and the shipped system's is 92.9%.

**Inferred — weaker than the measurements above, and labelled as such.**

- The mechanism we *infer* behind (5) and (6) is that this benchmark scores retrieval by whether a required entity *name* reaches the context, and a wrongly-merged node still carries both names (the absorbed name is kept as an alias and the canonical name is the longer one). Over-merging therefore destroys *structure* — it grafts foreign edges on and invents paths — long before it destroys *name reachability*, which is what the metric can see. A benchmark that scored whether the retrieved edges were **true** would, we expect, fall off much earlier. This experiment cannot show that; it is a prediction, not a result.
- We infer that the practitioner-facing consequence is a *precision* argument rather than a *recall* argument: the reason to keep the thresholds strict is not that loose thresholds visibly cost retrieval on this corpus (they mostly do not), but that they silently manufacture false relationships that a downstream reader — human or LLM — has no way to detect. Testing that costs a generation-side experiment and an LLM budget, which this one deliberately does not have.

## 8. Threats to validity

- **Small n.** 14 questions. The bootstrap resamples questions, so the effective sample size is 14, not the number of facts. Every CI here is wide, and one question is worth 7.1 pp — treat any gap below that as a wash, which several of the comparisons above are.
- **The noise model is ours, not nature's.** The duplicate distribution is synthetic: 36 injected variants (surname-only and one-character deletion), half carrying the source description and half not, with every second edge and chunk link re-pointed. Those choices set where the true pairs land on the decision surface and therefore how hard the problem is. A real extractor's duplicates would land somewhere else, and the whole curve would move with them. Nothing here measures what an LLM extractor actually produces — doing that would require LLM calls, which this experiment is forbidden.
- **The decoys set the difficulty of the precision side.** The 11 adversarial entities were written by hand before any result was seen, but they were written to be hard. A corpus without same-surname collisions would show a much wider precise operating region, and one with more of them a narrower one. The 'perfect resolution is unreachable' finding is a statement about this corpus, not a theorem.
- **Retrieval only, no generation.** The metric is whether a required entity name reaches the context. It cannot see a hallucinated relationship, a false reasoning path, or an answer that is wrong for a reason a human would notice. That is precisely the failure mode over-merging is supposed to cause, so the most important consequence of aggressive resolution is invisible to this instrument.
- **The permissive rule flatters merged graphs.** A required fact scores when its name appears anywhere in the context, including in a neighbour line that a wrong merge put there. The strict rule (prose only) is reported for D throughout, and C scores 0.0 under it by construction. Read permissive as an upper bound.
- **One corpus, one embedder, one seed count.** `benchmarks/dataset.json` at k=8 with `fastembed` embeddings and `retrieval_max_hops=2`. The corpus is small enough (34 passages) that the chunk channel saturates, which is exactly why D is nearly flat; on a large corpus D would have more room to be hurt.
- **The grid is a sample.** 15 × 14 thresholds, denser where the pairs pile up. Behaviour *between* sampled thresholds is interpolated by the reader, not measured. The 210 cells collapse to only 62 distinct graphs, so the grid is finer than the outcome space and the curve above has that many points and no more.
- **Modularity is a heuristic number.** Greedy modularity maximisation on the undirected simple graph, inserted in sorted order for determinism. It is comparable across the cells of this run and should not be compared with modularity computed any other way.
- **In production this resolver runs per document as well as whole-graph.** This experiment resolves one batch of entities in one pass. An incremental ingest reaches a different fixed point, and that path is not measured here.
- **A shared database.** The research Neo4j is Community edition — one database, no namespaces — and other experiments write to it concurrently. Every cell is therefore verified after it runs (exact entity count, exact chunk count, plus the harness's own contamination check) and re-seeded and re-run when the state drifted. This run detected 0 such external writes and recovered 0 cell measurement(s) by re-running them; a cell that could not be measured cleanly would have aborted the run rather than been reported.

## 9. Claim we could defend in a paper

> **Defensible.** On a 79-entity multi-hop benchmark corpus into which 36 realistic surface-form duplicates and 11 adversarial same-surface distinct entities were injected, sweeping Synapse's two entity-resolution thresholds across 210 settings (62 distinct resulting graphs, from 0 to 96 absorbed nodes) moves graph-only retrieval fact recall by 44.8 pp end to end (best − worst +26.2 to +63.2 pp, paired bootstrap 95% CI, n=14) and the shipped graph+chunks system by 5.4 pp — while the same sweep changes the graph from 126 nodes to 30 and modularity from 0.95 to 0.31. Structure is destroyed long before retrieval notices.

> **On the hypothesis:** the predicted inverted U is present: graph-only fact recall peaks at 39 absorbed nodes (83.9%), rising +16.4 pp from the unresolved graph and falling -44.8 pp to the most aggressive setting.

> **On the shipped defaults (0.93 / 0.87):** they sit in the fully-precise region of the grid — 0 wrong pair(s) at 100.0% precision — and are worth +9.3 pp [95% CI -1.8, +23.9 pp; includes 0] of graph-only fact recall against resolution being off, a difference whose CI includes 0. Perfect resolution would add a further +10.7 pp [95% CI +3.6, +19.6 pp; excludes 0]. They are a defensible choice — the cost of being stricter than optimal is small and the cost of being looser is a graph that is quietly wrong.

> **NOT defensible from this run:** that entity-resolution thresholds are an important tuning knob for retrieval quality. On this corpus they are not; the measurable consequence of getting them wrong is structural, and the instrument used here (does a required entity *name* reach the context?) cannot see structural damage. Nor can this run say anything about answer quality, which is where a false merged edge would actually do its damage.

---

Generated by `backend/research/rq3_entity_resolution/run_rq3.py` — Synapse, © 2026 Ahmed Maaloul, AGPL-3.0-or-later.
