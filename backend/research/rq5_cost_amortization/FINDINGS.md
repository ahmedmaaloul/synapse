# RQ5 — When does GraphRAG amortize its extraction cost?

*Generated 2026-07-21 09:56 UTC by `research/rq5_cost_amortization/run_rq5.py`. Every number below is written by that script; none is typed by hand.*

```
cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=research_secret \
    python research/rq5_cost_amortization/run_rq5.py
```

Seed `20260721` · bootstrap `10,000` iterations · chat model `gpt-4o-mini` · embeddings `text-embedding-3-small` · prices hand-recorded on `2026-07-21` from https://openai.com/api/pricing/.

> **Every USD figure in this document is an ESTIMATE.** Prices are a hard-coded, dated table, not a live feed, and token counts use the repo's `len/4` rule rather than a real tokenizer. No chat model was called at any point: this experiment cost $0.00 to run.

## 1. Hypothesis and prediction (stated before the results)

**H5.** GraphRAG pays a large *one-off* cost — LLM extraction over the whole corpus — in order to reduce the *per-query* cost. There is therefore a break-even query count **Q\*** below which plain vector RAG is strictly the better engineering choice.

**Prediction.** Q\* is finite and large (hundreds to thousands of queries per corpus), and grows with corpus size. Most demo projects sit below it.

**The model.** For a corpus of *N* characters answered with *Q* queries:

```
  total(system) = setup(system) + Q · per_query(system)

  setup(vector)   = embedding of the corpus                       (no LLM)
  setup(graph)    = extraction + community summaries + embeddings (LLM)

  per_query(sys)  = p_in · (rag_overhead + question + CONTEXT(sys))
                  + p_out · answer
                  + p_embed · question

  Q* = setup(graph) − setup(vector)
       ─────────────────────────────
       per_query(vector) − per_query(graph)
```

**The algebra matters.** `rag_overhead`, `question` and `answer` are identical for every retrieval system, so they cancel exactly out of the denominator. What is left is

```
  Q* = Δsetup / ( p_in · (CONTEXT(vector) − CONTEXT(graph)) / 1e6 )
```

so **a finite break-even exists if and only if GraphRAG retrieves *fewer* context tokens than the baseline it is being compared against.** Answer length, chat verbosity and the output price are irrelevant to *whether* GraphRAG amortizes — they only change the absolute bill. This is the single load-bearing quantity in the whole analysis, and it is measured, not assumed.

## 2. Measured inputs

### 2.1 Corpus and chunking (production chunker, 1000/200)

| Quantity | Value | How obtained |
| --- | ---: | --- |
| Passages in `benchmarks/dataset.json` | 34 | fixture |
| Corpus characters | 6,816 | measured |
| Corpus tokens (`len/4`) | 1,704 | measured |
| Chunks at 1000/200 | 9 | `pdf_parser.chunk_text`, run |
| Characters actually sent (overlap re-sent) | 8,402 | measured |
| Overlap inflation | 1.233x | derived |
| Extraction prompt template | 1,250 chars = 313 tokens | `get_extraction_prompt('Generic')`, rendered |
| RAG generation prompt overhead | 229 tokens | `chat_engine.RAG_PROMPT`, rendered |

On a corpus this small the **prompt template is 57% of everything extraction sends** (2,817 of 4,922 prompt tokens). That share falls as documents get longer, which is the one term that genuinely favours GraphRAG at scale.

### 2.2 Extraction output — derived from the fixture, not guessed

`benchmarks/dataset.json` *is* the extraction result for this corpus. Re-serialising its entities and relationships into the exact JSON shape the prompt demands gives the completion the model would have had to emit.

| Quantity | Value |
| --- | ---: |
| Unique entities | 79 |
| Unique relationships | 81 |
| Serialized JSON (unique only) | 24,774 chars = 6,194 tokens |
| Entity mentions across passages | 111 |
| Re-emission factor (mentions / unique) | 1.41x |
| **Completion tokens charged (realistic)** | **8,820** |
| Completion tokens if extraction never repeated itself (lower bound) | 6,311 |

Real extraction re-emits an entity in every chunk that mentions it and the dedup happens afterwards in Python, for free. Charging only the unique count would under-bill GraphRAG, so the re-emission factor is applied; both figures are shown so the choice is auditable.

### 2.3 Community summarisation (Louvain run for real, no LLM)

Louvain over the fixture graph yields **9 communities** (modularity 0.7836). Rendering the production `SUMMARY_PROMPT` for each gives 4,397 prompt tokens; completions are assumed at 120 tokens each (1,080 total) — an ASSUMPTION, sized from the prompt's own "6-word title, 2-3 sentences" rule.

### 2.4 Retrieved context — measured by running the real systems

Systems imported read-only from `benchmarks/run_benchmark.py` and run against Neo4j, so these are directly comparable with `benchmarks/results.md`. n = 14 questions; CIs are percentile bootstrap over questions.

| System | Mean context (chars) | 95% CI (chars) | Mean context (tokens) | Fact recall | Full coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Passage vector RAG (top-4) | 831 | [806, 856] | 208 | 89.3% | 71.4% |
| A*. Passage vector RAG (top-8, quality-matched to D) | 1,672 | [1,622, 1,725] | 418 | 96.4% | 92.9% |
| C. GraphRAG, graph only (k=8) | 2,056 | [1,924, 2,176] | 514 | 87.5% | 64.3% |
| D. GraphRAG + source chunks (k=8) | 3,515 | [3,398, 3,654] | 879 | 94.6% | 85.7% |

**A\*** is the cheapest passage baseline that equals or beats D on *both* metrics (top-8: 96.4% / 92.9% against D's 94.6% / 85.7%). It is the fair comparison and every headline number below is stated against it; A (top-4) is kept because it is the configuration `benchmarks/results.md` reports, but it scores *below* D and so flatters the graph's token ratio.

**Sanity check against `benchmarks/results.md`** — the figures below are *parsed* out of that file at run time (a different workflow owns and regenerates it), not transcribed:

| System | This run (mean chars) | `results.md` | Agreement |
| --- | ---: | ---: | ---: |
| A | 831 | 831 | ✅ 0.02% apart |
| C | 2,056 | 2,057 | ✅ 0.06% apart |
| D | 3,515 | 3,516 | ✅ 0.02% apart |

Worst disagreement 0.06% — the retrieval measurements in this experiment reproduce the repo's independently generated benchmark, so the cost model is built on the same numbers the project already publishes.

## 3. The one-off bill

For the 6,816-character benchmark corpus, at `gpt-4o-mini` ($0.15/1M in, $0.60/1M out):

| Term | LLM calls | Prompt tokens | Completion tokens | USD (est.) | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extraction | 9 | 4,922 | 8,820 | $0.006030 | 81.2% |
| Community summaries | 9 | 4,397 | 1,080 | $0.001308 | 17.6% |
| Embeddings (`text-embedding-3-small`) | — | 4,593 | — | $0.000092 | 1.2% |
| **Total** | **18** | | | **$0.007430** | |

**Which term dominates:** extraction — 81.2% of the setup bill. Embeddings are 1.2% of it, and Synapse's default `EMBEDDING_PROVIDER=fastembed` makes them **$0.00** (local model, no API), leaving $0.007338. **The extraction LLM is the graph's cost. Nothing else is close.**

## 4. The per-query bill

Assuming a 250-token answer (an ASSUMPTION that cancels out of every difference below) and hosted query embeddings:

| System | Context tokens | Total prompt tokens | USD / query (est.) | USD / 1,000 queries |
| --- | ---: | ---: | ---: | ---: |
| A. Passage vector RAG (top-4) | 208 | 460 | $0.000219 | $0.2195 |
| A*. Passage vector RAG (top-8, quality-matched to D) | 418 | 670 | $0.000251 | $0.2510 |
| C. GraphRAG, graph only (k=8) | 514 | 766 | $0.000265 | $0.2654 |
| D. GraphRAG + source chunks (k=8) | 879 | 1,131 | $0.000320 | $0.3201 |

**Paired bootstrap on the load-bearing quantity** — CONTEXT(baseline) − CONTEXT(D), resampling the 14 questions with the pairing intact, 10,000 iterations, seed 20260721. A finite break-even requires this to be **positive**:

| Baseline | Δ context tokens (baseline − D) | 95% CI | Interval |
| --- | ---: | ---: | --- |
| A. Passage vector RAG (top-4) | -671 | [-703, -644] | entirely below zero |
| A*. Passage vector RAG (top-8, quality-matched to D) | -461 | [-491, -432] | entirely below zero |

Both intervals sit entirely below zero. Against the *quality-matched* baseline — the fair one — GraphRAG reads **461 tokens more per query** (95% CI [432, 491] more), and the confidence interval never crosses into the amortizing regime. The sign of this quantity is the entire result.

## 5. Break-even

| Comparison | Δsetup (est.) | Saving / query | Break-even Q\* |
| --- | ---: | ---: | ---: |
| **D (graph + chunks) vs A\* (top-8, quality-matched)** — the fair one | $0.007338 | $-0.00006915 | **never** (∞) |
| D (graph + chunks) vs A (top-4, scores below D) | $0.007338 | $-0.00010065 | **never** (∞) |
| C (graph only) vs A (top-4) | $0.007338 | $-0.00004590 | **never** (∞) |

### The result

> **NULL RESULT — the break-even does not exist on this corpus.** GraphRAG is not trading a one-off cost for a per-query saving. It costs more up front **and** more per query than a passage baseline that already matches it on both retrieval metrics. Q\* is not large; it is **infinite**. Every additional query widens the gap rather than closing it.

The hypothesis has a false premise. It assumed GraphRAG's context is *smaller* — a compressed graph standing in for raw passages. Measured, Synapse's shipped context is **2.10x** the quality-matched baseline's (879 vs 418 tokens), and **4.22x** the top-4 baseline's, because the shipped system returns the graph scaffolding *and* the source chunks the entities came from. The graph is additive, not substitutive.

Even the graph-*only* configuration (C, no source chunks — the most compressed thing Synapse can emit) reads 2.47x the top-4 baseline and 1.23x the quality-matched one, while scoring *below* both (87.5% / 64.3%). There is no Synapse configuration on this corpus whose context is smaller than the baseline it would have to beat.

## 6. What GraphRAG would have to achieve — the inversion

Since a finite Q\* requires CONTEXT(graph) < CONTEXT(vector), the useful question is not *when* does it amortize but *how much compression would it need to*. Rearranging:

```
  CONTEXT(graph) < CONTEXT(vector) − 1e6 · Δsetup / (p_in · Q)
```

With the quality-matched baseline context (418 tokens, top-8) and the measured setup cost ($0.007338), the graph's context budget at a given query volume is:

| Queries per corpus | Max context GraphRAG may use and still win | vs. its measured context |
| ---: | ---: | ---: |
| 10 | impossible — the setup cost alone exceeds 10 queries' worth of the baseline's entire context | — |
| 100 | impossible — the setup cost alone exceeds 100 queries' worth of the baseline's entire context | — |
| 1,000 | 370 tokens | needs 2.4x less than it uses |
| 10,000 | 414 tokens | needs 2.1x less than it uses |
| 100,000 | 418 tokens | needs 2.1x less than it uses |
| 1,000,000 | 418 tokens | needs 2.1x less than it uses |

Read the last column as the compression ratio GraphRAG owes. It is asymptotically **2.1x** — even with infinite queries to amortize over, GraphRAG must first stop being bigger than the baseline before the amortization argument can start. Note also how fast the setup term vanishes down the table: by 100,000 queries the one-off extraction bill has stopped mattering, and the answer is decided entirely by the per-query token ratio.

## 7. Break-even as a function of corpus size

Extraction scales linearly with corpus tokens; the retrieved context does **not** grow with the corpus (top-k and k-seeds are fixed budgets). So in the counterfactual world where GraphRAG *does* compress by a factor ρ = CONTEXT(graph)/CONTEXT(vector) < 1, the break-even is

```
  Q*(N, ρ) = setup_per_token · N / ( p_in · CONTEXT(vector) · (1 − ρ) / 1e6 )
```

Measured setup cost per corpus token: **$0.00000431/token** ($4.31 per 1M corpus tokens) at `gpt-4o-mini` with local embeddings.

**Q\*(corpus size, ρ)** — queries needed before the graph pays for itself. ρ = 2.10 is the MEASURED value (D against the quality-matched baseline); every ρ < 1 column is a COUNTERFACTUAL, shown to bound what compression would have to buy.

| Corpus | Corpus tokens | Setup (est.) | ρ=0.25 | ρ=0.50 | ρ=0.75 | ρ=0.90 | ρ=2.10 *(measured)* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 CV / résumé (~4 KB) | 1,000 | $0.0043 | 91 | 137 | 274 | 686 | **never** |
| benchmark fixture (this corpus) | 1,704 | $0.0073 | 156 | 234 | 468 | 1,169 | **never** |
| 1 report (~50 KB) | 12,500 | $0.0538 | 1,143 | 1,715 | 3,430 | 8,576 | **never** |
| 1 book (~500 KB) | 125,000 | $0.5383 | 11,435 | 17,152 | 34,305 | 85,762 | **never** |
| small wiki (~5 MB) | 1,250,000 | $5.3828 | 114,350 | 171,525 | 343,049 | 857,623 | **never** |
| corpus (~50 MB) | 12,500,000 | $53.8281 | 1,143,498 | 1,715,247 | 3,430,494 | 8,576,235 | **never** |

Two things fall out, and the second is the counterintuitive one:

1. **At the measured ρ the entire right-hand column is `never`.** No corpus size rescues the graph, because the per-query term has the wrong sign.
2. **Q\* grows *linearly* with corpus size.** The usual intuition — "the graph pays off on big corpora" — is backwards *in this model*: extraction cost scales with N while the per-query saving is capped by a fixed context budget, so a 100x bigger corpus needs 100x more queries to amortize. What actually rescues GraphRAG on a large corpus is not size itself but the possibility that ρ falls — that vector RAG needs a much wider top-k to reach the same recall when the corpus is large. **This experiment did not measure that, and the model cannot settle it.**

## 8. Price sensitivity

The conclusion is about the *sign* of the per-query difference, so it is invariant to price — but the size of the one-off bill is not. Same corpus, every priced chat model:

| Chat model | $/1M in | $/1M out | Setup (est.) | Extra cost per 1,000 queries vs A\* (top-8) |
| --- | ---: | ---: | ---: | ---: |
| `gpt-4.1` | 2.00 | 8.00 | $0.097838 | +$0.9220 |
| `gpt-4.1-mini` | 0.40 | 1.60 | $0.019568 | +$0.1844 |
| `gpt-4.1-nano` | 0.10 | 0.40 | $0.004892 | +$0.0461 |
| `gpt-4o` | 2.50 | 10.00 | $0.122298 | +$1.1525 |
| `gpt-4o-mini` | 0.15 | 0.60 | $0.007338 | +$0.0691 |

The last column is positive for every model: GraphRAG is more expensive per query regardless of price point, because it is more expensive in *tokens*.

## 9. Wall-clock, not just money

| Stage | Seconds | Measured or estimated |
| --- | ---: | --- |
| Embed 79 entities + write 79 nodes / 81 edges to Neo4j | 1.7 | MEASURED (fastembed, local) |
| Embed 127 documents for the benchmark's own retrieval | 0.6 | MEASURED (model already warm — the row above paid the one-off fastembed model load, so these two rows are not comparable to each other) |
| Store 34 chunks + entity links | 0.7 | MEASURED |
| Extraction: 9 LLM calls at concurrency 5 | ~5.4 | ESTIMATED at 3.0s/call — NOT measured, no LLM was called |

Even on this tiny corpus the estimated LLM latency (~5s) is 1.8x the entire measured non-LLM ingestion (3.0s). Latency has the same shape as cost: **the extraction LLM dominates, everything else is noise.** Vector RAG's ingestion is the 0.6s embedding pass and nothing else.

## 10. The rule of thumb

> **Do not build a knowledge graph to save money. It does not.**
>
> On this corpus GraphRAG costs $0.0073 up front *and* $0.00006915 more per query than a plain top-8 passage baseline that already equals or beats it on both retrieval metrics (96.4% / 92.9% vs 94.6% / 85.7%). The break-even query count is infinite: there is no N at which the graph becomes the cheaper choice.
>
> The practitioner's version: **if your reason for building a graph is cost, stop.** Build one only if you want something vector RAG cannot do at any budget — explicit multi-hop paths, an auditable reasoning chain, a browsable entity view, global community summaries. Those are product features, and you should price them as features, not as savings.

A crude budgeting heuristic that does fall out of the measured numbers:

* **One-off:** ~$4.31 per 1M corpus tokens at `gpt-4o-mini` — about $0.0043 for a résumé, $0.54 for a book. Cheap in absolute terms at the small end; this is why demo projects never notice.
* **Recurring:** ~+461 prompt tokens on *every* query, forever. At 1M queries that is $69.15 of pure surcharge — 9,424x the one-off extraction bill for this corpus.
* **The crossover in *queries*, not dollars:** the surcharge overtakes the entire one-off extraction bill after only 106 queries. That is the honest form of the "break-even" practitioners are looking for — except it runs the wrong way: it is the point after which the graph's *recurring* cost, not its setup cost, is the thing you are paying for.
* **Therefore:** the extraction cost is the part everybody worries about and it is the part that does not matter. The context surcharge is the part nobody budgets for and it is unbounded in the number of queries.

## 11. Threats to validity

1. **n = 14 questions, one corpus, 6,816 characters.** This is a fixture, not a workload. Every context measurement carries the bootstrap CI printed in §2.4; the corpus itself has no error bar at all because there is only one of it. Nothing here generalises to a different corpus without re-measuring.
2. **The corpus saturates, and this is the load-bearing threat.** `benchmarks/results.md` already notes that top-20 reads 59% of this corpus and scores 100%. A tiny corpus flatters top-k retrieval, and top-k is the system GraphRAG loses to here. The whole result rests on ρ = 2.10 > 1; on a large corpus where a passage baseline needed a much wider k to reach the same multi-hop recall, the quality-matched baseline's context would grow, ρ could fall below 1, and a finite Q\* would appear. **This is the single most likely way the conclusion is wrong, and this experiment cannot rule it out** — it would take a large-corpus study (e.g. the HotpotQA harness) to settle, and that costs real LLM money.
3. **Tokens are estimated, not tokenized.** Everything uses the repo's `len/4` rule (`cost.CHARS_PER_TOKEN`). A real BPE tokenizer would shift absolute counts by ~10-20%, and would shift both systems in the same direction. It would not flip a 2.10x ratio.
4. **Prices are a dated hand-recorded table** (2026-07-21, https://openai.com/api/pricing/) and are not fetched live. §8 shows the conclusion is invariant to the price point, but the absolute dollars are estimates and must be checked against an invoice before quoting.
5. **Answer length (250 tokens) and community-summary length (120 tokens) are assumptions.** The first cancels out of every difference. The second affects only the community term, which is 17.6% of setup.
6. **Extraction completion size is derived from the fixture, not observed.** The fixture is a clean, hand-checked extraction; a real LLM emits more slop, more duplicates and retried calls. That biases the setup cost **downward** — i.e. in GraphRAG's favour — so it cannot explain away the null result.
7. **Extraction wall-clock is an estimate** at 3.0s/call. No LLM was called. Treat §9's last row as an order of magnitude only.
8. **This measures retrieval cost, not answer quality.** A larger context is not automatically worse for the generator — it may produce better answers, or worse ones through distraction. This experiment prices tokens; it does not judge answers. `benchmarks/results.md` shows GraphRAG does not *retrieve* better here, which is what makes the extra tokens hard to justify — but an end-to-end answer-quality study could still find value the retrieval metrics miss.
9. **One retrieval configuration.** k=8 seeds, 2 hops, chunk_top_k=4. A GraphRAG tuned for terseness (fewer seeds, 1 hop, no chunks) would move ρ. C is the closest thing to that here and still sits at ρ = 1.23 against the quality-matched baseline — while scoring below it, so it is not a configuration that would rescue the argument.
10. **`benchmarks/public/PILOT.md` did not exist when this ran**, so no HotpotQA cost figures could be cross-checked. The only external cross-check is `benchmarks/results.md` (§2.4), which agrees.

## 12. Measured vs. inferred

| | |
| --- | --- |
| **MEASURED** | corpus and chunk token counts; extraction and RAG prompt template sizes; the extraction JSON payload size; the Louvain partition and its summary prompt sizes; per-question retrieved-context sizes for A, A\*, C and D, and the retrieval quality of each; non-LLM ingestion wall-clock. |
| **DERIVED** (arithmetic on measurements + published prices) | every USD figure; the per-query difference and its bootstrap CI; Q\*; the inversion table; the ρ-scaling table. |
| **ASSUMED** (stated, and the sensitivity shown) | answer length; community-summary length; extraction latency; that entity density stays constant as corpora scale; that retrieved-context size is independent of corpus size. |
| **NOT MEASURED** | answer quality; whether ρ falls below 1 on a large corpus; real provider tokenization; real extraction slop and retry rates. |

## 13. Claim we could defend in a paper

> On a 34-passage multi-hop benchmark, GraphRAG's cost disadvantage is **not** a one-off extraction charge that amortizes. The shipped system retrieves 2.10x the prompt tokens of a passage baseline that already equals or beats it on both retrieval metrics (top-8; paired bootstrap over 14 questions, 95% CI on the per-query context-token difference [-491, -432], entirely on the wrong side of zero). Because the break-even query count is Δsetup divided by that difference, it is **infinite rather than large**: there is no query volume at which the graph becomes cheaper. The practical consequence is that extraction cost — the term practitioners budget for, here $0.0073 — is swamped after roughly 106 queries by a per-query context surcharge that grows without bound in query volume.

*(Scope: one small corpus (6,816 chars), n = 14 questions, one retrieval configuration, token counts estimated at 4 chars/token. The claim is about this corpus and the structure of the cost model, not about GraphRAG in general. §11.2 states the condition under which it would flip.)*

---

Numbers generated by `research/rq5_cost_amortization/run_rq5.py`; raw values in `results.json`. Synapse, © 2026 Ahmed Maaloul, AGPL-3.0-or-later.
