# GraphRAG vs. vector RAG — a head-to-head benchmark

A GraphRAG project that never measures itself against plain vector RAG is asking
you to take its central claim on faith. This directory is the attempt not to.

```bash
docker compose up -d neo4j
make benchmark          # no API key, no network
```

The run writes [`results.md`](results.md) — the full generated report — and the
generated block below. Nothing numeric in this file is typed by a human.

> **The result, stated up front, because it is not the flattering one.**
>
> **On this corpus a plain passage baseline is competitive with, and at a
> matched context budget better than, Synapse GraphRAG — under both scoring
> rules, and more decisively under the strict one.** The shipped system (row
> **D**) does beat the passage baseline at the configured `k`, but the sweep
> shows a *cheaper* baseline setting reaching the same scores on a fraction of
> the text, and tightening the scoring rule makes that baseline setting cheaper
> still. The exact k's and shares are in the generated block; they are computed
> by `headline()`, which is not allowed to claim a win the sweep contradicts.
>
> Two further findings the author would rather not have had to publish:
>
> - **The scoring rule this benchmark used to call fair was not neutral.** It
>   credits a required fact whose *name* appears anywhere in the context, and a
>   GraphRAG context prints entity names as scaffolding. Measured, not
>   estimated: nearly half of the graph-only row's credited facts come from a
>   name appearing **only** in a relationship line or a reasoning path — the
>   graph mentioned the entity as a neighbour, and the evidence needed to answer
>   was never retrieved. The passage baseline has no such channel. See *Where
>   the scored hits come from* in [`results.md`](results.md).
> - **Under the strict rule the graph-only row scores zero**, as do all three
>   entity-granular rows, because they return a compressed representation of the
>   corpus rather than the corpus. That is a property of what those systems
>   *are*, not a bug in the run — but it means the graph ablation (C − B′) only
>   exists under the permissive rule, and the report now says so where the
>   ablation is stated.

## The numbers — generated, never typed

Everything between the markers is written by `run_benchmark.py` into *both* this
file and [`results.md`](results.md), from one function. A previous version of
this README promised its numbers were "copied from the generated report and
reproduced by re-running the script"; an audit found several of them stale. The
promise has been replaced by a mechanism, and the mechanism by a test:
`test_benchmark.py` fails if this region is missing, if it is not present
verbatim inside `results.md`, or if any *other* number anywhere in this file
cannot be found in `results.md`.

<!-- BEGIN GENERATED — written by run_benchmark.py, do not edit by hand -->

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

<!-- END GENERATED -->

Per-question breakdowns, both k sweeps, the channel attribution and the graph
ablation are in [`results.md`](results.md).

---

## What is being measured

**Retrieval, not generation.** The systems differ in *what context they put in
front of the model*; everything after that is the same model doing the same job.
Scoring retrieval directly makes the benchmark LLM-free, free to run and
reproducible in CI — an LLM-judged answer-quality score would be none of those
things, and would mostly measure the judge.

Each question in [`dataset.json`](dataset.json) lists `required_facts`: entity
names that must be in the retrieved context for the question to be answerable at
all.

| Metric | Meaning |
| --- | --- |
| **Fact Recall** | Mean fraction of required facts retrieved. |
| **Full-Coverage Rate** | Fraction of questions where *all* required facts were retrieved. This is the one that matters: a multi-hop question answered from three of its four facts is answered wrongly, with confidence. |
| **Mean context size** | Characters handed to the generator, so a win cannot be bought by simply returning more text. |

### The scoring rule is not neutral — so both rules are reported

An earlier version of this file defended the metric with "the identical rule is
applied to every system". That defends the rule's *uniformity*. It says nothing
about its *neutrality*, and the two are not the same thing, because the systems
put radically different strings in front of the rule:

- The **passage** systems return corpus prose and nothing else. A required fact
  scores only when its name occurs in text they actually read.
- The **graph** systems inject entity names as structured context — `Entity:`
  headers, `Description:` lines, `→ REL → Neighbour` relationship lines, and a
  rendered reasoning-path section. A required fact can score there because the
  graph *listed it as a neighbour of something else*, with none of the evidence
  needed to answer it retrieved.

That is a systematic advantage to the graph rows, and this benchmark used to
give it away for free without sizing it. It is now measured two ways, both
generated:

- **Permissive rule** (as before) — the name appears anywhere in the context.
- **Strict rule** — the name appears in *retrieved prose*: verbatim corpus text
  under `Source excerpts:`, with the `[Sn]` provenance headers excluded.

And every credited fact is attributed to the channel it was found in — prose, a
materialised entity block, or an edge/reasoning-path mention and nothing else —
with the split printed per system. Attribution is deliberately conservative
*against* the criticism: a fact present in both prose and scaffolding is
credited to the prose.

**Which rule favours whom, plainly.** The permissive rule is worth nothing to
the passage rows, because everything they return is prose and the two rules
score them identically. It is worth a small amount to the shipped system, whose
credited facts come overwhelmingly from its excerpt channel. It is worth the
*entire score* of the entity-granular rows and of the graph-only row, which
retrieve no prose at all and therefore score zero when it is withdrawn. The
per-system point deltas are generated into *How much the permissive rule is
worth* in [`results.md`](results.md).

**Neither rule is the truth.** The permissive rule over-credits the graph: a
neighbour mention is not evidence. The strict rule under-credits it: entity
descriptions are distillations of the corpus and do carry information a model
could answer from, and the strict rule ignores them entirely. Read permissive as
an upper bound and strict as a lower bound. Actual answerability is between
them, and this benchmark does not measure it — see *Threats to validity*.

## The systems

| | System | What it is |
| --- | --- | --- |
| **A** | Passage vector RAG (top-4) | The naive baseline. Embed each passage whole, embed the question, take the top-k by cosine, concatenate. |
| **A′** | Passage vector RAG, budget-matched | The same baseline at the smallest `k` whose mean context reaches D's. Computed from the sweep. |
| **B** | Entity vector RAG, no edges (top-8) | Same retrieval *granularity* as GraphRAG — entity documents, `k=8` — with no graph at all. |
| **B″** | Entity vector RAG, budget-matched | B at the smallest `k` whose mean context reaches C's. The ablation's edgeless side, given equal budget. |
| **B′** | GraphRAG seeds, graph stripped (k=8) | C's own context with every edge-derived part deleted. |
| **C** | Synapse GraphRAG, graph only (k=8) | `chat_engine.retrieve_subgraph`, unmodified, with no source chunks stored. What Synapse was before source chunks. |
| **D** | Synapse GraphRAG + source chunks (k=8) | The shipped system: the same retrieval with the corpus stored as `:Chunk` text units linked to the entities extracted from them. |

**Why B, B″ and B′ exist.** A − D changes several things at once: graph
traversal, retrieval granularity, *and* the excerpt channel. A reader is
entitled to answer "you compared a handful of chunks with entity records plus
their neighbours plus some passages", and they would be right. **C − B′ is the
controlled comparison**: same seeds, same order, same per-entity blocks, the
relationships the only variable. B is the same idea as a standalone system that
needs no Neo4j; B″ is B given C's character budget.

**What C − B′ does *not* control.** It holds the seed set fixed, which is what
makes it the clean ablation — but deleting the edges also deletes their
characters, so B′ reads *less* text than C. **C − B′ is seed-matched, not
budget-matched.** An earlier version of this README claimed the ablation "is now
budget-matched"; that was false, and the sentence is deleted rather than
softened. B″ is the budget-matched edgeless row, and the generated ablation
block states the residual character gap on its own line.

**What B′ actually removes.** C's context contains per-entity `Relationships:`
lines *and* a rendered `Reasoning paths:` section. Both are built from edges, so
B′ deletes both, by literal text surgery on the context C returned. The test
`test_strip_graph_structure_removes_edges_and_paths_and_nothing_else` pins what
the surgery does, and a companion test pins that the surgery's complement —
`edge_segment` — together with the prose segment covers every entity name in the
context, so no credited fact can escape attribution.

The passages are already paragraph-sized (one to three sentences), which is the
*best case* for chunking — the baseline is not handicapped by a bad splitter.
Every system uses the same embedding model (`fastembed` /
`BAAI/bge-small-en-v1.5`), so nobody gets a better encoder.

The extraction ships pre-computed in `dataset.json` rather than being run by an
LLM. That is not a shortcut around the hard part — it is what makes the
comparison *about retrieval*. If a model extracted the graph at benchmark time,
every run would score a different graph and nothing would be reproducible.

## The dataset

Corpus size, entity count, relationship count and question count are printed in
the first line of [`results.md`](results.md), and every structural claim below is
enforced by a test in `tests/test_benchmark.py` rather than asserted here.

- The **core passages** carry every fact the questions need.
- The **distractor passages** are on adjacent topics (ENIAC, Colossus, ARPANET,
  AlphaGo, FAISS…). They exist because corpus size is what gives a retrieval
  benchmark its discriminating power: on a corpus of core passages alone, the
  configured `k` is a large share of everything that exists, every question is
  trivially coverable, and the systems tie at the ceiling. That is an artifact
  of a toy corpus, not a finding.
- **Some distractors do mention a required entity in passing.** An earlier
  version of this README claimed the distractors "carry none of" the required
  facts; that was false. The exact list lives in `dataset.json` under
  `distractors_leaking_required_facts` and is pinned by a test, so it cannot
  drift again. A test also asserts that no distractor is ever *needed* to answer
  a question.

  The same README went on to say a leaked mention "can only ever help the
  passage baseline, never GraphRAG". **That is also false and is deleted.** The
  shipped system's excerpt channel retrieves the same passages the baseline
  ranks, so a leaked mention can help any system that returns prose — A, A′ and
  D alike. It cannot help the rows that return no prose (B, B″, B′, C).

### Fairness rules, all enforced by tests

**R1 — a question names at most one of its required facts** (its *anchor*), and
shares no content-word *family* with the **name** or the **description** of any
other required fact. The description half is the one that matters: descriptions
are what GraphRAG embeds and full-text-indexes, so a description echoing the
question's distinctive words hands over a required fact with no edge walked at
all.

*Family*, not *stem*. Stemming only ever collides words that differ at the end
("train"/"training"), and an auditor pointed out that the leaks actually present
in this dataset differed in the middle — a question asking where a **dataset**
was assembled, next to a required fact described as directing an image
**database**. Two stems now count as one family when they share a prefix of
`PREFIX_FAMILY_LEN` characters, which catches that class. Violations surfaced
when the rule was tightened, and they were fixed in the dataset rather than the
rule being loosened; the rule runs as a test, so the shipped dataset is clean by
construction.

What R1 cost GraphRAG's numbers is deliberately **not** quoted. A previous
version of this README gave a precise before/after for the rewrite — against a
dataset version that exists nowhere in this repository, so nobody could check
it. An unverifiable number in a credibility document is worse than no number.

**R2 — no single passage contains all of a question's required facts.** This is
the definition of multi-hop, and it is checked exhaustively rather than asserted
in prose.

**R3 — no entity description names another entity.** Otherwise "Charles
Babbage — Victorian inventor who designed the *Analytical Engine*" would credit
a second required fact to anything that retrieved Babbage, and the edgeless
systems would be scoring the graph's job. This rule is conservative *against*
GraphRAG: a real LLM extraction does mention related entities in descriptions,
so the graph here has to carry relationships entirely on its own.

### R1 is asymmetric, and the asymmetry is disclosed

R1 is enforced against entity **names** and **descriptions** — the graph-side
text. It is **not** enforced against **passage text**. A question is therefore
free to share distinctive words with the very passages that contain its required
facts, and in this dataset it does; a test measures that this overlap is real
rather than hypothetical, so the disclosure cannot quietly become false.

This cuts *against* GraphRAG, not for it: the passage baseline's retriever is
allowed lexical and semantic help that the graph's seed search is forbidden.
Symmetrising R1 would mean rewriting the corpus until no question resembled the
prose that answers it, which would make the passage-retrieval task artificial in
a different direction. The choice is defensible; leaving it implicit was not.

`tests/test_benchmark.py` additionally verifies that every declared mention
really occurs in the passage text, that declared mentions are *exactly* the
detected ones (no inflation, no omission), that no entity name is a substring of
another **and never matches inside a longer word** (the scoring rule is a
substring test, so substrings must not lie), that every relationship endpoint is
a declared entity, and that the graph is connected.

## Reading the results honestly

- **The headline is computed, not written.** `headline()` refuses to claim a win
  on any metric that a cheaper baseline setting reaches; `verdict()` prints, per
  metric and per scoring rule, the smallest baseline `k` that matches or beats
  the shipped system and what that costs in characters. Both currently print
  that a plain passage baseline matches D on *every* metric while reading less
  text, and those sentences are generated, not conceded.
- **D contains the baseline.** `chunk_store.search_chunks` is a cosine search
  over the same passages A ranks, with the same embedder. The report measures
  the overlap and prints it. Only the *structural* channel — chunks reached
  through the entities the graph selected, including multi-hop ones — is
  something plain vector RAG cannot do. Read **D vs. A′** and **D vs. C**; do
  not read D vs. A as a result.
- **The corpus is small, so a "budget-matched" baseline is reading most of it.**
  A′ reads a large share of the corpus, where it trivially saturates. That is an
  upper bound on what top-k can do here, not a deployable configuration, and the
  generated scale caveat says so in those words. The whole-corpus row at the end
  of each sweep exists to make the ceiling explicit instead of letting a
  truncated sweep imply one.
- **Per-question failures are printed for every system, including the shipped
  one, and the summary sentence is generated from the same table.** An earlier
  version of this README asserted that three named questions explained
  GraphRAG's misses while the generated table right below it showed five. No
  count and no question id is typed here any more: `misses_note()` produces the
  sentence from the results themselves — see **Where retrieval still fails** in
  [`results.md`](results.md), which also lists D's misses under the strict rule.
- **The remaining failures have two identifiable causes**, both visible by
  re-running `retrieve_subgraph` on the questions the generated sentence names:
  - *One hop too far.* `_expand_and_format` materialises seeds plus their
    **immediate** neighbours, while `retrieval_max_hops` only widens the
    reasoning *paths* between seeds. A required fact two hops from every seed is
    never written into the context unless a source chunk happens to carry it.
  - *Bad seeding.* On at least one question the hybrid seed search returns
    entities of which only the anchor is relevant. No traversal recovers a fact
    whose region of the graph was never seeded — and the passage baseline misses
    that question too, so it is a hard question rather than a graph-specific
    failure.
- **Scores are stable across runs; the character counts wobble.** Neo4j's vector
  index is approximate and its state depends on write history, so the mean
  context column moves slightly between runs of the same code against the same
  data, while the score columns have not. If your run differs from `results.md`
  in the context column by a few characters, that is why; if it differs in the
  score columns, something is actually wrong. This paragraph is a qualitative
  observation, not a measurement: no run-to-run variance study is shipped, and
  an earlier version of this file quoted a five-run tally that nothing in the
  repository reproduces.

A benchmark that always flatters its author is worth nothing to a reader, which
is the opposite of the point.

## Threats to validity

Stated by the author, before a referee has to find them.

- **One small corpus.** 34 passages, 14 questions, one domain (the history of
  computing and AI), one author. These results are **illustrative, not
  general**. Nothing here licenses a claim about GraphRAG or vector RAG in
  general, on other corpora, at other scales, or in other domains. A corpus this
  small also saturates: the budget-matched baseline reads most of the text that
  exists, which is why its scores hit the ceiling.
- **The questions were authored by the same process that built the graph.** The
  required facts are entity names *from the extraction*, and the questions were
  written to be multi-hop *over that graph*. Fairness rules R1–R3 constrain the
  most obvious leaks, but they cannot remove the fact that the evaluation set and
  the retrieval index share an author and a construction. An independently
  authored question set over an independently extracted graph would be a
  stronger design, and is not what this is.
- **String containment is a proxy for answerability, not answerability.** Both
  rules ask whether an entity *name* appears in the context. Neither asks
  whether the retrieved text actually supports the answer, and no LLM judges
  anything here. A context can contain every required name and still fail to
  license the inference; a context can lack a name and still support the answer
  through a paraphrase. The permissive/strict pair brackets the error rather
  than eliminating it.
- **The embedding model is held constant, so the results may not transfer.**
  Every system uses `BAAI/bge-small-en-v1.5` via `fastembed`. Holding it fixed
  is what makes the comparison internal-validity-clean, and is also why nothing
  here predicts what happens with a stronger encoder — a better retriever
  plausibly helps the passage baseline more than the graph, since the baseline's
  entire performance is retrieval ranking.
- **No statistical significance is claimed.** Fourteen questions. **There is no
  significance test in this benchmark, and none is implied by any number in it.**
  Differences of a few percentage points correspond to a fraction of one
  question and should be read as noise. Confidence intervals over a set this
  small would be wide enough to contain every system's score, which is why none
  are quoted rather than quoted misleadingly.
- **The strict rule is conservative against the graph, by design.** It scores
  entity descriptions as non-evidence, though they are distillations of corpus
  prose and a model could answer from them. Someone who considers descriptions
  to be evidence should read the permissive column. The point of shipping both
  is that no reading of this table depends on the author's choice of rule.
- **The ablation is rule-dependent.** C − B′ measures the relationships'
  contribution under the permissive rule only. Under the strict rule both sides
  score zero and the difference is undefined. Any statement of the form "the
  graph is worth N points" in this repository is a permissive-rule statement.

## Files

| File | Purpose |
| --- | --- |
| `dataset.json` | Corpus, pre-computed extraction, questions, required facts |
| `run_benchmark.py` | Harness — seeds Neo4j, runs all five systems (seven rows), scores both rules, prints and writes the report, regenerates this file's generated block |
| `results.md` | Generated report (regenerate with `make benchmark`) |
| `../tests/test_benchmark.py` | Hermetic tests: dataset integrity, metric correctness, channel attribution, and the README/results consistency check |

## Notes

- Seeding **replaces** the `:Entity` / `:Community` / `:Chunk` contents of the
  target Neo4j database. Restore the demo graph afterwards with `make
  demo-local`. Chunks are cleared too, and the harness *verifies* that the
  graph-only row returned no excerpts — a leftover demo `:Chunk` would otherwise
  be retrieved by the production path and silently inflate row C.
- The harness refuses to report a run whose channel attribution disagrees with
  the scoring rule, because a hit it cannot place would be silently dropped from
  the strict column.
- `python -m benchmarks.run_benchmark --help` for the flags and the exit codes.

---

Synapse — © 2026 Ahmed Maaloul — AGPL-3.0-or-later
