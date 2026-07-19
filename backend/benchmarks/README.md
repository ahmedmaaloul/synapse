# GraphRAG vs. vector RAG — a head-to-head benchmark

A GraphRAG project that never measures itself against plain vector RAG is asking
you to take its central claim on faith. This directory is the attempt not to.

```bash
docker compose up -d neo4j
make benchmark          # ~6 s, no API key, no network
```

Results land in [`results.md`](results.md) and on stdout.

**Latest numbers are in [`results.md`](results.md)** — that file is generated, so
it is always the run, never a claim typed by hand. Every number quoted in *this*
file is copied from that generated report and is reproduced by re-running the
script.

> **The result, stated up front, because it is still not the flattering one.**
>
> The shipped system — Synapse GraphRAG **plus** the source chunks the entities
> were extracted from (row **D**) — scores **94.6% fact recall / 85.7% full
> coverage on 3,522 characters** of context. It beats a top-4 passage baseline
> (89.3% / 71.4%). It **does not beat a top-8 passage baseline, which scores
> 96.4% / 92.9% while reading 1,673 characters — 47% of D's context.** On this
> corpus, plain vector RAG is still ahead of Synapse at less than half the
> budget, and no amount of framing changes that.
>
> What *is* now established, and was the point of this wave:
>
> - **Source chunks fixed the regression they were meant to fix.** Graph-only
>   retrieval (row **C**) scores 87.5% / 64.3%; adding the text units takes it to
>   94.6% / 85.7% — **+7.1 pp recall, +21.4 pp coverage** for 1.71x the context.
>   The old "GraphRAG loses to plain passage RAG on raw recall" gap is closed at
>   the configured k. It is *not* closed against a budget-matched baseline.
> - **The graph itself carries most of what GraphRAG retrieves, and it is now
>   budget-matched.** Deleting every edge-derived part of C's context costs
>   **−39.4 pp recall and −57.1 pp coverage** (row **B′**). Given C's *own*
>   character budget, the edgeless entity system still only reaches 63.0% / 21.4%
>   (row **B″**); it needs **top-48 of 79 entities — 2.6x C's context** — before
>   it reaches C's recall.
> - **Part of D's gain is arithmetic, not evidence.** D's semantic excerpt
>   channel is a top-k cosine search over exactly the passages system A ranks.
>   Measured: **100% of A's top-4 passages also appear among D's excerpts.** D
>   contains the baseline. Read D vs. A′ and D vs. C; do not read D vs. A as a
>   result.

---

## What is being measured

**Retrieval, not generation.** The systems differ in *what context they put in
front of the model*; everything after that is the same model doing the same job.
Scoring retrieval directly makes the benchmark LLM-free, free to run and
reproducible in CI — an LLM-judged answer-quality score would be none of those
things, and would mostly measure the judge.

Each question in [`dataset.json`](dataset.json) lists `required_facts`: entity
names that *must* be in the retrieved context for the question to be answerable
at all. A fact counts as retrieved when its name occurs (case-insensitively) in
the retrieved text. **The identical rule is applied to every system.**

| Metric | Meaning |
| --- | --- |
| **Fact Recall** | Mean fraction of required facts retrieved. |
| **Full-Coverage Rate** | Fraction of questions where *all* required facts were retrieved. This is the one that matters: a multi-hop question answered from three of its four facts is answered wrongly, with confidence. |
| **Mean context size** | Characters handed to the generator, so a win cannot be bought by simply returning more text. |

## The systems

| | System | What it is |
| --- | --- | --- |
| **A** | Passage vector RAG (top-4) | The naive baseline. Embed each passage whole, embed the question, take the top-k by cosine, concatenate. |
| **A′** | Passage vector RAG, budget-matched | The same baseline at the smallest `k` whose mean context reaches D's. Computed from the sweep. |
| **B** | Entity vector RAG, no edges (top-8) | Same retrieval *granularity* as GraphRAG — entity documents, `k=8` — with no graph at all. |
| **B″** | Entity vector RAG, budget-matched | B at the smallest `k` whose mean context reaches C's. The ablation's edgeless side, given equal budget. |
| **B′** | GraphRAG seeds, graph stripped (k=8) | C's own context with every edge-derived part deleted. |
| **C** | Synapse GraphRAG, graph only (k=8) | `chat_engine.retrieve_subgraph`, unmodified, with no source chunks stored. What Synapse was before this wave. |
| **D** | Synapse GraphRAG + source chunks (k=8) | The shipped system: the same retrieval with the corpus stored as `:Chunk` text units linked to the entities extracted from them. |

**Why B, B″ and B′ exist.** A − D changes several things at once: graph
traversal, retrieval granularity, *and* the excerpt channel. A reader is
entitled to answer "you compared 4 chunks with 8 entity records plus their
neighbours plus 6 passages", and they would be right. **C − B′ is the controlled
comparison**: same seeds, same order, same per-entity blocks, the relationships
the only variable. B is the same idea as a standalone system that needs no
Neo4j; B″ is B given C's budget, so the ablation is not a comparison between
unequal amounts of text.

**What B′ actually removes.** C's context contains per-entity
`Relationships:` lines *and* a rendered `Reasoning paths:` section. Both are
built from edges, so B′ deletes both, by literal text surgery on the context C
returned. An earlier version of this file said the relationship lines were "the
only thing taken away"; that was false, and the test
`test_strip_graph_structure_removes_edges_and_paths_and_nothing_else` now pins
what the surgery does. **C − B′ is therefore the contribution of the
relationships**, not of the relationship lines alone.

The passages are already paragraph-sized (1–3 sentences), which is the *best
case* for chunking — the baseline is not handicapped by a bad splitter. Every
system uses the same embedding model (`fastembed` / `BAAI/bge-small-en-v1.5`),
so nobody gets a better encoder.

The extraction ships pre-computed in `dataset.json` rather than being run by an
LLM. That is not a shortcut around the hard part — it is what makes the
comparison *about retrieval*. If a model extracted the graph at benchmark time,
every run would score a different graph and nothing would be reproducible.

## Retrieval budgets, in one place

| Side | Budget | Share of what exists |
| --- | --- | --- |
| Passage baseline (A) | 4 of 34 passages | ~12% of the corpus |
| Graph only (C) | 8 seed entities of 79, plus each seed's immediate neighbours | ~10% seeded, more materialised |
| Shipped (D) | C, plus up to `chunk_top_k=4` excerpts per channel (structural + semantic), capped at `chunk_context_max_chars` | 52% of the corpus by characters |

Because those are not equal in characters, **every** comparison in the report is
budget-matched, and both sweeps now run all the way to *k = everything that
system has* (all 34 passages, all 79 entity records). That last row is what makes
a budget match always reachable: an earlier version stopped the sweep at k=12 —
where, in the current run too, the edgeless entity system reads 1,367 characters
against C's 2,060, i.e. **66%** of the context it was being compared with — and
then compared them as though that were equal. It no longer can; `sweep_ks()` is
tested to end at the whole corpus.

## The dataset

34 passages on the history of computing and AI, 79 entities, 81 relationships,
14 questions.

- **18 core passages** carry every fact the questions need.
- **16 distractor passages** are on adjacent topics (ENIAC, Colossus, ARPANET,
  AlphaGo, FAISS…). They exist because corpus size is what gives a retrieval
  benchmark its discriminating power: on an 18-passage corpus, `k=4` is 22% of
  everything that exists, every question is trivially coverable, and the systems
  tie at ~100%. That is an artifact of a toy corpus, not a finding.
- **Eight of those distractors do mention a required entity in passing** — IBM
  in `d04`/`d05`, Stanford University in `d07`, Deep Blue in `d10`, Google in
  `d11`, Retrieval-Augmented Generation in `d12`, GPU in `d13`, Transformer in
  `d14`. An earlier version of this README claimed the distractors "carry none
  of" the required facts; that was **false**, and it is corrected here rather
  than papered over. The mentions are harmless and realistic: no distractor is
  ever *needed* to answer a question (asserted by a test), and a leaked mention
  can only ever help the passage baseline — never GraphRAG. The exact list lives
  in `dataset.json` and is pinned by a test, so it cannot drift again.

### Three fairness rules, all enforced by tests

**R1 — a question names at most ONE of its required facts** (its *anchor*), and
shares no content-word *family* with the **name** or the **description** of any
other required fact. The description half is the one that matters: descriptions
are what GraphRAG embeds and full-text-indexes, so a description echoing the
question's distinctive words hands over a required fact with no edge walked at
all.

*Family*, not *stem*. Stemming only ever collides words that differ at the end
("train"/"training"), and an auditor pointed out that the leaks actually present
in this dataset differed in the middle — a question asking where a **dataset**
was assembled, next to a required fact described as directing an image
**database**; a question asking what Hinton popularised "**back** in 1986", next
to the answer **Backpropagation**. Two stems now count as one family when they
share a 4-character prefix, which catches all of those. Three violations
surfaced when the rule was tightened, and all three were fixed in the dataset
(Hollerith's and Fei-Fei Li's descriptions were reworded, and "back in 1986"
became "in 1986").

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

`tests/test_benchmark.py` additionally verifies that every declared mention
really occurs in the passage text, that declared mentions are *exactly* the
detected ones (no inflation, no omission), that no entity name is a substring of
another **and never matches inside a longer word** (the scoring rule is a
substring test, so substrings must not lie), that every relationship endpoint is
a declared entity, and that the graph is connected.

## Reading the results honestly

Six things worth knowing before you quote this benchmark:

1. **The headline is computed, not written.** `headline()` refuses to claim a
   win on any metric that a cheaper baseline setting reaches; `verdict()` prints,
   per metric, the smallest baseline `k` that matches or beats the shipped system
   and what that costs in characters. It currently prints that a plain passage
   baseline matches D on **every** metric while reading less text, and that
   sentence is generated, not conceded.
2. **D contains the baseline.** `chunk_store.search_chunks` is a cosine search
   over the same passages A ranks, with the same embedder. The report measures
   the overlap (100% of A's top-4 passages appear among D's excerpts; D returns
   5.6 excerpts per question, of which 1.6 are passages A did not retrieve).
   Only the *structural* channel — chunks reached through the entities the graph
   selected, including multi-hop ones — is something plain vector RAG cannot do.
3. **The corpus is small, so a "budget-matched" baseline is reading most of it.**
   A′ reads 20 of 34 passages: 60% of a 6,758-character corpus, where it
   trivially scores 100% / 100%. That is an upper bound on what top-k can do
   here, not a deployable configuration, and the report says so in those words.
   The whole-corpus row at the end of each sweep exists to make the ceiling
   explicit instead of letting a truncated sweep imply one.
4. **Per-question failures are printed for every system, including the shipped
   one, and the summary sentence is generated from the same table.** An earlier
   version of this README asserted that three named questions explained
   GraphRAG's misses while the generated table right below it showed five. That
   sentence is now produced by `misses_note()` from the results themselves, so
   the two cannot disagree again — see **Where retrieval still fails** in
   [`results.md`](results.md).
5. **The two remaining failures have two different causes**, both visible by
   re-running `retrieve_subgraph` on the questions the generated sentence names:
   - *One hop too far.* `_expand_and_format` materialises seeds plus their
     **immediate** neighbours, while `retrieval_max_hops` only widens the
     reasoning *paths* between seeds. A required fact two hops from every seed
     is never written into the context unless a source chunk happens to carry
     it.
   - *Bad seeding.* On one question the hybrid seed search returns eight
     entities of which only the anchor is relevant. No traversal recovers a fact
     whose region of the graph was never seeded — and the passage baseline misses
     that question too, so it is a hard question rather than a graph-specific
     failure.
6. **Scores are stable across runs; the character counts wobble.** This is the
   one paragraph here whose numbers are a *record* rather than something a
   re-run reproduces — which is the point of it. Across five consecutive runs of
   the final harness the two score columns never moved at all, and the mean
   context moved by at most one character (D: 3,522 in four runs, 3,523 in one).
   The very first run after the vector indexes were (re)created was further off
   — C 2,047 chars instead of 2,060, D 4,092 instead of 3,522 — again with
   **identical** scores, because Neo4j's vector index is approximate and its
   state depends on write history. So: if your run differs from `results.md` in
   the context column by a few percent, that is why; if it differs in the score
   columns, something is actually wrong.

A benchmark that always flatters its author is worth nothing to a reader, which
is the opposite of the point.

## Files

| File | Purpose |
| --- | --- |
| `dataset.json` | Corpus, pre-computed extraction, questions, required facts |
| `run_benchmark.py` | Harness — seeds Neo4j, runs all five systems (seven rows), prints and writes the report |
| `results.md` | Generated report (regenerate with `make benchmark`) |
| `../tests/test_benchmark.py` | Hermetic tests: dataset integrity + metric correctness |

## Notes

- Seeding **replaces** the `:Entity` / `:Community` / `:Chunk` contents of the
  target Neo4j database. Restore the demo graph afterwards with `make
  demo-local`. Chunks are cleared too, and the harness *verifies* that the
  graph-only row returned no excerpts — a leftover demo `:Chunk` would otherwise
  be retrieved by the production path and silently inflate row C.
- Exit codes: `0` success, `1` invalid dataset, `2` Neo4j unreachable, `3` the
  run produced something the harness refuses to report.
- `python -m benchmarks.run_benchmark --help` for `--baseline-k`, `--graph-k`,
  `--dataset`, `--no-write` and `--validate-only`.

---

Synapse — © 2026 Ahmed Maaloul — AGPL-3.0-or-later
