# Procedural Graphs on the GraphRAG Navigator: the benchmark harness

Synapse implements *Procedural Graphs* (Lu, Chen, Wu, Arık, "Procedural Graphs:
Self-Evolving Execution Structures for LLM Agents", Google,
[arXiv:2609.09153](https://arxiv.org/abs/2609.09153)) on top of its own GraphRAG
Navigator, a ReAct agent that answers questions by walking the knowledge graph
with deterministic tools (`app/services/graph_agent.py`). This directory holds
the harness that measures whether procedural guidance makes the Navigator
**better, or only more expensive**.

The paper reports both effects: guidance helps, and it costs tokens (Table 9,
HotpotQA: 10,116 vs 4,003 tokens per question for its best mode). It also reports
that evolution is no sure win: on HotpotQA, evolving the expert graph (mode 3,
the harness's `--evolve`) scored 76.34 answer F1 at 10,658 tokens per question,
below the unevolved expert graph (76.61 at 9,046). Synapse adds
two things the paper never measured, and both are **hypotheses, not claims**:

- **raw local guidance** (Synapse's default): the serialized 2-hop subgraph goes
  straight into the solver prompt, with zero extra LLM calls;
- **a localization cascade**: start → exact → normalized → semantic → full graph.

The harness is how they get tested. **No results are committed.** A run writes
`results_procedural.md` next to this file, which is gitignored because it
describes one run (model, date, spend), not the repository. So far the harness
has only been exercised by its hermetic tests (`tests/test_procedural_bench.py`,
fake models throughout). The first real run is yours, and it spends real LLM
calls, so start with `--dry-run`.

## Quick start

```bash
docker compose up -d neo4j
cd backend

# 1. Seed the zero-key demo graph (free: no LLM, only embeddings).
NEO4J_URI=bolt://localhost:7687 python -m scripts.seed_demo --clear

# 2. The bill, as an upper bound. Calls no model and touches no database.
python -m benchmarks.procedural.run_procedural --dry-run

# 3. The cheapest informative comparison: no graph vs Synapse's default.
NEO4J_URI=bolt://localhost:7687 python -m benchmarks.procedural.run_procedural \
    --systems no_pg,pg_raw_local
```

Seed and run with the same `EMBEDDING_PROVIDER`. `search_entities` compares the
query's embedding with the entity embeddings written at seed time.

The demo graph must hold the fixture **and nothing else**. When Neo4j also holds
other documents (more entities than the fixture's 53, or any source chunk), their
content surfaces in `search_entities` and `search_passages`, and the numbers stop
being about the demo set. The run is then refused with exit 4. `seed_demo --clear`
fixes it (free, but it deletes those other documents from the knowledge graph;
procedural memory is kept). `--allow-mixed-graph` runs anyway and marks the
report's provenance **MIXED**.

## What is compared

Every system is the same Navigator, with the same tools, the same model and the
same test questions. Only the Procedural Graph guidance changes.

| System | Graph | Guidance | What it stands for |
| --- | --- | --- | --- |
| `no_pg` | none | none | the paper's unguided baseline |
| `pg_raw_local` | expert prior | raw local subgraph | Synapse's default; the (subgraph × raw) cell missing from the paper's Table 3 |
| `pg_gen_local` | expert prior | generative, local scope | the paper's configuration: one guidance LLM call per step. On the prior, localization is the paper's exact match: every parsed action has a node, and a failed parse gets the full graph |
| `pg_gen_full` | expert prior | generative, full graph (forced) | the paper's scope ablation |
| `pg_evolved_raw` | evolved graph | raw local subgraph | with `--evolve ROUNDS` |
| `pg_evolved_gen` | evolved graph | generative, local scope | with `--evolve ROUNDS` |

The expert prior is the **bundled** JSON (`app/data/procedural/graphrag-navigator.json`),
loaded from disk. The stored graph is never used, because it may have been evolved
or hand-edited, and a benchmark row has to name a fixed input.

`--systems` takes any comma-separated subset. By default the four base systems
run, plus the two evolved ones when `--evolve` is given. The report compares
each pair below when both systems ran:

| Pair | Question it answers |
| --- | --- |
| `pg_raw_local` vs `no_pg` | does Synapse's default help at all? |
| `pg_gen_local` vs `no_pg` | does the paper's configuration help here? |
| `pg_gen_full` vs `no_pg` | does full-graph guidance help? |
| `pg_raw_local` vs `pg_gen_local` | is the guidance LLM needed, on the same subgraph? |
| `pg_gen_local` vs `pg_gen_full` | local vs full scope (the paper's ablation) |
| `pg_evolved_*` vs `pg_*_local` | did evolution beat the expert prior? |

## What is measured

The columns follow the paper's Table 9. Every figure is a per-question mean over
the **test split**:

- **EM and F1**: SQuAD/HotpotQA answer normalization, including the official
  yes/no rule (`app/services/qa_metrics.py`). A gold answer can list aliases,
  and the best match counts.
- **Steps, LLM calls, guidance LLM calls, parse failures, latency.**
- **Input and output tokens**, read from the provider's own usage metadata.
  When the provider reports nothing, tokens are estimated at chars/4 and marked
  `≈`.
- **Localization counts**: how often each method of the cascade fired, and how
  many characters of guidance were injected.

**The effect floor.** With N test questions, one question is worth 100/N
points (11.1 on the demo's 9). A gap below that is reported as **TOO CLOSE TO
CALL**, never as a difference. A gap of exactly one question is called, as a
*direction* rather than a finding. No significance test is run. This is the same
rule the retrieval benchmarks in `../` and `../public/` use.

## The datasets

### `--dataset demo` (the default): `demo_qa.json`

There are 30 questions over the zero-key demo graph (`backend/scripts/demo_graph.json`,
53 entities and 106 relations about the history of AI). Short exact answers,
names or years:

| Split | Questions | single | bridge | comparison |
| --- | ---: | ---: | ---: | ---: |
| train | 12 | 4 | 6 | 2 |
| val | 9 | 2 | 5 | 2 |
| test | 9 | 2 | 6 | 1 |

22 of the 30 questions (73%) need two or more graph facts: 21 of them need two,
and one question needs three hops.

The file is a JSON array. Each item carries `id`, `split`, `type`, `hops`,
`question` and `answer`, plus `evidence` (and `compare` for comparisons).
`answer` is a string, or `[canonical, *aliases]` for the two questions with an
accepted short form. The API, the client CLI and `qa_metrics` all score that
shape.

**How each answer was verified.** `verify_demo_qa` in `run_procedural.py` runs
before every run, including `--dry-run`, and in the test suite. A file that fails
is refused (exit 1). Evidence is limited to the two kinds of fact the
Navigator's tools can show: relation triples (`neighbors`, `find_path`) and
entity descriptions (`search_entities`, `neighbors`). No tool prints a
relationship's own description, and the seeded demo graph has no source
passages. The checks, against the fixture:

1. Every relation triple exists verbatim in the fixture, with the same source,
   type, target and direction. Every description snippet is a case-insensitive
   substring of that entity's fixture description.
2. `hops` equals the number of evidence facts, and `type` agrees with it: a
   `single` question rests on one fact, a `bridge` on two or more facts chained
   through shared entities, and a `comparison` on the two compared descriptions.
3. The answer is grounded. For a comparison it is **recomputed** from the years
   in the two fixture descriptions. Otherwise it must be an entity named in the
   evidence or a substring of a description snippet, and it must not appear in
   its own question.
4. When the answer is a relation endpoint, the final hop is unambiguous: no
   other entity stands in the same relation to the same other end.
5. Across the set: ids and questions are unique, the splits are 12/9/9, every
   answer form is at most four words, aliases do not repeat, and at least 60%
   of the questions need two facts.

The set is **self-authored**. The verifier proves that each answer can be derived
from facts the tools show. It says nothing about difficulty or bias. HotpotQA
is the external check.

The file also feeds the client's evolution CLI unchanged (from the repository
root):

```bash
synapse-graphrag evolve graphrag-navigator \
    --train backend/benchmarks/procedural/demo_qa.json --train-split train \
    --val   backend/benchmarks/procedural/demo_qa.json --val-split val \
    --rounds 1 --batch-size 6 --max-llm-calls 200
```

The cap is sized to run one round. A baseline alone decides nothing, so the CLI
and the API refuse, before any call, a cap below the worst case of the baseline
plus one whole round: (|val| + batch + |val|) × S + 1, with S = `AGENT_MAX_STEPS`
(8 by default, × 2 with generative guidance). Here: (9 + 6 + 9) × 8 + 1 = 193.
With the CLI's default batch of 10 it would be (9 + 10 + 9) × 8 + 1 = 225.

The command above evolves the **stored** graph: each accepted round becomes a new
version, and `synapse-graphrag procedures rollback NAME VERSION` undoes it. The
harness's own `--evolve` never touches that graph (see below).

### `--dataset hotpotqa --n N --reuse-graph`

The seeded HotpotQA sample comes from `../public/hotpotqa.py` (dev/distractor,
CC BY-SA 4.0; nothing from it enters the repository). It is split 40/30/30 by
position; the sample is already a seeded shuffle, so this is a random split
that `(--seed, --n)` names.

**This harness never ingests.** The corpus must already be in Neo4j, built by
`python -m benchmarks.public.run_hotpotqa --questions M --seed S` with `M ≥ N`
and the same seed. The sample is a seeded shuffle followed by a head, so the
first N questions of a bigger sample are the N-sample. `--reuse-graph` is
therefore required. Without it, or when any paragraph of the sample is missing
from the graph, the run is refused with exit 4.

`--export-splits DIR` writes `train.json`, `val.json` and `test.json` as
`[{question, answer}]` and exits without spending anything. That is the input
the CLI's `evolve --train/--val` takes.

## Evolution (`--evolve ROUNDS`)

Algorithm 1 of the paper (`app/services/procedural_evolution.py`) runs on the
**train** split, gated on the **val** split, in static mode starting from the
bundled prior. That is the paper's mode 3, which on HotpotQA ended below the prior
it started from (76.34 vs 76.61 answer F1, Table 9), so the `pg_evolved_*` vs
`pg_*_local` rows are a real test, not a formality:

- `--evolve-batch-size` sets the training questions per round. It defaults to
  `EVOLUTION_DEFAULT_BATCH_SIZE`, capped at the train split.
- `--evolve-guidance raw|generative` sets the guidance used during rollouts.
  The default, `raw`, costs half the calls.
- `--metric f1|em` sets the gate's metric. The report shows both either way.

Accepted versions are saved as **`graphrag-navigator-evolved-bench`**, so the
default graph is never touched. Those versions, the rejections and the rollout
trajectories under that name are the harness's **only** database writes. To
delete them: `DELETE /api/procedures/graphrag-navigator-evolved-bench`.

The evolved rows are scored on the same test split as the prior. When no round
is accepted, the evolved rows run the prior under the new name, and the report
says so. The gate compares two means over 9 validation questions, so its
resolution is one question (11.1 points): an accept or reject is **a step in a
search, not a significance test**. The paper says the same about its own gate.

## Cost

Every system spends real LLM calls, and generative guidance doubles them.

- **`--dry-run`** prints an **upper bound** and exits. It calls no model and
  no embedder, and it does not touch Neo4j. The bound assumes that every episode
  uses all `--max-steps` steps, that every observation hits
  `AGENT_OBSERVATION_MAX_CHARS`, and that every guided step after the first
  carries the full graph (the fallback when a step matches no node). Prompt
  sizes come from rendering the real solver, guidance and refiner templates.
  Only the completions are assumed: 120 solver, 350 guidance and 1,500 refiner
  tokens.
- **`--max-usd`** (default 2.00) works twice. A run whose upper bound is over
  the cap refuses to start. During the run, every model call is metered and
  checked **before** it is made, so the overshoot is at most the one call that
  crossed the cap. Prices come from `../public/cost.py`, a hand-recorded table,
  so every USD figure is an estimate.
- **A real run is priced as the model it calls.** `--model` re-prices a
  `--dry-run`. In a real run it is used only when the configured model id has no
  price on file (a deployment alias, a local model), and the provenance says
  "priced as". A cheaper `--model` can never weaken the cap of a priced model.
- **`--max-llm-calls`** is a second cap, counted in calls. It defaults to the
  dry-run's upper bound, and it is the only cap that still works for a model with
  no price on file. With `--evolve`, a lower cap must still cover the worst case
  of the systems that run before evolution, evolution's baseline plus one whole
  round, and the evolved rows. A cap below that sum is refused with exit 5
  before any call (and flagged by `--dry-run`), instead of being refused by
  evolution after the first systems were paid for.
- **Evolution cannot starve the evolved rows.** It gets the remaining call
  budget minus the worst case of the evolved systems that still have to run.
  Inside that budget, a round starts only when its whole worst case (batch,
  refiner call and validation) fits. When a cap refuses the refiner call, the
  evolution report says `stopped: budget` and the run exits 5.
- Embeddings are not included. The tools embed each search query, and semantic
  localization embeds unmatched steps. With `EMBEDDING_PROVIDER=fastembed`
  they run locally and cost nothing.

A run stopped by a cap prints what finished as **INCOMPLETE** and writes no
report.

## All flags

| Flag | Meaning |
| --- | --- |
| `--dataset demo\|hotpotqa` | the question set (default `demo`) |
| `--n N` | HotpotQA only: sampled questions (default 20) |
| `--reuse-graph` | HotpotQA only, **required**: score the corpus already ingested |
| `--data PATH` / `--no-download` | HotpotQA only: dataset file / never reach the network |
| `--qa PATH` | demo only: another QA file (verified before use) |
| `--allow-mixed-graph` | demo only: run even when Neo4j holds more than the fixture (flagged MIXED) |
| `--seed S` | HotpotQA sampling seed (recorded for the demo set) |
| `--systems a,b,…` | subset of the six systems |
| `--max-steps N` | Navigator turns per question, 1 to 20 (default `AGENT_MAX_STEPS`) |
| `--evolve ROUNDS` | run Algorithm 1 first (0 to 20 rounds) and add the evolved rows |
| `--evolve-batch-size N`, `--evolve-guidance raw\|generative`, `--metric f1\|em` | evolution settings |
| `--dry-run` | print the upper-bound estimate and exit |
| `--max-usd X`, `--max-llm-calls N` | the two hard caps |
| `--model NAME` | price the `--dry-run` as this model; a real run uses it only when the configured model has no price |
| `--export-splits DIR` | write the splits as `[{question, answer}]` and exit |
| `--out PATH`, `--no-write` | where the report goes / write nothing |

Exit codes:

| Code | Meaning |
| ---: | --- |
| 0 | success |
| 1 | dataset or arguments unusable (including a QA file that fails verification) |
| 2 | Neo4j unreachable |
| 3 | the run produced a result the harness cannot honestly report, e.g. every run of a system ended in a provider error |
| 4 | the graph is not prepared: demo not seeded or mixed with other documents, HotpotQA corpus not ingested, or HotpotQA without `--reuse-graph` |
| 5 | a cost cap was hit, before the run or during it |
| 6 | no chat model is configured |

## Reading the report

`results_procedural.md` is generated and never edited by hand. It contains:

- a **provenance** block: date, commit (marked when the tree is dirty), dataset
  and splits, seed, graph size, model and provider, Navigator and guidance
  settings, the prior, evolution settings, the price-table date, and the command
  that reproduces the run;
- the results table;
- every applicable pair compared under the effect floor;
- localization counts;
- evolution rounds;
- per-question F1;
- metered cost per system;
- the threats to validity.

## Honest limitations

- **Not a reproduction.** The paper evaluates its own HotpotQA agent with Gemini
  models on 1,000 test questions. This harness runs Synapse's Navigator over a
  knowledge graph with whatever model is configured. Its numbers compare systems
  with each other (F1 and tokens per question, under the same model), not with the
  paper's Table 9.
- **Small N.** Nine demo test questions put the floor at 11.1 points. Most gaps
  will, correctly, be refused.
- **No source passages on the demo graph.** Seeding writes entities and
  relations only, so on a clean demo graph (the only kind the harness accepts
  without `--allow-mixed-graph`) `read_sources` and `search_passages` find
  nothing. Prior edges that recommend those tools cost steps on the demo but not
  on HotpotQA.
- **Localization by tool name.** Several procedure steps behind one tool cannot
  be told apart.
- **Temperature 0 is not determinism.** Re-run before reading a one-question gap.
- **One model.** The solver, the guidance model and the refiner are the same
  configured model.
