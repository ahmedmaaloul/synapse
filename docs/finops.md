# FinOps: where a GraphRAG dollar goes — and how Synapse lets you cap it

Synapse's core pipeline spends money in exactly three places, and every one of
them has a knob. The opt-in procedural-memory features (the Navigator agent,
generative guidance and self-evolution) add three more, each with its own cap:
see [§5](#5-procedural-guidance-the-navigator-and-evolution).

This page covers the honest cost model behind the defaults, the settings that
bound each cost, and the 0.4.0 additions that let an MCP host (Claude, Cursor,
…) consume **budgeted context instead of paying for a second LLM call**. The
[Synapse Lab](./lab.md) measures the choice behind the per-question bill: which
retrieval approach, at which context budget, reaches a correct answer for the
fewest tokens and dollars ([§7](#7-the-lab-what-a-correct-answer-costs)).

> Every number below is a *shape*, not a price. Provider prices change; measure
> your own runs with the ledger in [`backend/benchmarks/public/cost.py`](../backend/benchmarks/public/cost.py)
> (it reads the provider's reported token usage and labels estimates as estimates).

---

## 1. The three cost centres

| When | What calls a paid model | Bounded by |
| --- | --- | --- |
| **Ingest** | One extraction call **per chunk** — `chunks × (prompt + output tokens)` | `MAX_CHUNKS` (default 40 per document), `EXTRACTION_CONCURRENCY` (5), `EXTRACTION_TIMEOUT` (180 s) |
| **Ingest → communities** | One summary call **per community** after every ingest (Louvain itself is free, runs in Python) | `COMMUNITY_DETECTION_ENABLED`, `COMMUNITY_MIN_SIZE` (3 — smaller clusters are never summarised), `COMMUNITY_MAX_MEMBERS_IN_SUMMARY` (30 — caps the prompt) |
| **Question** | One streamed generation call, whose prompt is graph facts + reasoning paths + source excerpts | `CHUNK_TOP_K` (4 excerpts), `CHUNK_CONTEXT_MAX_CHARS` (4 000), `MAX_REASONING_PATHS` (6), `RETRIEVAL_MAX_HOPS` (2) |

Everything else is free by design:

- **Embeddings** default to local `fastembed` (`bge-small-en-v1.5`, 384-dim). Cloud embeddings are a one-line swap, and the only reason to pay for them.
- **Entity resolution** is cosine similarity over those embeddings plus fuzzy name matching — no LLM.
- **Query routing** (local vs. global search) is deterministic — no LLM.
- **Retrieval** is Neo4j vector + full-text indexes — no LLM.
- **Procedural guidance in its default `raw` mode** is a serialized subgraph — no LLM
  (see [§5](#5-procedural-guidance-the-navigator-and-evolution)).

So a corpus of 10 documents × 40 chunks costs ~400 extraction calls once, plus one
summary per community after each of the 10 ingests, and then each question costs
**one** generation call whose input is capped by the settings above.

---

## 2. Cost profiles

| Profile | Settings | What you pay |
| --- | --- | --- |
| **Zero** — demo, tests, UI work | `make demo`, `EMBEDDING_PROVIDER=fastembed`, no `LLM_PROVIDER` key | Nothing. Graph view, retrieval, `POST /api/retrieve`, the procedural-graph view, `raw` procedural guidance, retrieve-only Lab runs and the whole test suite work without a model. |
| **Lean cloud** — most teams | `LLM_PROVIDER=gemini` (free tier) or `groq`, `openai` with `gpt-4o-mini`; keep `MAX_CHUNKS=40` | Cents per document; fractions of a cent per question. |
| **Private** — regulated data | `LLM_PROVIDER=ollama` (+ `EMBEDDING_PROVIDER=ollama` or `fastembed`) | Hardware only; no tokens leave the machine. |
| **Premium answers, cheap ingest** | Any provider; ingest with a small model, then switch `LLM_PROVIDER`/model for chat | Extraction is the volume cost — spend the strong model only on answers. |

`LLM_PROVIDER` is a single env var: switching provider or model needs no code change
and no re-ingest (the graph is provider-agnostic; only re-embedding requires a
matching `EMBEDDING_DIM`).

---

## 3. Context management at answer time (new in 0.4.0)

The expensive part of a question is *tokens in*, and with an MCP host you are
already paying for a capable model on the other end. Synapse therefore separates
**retrieval** from **generation**:

- **`POST /api/retrieve`** returns the same context the chat endpoint would have
  sent to its own LLM — graph facts, reasoning paths, source excerpts, citations —
  and *no* generated answer. Pass `max_context_chars` to cap it; the response's
  `usage` block reports `context_chars`, `context_tokens_est` (≈ chars / 4,
  a heuristic) and whether it was `truncated`.
- **`synapse_retrieve`** (MCP tool) is the default way an agent should use Synapse:
  the host model reads the budgeted context and answers itself. `synapse_ask`
  exists when you *want* Synapse's configured model to answer — its description
  says plainly that it costs a second LLM call.
- **`usage` on every chat `done` event** — `context_chars`, `context_tokens_est`,
  `answer_chars` — so a dashboard can attribute spend per question without
  scraping logs.
- **Server-side TTL cache** in the MCP server (`SYNAPSE_CACHE_TTL`, default 300 s):
  a repeated `(query, k, budget)` costs nothing and reports `usage.cached: true`.
- **CLI:** `synapse-graphrag retrieve "…" --budget 2000 --json` gives you the same
  thing from a shell or a pipeline.

Rule of thumb: start at `SYNAPSE_MAX_CONTEXT_CHARS=6000` (~1 500 tokens). Below
~2 000 chars the source excerpts are the first thing to go, which is exactly the
evidence a grounded answer needs — watch `usage.truncated`.

---

## 4. Measuring instead of guessing

`backend/benchmarks/public/cost.py` is a small, dependency-free ledger:

```python
from benchmarks.public.cost import CostLedger, metered

with CostLedger("gpt-4o-mini", label="ingestion") as ledger:
    chain = prompt | metered(llm, ledger)
    await chain.ainvoke(...)
print("\n".join(ledger.lines()))   # tokens, calls, elapsed, estimated USD
```

It reads the provider's reported usage when present and falls back to a
character-based estimate that stays flagged as an estimate for the life of the
ledger. Its price table is hand-recorded with a date — treat every USD figure as
an estimate to check against the invoice. An entry verified on another day
carries its own date (the gpt-5 family does), and `usd(usage, price, batch=True)`
prices a request at the OpenAI Batch rate: the same row × `BATCH_PRICE_MULTIPLIER`
(0.5, dated separately), never a second table.

---

## 5. Procedural guidance, the Navigator and evolution

[Procedural memory](./procedural-graphs.md) adds three opt-in ways to spend, next to the three
cost centres above. The Procedural Graphs paper (arXiv:2609.09153) is candid about the price of
its own design:

- **Guidance tokens (Table 3).** Localized generative guidance raises total tokens by 33.4 %
  (GDPval) and 55.4 % (ALFWorld) over the no-graph baseline.
- **Tokens per question (Table 9, HotpotQA).** Its most accurate construction mode uses 10,116
  tokens per question, against 4,003 unguided.
- **Evolution can cost more and score less (Table 9).** Evolving the hand-crafted expert graph
  online (mode 3, exactly what `evolve --mode static` runs) scored 76.34 answer F1 at 10,658
  tokens per question, below the unevolved expert graph (76.61 at 9,046). Only evolution from
  scratch (mode 5, `--mode scratch`) beat the prior among the modes Synapse implements (78.79 at
  10,116). An evolution run is paid search, not a guaranteed improvement.

Synapse's defaults are built around those numbers:

| What | Paid LLM calls | Bounded by |
| --- | --- | --- |
| **`raw` guidance** (the default: API, MCP, CLI, Navigator) | **0**. The serialized local subgraph is handed over as-is. | `PROCEDURAL_HOPS` (2) sets its size; every response reports `usage.context_chars` / `context_tokens_est` |
| **`generative` guidance** (the paper's mode, opt-in) | 1 per step; 0 when the exact situation is cached | `PROCEDURAL_GUIDANCE_MODE`, `PROCEDURAL_WINDOW` (3 recent steps in its prompt), `PROCEDURAL_GUIDANCE_CACHE_SIZE` (256) |
| **A Navigator question** (`/api/agent/ask`, `synapse_agent_ask`, the UI's *Navigator* mode) | up to `AGENT_MAX_STEPS` (8) solver calls, twice that with generative guidance. `/api/chat` makes **one**. | `AGENT_MAX_STEPS` (and `max_steps` per request, 1–20), `AGENT_OBSERVATION_MAX_CHARS` (1 500 per tool output) |
| **An evolution run** (`synapse-graphrag evolve`, `POST …/evolve`) | ≤ (rounds × (batch + \|val\|) + \|val\|) × `AGENT_MAX_STEPS` (× 2 generative) + one refiner call per round | a hard `max_llm_calls` (default `EVOLUTION_MAX_LLM_CALLS=400`); `EVOLUTION_TRAJECTORY_MAX_CHARS` (24 000) caps the refiner prompt's trace text. A cap below one round's worth, (\|val\| + batch + \|val\|) × S + 1 with S = `AGENT_MAX_STEPS` (× 2 generative), is refused before any call |

What that means in practice:

- **Sizes on the bundled priors.** Computed with the real serializer over the nodes an agent can
  actually be localized on, these are deterministic properties of the graphs, not measurements
  of a run:
  - `graphrag-navigator` (the Navigator): 1,153–3,083 characters per step (≈ 289–771 tokens at
    chars / 4), at `Start` and at the tool nodes other than `answer`, which ends the run. `End`
    (203 characters, a terminal) and `Decompose_Question` (3,336, a `REASONING` node) are
    excluded: the Navigator's actions are tool names, so it is never localized on them. The
    full-graph fallback (after a failed parse) is 8,386 characters (≈ 2,097 tokens).
  - `mcp-host` (MCP hosts): 687–7,878 characters per step (≈ 172–1,970 tokens) at its ten
    non-terminal nodes, and 11,486 characters (≈ 2,872 tokens) for the full graph. A host is
    told to report "the tool or node name", so it can be localized on any of them: the six
    `synapse_*` tool nodes cost 1,283–2,386 characters, `Start` 3,976, and `Plan_Question`
    (the hub that fans out to all six tools) 7,878 (≈ 1,970 tokens) whenever a host reports it
    as its last action. `End` (225 characters) is a terminal.
- **Raw injection is Synapse's choice, not the paper's result.** The paper always pays for a
  guidance call. Synapse's default skips it, consistent with the retrieve-only stance in §3.
  Whether raw guidance helps as much is exactly what the benchmark harness's `pg_raw_local` vs
  `pg_gen_local` pair measures. Until a run says so, it is a hypothesis.
- **The localization cascade** saves tokens by keeping more steps local instead of falling back
  to the full graph. Its `semantic` step embeds the last action and observation, plus each graph
  version's node descriptions once. With the default `fastembed` that is free; with a cloud
  embedder it is one small embedding call per unmatched step. A failed parse and a skeleton graph
  skip it (nothing to embed, or no candidate).
- **Evolution never overspends, and never wastes a round.** A round starts only when its whole
  worst case (training batch, refiner call and validation) fits in the remaining budget, so no
  paid work is thrown away half-done. Otherwise the run stops cleanly (`stopped: "budget"`),
  never in the middle of a save. A cap too small for the baseline plus one whole round would
  pay for a baseline that decides nothing, so it is refused before any call (the API answers
  422 with the minimum, the CLI names it): (|val| + batch + |val|) × S + 1, with S =
  `AGENT_MAX_STEPS` (× 2 generative). On the demo QA set (9 validation questions) with
  `--batch-size 6`, that is (9 + 6 + 9) × 8 + 1 = 193 calls. The CLI prints the upper bound,
  asks for consent before sending anything and sends that bound as the cap, and there is
  deliberately no MCP tool that starts a run.

**Measure it.** `cd backend && python -m benchmarks.procedural.run_procedural --dry-run` prints an
upper-bound bill for every system without calling a model. A real run is metered with the ledger
from §4 and stops at `--max-usd` (default 2.00) or `--max-llm-calls`, whichever comes first. See
[`backend/benchmarks/procedural/README.md`](../backend/benchmarks/procedural/README.md).

---

## 6. Reasoning models (the gpt-5 family)

OpenAI's reasoning models — the gpt-5 family and the o-series — change three things about the
bill. Synapse recognizes them by model id (`^(gpt-5|o[1-9])`) on the `openai`,
`azure_openai` (the *deployment* name, so name the deployment after the model) and
`openai_compatible` providers:

- **No temperature.** They accept only their default temperature, so Synapse sends none.
  `CHAT_TEMPERATURE`, `EXTRACTION_TEMPERATURE`, `AGENT_TEMPERATURE` and the temperature 0 of the
  guidance and refiner calls do not apply to them. The model samples at its own default, so two
  runs of the same question can differ more than on a classic model. Re-run before reading a
  one-question gap in the benchmarks.
- **A reasoning effort instead: `OPENAI_REASONING_EFFORT`** (`minimal` by default; `minimal`,
  `low`, `medium` or `high`, passed through for the API to validate). `minimal` exists only on
  the gpt-5 family, so an o-series model is sent `low` instead. Every other model ignores the
  setting.
- **Hidden reasoning tokens are billed as output.** The model thinks in tokens you never see,
  and they are priced at the output rate. That is why the default effort is the lowest: output
  tokens cost several times more than input (8× on `gpt-5-nano` and `gpt-5-mini` in the
  ledger's price table). A short visible answer can carry a large output bill, so a
  characters / 4 estimate of the visible text undercounts badly. Read the provider's reported
  usage instead: its output count already includes the reasoning, and the ledger from §4 prints
  that part separately ("of which N hidden reasoning, billed as output").

The Navigator and evolution multiply this: every step of every rollout is one such call. The
Lab's reader caps it per call ([§7](#reasoning-models-in-the-lab)).

---

## 7. The Lab: what a correct answer costs

§1 to §6 bound what Synapse spends. The [Synapse Lab](./lab.md) measures the choice that sets the
biggest recurring cost, what goes into the prompt of every question: it runs retrieval approaches
("arms") side by side on your own questions, under the same token budgets, and ranks them by what
a *correct* answer costs. Every arm retrieves with zero LLM calls, so an arm's query cost is its
reader call and nothing else.

### Leaderboard metrics

| Metric | Definition | Why it is on a FinOps table |
| --- | --- | --- |
| **$ per 100 correct** (the ranking) | reader $ ÷ EM-correct answers × 100: cost-of-pass ([Erol et al.](https://arxiv.org/abs/2504.13359)) | tokens per call rewards a context that is small and wrong; this rewards a context that is small *and* answers |
| tokens per correct | total tokens ÷ correct answers (and ÷ ΣF1) | the same, in tokens, independent of price changes |
| amortized cost-of-pass(Q) | `(C_ingest / Q + C_query) ÷ accuracy`, Q = 100, 1,000, 10,000 queries per corpus | a graph arm also carries the extraction bill; this spreads it over the questions the corpus will serve. `C_ingest` is the measured spend (`--ingest-usd`) and is charged only to arms that need the extracted graph. Unknown ingest → no figure, never a silent $0. |
| gain above N0 · gain above N2 | F1 − F1(closed-book); F1 − F1(random context at the same budget) | context tokens that do not beat no evidence, or do no better than random passages of the same size, buy tokens rather than retrieval |
| graph premium at B | F1(graph arm, B) − F1(the better of BM25 and dense at B) | what the graph adds over plain passages at the same token budget |
| Pareto frontiers | the non-dominated (arm, budget) points for F1 vs $ per query and F1 vs tokens per query | the cheapest arm for each level of quality, in both units, since dollars also weigh output and reasoning tokens |

Differences are paired-bootstrap estimates (10,000 resamples, 95% CI) and are reported only when
they clear a one-question effect floor (100 / n points) and the CI excludes 0. See
[lab.md](./lab.md#the-leaderboard) for every column, including the free retrieve-only ones.

### Spend controls

- **Retrieve-only is the default and costs $0.** No model is called. It reports context tokens,
  units by kind and whether the gold answer reached the context, next to the floors.
- **A free estimate first.** Per phase and per arm × budget: a point estimate (real-tokenizer
  prompts, assumed answer length) and an upper bound (output at the request's cap, 0% cache hits,
  +10% on every prompt). `refuse` when the upper bound exceeds `--max-usd` or when a model has no
  price on file.
- **Three checks.** Before the run starts (the API answers 422 with the estimate); after the free
  retrieve phase, re-priced on the contexts actually packed; and, in realtime, a spend guard that
  reserves each request's worst case before sending it and stops *before* the cap.
- **Price on measured contexts.** A capped context is assumed to fill its budget until measured.
  Run the configuration retrieve-only first (free), then estimate the paid run with
  `--measured-from RUN_ID`.
- **Calibration.** `backend/lab_runs/calibration.json` replaces the assumed answer and reasoning
  lengths and the ingest tokens per paragraph with measured ones; every run records its estimate
  next to its actual spend.
- **Identical requests are paid once.** Closed-book reads the same empty context at every budget,
  and an arm whose context is the same at two budgets sends one request for both.

### Batch

The OpenAI Batch API bills input and output at **half** the realtime rate, for results within 24
hours. The Lab uses it for both of its paid phases:

- **Reader** (`--mode batch`): the run retrieves, passes the measured-context check, submits and
  parks as `batch_submitted`; `lab resume RUN_ID` collects and scores. Requests that expire or fail
  come back as errors and can be resent with `--retry-failed`; nothing is sent twice.
- **Ingest** (`app/lab/ingest.py`, Python): one extraction request per document, with the same
  prompt, parser and writes as the realtime pipeline. Its estimate uses tokens per paragraph
  measured on HotpotQA with `gpt-4o-mini` (434 prompt, 464 completion) until a calibration file
  says otherwise. **Community summaries are off by default**: they would be realtime calls on the
  chat provider, neither batched nor half price, and no Lab arm needs them except `synapse_d`'s
  global route.
- **Queue limits.** OpenAI caps the tokens an organisation may have enqueued per model.
  `SYNAPSE_LAB_BATCH_MAX_ENQUEUED_TOKENS` sizes parts under that cap and submits them in waves.

### Reasoning models in the Lab

The default reader, `gpt-5-nano`, is a reasoning model (§6). The Lab bounds its hidden reasoning
per request: `max_completion_tokens` = 32 answer tokens + a **reasoning allowance** (512 by
default, `--reasoning-allowance`). The upper bound charges every call at that cap, so it holds
whatever the model does. The point estimate assumes 64 reasoning tokens per call at effort
`minimal`, which is an assumption until a calibrated run replaces it.

Lowering the allowance lowers the bound and the request's own cap together, so the bound stays
true. But a call that spends its whole allowance thinking returns no visible answer
(`finish_reason: "length"` in the run's rows) and scores 0. Check for those before tightening the
allowance further. Tokens per correct counts reasoning tokens, since they are billed as output.

---

## 8. What is *not* done yet (roadmap)

- **Per-request USD accounting in the app.** The Lab meters its own reader calls per request
  (tokens and $ in each run's `responses.jsonl`). Chat and ingest still report characters; wire
  the ledger above into them so `usage` carries measured tokens (and an estimated cost) per
  provider.
- **Cached-input pricing.** The price table has no cached-input rate, so the Lab prices cached
  prompt tokens at the full input rate, and its upper bound assumes no cache hits at all.
- **Incremental community summaries.** Community ids are content-derived and
  stable, but summaries are regenerated after every ingest; reusing the summary
  of an unchanged community would make the per-ingest cost proportional to what
  actually changed.
- **Prompt caching** for the (large, static) extraction system prompt on providers
  that support it.
- **Semantic answer cache** across users for identical or near-identical questions.
- **Model tiering in config** — a separate `EXTRACTION_MODEL` so ingest can use a
  cheaper model than chat without flipping env vars between phases.
- **Selective generative guidance** — call the guidance LLM only when it is likely to
  matter (a localization miss, a failed step) instead of on every step. The Procedural
  Graphs paper suggests this as future work, and today's exact-situation cache is only
  the first half of it.

If one of these matters to you, open an issue — the settings and the ledger are
the seams they plug into.
