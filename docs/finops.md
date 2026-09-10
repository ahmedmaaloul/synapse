# FinOps: where a GraphRAG dollar goes — and how Synapse lets you cap it

Synapse spends money in exactly three places, and every one of them has a knob.
This page is the honest cost model behind the defaults, the settings that bound
each cost, and the 0.4.0 additions that let an MCP host (Claude, Cursor, …)
consume **budgeted context instead of paying for a second LLM call**.

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

So a corpus of 10 documents × 40 chunks costs ~400 extraction calls once, plus one
summary per community after each of the 10 ingests, and then each question costs
**one** generation call whose input is capped by the settings above.

---

## 2. Cost profiles

| Profile | Settings | What you pay |
| --- | --- | --- |
| **Zero** — demo, tests, UI work | `make demo`, `EMBEDDING_PROVIDER=fastembed`, no `LLM_PROVIDER` key | Nothing. Graph view, retrieval, `POST /api/retrieve` and the whole test suite work without a model. |
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
an estimate to check against the invoice.

---

## 5. What is *not* done yet (roadmap)

- **Per-request USD accounting in the app.** Wire the ledger above into ingest and
  chat so `usage` carries measured tokens (and an estimated cost) per provider.
- **Incremental community summaries.** Community ids are content-derived and
  stable, but summaries are regenerated after every ingest; reusing the summary
  of an unchanged community would make the per-ingest cost proportional to what
  actually changed.
- **Prompt caching** for the (large, static) extraction system prompt on providers
  that support it.
- **Semantic answer cache** across users for identical or near-identical questions.
- **Model tiering in config** — a separate `EXTRACTION_MODEL` so ingest can use a
  cheaper model than chat without flipping env vars between phases.

If one of these matters to you, open an issue — the settings and the ledger are
the seams they plug into.
