# Synapse for AI-safety insight

Safety evidence is relational. A model *exhibits* a failure mode, a mitigation
*addresses* a risk, an evaluation *measures* a capability, an incident *involved*
a deployment, a policy *governs* an organisation. Those are edges, and the
questions safety teams ask are walks over them ("which mitigations have been
evaluated against the risks this system card admits to?") or properties of the
whole graph ("what recurs across two years of incident reports?"). That is
precisely the local/global split Synapse's GraphRAG engine is built around.

This page is the shortest path from a folder of PDFs to grounded answers.

---

## 1. Ingest with the `AI Safety` theme

The theme selects the extraction schema — the entity and relationship types the
model is allowed to use — so the graph comes out in safety vocabulary instead of
generic `CONCEPT`/`RELATED_TO` blobs.

```bash
synapse-graphrag ingest system-card.pdf --theme "AI Safety"
synapse-graphrag ingest incident-report.pdf --theme "AI Safety"
```

(Or pick **AI Safety / Evals** in the upload panel of the web UI.)

| Entity types | Relationship types |
| --- | --- |
| `MODEL` `ORGANIZATION` `PERSON` `CAPABILITY` `RISK` `FAILURE_MODE` `MITIGATION` `EVALUATION` `BENCHMARK` `INCIDENT` `POLICY` `DATASET` `CONCEPT` | `DEVELOPED_BY` `EXHIBITS` `POSES` `MITIGATES` `EVALUATED_BY` `MEASURES` `INVOLVED_IN` `GOVERNS` `PROPOSED_BY` `RELATED_TO` |

Theme-specific extraction rules (see `get_extraction_prompt` in
[`backend/app/services/graph_builder.py`](../backend/app/services/graph_builder.py)):

- **Demonstrated vs. hypothesised.** A risk or failure mode's description starts
  with `demonstrated:` when the source reports an observation, and
  `hypothesised:` when it is argued or forecast. The distinction survives into
  retrieval, so an answer can say which is which.
- **Canonical model names** (`GPT-4`, `Claude 3 Opus`, `Llama 3 70B`) so versions
  merge into one node instead of five.
- **No invented incidents.** Only incidents, evaluations and benchmarks named in
  the text are extracted — the graph must never contain an event the corpus does
  not.

What to feed it (all public, all PDF): model and system cards, frontier-safety
frameworks and preparedness policies, red-team and evaluation reports, incident
write-ups and post-mortems, standards and regulation (risk-management frameworks,
AI-act style texts), and the eval papers your team argues about.

---

## 2. Ask — locally, globally, and with a budget

```bash
# local: specific entities, multi-hop
synapse-graphrag ask "Which mitigations are claimed for deceptive behaviour, and which were actually evaluated?"

# global: the corpus as a whole (routed to community summaries)
synapse-graphrag ask "What risk themes recur across these incident reports?"

# retrieval only: the evidence, capped, for your own model or a human
synapse-graphrag retrieve "sandbagging evaluations" --budget 3000 --json
```

Every answer streams its **citations** (the entities that grounded it), the
**reasoning paths** it walked (`Model -[EXHIBITS]-> Failure mode`, `Risk <-[MITIGATES]- Mitigation`)
and the **source excerpts** the sentences were written from. If the graph does
not contain the answer, the prompt tells the model to say so rather than fill in.

---

## 3. From Claude, Cursor or any MCP client

Install the MCP server once (see [`docs/mcp.md`](./mcp.md)), then:

- `synapse_retrieve` — the default: budgeted context, your model answers, no
  second LLM bill (see [`docs/finops.md`](./finops.md)).
- `synapse_communities` — the detected themes of the corpus, a good first call on
  a new batch of reports.
- The **`safety_brief`** prompt turns a topic into a structured brief — *claims,
  evidence (demonstrated vs. hypothesised), mitigations and their evaluation
  status, open questions* — with every line tied to a cited entity or excerpt, and
  an explicit "not in the corpus" section for what the graph cannot support.

---

## 4. Ground rules the tool enforces — and the ones you must

Synapse enforces: citations point at real nodes; excerpts are verbatim source
prose; retrieval never calls a model; global summaries are marked as summaries.

You must: treat community summaries as LLM-written *overviews* to be checked
against the excerpts; remember the graph is only as complete as the corpus you
ingested; keep the backend on a private network (it has no authentication of its
own — see the security note in [`docs/mcp.md`](./mcp.md)); and, when a number in
an answer matters, open the excerpt. A retrieval system that names the right
entity is not the same as one that returned the evidence — read the prose.
