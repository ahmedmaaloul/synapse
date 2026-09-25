// Shared domain types for the Synapse frontend.

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  // Populated at runtime by react-force-graph's simulation.
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  type: string;
  properties: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export type ChatRole = "user" | "assistant" | "system";

/**
 * A Louvain community — a cluster of densely connected entities, summarized by
 * the LLM. These are the corpus-level "themes" that power global search.
 */
export interface Community {
  id: string;
  title: string;
  summary: string;
  size: number;
  members: string[];
  /** Only present on vector-search results. */
  score?: number;
}

export interface CommunityList {
  communities: Community[];
  count: number;
}

/**
 * Which retrieval strategy the router picked. "local" walks entity
 * neighborhoods; "global" answers from community summaries.
 */
export type RetrievalMode = "local" | "global";

/** Citations point at an entity (local search) or a community (global search). */
export type CitationKind = "entity" | "community";

export interface Citation {
  name: string;
  type: string | null;
  kind?: CitationKind;
  /** Community citations only. */
  id?: string;
  size?: number;
  members?: string[];
}

/** One chain of relationships connecting two of the answer's seed entities. */
export interface ReasoningPath {
  /** Entity names, in order: nodes[i] --rels[i]--> nodes[i + 1]. */
  nodes: string[];
  /** Relationship types; always `nodes.length - 1` long. */
  rels: string[];
  /**
   * Per-hop direction: `true` when the edge points from nodes[i] to
   * nodes[i + 1], `false` when the chain traverses it backwards. The graph is
   * walked undirected, so a hop can genuinely run against its edge — rendering
   * every hop forward would misstate the graph. Optional for resilience
   * against older backends; hops default to forward when absent.
   */
  dirs?: boolean[];
  /** Pre-rendered "A -[REL]-> B" form, used as a stable key. */
  text: string;
}

/**
 * A source excerpt — a passage of the *original document* that was retrieved
 * alongside the graph and fed to the model. Entity descriptions are lossy
 * summaries; these are the actual prose, so they are what a reader can check
 * the answer against.
 */
export interface Source {
  /** Stable chunk id, used as a render key. */
  id: string;
  /** Filename the passage came from. */
  document: string;
  /** 0-based position of the chunk within that document. */
  index: number;
  /** The excerpt itself, already truncated server-side for transport. */
  text: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  citations?: Citation[];
  paths?: ReasoningPath[];
  sources?: Source[];
  /** Absent means "chat" — every message predating Navigator mode. */
  mode?: ChatMode;
  /** Navigator answers only: the full step-by-step run. */
  agent?: AgentResult;
  /** Navigator answers only: a failed run, already phrased for a person. */
  agentError?: { message: string; detail?: string };
}

// Server-Sent Events emitted by the chat endpoint.
export type ChatEvent =
  | { type: "citations"; data: Citation[] }
  | { type: "paths"; data: ReasoningPath[] }
  // Emitted before the first token, and only when source excerpts were
  // actually retrieved — an answer grounded in the graph alone has none.
  | { type: "sources"; data: Source[] }
  | { type: "token"; data: string }
  // Closes the stream. `usage` is what this answer cost to prompt and to
  // write (`context_tokens_est` is a chars/4 heuristic, not a tokenizer
  // count); optional for resilience against older backends.
  | {
      type: "done";
      usage?: {
        context_chars: number;
        context_tokens_est: number;
        answer_chars: number;
      };
    }
  | { type: "error"; data: string };

// Server-Sent Events emitted by the ingestion endpoint.
export type IngestStage =
  | "extracting"
  // Entity resolution runs between extraction and embedding whenever
  // `entity_resolution_enabled` is on (the default).
  | "resolving_entities"
  | "embedding"
  | "writing_nodes"
  | "writing_edges"
  // Source chunks are persisted last, whenever `store_source_chunks` is on
  // (the default). Missing this stage makes computePct fall through to its
  // default and the progress bar visibly rewinds.
  | "storing_chunks";

export interface IngestResult {
  filename: string;
  chunks_processed: number;
  nodes_created: number;
  relationships_created: number;
  entities_extracted: number;
  unique_entities: number;
  entities_merged?: number;
  // Present when community detection ran as part of the ingest job.
  communities?: number;
  communities_summarized?: number;
  modularity?: number;
}

export type IngestEvent =
  | {
      type: "progress";
      stage: IngestStage;
      processed?: number;
      total?: number;
      entities_so_far?: number;
      /** "resolving_entities" only — duplicates collapsed so far. */
      merged?: number;
    }
  // Clustering runs after extraction and reports under its own type so it
  // cannot rewind the extraction progress bar.
  | {
      type: "community_progress";
      stage: CommunityStage;
      processed?: number;
      total?: number;
      title?: string;
    }
  | { type: "done"; data: IngestResult }
  | { type: "error"; data: string };

export interface UploadJob {
  job_id: string;
  filename: string;
  total_chunks: number;
  status: string;
}

// Server-Sent Events emitted by the community-rebuild endpoint.
export type CommunityStage =
  | "loading_graph"
  | "summarizing"
  | "embedding_communities"
  | "writing_communities";

export interface CommunityRebuildResult {
  communities: number;
  summarized: number;
  modularity: number;
}

export type CommunityEvent =
  | {
      type: "progress";
      stage: CommunityStage;
      processed?: number;
      total?: number;
      title?: string;
    }
  | { type: "done"; data: CommunityRebuildResult }
  | { type: "error"; data: string };

export interface CommunityJob {
  job_id: string;
  status: string;
}

export type Theme =
  | "Personal CV / Resume"
  | "Technology, Tools & Docs"
  | "Generic"
  | "Medical/Scientific"
  | "Business/Legal"
  | "AI Safety";

// ── Procedural memory ─────────────────────────────────
// A procedural graph (Lu et al., "Procedural Graphs: Self-Evolving Execution
// Structures for LLM Agents", arXiv:2609.09153) is the *how* next to the
// knowledge graph's *what*: a small directed graph of tool actions, reasoning
// steps and status markers whose transitions carry a condition, guidance and
// pitfalls. Shapes below mirror the backend's /api/procedures contract.

/** A tool call, a reasoning step, or a status marker (Start / End). */
export type ProcNodeType = "ACTION" | "REASONING" | "STATUS";

/** The paper's fixed transition vocabulary. */
export type ProcRelation =
  | "LEADS_TO"
  | "TRIGGERS"
  | "PROVIDES_INPUT_FOR"
  | "CONVERGES_TO";

/** One row of `GET /api/procedures`. */
export interface ProceduralGraphSummary {
  name: string;
  version: number;
  /** Validation score of the live version; null for a hand-written prior. */
  score: number | null;
  nodes: number;
  edges: number;
  updated_at: string | null;
  description: string;
}

export interface ProcedureList {
  graphs: ProceduralGraphSummary[];
}

export interface ProcNode {
  id: string;
  label: string;
  // Typed loosely so an unknown type from a newer backend still renders.
  type: ProcNodeType | string;
  description: string;
  is_start: boolean;
  /** Out-degree 0 — where a run is allowed to end. */
  is_terminal: boolean;
  // Populated at runtime by react-force-graph's simulation.
  x?: number;
  y?: number;
}

export interface ProcLink {
  source: string | ProcNode;
  target: string | ProcNode;
  relation: ProcRelation | string;
  /** Natural-language precondition; null means the transition is unconditional. */
  condition: string | null;
  guidance: string;
  pitfalls: string;
}

/** `GET /api/procedures/{name}/graph-data` — react-force-graph's shape. */
export interface ProcGraphData {
  nodes: ProcNode[];
  links: ProcLink[];
  version: number;
  score: number | null;
}

/** Structural change a version introduced (`graph_diff`); keys may be absent. */
export interface ProcGraphDiff {
  added_nodes?: unknown[];
  removed_nodes?: unknown[];
  changed_nodes?: unknown[];
  added_edges?: unknown[];
  removed_edges?: unknown[];
  changed_edges?: unknown[];
}

export interface ProcVersion {
  version: number;
  score: number | null;
  accepted: boolean;
  created_at: string | null;
  note: string;
  diff: ProcGraphDiff;
}

export interface ProcVersionList {
  versions: ProcVersion[];
}

export interface RollbackResult {
  name: string;
  version: number;
}

/**
 * How the procedural graph steers the navigator. "raw" hands the solver the
 * serialized 2-hop subgraph (no extra LLM call); "generative" asks an LLM to
 * turn it into prose guidance first (the paper's configuration).
 */
export type GuidanceMode = "none" | "raw" | "generative";

/** How a step was placed on the graph — the guidance scope follows from it. */
export type LocalizationMethod = "start" | "exact" | "normalized" | "semantic" | "none";

export interface Localization {
  method: LocalizationMethod | string;
  node_id?: string | null;
  /** Cosine similarity, `semantic` matches only. */
  score?: number | null;
}

export interface AgentStep {
  thought: string;
  /** Tool name; null/empty when the model's action could not be parsed. */
  action: string | null;
  args: Record<string, unknown> | null;
  observation: string;
  guidance_context_chars: number;
  /** Null when the run used no procedural graph (or its guidance failed). */
  localization: Localization | LocalizationMethod | string | null;
  /** The node this step was localized on; null → full-graph scope. */
  active_node?: string | null;
  /** Unparseable turns only: what the model actually wrote (clipped). */
  raw_action?: string;
  /** Guidance lookup failed for this step; the run continued without it. */
  guidance_error?: string;
}

export interface AgentUsage {
  llm_calls: number;
  guidance_llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  /** True when a provider reported no usage and tokens were estimated (chars/4). */
  estimated: boolean;
  context_chars: number;
}

export type AgentStopReason = "answer" | "max_steps" | "error";

/** `POST /api/agent/ask` — one GraphRAG Navigator run, step by step. */
export interface AgentResult {
  question?: string;
  answer: string | null;
  steps: AgentStep[];
  stopped: AgentStopReason | string;
  parse_failures: number;
  usage: AgentUsage;
  graph: { name: string; version: number } | null;
  latency_s: number;
  /** Set when `stopped === "error"`. */
  error?: string | null;
  /** Whether `record: true` actually stored the trajectory. */
  recorded?: boolean;
}

export interface AgentAskOptions {
  /** Omitted → the backend's default graph; null → run without a procedural graph. */
  graph?: string | null;
  guidance?: GuidanceMode;
  /** 1..20; omitted → the backend's `agent_max_steps`. */
  maxSteps?: number;
  /** Store the trajectory (score null) for later evolution. */
  record?: boolean;
}

/** The chat panel either streams a GraphRAG answer or runs the navigator agent. */
export type ChatMode = "chat" | "navigator";

// ── Synapse Lab ───────────────────────────────────────
// Compare retrieval approaches ("arms") side by side on your own questions,
// FinOps-first: every arm is scored on quality AND on what it costs, next to
// three evidence floors. Shapes below mirror the backend's /api/lab contract
// (app/lab/{arms,estimate,metrics,runner}.py); every field a newer or older
// router might omit is optional.

/** Evidence floors (null controls) · passage baselines · graph arms. */
export type LabFamily = "null" | "passage" | "graph";

/** One row of `GET /api/lab/arms`. */
export interface LabArm {
  name: string;
  // Typed loosely so an unknown family from a newer backend still renders.
  family: LabFamily | string;
  family_title?: string;
  title: string;
  description: string;
  source: { citation: string; url: string };
  /** LLM calls the arm makes while *retrieving* (0 for every arm today). */
  retrieval_llm_calls: number;
  /** True when the arm reads the LLM-extracted knowledge graph. */
  needs_graph: boolean;
  /** A null control (N0 closed-book, N1 vocabulary, N2 random context). */
  is_null: boolean;
  k_role?: string;
}

/**
 * "retrieve" is the free tier: retrieval only, no reader, no LLM, $0.
 * "realtime" reads with the official client, metered against the cap;
 * "batch" hands the reads to the OpenAI Batch API (half price, up to 24h).
 */
export type LabMode = "retrieve" | "realtime" | "batch";

/** A token budget; `null` is the arm's own default (uncapped) context. */
export type LabBudget = number | null;

/** `GET /api/lab/arms` — the catalog plus what a run can be configured with. */
export interface LabArmCatalog {
  arms: LabArm[];
  /** False when this build has no Batch layer (batch runs are a 501). */
  batchAvailable: boolean;
}

/** A reader model with its hand-recorded price (`GET /api/lab/models`). */
export interface LabReaderModel {
  id: string;
  /** USD per 1M tokens, standard (realtime) rate. */
  input: number;
  output: number;
  /** Hidden reasoning tokens are billed as output (gpt-5*, o-series). */
  reasoning: boolean;
  /** When this price was recorded by hand. */
  checkedOn: string;
}

/** One selectable dataset, normalized from `GET /api/lab/datasets`. */
export interface LabDatasetInfo {
  /** The router's dataset id (`demo`, `hotpotqa`, `qa-file:<name>`); the UI key. */
  key: string;
  /** What the API takes as `dataset`: demo | qa-file | hotpotqa. */
  dataset: string;
  /** Uploaded QA files only: the stored file name, sent as `file`. */
  file: string | null;
  label: string;
  description: string | null;
  splits: string[];
  /** Questions per split, when the backend counted them. */
  splitCounts: Record<string, number>;
  defaultSplit: string | null;
  /** Questions available, when the backend says. */
  n: number | null;
  /** The sample size used when n is left empty (HotpotQA). */
  defaultN: number | null;
  available: boolean;
  /** Why it is unavailable (e.g. the HotpotQA corpus is not ingested). */
  reason: string | null;
}

/** Body of `POST /api/lab/estimate` and (with `seed`) `POST /api/lab/runs`. */
export interface LabRunRequest {
  dataset: string;
  split?: string | null;
  n?: number | null;
  /** qa-file only: the uploaded file's name (see `POST /api/lab/qa-files`). */
  file?: string | null;
  arms: string[];
  budgets: LabBudget[];
  k?: number;
  reader_model: string;
  mode: LabMode;
  max_usd: number | null;
  seed?: number;
  /** Price on the MEASURED contexts of an earlier retrieve-only run of this config. */
  measured_run_id?: string | null;
}

/** One phase of the dry-run estimate (ingest, retrieval_llm, reader). */
export interface LabPhaseEstimate {
  phase: string;
  model: string;
  batch: boolean;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens?: number;
  upper_prompt_tokens: number;
  upper_completion_tokens: number;
  point_usd: number | null;
  upper_usd: number | null;
  price_checked_on?: string | null;
  notes?: string[];
}

/** One (arm × budget) cell of the estimate. */
export interface LabCellEstimate {
  arm: string;
  budget: LabBudget;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  upper_prompt_tokens: number;
  upper_completion_tokens: number;
  point_usd: number | null;
  upper_usd: number | null;
  /** Mean context tokens assumed per question (measured when `measured`). */
  context_tokens: number;
  measured?: boolean;
  note?: string;
}

/** `POST /api/lab/estimate` — free: no model, no Neo4j, no network. */
export interface LabEstimate {
  mode: LabMode | string;
  reader_model: string;
  batch: boolean;
  n_questions: number;
  phases: LabPhaseEstimate[];
  cells: LabCellEstimate[];
  total_point_usd: number | null;
  total_upper_usd: number | null;
  max_usd: number | null;
  /** The upper bound exceeds the cap (or cannot be computed): no run. */
  refuse: boolean;
  refuse_reason: string | null;
  tokenizer?: string;
  reasoning_allowance?: number;
  max_output_tokens?: number;
  price_dates?: Record<string, string>;
  assumptions?: string[];
  pricing_url?: string;
  batch_multiplier?: number;
  /** Set when priced on the MEASURED contexts of an earlier retrieve-only run. */
  measured_from?: {
    run_id: string;
    cells: { arm: string; budget: LabBudget }[];
    unmeasured: { arm: string; budget: LabBudget }[];
  } | null;
}

/** A paired-bootstrap difference, in F1 points. */
export interface LabComparison {
  diff: number;
  ci_low: number;
  ci_high: number;
  n: number;
  /** Effect floor: one question, 100/n points. */
  floor: number;
  /** Exceeds the floor AND the 95% CI excludes 0. */
  reportable: boolean;
  against: string;
  verdict?: string;
}

/** One (arm, budget) row of a run's leaderboard. */
export interface LabLeaderboardRow {
  arm: string;
  budget: LabBudget;
  budget_label?: string;
  family?: string | null;
  title?: string;
  is_null: boolean;
  n: number;
  // Retrieval-level — always present.
  context_tokens_mean?: number | null;
  context_tokens_max?: number | null;
  truncated_rate?: number | null;
  units_by_kind_mean?: Record<string, number>;
  units_mean?: number | null;
  /** % of contexts containing a gold answer string (QA files, demo). */
  containment?: number | null;
  recall_permissive?: number | null;
  recall_strict?: number | null;
  both_gold_permissive?: number | null;
  both_gold_strict?: number | null;
  retrieval_errors?: number;
  // Read runs only (points 0–100, tokens, $).
  answered?: number;
  read_errors?: number;
  em?: number | null;
  f1?: number | null;
  correct?: number;
  total_tokens?: number;
  tokens_per_query?: number | null;
  tokens_per_correct?: number | null;
  tokens_per_f1?: number | null;
  usd?: number | null;
  usd_per_query?: number | null;
  usd_per_100_correct?: number | null;
  amortized_cost_of_pass?: Record<string, number | null>;
  comparisons?: {
    gain_above_n0?: LabComparison;
    gain_above_n2?: LabComparison;
    graph_premium?: LabComparison;
  };
  /** Rank by $ per 100 correct; null when unpriced or nothing correct. */
  rank?: number | null;
}

export interface LabFrontierPoint {
  arm: string;
  budget: LabBudget;
  f1: number | null;
  usd_per_query: number | null;
  tokens_per_query: number | null;
  is_null: boolean;
}

export interface LabFrontiers {
  f1_vs_usd: LabFrontierPoint[];
  f1_vs_tokens: LabFrontierPoint[];
}

/** `leaderboard.json` — `metrics.leaderboard` plus the run's labels. */
export interface LabLeaderboard {
  /** "read" (answers scored) or "retrieve" (retrieval-level columns only). */
  mode: "read" | "retrieve" | string;
  n_questions: number;
  effect_floor_points: number;
  bootstrap?: { iterations: number; seed: number; confidence: number };
  rows: LabLeaderboardRow[];
  floors?: { n0_f1: number | null; n2_f1_by_budget: Record<string, number | null> };
  frontiers?: LabFrontiers;
  run_id?: string;
  dataset?: { name: string; split: string | null; n: number };
  reader_model?: string | null;
  ingest_usd?: number | null;
  notes?: string[];
}

export type LabRunStatus =
  | "created"
  | "retrieving"
  | "retrieved"
  | "reading"
  | "batch_submitted"
  | "refused"
  | "aborted"
  | "done"
  | "failed";

/** One row of `GET /api/lab/runs`. */
export interface LabRunSummary {
  run_id: string;
  status: LabRunStatus | string;
  created_at: string | null;
  updated_at: string | null;
  mode: LabMode | string | null;
  dataset: string | null;
  n: number | null;
  arms: string[];
  budgets: LabBudget[];
  reader_model: string | null;
  /** A job is working on the run in this backend right now. */
  active?: boolean;
}

export interface LabPhaseState {
  status?: string;
  reason?: string;
  total?: number;
  requests?: number;
  pending?: number;
  failed?: number;
  skipped?: number;
  started_at?: string;
  finished_at?: string;
}

/** A run's manifest — only the fields the UI reads are typed. */
export interface LabManifest {
  run_id?: string;
  status?: LabRunStatus | string;
  created_at?: string;
  updated_at?: string;
  finished_at?: string;
  /** The run's config; `dataset` is the runner's DatasetSpec (or a bare name). */
  config?: Omit<Partial<LabRunRequest>, "dataset"> & {
    dataset?:
      | { name?: string; split?: string | null; n?: number | null; path?: string | null }
      | string;
  };
  code?: {
    git_sha?: string | null;
    git_dirty?: boolean | null;
    git_source?: string | null;
    /** Why the commit is unknown (e.g. no git in the Docker image). */
    git_unavailable?: string | null;
    synapse_version?: string;
  };
  dataset?: { name?: string; split?: string | null; n?: number; sha256?: string; source?: string };
  tokenizer?: string;
  budget_cap_usd?: number | null;
  phases?: Record<string, LabPhaseState>;
  estimate?: { pre_run?: LabEstimate; read?: LabEstimate; read_pending?: LabPendingReads };
  actual?: { reader?: { calls?: number; usd?: number } };
  refuse_reason?: string;
  abort_reason?: string;
  error?: string;
}

/** `manifest.estimate.read_pending`: what resuming would still read. */
export interface LabPendingReads {
  requests?: number;
  upper_usd?: number | null;
  spent_usd?: number;
}

/** One scored (arm, budget, question) row — the drill-down under a leaderboard row. */
export interface LabQuestionRow {
  arm: string;
  budget: LabBudget;
  qid: string;
  question: string;
  gold: string[];
  context_tokens?: number | null;
  containment?: boolean | null;
  recall_permissive?: number;
  recall_strict?: number;
  retrieval_error?: string | null;
  answer?: string | null;
  em?: number;
  f1?: number;
  usd?: number | null;
  read_error?: string;
}

export interface LabRowsPage {
  rows: LabQuestionRow[];
  total: number;
  offset: number;
}

/** `GET /api/lab/runs/{id}`. */
export interface LabRunDetail {
  run_id: string;
  /** A job for this run is in flight in the backend process. */
  active: boolean;
  manifest: LabManifest;
  /** Null until the run has been scored. */
  leaderboard: LabLeaderboard | null;
  frontiers: LabFrontiers | null;
  rows: LabRowsPage | null;
}

/** `POST /api/lab/runs` and `POST /api/lab/runs/{id}/resume`. */
export interface LabRunStarted {
  run_id: string;
  /** The SSE job to follow, when the router runs it in the background. */
  job_id: string | null;
  status: string | null;
  estimate: LabEstimate | null;
}

// Server-Sent Events of a running Lab job (the runner's own events).
export type LabEvent =
  | { type: "phase"; phase: string; status: string; total?: number }
  | {
      type: "progress";
      phase: string;
      done: number;
      total: number;
      arm?: string;
      spent_usd?: number;
    }
  // The measured re-estimate broke the cap before any read: nothing spent.
  | { type: "refused"; reason: string; estimate?: LabEstimate }
  | { type: "batch_submitted"; requests: number; batch?: unknown }
  | { type: "batch_status"; status: unknown }
  // The metered cap stopped the reads; followed by `done`.
  | { type: "aborted"; reason: string }
  // The job's ending: `data` summarizes the run's final state.
  | { type: "done"; status?: string; data?: LabJobSummary }
  | { type: "error"; error?: string; data?: string };

/** The `done` event's summary of a finished Lab job. */
export interface LabJobSummary {
  run_id?: string;
  /** done | batch_submitted | refused | aborted */
  status?: string;
  reason?: string | null;
  spent_usd?: number | null;
  cap_usd?: number | null;
  estimate_upper_usd?: number | null;
}
