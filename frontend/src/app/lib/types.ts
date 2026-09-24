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
