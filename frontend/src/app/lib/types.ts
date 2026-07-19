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
}

// Server-Sent Events emitted by the chat endpoint.
export type ChatEvent =
  | { type: "citations"; data: Citation[] }
  | { type: "paths"; data: ReasoningPath[] }
  // Emitted before the first token, and only when source excerpts were
  // actually retrieved — an answer grounded in the graph alone has none.
  | { type: "sources"; data: Source[] }
  | { type: "token"; data: string }
  | { type: "done" }
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
  | "Business/Legal";
