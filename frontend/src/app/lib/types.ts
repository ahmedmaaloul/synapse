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

export interface Citation {
  name: string;
  type: string | null;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  citations?: Citation[];
}

// Server-Sent Events emitted by the chat endpoint.
export type ChatEvent =
  | { type: "citations"; data: Citation[] }
  | { type: "token"; data: string }
  | { type: "done" }
  | { type: "error"; data: string };

// Server-Sent Events emitted by the ingestion endpoint.
export type IngestStage =
  | "extracting"
  | "embedding"
  | "writing_nodes"
  | "writing_edges";

export interface IngestResult {
  filename: string;
  chunks_processed: number;
  nodes_created: number;
  relationships_created: number;
  entities_extracted: number;
  unique_entities: number;
}

export type IngestEvent =
  | {
      type: "progress";
      stage: IngestStage;
      processed?: number;
      total?: number;
      entities_so_far?: number;
    }
  | { type: "done"; data: IngestResult }
  | { type: "error"; data: string };

export interface UploadJob {
  job_id: string;
  filename: string;
  total_chunks: number;
  status: string;
}

export type Theme =
  | "Personal CV / Resume"
  | "Technology, Tools & Docs"
  | "Generic"
  | "Medical/Scientific"
  | "Business/Legal";
