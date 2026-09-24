import type { IngestStage, LocalizationMethod, Theme } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Entity-type → color. Mirrors the extraction schema in the backend.
export const TYPE_COLORS: Record<string, string> = {
  PERSON: "#818cf8", // Indigo 400
  ORGANIZATION: "#6366f1", // Indigo 500
  COMPANY: "#6366f1", // Indigo 500
  UNIVERSITY: "#0ea5e9", // Sky 500
  EDUCATION: "#3b82f6", // Blue 500
  ROLE: "#10b981", // Emerald 500
  PROJECT: "#f59e0b", // Amber 500
  SKILL: "#22d3ee", // Cyan 400
  TOOL: "#06b6d4", // Cyan 500
  FRAMEWORK: "#14b8a6", // Teal 500
  DATABASE: "#0d9488", // Teal 600
  LANGUAGE: "#a855f7", // Purple 500
  CERTIFICATION: "#f43f5e", // Rose 500
  LOCATION: "#94a3b8", // Slate 400
  // Medical / scientific
  DISEASE: "#ef4444",
  SYMPTOM: "#f97316",
  DRUG: "#8b5cf6",
  TREATMENT: "#10b981",
  GENE: "#ec4899",
  // Business / legal
  CONTRACT: "#eab308",
  LAW: "#f43f5e",
  PRODUCT: "#f59e0b",
  FINANCIAL_METRIC: "#22c55e",
  // AI safety / evals
  MODEL: "#6366f1",
  CAPABILITY: "#22d3ee",
  RISK: "#ef4444",
  FAILURE_MODE: "#f97316",
  MITIGATION: "#10b981",
  EVALUATION: "#0ea5e9",
  BENCHMARK: "#3b82f6",
  INCIDENT: "#dc2626",
  POLICY: "#eab308",
  DATASET: "#14b8a6",
  // Generic
  CONCEPT: "#a78bfa",
  EVENT: "#fb923c",
  THING: "#94a3b8",
};

export const FALLBACK_COLOR = "#71717a"; // Zinc 500

export function colorForType(type: string): string {
  return TYPE_COLORS[type?.toUpperCase()] || FALLBACK_COLOR;
}

export const THEMES: { value: Theme; label: string }[] = [
  { value: "Personal CV / Resume", label: "Personal CV / Resume" },
  { value: "Technology, Tools & Docs", label: "Technology (Wiki / Docs)" },
  { value: "Generic", label: "Generic / Other" },
  { value: "Medical/Scientific", label: "Medical / Scientific" },
  { value: "Business/Legal", label: "Business / Legal" },
  { value: "AI Safety", label: "AI Safety / Evals" },
];

// Pipeline stage → human label. Keep in sync with `IngestStage` and with
// `INGEST_STAGE_PCT` below; a missing stage would leave the bar unlabeled.
export const INGEST_STAGE_LABEL: Record<string, string> = {
  extracting: "Extracting entities",
  resolving_entities: "Resolving duplicates",
  embedding: "Embedding for retrieval",
  writing_nodes: "Writing nodes",
  writing_edges: "Linking relationships",
  storing_chunks: "Storing source text",
};

/**
 * Where each stage lands on the 0–100 bar.
 *
 * graph_builder emits, in order: extracting → embedding → resolving_entities →
 * writing_nodes → writing_edges → done, plus a *trailing* resolving_entities
 * event for the cross-document pass that runs after the edges are written.
 * The slots below are non-decreasing in that order; the trailing event is
 * absorbed by the monotonic clamp in `computePct`, which also keeps the bar
 * honest if the pipeline is ever reordered.
 * "extracting" is the fraction-driven ramp and owns everything up to its slot.
 */
export const INGEST_STAGE_PCT: Record<IngestStage, number> = {
  extracting: 70,
  embedding: 76,
  resolving_entities: 82,
  writing_nodes: 88,
  writing_edges: 96,
  // Source chunks are persisted after the edges are written; the trailing
  // cross-document resolving_entities event is absorbed by the monotonic clamp.
  storing_chunks: 98,
};

// ── Procedural memory ─────────────────────────────────

/** Seeded by the backend on startup (`procedural_default_graph`). */
export const DEFAULT_PROCEDURE = "graphrag-navigator";

/**
 * Guidance scope: the active node plus its *outgoing* transitions up to this
 * many hops (the paper's N_h with h = 2, and the backend's `procedural_hops`
 * default). Used to preview what the navigator sees from a given node.
 */
export const GUIDANCE_HOPS = 2;

// Node type → color. Tools read as the brand accent, reasoning as teal, and
// status markers stay neutral so Start / End are told apart by their rings.
export const PROC_TYPE_COLORS: Record<string, string> = {
  ACTION: "#818cf8", // Indigo 400
  REASONING: "#2dd4bf", // Teal 400
  STATUS: "#a1a1aa", // Zinc 400
};

export const PROC_START_COLOR = "#34d399"; // Emerald 400
export const PROC_TERMINAL_COLOR = "#fafafa"; // Zinc 50
/** Visited-by-the-last-navigator-run badges. */
export const PROC_TRACE_COLOR = "#fbbf24"; // Amber 400

export function colorForProcType(type: string): string {
  return PROC_TYPE_COLORS[type?.toUpperCase()] || FALLBACK_COLOR;
}

// Relation → label color; the lines themselves stay neutral.
export const PROC_RELATION_COLORS: Record<string, string> = {
  LEADS_TO: "#71717a", // Zinc 500
  TRIGGERS: "#f59e0b", // Amber 500
  PROVIDES_INPUT_FOR: "#38bdf8", // Sky 400
  CONVERGES_TO: "#a78bfa", // Violet 400
};

export function colorForRelation(relation: string): string {
  return PROC_RELATION_COLORS[relation?.toUpperCase()] || FALLBACK_COLOR;
}

/**
 * The localization cascade, in the order the backend tries it. "none" means
 * nothing matched and the guidance fell back to the whole graph.
 */
export const LOCALIZATION_INFO: Record<
  LocalizationMethod,
  { label: string; hint: string; className: string }
> = {
  start: {
    label: "start",
    hint: "Empty trajectory: placed on the Start node",
    className: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300/90",
  },
  exact: {
    label: "exact",
    hint: "The last action matched a node id exactly",
    className: "border-indigo-500/30 bg-indigo-500/10 text-indigo-300/90",
  },
  normalized: {
    label: "normalized",
    hint: "Matched once case, punctuation and arguments were stripped",
    className: "border-sky-400/30 bg-sky-400/10 text-sky-300/90",
  },
  semantic: {
    label: "semantic",
    hint: "Matched by embedding similarity to the node descriptions",
    className: "border-violet-400/30 bg-violet-400/10 text-violet-300/90",
  },
  none: {
    label: "full graph",
    hint: "No node matched: guidance used the whole graph",
    className: "border-[#3f3f46] bg-[#27272a]/60 text-[#a1a1aa]",
  },
};
