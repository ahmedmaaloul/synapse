import type { IngestStage, Theme } from "./types";

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
