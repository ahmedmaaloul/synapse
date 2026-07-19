import type { Theme } from "./types";

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

export const INGEST_STAGE_LABEL: Record<string, string> = {
  extracting: "Extracting entities",
  embedding: "Embedding for retrieval",
  writing_nodes: "Writing nodes",
  writing_edges: "Linking relationships",
};
