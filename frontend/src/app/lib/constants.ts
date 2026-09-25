import type {
  IngestStage,
  LabBudget,
  LabMode,
  LabReaderModel,
  LocalizationMethod,
  Theme,
} from "./types";

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

// ── Synapse Lab ───────────────────────────────────────

/** Token budgets on offer; `null` is each arm's own default (uncapped) context. */
export const LAB_BUDGETS: LabBudget[] = [500, 1000, 2000, 4000, 8000, null];
/** A spread wide enough to draw a curve per arm on the Pareto plot. */
export const LAB_DEFAULT_BUDGETS: LabBudget[] = [500, 2000, 4000];
/** Mirrors the runner's DEFAULT_K / DEFAULT_SEED / DEFAULT_READER. */
export const LAB_DEFAULT_K = 8;
export const LAB_DEFAULT_SEED = 20260924;
export const LAB_DEFAULT_READER = "gpt-5-nano";
export const LAB_DEFAULT_MAX_USD = 0.5;
/** Questions per page in a leaderboard row's drill-down. */
export const LAB_ROWS_PAGE = 25;

/**
 * Fallback reader list for a backend without `GET /api/lab/models`. MIRRORS
 * the hand-recorded table in backend/benchmarks/public/cost.py — prices are NOT
 * fetched live, go stale, and must be re-verified at LAB_PRICING_URL; the
 * backend's estimate (which quotes its own dates) is what gates a run.
 */
export const LAB_READER_MODELS: LabReaderModel[] = [
  { id: "gpt-5-nano", input: 0.05, output: 0.4, reasoning: true, checkedOn: "2026-09-24" },
  { id: "gpt-5-mini", input: 0.25, output: 2.0, reasoning: true, checkedOn: "2026-09-24" },
  { id: "gpt-4.1-nano", input: 0.1, output: 0.4, reasoning: false, checkedOn: "2026-07-21" },
  { id: "gpt-4o-mini", input: 0.15, output: 0.6, reasoning: false, checkedOn: "2026-07-21" },
  { id: "gpt-4.1-mini", input: 0.4, output: 1.6, reasoning: false, checkedOn: "2026-07-21" },
  { id: "gpt-4.1", input: 2.0, output: 8.0, reasoning: false, checkedOn: "2026-07-21" },
  { id: "gpt-4o", input: 2.5, output: 10.0, reasoning: false, checkedOn: "2026-07-21" },
];

/** OpenAI Batch API: half the standard rate on input and output. */
export const LAB_BATCH_MULTIPLIER = 0.5;
export const LAB_PRICING_URL = "https://openai.com/api/pricing/";

export const LAB_MODE_INFO: Record<LabMode, { label: string; hint: string }> = {
  retrieve: {
    label: "Retrieve · $0",
    hint: "Retrieval only: no reader, no LLM, $0. Scores context size, units by kind and whether the gold answer is in the context.",
  },
  realtime: {
    label: "Realtime",
    hint: "One reader call per (arm, budget, question), metered against the cap: the run stops cleanly before it would exceed it.",
  },
  batch: {
    label: "Batch",
    hint: "Reads go through the OpenAI Batch API: half price, results within 24h. Check back to collect them.",
  },
};

/** Display order of the arm families. */
export const LAB_FAMILY_ORDER = ["null", "passage", "graph"];

/**
 * Family → label and mark color. Two validated categorical hues (passage vs
 * graph: CVD ΔE 32, normal-vision ΔE 35 on #09090b) plus a neutral gray for
 * the floors, which are also drawn hollow so they never read as a series.
 */
export const LAB_FAMILY_INFO: Record<string, { title: string; hint: string; color: string }> = {
  null: {
    title: "Evidence floors",
    hint: "Null controls: the score with no evidence, meaningless evidence or random evidence. A gain only counts above these.",
    color: "#71717a", // Zinc 500
  },
  passage: {
    title: "Passage baselines",
    hint: "Plain text retrieval over the source chunks: no graph needed.",
    color: "#d97706", // Amber 600
  },
  graph: {
    title: "Graph arms",
    hint: "Retrieval over the LLM-extracted knowledge graph.",
    color: "#6366f1", // Indigo 500
  },
};

export function colorForFamily(family: string | null | undefined): string {
  return LAB_FAMILY_INFO[family ?? ""]?.color ?? FALLBACK_COLOR;
}

export type LabShape = "circle" | "square" | "triangle" | "diamond";

/** Marker shape per arm: identity within a family is never color alone. */
export const LAB_ARM_SHAPES: Record<string, LabShape> = {
  null_closed_book: "circle",
  null_vocabulary: "square",
  null_random: "triangle",
  bm25: "circle",
  dense: "square",
  synapse_d: "circle",
  synapse_lean: "diamond",
  ppr: "triangle",
};

/** Run status → chip label and colors. */
export const LAB_STATUS_INFO: Record<string, { label: string; className: string }> = {
  created: {
    label: "created",
    className: "border-[#3f3f46] bg-[#27272a]/60 text-[#a1a1aa]",
  },
  retrieving: {
    label: "retrieving",
    className: "border-indigo-500/30 bg-indigo-500/10 text-indigo-300/90",
  },
  retrieved: {
    label: "retrieved",
    className: "border-indigo-500/30 bg-indigo-500/10 text-indigo-300/90",
  },
  reading: {
    label: "reading",
    className: "border-indigo-500/30 bg-indigo-500/10 text-indigo-300/90",
  },
  batch_submitted: {
    label: "batch submitted",
    className: "border-amber-400/30 bg-amber-400/10 text-amber-300/90",
  },
  refused: {
    label: "refused",
    className: "border-rose-500/30 bg-rose-500/10 text-rose-300/90",
  },
  aborted: {
    label: "aborted at cap",
    className: "border-amber-400/30 bg-amber-400/10 text-amber-300/90",
  },
  done: {
    label: "done",
    className: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300/90",
  },
  failed: {
    label: "failed",
    className: "border-rose-500/30 bg-rose-500/10 text-rose-300/90",
  },
};

/** Statuses that mean the backend is still working on the run. */
export const LAB_ACTIVE_STATUSES = new Set(["created", "retrieving", "retrieved", "reading"]);
