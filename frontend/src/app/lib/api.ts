import { API_URL } from "./constants";
import type {
  AgentAskOptions,
  AgentResult,
  ChatEvent,
  ChatMessage,
  Community,
  CommunityEvent,
  CommunityJob,
  CommunityList,
  GraphData,
  IngestEvent,
  LabArm,
  LabArmCatalog,
  LabBudget,
  LabDatasetInfo,
  LabEstimate,
  LabEvent,
  LabFrontiers,
  LabLeaderboard,
  LabManifest,
  LabQuestionRow,
  LabReaderModel,
  LabRowsPage,
  LabRunDetail,
  LabRunRequest,
  LabRunStarted,
  LabRunSummary,
  ProcedureList,
  ProceduralGraphSummary,
  ProcGraphData,
  ProcVersion,
  ProcVersionList,
  RollbackResult,
  Source,
  Theme,
  UploadJob,
} from "./types";

export async function fetchGraph(signal?: AbortSignal): Promise<GraphData> {
  const res = await fetch(`${API_URL}/api/graph-data`, { signal });
  if (!res.ok) throw new Error(`graph-data ${res.status}`);
  return res.json();
}

/** The detected corpus themes, largest community first. */
export async function fetchCommunities(
  limit = 20,
  signal?: AbortSignal,
): Promise<Community[]> {
  const res = await fetch(`${API_URL}/api/communities?limit=${limit}`, { signal });
  if (!res.ok) throw new Error(`communities ${res.status}`);
  const body: CommunityList = await res.json();
  return body.communities ?? [];
}

/** Kick off a background re-clustering; watch it with `subscribeCommunityRebuild`. */
export async function rebuildCommunities(): Promise<CommunityJob> {
  const res = await fetch(`${API_URL}/api/communities/rebuild`, { method: "POST" });
  if (!res.ok) throw new Error(`rebuild communities ${res.status}`);
  return res.json();
}

export async function subscribeCommunityRebuild(
  jobId: string,
  onEvent: (event: CommunityEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(
    `${API_URL}/api/communities/rebuild/${jobId}/events`,
    { signal },
  );
  if (!res.ok) throw new Error(`community events ${res.status}`);
  await readSSE<CommunityEvent>(res, onEvent);
}

export async function clearGraph(): Promise<void> {
  const res = await fetch(`${API_URL}/api/graph`, { method: "DELETE" });
  if (!res.ok) throw new Error(`clear graph ${res.status}`);
}

export async function uploadDocument(file: File, theme: Theme): Promise<UploadJob> {
  const form = new FormData();
  form.append("file", file);
  form.append("theme", theme);
  const res = await fetch(`${API_URL}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed." }));
    throw new Error(err.detail || "Upload failed.");
  }
  return res.json();
}

/**
 * Read a fetch Response body as a Server-Sent-Events stream, invoking `onEvent`
 * for every `data: {json}` frame. Works for both GET and POST SSE endpoints.
 */
async function readSSE<T>(
  res: Response,
  onEvent: (event: T) => void,
): Promise<void> {
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          onEvent(JSON.parse(payload) as T);
        } catch {
          // Ignore malformed frames rather than killing the stream.
        }
      }
    }
  }
}

export async function subscribeIngest(
  jobId: string,
  onEvent: (event: IngestEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/api/upload/${jobId}/events`, { signal });
  if (!res.ok) throw new Error(`ingest events ${res.status}`);
  await readSSE<IngestEvent>(res, onEvent);
}

/** A source excerpt is only provenance if it actually carries a quote. */
function hasQuote(source: Source): boolean {
  return typeof source?.text === "string" && source.text.trim().length > 0;
}

/**
 * `readSSE` trusts the wire, so a `sources` frame that arrived empty or
 * malformed would still become an "N sources" affordance that expands to
 * nothing — the opposite of the credibility this feature exists for. Drop the
 * unusable excerpts, and the whole frame if none survive; every other event
 * passes through untouched.
 */
function withVerifiableSources(event: ChatEvent): ChatEvent | null {
  if (event.type !== "sources") return event;
  if (!Array.isArray(event.data)) return null;
  const sources = event.data.filter(hasQuote);
  return sources.length > 0 ? { type: "sources", data: sources } : null;
}

export async function streamChat(
  query: string,
  history: Pick<ChatMessage, "role" | "content">[],
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history }),
    signal,
  });
  if (!res.ok) throw new Error(`chat ${res.status}`);
  await readSSE<ChatEvent>(res, (event) => {
    const usable = withVerifiableSources(event);
    if (usable) onEvent(usable);
  });
}

// ── Procedural memory ─────────────────────────────────

/**
 * A non-2xx response that keeps the status and the server's own explanation,
 * so the UI can tell "no LLM key configured" (503) apart from "no such graph"
 * (404) instead of printing a bare status code.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, message: string, detail: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * FastAPI's `detail` comes in three shapes: a string, the procedures router's
 * `{message, diagnostics}` object, or request validation's list of `{msg}`.
 */
function describeDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
      .join("; ");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String(detail.message);
  }
  return "";
}

async function apiError(res: Response, what: string): Promise<ApiError> {
  const body = await res.json().catch(() => null);
  const detail = describeDetail(body?.detail);
  return new ApiError(res.status, `${what} ${res.status}`, detail);
}

const procedurePath = (name: string) =>
  `${API_URL}/api/procedures/${encodeURIComponent(name)}`;

/** Every stored procedural graph, with its live version and validation score. */
export async function fetchProcedures(
  signal?: AbortSignal,
): Promise<ProceduralGraphSummary[]> {
  const res = await fetch(`${API_URL}/api/procedures`, { signal });
  if (!res.ok) throw await apiError(res, "procedures");
  const body: ProcedureList = await res.json();
  return body.graphs ?? [];
}

/** One procedural graph in react-force-graph's `{nodes, links}` shape. */
export async function fetchProcedureGraph(
  name: string,
  signal?: AbortSignal,
): Promise<ProcGraphData> {
  const res = await fetch(`${procedurePath(name)}/graph-data`, { signal });
  if (!res.ok) throw await apiError(res, "procedure graph");
  return res.json();
}

/** Version history, newest first. */
export async function fetchProcedureVersions(
  name: string,
  signal?: AbortSignal,
): Promise<ProcVersion[]> {
  const res = await fetch(`${procedurePath(name)}/versions`, { signal });
  if (!res.ok) throw await apiError(res, "procedure versions");
  const body: ProcVersionList = await res.json();
  return body.versions ?? [];
}

/**
 * Re-save an old version as a *new* one (history is append-only: nothing is
 * deleted, and the rollback itself can be rolled back).
 */
export async function rollbackProcedure(
  name: string,
  version: number,
): Promise<RollbackResult> {
  const res = await fetch(`${procedurePath(name)}/rollback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
  if (!res.ok) throw await apiError(res, "rollback");
  return res.json();
}

/**
 * Ask the GraphRAG Navigator: a ReAct agent that answers by walking the
 * knowledge graph with deterministic tools, steered by a procedural graph.
 * Non-streaming — the whole run (every step) comes back at once.
 */
export async function agentAsk(
  query: string,
  { graph, guidance, maxSteps, record }: AgentAskOptions = {},
  signal?: AbortSignal,
): Promise<AgentResult> {
  const body: Record<string, unknown> = { query };
  // `graph: null` is meaningful (run without a procedural graph), so only an
  // omitted graph falls back to the server's default.
  if (graph !== undefined) body.graph = graph;
  if (guidance) body.guidance = guidance;
  if (maxSteps != null) body.max_steps = maxSteps;
  if (record != null) body.record = record;
  const res = await fetch(`${API_URL}/api/agent/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await apiError(res, "agent");
  return res.json();
}

// ── Synapse Lab ───────────────────────────────────────
// Every Lab endpoint lives under /api/lab (backend/app/routers/lab.py). The
// readers below normalize what they get and tolerate absent fields.

const LAB = `${API_URL}/api/lab`;

type Json = Record<string, unknown>;

function asObject(value: unknown): Json | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Json) : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function listOf(body: unknown, key: string): unknown[] {
  if (Array.isArray(body)) return body;
  const obj = asObject(body);
  return obj && Array.isArray(obj[key]) ? (obj[key] as unknown[]) : [];
}

/** The leaf of a stored path ("a/b/c.json" → "c.json"). */
function leafName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

/**
 * A run the backend refused before spending anything: the estimate's upper
 * bound is over the cap (422 whose `detail.estimate` is the estimate).
 */
export class LabRefusedError extends ApiError {
  readonly estimate: LabEstimate | null;

  constructor(status: number, detail: string, estimate: LabEstimate | null) {
    super(status, `lab refused ${status}`, detail);
    this.name = "LabRefusedError";
    this.estimate = estimate;
  }
}

function looksLikeEstimate(value: unknown): value is LabEstimate {
  const obj = asObject(value);
  return !!obj && Array.isArray(obj.cells) && "refuse" in obj;
}

/**
 * The Lab router's refusals are `{message, diagnostics: [...]}`: the message
 * alone ("Invalid Lab request") hides the reason, so the diagnostics follow.
 */
function describeLabDetail(detail: unknown): string {
  const obj = asObject(detail);
  if (obj && typeof obj.message === "string") {
    const diagnostics = Array.isArray(obj.diagnostics)
      ? obj.diagnostics.filter((d): d is string => typeof d === "string" && !!d.trim())
      : [];
    return diagnostics.length ? `${obj.message}: ${diagnostics.join("; ")}` : obj.message;
  }
  return describeDetail(detail);
}

/** A 422 carrying an estimate is a refusal; anything else is a plain ApiError. */
async function labError(res: Response, what: string): Promise<ApiError> {
  const body = asObject(await res.json().catch(() => null));
  const detail = body?.detail;
  const estimate = [asObject(detail)?.estimate, body?.estimate].find(looksLikeEstimate);
  const message = describeLabDetail(detail);
  if (estimate) return new LabRefusedError(res.status, message, estimate);
  return new ApiError(res.status, `${what} ${res.status}`, message);
}

async function postJson(url: string, body: unknown, what: string, signal?: AbortSignal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await labError(res, what);
  return res.json();
}

/** The arm catalog (floors · passage baselines · graph arms) and batch availability. */
export async function fetchLabArms(signal?: AbortSignal): Promise<LabArmCatalog> {
  const res = await fetch(`${LAB}/arms`, { signal });
  if (!res.ok) throw await labError(res, "lab arms");
  const body = await res.json();
  const arms = listOf(body, "arms")
    .map(asObject)
    .filter((a): a is Json => !!a && typeof a.name === "string")
    .map((a): LabArm => {
      const source = asObject(a.source);
      const family = asString(a.family) ?? "graph";
      return {
        name: String(a.name),
        family,
        family_title: asString(a.family_title) ?? undefined,
        title: asString(a.title) ?? String(a.name),
        description: asString(a.description) ?? "",
        source: {
          citation: asString(source?.citation) ?? "",
          url: asString(source?.url) ?? "",
        },
        retrieval_llm_calls: asNumber(a.retrieval_llm_calls) ?? 0,
        needs_graph: Boolean(a.needs_graph),
        is_null: typeof a.is_null === "boolean" ? a.is_null : family === "null",
        k_role: asString(a.k_role) ?? undefined,
      };
    });
  const batch = asObject(body)?.batch_available;
  return { arms, batchAvailable: typeof batch === "boolean" ? batch : true };
}

/** Reader models with their hand-recorded prices; `[]` when the backend has no list. */
export async function fetchLabModels(signal?: AbortSignal): Promise<LabReaderModel[]> {
  const res = await fetch(`${LAB}/models`, { signal });
  if (!res.ok) throw await labError(res, "lab models");
  return listOf(await res.json(), "models")
    .map(asObject)
    .filter((m): m is Json => !!m && typeof (m.name ?? m.id) === "string")
    .map((m) => {
      const id = String(m.name ?? m.id);
      return {
        id,
        input: asNumber(m.input_usd_per_1m) ?? NaN,
        output: asNumber(m.output_usd_per_1m) ?? NaN,
        reasoning: typeof m.reasoning === "boolean" ? m.reasoning : /^(gpt-5|o[1-9])/.test(id),
        checkedOn: asString(m.price_checked_on) ?? "",
      };
    });
}

const KNOWN_DATASETS = new Set(["demo", "qa-file", "hotpotqa"]);
const QA_FILE_PREFIX = "qa-file:";

/** `{test: 10, train: 14}` or `["test", "train"]` → names + counts. */
function splitsOf(value: unknown): { splits: string[]; counts: Record<string, number> } {
  if (Array.isArray(value)) {
    return { splits: value.filter((s): s is string => typeof s === "string"), counts: {} };
  }
  const obj = asObject(value);
  if (!obj) return { splits: [], counts: {} };
  const counts: Record<string, number> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === "number") counts[k] = v;
  }
  return { splits: Object.keys(obj), counts };
}

function datasetEntry(raw: Json, available: boolean): LabDatasetInfo | null {
  const id = asString(raw.id);
  const name = asString(raw.name) ?? id;
  if (!name) return null;
  const fromId = id?.startsWith(QA_FILE_PREFIX) ? id.slice(QA_FILE_PREFIX.length) : null;
  const file = asString(raw.file) ?? fromId ?? (name.startsWith(QA_FILE_PREFIX) ? name.slice(QA_FILE_PREFIX.length) : null);
  const dataset = file ? "qa-file" : KNOWN_DATASETS.has(name) ? name : (id ?? name);
  if (dataset === "qa-file" && !file) return null; // a group header without a file
  const { splits, counts } = splitsOf(raw.splits);
  const reason = asString(raw.reason) ?? asString(raw.error);
  return {
    key: file ? `${QA_FILE_PREFIX}${leafName(file)}` : (id ?? dataset),
    dataset,
    file: file ? leafName(file) : null,
    label: asString(raw.title) ?? asString(raw.label) ?? (file ? leafName(file) : dataset),
    description: asString(raw.description) ?? (available ? asString(raw.note) : null),
    splits,
    splitCounts: counts,
    defaultSplit: asString(raw.default_split),
    n: asNumber(raw.n),
    defaultN: asNumber(raw.default_n),
    available: available && !reason,
    reason,
  };
}

/**
 * Datasets a run can use (demo, uploaded QA files, HotpotQA when its graph is
 * present) followed by the ones that are not usable yet, with the reason.
 */
export async function fetchLabDatasets(signal?: AbortSignal): Promise<LabDatasetInfo[]> {
  const res = await fetch(`${LAB}/datasets`, { signal });
  if (!res.ok) throw await labError(res, "lab datasets");
  const body = await res.json();
  const out: LabDatasetInfo[] = [];
  const add = (item: unknown, available: boolean) => {
    const raw = typeof item === "string" ? { name: item } : asObject(item);
    const entry = raw && datasetEntry(raw, available);
    if (entry && !out.some((d) => d.key === entry.key)) out.push(entry);
  };
  for (const item of listOf(body, "datasets")) add(item, true);
  for (const item of listOf(asObject(body)?.unavailable, "unavailable")) add(item, false);
  return out;
}

/** The free dry-run estimate: per (arm × budget) calls, tokens, point and upper $. */
export async function estimateLab(
  request: LabRunRequest,
  signal?: AbortSignal,
): Promise<LabEstimate> {
  const body = await postJson(`${LAB}/estimate`, request, "lab estimate", signal);
  const obj = asObject(body);
  return (looksLikeEstimate(obj?.estimate) ? obj.estimate : body) as LabEstimate;
}

function started(body: unknown, fallbackRunId = ""): LabRunStarted {
  const obj = asObject(body) ?? {};
  return {
    run_id: asString(obj.run_id) ?? fallbackRunId,
    job_id: asString(obj.job_id),
    status: asString(obj.status),
    estimate: looksLikeEstimate(obj.estimate) ? obj.estimate : null,
  };
}

/**
 * Start a run (a background job). Refused with a `LabRefusedError` (422 + the
 * estimate) when the upper bound is over the cap — nothing is spent then.
 */
export async function startLabRun(request: LabRunRequest): Promise<LabRunStarted> {
  return started(await postJson(`${LAB}/runs`, request, "lab run"));
}

/**
 * Continue a run: a submitted batch is polled and, once done, collected and
 * scored (no new spend); an aborted or refused run reads what is left under
 * `max_usd` — which DOES spend.
 */
export async function resumeLabRun(
  runId: string,
  options: { max_usd?: number | null; retry_failed?: boolean } = {},
): Promise<LabRunStarted> {
  const body: Record<string, unknown> = {};
  if (options.max_usd != null) body.max_usd = options.max_usd;
  if (options.retry_failed) body.retry_failed = true;
  return started(
    await postJson(`${LAB}/runs/${encodeURIComponent(runId)}/resume`, body, "lab resume"),
    runId,
  );
}

function isUnknownRun(event: LabEvent): boolean {
  if (event.type !== "error") return false;
  return /unknown\b.*\b(job|run)|not found/i.test(String(event.error ?? event.data ?? ""));
}

/**
 * Follow a run's job (SSE): the runner's phase / progress / refused / aborted /
 * batch_submitted events, then `done` (with the run's final state) or `error`.
 * `ids` are tried in order — the run id, then the job id.
 */
export async function subscribeLabRun(
  ids: (string | null | undefined)[],
  onEvent: (event: LabEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const candidates = Array.from(new Set(ids.filter((id): id is string => !!id)));
  for (let i = 0; i < candidates.length; i++) {
    const last = i === candidates.length - 1;
    const res = await fetch(`${LAB}/runs/${encodeURIComponent(candidates[i])}/events`, {
      signal,
    });
    if (!res.ok) {
      if (!last && res.status === 404) continue;
      throw await labError(res, "lab events");
    }
    let first = true;
    let unknown = false;
    await readSSE<LabEvent>(res, (event) => {
      if (first) {
        first = false;
        if (!last && isUnknownRun(event)) unknown = true;
      }
      if (!unknown) onEvent(event);
    });
    if (!unknown) return;
  }
}

function budgetsOf(value: unknown): LabBudget[] {
  if (!Array.isArray(value)) return [];
  return value.map((b) => (typeof b === "number" ? b : null));
}

function runSummary(raw: Json): LabRunSummary | null {
  const config = asObject(raw.config) ?? {};
  const runId = asString(raw.run_id) ?? asString(raw.id);
  if (!runId) return null;
  const dataset = raw.dataset ?? config.dataset;
  const datasetObj = asObject(dataset);
  const arms = raw.arms ?? config.arms;
  return {
    run_id: runId,
    status: asString(raw.status) ?? "created",
    created_at: asString(raw.created_at),
    updated_at: asString(raw.updated_at),
    mode: asString(raw.mode) ?? asString(config.mode),
    dataset: asString(dataset) ?? asString(datasetObj?.name),
    n: asNumber(raw.n) ?? asNumber(datasetObj?.n),
    arms: Array.isArray(arms) ? arms.map(String) : [],
    budgets: budgetsOf(raw.budgets ?? config.budgets),
    reader_model: asString(raw.reader_model) ?? asString(config.reader_model),
  };
}

/** Every stored run, newest first. */
export async function fetchLabRuns(signal?: AbortSignal): Promise<LabRunSummary[]> {
  const res = await fetch(`${LAB}/runs`, { signal });
  if (!res.ok) throw await labError(res, "lab runs");
  return listOf(await res.json(), "runs")
    .map(asObject)
    .filter((r): r is Json => !!r)
    .map(runSummary)
    .filter((r): r is LabRunSummary => r != null);
}

function rowsPage(value: unknown): LabRowsPage | null {
  if (Array.isArray(value)) {
    return { rows: value as LabQuestionRow[], total: value.length, offset: 0 };
  }
  const obj = asObject(value);
  if (!obj || !Array.isArray(obj.rows)) return null;
  return {
    rows: obj.rows as LabQuestionRow[],
    total: asNumber(obj.total) ?? obj.rows.length,
    offset: asNumber(obj.offset) ?? 0,
  };
}

/**
 * One run: manifest, leaderboard (null until scored), frontiers, and a page of
 * per-question rows — optionally for one (arm, budget) cell. `limit: 0` skips
 * the rows (the router still reports their total).
 */
export async function fetchLabRun(
  runId: string,
  options: { arm?: string; budget?: LabBudget; offset?: number; limit?: number } = {},
  signal?: AbortSignal,
): Promise<LabRunDetail> {
  const params = new URLSearchParams();
  if (options.arm) params.set("arm", options.arm);
  if (options.arm && options.budget !== undefined) {
    params.set("budget", options.budget == null ? "default" : String(options.budget));
  }
  if (options.offset != null) params.set("offset", String(options.offset));
  if (options.limit != null) params.set("limit", String(options.limit));
  const query = params.toString();
  const res = await fetch(
    `${LAB}/runs/${encodeURIComponent(runId)}${query ? `?${query}` : ""}`,
    { signal },
  );
  if (!res.ok) throw await labError(res, "lab run");
  const body = asObject(await res.json()) ?? {};
  const manifest = (asObject(body.manifest) ?? {}) as LabManifest;
  if (!manifest.status && typeof body.status === "string") manifest.status = body.status;
  const board = asObject(body.leaderboard);
  const leaderboard = board && Array.isArray(board.rows) ? (board as unknown as LabLeaderboard) : null;
  const frontiers = (asObject(body.frontiers) ?? asObject(board?.frontiers) ??
    null) as LabFrontiers | null;
  return {
    run_id: asString(body.run_id) ?? manifest.run_id ?? runId,
    active: body.active === true,
    manifest,
    leaderboard,
    frontiers,
    rows: rowsPage(body.rows),
  };
}

/**
 * Upload a QA file (JSON array or JSONL of {question, answer[, id][, split]}).
 * Validated in full server-side; a different file under an existing name is a
 * 409 (past runs refer to it by content hash).
 */
export async function uploadLabQaFile(
  file: File,
): Promise<{ file: string; n: number | null; status: string | null }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${LAB}/qa-files`, { method: "POST", body: form });
  if (!res.ok) throw await labError(res, "qa file upload");
  const body = asObject(await res.json()) ?? {};
  const id = asString(body.id);
  const stored =
    asString(body.file) ??
    (id?.startsWith(QA_FILE_PREFIX) ? id.slice(QA_FILE_PREFIX.length) : null) ??
    file.name;
  return { file: leafName(stored), n: asNumber(body.n), status: asString(body.status) };
}
