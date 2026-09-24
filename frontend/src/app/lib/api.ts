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
