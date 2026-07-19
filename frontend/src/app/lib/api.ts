import { API_URL } from "./constants";
import type {
  ChatEvent,
  ChatMessage,
  Community,
  CommunityEvent,
  CommunityJob,
  CommunityList,
  GraphData,
  IngestEvent,
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
