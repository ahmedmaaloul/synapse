import { API_URL } from "./constants";
import type {
  ChatEvent,
  ChatMessage,
  GraphData,
  IngestEvent,
  Theme,
  UploadJob,
} from "./types";

export async function fetchGraph(signal?: AbortSignal): Promise<GraphData> {
  const res = await fetch(`${API_URL}/api/graph-data`, { signal });
  if (!res.ok) throw new Error(`graph-data ${res.status}`);
  return res.json();
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
  await readSSE<ChatEvent>(res, onEvent);
}
