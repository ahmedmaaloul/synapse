"use client";

import { useEffect, useRef, useState } from "react";
import {
  ChevronRight,
  FileText,
  Globe2,
  Layers,
  Send,
  Sparkles,
  Target,
  User,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamChat } from "../lib/api";
import { colorForType } from "../lib/constants";
import type {
  ChatMessage,
  Citation,
  IngestResult,
  ReasoningPath,
  RetrievalMode,
  Source,
} from "../lib/types";

interface ChatPanelProps {
  ingestResult: IngestResult | null;
  onCitations: (names: string[]) => void;
  onFocusCitation: (name: string) => void;
}

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

const SUGGESTIONS = [
  "Summarize the key entities",
  "How is everything connected?",
  "What are the main themes across the corpus?",
];

/** Paths shown before the "show all" fold. */
const PATH_PREVIEW = 3;

/**
 * Which retrieval path the backend router picked, inferred from the citations
 * it returned: community citations mean the question was answered from
 * corpus-level summaries ("global search") rather than entity neighborhoods.
 */
function retrievalMode(citations: Citation[]): RetrievalMode {
  return citations.some((c) => c.kind === "community") ? "global" : "local";
}

/** Names to highlight in the graph — communities highlight their members. */
function highlightNames(citations: Citation[]): string[] {
  return citations.flatMap((c) =>
    c.kind === "community" ? (c.members ?? []) : [c.name],
  );
}

function ModeBadge({ mode }: { mode: RetrievalMode }) {
  const global = mode === "global";
  const Icon = global ? Globe2 : Target;
  return (
    <span
      title={
        global
          ? "Global search — answered from corpus-level community summaries"
          : "Local search — answered from entity neighborhoods and reasoning paths"
      }
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${
        global
          ? "border-amber-400/30 bg-amber-400/10 text-amber-300/90"
          : "border-indigo-500/30 bg-indigo-500/10 text-indigo-300/90"
      }`}
    >
      <Icon size={9} strokeWidth={2.5} />
      {global ? "Global" : "Local"}
    </span>
  );
}

/**
 * Direction of hop `i`. The backend walks the graph undirected, so a chain can
 * traverse an edge backwards; `dirs[i] === false` means the relationship
 * actually points from nodes[i + 1] back to nodes[i]. Missing `dirs` (older
 * backend) degrades to forward rather than throwing away the path.
 */
function isForward(path: ReasoningPath, i: number): boolean {
  return path.dirs?.[i] ?? true;
}

/** One "—REL→" / "←REL—" connector between two entities in a reasoning chain. */
function Hop({ rel, forward }: { rel: string; forward: boolean }) {
  const arrow = <span className="text-indigo-400/60">{forward ? "→" : "←"}</span>;
  const tail = <span className="text-[#3f3f46]">—</span>;
  return (
    <span
      title={forward ? `points forward: ${rel}` : `points backward: ${rel}`}
      className="flex items-center gap-0.5 font-mono text-[9px] uppercase tracking-wide text-[#52525b]"
    >
      {forward ? tail : arrow}
      {rel}
      {forward ? arrow : tail}
    </span>
  );
}

/** The multi-hop chain that connects the answer's entities. */
function ReasoningTrail({
  paths,
  onFocus,
}: {
  paths: ReasoningPath[];
  onFocus: (name: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? paths : paths.slice(0, PATH_PREVIEW);

  return (
    <div className="mt-2.5 rounded-md border border-[#27272a] bg-[#18181b]/60 px-2.5 py-2">
      <div className="mb-1.5 flex items-center gap-1.5">
        <Sparkles size={9} className="text-indigo-400/70" />
        <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
          Reasoning {paths.length > 1 ? "paths" : "path"}
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {shown.map((path) => (
          <div key={path.text} className="flex flex-wrap items-center gap-x-1 gap-y-1">
            {path.nodes.map((node, i) => (
              <span key={`${path.text}:${i}`} className="flex items-center gap-1">
                <button
                  onClick={() => onFocus(node)}
                  title="Center in graph"
                  className="rounded border border-transparent px-1 py-px text-[11px] font-medium text-[#d4d4d8] transition-colors hover:border-indigo-500/40 hover:bg-[#27272a] hover:text-[#fafafa]"
                >
                  {node}
                </button>
                {i < path.rels.length && (
                  <Hop rel={path.rels[i]} forward={isForward(path, i)} />
                )}
              </span>
            ))}
          </div>
        ))}
      </div>
      {paths.length > PATH_PREVIEW && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 text-[10px] font-medium text-[#52525b] transition-colors hover:text-[#a1a1aa]"
        >
          {expanded ? "Show fewer" : `+${paths.length - PATH_PREVIEW} more paths`}
        </button>
      )}
    </div>
  );
}

/**
 * The passages of the original documents that grounded the answer.
 *
 * Collapsed to a single "N sources" line by default: provenance should stay out
 * of the way while reading, and be one click away when a claim needs checking.
 * Unlike citations (entity names) and reasoning paths (graph structure), these
 * are verbatim source prose — the only thing here a reader can actually audit
 * the answer against.
 */
function SourceTrail({ sources }: { sources: Source[] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-2.5 overflow-hidden rounded-md border border-[#27272a] bg-[#18181b]/60">
      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        title={expanded ? "Hide source excerpts" : "Show the source excerpts behind this answer"}
        className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left transition-colors hover:bg-[#27272a]/40"
      >
        <ChevronRight
          size={9}
          className={`shrink-0 text-[#52525b] transition-transform ${
            expanded ? "rotate-90" : ""
          }`}
        />
        <FileText size={9} className="shrink-0 text-indigo-400/70" />
        <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
      </button>

      {expanded && (
        <ul className="flex flex-col gap-2 border-t border-[#27272a] px-2.5 py-2">
          {sources.map((source, i) => (
            <li key={source.id || `${source.document}:${source.index}:${i}`}>
              <div
                className="truncate font-mono text-[10px] uppercase tracking-wider text-[#71717a]"
                title={source.document || "Unknown document"}
              >
                {source.document || "Unknown document"}
                <span className="text-[#3f3f46]"> · chunk {source.index + 1}</span>
              </div>
              <blockquote className="mt-0.5 border-l border-indigo-500/30 pl-2 text-[11px] leading-relaxed text-[#a1a1aa]">
                {source.text}
              </blockquote>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function ChatPanel({
  ingestResult,
  onCitations,
  onFocusCitation,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      content: "Synapse engine ready. Ingest a document, then ask anything.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!ingestResult) return;
    const themes = ingestResult.communities
      ? ` ${ingestResult.communities} themes detected.`
      : "";
    setMessages((prev) => [
      ...prev,
      {
        id: nextId(),
        role: "system",
        content: `Ingested "${ingestResult.filename}" → ${ingestResult.nodes_created} entities, ${ingestResult.relationships_created} relationships.${themes} Context loaded.`,
      },
    ]);
  }, [ingestResult]);

  const send = async (text?: string) => {
    const query = (text ?? input).trim();
    if (!query || isLoading) return;

    const history = messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.content)
      .map((m) => ({ role: m.role, content: m.content }));

    const userMsg: ChatMessage = { id: nextId(), role: "user", content: query };
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      userMsg,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        paths: [],
        sources: [],
      },
    ]);
    setInput("");
    setIsLoading(true);

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    try {
      await streamChat(query, history, (event) => {
        if (event.type === "citations") {
          const cites = event.data;
          patch((m) => ({ ...m, citations: cites }));
          onCitations(highlightNames(cites));
        } else if (event.type === "paths") {
          const paths = event.data;
          patch((m) => ({ ...m, paths }));
        } else if (event.type === "sources") {
          const sources = event.data;
          patch((m) => ({ ...m, sources }));
        } else if (event.type === "token") {
          patch((m) => ({ ...m, content: m.content + event.data }));
        } else if (event.type === "error") {
          patch((m) => ({ ...m, content: `⚠️ ${event.data}` }));
        }
      });
    } catch (err) {
      console.error("Chat error:", err);
      patch((m) => ({
        ...m,
        content: m.content || "⚠️ Connection refused. Is the backend running?",
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const showSuggestions =
    messages.filter((m) => m.role === "user").length === 0 && !isLoading;

  return (
    <div className="relative flex h-full w-full flex-col bg-[#09090b] pt-2">
      {isLoading && (
        <div className="loading-bar">
          <div className="loading-indicator" />
        </div>
      )}

      <div className="flex items-center justify-between border-b border-[#27272a] px-5 py-2.5">
        <h3 className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-wider text-[#e4e4e7]">
          <Sparkles size={13} className="text-indigo-400" />
          GraphRAG Chat
        </h3>
        <div className="font-mono text-[11px] text-[#71717a]">v0.3.0</div>
      </div>

      {/* Messages */}
      <div className="w-full flex-1 overflow-y-auto px-5 py-4 text-[13px]">
        <div className="flex w-full max-w-full flex-col pb-24">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className="msg-enter group flex w-full flex-col border-b border-[#27272a]/50 py-3 last:border-0"
            >
              {msg.role === "system" ? (
                <div className="flex items-center gap-2 py-1 font-mono text-[12px] tracking-tight text-[#71717a]">
                  <ChevronRight size={12} />
                  {msg.content}
                </div>
              ) : (
                <div className="flex gap-3">
                  <div className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded border border-[#27272a] bg-[#18181b]">
                    {msg.role === "user" ? (
                      <User size={12} className="text-[#a1a1aa]" />
                    ) : (
                      <div className="h-1.5 w-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    {msg.role === "user" ? (
                      <div className="whitespace-pre-wrap font-medium leading-relaxed text-[#fafafa]">
                        {msg.content}
                      </div>
                    ) : (
                      <div className="prose-chat leading-relaxed text-[#d4d4d8]">
                        {msg.content ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        ) : (
                          isLoading && <span className="caret" />
                        )}
                      </div>
                    )}

                    {/* Reasoning chain (multi-hop local search) */}
                    {msg.role === "assistant" &&
                      msg.paths &&
                      msg.paths.length > 0 && (
                        <ReasoningTrail
                          paths={msg.paths}
                          onFocus={onFocusCitation}
                        />
                      )}

                    {/* Citations + which retrieval path the router picked */}
                    {msg.role === "assistant" &&
                      msg.citations &&
                      msg.citations.length > 0 && (
                        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                          <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
                            Grounded in
                          </span>
                          {msg.citations.slice(0, 8).map((c) => {
                            const isCommunity = c.kind === "community";
                            return (
                              <button
                                key={c.id ?? c.name}
                                onClick={() =>
                                  onFocusCitation(
                                    isCommunity ? (c.members?.[0] ?? c.name) : c.name,
                                  )
                                }
                                className={`flex max-w-full items-center gap-1 truncate rounded border bg-[#18181b] px-1.5 py-0.5 text-[10px] font-medium text-[#d4d4d8] transition-colors hover:text-[#fafafa] ${
                                  isCommunity
                                    ? "border-amber-400/30 hover:border-amber-400/60"
                                    : "border-[#27272a] hover:border-indigo-500/50"
                                }`}
                                title={
                                  isCommunity
                                    ? `Theme · ${c.size ?? c.members?.length ?? 0} entities`
                                    : "Center in graph"
                                }
                              >
                                {isCommunity ? (
                                  <Layers size={9} className="shrink-0 text-amber-400" />
                                ) : (
                                  <span
                                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                                    style={{
                                      backgroundColor: colorForType(c.type || ""),
                                    }}
                                  />
                                )}
                                {c.name}
                              </button>
                            );
                          })}
                          <ModeBadge mode={retrievalMode(msg.citations)} />
                        </div>
                      )}

                    {/* Provenance: the source prose behind the answer */}
                    {msg.role === "assistant" &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <SourceTrail sources={msg.sources} />
                      )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {showSuggestions && (
            <div className="mt-3 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-md border border-[#27272a] bg-[#18181b] px-2.5 py-1.5 text-[12px] text-[#a1a1aa] transition-colors hover:border-indigo-500/40 hover:text-[#fafafa]"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className="absolute bottom-0 w-full border-t border-[#27272a] bg-[#09090b] pb-4 pt-2">
        <div className="mx-auto max-w-full px-4">
          <div className="relative flex w-full items-center rounded-md border border-[#27272a] bg-[#18181b] shadow-sm transition-colors focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/20">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Query the knowledge graph…"
              disabled={isLoading}
              aria-label="Chat message"
              className="w-full rounded-md border-0 bg-transparent py-2.5 pl-3 pr-10 text-[13px] font-medium text-[#fafafa] outline-none placeholder:text-[#71717a] focus:ring-0"
            />
            <button
              className="absolute right-1.5 rounded p-1.5 text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa] disabled:opacity-50"
              onClick={() => send()}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
