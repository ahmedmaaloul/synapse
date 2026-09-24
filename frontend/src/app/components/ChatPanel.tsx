"use client";

import { useEffect, useRef, useState } from "react";
import pkg from "../../../package.json";
import {
  ChevronRight,
  Compass,
  FileText,
  Globe2,
  Layers,
  Loader2,
  MessageSquare,
  Send,
  Sparkles,
  Target,
  TriangleAlert,
  User,
  Waypoints,
  Workflow,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { agentAsk, ApiError, streamChat } from "../lib/api";
import { colorForType, LOCALIZATION_INFO } from "../lib/constants";
import type {
  AgentResult,
  AgentStep,
  ChatMessage,
  ChatMode,
  Citation,
  IngestResult,
  LocalizationMethod,
  ReasoningPath,
  RetrievalMode,
  Source,
} from "../lib/types";
import SegmentedToggle, { type SegmentedOption } from "./SegmentedToggle";

interface ChatPanelProps {
  ingestResult: IngestResult | null;
  onCitations: (names: string[]) => void;
  onFocusCitation: (name: string) => void;
  /** Fires after every successful Navigator run (for the Procedures overlay). */
  onNavigatorRun?: (result: AgentResult) => void;
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

const MODE_OPTIONS: SegmentedOption<ChatMode>[] = [
  {
    value: "chat",
    label: "Chat",
    icon: MessageSquare,
    title: "One retrieval, one streamed answer",
  },
  {
    value: "navigator",
    label: "Navigator",
    icon: Compass,
    title:
      "An agent walks the knowledge graph step by step, steered by the procedural graph",
  },
];

/**
 * Navigator tools whose arguments name knowledge-graph entities. After a run,
 * those entities are highlighted in the topology, like chat citations.
 */
const ENTITY_TOOLS = new Set(["neighbors", "read_sources", "find_path"]);

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

// ── Navigator mode ─────────────────────────────────────

/** `tool(arg="value", …)` — the action exactly as the agent issued it. */
function formatAction(step: AgentStep): string {
  const args = step.args as unknown;
  let inner = "";
  if (typeof args === "string") inner = JSON.stringify(args);
  else if (Array.isArray(args)) inner = args.map((a) => JSON.stringify(a)).join(", ");
  else if (args && typeof args === "object") {
    inner = Object.entries(args)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(", ");
  }
  return `${step.action}(${inner})`;
}

/** Entity names the run touched, for highlighting in the knowledge graph. */
function visitedEntities(result: AgentResult): string[] {
  const names = new Set<string>();
  for (const step of result.steps) {
    if (!step.action || !ENTITY_TOOLS.has(step.action)) continue;
    const args = step.args;
    if (!args || typeof args !== "object") continue;
    for (const value of Object.values(args)) {
      if (typeof value === "string" && value.trim()) names.add(value.trim());
    }
  }
  return Array.from(names);
}

/** Turn a failed run into something a person can act on. */
function describeNavigatorError(err: unknown): { message: string; detail?: string } {
  if (err instanceof ApiError) {
    const detail = err.detail || undefined;
    if (err.status === 503) {
      return {
        message:
          "The Navigator needs an LLM to reason with, and the backend has no provider configured. Add an LLM key to the backend and try again.",
        detail,
      };
    }
    if (err.status === 404) {
      return {
        message:
          "The Navigator isn't available: this backend has no navigator endpoint or no procedural graph to follow.",
        detail,
      };
    }
    return { message: "The Navigator run failed.", detail: detail ?? err.message };
  }
  return { message: "Connection refused. Is the backend running?" };
}

/** "PG: exact" — how this step was placed on the procedural graph. */
function LocalizationChip({ step }: { step: AgentStep }) {
  if (step.guidance_error) {
    return (
      <span
        title={`Guidance failed for this step; the agent continued without it.\n${step.guidance_error}`}
        className="inline-flex shrink-0 items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-rose-300/90"
      >
        <Workflow size={9} strokeWidth={2.5} />
        PG: failed
      </span>
    );
  }
  const raw = step.localization;
  if (!raw) return null;
  const loc = typeof raw === "string" ? { method: raw } : raw;
  const info = LOCALIZATION_INFO[loc.method as LocalizationMethod];
  const node = loc.node_id ?? step.active_node;
  const hint = [
    info?.hint ?? `Localization: ${loc.method}`,
    node ? `Active node: ${node}` : "",
    loc.score != null ? `Similarity: ${loc.score.toFixed(2)}` : "",
    step.guidance_context_chars
      ? `${step.guidance_context_chars.toLocaleString()} chars of guidance in the prompt`
      : "",
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <span
      title={hint}
      className={`inline-flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider ${
        info?.className ?? "border-[#3f3f46] bg-[#27272a]/60 text-[#a1a1aa]"
      }`}
    >
      <Workflow size={9} strokeWidth={2.5} />
      PG: {info?.label ?? loc.method}
    </span>
  );
}

/** Collapsed by default: tool output is long, and the thought says what mattered. */
function ObservationFold({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  if (!text) return null;
  return (
    <div className="mt-1.5">
      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-[#52525b] transition-colors hover:text-[#a1a1aa]"
      >
        <ChevronRight
          size={9}
          className={`shrink-0 transition-transform ${expanded ? "rotate-90" : ""}`}
        />
        Observation
        <span className="font-mono normal-case tracking-normal text-[#3f3f46]">
          {text.length.toLocaleString()} chars
        </span>
      </button>
      {expanded && (
        <pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded border border-[#27272a] bg-[#09090b] px-2 py-1.5 font-mono text-[10.5px] leading-relaxed text-[#a1a1aa]">
          {text}
        </pre>
      )}
    </div>
  );
}

const STOP_LABEL: Record<string, { label: string; className: string }> = {
  max_steps: {
    label: "Step limit reached",
    className: "border-amber-400/30 bg-amber-400/10 text-amber-300/90",
  },
  error: {
    label: "Stopped on error",
    className: "border-rose-500/30 bg-rose-500/10 text-rose-300/90",
  },
};

/** A Navigator run: numbered steps, then the answer and what it cost. */
function NavigatorRun({ result }: { result: AgentResult }) {
  const { usage } = result;
  const stop = result.stopped !== "answer" ? STOP_LABEL[result.stopped] : undefined;

  return (
    <div className="flex flex-col gap-2.5">
      <div className="overflow-hidden rounded-md border border-[#27272a] bg-[#18181b]/60">
        <div className="flex items-center gap-1.5 border-b border-[#27272a] px-2.5 py-2">
          <Waypoints size={9} className="shrink-0 text-indigo-400/70" />
          <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
            Navigator trace · {result.steps.length}{" "}
            {result.steps.length === 1 ? "step" : "steps"}
          </span>
          <span
            className="ml-auto truncate font-mono text-[10px] text-[#52525b]"
            title="The procedural graph that steered this run"
          >
            {result.graph
              ? `${result.graph.name} v${result.graph.version}`
              : "no procedural graph"}
          </span>
        </div>
        {result.steps.length > 0 ? (
          <ol className="flex flex-col divide-y divide-[#27272a]/60">
            {result.steps.map((step, i) => (
              <li key={i} className="flex gap-2.5 px-2.5 py-2">
                <span className="mt-px w-4 shrink-0 text-right font-mono text-[10px] text-[#52525b]">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  {step.thought && (
                    <p className="text-[12px] leading-relaxed text-[#a1a1aa]">
                      <span className="mr-1.5 text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
                        Thought
                      </span>
                      {step.thought}
                    </p>
                  )}
                  <div className="mt-1 flex flex-wrap items-center gap-1.5">
                    {step.action ? (
                      <code className="max-w-full break-all rounded border border-[#27272a] bg-[#09090b] px-1.5 py-0.5 font-mono text-[11px] text-indigo-300/90">
                        {formatAction(step)}
                      </code>
                    ) : (
                      <>
                        <span className="rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 font-mono text-[10px] text-rose-300/90">
                          unparseable
                        </span>
                        {step.raw_action && (
                          <code
                            title="What the model wrote instead of a valid action"
                            className="max-w-full break-all font-mono text-[11px] text-[#71717a] line-through decoration-rose-500/40"
                          >
                            {step.raw_action}
                          </code>
                        )}
                      </>
                    )}
                    <LocalizationChip step={step} />
                  </div>
                  <ObservationFold text={step.observation} />
                </div>
              </li>
            ))}
          </ol>
        ) : (
          <p className="px-2.5 py-2 text-[11px] text-[#52525b]">No steps were taken.</p>
        )}
      </div>

      <div>
        <div className="mb-1 flex items-center gap-1.5">
          <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
            Answer
          </span>
          {stop && (
            <span
              className={`rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${stop.className}`}
            >
              {stop.label}
            </span>
          )}
        </div>
        <div className="prose-chat leading-relaxed text-[#d4d4d8]">
          {result.answer ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
          ) : (
            <span className="text-[#71717a]">
              No answer: the Navigator stopped before calling{" "}
              <code>answer</code>.
            </span>
          )}
        </div>
        {result.stopped === "error" && result.error && (
          <p className="mt-1 break-words font-mono text-[10px] leading-relaxed text-rose-300/60">
            {result.error}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 font-mono text-[10px] text-[#52525b]">
        <span>
          {usage.llm_calls} LLM {usage.llm_calls === 1 ? "call" : "calls"}
        </span>
        <span className="text-[#3f3f46]">·</span>
        <span title="Extra LLM calls spent turning the procedural graph into prose (0 in raw mode)">
          {usage.guidance_llm_calls} guidance
        </span>
        <span className="text-[#3f3f46]">·</span>
        <span
          title={
            usage.estimated
              ? "The provider reported no token usage; counts are estimated at 4 characters per token"
              : "Token counts reported by the provider"
          }
        >
          {usage.input_tokens.toLocaleString()} in / {usage.output_tokens.toLocaleString()}{" "}
          out tokens{usage.estimated ? " (est.)" : ""}
        </span>
        <span className="text-[#3f3f46]">·</span>
        <span>{result.latency_s.toFixed(1)}s</span>
        {result.parse_failures > 0 && (
          <>
            <span className="text-[#3f3f46]">·</span>
            <span className="text-amber-400/70">
              {result.parse_failures} parse{" "}
              {result.parse_failures === 1 ? "failure" : "failures"}
            </span>
          </>
        )}
      </div>
    </div>
  );
}

export default function ChatPanel({
  ingestResult,
  onCitations,
  onFocusCitation,
  onNavigatorRun,
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
  const [mode, setMode] = useState<ChatMode>("chat");
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

  const patchMessage = (id: string, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));

  /**
   * One Navigator run. Default procedural graph, raw guidance (the serialized
   * subgraph goes straight into the agent's prompt: no extra LLM call).
   */
  const navigate = async (query: string) => {
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", content: query, mode: "navigator" },
      { id: assistantId, role: "assistant", content: "", mode: "navigator" },
    ]);
    setInput("");
    setIsLoading(true);

    try {
      const result = await agentAsk(query, { guidance: "raw" });
      patchMessage(assistantId, (m) => ({
        ...m,
        content: result.answer ?? "",
        agent: result,
      }));
      const visited = visitedEntities(result);
      if (visited.length > 0) onCitations(visited);
      onNavigatorRun?.(result);
    } catch (err) {
      console.error("Navigator error:", err);
      patchMessage(assistantId, (m) => ({
        ...m,
        agentError: describeNavigatorError(err),
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const send = async (text?: string) => {
    const query = (text ?? input).trim();
    if (!query || isLoading) return;
    if (mode === "navigator") return navigate(query);

    // Navigator turns stay out of the chat history: the agent is stateless,
    // and its answers rest on tool observations this retrieval never saw.
    const history = messages
      .filter(
        (m) =>
          (m.role === "user" || m.role === "assistant") &&
          m.content &&
          m.mode !== "navigator",
      )
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
  const showNavigatorIntro =
    mode === "navigator" && !messages.some((m) => m.mode === "navigator");

  return (
    <div className="relative flex h-full w-full flex-col bg-[#09090b] pt-2">
      {isLoading && (
        <div className="loading-bar">
          <div className="loading-indicator" />
        </div>
      )}

      <div className="flex items-center justify-between border-b border-[#27272a] px-5 py-2.5">
        <h3 className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-wider text-[#e4e4e7]">
          {mode === "navigator" ? (
            <Compass size={13} className="text-indigo-400" />
          ) : (
            <Sparkles size={13} className="text-indigo-400" />
          )}
          {mode === "navigator" ? "GraphRAG Navigator" : "GraphRAG Chat"}
        </h3>
        <div className="flex items-center gap-3">
          <SegmentedToggle
            label="Chat mode"
            value={mode}
            options={MODE_OPTIONS}
            onChange={setMode}
          />
          <div className="font-mono text-[11px] text-[#71717a]">v{pkg.version}</div>
        </div>
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
                    ) : msg.mode === "navigator" ? (
                      <Compass size={12} className="text-indigo-400" />
                    ) : (
                      <div className="h-1.5 w-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    {msg.role === "user" ? (
                      <div className="whitespace-pre-wrap font-medium leading-relaxed text-[#fafafa]">
                        {msg.content}
                      </div>
                    ) : msg.mode === "navigator" ? (
                      msg.agent ? (
                        <NavigatorRun result={msg.agent} />
                      ) : msg.agentError ? (
                        <div className="flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-2.5 py-2">
                          <TriangleAlert
                            size={12}
                            className="mt-0.5 shrink-0 text-rose-400/80"
                          />
                          <div className="min-w-0">
                            <p className="text-[12px] leading-snug text-rose-100/90">
                              {msg.agentError.message}
                            </p>
                            {msg.agentError.detail && (
                              <p className="mt-1 break-words font-mono text-[10px] leading-relaxed text-rose-200/50">
                                {msg.agentError.detail}
                              </p>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 py-0.5 text-[12px] text-[#71717a]">
                          <Loader2 size={12} className="animate-spin text-indigo-400" />
                          Navigating the graph: thought, action, observation…
                        </div>
                      )
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

          {showNavigatorIntro && (
            <div className="msg-enter mt-3 flex items-start gap-2.5 rounded-md border border-dashed border-[#27272a] px-3 py-2.5 text-[12px] leading-relaxed text-[#71717a]">
              <Compass size={13} className="mt-0.5 shrink-0 text-indigo-400/70" />
              <span>
                The Navigator is an agent that answers by <em>walking</em> the
                knowledge graph: search, neighbors, sources, paths, one tool call
                per step. Before each step it is shown the procedural graph around
                its last action (raw guidance, no extra LLM call). Every thought,
                action and observation is shown with the answer.
              </span>
            </div>
          )}

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
              placeholder={
                mode === "navigator"
                  ? "Ask the Navigator: it walks the graph step by step…"
                  : "Query the knowledge graph…"
              }
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
