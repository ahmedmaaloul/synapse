"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronRight, Send, Sparkles, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamChat } from "../lib/api";
import { colorForType } from "../lib/constants";
import type { ChatMessage, Citation, IngestResult } from "../lib/types";

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
  "What are the main skills or topics?",
];

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
    setMessages((prev) => [
      ...prev,
      {
        id: nextId(),
        role: "system",
        content: `Ingested "${ingestResult.filename}" → ${ingestResult.nodes_created} entities, ${ingestResult.relationships_created} relationships. Context loaded.`,
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
      { id: assistantId, role: "assistant", content: "", citations: [] },
    ]);
    setInput("");
    setIsLoading(true);

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    try {
      await streamChat(query, history, (event) => {
        if (event.type === "citations") {
          const cites = event.data as Citation[];
          patch((m) => ({ ...m, citations: cites }));
          onCitations(cites.map((c) => c.name));
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

                    {/* Citations */}
                    {msg.role === "assistant" &&
                      msg.citations &&
                      msg.citations.length > 0 && (
                        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                          <span className="text-[10px] font-medium uppercase tracking-wider text-[#52525b]">
                            Grounded in
                          </span>
                          {msg.citations.slice(0, 8).map((c) => (
                            <button
                              key={c.name}
                              onClick={() => onFocusCitation(c.name)}
                              className="flex items-center gap-1 rounded border border-[#27272a] bg-[#18181b] px-1.5 py-0.5 text-[10px] font-medium text-[#d4d4d8] transition-colors hover:border-indigo-500/50 hover:text-[#fafafa]"
                              title="Center in graph"
                            >
                              <span
                                className="h-1.5 w-1.5 rounded-full"
                                style={{ backgroundColor: colorForType(c.type || "") }}
                              />
                              {c.name}
                            </button>
                          ))}
                        </div>
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
