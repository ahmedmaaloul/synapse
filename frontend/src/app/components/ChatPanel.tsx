"use client";

import { useState, useRef, useEffect } from "react";
import { Send, User, ChevronRight } from "lucide-react";

interface ChatMessage {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ChatPanelProps {
    uploadResult?: {
        filename: string;
        nodes_created: number;
        relationships_created: number;
    } | null;
}

export default function ChatPanel({ uploadResult }: ChatPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: "welcome",
            role: "system",
            content: "Synapse Engine Ready. Awaiting document ingestion.",
        },
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // ── Auto-scroll to bottom ──
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Listen for external uploads
    useEffect(() => {
        if (uploadResult) {
            const sysMsg: ChatMessage = {
                id: `sys-${Date.now()}`,
                role: "system",
                content: `Ingestion complete. Extracted [${uploadResult.nodes_created}] entities, [${uploadResult.relationships_created}] links. Context loaded.`,
            };
            setMessages((prev) => [...prev, sysMsg]);
        }
    }, [uploadResult]);

    // ── Handle send ──
    const handleSend = async () => {
        const query = input.trim();
        if (!query || isLoading) return;

        const userMsg: ChatMessage = {
            id: `user-${Date.now()}`,
            role: "user",
            content: query,
        };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        const assistantId = `assistant-${Date.now()}`;
        setMessages((prev) => [
            ...prev,
            { id: assistantId, role: "assistant", content: "" },
        ]);

        try {
            const res = await fetch(`${API_URL}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            });

            if (!res.ok) throw new Error("Chat request failed");

            const reader = res.body?.getReader();
            const decoder = new TextDecoder();

            if (reader) {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantId
                                ? { ...msg, content: msg.content + chunk }
                                : msg
                        )
                    );
                }
            }
        } catch (err) {
            console.error("Chat error:", err);
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === assistantId
                        ? {
                            ...msg,
                            content: "Error: Connection refused. Check backend status.",
                        }
                        : msg
                )
            );
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#09090b] w-full relative pt-2">
            {isLoading && (
                <div className="loading-bar">
                    <div className="loading-indicator"></div>
                </div>
            )}

            {/* ── Chat Header ── */}
            <div className="px-5 py-2.5 flex items-center justify-between border-b border-[#27272a]">
                <h3 className="text-[12px] font-semibold text-[#e4e4e7] uppercase tracking-wider">Terminal Workspace</h3>
                <div className="text-[11px] font-mono text-[#71717a]">v0.2.1</div>
            </div>

            {/* ── Messages Area ── */}
            <div className="flex-1 overflow-y-auto w-full px-5 py-4 font-sans text-[13px]">
                <div className="max-w-full flex flex-col w-full pb-20">
                    {messages.map((msg) => (
                        <div key={msg.id} className="w-full flex flex-col py-3 border-b border-[#27272a]/50 last:border-0 group">
                            {msg.role === "system" ? (
                                <div className="text-[12px] text-[#71717a] py-1 font-mono tracking-tight flex items-center gap-2">
                                    <ChevronRight size={12} />
                                    {msg.content}
                                </div>
                            ) : (
                                <div className="flex gap-3">
                                    {/* Minimal Avatar Grid */}
                                    <div className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0 mt-0.5 border border-[#27272a] bg-[#18181b]">
                                        {msg.role === "user" ? (
                                            <User size={12} className="text-[#a1a1aa]" />
                                        ) : (
                                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
                                        )}
                                    </div>

                                    {/* Text Content */}
                                    <div className={`flex-1 text-[13px] leading-relaxed tracking-tight ${msg.role === "user"
                                            ? "text-[#fafafa]"
                                            : "text-[#d4d4d8]"
                                        }`}>
                                        <div className="whitespace-pre-wrap">{msg.content}</div>
                                        {msg.role === "assistant" && !msg.content && isLoading && (
                                            <span className="inline-block w-1.5 h-3 ml-1 bg-indigo-500 animate-pulse mt-1" />
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* ── Prompt Input Area (Linear Style) ── */}
            <div className="absolute bottom-0 w-full bg-[#09090b] pt-2 pb-4 border-t border-[#27272a]">
                <div className="max-w-full mx-auto px-4">
                    <div className="relative flex items-center w-full bg-[#18181b] border border-[#27272a] rounded-md shadow-sm transition-colors focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/20">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSend()}
                            placeholder="Query the engine..."
                            disabled={isLoading}
                            className="bg-transparent text-[#fafafa] placeholder-[#71717a] border-0 focus:ring-0 outline-none w-full py-2.5 pl-3 pr-10 text-[13px] rounded-md font-medium"
                        />
                        <button
                            className="absolute right-1.5 p-1.5 rounded text-[#a1a1aa] hover:text-[#fafafa] hover:bg-[#27272a] disabled:opacity-50 transition-colors"
                            onClick={handleSend}
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
