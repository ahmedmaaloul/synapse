"use client";

import { useRef, useState } from "react";
import { Upload, Loader2, CheckCircle2, AlertCircle } from "lucide-react";

interface FileUploadProps {
    onUploadComplete: (result: any) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [theme, setTheme] = useState("Personal CV / Resume");
    const [isUploading, setIsUploading] = useState(false);
    const [status, setStatus] = useState<{
        type: "success" | "error";
        message: string;
    } | null>(null);

    const handleUpload = async (file: File) => {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            setStatus({ type: "error", message: "Invalid type. PDF required." });
            return;
        }

        setIsUploading(true);
        setStatus(null);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5 min

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("theme", theme);

            const res = await fetch(`${API_URL}/api/upload`, {
                method: "POST",
                body: formData,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Engine failed to process document." }));
                throw new Error(err.detail || "Engine timeout.");
            }

            const result = await res.json();
            setStatus({
                type: "success",
                message: `${result.nodes_created} nodes, ${result.relationships_created} edges active.`,
            });
            onUploadComplete(result);
        } catch (err: any) {
            clearTimeout(timeoutId);
            let message = "API unreachable. Port 8000 blocked?";
            if (err.name === "AbortError") {
                message = "Timeout. Payload exceeded limits.";
            } else if (err.message) {
                message = err.message;
            }
            setStatus({ type: "error", message });
        } finally {
            setIsUploading(false);
            if (inputRef.current) inputRef.current.value = "";
        }
    };

    return (
        <div className="w-full flex flex-col gap-3">
            <input
                ref={inputRef}
                type="file"
                accept=".pdf"
                hidden
                onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleUpload(file);
                }}
            />

            <div className="relative group">
                <select
                    value={theme}
                    onChange={(e) => setTheme(e.target.value)}
                    disabled={isUploading}
                    className="w-full bg-[#09090b] border border-[#27272a] rounded-md text-[13px] text-[#fafafa] font-medium px-3 py-2 outline-none focus:border-indigo-500/50 hover:border-[#3f3f46] transition-colors appearance-none disabled:opacity-50"
                >
                    <option value="Personal CV / Resume">Personal CV / Resume</option>
                    <option value="Technology, Tools & Docs">Technology (Wiki/Docs)</option>
                    <option value="Generic">Generic / Other</option>
                    <option value="Medical/Scientific">Medical / Scientific</option>
                    <option value="Business/Legal">Business / Legal</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-[#71717a] group-hover:text-[#a1a1aa] transition-colors">
                    <svg className="fill-current h-3.5 w-3.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
                </div>
            </div>

            <button
                className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md text-[12px] font-semibold tracking-wide transition-all shadow-sm ${isUploading
                        ? "opacity-50 cursor-not-allowed bg-[#18181b] text-[#71717a] border border-[#27272a]"
                        : "bg-[#fafafa] text-[#09090b] hover:bg-[#e4e4e7] border border-transparent shadow-[0_0_10px_rgba(255,255,255,0.05)]"
                    }`}
                onClick={() => inputRef.current?.click()}
                disabled={isUploading}
            >
                {isUploading ? (
                    <Loader2 size={14} className="animate-spin text-[#71717a]" />
                ) : (
                    <Upload size={14} strokeWidth={2.5} />
                )}
                {isUploading ? "Processsing context..." : "Upload Document"}
            </button>

            {status && (
                <div className={`flex items-center gap-2 text-[11px] mt-1 px-1 font-medium tracking-tight ${status.type === "success" ? "text-emerald-500" : "text-rose-500"}`}>
                    <div className="flex-shrink-0">
                        {status.type === "success" ? <CheckCircle2 size={13} strokeWidth={2.5} /> : <AlertCircle size={13} strokeWidth={2.5} />}
                    </div>
                    <span>{status.message}</span>
                </div>
            )}
        </div>
    );
}
