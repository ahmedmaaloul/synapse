"use client";

import { useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Upload } from "lucide-react";
import { subscribeIngest, uploadDocument } from "../lib/api";
import { INGEST_STAGE_LABEL, THEMES } from "../lib/constants";
import type { IngestResult, Theme } from "../lib/types";

interface FileUploadProps {
  onComplete: (result: IngestResult) => void;
  onIngestingChange: (ingesting: boolean) => void;
}

interface Progress {
  stage: string;
  processed?: number;
  total?: number;
  pct: number;
}

// Rough mapping of pipeline stages onto a 0–100 bar.
function computePct(stage: string, processed?: number, total?: number): number {
  const frac = total && total > 0 ? (processed ?? 0) / total : 0;
  switch (stage) {
    case "extracting":
      return Math.round(frac * 70);
    case "embedding":
      return 75;
    case "writing_nodes":
      return 88;
    case "writing_edges":
      return 96;
    default:
      return 0;
  }
}

export default function FileUpload({
  onComplete,
  onIngestingChange,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [theme, setTheme] = useState<Theme>("Personal CV / Resume");
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [status, setStatus] = useState<
    { type: "success" | "error"; message: string } | null
  >(null);

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus({ type: "error", message: "Invalid type. PDF required." });
      return;
    }

    setIsUploading(true);
    setStatus(null);
    setProgress({ stage: "parsing", pct: 3 });
    onIngestingChange(true);

    try {
      const job = await uploadDocument(file, theme);

      await subscribeIngest(job.job_id, (event) => {
        if (event.type === "progress") {
          setProgress({
            stage: event.stage,
            processed: event.processed,
            total: event.total,
            pct: computePct(event.stage, event.processed, event.total),
          });
        } else if (event.type === "done") {
          const result = event.data as IngestResult;
          setProgress({ stage: "done", pct: 100 });
          setStatus({
            type: "success",
            message: `${result.nodes_created} nodes · ${result.relationships_created} edges`,
          });
          onComplete(result);
        } else if (event.type === "error") {
          setStatus({ type: "error", message: event.data });
        }
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "API unreachable. Is port 8000 up?";
      setStatus({ type: "error", message });
    } finally {
      setIsUploading(false);
      setProgress(null);
      onIngestingChange(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const stageLabel = progress
    ? INGEST_STAGE_LABEL[progress.stage] ??
      (progress.stage === "parsing"
        ? "Parsing PDF"
        : progress.stage === "done"
          ? "Complete"
          : "Processing")
    : "";

  return (
    <div className="flex w-full flex-col gap-3">
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

      <div className="group relative">
        <select
          value={theme}
          onChange={(e) => setTheme(e.target.value as Theme)}
          disabled={isUploading}
          aria-label="Document theme"
          className="w-full appearance-none rounded-md border border-[#27272a] bg-[#09090b] px-3 py-2 text-[13px] font-medium text-[#fafafa] outline-none transition-colors hover:border-[#3f3f46] focus:border-indigo-500/50 disabled:opacity-50"
        >
          {THEMES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-[#71717a] transition-colors group-hover:text-[#a1a1aa]">
          <svg className="h-3.5 w-3.5 fill-current" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
          </svg>
        </div>
      </div>

      <button
        className={`flex w-full items-center justify-center gap-2 rounded-md px-4 py-2 text-[12px] font-semibold tracking-wide shadow-sm transition-all ${
          isUploading
            ? "cursor-not-allowed border border-[#27272a] bg-[#18181b] text-[#71717a]"
            : "border border-transparent bg-[#fafafa] text-[#09090b] hover:bg-[#e4e4e7]"
        }`}
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
      >
        {isUploading ? (
          <Loader2 size={14} className="animate-spin text-[#71717a]" />
        ) : (
          <Upload size={14} strokeWidth={2.5} />
        )}
        {isUploading ? "Processing…" : "Upload Document"}
      </button>

      {/* Live progress */}
      {progress && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-[#a1a1aa]">
            <span className="font-medium">{stageLabel}</span>
            <span className="font-mono text-[#71717a]">
              {progress.stage === "extracting" && progress.total
                ? `${progress.processed ?? 0}/${progress.total}`
                : `${progress.pct}%`}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#18181b]">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${progress.pct}%` }}
            />
          </div>
        </div>
      )}

      {status && (
        <div
          className={`mt-1 flex items-center gap-2 px-1 text-[11px] font-medium tracking-tight ${
            status.type === "success" ? "text-emerald-500" : "text-rose-500"
          }`}
        >
          <span className="shrink-0">
            {status.type === "success" ? (
              <CheckCircle2 size={13} strokeWidth={2.5} />
            ) : (
              <AlertCircle size={13} strokeWidth={2.5} />
            )}
          </span>
          <span>{status.message}</span>
        </div>
      )}
    </div>
  );
}
