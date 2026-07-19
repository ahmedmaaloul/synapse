"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Layers, Loader2, RefreshCw } from "lucide-react";
import {
  fetchCommunities,
  rebuildCommunities,
  subscribeCommunityRebuild,
} from "../lib/api";
import type { Community, CommunityStage } from "../lib/types";

interface ThemesPanelProps {
  /** Bumps whenever the graph changed (ingest finished, DB cleared). */
  refreshSignal: number;
  /** Id of the community currently highlighted in the graph. */
  selectedId: string | null;
  /** Null when the user deselects — clears the graph highlight. */
  onSelect: (community: Community | null) => void;
  /** Center a single member node in the graph. */
  onFocusEntity: (name: string) => void;
}

const STAGE_LABEL: Record<CommunityStage, string> = {
  loading_graph: "Loading graph",
  summarizing: "Summarizing clusters",
  embedding_communities: "Embedding themes",
  writing_communities: "Writing themes",
};

interface Progress {
  label: string;
  processed?: number;
  total?: number;
}

/** Members shown before the "+n more" fold. */
const MEMBER_PREVIEW = 10;

export default function ThemesPanel({
  refreshSignal,
  selectedId,
  onSelect,
  onFocusEntity,
}: ThemesPanelProps) {
  const [open, setOpen] = useState(true);
  const [communities, setCommunities] = useState<Community[]>([]);
  const [loading, setLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadedOnce = useRef(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const next = await fetchCommunities(20, signal);
      if (signal?.aborted) return;
      setCommunities(next);
      setError(null);
      loadedOnce.current = true;
    } catch (err) {
      if (signal?.aborted) return;
      console.error("Failed to fetch communities:", err);
      setError("Themes unavailable.");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load, refreshSignal]);

  // A cleared / re-ingested graph invalidates whatever theme was highlighted.
  useEffect(() => {
    onSelect(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshSignal]);

  const rebuild = async () => {
    if (rebuilding) return;
    setRebuilding(true);
    setError(null);
    setProgress({ label: "Starting" });
    try {
      const job = await rebuildCommunities();
      await subscribeCommunityRebuild(job.job_id, (event) => {
        if (event.type === "progress") {
          setProgress({
            label: STAGE_LABEL[event.stage] ?? "Clustering",
            processed: event.processed,
            total: event.total,
          });
        } else if (event.type === "error") {
          setError(event.data);
        }
      });
      await load();
    } catch (err) {
      console.error("Community rebuild failed:", err);
      setError(err instanceof Error ? err.message : "Rebuild failed.");
    } finally {
      setRebuilding(false);
      setProgress(null);
    }
  };

  const toggle = (community: Community) =>
    onSelect(community.id === selectedId ? null : community);

  const largest = communities[0]?.size ?? 1;
  const busy = rebuilding || (loading && !loadedOnce.current);

  return (
    <div className="flex min-h-0 flex-col gap-2">
      {/* ── Section header ── */}
      <div className="flex items-center gap-1.5 px-1">
        <button
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="group flex flex-1 items-center gap-1.5 text-left"
        >
          <ChevronDown
            size={12}
            className={`text-[#52525b] transition-transform duration-200 group-hover:text-[#a1a1aa] ${
              open ? "" : "-rotate-90"
            }`}
          />
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#a1a1aa] transition-colors group-hover:text-[#e4e4e7]">
            Themes
          </h3>
          {communities.length > 0 && (
            <span className="rounded border border-[#27272a] px-1 font-mono text-[10px] leading-[15px] text-[#52525b]">
              {communities.length}
            </span>
          )}
        </button>
        <button
          onClick={rebuild}
          disabled={rebuilding}
          title="Re-detect communities (Louvain + LLM summaries)"
          aria-label="Rebuild themes"
          className="rounded p-1 text-[#52525b] transition-colors hover:bg-[#18181b] hover:text-[#e4e4e7] disabled:opacity-50"
        >
          <RefreshCw size={11} className={rebuilding ? "animate-spin" : ""} />
        </button>
      </div>

      {open && (
        <div className="flex min-h-0 flex-col gap-1.5">
          {/* Rebuild progress */}
          {progress && (
            <div className="flex items-center justify-between gap-2 rounded-md border border-[#27272a] bg-[#18181b] px-2 py-1.5 text-[11px]">
              <span className="truncate text-[#a1a1aa]">{progress.label}</span>
              {progress.total ? (
                <span className="shrink-0 font-mono text-[10px] text-[#71717a]">
                  {progress.processed ?? 0}/{progress.total}
                </span>
              ) : (
                <Loader2 size={11} className="shrink-0 animate-spin text-[#71717a]" />
              )}
            </div>
          )}

          {error && (
            <p className="px-1 text-[11px] leading-snug text-rose-500/90">{error}</p>
          )}

          {/* Empty / loading state */}
          {communities.length === 0 && !error && (
            <div className="flex items-start gap-2 rounded-md border border-dashed border-[#27272a] px-2.5 py-2.5 text-[11px] leading-snug text-[#52525b]">
              <Layers size={12} className="mt-px shrink-0 opacity-70" />
              <span>
                {busy
                  ? "Detecting communities…"
                  : "No themes yet. Ingest a document, or rebuild to cluster the graph."}
              </span>
            </div>
          )}

          {/* ── Theme list ── */}
          {communities.length > 0 && (
            <div className="flex flex-col gap-1 pb-1">
              {communities.map((community) => {
                const active = community.id === selectedId;
                const share = Math.max(
                  8,
                  Math.round((community.size / largest) * 100),
                );
                return (
                  <div
                    key={community.id}
                    className={`overflow-hidden rounded-md border transition-colors ${
                      active
                        ? "border-amber-400/40 bg-amber-400/[0.06]"
                        : "border-[#27272a] bg-[#18181b] hover:border-[#3f3f46]"
                    }`}
                  >
                    <button
                      onClick={() => toggle(community)}
                      aria-expanded={active}
                      className="flex w-full flex-col gap-1.5 px-2.5 py-2 text-left"
                    >
                      <div className="flex items-start gap-2">
                        <span
                          className={`mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full transition-colors ${
                            active
                              ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.7)]"
                              : "bg-[#3f3f46]"
                          }`}
                        />
                        <span
                          className={`min-w-0 flex-1 text-[12px] font-medium leading-snug ${
                            active ? "text-amber-50" : "text-[#e4e4e7]"
                          }`}
                        >
                          {community.title}
                        </span>
                        <span className="mt-px shrink-0 font-mono text-[10px] text-[#52525b]">
                          {community.size}
                        </span>
                      </div>
                      {/* Relative-size bar: cluster weight at a glance. */}
                      <div className="ml-3.5 h-[2px] w-full overflow-hidden rounded-full bg-[#27272a]">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            active ? "bg-amber-400/80" : "bg-indigo-500/60"
                          }`}
                          style={{ width: `${share}%` }}
                        />
                      </div>
                    </button>

                    {/* Expanded detail */}
                    {active && (
                      <div className="border-t border-amber-400/15 px-2.5 py-2">
                        <p className="text-[11px] leading-relaxed text-[#a1a1aa]">
                          {community.summary}
                        </p>
                        {community.members.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {community.members
                              .slice(0, MEMBER_PREVIEW)
                              .map((member) => (
                                <button
                                  key={member}
                                  onClick={() => onFocusEntity(member)}
                                  title="Center in graph"
                                  className="max-w-full truncate rounded border border-[#27272a] bg-[#09090b] px-1.5 py-0.5 text-[10px] font-medium text-[#d4d4d8] transition-colors hover:border-amber-400/50 hover:text-[#fafafa]"
                                >
                                  {member}
                                </button>
                              ))}
                            {community.members.length > MEMBER_PREVIEW && (
                              <span className="px-1 py-0.5 font-mono text-[10px] text-[#52525b]">
                                +{community.members.length - MEMBER_PREVIEW}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
