"use client";

import { useCallback, useRef, useState } from "react";
import { Database, Github, Trash2 } from "lucide-react";
import ChatPanel from "./components/ChatPanel";
import FileUpload from "./components/FileUpload";
import GraphPanel from "./components/GraphPanel";
import Inspector from "./components/Inspector";
import ThemesPanel from "./components/ThemesPanel";
import type { Community, GraphNode, IngestResult } from "./lib/types";

export default function Home() {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [highlightedNames, setHighlightedNames] = useState<string[]>([]);
  const [activeCommunity, setActiveCommunity] = useState<Community | null>(null);
  const [refreshSignal, setRefreshSignal] = useState(0);
  const [clearSignal, setClearSignal] = useState(0);
  // Themes get their own counter: it must only advance once the graph has
  // *actually* changed on the server, never when a mutation is merely
  // requested (see `bumpThemes`).
  const [themesVersion, setThemesVersion] = useState(0);
  const [focusTarget, setFocusTarget] = useState<{ name: string; nonce: number } | null>(
    null,
  );
  const focusNonce = useRef(0);

  const bumpThemes = useCallback(() => setThemesVersion((n) => n + 1), []);

  const handleComplete = useCallback(
    (result: IngestResult) => {
      setIngestResult(result);
      setRefreshSignal((n) => n + 1);
      bumpThemes();
    },
    [bumpThemes],
  );

  const focusNode = useCallback((name: string) => {
    focusNonce.current += 1;
    setFocusTarget({ name, nonce: focusNonce.current });
  }, []);

  // Selecting a theme takes over the graph highlight; chat citations own it
  // otherwise, so the two never fight over the same visual channel.
  const selectCommunity = useCallback((community: Community | null) => {
    setActiveCommunity(community);
    if (community) setHighlightedNames([]);
  }, []);

  const showCitations = useCallback((names: string[]) => {
    setHighlightedNames(names);
    if (names.length > 0) setActiveCommunity(null);
  }, []);

  const clearDatabase = () => {
    setClearSignal((n) => n + 1);
    setSelectedNode(null);
    setHighlightedNames([]);
    setActiveCommunity(null);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-[#09090b] font-sans text-[#fafafa] selection:bg-indigo-500/30">
      {/* ── Left sidebar ── */}
      <aside className="z-20 flex w-[260px] flex-shrink-0 flex-col border-r border-[#27272a] bg-[#09090b] p-4">
        <div className="mb-6 flex items-center gap-2.5 px-1 pt-1">
          <div className="flex h-6 w-6 items-center justify-center rounded bg-gradient-to-br from-indigo-400 to-indigo-600 font-bold text-white shadow-sm">
            <Database size={13} strokeWidth={2.5} />
          </div>
          <h1 className="text-[14px] font-semibold tracking-tight text-[#fafafa]">
            Synapse
          </h1>
          <span className="ml-auto rounded border border-[#27272a] px-1.5 py-0.5 font-mono text-[10px] text-[#71717a]">
            GraphRAG
          </span>
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-6 pt-2">
          <div className="flex flex-col gap-2">
            <h3 className="px-1 text-[11px] font-semibold uppercase tracking-wider text-[#a1a1aa]">
              Knowledge Base
            </h3>
            <FileUpload
              onComplete={handleComplete}
              onIngestingChange={setIngesting}
            />
          </div>

          <div className="-mr-1 min-h-0 flex-1 overflow-y-auto pr-1">
            <ThemesPanel
              refreshSignal={themesVersion}
              selectedId={activeCommunity?.id ?? null}
              onSelect={selectCommunity}
              onFocusEntity={focusNode}
            />
          </div>

          <div className="flex flex-col gap-2">
            <a
              href="https://github.com/ahmedmaaloul/synapse"
              target="_blank"
              rel="noreferrer"
              className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-[13px] font-medium text-[#a1a1aa] transition-colors hover:bg-[#18181b] hover:text-[#fafafa]"
            >
              <Github size={14} className="opacity-70" />
              View source
            </a>
            <div className="-mt-1 flex items-center justify-between gap-2 px-3">
              <a
                href="https://github.com/ahmedmaaloul"
                target="_blank"
                rel="noreferrer"
                className="group text-[11px] text-[#52525b] transition-colors hover:text-[#d4d4d8]"
              >
                Built by{" "}
                <span className="font-medium text-[#71717a] transition-colors group-hover:text-[#fafafa]">
                  Ahmed Maaloul
                </span>
              </a>
              <a
                href="https://github.com/ahmedmaaloul/synapse/blob/main/LICENSE"
                target="_blank"
                rel="noreferrer"
                title="Licensed under the GNU AGPL v3.0 or later"
                className="rounded border border-[#27272a] px-1.5 py-0.5 font-mono text-[10px] text-[#52525b] transition-colors hover:border-indigo-500/40 hover:text-[#a1a1aa]"
              >
                AGPL-3.0
              </a>
            </div>
            <div className="mb-1 h-px w-full bg-[#27272a]" />
            <button
              onClick={clearDatabase}
              className="group flex w-full items-center gap-2.5 rounded-md border border-transparent px-3 py-2 text-[13px] font-medium text-[#ef4444] transition-colors hover:bg-[#ef4444]/10"
            >
              <Trash2 size={14} className="opacity-70 group-hover:opacity-100" />
              Clear Database
            </button>
          </div>
        </div>
      </aside>

      {/* ── Center ── */}
      <main className="relative flex min-w-0 flex-1 flex-col bg-[#09090b]">
        <div className="relative flex-1 overflow-hidden border-b border-[#27272a]">
          <GraphPanel
            ingesting={ingesting}
            refreshSignal={refreshSignal}
            clearSignal={clearSignal}
            highlightedNames={highlightedNames}
            communityNames={activeCommunity?.members}
            communityTitle={activeCommunity?.title ?? null}
            focusTarget={focusTarget}
            onNodeSelected={setSelectedNode}
            // Fires only after the DELETE resolves — refetching any earlier
            // races the wipe and re-seats the themes that were just deleted.
            onClearComplete={bumpThemes}
          />
        </div>

        <div className="flex h-[40%] min-h-[350px] flex-col bg-[#09090b]">
          <ChatPanel
            ingestResult={ingestResult}
            onCitations={showCitations}
            onFocusCitation={focusNode}
          />
        </div>
      </main>

      {/* ── Right inspector ── */}
      {selectedNode && (
        <Inspector
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
          onFocus={(n) => focusNode(n.label)}
        />
      )}
    </div>
  );
}
