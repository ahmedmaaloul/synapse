"use client";

import { useState } from "react";
import { Trash2, PanelRightClose, Database } from "lucide-react";
import GraphPanel from "./components/GraphPanel";
import ChatPanel from "./components/ChatPanel";
import FileUpload from "./components/FileUpload";

export default function Home() {
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [clearGraphTrigger, setClearGraphTrigger] = useState(0);

  // UX Enhancement: Right Properties Panel State
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

  return (
    <div className="flex h-screen w-full bg-[#09090b] text-[#fafafa] overflow-hidden font-sans selection:bg-indigo-500/30">

      {/* ── 1. Left Sidebar (Project Controls) ── */}
      <aside className="w-[260px] bg-[#09090b] flex flex-col p-4 flex-shrink-0 z-20 border-r border-[#27272a]">
        <div className="mb-6 flex items-center gap-2.5 px-1 pt-1">
          <div className="w-6 h-6 rounded bg-[#fafafa] flex items-center justify-center font-bold text-[#09090b] shadow-sm">
            <Database size={13} strokeWidth={2.5} />
          </div>
          <h1 className="text-[14px] font-semibold tracking-tight text-[#fafafa]">Synapse</h1>
        </div>

        <div className="flex flex-col gap-6 flex-1 pt-2">
          <div className="flex flex-col gap-2">
            <h3 className="text-[11px] font-semibold text-[#a1a1aa] uppercase tracking-wider px-1">Knowledge Base</h3>
            <FileUpload onUploadComplete={(res) => setUploadResult(res)} />
          </div>

          <div className="flex flex-col gap-2 mt-auto">
            <div className="h-[1px] bg-[#27272a] w-full mb-2"></div>
            <button
              onClick={() => {
                setClearGraphTrigger(prev => prev + 1);
                setSelectedNode(null);
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 text-[13px] text-[#ef4444] hover:bg-[#ef4444]/10 rounded-md transition-colors border border-transparent font-medium group"
            >
              <Trash2 size={14} className="opacity-70 group-hover:opacity-100" />
              Clear Database
            </button>
          </div>
        </div>
      </aside>

      {/* ── 2. Center Content (Focus Area) ── */}
      <main className="flex-1 flex flex-col relative min-w-0 bg-[#09090b]">
        {/* Graph Canvas */}
        <div className="flex-1 overflow-hidden relative border-b border-[#27272a]">
          <GraphPanel
            clearTrigger={clearGraphTrigger}
            onNodeSelected={setSelectedNode}
          />
        </div>

        {/* Integrated Chat Drawer */}
        <div className="h-[40%] min-h-[350px] flex flex-col bg-[#09090b]">
          <ChatPanel uploadResult={uploadResult} />
        </div>
      </main>

      {/* ── 3. Right Properties Panel (Data Inspector) ── */}
      {selectedNode && (
        <aside className="w-[340px] bg-[#09090b] border-l border-[#27272a] flex flex-col p-5 overflow-y-auto z-20">
          <div className="flex justify-between items-center mb-6 pb-4 border-b border-[#27272a]">
            <h2 className="text-[14px] font-semibold text-[#fafafa] tracking-tight">Inspector</h2>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-[#a1a1aa] hover:text-[#fafafa] p-1 rounded-md hover:bg-[#27272a] transition-colors"
              title="Close Panel"
            >
              <PanelRightClose size={16} />
            </button>
          </div>

          <div className="flex flex-col gap-5">
            {/* Core Identity */}
            <div className="flex flex-col gap-1.5">
              <div className="text-[11px] font-semibold text-[#71717a] uppercase tracking-wider">Label</div>
              <div className="text-[14px] text-[#fafafa] font-medium leading-snug">
                {selectedNode.label}
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="text-[11px] font-semibold text-[#71717a] uppercase tracking-wider">Type</div>
              <div className="inline-flex px-2 py-0.5 bg-[#27272a]/50 text-[#e4e4e7] text-[11px] font-medium rounded border border-[#27272a] w-fit">
                {selectedNode.type}
              </div>
            </div>

            {/* Injected Meta Properties */}
            {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
              <div className="mt-2 pt-5 border-t border-[#27272a]">
                <div className="text-[11px] font-semibold text-[#71717a] uppercase tracking-wider mb-3">Properties</div>
                <div className="flex flex-col gap-3">
                  {Object.entries(selectedNode.properties).map(([key, value]) => (
                    <div key={key} className="flex flex-col gap-1">
                      <span className="block text-[#a1a1aa] text-[12px] font-medium capitalize">{key.replace(/_/g, ' ')}</span>
                      <div className="text-[#fafafa] text-[13px] leading-relaxed break-words bg-[#18181b] px-3 py-2.5 rounded-md border border-[#27272a] selection:bg-indigo-500/30">
                        {String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}
