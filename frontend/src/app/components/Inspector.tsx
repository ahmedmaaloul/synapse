"use client";

import { Crosshair, PanelRightClose } from "lucide-react";
import { colorForType } from "../lib/constants";
import type { GraphNode } from "../lib/types";

interface InspectorProps {
  node: GraphNode;
  onClose: () => void;
  onFocus: (node: GraphNode) => void;
}

const HIDDEN_KEYS = new Set(["name", "type", "embedding"]);

export default function Inspector({ node, onClose, onFocus }: InspectorProps) {
  const props = Object.entries(node.properties || {}).filter(
    ([key]) => !HIDDEN_KEYS.has(key),
  );

  return (
    <aside className="z-20 flex w-[340px] flex-shrink-0 flex-col overflow-y-auto border-l border-[#27272a] bg-[#09090b] p-5">
      <div className="mb-6 flex items-center justify-between border-b border-[#27272a] pb-4">
        <h2 className="text-[14px] font-semibold tracking-tight text-[#fafafa]">
          Inspector
        </h2>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onFocus(node)}
            className="rounded-md p-1 text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
            title="Center in graph"
          >
            <Crosshair size={15} />
          </button>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
            title="Close panel"
          >
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
            Label
          </div>
          <div className="flex items-center gap-2 text-[14px] font-medium leading-snug text-[#fafafa]">
            <span
              className="h-2.5 w-2.5 flex-shrink-0 rounded-full"
              style={{ backgroundColor: colorForType(node.type) }}
            />
            {node.label}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
            Type
          </div>
          <div
            className="inline-flex w-fit rounded border px-2 py-0.5 text-[11px] font-medium"
            style={{
              color: colorForType(node.type),
              borderColor: `${colorForType(node.type)}40`,
              backgroundColor: `${colorForType(node.type)}12`,
            }}
          >
            {node.type}
          </div>
        </div>

        {props.length > 0 && (
          <div className="mt-2 border-t border-[#27272a] pt-5">
            <div className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
              Properties
            </div>
            <div className="flex flex-col gap-3">
              {props.map(([key, value]) => (
                <div key={key} className="flex flex-col gap-1">
                  <span className="text-[12px] font-medium capitalize text-[#a1a1aa]">
                    {key.replace(/_/g, " ")}
                  </span>
                  <div className="break-words rounded-md border border-[#27272a] bg-[#18181b] px-3 py-2.5 text-[13px] leading-relaxed text-[#fafafa]">
                    {String(value)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
