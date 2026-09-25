// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { Check, ExternalLink, Share2 } from "lucide-react";
import { LAB_FAMILY_INFO, LAB_FAMILY_ORDER } from "../lib/constants";
import { shortCitation } from "../lib/lab";
import type { LabArm } from "../lib/types";
import { ArmMark, FLOOR_HATCH, FLOOR_TAG, Tag } from "./LabShared";

interface LabArmPickerProps {
  arms: LabArm[];
  selected: string[];
  onChange: (next: string[]) => void;
}

/** Families in display order; an unknown family from a newer backend goes last. */
function groupByFamily(arms: LabArm[]): { family: string; arms: LabArm[] }[] {
  const groups = new Map<string, LabArm[]>();
  for (const arm of arms) {
    groups.set(arm.family, [...(groups.get(arm.family) ?? []), arm]);
  }
  const rank = (family: string) => {
    const i = LAB_FAMILY_ORDER.indexOf(family);
    return i === -1 ? LAB_FAMILY_ORDER.length : i;
  };
  return Array.from(groups, ([family, list]) => ({ family, arms: list })).sort(
    (a, b) => rank(a.family) - rank(b.family),
  );
}

function CostChip({ calls }: { calls: number }) {
  const free = calls === 0;
  return (
    <Tag
      caps={false}
      title={
        free
          ? "Retrieval itself makes no LLM call: the only model call is the shared reader"
          : "LLM calls this arm makes per question while retrieving, on top of the reader"
      }
      className={
        free
          ? "border-emerald-400/25 bg-emerald-400/[0.07] text-emerald-300/80"
          : "border-amber-400/30 bg-amber-400/10 text-amber-300/90"
      }
    >
      {calls} retrieval LLM {calls === 1 ? "call" : "calls"}
    </Tag>
  );
}

/** The arms, grouped Evidence floors · Passage baselines · Graph arms. */
export default function LabArmPicker({ arms, selected, onChange }: LabArmPickerProps) {
  const chosen = new Set(selected);
  // Emit in catalog order so the request (and its estimate key) is stable.
  const emit = (next: Set<string>) =>
    onChange(arms.map((a) => a.name).filter((name) => next.has(name)));

  const toggle = (name: string) => {
    const next = new Set(chosen);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    emit(next);
  };

  const setFamily = (list: LabArm[], on: boolean) => {
    const next = new Set(chosen);
    for (const arm of list) {
      if (on) next.add(arm.name);
      else next.delete(arm.name);
    }
    emit(next);
  };

  return (
    <div className="flex flex-col gap-3">
      {groupByFamily(arms).map(({ family, arms: list }) => {
        const info = LAB_FAMILY_INFO[family];
        const count = list.filter((a) => chosen.has(a.name)).length;
        const all = count === list.length;
        return (
          <div key={family}>
            <div className="mb-1 flex items-center gap-1.5 px-0.5">
              <span
                title={info?.hint}
                className="text-[10px] font-semibold uppercase tracking-wider text-[#a1a1aa]"
              >
                {info?.title ?? list[0]?.family_title ?? family}
              </span>
              <span className="rounded border border-[#27272a] px-1 font-mono text-[10px] leading-[15px] text-[#52525b]">
                {count}/{list.length}
              </span>
              <button
                onClick={() => setFamily(list, !all)}
                className="ml-auto text-[10px] font-medium text-[#52525b] transition-colors hover:text-[#a1a1aa]"
              >
                {all ? "None" : "All"}
              </button>
            </div>
            <div className="flex flex-col gap-1">
              {list.map((arm) => {
                const on = chosen.has(arm.name);
                return (
                  <div
                    key={arm.name}
                    style={arm.is_null ? FLOOR_HATCH : undefined}
                    className={`rounded-md border px-2.5 py-1.5 transition-colors ${
                      on
                        ? "border-indigo-500/35 bg-indigo-500/[0.05]"
                        : "border-[#27272a] bg-[#18181b]/40 hover:border-[#3f3f46]"
                    } ${arm.is_null ? "border-dashed" : ""}`}
                  >
                    <label className="flex cursor-pointer items-start gap-2">
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() => toggle(arm.name)}
                        className="peer sr-only"
                      />
                      <span
                        aria-hidden="true"
                        className={`mt-[3px] flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-[3px] border transition-colors peer-focus-visible:ring-1 peer-focus-visible:ring-indigo-400 ${
                          on
                            ? "border-indigo-400/70 bg-indigo-500/25 text-indigo-200"
                            : "border-[#3f3f46] bg-[#09090b] text-transparent"
                        }`}
                      >
                        <Check size={9} strokeWidth={3.5} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-1.5">
                          <ArmMark arm={arm.name} family={arm.family} isNull={arm.is_null} />
                          <span
                            className={`truncate text-[12px] font-medium ${
                              on ? "text-[#fafafa]" : "text-[#d4d4d8]"
                            }`}
                          >
                            {arm.title}
                          </span>
                          {arm.is_null && (
                            <Tag
                              className={`ml-auto ${FLOOR_TAG}`}
                              title="Evidence floor: a gain only counts above it"
                            >
                              floor
                            </Tag>
                          )}
                        </span>
                        {arm.description && (
                          <span
                            title={arm.description}
                            className="mt-0.5 block truncate text-[11px] leading-snug text-[#71717a]"
                          >
                            {arm.description}
                          </span>
                        )}
                      </span>
                    </label>
                    <div className="mt-1 flex min-w-0 items-center gap-1 pl-[22px]">
                      <CostChip calls={arm.retrieval_llm_calls} />
                      {arm.needs_graph && (
                        <Tag
                          caps={false}
                          title="Reads the LLM-extracted knowledge graph: its ingest cost is charged to it in the amortized cost-of-pass"
                          className="border-indigo-500/25 bg-indigo-500/[0.07] text-indigo-300/80"
                        >
                          <Share2 size={8} strokeWidth={2.5} />
                          graph
                        </Tag>
                      )}
                      {arm.source.url && (
                        <a
                          href={arm.source.url}
                          target="_blank"
                          rel="noreferrer"
                          title={arm.source.citation || arm.source.url}
                          className="ml-auto flex min-w-0 items-center gap-1 rounded px-0.5 text-[10px] text-[#52525b] transition-colors hover:text-[#a1a1aa]"
                        >
                          <ExternalLink size={9} className="shrink-0" />
                          <span className="truncate">{shortCitation(arm.source.citation)}</span>
                        </a>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
