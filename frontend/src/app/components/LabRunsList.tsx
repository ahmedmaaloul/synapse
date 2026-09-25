// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { ChevronRight, History } from "lucide-react";
import { LAB_ACTIVE_STATUSES } from "../lib/constants";
import { budgetLabel, formatWhen, sortBudgets } from "../lib/lab";
import type { LabRunSummary } from "../lib/types";
import { StatusChip } from "./LabShared";

interface LabRunsListProps {
  runs: LabRunSummary[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
  error: string | null;
}

const MODE_LABEL: Record<string, string> = {
  retrieve: "retrieve · $0",
  realtime: "realtime",
  batch: "batch",
};

/** Every stored run, newest first; a click opens its leaderboard. */
export default function LabRunsList({ runs, selectedId, onSelect, error }: LabRunsListProps) {
  if (error) {
    return (
      <p className="rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-3 py-2 text-[12px] text-rose-100/80">
        {error}
      </p>
    );
  }
  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 px-6 py-10 text-center text-[#71717a]">
        <History size={26} strokeWidth={1.5} className="opacity-40" />
        <p className="text-[13px] font-medium">No runs yet.</p>
        <p className="max-w-[360px] text-[12px] leading-relaxed text-[#52525b]">
          Estimate, then run. Runs are stored under{" "}
          <span className="font-mono text-[#a1a1aa]">backend/lab_runs/</span> with their manifest,
          contexts and leaderboard, and can be re-scored without a model.
        </p>
      </div>
    );
  }
  return (
    <ul className="flex flex-col divide-y divide-[#27272a]/60 overflow-hidden rounded-md border border-[#27272a]">
      {runs.map((run) => {
        const active = run.run_id === selectedId;
        const budgets = sortBudgets(run.budgets);
        return (
          <li key={run.run_id}>
            <button
              onClick={() => onSelect(run.run_id)}
              aria-current={active}
              className={`group flex w-full items-center gap-3 px-3 py-2 text-left transition-colors ${
                active ? "bg-indigo-500/[0.07]" : "hover:bg-[#18181b]"
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span
                    className={`truncate font-mono text-[11px] font-medium ${
                      active ? "text-indigo-200" : "text-[#e4e4e7]"
                    }`}
                  >
                    {run.run_id}
                  </span>
                  <StatusChip
                    status={run.status}
                    stalled={run.active === false && LAB_ACTIVE_STATUSES.has(run.status)}
                  />
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 font-mono text-[10px] text-[#71717a]">
                  <span className="text-[#a1a1aa]">{run.dataset ?? "—"}</span>
                  {run.n != null && <span>n={run.n}</span>}
                  <span className="text-[#3f3f46]">·</span>
                  <span>{MODE_LABEL[run.mode ?? ""] ?? run.mode ?? "—"}</span>
                  {run.mode !== "retrieve" && run.reader_model && (
                    <>
                      <span className="text-[#3f3f46]">·</span>
                      <span>{run.reader_model}</span>
                    </>
                  )}
                  <span className="text-[#3f3f46]">·</span>
                  <span title={run.arms.join(", ")}>
                    {run.arms.length} {run.arms.length === 1 ? "arm" : "arms"}
                  </span>
                  {budgets.length > 0 && (
                    <>
                      <span className="text-[#3f3f46]">×</span>
                      <span>{budgets.map(budgetLabel).join("/")}</span>
                    </>
                  )}
                </div>
              </div>
              <span className="shrink-0 font-mono text-[10px] text-[#52525b]">
                {formatWhen(run.created_at)}
              </span>
              <ChevronRight
                size={12}
                className="shrink-0 text-[#3f3f46] transition-colors group-hover:text-[#a1a1aa]"
              />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
