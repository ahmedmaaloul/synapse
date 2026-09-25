// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { useEffect, useState } from "react";
import { Check, Loader2, Minus, X } from "lucide-react";
import { ApiError, fetchLabRun } from "../lib/api";
import { LAB_ROWS_PAGE } from "../lib/constants";
import { budgetKey, budgetLabel, formatPts, formatTokens } from "../lib/lab";
import type { LabBudget, LabQuestionRow } from "../lib/types";

interface LabQuestionsProps {
  runId: string;
  arm: string;
  armTitle: string;
  budget: LabBudget;
  read: boolean;
  onClose: () => void;
}

interface Page {
  rows: LabQuestionRow[];
  total: number;
}

/** Rows of this one cell (the router may ignore the filter: filter here too). */
function ofCell(rows: LabQuestionRow[], arm: string, budget: LabBudget): LabQuestionRow[] {
  return rows.filter((r) => r.arm === arm && budgetKey(r.budget) === budgetKey(budget));
}

/** The per-question rows behind one leaderboard cell: what the arm got right. */
export default function LabQuestions({
  runId,
  arm,
  armTitle,
  budget,
  read,
  onClose,
}: LabQuestionsProps) {
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    fetchLabRun(runId, { arm, budget, offset, limit: LAB_ROWS_PAGE }, controller.signal)
      .then((detail) => {
        const rows = ofCell(detail.rows?.rows ?? [], arm, budget);
        setPage((prev) => ({
          rows: offset === 0 ? rows : [...(prev?.rows ?? []), ...rows],
          total: detail.rows?.total ?? rows.length,
        }));
        setError(null);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiError && err.detail ? err.detail : "Could not load the questions.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [runId, arm, budget, offset]);

  const rows = page?.rows ?? [];
  const more = page != null && offset + LAB_ROWS_PAGE < page.total;

  return (
    <div className="overflow-hidden rounded-md border border-[#27272a]">
      <div className="flex items-center gap-2 border-b border-[#27272a] bg-[#18181b]/60 px-3 py-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
          Questions
        </span>
        <span className="truncate text-[11px] font-medium text-[#e4e4e7]">{armTitle}</span>
        <span className="font-mono text-[10px] text-[#71717a]">@ {budgetLabel(budget)}</span>
        {loading && <Loader2 size={11} className="animate-spin text-[#71717a]" />}
        <button
          onClick={onClose}
          className="ml-auto rounded-md p-0.5 text-[#71717a] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
          title="Close"
          aria-label="Close questions"
        >
          <X size={12} />
        </button>
      </div>
      {error ? (
        <p className="px-3 py-2 text-[11px] text-rose-300/80">{error}</p>
      ) : rows.length === 0 && !loading ? (
        <p className="px-3 py-2 text-[11px] text-[#52525b]">No rows for this cell.</p>
      ) : (
        <ul className="flex max-h-[320px] flex-col divide-y divide-[#27272a]/60 overflow-y-auto">
          {rows.map((r) => {
            const hit = read ? (r.em ?? 0) >= 1 : r.containment === true;
            const partial = read && !hit && (r.f1 ?? 0) > 0;
            return (
              <li key={r.qid} className="flex gap-2.5 px-3 py-2">
                <span
                  title={
                    read
                      ? hit
                        ? "Exact match"
                        : partial
                          ? "Partial (F1 > 0)"
                          : "Wrong"
                      : hit
                        ? "The gold answer is in the context"
                        : "The gold answer is not in the context"
                  }
                  className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                    hit
                      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                      : partial
                        ? "border-amber-400/30 bg-amber-400/10 text-amber-300"
                        : "border-[#3f3f46] bg-[#18181b] text-[#71717a]"
                  }`}
                >
                  {hit ? <Check size={10} strokeWidth={3} /> : partial ? <Minus size={10} /> : <X size={10} />}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-[12px] leading-snug text-[#d4d4d8]">{r.question}</p>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10px] text-[#71717a]">
                    <span>
                      gold <span className="text-[#a1a1aa]">{(r.gold ?? []).join(" | ") || "—"}</span>
                    </span>
                    {read && (
                      <span>
                        answer{" "}
                        <span className={r.read_error ? "text-rose-300/80" : "text-[#e4e4e7]"}>
                          {r.read_error ? `not answered (${r.read_error})` : (r.answer ?? "—")}
                        </span>
                      </span>
                    )}
                    {read && !r.read_error && <span>F1 {formatPts((r.f1 ?? 0) * 100, 0)}</span>}
                    <span>{formatTokens(r.context_tokens)} ctx tok</span>
                    {r.recall_permissive != null && (
                      <span>
                        recall {formatPts((r.recall_strict ?? 0) * 100, 0)}/
                        {formatPts(r.recall_permissive * 100, 0)}
                      </span>
                    )}
                    {r.retrieval_error && (
                      <span className="text-amber-300/80" title={r.retrieval_error}>
                        retrieval error
                      </span>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {more && !error && (
        <button
          onClick={() => {
            setLoading(true);
            setOffset((o) => o + LAB_ROWS_PAGE);
          }}
          disabled={loading}
          className="w-full border-t border-[#27272a] px-3 py-1.5 text-[11px] font-medium text-[#71717a] transition-colors hover:bg-[#18181b] hover:text-[#fafafa] disabled:opacity-50"
        >
          Load {Math.min(LAB_ROWS_PAGE, (page?.total ?? 0) - offset - LAB_ROWS_PAGE)} more of{" "}
          {page?.total}
        </button>
      )}
    </div>
  );
}
