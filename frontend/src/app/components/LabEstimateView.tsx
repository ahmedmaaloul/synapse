// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { Fragment, useState } from "react";
import { ChevronDown, CircleCheck, Info, ShieldCheck, TriangleAlert } from "lucide-react";
import { LAB_PRICING_URL } from "../lib/constants";
import { budgetLabel, formatTokens, formatUsd } from "../lib/lab";
import type { LabArm, LabCellEstimate, LabEstimate } from "../lib/types";
import { ArmMark, FLOOR_HATCH } from "./LabShared";

interface LabEstimateViewProps {
  estimate: LabEstimate;
  /** The configuration changed since this estimate was made. */
  stale: boolean;
  arms: Map<string, LabArm>;
}

const PHASE_LABEL: Record<string, string> = {
  ingest: "Ingest (extraction)",
  retrieval_llm: "Retrieval LLM calls",
  reader: "Reader",
};

function Stat({
  label,
  value,
  hint,
  tone = "text-[#fafafa]",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: string;
}) {
  return (
    <div
      title={hint}
      className="min-w-0 rounded-md border border-[#27272a] bg-[#18181b]/60 px-3 py-2"
    >
      <div className="truncate text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
        {label}
      </div>
      <div className={`mt-0.5 truncate font-mono text-[15px] font-medium tabular-nums ${tone}`}>
        {value}
      </div>
    </div>
  );
}

/**
 * Point and upper bound against the cap on one track. The cap is a hard line:
 * the upper bound has to land left of it for the run to be allowed.
 */
function CapMeter({
  point,
  upper,
  cap,
}: {
  point: number | null;
  upper: number | null;
  cap: number | null;
}) {
  if (upper == null || cap == null || cap <= 0) return null;
  const scale = Math.max(upper, cap) * 1.08 || 1;
  const pct = (v: number) => `${Math.min(100, (v / scale) * 100)}%`;
  const over = upper > cap;
  return (
    <div className="mt-3">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-[#18181b]">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${
            over ? "bg-rose-500/25" : "bg-indigo-500/25"
          }`}
          style={{ width: pct(upper) }}
        />
        {point != null && (
          <div
            className={`absolute inset-y-0 left-0 rounded-full ${
              over ? "bg-rose-400/80" : "bg-indigo-400/90"
            }`}
            style={{ width: pct(point) }}
          />
        )}
        <div
          className="absolute inset-y-0 w-[2px] bg-[#fafafa]"
          style={{ left: `calc(${pct(cap)} - 1px)` }}
        />
      </div>
      <div className="mt-1 flex items-center gap-3 font-mono text-[10px] text-[#52525b]">
        <span className="flex items-center gap-1">
          <span className={`h-1.5 w-2.5 rounded-sm ${over ? "bg-rose-400/80" : "bg-indigo-400/90"}`} />
          point {formatUsd(point)}
        </span>
        <span className="flex items-center gap-1">
          <span className={`h-1.5 w-2.5 rounded-sm ${over ? "bg-rose-500/25" : "bg-indigo-500/25"}`} />
          upper {formatUsd(upper)}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-[2px] bg-[#fafafa]" />
          cap {formatUsd(cap)}
        </span>
      </div>
    </div>
  );
}

function tokens(cell: { prompt_tokens: number; completion_tokens: number }): number {
  return (cell.prompt_tokens ?? 0) + (cell.completion_tokens ?? 0);
}

function upperTokens(cell: {
  upper_prompt_tokens: number;
  upper_completion_tokens: number;
}): number {
  return (cell.upper_prompt_tokens ?? 0) + (cell.upper_completion_tokens ?? 0);
}

/** The estimate: per arm × budget calls, tokens, point and upper $, and the gate. */
export default function LabEstimateView({ estimate, stale, arms }: LabEstimateViewProps) {
  const [showAssumptions, setShowAssumptions] = useState(false);
  const free = estimate.mode === "retrieve";
  const cap = estimate.max_usd;
  const reader = estimate.phases.find((p) => p.phase === "reader");
  const otherPhases = estimate.phases.filter(
    (p) => p.phase !== "reader" && (p.calls > 0 || (p.upper_usd ?? 0) > 0),
  );
  const readerCalls = reader?.calls ?? estimate.cells.reduce((s, c) => s + c.calls, 0);

  // Cells grouped by arm, in the order the backend listed them.
  const byArm = new Map<string, LabCellEstimate[]>();
  for (const cell of estimate.cells) {
    byArm.set(cell.arm, [...(byArm.get(cell.arm) ?? []), cell]);
  }
  const cellsPerArm = byArm.size ? Math.round(estimate.cells.length / byArm.size) : 0;

  const dates = Object.entries(estimate.price_dates ?? {});
  const assumptions = estimate.assumptions ?? [];
  const pricingUrl = estimate.pricing_url || LAB_PRICING_URL;

  return (
    <div className="flex flex-col gap-3">
      {/* Gate */}
      {estimate.refuse ? (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-md border border-rose-500/40 bg-rose-500/[0.09] px-3 py-2.5"
        >
          <TriangleAlert size={14} className="mt-0.5 shrink-0 text-rose-400" />
          <div className="min-w-0">
            <p className="text-[12px] font-semibold text-rose-200">
              Refused: the upper bound is over your cap.
            </p>
            {estimate.refuse_reason && (
              <p className="mt-0.5 break-words font-mono text-[11px] leading-relaxed text-rose-200/70">
                {estimate.refuse_reason}
              </p>
            )}
            <p className="mt-1 text-[11px] leading-relaxed text-rose-100/60">
              Lower n, drop budgets or arms, switch to Batch (half price), or raise the cap. Run
              retrieve-only first ($0): a paid run re-checks the cap on the measured contexts.
            </p>
          </div>
        </div>
      ) : free ? (
        <div className="flex items-start gap-2.5 rounded-md border border-emerald-400/25 bg-emerald-400/[0.06] px-3 py-2.5">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-emerald-400/90" />
          <p className="text-[12px] leading-relaxed text-emerald-100/80">
            <span className="font-semibold text-emerald-200">$0.</span> Retrieve-only never
            calls a model: every arm retrieves and packs its context at each budget, then the
            leaderboard scores context size, units by kind and whether the gold answer is in the
            context, next to the floors.
          </p>
        </div>
      ) : (
        <div className="flex items-start gap-2.5 rounded-md border border-emerald-400/25 bg-emerald-400/[0.06] px-3 py-2.5">
          <CircleCheck size={14} className="mt-0.5 shrink-0 text-emerald-400/90" />
          <p className="text-[12px] leading-relaxed text-emerald-100/80">
            Upper bound <span className="font-mono">{formatUsd(estimate.total_upper_usd)}</span>
            {cap != null ? (
              <>
                {" "}
                fits the cap <span className="font-mono">{formatUsd(cap)}</span>.
              </>
            ) : (
              " (no cap set)."
            )}{" "}
            {estimate.batch
              ? "Batch reads are billed at half price when OpenAI completes them; the cap is re-checked on the measured contexts first."
              : "Realtime reads stay metered against the cap and stop cleanly before exceeding it."}
          </p>
        </div>
      )}

      {!free && estimate.measured_from && (
        <div className="flex items-start gap-2 rounded-md border border-[#27272a] bg-[#18181b]/60 px-3 py-2 text-[11px] leading-relaxed text-[#a1a1aa]">
          <Info size={12} className="mt-0.5 shrink-0 text-indigo-400/80" />
          <span>
            Priced on the contexts measured by{" "}
            <span className="font-mono text-[#e4e4e7]">{estimate.measured_from.run_id}</span>:{" "}
            {estimate.measured_from.cells.length} cells measured
            {estimate.measured_from.unmeasured.length > 0
              ? `, ${estimate.measured_from.unmeasured.length} still bounded by their budget`
              : ""}
            . Cells marked <span className="text-emerald-400/80">m</span> use the real sizes.
          </span>
        </div>
      )}

      {stale && (
        <div className="flex items-center gap-2 rounded-md border border-amber-400/30 bg-amber-400/[0.07] px-3 py-2 text-[11px] text-amber-100/80">
          <Info size={12} className="shrink-0 text-amber-400" />
          The configuration changed since this estimate. Estimate again to run.
        </div>
      )}

      {/* Headline numbers */}
      {free ? (
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          <Stat label="Questions" value={estimate.n_questions.toLocaleString()} />
          <Stat label="Arms × budgets" value={`${byArm.size} × ${cellsPerArm}`} />
          <Stat
            label="Contexts packed"
            value={(estimate.cells.length * estimate.n_questions).toLocaleString()}
            hint="Every (arm, budget, question) context is stored with its sha256, so the run can be re-scored without Neo4j or a model"
          />
          <Stat label="Model calls · cost" value="0 · $0" tone="text-emerald-300" />
        </div>
      ) : (
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="Questions" value={estimate.n_questions.toLocaleString()} />
        <Stat
          label="Reader calls"
          value={readerCalls.toLocaleString()}
          hint="One per (arm, budget, question); identical requests are counted once (N0 reads the same empty context at every budget)"
        />
        <Stat
          label="Point estimate"
          value={formatUsd(estimate.total_point_usd)}
          hint="Real tokenizer on the rendered prompts; context at the budget; answer at the assumed length"
        />
        <Stat
          label="Upper bound"
          value={formatUsd(estimate.total_upper_usd)}
          tone={estimate.refuse ? "text-rose-300" : "text-[#fafafa]"}
          hint="Output at the request cap (incl. the reasoning allowance), 0% cache hits, +10% tokenizer margin"
        />
      </div>
      )}
      {!free && (
        <CapMeter point={estimate.total_point_usd} upper={estimate.total_upper_usd} cap={cap} />
      )}

      {/* Per arm × budget (a table of zeros says nothing in the free tier) */}
      {!free && (
      <div className="overflow-x-auto rounded-md border border-[#27272a]">
        <table className="w-full min-w-[560px] border-collapse text-[11px]">
          <thead>
            <tr className="border-b border-[#27272a] bg-[#18181b]/60 text-left text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
              <th className="px-2.5 py-1.5 font-semibold">Arm</th>
              <th className="px-2 py-1.5 font-semibold">Budget</th>
              <th className="px-2 py-1.5 text-right font-semibold">Calls</th>
              <th
                className="px-2 py-1.5 text-right font-semibold"
                title="Context tokens per question: the budget (or measured, when marked)"
              >
                Ctx / q
              </th>
              <th className="px-2 py-1.5 text-right font-semibold" title="Prompt + output, point">
                Tokens
              </th>
              <th className="px-2 py-1.5 text-right font-semibold" title="Prompt + output, upper bound">
                Upper tok
              </th>
              <th className="px-2 py-1.5 text-right font-semibold">Point $</th>
              <th className="px-2.5 py-1.5 text-right font-semibold">Upper $</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {Array.from(byArm, ([armName, cells]) => {
              const arm = arms.get(armName);
              const isNull = arm?.is_null ?? armName.startsWith("null_");
              return (
                <Fragment key={armName}>
                  {cells.map((cell, i) => (
                    <tr
                      key={`${armName}:${budgetLabel(cell.budget)}`}
                      style={isNull ? FLOOR_HATCH : undefined}
                      className={`border-b border-[#27272a]/60 ${
                        i === cells.length - 1 ? "border-[#27272a]" : ""
                      } text-[#d4d4d8]`}
                    >
                      <td className="px-2.5 py-1 font-sans">
                        {i === 0 ? (
                          <span className="flex items-center gap-1.5">
                            <ArmMark arm={armName} family={arm?.family} isNull={isNull} />
                            <span
                              className={`truncate text-[11px] font-medium ${
                                isNull ? "text-[#a1a1aa]" : "text-[#e4e4e7]"
                              }`}
                            >
                              {arm?.title ?? armName}
                            </span>
                          </span>
                        ) : null}
                      </td>
                      <td className="px-2 py-1 text-[#a1a1aa]">{budgetLabel(cell.budget)}</td>
                      <td className="px-2 py-1 text-right">
                        {cell.calls.toLocaleString()}
                        {cell.note && (
                          <span title={cell.note} className="ml-1 cursor-help text-[#52525b]">
                            *
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1 text-right text-[#a1a1aa]">
                        {formatTokens(cell.context_tokens)}
                        {cell.measured && armName !== "null_closed_book" && (
                          <span title="Measured on a completed retrieve phase" className="ml-0.5 text-emerald-400/70">
                            m
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1 text-right">{formatTokens(tokens(cell))}</td>
                      <td className="px-2 py-1 text-right text-[#a1a1aa]">
                        {formatTokens(upperTokens(cell))}
                      </td>
                      <td className="px-2 py-1 text-right">{formatUsd(cell.point_usd)}</td>
                      <td className="px-2.5 py-1 text-right text-[#e4e4e7]">
                        {formatUsd(cell.upper_usd)}
                      </td>
                    </tr>
                  ))}
                </Fragment>
              );
            })}
          </tbody>
          <tfoot className="font-mono tabular-nums">
            {reader && (
              <tr className="border-b border-[#27272a]/60 text-[#a1a1aa]">
                <td className="px-2.5 py-1 font-sans text-[11px]" colSpan={2}>
                  {PHASE_LABEL.reader}
                  {reader.batch && (
                    <span className="ml-1.5 font-mono text-[10px] text-amber-300/80">
                      ×{estimate.batch_multiplier ?? 0.5} batch
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 text-right">{reader.calls.toLocaleString()}</td>
                <td />
                <td className="px-2 py-1 text-right">{formatTokens(tokens(reader))}</td>
                <td className="px-2 py-1 text-right">{formatTokens(upperTokens(reader))}</td>
                <td className="px-2 py-1 text-right">{formatUsd(reader.point_usd)}</td>
                <td className="px-2.5 py-1 text-right">{formatUsd(reader.upper_usd)}</td>
              </tr>
            )}
            {otherPhases.map((phase) => (
              <tr key={phase.phase} className="border-b border-[#27272a]/60 text-[#a1a1aa]">
                <td className="px-2.5 py-1 font-sans text-[11px]" colSpan={2}>
                  {PHASE_LABEL[phase.phase] ?? phase.phase}
                  <span className="ml-1.5 font-mono text-[10px] text-[#52525b]">{phase.model}</span>
                </td>
                <td className="px-2 py-1 text-right">{phase.calls.toLocaleString()}</td>
                <td />
                <td className="px-2 py-1 text-right">{formatTokens(tokens(phase))}</td>
                <td className="px-2 py-1 text-right">{formatTokens(upperTokens(phase))}</td>
                <td className="px-2 py-1 text-right">{formatUsd(phase.point_usd)}</td>
                <td className="px-2.5 py-1 text-right">{formatUsd(phase.upper_usd)}</td>
              </tr>
            ))}
            <tr className="bg-[#18181b]/60 font-semibold text-[#fafafa]">
              <td className="px-2.5 py-1.5 font-sans text-[11px]" colSpan={6}>
                Total
                {cap != null && !free && (
                  <span className="ml-2 font-mono text-[10px] font-normal text-[#71717a]">
                    cap {formatUsd(cap)}
                  </span>
                )}
              </td>
              <td className="px-2 py-1.5 text-right">{formatUsd(estimate.total_point_usd)}</td>
              <td
                className={`px-2.5 py-1.5 text-right ${
                  estimate.refuse ? "text-rose-300" : "text-[#fafafa]"
                }`}
              >
                {formatUsd(estimate.total_upper_usd)}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      )}

      {/* Provenance of the numbers */}
      <div className="flex flex-col gap-1.5 text-[11px] leading-relaxed text-[#71717a]">
        {!free && (
        <p>
          <span className="font-mono text-[#a1a1aa]">{estimate.reader_model}</span>
          {estimate.tokenizer && (
            <>
              {" · tokenizer "}
              <span className="font-mono text-[#a1a1aa]">{estimate.tokenizer}</span>
            </>
          )}
          {estimate.max_output_tokens != null && (
            <>
              {" · output cap "}
              <span className="font-mono text-[#a1a1aa]">{estimate.max_output_tokens}</span>
            </>
          )}
          {estimate.reasoning_allowance ? (
            <>
              {" (incl. "}
              <span className="font-mono text-[#a1a1aa]">{estimate.reasoning_allowance}</span>
              {" reasoning)"}
            </>
          ) : null}
        </p>
        )}
        {!free && dates.length > 0 && (
          <p>
            Prices hand-recorded, not fetched:{" "}
            {dates.map(([what, date], i) => (
              <span key={what}>
                {i > 0 && " · "}
                <span className="text-[#a1a1aa]">{what}</span>{" "}
                <span className="font-mono">{date}</span>
              </span>
            ))}
            {" — "}
            <a
              href={pricingUrl}
              target="_blank"
              rel="noreferrer"
              className="text-indigo-300/80 underline decoration-indigo-400/30 underline-offset-2 hover:text-indigo-200"
            >
              verify
            </a>
          </p>
        )}
        {assumptions.length > 0 && (
          <div>
            <button
              onClick={() => setShowAssumptions((v) => !v)}
              aria-expanded={showAssumptions}
              className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[#52525b] transition-colors hover:text-[#a1a1aa]"
            >
              <ChevronDown
                size={11}
                className={`transition-transform ${showAssumptions ? "" : "-rotate-90"}`}
              />
              {assumptions.length} {assumptions.length === 1 ? "assumption" : "assumptions"}
            </button>
            {showAssumptions && (
              <ul className="mt-1 flex list-disc flex-col gap-1 pl-5 text-[11px] leading-relaxed text-[#a1a1aa] marker:text-[#3f3f46]">
                {assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
                {estimate.phases
                  .flatMap((p) => (p.notes ?? []).map((note) => `${PHASE_LABEL[p.phase] ?? p.phase}: ${note}`))
                  .map((note) => (
                    <li key={note} className="text-[#71717a]">
                      {note}
                    </li>
                  ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
