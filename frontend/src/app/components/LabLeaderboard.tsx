// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { Fragment, type KeyboardEvent, type ReactNode, useState } from "react";
import { TriangleAlert } from "lucide-react";
import {
  budgetLabel,
  cellKey,
  formatDelta,
  formatPts,
  formatTokens,
  formatUsd,
} from "../lib/lab";
import type { LabArm, LabBudget, LabComparison, LabLeaderboard, LabLeaderboardRow } from "../lib/types";
import SegmentedToggle, { type SegmentedOption } from "./SegmentedToggle";
import { ArmMark, FLOOR_HATCH, FLOOR_TAG, Tag } from "./LabShared";

interface LabLeaderboardProps {
  board: LabLeaderboard;
  arms: Map<string, LabArm>;
  selected: string | null;
  onSelect: (arm: string, budget: LabBudget) => void;
}

type Columns = "answers" | "retrieval";

const COLUMN_OPTIONS: SegmentedOption<Columns>[] = [
  { value: "answers", label: "Answers", title: "Quality, tokens and cost of the answers" },
  {
    value: "retrieval",
    label: "Retrieval",
    title: "Context size, units by kind, answer containment and paragraph recall",
  },
];

/** Evidence units, by the packer's section names. */
const KIND_LABEL: Record<string, string> = {
  prose: "excerpts",
  entity: "facts",
  relation: "relations",
  path: "paths",
  community: "themes",
  name_list: "names",
};

function rowKey(row: LabLeaderboardRow): string {
  return cellKey(row.arm, row.budget);
}

function isNullRow(row: LabLeaderboardRow, arms: Map<string, LabArm>): boolean {
  return row.is_null ?? arms.get(row.arm)?.is_null ?? false;
}

/**
 * A paired-bootstrap difference. Colored only when it is reportable (clears the
 * effect floor AND its 95% CI excludes 0); otherwise it reads "n.s." in gray.
 */
function DeltaCell({ c }: { c: LabComparison | undefined }) {
  if (!c || c.n === 0) return <span className="text-[#3f3f46]">—</span>;
  const tone = !c.reportable
    ? "text-[#71717a]"
    : c.diff > 0
      ? "text-emerald-300"
      : "text-rose-300";
  const title =
    `${c.verdict ?? (c.reportable ? (c.diff > 0 ? "above" : "below") : "too close to call")}` +
    ` · vs ${c.against}\n` +
    `95% CI [${formatDelta(c.ci_low)}, ${formatDelta(c.ci_high)}] points · ` +
    `effect floor ±${c.floor.toFixed(1)} · n = ${c.n}`;
  return (
    <span title={title} className={`cursor-help whitespace-nowrap ${tone}`}>
      {formatDelta(c.diff)}
      {!c.reportable && <span className="ml-0.5 text-[9px] text-[#52525b]">n.s.</span>}
    </span>
  );
}

/** A percentage with a bar, and a tick at the best evidence floor at that budget. */
function MeterCell({
  value,
  floor,
  isFloor,
}: {
  value: number | null | undefined;
  floor: number | null;
  isFloor: boolean;
}) {
  if (value == null) return <span className="text-[#3f3f46]">—</span>;
  const above = floor != null && !isFloor ? value - floor : null;
  return (
    <span className="flex items-center justify-end gap-2">
      <span
        title={
          floor != null
            ? `Best evidence floor at this budget: ${formatPts(floor)}%`
            : undefined
        }
        className="relative h-1.5 w-14 shrink-0 overflow-hidden rounded-full bg-[#27272a]"
      >
        <span
          className={`absolute inset-y-0 left-0 rounded-full ${
            isFloor ? "bg-[#71717a]" : "bg-indigo-400/80"
          }`}
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        />
        {floor != null && (
          <span
            className="absolute inset-y-[-1px] w-[2px] bg-[#fafafa]"
            style={{ left: `calc(${Math.max(0, Math.min(100, floor))}% - 1px)` }}
          />
        )}
      </span>
      <span className="w-10 text-right">{formatPts(value)}</span>
      {above != null && (
        <span
          title="Points above the best evidence floor at this budget (no significance test)"
          className={`w-9 text-right text-[10px] ${above > 0 ? "text-[#a1a1aa]" : "text-[#52525b]"}`}
        >
          {formatDelta(above)}
        </span>
      )}
    </span>
  );
}

function ArmCell({
  row,
  arms,
  warn,
}: {
  row: LabLeaderboardRow;
  arms: Map<string, LabArm>;
  warn?: ReactNode;
}) {
  const arm = arms.get(row.arm);
  const isNull = isNullRow(row, arms);
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <ArmMark arm={row.arm} family={row.family ?? arm?.family} isNull={isNull} />
      <span
        title={row.arm}
        className={`truncate font-sans text-[11px] font-medium ${
          isNull ? "text-[#a1a1aa]" : "text-[#e4e4e7]"
        }`}
      >
        {row.title ?? arm?.title ?? row.arm}
      </span>
      {isNull && <Tag className={FLOOR_TAG}>floor</Tag>}
      {warn}
    </span>
  );
}

function ErrorFlag({ row }: { row: LabLeaderboardRow }) {
  const retrieval = row.retrieval_errors ?? 0;
  const read = row.read_errors ?? 0;
  if (!retrieval && !read) return null;
  const parts = [
    retrieval ? `${retrieval} retrieval error${retrieval === 1 ? "" : "s"}` : "",
    read ? `${read} unanswered (failed or stopped at the cap; not scored as 0)` : "",
  ].filter(Boolean);
  return (
    <span title={parts.join("\n")} className="shrink-0 cursor-help text-amber-400/80">
      <TriangleAlert size={10} />
    </span>
  );
}

function kindSummary(kinds: Record<string, number> | undefined): string {
  const entries = Object.entries(kinds ?? {})
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return "—";
  return entries
    .slice(0, 3)
    .map(([k, v]) => `${v.toFixed(v >= 10 ? 0 : 1)} ${KIND_LABEL[k] ?? k}`)
    .join(" · ");
}

function Th({
  children,
  hint,
  right = true,
}: {
  children: ReactNode;
  hint?: string;
  right?: boolean;
}) {
  return (
    <th
      title={hint}
      className={`whitespace-nowrap px-2 py-1.5 font-semibold ${right ? "text-right" : "text-left"} ${
        hint ? "cursor-help" : ""
      }`}
    >
      {children}
    </th>
  );
}

/** One clickable row: a click (or Enter) opens that cell's questions. */
function Row({
  row,
  arms,
  selected,
  onSelect,
  children,
}: {
  row: LabLeaderboardRow;
  arms: Map<string, LabArm>;
  selected: string | null;
  onSelect: (arm: string, budget: LabBudget) => void;
  children: ReactNode;
}) {
  const isSelected = selected === rowKey(row);
  const open = () => onSelect(row.arm, row.budget);
  return (
    <tr
      tabIndex={0}
      onClick={open}
      onKeyDown={(e: KeyboardEvent) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      }}
      aria-selected={isSelected}
      style={isNullRow(row, arms) ? FLOOR_HATCH : undefined}
      className={`cursor-pointer border-b border-[#27272a]/60 outline-none transition-colors focus-visible:bg-[#27272a]/50 ${
        isSelected ? "bg-indigo-500/[0.08]" : "hover:bg-[#18181b]"
      }`}
    >
      {children}
    </tr>
  );
}

function AnswerTable({ board, arms, selected, onSelect }: LabLeaderboardProps) {
  const hasPremium = board.rows.some((r) => r.comparisons?.graph_premium);
  return (
    <table className="w-full min-w-[820px] border-collapse text-[11px]">
      <thead>
        <tr className="border-b border-[#27272a] bg-[#18181b]/60 text-[10px] uppercase tracking-wider text-[#71717a]">
          <Th hint="Rank by $ per 100 correct (cost-of-pass); unpriced or zero-correct rows are unranked">
            #
          </Th>
          <Th right={false}>Arm</Th>
          <Th right={false}>Budget</Th>
          <Th hint="Token F1 against the gold answer, in points (HotpotQA normalisation)">F1</Th>
          <Th hint="Exact match, in points">EM</Th>
          <Th hint="Reader tokens per question: prompt + output (hidden reasoning included)">
            Tok / q
          </Th>
          <Th hint="Reader tokens ÷ exact-match correct answers">Tok / correct</Th>
          <Th hint="Cost-of-pass: reader $ ÷ correct answers × 100 (Erol et al., arXiv:2504.13359). Rows are ranked by this.">
            $ / 100 correct
          </Th>
          <Th hint="Gain above closed-book: F1 − F1(N0), paired bootstrap">Δ N0</Th>
          <Th hint="Gain above random context at the same budget: F1 − F1(N2), paired bootstrap">
            Δ N2
          </Th>
          {hasPremium && (
            <Th hint="Graph arms only: F1 − max(F1 BM25, F1 dense) at the same budget, paired bootstrap">
              Graph premium
            </Th>
          )}
        </tr>
      </thead>
      <tbody className="font-mono tabular-nums text-[#d4d4d8]">
        {board.rows.map((row) => (
          <Row key={rowKey(row)} row={row} arms={arms} selected={selected} onSelect={onSelect}>
            <td
              className={`px-2 py-1.5 text-right ${
                row.rank === 1 ? "font-semibold text-indigo-300" : "text-[#52525b]"
              }`}
            >
              {row.rank ?? "—"}
            </td>
            <td className="max-w-[220px] px-2 py-1.5">
              <ArmCell row={row} arms={arms} warn={<ErrorFlag row={row} />} />
            </td>
            <td className="px-2 py-1.5 text-[#a1a1aa]">{budgetLabel(row.budget)}</td>
            <td className="px-2 py-1.5 text-right font-medium text-[#fafafa]">
              {formatPts(row.f1)}
            </td>
            <td className="px-2 py-1.5 text-right">{formatPts(row.em)}</td>
            <td className="px-2 py-1.5 text-right text-[#a1a1aa]">
              {formatTokens(row.tokens_per_query)}
            </td>
            <td className="px-2 py-1.5 text-right">{formatTokens(row.tokens_per_correct)}</td>
            <td className="px-2 py-1.5 text-right text-[#fafafa]">
              {formatUsd(row.usd_per_100_correct)}
            </td>
            <td className="px-2 py-1.5 text-right">
              <DeltaCell c={row.comparisons?.gain_above_n0} />
            </td>
            <td className="px-2 py-1.5 text-right">
              <DeltaCell c={row.comparisons?.gain_above_n2} />
            </td>
            {hasPremium && (
              <td className="px-2 py-1.5 text-right">
                <DeltaCell c={row.comparisons?.graph_premium} />
              </td>
            )}
          </Row>
        ))}
      </tbody>
    </table>
  );
}

/**
 * Retrieval-level columns, one block per budget with the floors first, so the
 * floor a containment or recall figure has to clear sits right above it.
 */
function RetrievalTable({ board, arms, selected, onSelect }: LabLeaderboardProps) {
  const hasContainment = board.rows.some((r) => r.containment != null);
  const hasRecall = board.rows.some(
    (r) => r.recall_permissive != null || r.recall_strict != null,
  );

  // One block per budget, ascending (the uncapped default last) — a read run
  // lists its rows by rank, which would scatter the budgets.
  const groups = new Map<string, LabLeaderboardRow[]>();
  const byBudget = [...board.rows].sort((a, b) => {
    if (a.budget == null) return b.budget == null ? 0 : 1;
    if (b.budget == null) return -1;
    return a.budget - b.budget;
  });
  for (const row of byBudget) {
    const key = budgetLabel(row.budget);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  const ordered = (rows: LabLeaderboardRow[]) =>
    [...rows].sort((a, b) => Number(isNullRow(b, arms)) - Number(isNullRow(a, arms)));
  const floorOf = (rows: LabLeaderboardRow[], key: keyof LabLeaderboardRow) => {
    const values = rows
      .filter((r) => isNullRow(r, arms))
      .map((r) => r[key])
      .filter((v): v is number => typeof v === "number");
    return values.length ? Math.max(...values) : null;
  };
  const columns = 5 + (hasContainment ? 1 : 0) + (hasRecall ? 2 : 0);

  return (
    <table className="w-full min-w-[760px] border-collapse text-[11px]">
      <thead>
        <tr className="border-b border-[#27272a] bg-[#18181b]/60 text-[10px] uppercase tracking-wider text-[#71717a]">
          <Th right={false}>Arm</Th>
          <Th hint="Mean packed context tokens per question (real tokenizer); max in the tooltip">
            Ctx tokens
          </Th>
          {hasContainment && (
            <Th hint="% of contexts that contain a gold answer string (normalised, word-bounded). A containment-style measure: the vocabulary null can game it, which is why the floors sit beside it. Tick = best floor at this budget.">
              Answer in context
            </Th>
          )}
          {hasRecall && (
            <>
              <Th hint="HotpotQA: gold paragraphs credited only when their prose is in the context. Tick = best floor.">
                Recall strict
              </Th>
              <Th hint="HotpotQA: gold paragraphs credited by prose OR by a graph unit naming them. Tick = best floor.">
                Recall permissive
              </Th>
            </>
          )}
          <Th hint="Mean evidence units packed per question">Units</Th>
          <Th right={false} hint="Mean units per question by kind (top 3)">
            By kind
          </Th>
          <Th hint="% of contexts where the budget cut at least one unit">Cut</Th>
        </tr>
      </thead>
      <tbody className="font-mono tabular-nums text-[#d4d4d8]">
        {Array.from(groups, ([label, rows]) => {
          const containmentFloor = floorOf(rows, "containment");
          const strictFloor = floorOf(rows, "recall_strict");
          const permissiveFloor = floorOf(rows, "recall_permissive");
          return (
            <Fragment key={label}>
              <tr className="border-b border-[#27272a]/60 bg-[#09090b]">
                <td
                  colSpan={columns}
                  className="px-2 pb-1 pt-2.5 font-sans text-[10px] font-semibold uppercase tracking-wider text-[#a1a1aa]"
                >
                  {label === "default" ? "Default context (uncapped)" : `Budget ${label} tokens`}
                </td>
              </tr>
              {ordered(rows).map((row) => {
                const isNull = isNullRow(row, arms);
                return (
                  <Row
                    key={rowKey(row)}
                    row={row}
                    arms={arms}
                    selected={selected}
                    onSelect={onSelect}
                  >
                    <td className="max-w-[240px] px-2 py-1.5">
                      <ArmCell row={row} arms={arms} warn={<ErrorFlag row={row} />} />
                    </td>
                    <td
                      className="px-2 py-1.5 text-right text-[#fafafa]"
                      title={
                        row.context_tokens_max != null
                          ? `max ${formatTokens(row.context_tokens_max)}`
                          : undefined
                      }
                    >
                      {formatTokens(row.context_tokens_mean)}
                    </td>
                    {hasContainment && (
                      <td className="px-2 py-1.5">
                        <MeterCell
                          value={row.containment}
                          floor={containmentFloor}
                          isFloor={isNull}
                        />
                      </td>
                    )}
                    {hasRecall && (
                      <>
                        <td className="px-2 py-1.5">
                          <MeterCell
                            value={row.recall_strict}
                            floor={strictFloor}
                            isFloor={isNull}
                          />
                        </td>
                        <td className="px-2 py-1.5">
                          <MeterCell
                            value={row.recall_permissive}
                            floor={permissiveFloor}
                            isFloor={isNull}
                          />
                        </td>
                      </>
                    )}
                    <td className="px-2 py-1.5 text-right">{formatPts(row.units_mean)}</td>
                    <td className="max-w-[200px] truncate px-2 py-1.5 text-left text-[10px] text-[#a1a1aa]">
                      {kindSummary(row.units_by_kind_mean)}
                    </td>
                    <td className="px-2 py-1.5 text-right text-[#a1a1aa]">
                      {row.truncated_rate != null ? `${formatPts(row.truncated_rate, 0)}%` : "—"}
                    </td>
                  </Row>
                );
              })}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

/** The leaderboard: rows (arm, budget), floors styled apart, cost-normalized ranking. */
export default function LabLeaderboard(props: LabLeaderboardProps) {
  const { board } = props;
  const read = board.mode !== "retrieve";
  const [columns, setColumns] = useState<Columns>("answers");
  const view: Columns = read ? columns : "retrieval";
  const floor = board.effect_floor_points;
  const iterations = board.bootstrap?.iterations ?? 10000;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <p className="min-w-0 flex-1 text-[11px] leading-relaxed text-[#71717a]">
          {read ? (
            <>
              Ranked by <span className="text-[#a1a1aa]">$ per 100 correct</span>. A difference
              is colored only when it clears the effect floor (
              <span className="font-mono text-[#a1a1aa]">±{formatPts(floor)}</span> points, one
              question in {board.n_questions}) and its 95% paired-bootstrap CI (
              {iterations.toLocaleString()} resamples) excludes 0; otherwise{" "}
              <span className="font-mono">n.s.</span>
            </>
          ) : (
            <>
              Retrieve-only: no reader, $0. Each budget lists the evidence floors first; the tick
              on a bar is the best floor at that budget, so a figure only means something above
              it.
            </>
          )}
        </p>
        {read && (
          <SegmentedToggle
            label="Leaderboard columns"
            value={columns}
            options={COLUMN_OPTIONS}
            onChange={setColumns}
          />
        )}
      </div>
      <div className="overflow-x-auto rounded-md border border-[#27272a]">
        {view === "answers" ? <AnswerTable {...props} /> : <RetrievalTable {...props} />}
      </div>
      {board.notes && board.notes.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-[10px] leading-snug text-[#52525b]">
          {board.notes.map((note) => (
            <li key={note}>· {note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
