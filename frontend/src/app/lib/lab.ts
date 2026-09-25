// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
//
// Synapse Lab — pure formatting and geometry helpers shared by the Lab view.

import type { LabBudget, LabFrontierPoint } from "./types";

/** The runner's own rule: `^gpt-5` or `^o[1-9]` bills hidden reasoning as output. */
export function isReasoningModel(model: string | null | undefined): boolean {
  return /^(gpt-5|o[1-9])/.test((model ?? "").trim().toLowerCase());
}

/** 500 → "500", 2000 → "2k", 2500 → "2.5k", null → "default". */
export function budgetLabel(budget: LabBudget | undefined): string {
  if (budget == null) return "default";
  if (budget >= 1000 && budget % 100 === 0) {
    return `${(budget / 1000).toString()}k`;
  }
  return budget.toLocaleString();
}

/** The wire / leaderboard key of a budget ("default" for the uncapped context). */
export function budgetKey(budget: LabBudget | undefined): string {
  return budget == null ? "default" : String(budget);
}

/** Ascending, the uncapped default last — the order budgets are read in. */
export function sortBudgets(budgets: LabBudget[]): LabBudget[] {
  return [...budgets].sort((a, b) => {
    if (a == null) return b == null ? 0 : 1;
    if (b == null) return -1;
    return a - b;
  });
}

function isNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Dollars with enough precision to be honest about tiny per-query costs:
 * "$1.23", "$0.045", "$0.0004", "$0.000062". Null → "—" (unpriced, never $0).
 */
export function formatUsd(value: number | null | undefined): string {
  if (!isNum(value)) return "—";
  if (value === 0) return "$0";
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (abs < 1e-6) return `${sign}<$0.000001`;
  // Two significant digits below a hundredth of a cent.
  if (abs < 0.0001) return `${sign}$${abs.toFixed(Math.ceil(-Math.log10(abs)) + 1)}`;
  if (abs < 0.01) return `${sign}$${abs.toFixed(4)}`;
  if (abs < 1) return `${sign}$${abs.toFixed(3)}`;
  return `${sign}$${abs.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/** 812 → "812", 4210 → "4,210", 38400 → "38.4k", 2.1e6 → "2.10M". */
export function formatTokens(value: number | null | undefined): string {
  if (!isNum(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  return Math.round(value).toLocaleString();
}

/** A score already in points (0–100). */
export function formatPts(value: number | null | undefined, digits = 1): string {
  return isNum(value) ? value.toFixed(digits) : "—";
}

/** A signed difference in points: "+4.2", "−1.0". */
export function formatDelta(value: number | null | undefined, digits = 1): string {
  if (!isNum(value)) return "—";
  const text = Math.abs(value).toFixed(digits);
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

export function formatWhen(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** "Robertson & Zaragoza, 'The Probabilistic…'" → "Robertson & Zaragoza". */
export function shortCitation(citation: string | null | undefined): string {
  const text = (citation ?? "").trim();
  if (!text) return "source";
  const head = text.split(/[,(:]/)[0].trim();
  return head.length >= 4 && head.length <= 48 ? head : `${text.slice(0, 44).trim()}…`;
}

/** The leaf name of a stored path, for display. */
export function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

// ── Plot geometry ─────────────────────────────────────

/** 1-2-5 ticks inside [min, max] on a log axis (only decades when crowded). */
export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > min)) return [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const pick = (mantissas: number[]) => {
    const ticks: number[] = [];
    for (let e = lo; e <= hi; e++) {
      for (const m of mantissas) {
        const v = m * 10 ** e;
        if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
      }
    }
    return ticks;
  };
  const dense = pick([1, 2, 5]);
  return dense.length > 7 ? pick([1]) : dense;
}

/** Non-dominated points: minimise x, maximise y — cheapest first. */
export function paretoFrontier<T extends { x: number; y: number }>(points: T[]): T[] {
  const frontier = points.filter(
    (p) =>
      !points.some(
        (q) => q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y),
      ),
  );
  return frontier.sort((a, b) => a.x - b.x || b.y - a.y);
}

/** A frontier point's identity, to match the server's frontier to plotted points. */
export function cellKey(arm: string, budget: LabBudget | undefined): string {
  return `${arm}\u0000${budgetKey(budget)}`;
}

export function frontierKeys(points: LabFrontierPoint[] | undefined | null): Set<string> {
  return new Set((points ?? []).map((p) => cellKey(p.arm, p.budget)));
}
