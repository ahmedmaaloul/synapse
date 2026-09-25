// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import { type MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { colorForFamily } from "../lib/constants";
import {
  budgetKey,
  budgetLabel,
  cellKey,
  formatPts,
  formatTokens,
  formatUsd,
  frontierKeys,
  logTicks,
  paretoFrontier,
} from "../lib/lab";
import type { LabArm, LabFrontiers, LabLeaderboard, LabLeaderboardRow } from "../lib/types";
import SegmentedToggle, { type SegmentedOption } from "./SegmentedToggle";
import { ArmMark, shapeFor, shapePath } from "./LabShared";

interface LabParetoProps {
  board: LabLeaderboard;
  frontiers: LabFrontiers | null;
  arms: Map<string, LabArm>;
}

type XAxis = "usd" | "tokens";
type YMetric = "f1" | "containment" | "recall_permissive" | "recall_strict";

const X_OPTIONS: SegmentedOption<XAxis>[] = [
  { value: "usd", label: "F1 vs $", title: "Reader $ per question, log scale" },
  { value: "tokens", label: "F1 vs tokens", title: "Reader tokens per question, log scale" },
];

const Y_LABEL: Record<YMetric, string> = {
  f1: "F1 (points)",
  containment: "Answer in context (%)",
  recall_permissive: "Paragraph recall, permissive (%)",
  recall_strict: "Paragraph recall, strict (%)",
};

const HEIGHT = 270;
const MARGIN = { top: 16, right: 20, bottom: 38, left: 44 };
const R = 4.6; // ≥ 8px markers
const SURFACE = "#09090b";
const INK_MUTED = "#71717a";
const INK_SECONDARY = "#a1a1aa";
const FRONTIER = "#e4e4e7";

interface Point {
  key: string;
  row: LabLeaderboardRow;
  arm: string;
  family: string | null;
  title: string;
  isNull: boolean;
  x: number;
  y: number;
}

function formatX(axis: XAxis | "context", v: number): string {
  if (axis === "usd") {
    if (v >= 1) return `$${v.toFixed(v >= 10 ? 0 : 1)}`;
    // Enough decimals to show the leading digit of a tiny per-query cost.
    const decimals = Math.min(8, Math.max(2, Math.ceil(-Math.log10(v))));
    return `$${v.toFixed(decimals).replace(/0+$/, "").replace(/\.$/, "")}`;
  }
  if (v >= 1000) return `${(v / 1000).toString()}k`;
  return String(v);
}

/**
 * F1 against what it cost, one point per (arm, budget). Log x axis; the band
 * is the evidence floor (min–max of the null controls); the dashed line joins
 * the Pareto frontier (nothing is both cheaper and better).
 */
export default function LabPareto({ board, frontiers, arms }: LabParetoProps) {
  const read = board.mode !== "retrieve";
  const [axis, setAxis] = useState<XAxis>("usd");
  const [hover, setHover] = useState<string | null>(null);
  const [width, setWidth] = useState(640);
  const boxRef = useRef<HTMLDivElement>(null);

  // Retrieve-only runs have no F1: plot the retrieval measure the run has.
  const yMetric: YMetric = read
    ? "f1"
    : board.rows.some((r) => r.recall_permissive != null)
      ? "recall_permissive"
      : "containment";
  const xAxis: XAxis | "context" = read ? axis : "context";

  const { points, offAxis } = useMemo(() => {
    const out: Point[] = [];
    let skipped = 0;
    for (const row of board.rows) {
      const y = row[yMetric];
      const x =
        xAxis === "usd"
          ? row.usd_per_query
          : xAxis === "tokens"
            ? row.tokens_per_query
            : row.context_tokens_mean;
      if (typeof y !== "number") continue;
      if (typeof x !== "number" || !(x > 0)) {
        skipped += 1;
        continue;
      }
      const arm = arms.get(row.arm);
      out.push({
        key: cellKey(row.arm, row.budget),
        row,
        arm: row.arm,
        family: row.family ?? arm?.family ?? null,
        title: row.title ?? arm?.title ?? row.arm,
        isNull: row.is_null ?? arm?.is_null ?? false,
        x,
        y,
      });
    }
    return { points: out, offAxis: skipped };
  }, [board.rows, arms, xAxis, yMetric]);

  // The server's frontier when it has one for this view; computed otherwise.
  const frontier = useMemo(() => {
    const server =
      xAxis === "usd" ? frontiers?.f1_vs_usd : xAxis === "tokens" ? frontiers?.f1_vs_tokens : null;
    if (server && server.length > 0) {
      const keys = frontierKeys(server);
      return points.filter((p) => keys.has(p.key)).sort((a, b) => a.x - b.x);
    }
    return paretoFrontier(points);
  }, [frontiers, points, xAxis]);
  const onFrontier = useMemo(() => new Set(frontier.map((p) => p.key)), [frontier]);

  // The plot box only exists once there is something to plot.
  const hasPoints = points.length > 0;
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    // The observer reports the initial size too: no synchronous read needed.
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.max(320, Math.round(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasPoints]);

  // The floor band spans every null control's score, plotted or not.
  const floorValues = board.rows
    .filter((r) => r.is_null ?? arms.get(r.arm)?.is_null)
    .map((r) => r[yMetric])
    .filter((v): v is number => typeof v === "number");
  const floorLo = floorValues.length ? Math.min(...floorValues) : null;
  const floorHi = floorValues.length ? Math.max(...floorValues) : null;

  if (!hasPoints) {
    return (
      <div className="rounded-md border border-dashed border-[#27272a] px-3 py-6 text-center text-[12px] text-[#52525b]">
        Nothing to plot yet
        {read && axis === "usd" ? ": no priced cell (a model without a price has no $)." : "."}
      </div>
    );
  }

  // ── Scales ──
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const xs = points.map((p) => p.x);
  let xMin = Math.min(...xs);
  let xMax = Math.max(...xs);
  if (xMax / xMin < 1.5) {
    xMin /= 2;
    xMax *= 2;
  } else {
    xMin /= 1.35;
    xMax *= 1.35;
  }
  const lx0 = Math.log10(xMin);
  const lx1 = Math.log10(xMax);
  const sx = (v: number) => MARGIN.left + ((Math.log10(v) - lx0) / (lx1 - lx0)) * innerW;

  const yTop = Math.min(
    100,
    Math.max(10, Math.ceil((Math.max(...points.map((p) => p.y), floorHi ?? 0) + 4) / 10) * 10),
  );
  const yStep = yTop <= 20 ? 5 : yTop <= 50 ? 10 : 20;
  const yTicks: number[] = [];
  for (let v = 0; v <= yTop + 1e-9; v += yStep) yTicks.push(v);
  const sy = (v: number) => MARGIN.top + innerH - (Math.max(0, Math.min(yTop, v)) / yTop) * innerH;

  const xTicks = logTicks(xMin, xMax);

  // Each arm's budget curve: its points joined in budget order.
  const curves = new Map<string, Point[]>();
  for (const p of points) curves.set(p.arm, [...(curves.get(p.arm) ?? []), p]);
  const budgetOrder = (p: Point) => (p.row.budget == null ? Infinity : p.row.budget);

  const hovered = points.find((p) => p.key === hover) ?? null;

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: Point | null = null;
    let bestD = 18 * 18; // hit target larger than the mark
    for (const p of points) {
      const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    setHover(best?.key ?? null);
  };

  const legendArms = Array.from(
    new Map(points.map((p) => [p.arm, p] as const)).values(),
  );
  const xTitle =
    xAxis === "usd"
      ? "Reader $ per question (log)"
      : xAxis === "tokens"
        ? "Reader tokens per question (log)"
        : "Context tokens per question (log)";

  // Tooltip placement, flipped left near the right edge.
  const tipX = hovered ? sx(hovered.x) : 0;
  const tipY = hovered ? sy(hovered.y) : 0;
  const flip = tipX > width - 200;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="min-w-0 flex-1 text-[11px] leading-relaxed text-[#71717a]">
          {read
            ? "Up and to the left is better. The dashed line is the Pareto frontier: no other cell is both cheaper and better."
            : "Retrieve-only: how much of the answer reaches the context, against its size. The dashed line is the frontier."}
        </p>
        {read && (
          <SegmentedToggle label="Pareto x axis" value={axis} options={X_OPTIONS} onChange={setAxis} />
        )}
      </div>

      <div ref={boxRef} className="relative w-full">
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`${Y_LABEL[yMetric]} against ${xTitle}: ${points.length} cells, ${frontier.length} on the Pareto frontier`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          className="block select-none"
        >
          {/* Grid + y axis */}
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={sy(v)}
                y2={sy(v)}
                stroke="#27272a"
                strokeWidth={1}
                strokeOpacity={v === 0 ? 1 : 0.55}
              />
              <text
                x={MARGIN.left - 8}
                y={sy(v)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={10}
                fill={INK_MUTED}
                className="font-mono"
              >
                {v}
              </text>
            </g>
          ))}
          {xTicks.map((v) => (
            <g key={`x${v}`}>
              <line
                x1={sx(v)}
                x2={sx(v)}
                y1={MARGIN.top}
                y2={MARGIN.top + innerH}
                stroke="#27272a"
                strokeWidth={1}
                strokeOpacity={0.35}
              />
              <text
                x={sx(v)}
                y={MARGIN.top + innerH + 14}
                textAnchor="middle"
                fontSize={10}
                fill={INK_MUTED}
                className="font-mono"
              >
                {formatX(xAxis, v)}
              </text>
            </g>
          ))}
          <text
            x={MARGIN.left + innerW / 2}
            y={HEIGHT - 6}
            textAnchor="middle"
            fontSize={10}
            fill={INK_MUTED}
          >
            {xTitle}
          </text>
          <text
            transform={`translate(11 ${MARGIN.top + innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={10}
            fill={INK_MUTED}
          >
            {Y_LABEL[yMetric]}
          </text>

          {/* Evidence floor band (N0–N2) */}
          {floorLo != null && floorHi != null && (
            <g>
              <rect
                x={MARGIN.left}
                width={innerW}
                y={sy(floorHi)}
                height={Math.max(2, sy(floorLo) - sy(floorHi))}
                fill="#a1a1aa"
                fillOpacity={0.08}
              />
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={sy(floorHi)}
                y2={sy(floorHi)}
                stroke="#52525b"
                strokeDasharray="2 3"
              />
              <text
                x={width - MARGIN.right - 4}
                y={sy(floorHi) - 4}
                textAnchor="end"
                fontSize={9.5}
                fill={INK_MUTED}
                stroke={SURFACE}
                strokeWidth={3}
                paintOrder="stroke"
                className="pointer-events-none"
              >
                evidence floor (N0–N2) {formatPts(floorHi)}
              </text>
            </g>
          )}

          {/* Per-arm budget curves */}
          {Array.from(curves, ([arm, list]) => {
            if (list.length < 2) return null;
            const sorted = [...list].sort((a, b) => budgetOrder(a) - budgetOrder(b));
            return (
              <polyline
                key={`curve:${arm}`}
                points={sorted.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
                fill="none"
                stroke={colorForFamily(sorted[0].family)}
                strokeOpacity={hover && !sorted.some((p) => p.key === hover) ? 0.12 : 0.35}
                strokeWidth={1.2}
                strokeDasharray={sorted[0].isNull ? "2 2" : undefined}
              />
            );
          })}

          {/* Pareto frontier */}
          {frontier.length > 1 && (
            <polyline
              points={frontier.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
              fill="none"
              stroke={FRONTIER}
              strokeOpacity={0.75}
              strokeWidth={1.5}
              strokeDasharray="5 3"
            />
          )}

          {/* Points */}
          {points.map((p) => {
            const color = colorForFamily(p.family);
            const cx = sx(p.x);
            const cy = sy(p.y);
            const isHover = p.key === hover;
            return (
              <g key={p.key} opacity={hover && !isHover ? 0.55 : 1}>
                {onFrontier.has(p.key) && (
                  <circle cx={cx} cy={cy} r={R + 3.6} fill="none" stroke={FRONTIER} strokeOpacity={0.45} />
                )}
                <path
                  d={shapePath(shapeFor(p.arm, p.family), cx, cy, isHover ? R + 1.4 : R)}
                  fill={p.isNull ? SURFACE : color}
                  stroke={p.isNull ? color : SURFACE}
                  strokeWidth={p.isNull ? 1.6 : 2}
                  paintOrder="stroke"
                />
              </g>
            );
          })}

          {/* Direct labels on the frontier only (tooltips carry the rest) */}
          {frontier.map((p) => {
            const px = sx(p.x);
            const py = sy(p.y);
            // Flip left near the right edge, below near the top: never clipped.
            const left = px > width - MARGIN.right - 150;
            return (
              <text
                key={`label:${p.key}`}
                x={left ? px - 9 : px + 9}
                y={py - 8 < MARGIN.top + 2 ? py + 15 : py - 8}
                textAnchor={left ? "end" : "start"}
                fontSize={9.5}
                fill={INK_SECONDARY}
                stroke={SURFACE}
                strokeWidth={3}
                paintOrder="stroke"
                className="pointer-events-none"
              >
                {p.title.split(/\s[·(]/)[0]} · {budgetLabel(p.row.budget)}
              </text>
            );
          })}
        </svg>

        {hovered && (
          <div
            className="pointer-events-none absolute z-10 min-w-[170px] rounded-md border border-[#27272a] bg-[#18181b]/95 px-2.5 py-2 shadow-xl backdrop-blur-md"
            style={{
              left: flip ? undefined : tipX + 14,
              right: flip ? width - tipX + 14 : undefined,
              top: Math.max(0, tipY - 18),
            }}
          >
            <div className="flex items-center gap-1.5">
              <ArmMark arm={hovered.arm} family={hovered.family} isNull={hovered.isNull} />
              <span className="text-[11px] font-medium text-[#fafafa]">{hovered.title}</span>
            </div>
            <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[10px] tabular-nums">
              <span className="text-[#71717a]">budget</span>
              <span className="text-right text-[#d4d4d8]">{budgetLabel(hovered.row.budget)}</span>
              {read ? (
                <>
                  <span className="text-[#71717a]">F1 / EM</span>
                  <span className="text-right text-[#d4d4d8]">
                    {formatPts(hovered.row.f1)} / {formatPts(hovered.row.em)}
                  </span>
                  <span className="text-[#71717a]">$ / question</span>
                  <span className="text-right text-[#d4d4d8]">
                    {formatUsd(hovered.row.usd_per_query)}
                  </span>
                  <span className="text-[#71717a]">$ / 100 correct</span>
                  <span className="text-right text-[#d4d4d8]">
                    {formatUsd(hovered.row.usd_per_100_correct)}
                  </span>
                  <span className="text-[#71717a]">tokens / q</span>
                  <span className="text-right text-[#d4d4d8]">
                    {formatTokens(hovered.row.tokens_per_query)}
                  </span>
                </>
              ) : (
                <>
                  <span className="text-[#71717a]">{Y_LABEL[yMetric].split(" (")[0]}</span>
                  <span className="text-right text-[#d4d4d8]">{formatPts(hovered.y)}%</span>
                  <span className="text-[#71717a]">context tokens</span>
                  <span className="text-right text-[#d4d4d8]">{formatTokens(hovered.x)}</span>
                </>
              )}
              {onFrontier.has(hovered.key) && (
                <span className="col-span-2 mt-0.5 font-sans text-[10px] text-[#e4e4e7]">
                  on the Pareto frontier
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend: identity is shape + color + name, never color alone */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[#a1a1aa]">
        {legendArms.map((p) => (
          <span key={p.arm} className="flex items-center gap-1.5">
            <ArmMark arm={p.arm} family={p.family} isNull={p.isNull} />
            {p.title}
          </span>
        ))}
        {floorLo != null && (
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-3.5 rounded-[2px] border-t border-dashed border-[#52525b] bg-[#a1a1aa]/10" />
            evidence floor
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="w-4 border-t-[1.5px] border-dashed border-[#e4e4e7]/75" />
          Pareto frontier
        </span>
      </div>
      {offAxis > 0 && (
        <p className="text-[10px] text-[#52525b]">
          {offAxis} {offAxis === 1 ? "cell" : "cells"} not plotted: no{" "}
          {xAxis === "usd" ? "price" : xAxis === "tokens" ? "token count" : "context"} (a log axis
          has no zero
          {xAxis === "context" ? "; closed-book has no context by construction" : ""}).
        </p>
      )}
      <span className="sr-only">
        {points
          .map((p) => `${p.title} at ${budgetKey(p.row.budget)}: ${formatPts(p.y)}`)
          .join("; ")}
      </span>
    </div>
  );
}
