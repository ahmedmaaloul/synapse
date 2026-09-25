// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import type { CSSProperties, ReactNode } from "react";
import { Loader2 } from "lucide-react";
import {
  LAB_ACTIVE_STATUSES,
  LAB_ARM_SHAPES,
  LAB_STATUS_INFO,
  type LabShape,
  colorForFamily,
} from "../lib/constants";

/** Diagonal hatching that marks an evidence-floor (null-control) row. */
export const FLOOR_HATCH: CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(135deg, #ffffff07 0px, #ffffff07 4px, transparent 4px, transparent 8px)",
};

export function shapeFor(arm: string, family: string | null | undefined): LabShape {
  if (LAB_ARM_SHAPES[arm]) return LAB_ARM_SHAPES[arm];
  // Unknown arms from a newer backend still get a stable, distinct-ish shape.
  const shapes: LabShape[] = ["circle", "square", "triangle", "diamond"];
  let hash = 0;
  for (const ch of `${family}:${arm}`) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return shapes[hash % shapes.length];
}

/** SVG path of a marker of radius-ish `r` centred on (x, y). */
export function shapePath(shape: LabShape, x: number, y: number, r: number): string {
  switch (shape) {
    case "square": {
      const h = r * 0.88;
      return `M${x - h},${y - h}H${x + h}V${y + h}H${x - h}Z`;
    }
    case "triangle": {
      const h = r * 1.15;
      return `M${x},${y - h}L${x + h * 0.95},${y + h * 0.7}H${x - h * 0.95}Z`;
    }
    case "diamond": {
      const h = r * 1.2;
      return `M${x},${y - h}L${x + h},${y}L${x},${y + h}L${x - h},${y}Z`;
    }
    default:
      return `M${x - r},${y}A${r},${r} 0 1,0 ${x + r},${y}A${r},${r} 0 1,0 ${x - r},${y}Z`;
  }
}

/**
 * The mark an arm wears everywhere (picker, tables, plot): family color, a
 * per-arm shape, hollow for the evidence floors.
 */
export function ArmMark({
  arm,
  family,
  isNull,
  size = 10,
}: {
  arm: string;
  family: string | null | undefined;
  isNull: boolean;
  size?: number;
}) {
  const color = colorForFamily(family);
  const c = size / 2;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className="shrink-0"
      aria-hidden="true"
    >
      <path
        d={shapePath(shapeFor(arm, family), c, c, c - 1.4)}
        fill={isNull ? "transparent" : color}
        stroke={color}
        strokeWidth={isNull ? 1.4 : 0}
      />
    </svg>
  );
}

export function SectionLabel({
  children,
  hint,
  aside,
}: {
  children: ReactNode;
  hint?: string;
  aside?: ReactNode;
}) {
  return (
    <div className="mb-1.5 flex items-center gap-2">
      <span
        title={hint}
        className="text-[10px] font-semibold uppercase tracking-wider text-[#71717a]"
      >
        {children}
      </span>
      {aside && <span className="ml-auto flex items-center gap-1.5">{aside}</span>}
    </div>
  );
}

/**
 * A run's status. `stalled`: an in-progress status with no job behind it (the
 * job died with the backend), so no spinner: nothing is moving until a resume.
 */
export function StatusChip({
  status,
  stalled = false,
}: {
  status: string | null | undefined;
  stalled?: boolean;
}) {
  const key = status ?? "created";
  const info = LAB_STATUS_INFO[key] ?? {
    label: key,
    className: "border-[#3f3f46] bg-[#27272a]/60 text-[#a1a1aa]",
  };
  const moving = LAB_ACTIVE_STATUSES.has(key) && !stalled;
  return (
    <span
      title={stalled ? "Stalled: no job is working on this run; resume it" : undefined}
      className={`inline-flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider ${info.className}`}
    >
      {moving && <Loader2 size={9} className="animate-spin" />}
      {info.label}
      {stalled && <span className="normal-case tracking-normal">· stalled</span>}
    </span>
  );
}

/** Small badge, e.g. "floor" or "reasoning"; `caps={false}` for a mono figure. */
export function Tag({
  children,
  className,
  title,
  caps = true,
}: {
  children: ReactNode;
  className: string;
  title?: string;
  caps?: boolean;
}) {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-1 rounded border px-1 py-px ${
        caps
          ? "text-[9px] font-semibold uppercase tracking-wider"
          : "font-mono text-[10px] font-medium"
      } ${className}`}
    >
      {children}
    </span>
  );
}

export const FLOOR_TAG = "border-[#3f3f46] bg-[#27272a]/60 text-[#a1a1aa]";
