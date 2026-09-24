"use client";

import type { LucideIcon } from "lucide-react";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
  /** Tooltip — what switching to this option shows. */
  title?: string;
}

interface SegmentedToggleProps<T extends string> {
  /** Accessible name of the whole control, e.g. "Topology view". */
  label: string;
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
}

/**
 * A compact two-or-more-state switch sized to sit in a panel header next to
 * the existing search box / icon buttons, in the same zinc chrome.
 */
export default function SegmentedToggle<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="group"
      aria-label={label}
      className="flex items-center rounded border border-[#27272a] bg-[#18181b] p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        const Icon = option.icon;
        return (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            aria-pressed={active}
            title={option.title}
            className={`flex items-center gap-1 rounded-[3px] px-2 py-[3px] text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              active
                ? "bg-[#27272a] text-[#fafafa] shadow-sm"
                : "text-[#71717a] hover:text-[#d4d4d8]"
            }`}
          >
            {Icon && (
              <Icon
                size={10}
                strokeWidth={2.5}
                className={active ? "text-indigo-400" : ""}
              />
            )}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
