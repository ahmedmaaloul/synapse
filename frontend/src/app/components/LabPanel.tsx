// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
// Synapse — https://github.com/ahmedmaaloul/synapse
"use client";

import {
  type ChangeEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Calculator,
  ChevronDown,
  FlaskConical,
  History,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  RefreshCw,
  ScanSearch,
  SlidersHorizontal,
  TriangleAlert,
  Trophy,
  Upload,
  Zap,
} from "lucide-react";
import {
  ApiError,
  LabRefusedError,
  estimateLab,
  fetchLabArms,
  fetchLabDatasets,
  fetchLabModels,
  fetchLabRun,
  fetchLabRuns,
  resumeLabRun,
  startLabRun,
  subscribeLabRun,
  uploadLabQaFile,
} from "../lib/api";
import {
  LAB_ACTIVE_STATUSES,
  LAB_BATCH_MULTIPLIER,
  LAB_BUDGETS,
  LAB_DEFAULT_BUDGETS,
  LAB_DEFAULT_K,
  LAB_DEFAULT_MAX_USD,
  LAB_DEFAULT_READER,
  LAB_DEFAULT_SEED,
  LAB_MODE_INFO,
  LAB_READER_MODELS,
} from "../lib/constants";
import {
  baseName,
  budgetKey,
  budgetLabel,
  cellKey,
  formatUsd,
  formatWhen,
  isReasoningModel,
  sortBudgets,
} from "../lib/lab";
import type {
  LabArm,
  LabBudget,
  LabDatasetInfo,
  LabEstimate,
  LabEvent,
  LabJobSummary,
  LabMode,
  LabReaderModel,
  LabRunDetail,
  LabRunRequest,
  LabRunSummary,
} from "../lib/types";
import LabArmPicker from "./LabArmPicker";
import LabEstimateView from "./LabEstimateView";
import LabLeaderboard from "./LabLeaderboard";
import LabPareto from "./LabPareto";
import LabQuestions from "./LabQuestions";
import LabRunsList from "./LabRunsList";
import { SectionLabel, StatusChip, Tag } from "./LabShared";
import SegmentedToggle, { type SegmentedOption } from "./SegmentedToggle";

interface LabPanelProps {
  /** The Knowledge | Procedures | Lab switch, rendered in this view's header. */
  viewToggle: ReactNode;
  /** The Lab covers the chat panel too (the chat stays mounted underneath). */
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

/** "setup" only exists in the narrow, one-column layout. */
type Tab = "setup" | "estimate" | "results" | "runs";

/** Below this width the configuration column becomes a tab. */
const NARROW_WIDTH = 880;

const SETUP_OPTION: SegmentedOption<Tab> = {
  value: "setup",
  label: "Setup",
  icon: SlidersHorizontal,
  title: "Arms, dataset, budgets, reader, mode and cap",
};
type CatalogStatus = "loading" | "ready" | "error";

/** Progress of the run this view is following over SSE. */
interface LiveRun {
  runId: string;
  phase: string | null;
  status: string;
  done: number;
  total: number;
  spentUsd: number | null;
  message: string | null;
  tone: "info" | "ok" | "warn" | "error";
  finished: boolean;
}

const MODE_OPTIONS: SegmentedOption<LabMode>[] = [
  {
    value: "retrieve",
    label: "Retrieve",
    icon: ScanSearch,
    title: "Retrieval only: no reader, no LLM, $0",
  },
  {
    value: "realtime",
    label: "Realtime",
    icon: Zap,
    title: "Read now, metered against the cap",
  },
  {
    value: "batch",
    label: "Batch",
    icon: Layers,
    title: "OpenAI Batch API: half price, within 24h",
  },
];

/** The demo set always exists: the view still works against a router that lists nothing. */
const DEMO_DATASET: LabDatasetInfo = {
  key: "demo",
  dataset: "demo",
  file: null,
  label: "Demo questions",
  description: null,
  splits: [],
  splitCounts: {},
  defaultSplit: "test",
  n: null,
  defaultN: null,
  available: true,
  reason: null,
};

const DEMO_SPLITS = ["test", "val", "train", "all"];

const INPUT =
  "w-full rounded border border-[#27272a] bg-[#18181b] px-2 py-1 font-mono text-[11px] text-[#fafafa] outline-none transition-colors placeholder:text-[#52525b] hover:border-[#3f3f46] focus:border-indigo-500/50 disabled:opacity-50";
const SELECT = `${INPUT} appearance-none pr-6`;

function describeError(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    if (err.status === 404) {
      return err.detail || "Not found. Does this backend have the Lab router (/api/lab)?";
    }
    return err.detail || `${fallback} (HTTP ${err.status})`;
  }
  if (err instanceof TypeError) return "Connection refused. Is the backend running?";
  return fallback;
}

/** How a finished job reads: its final status decides the tone. */
function toneFor(status: string): LiveRun["tone"] {
  if (status === "refused" || status === "failed") return "error";
  if (status === "aborted" || status === "batch_submitted") return "warn";
  return "ok";
}

function doneMessage(summary: LabJobSummary | undefined, status: string): string | null {
  if (summary?.reason) return summary.reason;
  if (status === "batch_submitted") {
    return "Submitted to the Batch API (half price, up to 24h). Use Check batch to collect the answers.";
  }
  if (status === "done" && summary?.spent_usd != null && summary.spent_usd > 0) {
    return `Scored. Reader spend ${formatUsd(summary.spent_usd)}${
      summary.cap_usd != null ? ` of a ${formatUsd(summary.cap_usd)} cap` : ""
    }.`;
  }
  return null;
}

function positiveInt(text: string): number | null {
  if (!text.trim()) return null;
  const v = Number(text);
  return Number.isInteger(v) && v > 0 ? v : NaN;
}

function describeBatchStatus(status: unknown): string {
  if (status && typeof status === "object") {
    const s = status as Record<string, unknown>;
    const counts = s.request_counts as Record<string, unknown> | undefined;
    const parts = [
      typeof s.status === "string" ? s.status : "",
      counts && typeof counts.completed === "number" && typeof counts.total === "number"
        ? `${counts.completed}/${counts.total} requests done`
        : "",
      counts && typeof counts.failed === "number" && counts.failed > 0
        ? `${counts.failed} failed`
        : "",
    ].filter(Boolean);
    if (parts.length) return `Batch: ${parts.join(" · ")}`;
  }
  return typeof status === "string" ? `Batch: ${status}` : "Batch polled.";
}

function Select({
  value,
  onChange,
  label,
  disabled,
  children,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="relative min-w-0">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        disabled={disabled}
        className={SELECT}
      >
        {children}
      </select>
      <ChevronDown
        size={11}
        className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[#71717a]"
      />
    </div>
  );
}

/**
 * Synapse Lab: compare retrieval approaches on your own questions, FinOps
 * first. Configure → Estimate (free) → Run (only when the estimate fits the
 * cap) → leaderboard + Pareto frontier, always next to the evidence floors.
 */
export default function LabPanel({ viewToggle, expanded, onExpandedChange }: LabPanelProps) {
  // ── Catalog ──
  const [arms, setArms] = useState<LabArm[]>([]);
  const [models, setModels] = useState<LabReaderModel[]>(LAB_READER_MODELS);
  const [batchAvailable, setBatchAvailable] = useState(true);
  const [datasets, setDatasets] = useState<LabDatasetInfo[]>([DEMO_DATASET]);
  const [catalogStatus, setCatalogStatus] = useState<CatalogStatus>("loading");
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogNonce, setCatalogNonce] = useState(0);

  // ── Configuration (retrieve-only, the $0 tier, is the default) ──
  const [pickedArms, setPickedArms] = useState<string[] | null>(null); // null → all
  const [datasetKey, setDatasetKey] = useState("demo");
  const [splitChoice, setSplitChoice] = useState<string | null>(null); // null → default
  const [nText, setNText] = useState("");
  const [budgets, setBudgets] = useState<LabBudget[]>(LAB_DEFAULT_BUDGETS);
  const [readerModel, setReaderModel] = useState(LAB_DEFAULT_READER);
  const [mode, setMode] = useState<LabMode>("retrieve");
  const [maxUsdText, setMaxUsdText] = useState(LAB_DEFAULT_MAX_USD.toFixed(2));
  const [kText, setKText] = useState(String(LAB_DEFAULT_K));
  const [seedText, setSeedText] = useState(String(LAB_DEFAULT_SEED));
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Price a paid run on the MEASURED contexts of a finished retrieve-only run.
  const [measuredRunId, setMeasuredRunId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // ── Estimate ──
  const [estimate, setEstimate] = useState<{ key: string; data: LabEstimate } | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState<string | null>(null);

  // ── Runs ──
  // null until the user picks: Setup first in the narrow layout, Estimate otherwise.
  const [tab, setTab] = useState<Tab | null>(null);
  const [runs, setRuns] = useState<LabRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<{ id: string; data: LabRunDetail } | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailNonce, setDetailNonce] = useState(0);
  const [live, setLive] = useState<LiveRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  // A paid action waiting for the user's yes: a new run, or resuming a stopped one.
  const [confirm, setConfirm] = useState<
    { kind: "run" } | { kind: "resume"; runId: string; pending: LabRunDetail } | null
  >(null);
  const [resuming, setResuming] = useState(false);
  const [cell, setCell] = useState<{ arm: string; budget: LabBudget } | null>(null);

  const [panelWidth, setPanelWidth] = useState(0);

  const rootRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<AbortController | null>(null);
  const estimateRef = useRef<AbortController | null>(null);
  const unmountedRef = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);

  // Stop following / estimating when the view goes away.
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      streamRef.current?.abort();
      estimateRef.current?.abort();
    };
  }, []);

  // ── Track the panel's width: two columns, or one with a Setup tab ──
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setPanelWidth(w); // 0 while hidden (another view is showing): keep the last
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Load the arm catalog + datasets ──
  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    Promise.all([
      fetchLabArms(signal),
      // The demo set needs no listing: a failing datasets call is not fatal.
      fetchLabDatasets(signal).catch((err) => {
        if (!signal.aborted) console.error("Failed to fetch Lab datasets:", err);
        return [] as LabDatasetInfo[];
      }),
      // Neither is the model list: the bundled mirror of the price table stands in.
      fetchLabModels(signal).catch((err) => {
        if (!signal.aborted) console.error("Failed to fetch Lab reader models:", err);
        return [] as LabReaderModel[];
      }),
    ])
      .then(([catalog, datasetList, modelList]) => {
        if (signal.aborted) return;
        setArms(catalog.arms);
        setBatchAvailable(catalog.batchAvailable);
        setModels(modelList.length > 0 ? modelList : LAB_READER_MODELS);
        setDatasets(
          datasetList.some((d) => d.dataset === "demo") ? datasetList : [DEMO_DATASET, ...datasetList],
        );
        setCatalogError(null);
        setCatalogStatus("ready");
      })
      .catch((err) => {
        if (signal.aborted) return;
        console.error("Failed to load the Lab catalog:", err);
        setCatalogError(describeError(err, "Could not load the Lab."));
        setCatalogStatus("error");
      });
    return () => controller.abort();
  }, [catalogNonce]);

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const list = await fetchLabRuns();
      if (unmountedRef.current) return;
      setRuns(list);
      setRunsError(null);
    } catch (err) {
      if (unmountedRef.current) return;
      setRunsError(describeError(err, "Could not list the runs."));
    } finally {
      if (!unmountedRef.current) setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchLabRuns(controller.signal)
      .then((list) => {
        setRuns(list);
        setRunsError(null);
      })
      .catch((err) => {
        if (!controller.signal.aborted) {
          setRunsError(describeError(err, "Could not list the runs."));
        }
      });
    return () => controller.abort();
  }, []);

  // ── The selected run's manifest + leaderboard ──
  useEffect(() => {
    if (!selectedRunId) return;
    const controller = new AbortController();
    // limit 0: the manifest and leaderboard only; questions load on demand.
    fetchLabRun(selectedRunId, { limit: 0 }, controller.signal)
      .then((data) => {
        setDetail({ id: selectedRunId, data });
        setDetailError(null);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setDetailError(describeError(err, "Could not load the run."));
      });
    return () => controller.abort();
  }, [selectedRunId, detailNonce]);

  const shown = detail && detail.id === selectedRunId ? detail.data : null;
  const shownStatus = shown?.manifest.status ?? null;
  const shownActive = shown?.active ?? false;
  const streaming = live != null && !live.finished;
  const followingShown = streaming && live?.runId === selectedRunId;

  // A run with a job in flight server-side that this view is not streaming
  // (opened from the list, or after a reload): poll it until the job ends. Only
  // `active` counts: a run left mid-phase by a dead job (a backend restart)
  // keeps an in-progress status forever, and is offered a Resume instead.
  useEffect(() => {
    if (!selectedRunId || followingShown || !shownActive) return;
    const timer = setInterval(() => {
      setDetailNonce((n) => n + 1);
      void refreshRuns();
    }, 4000);
    return () => clearInterval(timer);
  }, [selectedRunId, followingShown, shownActive, refreshRuns]);
  // In-progress status, but no job behind it: the job died with the backend.
  const shownStale =
    shown != null &&
    !shownActive &&
    !followingShown &&
    shownStatus != null &&
    LAB_ACTIVE_STATUSES.has(shownStatus);
  const shownMode = typeof shown?.manifest.config?.mode === "string" ? shown.manifest.config.mode : null;

  // ── Derived configuration ──
  const armsByName = useMemo(() => new Map(arms.map((a) => [a.name, a])), [arms]);
  const chosenArms = useMemo(
    () => (pickedArms ?? arms.map((a) => a.name)).filter((name) => armsByName.has(name)),
    [pickedArms, arms, armsByName],
  );
  const dataset =
    datasets.find((d) => d.key === datasetKey) ??
    datasets.find((d) => d.dataset === "demo") ??
    DEMO_DATASET;
  const isHotpot = dataset.dataset === "hotpotqa";
  const isQaFile = dataset.dataset === "qa-file";
  // "all" is always on offer (the runner reads it as "no split filter"); the
  // router lists only the splits the file actually has.
  const namedSplits = (dataset.splits.length > 0 || isQaFile ? dataset.splits : DEMO_SPLITS).filter(
    (s) => s !== "all",
  );
  const splitOptions = isHotpot
    ? []
    : isQaFile
      ? ["all", ...namedSplits]
      : [...namedSplits, "all"];
  const countedSplits = Object.values(dataset.splitCounts);
  const splitCount = (s: string): number | null =>
    dataset.splitCounts[s] ??
    (s === "all" && countedSplits.length > 0 ? countedSplits.reduce((a, b) => a + b, 0) : null);
  const defaultSplit =
    dataset.defaultSplit && splitOptions.includes(dataset.defaultSplit)
      ? dataset.defaultSplit
      : isQaFile
        ? "all"
        : (splitOptions[0] ?? null);
  const split =
    splitChoice && splitOptions.includes(splitChoice) ? splitChoice : defaultSplit;

  const nValue = positiveInt(nText);
  const kValue = positiveInt(kText);
  const seedNumber = seedText.trim() === "" ? LAB_DEFAULT_SEED : Number(seedText);
  const seedValue = Number.isInteger(seedNumber) && seedNumber >= 0 ? seedNumber : NaN;
  const maxUsdNumber = maxUsdText.trim() === "" ? null : Number(maxUsdText);
  const maxUsdValid = maxUsdNumber == null || (Number.isFinite(maxUsdNumber) && maxUsdNumber >= 0);
  const paid = mode !== "retrieve";
  const reader = models.find((m) => m.id === readerModel);
  // Only a finished retrieve-only run has measured contexts to price on.
  const measuredCandidates = runs.filter((r) => r.mode === "retrieve" && r.status === "done");
  const measured = measuredCandidates.some((r) => r.run_id === measuredRunId)
    ? measuredRunId
    : null;

  const blocker: string | null =
    catalogStatus !== "ready"
      ? "The Lab catalog is not loaded."
      : chosenArms.length === 0
        ? "Pick at least one arm."
        : budgets.length === 0
          ? "Pick at least one budget."
          : !dataset.available
            ? (dataset.reason ?? `${dataset.label} is not available.`)
            : Number.isNaN(nValue)
              ? "n must be a positive whole number (or empty for all)."
              : kValue == null || Number.isNaN(kValue)
                ? "k must be a positive whole number."
                : Number.isNaN(seedValue)
                  ? "The seed must be a whole number."
                  : !maxUsdValid
                    ? "The cap must be a dollar amount."
                    : paid && (maxUsdNumber == null || maxUsdNumber <= 0)
                      ? "A paid run needs a spend cap above $0."
                      : null;

  const request: LabRunRequest | null = useMemo(() => {
    if (blocker) return null;
    const body: LabRunRequest = {
      dataset: dataset.dataset,
      split: isHotpot ? null : split,
      n: nValue,
      arms: chosenArms,
      budgets: sortBudgets(budgets),
      k: kValue ?? LAB_DEFAULT_K,
      reader_model: readerModel,
      mode,
      max_usd: maxUsdNumber,
      // Part of the estimate too: a measured run prices this one only on the same seed.
      seed: seedValue,
    };
    if (dataset.file) body.file = dataset.file;
    if (measured && paid) body.measured_run_id = measured;
    return body;
  }, [
    blocker,
    dataset.dataset,
    dataset.file,
    isHotpot,
    split,
    nValue,
    chosenArms,
    budgets,
    kValue,
    readerModel,
    mode,
    maxUsdNumber,
    seedValue,
    measured,
    paid,
  ]);
  const requestKey = request ? JSON.stringify(request) : null;

  const estimateFresh = estimate != null && estimate.key === requestKey;
  const runBlocker: string | null =
    blocker ??
    (!estimate
      ? "Estimate first: Run unlocks after an estimate that fits the cap."
      : !estimateFresh
        ? "The configuration changed since the estimate: estimate again."
        : estimate.data.refuse
          ? "Refused: the upper bound is over the cap."
          : mode === "batch" && !batchAvailable
            ? "Batch mode is not available in this build (no Batch layer): use Realtime or Retrieve."
            : streaming
              ? "A run is in progress."
              : null);

  // ── Actions ──
  const runEstimate = async () => {
    if (!request || !requestKey) return;
    estimateRef.current?.abort();
    const controller = new AbortController();
    estimateRef.current = controller;
    setEstimating(true);
    setEstimateError(null);
    setRunError(null);
    setTab("estimate");
    try {
      const data = await estimateLab(request, controller.signal);
      setEstimate({ key: requestKey, data });
    } catch (err) {
      if (controller.signal.aborted) return;
      if (err instanceof LabRefusedError && err.estimate) {
        setEstimate({ key: requestKey, data: err.estimate });
      } else {
        setEstimateError(describeError(err, "The estimate failed."));
      }
    } finally {
      if (!controller.signal.aborted) setEstimating(false);
    }
  };

  const follow = useCallback(
    (runId: string, jobId: string | null) => {
      streamRef.current?.abort();
      const controller = new AbortController();
      streamRef.current = controller;
      setLive({
        runId,
        phase: null,
        status: "starting",
        done: 0,
        total: 0,
        spentUsd: null,
        message: null,
        tone: "info",
        finished: false,
      });
      const patch = (fn: (run: LiveRun) => LiveRun) =>
        setLive((prev) => (prev && prev.runId === runId ? fn(prev) : prev));

      const onEvent = (event: LabEvent) => {
        switch (event.type) {
          case "phase":
            patch((l) => ({
              ...l,
              phase: event.phase,
              status: event.status,
              done: event.status === "running" ? 0 : l.done,
              total: event.total ?? l.total,
            }));
            break;
          case "progress":
            patch((l) => ({
              ...l,
              phase: event.phase,
              status: "running",
              done: event.done,
              total: event.total,
              spentUsd: event.spent_usd ?? l.spentUsd,
            }));
            break;
          case "refused":
            // The measured re-estimate broke the cap before any read: $0 spent.
            patch((l) => ({ ...l, status: "refused", message: event.reason, tone: "error" }));
            break;
          case "batch_submitted":
            patch((l) => ({
              ...l,
              status: "batch_submitted",
              message: `${event.requests.toLocaleString()} requests submitted to the Batch API (half price, up to 24h). Use Check batch to collect the answers.`,
              tone: "warn",
            }));
            break;
          case "batch_status":
            patch((l) => ({ ...l, message: describeBatchStatus(event.status) }));
            break;
          case "aborted":
            patch((l) => ({ ...l, status: "aborted", message: event.reason, tone: "warn" }));
            break;
          case "done": {
            // The job's own ending: `data` is the run's final state.
            const summary = event.data;
            const status = summary?.status ?? event.status ?? "done";
            patch((l) => ({
              ...l,
              status,
              // A poll that ends with the batch still out keeps its progress line.
              message:
                status === "batch_submitted"
                  ? (l.message ?? doneMessage(summary, status))
                  : (doneMessage(summary, status) ?? l.message),
              spentUsd: summary?.spent_usd ?? l.spentUsd,
              tone: toneFor(status),
              finished: true,
            }));
            break;
          }
          case "error":
            patch((l) => ({
              ...l,
              status: "failed",
              message: event.error ?? event.data ?? "The run failed.",
              tone: "error",
              finished: true,
            }));
            break;
        }
      };

      subscribeLabRun([runId, jobId], onEvent, controller.signal)
        .catch((err) => {
          if (controller.signal.aborted) return;
          patch((l) => ({
            ...l,
            message: describeError(err, "Lost the progress stream."),
            tone: "error",
          }));
        })
        .finally(() => {
          if (streamRef.current === controller) streamRef.current = null;
          if (unmountedRef.current) return;
          patch((l) => ({ ...l, finished: true }));
          void refreshRuns();
          setDetailNonce((n) => n + 1);
        });
    },
    [refreshRuns],
  );

  const startRun = async () => {
    if (!request || !requestKey || runBlocker) return;
    setConfirm(null);
    setStarting(true);
    setRunError(null);
    try {
      const res = await startLabRun(request);
      if (res.estimate) setEstimate({ key: requestKey, data: res.estimate });
      const runId = res.run_id || res.job_id;
      if (!runId) throw new Error("The backend did not return a run id.");
      setSelectedRunId(runId);
      setCell(null);
      setTab("results");
      void refreshRuns();
      if (res.status === "batch_submitted" && !res.job_id) {
        setDetailNonce((n) => n + 1);
        return;
      }
      follow(runId, res.job_id);
    } catch (err) {
      if (err instanceof LabRefusedError) {
        if (err.estimate) setEstimate({ key: requestKey, data: err.estimate });
        else setRunError(err.detail || "Refused: the upper bound is over the cap.");
        setTab("estimate");
      } else {
        setRunError(
          err instanceof ApiError || err instanceof TypeError || !(err instanceof Error)
            ? describeError(err, "Could not start the run.")
            : err.message || "Could not start the run.",
        );
      }
    } finally {
      setStarting(false);
    }
  };

  const onRunClick = () => {
    if (runBlocker || starting) return;
    // Retrieval is free; anything that reads asks once, with the bound.
    if (paid) setConfirm({ kind: "run" });
    else void startRun();
  };

  /**
   * Continue a run. A submitted batch is only polled / collected (no new
   * spend); an aborted or refused run reads what is left under `maxUsd`.
   */
  const resume = async (runId: string, maxUsd?: number | null) => {
    setConfirm(null);
    setResuming(true);
    setRunError(null);
    try {
      const res = await resumeLabRun(runId, { max_usd: maxUsd ?? undefined });
      if (res.job_id) follow(runId, res.job_id);
      else setDetailNonce((n) => n + 1);
      void refreshRuns();
    } catch (err) {
      setRunError(describeError(err, "Could not resume the run."));
    } finally {
      setResuming(false);
    }
  };

  /** Load a stored run's configuration into the form. */
  const applyRunConfig = (d: LabRunDetail) => {
    const cfg = d.manifest.config ?? {};
    const ds = cfg.dataset;
    const spec = ds && typeof ds === "object" ? ds : null;
    const name = typeof ds === "string" ? ds : spec?.name;
    if (name === "qa-file" && spec?.path) setDatasetKey(`qa-file:${baseName(spec.path)}`);
    else if (name) setDatasetKey(name);
    setSplitChoice(spec?.split ?? null);
    setNText(spec?.n != null ? String(spec.n) : "");
    if (Array.isArray(cfg.arms)) setPickedArms(cfg.arms.filter((a) => armsByName.has(a)));
    if (Array.isArray(cfg.budgets) && cfg.budgets.length > 0) {
      setBudgets(cfg.budgets.map((b) => (typeof b === "number" ? b : null)));
    }
    if (typeof cfg.k === "number") setKText(String(cfg.k));
    if (typeof cfg.seed === "number") setSeedText(String(cfg.seed));
  };

  /**
   * The FinOps loop: a free retrieve-only run measured every context, so a
   * paid run of the same configuration can be priced on real sizes rather
   * than on "every context fills its budget".
   */
  const priceOnMeasured = (d: LabRunDetail) => {
    applyRunConfig(d);
    setMeasuredRunId(d.run_id);
    if (mode === "retrieve") setMode("realtime");
    setAdvancedOpen(true);
    setEstimate(null);
    setEstimateError(null);
    setTab("estimate");
  };

  const openRun = (runId: string) => {
    setSelectedRunId(runId);
    setCell(null);
    setDetailError(null);
    setTab("results");
    setDetailNonce((n) => n + 1);
  };

  const onUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // the same file can be picked again after a fix
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const uploaded = await uploadLabQaFile(file);
      const list = await fetchLabDatasets().catch(() => null);
      const entry: LabDatasetInfo = {
        key: `qa-file:${uploaded.file}`,
        dataset: "qa-file",
        file: uploaded.file,
        label: uploaded.file,
        description: null,
        splits: [],
        splitCounts: {},
        defaultSplit: null,
        n: uploaded.n,
        defaultN: null,
        available: true,
        reason: null,
      };
      setDatasets((prev) => {
        const base = list && list.length > 0 ? list : prev;
        const withDemo = base.some((d) => d.dataset === "demo") ? base : [DEMO_DATASET, ...base];
        return withDemo.some((d) => d.file && baseName(d.file) === uploaded.file)
          ? withDemo
          : [...withDemo, entry];
      });
      setDatasetKey(`qa-file:${uploaded.file}`);
      setSplitChoice(null);
    } catch (err) {
      setUploadError(describeError(err, "Upload failed."));
    } finally {
      setUploading(false);
    }
  };

  const toggleBudget = (budget: LabBudget) =>
    setBudgets((prev) =>
      prev.some((b) => budgetKey(b) === budgetKey(budget))
        ? prev.filter((b) => budgetKey(b) !== budgetKey(budget))
        : [...prev, budget],
    );

  // ── Rollback-style dialog: Escape cancels, focus starts on "Cancel" ──
  const confirmOpen = confirm != null;
  useEffect(() => {
    if (!confirmOpen) return;
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setConfirm(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirmOpen]);

  const tabOptions: SegmentedOption<Tab>[] = [
    { value: "estimate", label: "Estimate", icon: Calculator, title: "What the run would cost" },
    { value: "results", label: "Results", icon: Trophy, title: "Leaderboard and Pareto frontier" },
    {
      value: "runs",
      label: runs.length > 0 ? `Runs · ${runs.length}` : "Runs",
      icon: History,
      title: "Every stored run",
    },
  ];

  const narrow = panelWidth > 0 && panelWidth < NARROW_WIDTH;
  const activeTab: Tab =
    tab == null ? (narrow ? "setup" : "estimate") : !narrow && tab === "setup" ? "estimate" : tab;

  const board = shown?.leaderboard ?? null;
  const cellArm = cell ? armsByName.get(cell.arm) : undefined;
  const liveHere = live && live.runId === selectedRunId ? live : null;
  const pct =
    liveHere && liveHere.total > 0 ? Math.min(100, (liveHere.done / liveHere.total) * 100) : null;
  const liveStatus =
    liveHere && !liveHere.finished
      ? liveHere.phase === "read"
        ? "reading"
        : "retrieving"
      : null;

  // ── Blocks shared by the wide (two-column) and narrow (tabbed) layouts ──
  const configBody = (
    <>
      <section className="mb-4">
        <SectionLabel
          hint="Every arm returns ranked evidence units; one shared packer fills each budget"
          aside={
            <span className="font-mono text-[10px] text-[#52525b]">
              {chosenArms.length}/{arms.length}
            </span>
          }
        >
          Arms
        </SectionLabel>
        {arms.length === 0 ? (
          <p className="text-[11px] text-[#52525b]">The backend lists no arms.</p>
        ) : (
          <LabArmPicker arms={arms} selected={chosenArms} onChange={setPickedArms} />
        )}
      </section>

      <section className="mb-4">
        <SectionLabel hint="The questions every arm is scored on">Dataset</SectionLabel>
        <Select
          value={dataset.key}
          onChange={(key) => {
            setDatasetKey(key);
            setSplitChoice(null);
          }}
          label="Dataset"
        >
          {datasets.map((d) => (
            <option key={d.key} value={d.key} disabled={!d.available}>
              {d.label}
              {d.n != null ? ` · ${d.n} q` : ""}
              {!d.available ? " (unavailable)" : ""}
            </option>
          ))}
        </Select>
        {(dataset.description || dataset.reason) && (
          <p
            className={`mt-1 text-[10px] leading-snug ${
              dataset.available ? "text-[#52525b]" : "text-amber-300/80"
            }`}
          >
            {dataset.available ? dataset.description : dataset.reason}
          </p>
        )}
        {/* A disabled option cannot be picked, so say here why it is disabled. */}
        {datasets
          .filter((d) => !d.available && d.key !== dataset.key)
          .map((d) => (
            <p key={d.key} className="mt-1 text-[10px] leading-snug text-[#52525b]">
              <span className="text-[#71717a]">{d.label}</span> unavailable
              {d.reason ? `: ${d.reason}` : ""}
            </p>
          ))}
        {isHotpot && (
          <p className="mt-1 text-[10px] leading-snug text-[#52525b]">
            Needs its corpus graph already ingested; also reports strict / permissive
            gold-paragraph recall per arm.
          </p>
        )}
        <div className="mt-1.5 grid grid-cols-2 gap-1.5">
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] text-[#52525b]">Split</span>
            {isHotpot ? (
              <span className="rounded border border-[#27272a] bg-[#18181b]/50 px-2 py-1 font-mono text-[11px] text-[#71717a]">
                dev · distractor
              </span>
            ) : (
              <Select
                value={split ?? ""}
                onChange={setSplitChoice}
                label="Split"
              >
                {splitOptions.map((s) => (
                  <option key={s} value={s}>
                    {s}
                    {splitCount(s) != null ? ` · ${splitCount(s)}` : ""}
                  </option>
                ))}
              </Select>
            )}
          </label>
          <label className="flex flex-col gap-0.5">
            <span className="text-[10px] text-[#52525b]">
              n <span className="text-[#3f3f46]">(questions)</span>
            </span>
            <input
              value={nText}
              onChange={(e) => setNText(e.target.value)}
              inputMode="numeric"
              placeholder={isHotpot ? String(dataset.defaultN ?? 20) : "all"}
              aria-label="Number of questions"
              className={INPUT}
            />
          </label>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".json,.jsonl,application/json"
          onChange={onUpload}
          className="hidden"
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          title="A JSON array or JSONL of {question, answer[, id][, split]}"
          className="mt-1.5 flex w-full items-center justify-center gap-1.5 rounded border border-dashed border-[#27272a] px-2 py-1.5 text-[11px] font-medium text-[#71717a] transition-colors hover:border-indigo-500/40 hover:text-[#d4d4d8] disabled:opacity-50"
        >
          {uploading ? (
            <Loader2 size={11} className="animate-spin" />
          ) : (
            <Upload size={11} />
          )}
          Upload your QA file (.json / .jsonl)
        </button>
        {uploadError && (
          <p className="mt-1 break-words text-[10px] leading-snug text-rose-300/80">
            {uploadError}
          </p>
        )}
      </section>

      <section className="mb-4">
        <SectionLabel hint="Context tokens each arm may use; 'default' is the arm's own uncapped context">
          Token budgets
        </SectionLabel>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Token budgets">
          {LAB_BUDGETS.map((b) => {
            const on = budgets.some((x) => budgetKey(x) === budgetKey(b));
            return (
              <button
                key={budgetKey(b)}
                onClick={() => toggleBudget(b)}
                aria-pressed={on}
                title={
                  b == null
                    ? "Each arm's default, uncapped context (not bounded before retrieval)"
                    : `${b.toLocaleString()} context tokens`
                }
                className={`rounded border px-2 py-[3px] font-mono text-[11px] transition-colors ${
                  on
                    ? "border-indigo-500/50 bg-indigo-500/15 text-indigo-200"
                    : "border-[#27272a] bg-[#18181b] text-[#71717a] hover:border-[#3f3f46] hover:text-[#d4d4d8]"
                }`}
              >
                {budgetLabel(b)}
              </button>
            );
          })}
        </div>
      </section>

      <section className="mb-4">
        <SectionLabel hint="How far the run goes: retrieval only ($0), or retrieval then one reader call per cell">
          Mode
        </SectionLabel>
        <div className="flex">
          <SegmentedToggle label="Run mode" value={mode} options={MODE_OPTIONS} onChange={setMode} />
        </div>
        <p className="mt-1.5 text-[10px] leading-snug text-[#52525b]">
          {LAB_MODE_INFO[mode].hint}
        </p>
        {mode === "batch" && !batchAvailable && (
          <p className="mt-1 text-[10px] leading-snug text-amber-300/80">
            This backend has no Batch layer: batch runs cannot start (estimates still
            work).
          </p>
        )}
      </section>

      <section className={`mb-4 transition-opacity ${paid ? "" : "opacity-60"}`}>
        <SectionLabel
          hint="One short-answer prompt for every arm: answer from the evidence only"
          aside={
            !paid ? (
              <span className="text-[10px] text-[#52525b]">unused at $0</span>
            ) : null
          }
        >
          Reader model
        </SectionLabel>
        <Select value={readerModel} onChange={setReaderModel} label="Reader model">
          {!reader && <option value={readerModel}>{readerModel} · unpriced</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id} · ${m.input} / ${m.output}
              {m.reasoning ? " · reasoning" : ""}
            </option>
          ))}
        </Select>
        <div className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-[#71717a]">
          {reader ? (
            <>
              <span title="USD per 1M tokens, standard rate">
                ${reader.input} in · ${reader.output} out / 1M
              </span>
              {mode === "batch" && (
                <span className="text-amber-300/80">×{LAB_BATCH_MULTIPLIER} batch</span>
              )}
              {reader.checkedOn && (
                <span className="text-[#52525b]" title="Hand-recorded, not fetched live">
                  checked {reader.checkedOn}
                </span>
              )}
            </>
          ) : (
            <span>no price on file: the estimate will refuse</span>
          )}
          {isReasoningModel(readerModel) && (
            <Tag
              title="Hidden reasoning tokens are billed as output: the upper bound allows for them"
              className="border-violet-400/30 bg-violet-400/10 text-violet-300/90"
            >
              reasoning model
            </Tag>
          )}
        </div>
      </section>

      <section className={`mb-3 transition-opacity ${paid ? "" : "opacity-60"}`}>
        <SectionLabel hint="Hard spend cap: refused up front if the upper bound exceeds it, and metered during realtime reads">
          Max spend
        </SectionLabel>
        <div className="relative">
          <span className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 font-mono text-[11px] text-[#71717a]">
            $
          </span>
          <input
            value={maxUsdText}
            onChange={(e) => setMaxUsdText(e.target.value)}
            inputMode="decimal"
            aria-label="Maximum spend in USD"
            className={`${INPUT} pl-5`}
          />
        </div>
      </section>

      <section>
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={advancedOpen}
          className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[#52525b] transition-colors hover:text-[#a1a1aa]"
        >
          <ChevronDown
            size={11}
            className={`transition-transform ${advancedOpen ? "" : "-rotate-90"}`}
          />
          Advanced
        </button>
        {advancedOpen && (
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            <label className="flex flex-col gap-0.5">
              <span
                className="text-[10px] text-[#52525b]"
                title="Seed entities for graph arms; passages at the default context for passage arms (under a token budget they retrieve as many passages as the budget holds)"
              >
                k
              </span>
              <input
                value={kText}
                onChange={(e) => setKText(e.target.value)}
                inputMode="numeric"
                aria-label="k"
                className={INPUT}
              />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-[#52525b]" title="Seeds the random-context null (N2); recorded in the manifest">
                Seed
              </span>
              <input
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                inputMode="numeric"
                aria-label="Seed"
                className={INPUT}
              />
            </label>
            <label className="col-span-2 flex flex-col gap-0.5">
              <span
                className="text-[10px] text-[#52525b]"
                title="Price a paid run on the real context sizes a finished retrieve-only run measured (same questions, k, seed and tokenizer), instead of assuming every context fills its budget"
              >
                Price on measured contexts{" "}
                <span className="text-[#3f3f46]">(paid modes)</span>
              </span>
              <Select
                value={measured ?? ""}
                onChange={setMeasuredRunId}
                label="Price on the measured contexts of"
                disabled={!paid || measuredCandidates.length === 0}
              >
                <option value="">
                  {measuredCandidates.length === 0
                    ? "no finished retrieve-only run"
                    : "budget-bound (no run)"}
                </option>
                {measuredCandidates.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id}
                    {r.dataset ? ` · ${r.dataset}` : ""}
                    {r.n != null ? ` n=${r.n}` : ""}
                  </option>
                ))}
              </Select>
            </label>
          </div>
        )}
      </section>
    </>
  );

  // Estimate / Run — always in view next to the configuration.
  const actionsBar = (
    <div className="border-t border-[#27272a] px-3 py-2.5">
      <div className="flex gap-1.5">
        <button
          onClick={runEstimate}
          disabled={!request || estimating}
          title={blocker ?? "Price the run: free, no model is called"}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-[#27272a] bg-[#18181b] px-3 py-1.5 text-[12px] font-medium text-[#d4d4d8] transition-colors hover:border-indigo-500/40 hover:text-[#fafafa] disabled:opacity-50"
        >
          {estimating ? (
            <Loader2 size={12} className="animate-spin" />
          ) : (
            <Calculator size={12} />
          )}
          Estimate
        </button>
        <button
          onClick={onRunClick}
          disabled={runBlocker != null || starting}
          title={runBlocker ?? (paid ? "Run: you confirm the upper bound first" : "Run: free")}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent bg-[#fafafa] px-3 py-1.5 text-[12px] font-semibold text-[#09090b] transition-colors hover:bg-[#e4e4e7] disabled:cursor-not-allowed disabled:bg-[#27272a] disabled:text-[#71717a]"
        >
          {starting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
          Run
        </button>
      </div>
      <p
        className={`mt-1.5 truncate text-[10px] ${
          runBlocker && estimate && !estimateFresh
            ? "text-amber-300/80"
            : estimate?.data.refuse && estimateFresh
              ? "text-rose-300/80"
              : "text-[#52525b]"
        }`}
        title={runBlocker ?? undefined}
      >
        {runBlocker ??
          (paid
            ? `Fits: upper ${formatUsd(estimate?.data.total_upper_usd)} ≤ cap ${formatUsd(maxUsdNumber)}`
            : "Ready: retrieve-only, $0")}
      </p>
    </div>
  );

  const tabBar = (
    <div className="flex items-center gap-2 border-b border-[#27272a] px-4 py-2">
      <SegmentedToggle
        label="Lab section"
        value={activeTab}
        options={narrow ? [SETUP_OPTION, ...tabOptions] : tabOptions}
        onChange={setTab}
      />
      {activeTab === "results" && selectedRunId && (
        <span className="ml-1 min-w-0 truncate font-mono text-[11px] text-[#71717a]">
          {selectedRunId}
        </span>
      )}
    </div>
  );

  const tabContent = (
    <>
      {runError && (
        <div
          role="alert"
          className="mb-3 flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-3 py-2"
        >
          <TriangleAlert size={12} className="mt-0.5 shrink-0 text-rose-400/80" />
          <p className="break-words text-[12px] leading-snug text-rose-100/90">{runError}</p>
        </div>
      )}

      {activeTab === "estimate" &&
        (estimateError ? (
          <div className="flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-3 py-2">
            <TriangleAlert size={12} className="mt-0.5 shrink-0 text-rose-400/80" />
            <p className="break-words text-[12px] leading-snug text-rose-100/90">
              {estimateError}
            </p>
          </div>
        ) : estimate ? (
          <LabEstimateView
            estimate={estimate.data}
            stale={!estimateFresh}
            arms={armsByName}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 px-6 py-10 text-center text-[#71717a]">
            <FlaskConical size={28} strokeWidth={1.5} className="opacity-40" />
            <p className="text-[13px] font-medium">Price a run before it spends anything.</p>
            <p className="max-w-[420px] text-[12px] leading-relaxed text-[#52525b]">
              Pick arms, a dataset and budgets, then <span className="text-[#a1a1aa]">Estimate</span>:
              calls, tokens, a point estimate and an upper bound per arm × budget. Run
              unlocks only when the upper bound fits your cap. Retrieve-only is free.
            </p>
          </div>
        ))}

      {activeTab === "results" &&
        (!selectedRunId ? (
          <div className="flex flex-col items-center gap-2 px-6 py-10 text-center text-[#71717a]">
            <Trophy size={28} strokeWidth={1.5} className="opacity-40" />
            <p className="text-[13px] font-medium">No run selected.</p>
            <p className="max-w-[380px] text-[12px] leading-relaxed text-[#52525b]">
              Start one, or open a stored run from{" "}
              <button
                onClick={() => setTab("runs")}
                className="text-indigo-300/80 underline decoration-indigo-400/30 underline-offset-2 hover:text-indigo-200"
              >
                Runs
              </button>
              .
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {/* Run header */}
            <div className="flex flex-col gap-2 rounded-md border border-[#27272a] bg-[#18181b]/40 px-3 py-2.5">
              <div className="flex flex-wrap items-center gap-2">
                {liveStatus || shownStatus || liveHere ? (
                <StatusChip
                  status={liveStatus ?? shownStatus ?? liveHere?.status}
                  stalled={!liveStatus && shownStale}
                />
              ) : (
                <Loader2 size={12} className="animate-spin text-[#71717a]" />
              )}
                <span className="font-mono text-[11px] text-[#e4e4e7]">{selectedRunId}</span>
                {shown?.manifest.created_at && (
                  <span className="font-mono text-[10px] text-[#52525b]">
                    {formatWhen(shown.manifest.created_at)}
                  </span>
                )}
                {shownStatus === "batch_submitted" && (
                  <button
                    onClick={() => resume(selectedRunId)}
                    disabled={resuming || streaming}
                    title="Poll the Batch API and, when it is done, collect and score the answers"
                    className="ml-auto flex items-center gap-1.5 rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-200 transition-colors hover:bg-amber-400/20 disabled:opacity-50"
                  >
                    {resuming ? (
                      <Loader2 size={11} className="animate-spin" />
                    ) : (
                      <RefreshCw size={11} />
                    )}
                    Check batch
                  </button>
                )}
                {(shownStatus === "aborted" ||
                  shownStatus === "refused" ||
                  shownStatus === "failed" ||
                  shownStale) &&
                  shown && (
                  <button
                    onClick={() =>
                      // Retrieve-only runs are free: no spend to confirm.
                      shownMode === "retrieve"
                        ? resume(selectedRunId)
                        : setConfirm({ kind: "resume", runId: selectedRunId, pending: shown })
                    }
                    disabled={resuming || streaming}
                    title={
                      shownStatus === "failed"
                        ? "The run stopped on an error: continue it from its manifest (completed work is kept)"
                        : shownStale
                          ? "No job is working on this run any more (e.g. the backend restarted): continue it from its manifest"
                          : "Read what is left under your Max spend setting (you confirm first)"
                    }
                    className="ml-auto flex items-center gap-1.5 rounded-md border border-[#27272a] bg-[#18181b] px-2.5 py-1 text-[11px] font-medium text-[#d4d4d8] transition-colors hover:border-amber-400/40 hover:text-[#fafafa] disabled:opacity-50"
                  >
                    <Play size={11} />
                    {shownMode === "retrieve" ? "Resume" : "Resume…"}
                  </button>
                )}
                {shownStatus === "done" && board?.mode === "retrieve" && shown && (
                  <button
                    onClick={() => priceOnMeasured(shown)}
                    title="Load this run's configuration and price a paid run on the context sizes it measured"
                    className="ml-auto flex items-center gap-1.5 rounded-md border border-[#27272a] bg-[#18181b] px-2.5 py-1 text-[11px] font-medium text-[#d4d4d8] transition-colors hover:border-indigo-500/40 hover:text-[#fafafa]"
                  >
                    <Calculator size={11} />
                    Price a paid run on these contexts
                  </button>
                )}
              </div>
              {shown && <RunFacts detail={shown} />}
              {liveHere && (
                <div className="flex flex-col gap-1">
                  {!liveHere.finished && (
                    <div className="flex items-center gap-2 font-mono text-[10px] text-[#a1a1aa]">
                      <span className="uppercase tracking-wider">
                        {liveHere.phase ?? "starting"}
                      </span>
                      {liveHere.total > 0 && (
                        <span>
                          {liveHere.done.toLocaleString()} / {liveHere.total.toLocaleString()}
                        </span>
                      )}
                      {liveHere.spentUsd != null && (
                        <span className="ml-auto">spent {formatUsd(liveHere.spentUsd)}</span>
                      )}
                    </div>
                  )}
                  {!liveHere.finished && (
                    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-[#27272a]">
                      {pct != null ? (
                        <div
                          className="absolute inset-y-0 left-0 rounded-full bg-indigo-400 transition-[width] duration-300"
                          style={{ width: `${pct}%` }}
                        />
                      ) : (
                        <div className="loading-indicator" />
                      )}
                    </div>
                  )}
                  {liveHere.message && (
                    <p
                      className={`break-words text-[11px] leading-snug ${
                        liveHere.tone === "error"
                          ? "text-rose-300/90"
                          : liveHere.tone === "warn"
                            ? "text-amber-200/80"
                            : "text-[#a1a1aa]"
                      }`}
                    >
                      {liveHere.message}
                    </p>
                  )}
                </div>
              )}
              {!liveHere && shown && <ManifestNotes detail={shown} />}
            </div>

            {detailError && !shown && (
              <p className="rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-3 py-2 text-[12px] text-rose-100/80">
                {detailError}
              </p>
            )}

            {board ? (
              <>
                <div>
                  <SectionLabel>Leaderboard</SectionLabel>
                  <LabLeaderboard
                    board={board}
                    arms={armsByName}
                    selected={cell ? cellKey(cell.arm, cell.budget) : null}
                    onSelect={(arm, budget) =>
                      setCell((prev) =>
                        prev && prev.arm === arm && budgetKey(prev.budget) === budgetKey(budget)
                          ? null
                          : { arm, budget },
                      )
                    }
                  />
                </div>
                {cell && (
                  <LabQuestions
                    key={`${selectedRunId}:${cellKey(cell.arm, cell.budget)}`}
                    runId={selectedRunId}
                    arm={cell.arm}
                    armTitle={cellArm?.title ?? cell.arm}
                    budget={cell.budget}
                    read={board.mode !== "retrieve"}
                    onClose={() => setCell(null)}
                  />
                )}
                <div>
                  <SectionLabel>
                    {board.mode !== "retrieve" ? "Pareto frontier" : "Retrieval frontier"}
                  </SectionLabel>
                  <LabPareto board={board} frontiers={shown?.frontiers ?? null} arms={armsByName} />
                </div>
              </>
            ) : (
              shown &&
              !streaming && (
                <p className="text-[12px] text-[#52525b]">
                  No leaderboard yet: the run has not been scored
                  {shownStatus === "batch_submitted" ? " (the batch is still out)" : ""}.
                </p>
              )
            )}
          </div>
        ))}

      {activeTab === "runs" && (
        <LabRunsList
          runs={runs}
          selectedId={selectedRunId}
          onSelect={openRun}
          error={runsError}
        />
      )}
    </>
  );

  // ── Render ──
  return (
    <div ref={rootRef} className="relative flex h-full w-full flex-col bg-[#09090b]">
      {(estimating || starting || streaming) && (
        <div className="loading-bar">
          <div className="loading-indicator" />
        </div>
      )}

      {/* Top bar — same chrome as the Knowledge and Procedures views */}
      <div className="absolute left-0 right-0 top-0 z-10 flex items-center justify-between border-b border-[#27272a]/50 bg-[#09090b]/80 px-5 py-2.5 backdrop-blur-md">
        <div className="flex min-w-0 items-center gap-2">
          <FlaskConical size={14} className="shrink-0 text-[#a1a1aa]" />
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[#e4e4e7]">
            Lab
          </h2>
          <div
            title="Compare retrieval approaches on your own questions: quality AND cost, always next to the evidence floors"
            className="ml-1.5 hidden shrink-0 items-center gap-1.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 py-0.5 pl-1.5 pr-2.5 lg:flex"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
            <span className="whitespace-nowrap text-[11px] font-medium text-indigo-200/90">
              Retrieval arena · FinOps first
            </span>
          </div>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          {onExpandedChange && (
            <button
              onClick={() => onExpandedChange(!expanded)}
              aria-pressed={!!expanded}
              className="shrink-0 rounded border border-[#27272a] bg-[#18181b] p-1.5 text-[#a1a1aa] transition-colors hover:text-[#fafafa]"
              title={expanded ? "Restore the chat panel" : "Give the Lab the full height (the chat is kept)"}
              aria-label={expanded ? "Restore the chat panel" : "Expand the Lab"}
            >
              {expanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
            </button>
          )}
          <button
            onClick={() => {
              setCatalogNonce((n) => n + 1);
              void refreshRuns();
            }}
            disabled={runsLoading}
            className="shrink-0 rounded border border-[#27272a] bg-[#18181b] p-1.5 text-[#a1a1aa] transition-colors hover:text-[#fafafa] disabled:opacity-50"
            title="Reload arms, datasets and runs"
            aria-label="Reload the Lab"
          >
            <RefreshCw size={12} className={runsLoading ? "animate-spin" : ""} />
          </button>
          <div className="h-4 w-px shrink-0 bg-[#27272a]" />
          <div className="shrink-0">{viewToggle}</div>
        </div>
      </div>

      <div className="relative mt-[41px] flex min-h-0 w-full flex-1">
        {catalogStatus !== "ready" ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center text-[#71717a]">
            {catalogStatus === "loading" ? (
              <>
                <Loader2 size={24} className="animate-spin opacity-60" />
                <p className="text-[13px] font-medium">Loading the Lab…</p>
              </>
            ) : (
              <>
                <TriangleAlert size={28} className="text-rose-500/70" strokeWidth={1.5} />
                <p className="text-[13px] font-medium text-[#a1a1aa]">Lab unavailable.</p>
                {catalogError && (
                  <p className="max-w-[420px] break-words text-[12px] leading-relaxed text-[#52525b]">
                    {catalogError}
                  </p>
                )}
                <button
                  onClick={() => {
                    setCatalogStatus("loading");
                    setCatalogNonce((n) => n + 1);
                  }}
                  className="mt-1 flex items-center gap-1.5 rounded-md border border-[#27272a] bg-[#18181b] px-3 py-1.5 text-[12px] font-medium text-[#a1a1aa] transition-colors hover:border-indigo-500/40 hover:text-[#fafafa]"
                >
                  <RefreshCw size={12} />
                  Retry
                </button>
              </>
            )}
          </div>
        ) : narrow ? (
          // Narrow panel: one column, the configuration becomes the Setup tab.
          <section className="flex min-w-0 flex-1 flex-col">
            {tabBar}
            {/* Keyed by tab: each tab opens at its top, not at the last one's scroll. */}
            <div key={activeTab} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
              {activeTab === "setup" ? (
                <div className="mx-auto max-w-[560px]">{configBody}</div>
              ) : (
                tabContent
              )}
            </div>
            {(activeTab === "setup" || activeTab === "estimate") && actionsBar}
          </section>
        ) : (
          <>
            {/* ── Left: configuration ── */}
            <aside className="flex w-[300px] shrink-0 flex-col border-r border-[#27272a]">
              <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">{configBody}</div>
              {actionsBar}
            </aside>

            {/* ── Right: estimate · results · runs ── */}
            <section className="flex min-w-0 flex-1 flex-col">
              {tabBar}
              <div key={activeTab} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
                {tabContent}
              </div>
            </section>
          </>
        )}
      </div>

      {/* Spend confirmation: a new paid run, or resuming a stopped one */}
      {confirm && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center bg-[#09090b]/70 px-4 backdrop-blur-sm"
          onClick={() => setConfirm(null)}
        >
          <div
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="lab-confirm-title"
            aria-describedby="lab-confirm-description"
            onClick={(e) => e.stopPropagation()}
            className="msg-enter w-full max-w-[400px] rounded-lg border border-[#27272a] bg-[#18181b] p-4 shadow-2xl"
          >
            {confirm.kind === "run" && estimate ? (
              <>
                <h3
                  id="lab-confirm-title"
                  className="flex items-center gap-2 text-[14px] font-semibold text-[#fafafa]"
                >
                  <Zap size={14} className="text-amber-400" />
                  Spend up to {formatUsd(estimate.data.total_upper_usd)}?
                </h3>
                <div
                  id="lab-confirm-description"
                  className="mt-2 flex flex-col gap-1.5 text-[12px] leading-relaxed text-[#a1a1aa]"
                >
                  <p>
                    A <span className="text-[#d4d4d8]">{mode}</span> run with{" "}
                    <span className="font-mono text-[#d4d4d8]">{readerModel}</span>: retrieval
                    first (free), then{" "}
                    {estimate.data.phases
                      .find((p) => p.phase === "reader")
                      ?.calls.toLocaleString() ?? "the"}{" "}
                    reader calls through the backend&apos;s OpenAI key.
                  </p>
                  <p className="font-mono text-[11px]">
                    point {formatUsd(estimate.data.total_point_usd)} · upper{" "}
                    {formatUsd(estimate.data.total_upper_usd)} · cap {formatUsd(maxUsdNumber)}
                  </p>
                  <p className="text-[11px] text-[#71717a]">
                    The cap is re-checked on the measured contexts before any read
                    {mode === "realtime"
                      ? ", and realtime reads are metered: the run stops cleanly before it would exceed the cap."
                      : "; a batch is billed at half price once OpenAI completes it."}
                  </p>
                </div>
                <div className="mt-4 flex justify-end gap-2">
                  <button
                    ref={cancelRef}
                    onClick={() => setConfirm(null)}
                    className="rounded-md border border-[#27272a] px-3 py-1.5 text-[12px] font-medium text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={startRun}
                    disabled={starting}
                    className="flex items-center gap-1.5 rounded-md border border-transparent bg-[#fafafa] px-3 py-1.5 text-[12px] font-semibold text-[#09090b] transition-colors hover:bg-[#e4e4e7] disabled:opacity-60"
                  >
                    {starting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                    Run
                  </button>
                </div>
              </>
            ) : confirm.kind === "resume" ? (
              <ResumeConfirm
                detail={confirm.pending}
                cap={maxUsdNumber}
                busy={resuming}
                cancelRef={cancelRef}
                onCancel={() => setConfirm(null)}
                onConfirm={() => resume(confirm.runId, maxUsdNumber)}
              />
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

/** Resuming an aborted / refused run reads what is left: that spends, so it is confirmed. */
function ResumeConfirm({
  detail,
  cap,
  busy,
  cancelRef,
  onCancel,
  onConfirm,
}: {
  detail: LabRunDetail;
  cap: number | null;
  busy: boolean;
  cancelRef: RefObject<HTMLButtonElement | null>;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const m = detail.manifest;
  const pending = m.estimate?.read_pending;
  const spent = m.actual?.reader?.usd ?? pending?.spent_usd;
  const reader = typeof m.config?.reader_model === "string" ? m.config.reader_model : null;
  const capOk = cap != null && cap > 0;
  return (
    <>
      <h3
        id="lab-confirm-title"
        className="flex items-center gap-2 text-[14px] font-semibold text-[#fafafa]"
      >
        <Zap size={14} className="text-amber-400" />
        Resume under a cap of {formatUsd(cap)}?
      </h3>
      <div
        id="lab-confirm-description"
        className="mt-2 flex flex-col gap-1.5 text-[12px] leading-relaxed text-[#a1a1aa]"
      >
        <p>
          <span className="font-mono text-[#d4d4d8]">{detail.run_id}</span> stopped as{" "}
          <span className="text-[#d4d4d8]">{m.status}</span>. Resuming reads the requests that
          have no answer yet
          {reader ? (
            <>
              {" "}
              with <span className="font-mono text-[#d4d4d8]">{reader}</span>
            </>
          ) : null}
          ; nothing already answered is sent again.
        </p>
        <p className="font-mono text-[11px]">
          {pending?.requests != null && <>pending {pending.requests.toLocaleString()} · </>}
          {pending?.upper_usd != null && <>upper {formatUsd(pending.upper_usd)} · </>}
          {spent != null && <>spent so far {formatUsd(spent)} · </>}
          new cap {formatUsd(cap)}
        </p>
        <p className="text-[11px] text-[#71717a]">
          The new cap is the run&apos;s total (your Max spend setting); it is re-checked on the measured
          contexts, and realtime reads stay metered under it.
        </p>
        {!capOk && (
          <p className="text-[11px] text-rose-300/90">Set a Max spend above $0 first.</p>
        )}
      </div>
      <div className="mt-4 flex justify-end gap-2">
        <button
          ref={cancelRef}
          onClick={onCancel}
          className="rounded-md border border-[#27272a] px-3 py-1.5 text-[12px] font-medium text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
        >
          Cancel
        </button>
        <button
          onClick={onConfirm}
          disabled={busy || !capOk}
          className="flex items-center gap-1.5 rounded-md border border-transparent bg-[#fafafa] px-3 py-1.5 text-[12px] font-semibold text-[#09090b] transition-colors hover:bg-[#e4e4e7] disabled:opacity-60"
        >
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
          Resume
        </button>
      </div>
    </>
  );
}

/** Dataset, mode, reader, cap and provenance of a stored run. */
function RunFacts({ detail }: { detail: LabRunDetail }) {
  const m = detail.manifest;
  const config = m.config ?? {};
  const configDataset = config.dataset;
  const datasetName =
    m.dataset?.name ?? (typeof configDataset === "string" ? configDataset : configDataset?.name);
  const mode = typeof config.mode === "string" ? config.mode : detail.leaderboard?.mode;
  const actual = m.actual?.reader?.usd;
  const sha = m.code?.git_sha;
  const facts: [string, string, string?][] = [
    ["dataset", `${datasetName ?? "—"}${m.dataset?.split ? ` · ${m.dataset.split}` : ""}`],
    ["n", String(m.dataset?.n ?? detail.leaderboard?.n_questions ?? "—")],
    ["mode", String(mode ?? "—")],
  ];
  if (mode !== "retrieve" && config.reader_model) facts.push(["reader", config.reader_model]);
  if (m.budget_cap_usd != null) facts.push(["cap", formatUsd(m.budget_cap_usd)]);
  if (mode !== "retrieve" && actual != null) {
    facts.push([
      "spent",
      formatUsd(actual),
      "Reader spend at recorded prices; a failed request is counted at its reserved upper bound",
    ]);
  }
  if (sha) {
    facts.push([
      "code",
      `${sha.slice(0, 7)}${m.code?.git_dirty ? "+dirty" : ""}`,
      "Git commit the run was made on (+dirty: uncommitted changes)",
    ]);
  } else if (m.code) {
    facts.push([
      "code",
      "unknown",
      m.code.git_unavailable ?? "The commit could not be read when this run was made",
    ]);
  }
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px]">
      {facts.map(([k, v, hint]) => (
        <span key={k} title={hint} className="text-[#52525b]">
          {k} <span className="text-[#a1a1aa]">{v}</span>
        </span>
      ))}
    </div>
  );
}

/** Why a run stopped, when it did not simply finish. */
function ManifestNotes({ detail }: { detail: LabRunDetail }) {
  const m = detail.manifest;
  const reason =
    m.status === "refused"
      ? (m.refuse_reason ?? m.phases?.read?.reason)
      : m.status === "aborted"
        ? (m.abort_reason ?? m.phases?.read?.reason)
        : m.status === "failed"
          ? m.error
          : null;
  if (!reason) return null;
  const tone = m.status === "aborted" ? "text-amber-200/80" : "text-rose-300/90";
  return (
    <p className={`break-words text-[11px] leading-snug ${tone}`}>
      {reason}
      {m.status === "aborted" && (
        <span className="text-[#71717a]">
          {" "}
          Answers so far are scored; nothing already answered is re-sent on resume:{" "}
          <span className="font-mono">synapse-graphrag lab resume {detail.run_id}</span>
        </span>
      )}
    </p>
  );
}
