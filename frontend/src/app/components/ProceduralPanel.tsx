"use client";

import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ArrowRight,
  ChevronDown,
  History,
  Loader2,
  Maximize2,
  RefreshCw,
  TriangleAlert,
  Undo2,
  Waypoints,
  Workflow,
  X,
} from "lucide-react";
import type {
  ForceGraphMethods,
  LinkObject,
  NodeObject,
} from "react-force-graph-2d";
import {
  ApiError,
  fetchProcedureGraph,
  fetchProcedureVersions,
  fetchProcedures,
  rollbackProcedure,
} from "../lib/api";
import {
  DEFAULT_PROCEDURE,
  GUIDANCE_HOPS,
  PROC_START_COLOR,
  PROC_TERMINAL_COLOR,
  PROC_TRACE_COLOR,
  PROC_TYPE_COLORS,
  colorForProcType,
  colorForRelation,
} from "../lib/constants";
import type {
  AgentResult,
  ProcGraphData,
  ProcGraphDiff,
  ProcLink,
  ProcNode,
  ProcVersion,
  ProceduralGraphSummary,
} from "../lib/types";

type PNode = NodeObject<ProcNode>;
type PLink = LinkObject<ProcNode, ProcLink>;
type FGMethods = ForceGraphMethods<PNode, PLink>;
type ForceGraphComponent = (typeof import("react-force-graph-2d"))["default"];

type Status = "loading" | "ready" | "empty" | "error";

/** Drawn node radius — also `nodeRelSize`, so arrowheads stop at the rim. */
const NODE_R = 5;
const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";
/** Hop-1 transitions ("Immediate Transition Options") vs. the hop-2 horizon. */
const SCOPE_HOP1 = "#818cf8";
const SCOPE_HOP2 = "#6366f199";
const SELECT_RING = "#fafafa";

interface ProceduralPanelProps {
  /** The Knowledge | Procedures switch, rendered in this view's header. */
  viewToggle: ReactNode;
  /** The latest Navigator run, overlaid as visited-step badges. */
  trace?: AgentResult | null;
}

function linkEndId(
  end: string | number | { id?: string | number } | undefined | null,
): string {
  if (end == null) return "";
  return typeof end === "object" ? String(end.id ?? "") : String(end);
}

/** Tooltips are rendered as HTML by force-graph; node text is model-written. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Case / punctuation-insensitive id, to match a tool call to its node. */
function normalizeId(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function formatScore(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return score.toFixed(3).replace(/\.?0+$/, "");
}

function formatWhen(iso: string | null): string {
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

/** "+2 −1 ~3" — what a version changed, structurally. */
function diffSummary(diff: ProcGraphDiff | undefined): { text: string; title: string } | null {
  if (!diff) return null;
  const n = (list?: unknown[]) => (Array.isArray(list) ? list.length : 0);
  const added = n(diff.added_nodes) + n(diff.added_edges);
  const removed = n(diff.removed_nodes) + n(diff.removed_edges);
  const changed = n(diff.changed_nodes) + n(diff.changed_edges);
  if (added + removed + changed === 0) return null;
  const parts = [
    added ? `+${added}` : "",
    removed ? `−${removed}` : "",
    changed ? `~${changed}` : "",
  ].filter(Boolean);
  const title =
    `${n(diff.added_nodes)} nodes / ${n(diff.added_edges)} transitions added, ` +
    `${n(diff.removed_nodes)} / ${n(diff.removed_edges)} removed, ` +
    `${n(diff.changed_nodes)} / ${n(diff.changed_edges)} rewritten`;
  return { text: parts.join(" "), title };
}

/**
 * Kahn's algorithm. A DAG gets a left-to-right layered layout (Start on the
 * left, terminals on the right); a graph with cycles (cycle_policy "allow")
 * falls back to a free force layout, which force-graph's DAG mode cannot do.
 */
function isAcyclic(nodes: ProcNode[], links: ProcLink[]): boolean {
  const indegree = new Map(nodes.map((n) => [n.id, 0]));
  const out = new Map<string, string[]>();
  for (const link of links) {
    const s = linkEndId(link.source);
    const t = linkEndId(link.target);
    if (!indegree.has(s) || !indegree.has(t)) continue;
    indegree.set(t, (indegree.get(t) ?? 0) + 1);
    out.set(s, [...(out.get(s) ?? []), t]);
  }
  const queue = [...indegree].filter(([, d]) => d === 0).map(([id]) => id);
  let seen = 0;
  while (queue.length > 0) {
    const id = queue.shift()!;
    seen += 1;
    for (const t of out.get(id) ?? []) {
      const d = (indegree.get(t) ?? 0) - 1;
      indegree.set(t, d);
      if (d === 0) queue.push(t);
    }
  }
  return seen === nodes.length;
}

/**
 * Bend parallel / opposite transitions apart so each keeps a readable label.
 * Curvatures are assigned in a canonical (sorted-endpoint) frame, then signed
 * by direction: force-graph offsets the control point perpendicular to the
 * link's *own* direction, so A→B and B→A with equal curvature would overlap.
 */
function linkCurvatures(links: ProcLink[]): Map<ProcLink, number> {
  const groups = new Map<string, ProcLink[]>();
  for (const link of links) {
    const s = linkEndId(link.source);
    const t = linkEndId(link.target);
    const key = s < t ? `${s}\u0000${t}` : `${t}\u0000${s}`;
    groups.set(key, [...(groups.get(key) ?? []), link]);
  }
  const curvature = new Map<ProcLink, number>();
  for (const group of groups.values()) {
    group.forEach((link, i) => {
      const s = linkEndId(link.source);
      const t = linkEndId(link.target);
      if (s === t) {
        curvature.set(link, 0.6 + i * 0.25); // self-loop
        return;
      }
      const canonical = group.length > 1 ? (i - (group.length - 1) / 2) * 0.35 : 0;
      curvature.set(link, s < t ? canonical : -canonical);
    });
  }
  return curvature;
}

/**
 * The guidance scope N_h(u): `root` plus everything reachable along OUTGOING
 * transitions within `hops` steps (BFS, so each node keeps its nearest hop).
 * It is exactly the subgraph the navigator is shown when it stands on `root`.
 */
function outgoingHops(
  root: string,
  adjacency: Map<string, string[]>,
  hops: number,
): Map<string, number> {
  const seen = new Map<string, number>([[root, 0]]);
  let frontier = [root];
  for (let hop = 1; hop <= hops; hop++) {
    const next: string[] = [];
    for (const id of frontier) {
      for (const target of adjacency.get(id) ?? []) {
        if (!seen.has(target)) {
          seen.set(target, hop);
          next.push(target);
        }
      }
    }
    frontier = next;
  }
  return seen;
}

/** Circle = tool call, diamond = reasoning step, square = status marker. */
function shapePath(
  ctx: CanvasRenderingContext2D,
  type: string,
  x: number,
  y: number,
  r: number,
) {
  ctx.beginPath();
  const kind = type?.toUpperCase();
  if (kind === "REASONING") {
    const d = r * 1.25;
    ctx.moveTo(x, y - d);
    ctx.lineTo(x + d, y);
    ctx.lineTo(x, y + d);
    ctx.lineTo(x - d, y);
    ctx.closePath();
  } else if (kind === "STATUS") {
    const h = r * 0.9;
    ctx.rect(x - h, y - h, h * 2, h * 2);
  } else {
    ctx.arc(x, y, r, 0, 2 * Math.PI);
  }
}

/** HTML swatch matching the canvas shape of a node type. */
function TypeSwatch({ type, size = 10 }: { type: string; size?: number }) {
  const kind = type?.toUpperCase();
  const color = colorForProcType(type);
  if (kind === "REASONING") {
    const side = Math.round(size * 0.78);
    return (
      <span
        className="inline-block shrink-0 rotate-45"
        style={{ width: side, height: side, backgroundColor: color }}
      />
    );
  }
  return (
    <span
      className={`inline-block shrink-0 ${kind === "STATUS" ? "rounded-[2px]" : "rounded-full"}`}
      style={{ width: size, height: size, backgroundColor: color }}
    />
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
      {children}
    </div>
  );
}

export default function ProceduralPanel({ viewToggle, trace }: ProceduralPanelProps) {
  const [ForceGraph, setForceGraph] = useState<ForceGraphComponent | null>(null);
  const [dims, setDims] = useState({ width: 800, height: 600 });
  const [graphs, setGraphs] = useState<ProceduralGraphSummary[]>([]);
  // What the user picked; `selected` is what actually loaded (the pick may
  // have vanished, e.g. deleted from the CLI, and fall back to the default).
  const [requested, setRequested] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [data, setData] = useState<ProcGraphData | null>(null);
  const [versions, setVersions] = useState<ProcVersion[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [reloadNonce, setReloadNonce] = useState(0);

  const [hoverId, setHoverId] = useState<string | null>(null);
  const [hoverLink, setHoverLink] = useState<PLink | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedLink, setSelectedLink] = useState<PLink | null>(null);

  // null until the user toggles it: then the canvas height decides (below).
  const [versionsOpen, setVersionsOpen] = useState<boolean | null>(null);
  const [pendingRollback, setPendingRollback] = useState<number | null>(null);
  const [rollingBack, setRollingBack] = useState(false);
  const [rollbackError, setRollbackError] = useState<string | null>(null);
  const [dismissedTrace, setDismissedTrace] = useState<AgentResult | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<FGMethods | undefined>(undefined);
  const fitRef = useRef(false);
  const cancelRef = useRef<HTMLButtonElement>(null);

  // ── Load the force-graph library on the client only ──
  useEffect(() => {
    let active = true;
    import("react-force-graph-2d").then((mod) => {
      if (active) setForceGraph(() => mod.default);
    });
    return () => {
      active = false;
    };
  }, []);

  // ── Track container size for a crisp, responsive canvas ──
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () =>
      setDims({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ForceGraph]);

  // ── Fetch the graph list, then the chosen graph + its history ──
  const load = useCallback(async (signal: AbortSignal, preferred: string | null) => {
    setRefreshing(true);
    try {
      const list = await fetchProcedures(signal);
      if (signal.aborted) return;
      setGraphs(list);
      if (list.length === 0) {
        setSelected(null);
        setData(null);
        setVersions([]);
        setError(null);
        setStatus("empty");
        return;
      }
      const names = list.map((g) => g.name);
      const name =
        preferred && names.includes(preferred)
          ? preferred
          : names.includes(DEFAULT_PROCEDURE)
            ? DEFAULT_PROCEDURE
            : names[0];
      const [graph, history] = await Promise.all([
        fetchProcedureGraph(name, signal),
        // History is secondary: the graph still renders without it.
        fetchProcedureVersions(name, signal).catch((err) => {
          if (!signal.aborted) console.error("Failed to fetch versions:", err);
          return [] as ProcVersion[];
        }),
      ]);
      if (signal.aborted) return;
      fitRef.current = false; // re-fit once the new layout settles
      setSelected(name);
      setData(graph);
      setVersions(history);
      setSelectedLink(null); // link objects belong to the previous data
      setSelectedNodeId(null);
      setHoverId(null);
      setHoverLink(null);
      setError(null);
      setStatus("ready");
    } catch (err) {
      if (signal.aborted) return;
      console.error("Failed to load procedural graph:", err);
      setError(
        err instanceof ApiError && err.detail
          ? err.detail
          : "Is the backend running with procedural memory enabled?",
      );
      setStatus("error");
    } finally {
      if (!signal.aborted) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal, requested);
    return () => controller.abort();
  }, [load, requested, reloadNonce]);

  const reload = () => setReloadNonce((n) => n + 1);

  // A fresh object per fetch: force-graph mutates it (positions, link ends).
  const graphData = useMemo(
    () => (data ? { nodes: data.nodes, links: data.links } : null),
    [data],
  );

  const adjacency = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const link of data?.links ?? []) {
      const s = linkEndId(link.source);
      map.set(s, [...(map.get(s) ?? []), linkEndId(link.target)]);
    }
    return map;
  }, [data]);

  const acyclic = useMemo(
    () => (data ? isAcyclic(data.nodes, data.links) : true),
    [data],
  );

  const curvature = useMemo(() => linkCurvatures(data?.links ?? []), [data]);

  const presentRelations = useMemo(
    () => Array.from(new Set((data?.links ?? []).map((l) => l.relation))).sort(),
    [data],
  );

  // Hover previews a node's guidance scope; a click pins it.
  const scopeRoot = hoverId ?? selectedNodeId;
  const scope = useMemo(
    () =>
      scopeRoot == null ? null : outgoingHops(scopeRoot, adjacency, GUIDANCE_HOPS),
    [scopeRoot, adjacency],
  );

  /** 1 or 2 when the link is a hop-1 / hop-2 transition of the scope, else null. */
  const linkHop = useCallback(
    (link: PLink): number | null => {
      if (!scope) return null;
      const from = scope.get(linkEndId(link.source));
      return from != null && from < GUIDANCE_HOPS ? from + 1 : null;
    },
    [scope],
  );

  // ── Overlay of the latest Navigator run: node id → step numbers ──
  const traceVisits = useMemo(() => {
    if (!trace || trace === dismissedTrace || !data) return null;
    if (!trace.graph || trace.graph.name !== selected) return null;
    const byNorm = new Map(data.nodes.map((n) => [normalizeId(n.id), n.id]));
    const visits = new Map<string, number[]>();
    trace.steps.forEach((step, i) => {
      if (!step.action) return;
      const id = byNorm.get(normalizeId(step.action));
      if (id) visits.set(id, [...(visits.get(id) ?? []), i + 1]);
    });
    return visits;
  }, [trace, dismissedTrace, data, selected]);

  // Spread the layers so relation labels have room.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !data || data.nodes.length === 0) return;
    const charge = fg.d3Force("charge");
    if (charge?.strength) charge.strength(-260);
    const link = fg.d3Force("link");
    if (link?.distance) link.distance(64);
    fg.d3ReheatSimulation?.();
  }, [ForceGraph, data]);

  // ── Rollback dialog: Escape cancels, focus starts on "Cancel" ──
  useEffect(() => {
    if (pendingRollback == null) return;
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !rollingBack) setPendingRollback(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pendingRollback, rollingBack]);

  const askRollback = (version: number) => {
    setRollbackError(null);
    setPendingRollback(version);
  };

  const confirmRollback = async () => {
    if (pendingRollback == null || !selected || rollingBack) return;
    setRollingBack(true);
    setRollbackError(null);
    try {
      await rollbackProcedure(selected, pendingRollback);
      setPendingRollback(null);
      reload();
    } catch (err) {
      console.error("Rollback failed:", err);
      setRollbackError(
        err instanceof ApiError && err.detail ? err.detail : "Rollback failed.",
      );
    } finally {
      setRollingBack(false);
    }
  };

  const zoomToFit = () => fgRef.current?.zoomToFit(500, 70);

  const selectNode = (id: string) => {
    setSelectedLink(null);
    setSelectedNodeId(id);
  };

  const selectLink = (link: PLink) => {
    setSelectedNodeId(null);
    setSelectedLink(link);
  };

  const clearSelection = () => {
    setSelectedNodeId(null);
    setSelectedLink(null);
  };

  // ── Canvas rendering ──
  const drawNode = useCallback(
    (node: PNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const hop = scope?.get(node.id);
      const onSelectedLink =
        selectedLink != null &&
        (linkEndId(selectedLink.source) === node.id ||
          linkEndId(selectedLink.target) === node.id);
      const dim =
        (scope != null && hop == null) || (selectedLink != null && !onSelectedLink);

      ctx.globalAlpha = dim ? 0.18 : 1;

      if (node.is_start) {
        // Soft emerald halo + ring: where every run is localized first (a0).
        ctx.beginPath();
        ctx.arc(x, y, NODE_R + 5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(52, 211, 153, 0.12)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, NODE_R + 2.6, 0, 2 * Math.PI);
        ctx.strokeStyle = PROC_START_COLOR;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
      if (node.is_terminal) {
        // A ring around the node, like a state machine's final state.
        ctx.beginPath();
        ctx.arc(x, y, NODE_R + 2.6, 0, 2 * Math.PI);
        ctx.strokeStyle = PROC_TERMINAL_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (hop === 0 || node.id === selectedNodeId) {
        ctx.beginPath();
        ctx.arc(x, y, NODE_R + 4.8, 0, 2 * Math.PI);
        ctx.strokeStyle = node.id === selectedNodeId ? SELECT_RING : SCOPE_HOP1;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }

      shapePath(ctx, node.type, x, y, NODE_R);
      ctx.fillStyle = colorForProcType(node.type);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#09090b";
      ctx.stroke();

      if (scale > 0.45) {
        const fontSize = Math.max(9 / scale, 2.4);
        ctx.font = `500 ${fontSize}px ${MONO}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = dim ? "#52525b" : "#e4e4e7";
        ctx.fillText(node.label || node.id, x, y + NODE_R + 4);
      }

      const visits = traceVisits?.get(node.id);
      if (visits) {
        const r = Math.max(5.5 / scale, 2.2);
        const bx = x + NODE_R * 0.95;
        const by = y - NODE_R * 0.95;
        ctx.beginPath();
        ctx.arc(bx, by, r, 0, 2 * Math.PI);
        ctx.fillStyle = PROC_TRACE_COLOR;
        ctx.fill();
        ctx.lineWidth = Math.max(1 / scale, 0.4);
        ctx.strokeStyle = "#09090b";
        ctx.stroke();
        ctx.font = `700 ${r * 1.15}px ${MONO}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#09090b";
        ctx.fillText(
          visits.length > 1 ? `${visits[0]}+` : String(visits[0]),
          bx,
          by + r * 0.05,
        );
      }
      ctx.globalAlpha = 1;
    },
    [scope, selectedLink, selectedNodeId, traceVisits],
  );

  /** Relation label at the link's midpoint (on the curve, kept upright). */
  const drawLinkLabel = useCallback(
    (link: PLink, ctx: CanvasRenderingContext2D, scale: number) => {
      const s = link.source;
      const t = link.target;
      if (typeof s !== "object" || typeof t !== "object") return;
      if (s.x == null || s.y == null || t.x == null || t.y == null) return;
      if (scale < 0.7) return;

      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const length = Math.hypot(dx, dy);
      if (length === 0) return; // self-loop: the side card carries the label

      // force-graph's quadratic control point sits `length × curvature` off the
      // midpoint; the curve's own midpoint (t = 0.5) is half that.
      const c = curvature.get(link) ?? 0;
      const angle = Math.atan2(dy, dx);
      const offset = (length * c) / 2;
      const mx = (s.x + t.x) / 2 + offset * Math.cos(angle - Math.PI / 2);
      const my = (s.y + t.y) / 2 + offset * Math.sin(angle - Math.PI / 2);

      const fontSize = Math.max(7 / scale, 1.8);
      ctx.font = `600 ${fontSize}px ${MONO}`;
      const text = link.relation;
      const width = ctx.measureText(text).width;
      const pad = fontSize * 0.35;
      // Skip labels that would spill over the endpoints.
      if (width + pad * 2 > length - NODE_R * 2 - 4) return;

      const hop = linkHop(link);
      const dim =
        (scope != null && hop == null) ||
        (selectedLink != null && link !== selectedLink);

      let upright = angle;
      if (upright > Math.PI / 2) upright -= Math.PI;
      if (upright < -Math.PI / 2) upright += Math.PI;

      ctx.save();
      ctx.globalAlpha = dim ? 0.15 : 1;
      ctx.translate(mx, my);
      ctx.rotate(upright);
      ctx.fillStyle = "#09090b";
      ctx.fillRect(
        -width / 2 - pad,
        -fontSize / 2 - pad * 0.6,
        width + pad * 2,
        fontSize + pad * 1.2,
      );
      ctx.fillStyle = colorForRelation(link.relation);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 0, 0);
      ctx.restore();
    },
    [curvature, linkHop, scope, selectedLink],
  );

  const linkColor = useCallback(
    (link: PLink) => {
      if (link === selectedLink) return SELECT_RING;
      if (scope) {
        const hop = linkHop(link);
        return hop === 1 ? SCOPE_HOP1 : hop === 2 ? SCOPE_HOP2 : "#ffffff0d";
      }
      if (selectedLink) return "#ffffff14";
      if (link === hoverLink) return "#a1a1aa";
      return "#52525b";
    },
    [hoverLink, linkHop, scope, selectedLink],
  );

  const linkWidth = useCallback(
    (link: PLink) => {
      if (link === selectedLink) return 2.2;
      const hop = linkHop(link);
      if (hop === 1) return 1.8;
      if (hop === 2) return 1.3;
      return link === hoverLink ? 1.8 : 1.1;
    },
    [hoverLink, linkHop, selectedLink],
  );

  // ── Side card content ──
  const selectedNode = useMemo(
    () => data?.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [data, selectedNodeId],
  );

  const outgoing = useMemo(
    () =>
      selectedNodeId == null
        ? []
        : (data?.links ?? []).filter((l) => linkEndId(l.source) === selectedNodeId),
    [data, selectedNodeId],
  );

  // Keep the history open only when it leaves the canvas (and a detail card,
  // if one is open) enough room; the user's own toggle always wins.
  const hasSelection = selectedNode != null || selectedLink != null;
  const versionsExpanded =
    versionsOpen ?? (dims.height >= 640 || (!hasSelection && dims.height >= 420));

  const liveVersion = data?.version ?? 0;
  const nextVersion =
    Math.max(liveVersion, ...versions.map((v) => v.version)) + 1;
  const summary = graphs.find((g) => g.name === selected);
  const visitedCount = traceVisits?.size ?? 0;

  return (
    <div className="relative flex h-full w-full flex-col bg-[#09090b]">
      {/* Top bar — same chrome as the Knowledge view */}
      <div className="absolute left-0 right-0 top-0 z-10 flex items-center justify-between border-b border-[#27272a]/50 bg-[#09090b]/80 px-5 py-2.5 backdrop-blur-md">
        <div className="flex min-w-0 items-center gap-2">
          <Workflow size={14} className="shrink-0 text-[#a1a1aa]" />
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[#e4e4e7]">
            Topology
          </h2>
          {data && (
            <div
              title={
                [
                  summary?.description,
                  summary?.updated_at ? `Updated ${formatWhen(summary.updated_at)}` : "",
                  "Score: validation score of this version (null for a hand-written prior)",
                ]
                  .filter(Boolean)
                  .join("\n")
              }
              className="ml-1.5 flex shrink-0 items-center gap-1.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 py-0.5 pl-1.5 pr-2.5"
            >
              <History size={10} className="shrink-0 text-indigo-300/80" />
              <span className="whitespace-nowrap font-mono text-[11px] font-medium text-indigo-200/90">
                v{data.version} · score {formatScore(data.score)}
              </span>
            </div>
          )}
        </div>
        <div className="flex min-w-0 items-center gap-2">
          {graphs.length > 1 && (
            <div className="relative min-w-0">
              <select
                value={selected ?? ""}
                onChange={(e) => setRequested(e.target.value)}
                aria-label="Procedural graph"
                className="w-full max-w-[170px] appearance-none truncate rounded border border-[#27272a] bg-[#18181b] py-1 pl-2 pr-6 font-mono text-[11px] text-[#fafafa] outline-none transition-colors hover:border-[#3f3f46] focus:border-indigo-500/50"
              >
                {graphs.map((g) => (
                  <option key={g.name} value={g.name}>
                    {g.name}
                  </option>
                ))}
              </select>
              <ChevronDown
                size={11}
                className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[#71717a]"
              />
            </div>
          )}
          <button
            onClick={reload}
            disabled={refreshing}
            className="shrink-0 rounded border border-[#27272a] bg-[#18181b] p-1.5 text-[#a1a1aa] transition-colors hover:text-[#fafafa] disabled:opacity-50"
            title="Reload (picks up versions saved by the CLI or an evolution run)"
            aria-label="Reload procedural graph"
          >
            <RefreshCw size={12} className={refreshing ? "animate-spin" : ""} />
          </button>
          <button
            onClick={zoomToFit}
            className="shrink-0 rounded border border-[#27272a] bg-[#18181b] p-1.5 text-[#a1a1aa] transition-colors hover:text-[#fafafa]"
            title="Fit to view"
            aria-label="Fit procedural graph to view"
          >
            <Maximize2 size={12} />
          </button>
          {/* Least important control: dropped first on a narrow panel. */}
          {data && (
            <div className="hidden shrink-0 rounded border border-[#27272a] bg-[#18181b] px-2.5 py-1 text-[11px] font-medium text-[#71717a] xl:block">
              {data.nodes.length} N / {data.links.length} E
            </div>
          )}
          <div className="h-4 w-px shrink-0 bg-[#27272a]" />
          <div className="shrink-0">{viewToggle}</div>
        </div>
      </div>

      {/* Canvas */}
      <div className="graph-grid relative mt-[41px] w-full flex-1" ref={containerRef}>
        {ForceGraph && graphData && graphData.nodes.length > 0 && (
          <ForceGraph
            ref={fgRef}
            graphData={graphData}
            width={dims.width}
            height={dims.height}
            nodeRelSize={NODE_R}
            nodeLabel={(node: PNode) => {
              const visits = traceVisits?.get(node.id);
              const steps = visits ? ` · navigator step ${visits.join(", ")}` : "";
              return escapeHtml(`${node.label || node.id} · ${node.type}${steps}`);
            }}
            nodeCanvasObject={drawNode}
            nodePointerAreaPaint={(
              node: PNode,
              color: string,
              ctx: CanvasRenderingContext2D,
            ) => {
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(node.x ?? 0, node.y ?? 0, NODE_R + 2, 0, 2 * Math.PI);
              ctx.fill();
            }}
            linkColor={linkColor}
            linkWidth={linkWidth}
            linkCurvature={(link: PLink) => curvature.get(link) ?? 0}
            linkLineDash={(link: PLink) => (link.condition ? [2.5, 1.5] : null)}
            linkDirectionalArrowLength={4.5}
            linkDirectionalArrowRelPos={1}
            linkLabel={(link: PLink) =>
              escapeHtml(
                link.condition
                  ? `${link.relation} · ${link.condition}`
                  : `${link.relation} · unconditional`,
              )
            }
            linkCanvasObjectMode={() => "after"}
            linkCanvasObject={drawLinkLabel}
            linkHoverPrecision={6}
            onNodeClick={(node: PNode) => selectNode(node.id)}
            onNodeHover={(node: PNode | null) => setHoverId(node ? node.id : null)}
            onLinkClick={(link: PLink) => selectLink(link)}
            onLinkHover={(link: PLink | null) => setHoverLink(link)}
            onBackgroundClick={clearSelection}
            dagMode={acyclic ? "lr" : undefined}
            dagLevelDistance={72}
            // Never throw on a cycle (the acyclic check above should already
            // keep DAG mode off for those); a free layout is the fallback.
            onDagError={() => undefined}
            onEngineStop={() => {
              if (!fitRef.current) {
                fgRef.current?.zoomToFit(500, 70);
                fitRef.current = true;
              }
            }}
            backgroundColor="transparent"
            d3VelocityDecay={0.35}
            warmupTicks={60}
            cooldownTicks={120}
          />
        )}

        {/* Loading / empty / error states */}
        {status !== "ready" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center text-[#71717a]">
            {status === "loading" && (
              <>
                <Loader2 size={24} className="animate-spin opacity-60" />
                <p className="text-[13px] font-medium">Loading procedural memory…</p>
              </>
            )}
            {status === "empty" && (
              <>
                <Workflow size={32} className="opacity-40" strokeWidth={1.5} />
                <p className="text-[13px] font-medium">No procedural graphs yet.</p>
                <p className="max-w-[360px] text-[12px] leading-relaxed text-[#52525b]">
                  The backend seeds{" "}
                  <span className="font-mono text-[#a1a1aa]">{DEFAULT_PROCEDURE}</span>{" "}
                  at startup when procedural memory is enabled, or import one with{" "}
                  <span className="whitespace-nowrap font-mono text-[#a1a1aa]">
                    synapse-graphrag procedures import
                  </span>
                  .
                </p>
              </>
            )}
            {status === "error" && (
              <>
                <TriangleAlert size={28} className="text-rose-500/70" strokeWidth={1.5} />
                <p className="text-[13px] font-medium text-[#a1a1aa]">
                  Procedural memory unavailable.
                </p>
                {error && (
                  <p className="max-w-[420px] break-words text-[12px] leading-relaxed text-[#52525b]">
                    {error}
                  </p>
                )}
                <button
                  onClick={reload}
                  disabled={refreshing}
                  className="mt-1 flex items-center gap-1.5 rounded-md border border-[#27272a] bg-[#18181b] px-3 py-1.5 text-[12px] font-medium text-[#a1a1aa] transition-colors hover:border-indigo-500/40 hover:text-[#fafafa] disabled:opacity-50"
                >
                  <RefreshCw size={12} className={refreshing ? "animate-spin" : ""} />
                  Retry
                </button>
              </>
            )}
          </div>
        )}

        {/* Last Navigator run */}
        {status === "ready" && trace && traceVisits && (
          <div className="absolute left-3 top-3 z-10 flex max-w-[calc(100%-340px)] items-center gap-2 rounded-full border border-amber-400/30 bg-[#09090b]/85 py-1 pl-2 pr-1 backdrop-blur-md">
            <Waypoints size={11} className="shrink-0 text-amber-400" />
            <span className="truncate text-[11px] font-medium text-amber-100/90">
              Last Navigator run
            </span>
            <span className="shrink-0 font-mono text-[10px] text-amber-200/60">
              {trace.steps.length} steps · {visitedCount} nodes
              {trace.graph && data && trace.graph.version !== data.version
                ? ` · ran on v${trace.graph.version}`
                : ""}
            </span>
            <button
              onClick={() => setDismissedTrace(trace)}
              className="rounded-full p-0.5 text-amber-200/60 transition-colors hover:bg-amber-400/10 hover:text-amber-100"
              title="Hide the run overlay"
              aria-label="Hide the Navigator run overlay"
            >
              <X size={11} />
            </button>
          </div>
        )}

        {/* Legend */}
        {status === "ready" && data && (
          <div className="absolute bottom-3 left-3 z-10 max-w-[280px] rounded-lg border border-[#27272a] bg-[#09090b]/85 px-2.5 py-2 backdrop-blur-md">
            <div className="flex flex-col gap-1 text-[10px] font-medium text-[#d4d4d8]">
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                {Object.keys(PROC_TYPE_COLORS).map((type) => (
                  <span
                    key={type}
                    className="flex items-center gap-1.5"
                    title={
                      type === "ACTION"
                        ? "A tool call"
                        : type === "REASONING"
                          ? "A reasoning step"
                          : "A status marker"
                    }
                  >
                    <TypeSwatch type={type} size={9} />
                    <span className="font-mono">{type}</span>
                  </span>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[#a1a1aa]">
                <span className="flex items-center gap-1.5" title="Where every run starts">
                  <span
                    className="h-2.5 w-2.5 rounded-full border-[1.5px]"
                    style={{ borderColor: PROC_START_COLOR }}
                  />
                  Start
                </span>
                <span className="flex items-center gap-1.5" title="No outgoing transitions">
                  <span
                    className="h-2.5 w-2.5 rounded-full border"
                    style={{ borderColor: PROC_TERMINAL_COLOR }}
                  />
                  Terminal
                </span>
                <span
                  className="flex items-center gap-1.5"
                  title="The transition has a natural-language precondition"
                >
                  <span className="w-3 border-t border-dashed border-[#a1a1aa]" />
                  Conditional
                </span>
              </div>
              {presentRelations.length > 0 && (
                <div className="flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[9px] uppercase tracking-wide">
                  {presentRelations.map((rel) => (
                    <span key={rel} style={{ color: colorForRelation(rel) }}>
                      {rel}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <p className="mt-1.5 border-t border-[#27272a] pt-1.5 text-[10px] leading-snug text-[#52525b]">
              Hover a node: its {GUIDANCE_HOPS}-hop guidance scope. Click a transition:
              condition, guidance, pitfalls.
            </p>
          </div>
        )}

        {/* Right column: detail card + version history */}
        {status === "ready" && data && (
          <div className="pointer-events-none absolute bottom-3 right-3 top-3 z-10 flex w-[300px] flex-col gap-2">
            {(selectedNode || selectedLink) && (
              <div className="pointer-events-auto flex min-h-0 shrink flex-col overflow-hidden rounded-lg border border-[#27272a] bg-[#09090b]/90 backdrop-blur-md">
                <div className="flex items-center justify-between border-b border-[#27272a] px-3 py-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
                    {selectedLink ? "Transition" : "Node"}
                  </span>
                  <button
                    onClick={clearSelection}
                    className="rounded-md p-0.5 text-[#71717a] transition-colors hover:bg-[#27272a] hover:text-[#fafafa]"
                    title="Close"
                    aria-label="Close details"
                  >
                    <X size={12} />
                  </button>
                </div>

                <div className="flex min-h-0 flex-col gap-3 overflow-y-auto px-3 py-2.5">
                  {selectedLink && (
                    <>
                      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
                        {[linkEndId(selectedLink.source), linkEndId(selectedLink.target)].map(
                          (id, i) => (
                            <span key={`${id}:${i}`} className="flex items-center gap-1.5">
                              {i === 1 && <ArrowRight size={11} className="text-[#52525b]" />}
                              <button
                                onClick={() => selectNode(id)}
                                title="Show this node"
                                className="rounded border border-[#27272a] bg-[#18181b] px-1.5 py-0.5 text-[#e4e4e7] transition-colors hover:border-indigo-500/50 hover:text-[#fafafa]"
                              >
                                {id}
                              </button>
                            </span>
                          ),
                        )}
                      </div>
                      <div>
                        <SectionLabel>Relation</SectionLabel>
                        <span
                          className="inline-flex rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium"
                          style={{
                            color: colorForRelation(selectedLink.relation),
                            borderColor: `${colorForRelation(selectedLink.relation)}40`,
                            backgroundColor: `${colorForRelation(selectedLink.relation)}12`,
                          }}
                        >
                          {selectedLink.relation}
                        </span>
                      </div>
                      <div>
                        <SectionLabel>Condition</SectionLabel>
                        <p className="rounded-md border border-[#27272a] bg-[#18181b] px-2.5 py-2 text-[12px] leading-relaxed text-[#d4d4d8]">
                          {selectedLink.condition || (
                            <span className="text-[#71717a]">
                              Unconditional: always available from here.
                            </span>
                          )}
                        </p>
                      </div>
                      <div>
                        <SectionLabel>Guidance</SectionLabel>
                        <p className="rounded-md border border-indigo-500/20 bg-indigo-500/[0.06] px-2.5 py-2 text-[12px] leading-relaxed text-[#d4d4d8]">
                          {selectedLink.guidance || <span className="text-[#71717a]">—</span>}
                        </p>
                      </div>
                      <div>
                        <SectionLabel>Pitfalls to avoid</SectionLabel>
                        <p className="rounded-md border border-rose-500/20 bg-rose-500/[0.06] px-2.5 py-2 text-[12px] leading-relaxed text-rose-100/80">
                          {selectedLink.pitfalls || <span className="text-[#71717a]">—</span>}
                        </p>
                      </div>
                    </>
                  )}

                  {selectedNode && (
                    <>
                      <div className="flex items-center gap-2">
                        <TypeSwatch type={selectedNode.type} size={11} />
                        <span className="break-all font-mono text-[13px] font-medium text-[#fafafa]">
                          {selectedNode.label || selectedNode.id}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        <span
                          className="inline-flex rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium"
                          style={{
                            color: colorForProcType(selectedNode.type),
                            borderColor: `${colorForProcType(selectedNode.type)}40`,
                            backgroundColor: `${colorForProcType(selectedNode.type)}12`,
                          }}
                        >
                          {selectedNode.type}
                        </span>
                        {selectedNode.is_start && (
                          <span className="inline-flex rounded border border-emerald-400/30 bg-emerald-400/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300/90">
                            Start
                          </span>
                        )}
                        {selectedNode.is_terminal && (
                          <span className="inline-flex rounded border border-[#3f3f46] bg-[#27272a]/60 px-1.5 py-0.5 text-[10px] font-medium text-[#e4e4e7]">
                            Terminal
                          </span>
                        )}
                        {traceVisits?.get(selectedNode.id) && (
                          <span className="inline-flex rounded border border-amber-400/30 bg-amber-400/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-amber-200/90">
                            step {traceVisits.get(selectedNode.id)!.join(", ")}
                          </span>
                        )}
                      </div>
                      <div>
                        <SectionLabel>Description</SectionLabel>
                        <p className="rounded-md border border-[#27272a] bg-[#18181b] px-2.5 py-2 text-[12px] leading-relaxed text-[#d4d4d8]">
                          {selectedNode.description || (
                            <span className="text-[#71717a]">No description.</span>
                          )}
                        </p>
                      </div>
                      {outgoing.length > 0 && (
                        <div>
                          <SectionLabel>Transitions out (hop 1)</SectionLabel>
                          <div className="flex flex-col gap-1">
                            {outgoing.map((link, i) => (
                              <button
                                key={`${linkEndId(link.target)}:${link.relation}:${i}`}
                                onClick={() => selectLink(link)}
                                className="flex flex-col gap-0.5 rounded-md border border-[#27272a] bg-[#18181b] px-2 py-1.5 text-left transition-colors hover:border-indigo-500/40"
                              >
                                <span className="flex items-center gap-1.5 font-mono text-[11px] text-[#e4e4e7]">
                                  <ArrowRight size={10} className="shrink-0 text-indigo-400/70" />
                                  <span className="truncate">{linkEndId(link.target)}</span>
                                  <span
                                    className="ml-auto shrink-0 text-[9px] uppercase"
                                    style={{ color: colorForRelation(link.relation) }}
                                  >
                                    {link.relation}
                                  </span>
                                </span>
                                {link.condition && (
                                  <span className="line-clamp-2 pl-4 text-[10px] leading-snug text-[#71717a]">
                                    {link.condition}
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      <p className="text-[10px] leading-snug text-[#52525b]">
                        Highlighted: the {GUIDANCE_HOPS}-hop outgoing scope the
                        Navigator is shown when its last action lands here.
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Version history */}
            <div className="pointer-events-auto mt-auto flex max-h-[45%] min-h-0 shrink-0 flex-col overflow-hidden rounded-lg border border-[#27272a] bg-[#09090b]/90 backdrop-blur-md">
              <button
                onClick={() => setVersionsOpen(!versionsExpanded)}
                aria-expanded={versionsExpanded}
                className="group flex w-full items-center gap-1.5 px-3 py-2 text-left"
              >
                <ChevronDown
                  size={12}
                  className={`text-[#52525b] transition-transform duration-200 group-hover:text-[#a1a1aa] ${
                    versionsExpanded ? "" : "-rotate-90"
                  }`}
                />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#a1a1aa] transition-colors group-hover:text-[#e4e4e7]">
                  Versions
                </span>
                {versions.length > 0 && (
                  <span className="rounded border border-[#27272a] px-1 font-mono text-[10px] leading-[15px] text-[#52525b]">
                    {versions.length}
                  </span>
                )}
              </button>

              {versionsExpanded && (
                <div className="min-h-0 overflow-y-auto border-t border-[#27272a]">
                  {versions.length === 0 ? (
                    <p className="px-3 py-2 text-[11px] text-[#52525b]">No history recorded.</p>
                  ) : (
                    <ul className="flex flex-col divide-y divide-[#27272a]/60">
                      {versions.map((v) => {
                        const live = v.version === liveVersion;
                        const diff = diffSummary(v.diff);
                        return (
                          <li key={v.version} className="flex items-start gap-2 px-3 py-1.5">
                            <span
                              className={`mt-px w-7 shrink-0 font-mono text-[11px] font-medium ${
                                live ? "text-indigo-300" : "text-[#e4e4e7]"
                              }`}
                            >
                              v{v.version}
                            </span>
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5 font-mono text-[10px]">
                                <span className="text-[#a1a1aa]" title="Validation score">
                                  {formatScore(v.score)}
                                </span>
                                {diff && (
                                  <span className="text-[#52525b]" title={diff.title}>
                                    {diff.text}
                                  </span>
                                )}
                                {live && (
                                  <span className="rounded border border-indigo-500/30 bg-indigo-500/10 px-1 text-[9px] uppercase tracking-wider text-indigo-300/90">
                                    live
                                  </span>
                                )}
                              </div>
                              <div
                                className="truncate text-[11px] text-[#a1a1aa]"
                                title={v.note || undefined}
                              >
                                {v.note || "—"}
                              </div>
                              <div className="font-mono text-[10px] text-[#52525b]">
                                {formatWhen(v.created_at)}
                              </div>
                            </div>
                            {!live && (
                              <button
                                onClick={() => askRollback(v.version)}
                                title={`Restore v${v.version} as a new version`}
                                className="mt-px flex shrink-0 items-center gap-1 rounded border border-[#27272a] bg-[#18181b] px-1.5 py-0.5 text-[10px] font-medium text-[#a1a1aa] transition-colors hover:border-amber-400/40 hover:text-[#fafafa]"
                              >
                                <Undo2 size={10} />
                                Rollback
                              </button>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Rollback confirmation */}
      {pendingRollback != null && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center bg-[#09090b]/70 px-4 backdrop-blur-sm"
          onClick={() => !rollingBack && setPendingRollback(null)}
        >
          <div
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="rollback-title"
            aria-describedby="rollback-description"
            onClick={(e) => e.stopPropagation()}
            className="msg-enter w-full max-w-[360px] rounded-lg border border-[#27272a] bg-[#18181b] p-4 shadow-2xl"
          >
            <h3
              id="rollback-title"
              className="flex items-center gap-2 text-[14px] font-semibold text-[#fafafa]"
            >
              <Undo2 size={14} className="text-amber-400" />
              Roll back to v{pendingRollback}?
            </h3>
            <p
              id="rollback-description"
              className="mt-2 text-[12px] leading-relaxed text-[#a1a1aa]"
            >
              The graph from v{pendingRollback} of{" "}
              <span className="font-mono text-[#d4d4d8]">{selected}</span> is saved
              as a new version, v{nextVersion}, and becomes live. Nothing is
              deleted: v{liveVersion} stays in the history and can be restored the
              same way.
            </p>
            {rollbackError && (
              <p className="mt-2 break-words text-[11px] leading-snug text-rose-500/90">
                {rollbackError}
              </p>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <button
                ref={cancelRef}
                onClick={() => setPendingRollback(null)}
                disabled={rollingBack}
                className="rounded-md border border-[#27272a] px-3 py-1.5 text-[12px] font-medium text-[#a1a1aa] transition-colors hover:bg-[#27272a] hover:text-[#fafafa] disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmRollback}
                disabled={rollingBack}
                className="flex items-center gap-1.5 rounded-md border border-transparent bg-[#fafafa] px-3 py-1.5 text-[12px] font-semibold text-[#09090b] transition-colors hover:bg-[#e4e4e7] disabled:opacity-60"
              >
                {rollingBack ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Undo2 size={12} />
                )}
                Roll back
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
