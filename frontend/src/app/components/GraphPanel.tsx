"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Search, Share2 } from "lucide-react";
import type {
  ForceGraphMethods,
  LinkObject,
  NodeObject,
} from "react-force-graph-2d";
import { clearGraph, fetchGraph } from "../lib/api";
import { colorForType } from "../lib/constants";
import type { GraphData, GraphLink, GraphNode } from "../lib/types";

type SNode = NodeObject<GraphNode>;
type SLink = LinkObject<GraphNode, GraphLink>;
type FGMethods = ForceGraphMethods<SNode, SLink>;
type ForceGraphComponent = (typeof import("react-force-graph-2d"))["default"];

/** Ring color for chat citations (indigo) vs. a selected community (amber). */
const CITATION_RING = "#818cf8";
const COMMUNITY_RING = "#fbbf24";
const SEARCH_RING = "#fafafa";

interface GraphPanelProps {
  ingesting: boolean;
  refreshSignal: number;
  clearSignal: number;
  highlightedNames: string[];
  /** Member names of the community selected in ThemesPanel, if any. */
  communityNames?: string[];
  /** Title of that community, shown as an overlay badge. */
  communityTitle?: string | null;
  focusTarget: { name: string; nonce: number } | null;
  onNodeSelected: (node: GraphNode | null) => void;
  onClearComplete?: () => void;
}

function linkEndId(
  end: string | number | { id?: string | number } | undefined | null,
): string {
  if (end == null) return "";
  return typeof end === "object" ? String(end.id ?? "") : String(end);
}

function graphSignature(data: GraphData): string {
  return `${data.nodes.length}:${data.links.length}:${data.nodes
    .map((n) => n.id)
    .join(",")}`;
}

export default function GraphPanel({
  ingesting,
  refreshSignal,
  clearSignal,
  highlightedNames,
  communityNames,
  communityTitle,
  focusTarget,
  onNodeSelected,
  onClearComplete,
}: GraphPanelProps) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [ForceGraph, setForceGraph] = useState<ForceGraphComponent | null>(null);
  const [dims, setDims] = useState({ width: 800, height: 600 });
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [hoverId, setHoverId] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<FGMethods | undefined>(undefined);
  const sigRef = useRef<string>("");
  const fitRef = useRef(false);

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

  // ── Fetch (diffed so unchanged data never re-heats the sim) ──
  const load = useCallback(async () => {
    try {
      const data = await fetchGraph();
      const sig = graphSignature(data);
      if (sig !== sigRef.current) {
        sigRef.current = sig;
        fitRef.current = false; // re-fit once the new layout settles
        setGraphData(data);
      }
    } catch (err) {
      console.error("Failed to fetch graph data:", err);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshSignal]);

  // Poll frequently while ingesting (watch it build), slowly when idle.
  useEffect(() => {
    const interval = setInterval(load, ingesting ? 2000 : 15000);
    return () => clearInterval(interval);
  }, [load, ingesting]);

  // ── Clear DB when the signal bumps ──
  useEffect(() => {
    if (clearSignal <= 0) return;
    (async () => {
      try {
        await clearGraph();
        sigRef.current = "";
        setGraphData({ nodes: [], links: [] });
        onNodeSelected(null);
        onClearComplete?.();
      } catch (err) {
        console.error("Failed to clear graph:", err);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearSignal]);

  // ── Adjacency for neighborhood highlighting ──
  const neighbors = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const link of graphData.links) {
      const s = linkEndId(link.source);
      const t = linkEndId(link.target);
      if (!map.has(s)) map.set(s, new Set());
      if (!map.has(t)) map.set(t, new Set());
      map.get(s)!.add(t);
      map.get(t)!.add(s);
    }
    return map;
  }, [graphData]);

  // Spread nodes out (stronger repulsion + link distance) for legible labels.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || graphData.nodes.length === 0) return;
    const charge = fg.d3Force("charge");
    if (charge?.strength) charge.strength(-160);
    const link = fg.d3Force("link");
    if (link?.distance) link.distance(70);
    fg.d3ReheatSimulation?.();
  }, [ForceGraph, graphData.nodes.length]);

  const presentTypes = useMemo(
    () => Array.from(new Set(graphData.nodes.map((n) => n.type))).sort(),
    [graphData],
  );

  const citedSet = useMemo(
    () => new Set(highlightedNames.map((n) => n.toLowerCase())),
    [highlightedNames],
  );

  // Same mechanism as citations, different accent — so a user can tell "the
  // answer used this" apart from "this theme contains this".
  const communitySet = useMemo(
    () => new Set((communityNames ?? []).map((n) => n.toLowerCase())),
    [communityNames],
  );

  const searchLower = search.trim().toLowerCase();

  // Frame the selected community once, so picking a theme actually *shows* it.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || communitySet.size === 0) return;
    const inCommunity = (node: SNode) =>
      communitySet.has(node.label.toLowerCase());
    if (graphData.nodes.some(inCommunity)) fg.zoomToFit(600, 80, inCommunity);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [communitySet]);

  // ── Focus a node when a citation chip / inspector button fires ──
  useEffect(() => {
    if (!focusTarget || !fgRef.current) return;
    const node = graphData.nodes.find(
      (n) => n.label.toLowerCase() === focusTarget.name.toLowerCase(),
    );
    if (node && node.x != null && node.y != null) {
      fgRef.current.centerAt(node.x, node.y, 600);
      fgRef.current.zoom(4, 600);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusTarget]);

  const toggleType = (type: string) => {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const isTypeActive = (type: string) =>
    activeTypes.size === 0 || activeTypes.has(type);

  const zoomToFit = () => fgRef.current?.zoomToFit(500, 60);

  // ── Canvas rendering ──
  const drawNode = useCallback(
    (node: SNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const active = activeTypes.size === 0 || activeTypes.has(node.type);
      const cited = citedSet.has(node.label.toLowerCase());
      const inCommunity = communitySet.has(node.label.toLowerCase());
      const searchHit =
        searchLower !== "" && node.label.toLowerCase().includes(searchLower);
      const neighborHi =
        hoverId != null &&
        (hoverId === node.id || (neighbors.get(hoverId)?.has(node.id) ?? false));
      // A selected theme also dims everything outside it, so the cluster reads
      // as a shape rather than as a handful of scattered rings.
      const outsideCommunity = communitySet.size > 0 && !inCommunity;
      const emphasized = cited || searchHit || inCommunity;
      const dim =
        !active || (hoverId != null && !neighborHi) || (outsideCommunity && !cited);

      const color = colorForType(node.type);
      const r = emphasized ? 6 : 4.5;

      ctx.globalAlpha = dim ? 0.18 : 1;

      if (emphasized) {
        if (inCommunity) {
          // Soft halo under the ring — the "premium" cue for a picked theme.
          ctx.beginPath();
          ctx.arc(x, y, r + 6, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(251, 191, 36, 0.12)";
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(x, y, r + 3.5, 0, 2 * Math.PI);
        ctx.strokeStyle = cited
          ? CITATION_RING
          : inCommunity
            ? COMMUNITY_RING
            : SEARCH_RING;
        ctx.lineWidth = inCommunity ? 1.8 : 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#09090b";
      ctx.stroke();

      const fontSize = Math.max(10 / scale, 2.6);
      if (scale > 1.2 || emphasized || neighborHi) {
        ctx.font = `500 ${fontSize}px var(--font-inter), sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = dim ? "#52525b" : "#e4e4e7";
        ctx.fillText(node.label, x, y + r + 2);
      }
      ctx.globalAlpha = 1;
    },
    [activeTypes, citedSet, communitySet, searchLower, hoverId, neighbors],
  );

  const linkColor = useCallback(
    (link: SLink) => {
      const s = linkEndId(link.source);
      const t = linkEndId(link.target);
      if (hoverId != null) {
        return hoverId === s || hoverId === t ? "#6366f1" : "#ffffff08";
      }
      return "#27272a";
    },
    [hoverId],
  );

  const linkWidth = useCallback(
    (link: SLink) => {
      if (hoverId == null) return 1;
      const s = linkEndId(link.source);
      const t = linkEndId(link.target);
      return hoverId === s || hoverId === t ? 2 : 0.5;
    },
    [hoverId],
  );

  return (
    <div className="relative flex h-full w-full flex-col bg-[#09090b]">
      {/* Top bar */}
      <div className="absolute left-0 right-0 top-0 z-10 flex items-center justify-between border-b border-[#27272a]/50 bg-[#09090b]/80 px-5 py-2.5 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <Share2 size={14} className="text-[#a1a1aa]" />
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[#e4e4e7]">
            Topology
          </h2>
          {communityTitle && communitySet.size > 0 && (
            <div className="ml-1.5 flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 py-0.5 pl-1.5 pr-2.5">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: COMMUNITY_RING }}
              />
              <span className="max-w-[220px] truncate text-[11px] font-medium text-amber-200/90">
                {communityTitle}
              </span>
              <span className="font-mono text-[10px] text-amber-200/50">
                {communitySet.size}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search
              size={12}
              className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[#71717a]"
            />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search nodes…"
              aria-label="Search nodes"
              className="w-36 rounded border border-[#27272a] bg-[#18181b] py-1 pl-6 pr-2 text-[11px] text-[#fafafa] outline-none placeholder:text-[#52525b] focus:border-indigo-500/50"
            />
          </div>
          <button
            onClick={zoomToFit}
            className="rounded border border-[#27272a] bg-[#18181b] p-1.5 text-[#a1a1aa] transition-colors hover:text-[#fafafa]"
            title="Fit to view"
            aria-label="Fit graph to view"
          >
            <Maximize2 size={12} />
          </button>
          {graphData.nodes.length > 0 && (
            <div className="rounded border border-[#27272a] bg-[#18181b] px-2.5 py-1 text-[11px] font-medium text-[#71717a]">
              {graphData.nodes.length} N / {graphData.links.length} E
            </div>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="graph-grid relative mt-[41px] w-full flex-1" ref={containerRef}>
        {ForceGraph && graphData.nodes.length > 0 && (
          <ForceGraph
            ref={fgRef}
            graphData={graphData}
            width={dims.width}
            height={dims.height}
            nodeLabel={(node: SNode) => `${node.label} · ${node.type}`}
            nodeCanvasObject={drawNode}
            nodePointerAreaPaint={(
              node: SNode,
              color: string,
              ctx: CanvasRenderingContext2D,
            ) => {
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(node.x ?? 0, node.y ?? 0, 6, 0, 2 * Math.PI);
              ctx.fill();
            }}
            linkColor={linkColor}
            linkWidth={linkWidth}
            linkDirectionalArrowLength={2.5}
            linkDirectionalArrowRelPos={1}
            linkLabel={(link: SLink) => link.type}
            onNodeClick={(node: SNode) => onNodeSelected(node)}
            onNodeHover={(node: SNode | null) =>
              setHoverId(node ? node.id : null)
            }
            onBackgroundClick={() => onNodeSelected(null)}
            onEngineStop={() => {
              // Frame the connected core once the layout settles. Stray degree-0
              // nodes drift far out; framing them would shrink the graph to a dot.
              if (!fitRef.current) {
                const hasEdges = (n: SNode) =>
                  (neighbors.get(n.id)?.size ?? 0) > 0;
                const anyConnected = graphData.nodes.some(hasEdges);
                fgRef.current?.zoomToFit(
                  500,
                  90,
                  anyConnected ? hasEdges : undefined,
                );
                fitRef.current = true;
              }
            }}
            backgroundColor="transparent"
            d3VelocityDecay={0.35}
            warmupTicks={80}
            cooldownTicks={120}
          />
        )}

        {graphData.nodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-[#71717a]">
            <Share2 size={32} className="opacity-40" strokeWidth={1.5} />
            <p className="text-[13px] font-medium">
              {ingesting ? "Building graph…" : "Ready for dataset ingestion."}
            </p>
          </div>
        )}

        {/* Interactive legend / filter */}
        {presentTypes.length > 0 && (
          <div className="absolute bottom-3 left-3 z-10 max-w-[240px] rounded-lg border border-[#27272a] bg-[#09090b]/85 p-2.5 backdrop-blur-md">
            <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-[#71717a]">
              Filter by type
            </div>
            <div className="flex flex-wrap gap-1">
              {presentTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleType(type)}
                  className={`flex items-center gap-1.5 rounded px-1.5 py-1 text-[10px] font-medium transition-opacity hover:bg-[#27272a] ${
                    isTypeActive(type) ? "opacity-100" : "opacity-35"
                  }`}
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: colorForType(type) }}
                  />
                  <span className="text-[#d4d4d8]">{type}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
