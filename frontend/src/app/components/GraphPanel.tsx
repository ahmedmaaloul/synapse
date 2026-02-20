"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Database, Share2 } from "lucide-react";

interface GraphNode {
    id: string;
    label: string;
    type: string;
    properties: Record<string, any>;
}

interface GraphLink {
    source: string;
    target: string;
    type: string;
    properties: Record<string, any>;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface GraphPanelProps {
    clearTrigger?: number;
    onNodeSelected?: (node: any) => void;
}

export default function GraphPanel({ clearTrigger, onNodeSelected }: GraphPanelProps) {
    const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
        nodes: [],
        links: [],
    });
    const containerRef = useRef<HTMLDivElement>(null);
    const [ForceGraph, setForceGraph] = useState<any>(null);

    // Dynamic import
    useEffect(() => {
        import("react-force-graph-2d").then((mod) => setForceGraph(() => mod.default));
    }, []);

    const fetchGraph = useCallback(async () => {
        try {
            const res = await fetch(`${API_URL}/api/graph-data`);
            if (res.ok) {
                const data = await res.json();
                setGraphData(data);
            }
        } catch (err) {
            console.error("Failed to fetch graph data:", err);
        }
    }, []);

    useEffect(() => {
        fetchGraph();
        const interval = setInterval(fetchGraph, 5000);
        return () => clearInterval(interval);
    }, [fetchGraph]);

    useEffect(() => {
        if (clearTrigger && clearTrigger > 0) {
            const clear = async () => {
                try {
                    const res = await fetch(`${API_URL}/api/graph`, { method: "DELETE" });
                    if (res.ok) {
                        setGraphData({ nodes: [], links: [] });
                        if (onNodeSelected) onNodeSelected(null);
                    }
                } catch (err) {
                    console.error("Failed to clear graph:", err);
                }
            };
            clear();
        }
    }, [clearTrigger, onNodeSelected]);

    // High Density Modern SaaS Palette (Zinc/Indigo/Emerald)
    const typeColors: Record<string, string> = {
        PERSON: "#fafafa",       // Zinc 50
        ORGANIZATION: "#6366f1", // Indigo 500
        COMPANY: "#6366f1",      // Indigo 500
        UNIVERSITY: "#0ea5e9",   // Sky 500
        ROLE: "#10b981",         // Emerald 500
        PROJECT: "#f59e0b",      // Amber 500
        SKILL: "#e4e4e7",        // Zinc 200
        TOOL: "#06b6d4",         // Cyan 500
        LANGUAGE: "#a855f7",     // Purple 500
        CERTIFICATION: "#f43f5e",// Rose 500
        LOCATION: "#71717a",     // Zinc 500
        EDUCATION: "#3b82f6",    // Blue 500
    };

    const getColor = (type: string) => typeColors[type] || "#71717a";

    return (
        <div className="w-full h-full flex flex-col bg-[#09090b] relative">
            {/* Minimalist Top Indicator */}
            <div className="absolute top-0 left-0 right-0 z-10 bg-[#09090b]/80 backdrop-blur-md px-5 py-2.5 flex justify-between items-center shadow-sm border-b border-[#27272a]/50">
                <div className="flex items-center gap-2">
                    <Database size={14} className="text-[#a1a1aa]" />
                    <h2 className="text-[12px] font-semibold text-[#e4e4e7] uppercase tracking-wider">Topology</h2>
                </div>

                {graphData.nodes.length > 0 && (
                    <div className="text-[11px] font-medium text-[#71717a] bg-[#18181b] px-2.5 py-1 rounded border border-[#27272a]">
                        {graphData.nodes.length} N / {graphData.links.length} E
                    </div>
                )}
            </div>

            {/* Canvas */}
            <div className="flex-1 w-full relative mt-9" ref={containerRef}>
                {ForceGraph && graphData.nodes.length > 0 && (
                    <ForceGraph
                        graphData={graphData}
                        width={containerRef.current?.clientWidth || 800}
                        height={(containerRef.current?.clientHeight || 600)}
                        nodeLabel={(node: GraphNode) => node.label}
                        nodeColor={(node: GraphNode) => getColor(node.type)}
                        nodeRelSize={5} /* slightly smaller for higher density */
                        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                            const label = node.label || "";
                            const fontSize = Math.max(10 / globalScale, 2.5); /* sharper, smaller text */
                            const r = 4.5;

                            ctx.beginPath();
                            ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
                            ctx.fillStyle = getColor(node.type);
                            ctx.fill();

                            // High contrast subtle stroke
                            ctx.lineWidth = 1;
                            ctx.strokeStyle = "#09090b";
                            ctx.stroke();

                            ctx.font = `500 ${fontSize}px Inter, sans-serif`;
                            ctx.textAlign = "center";
                            ctx.textBaseline = "top";
                            ctx.fillStyle = "#fafafa";
                            ctx.fillText(label, node.x!, node.y! + r + 3);
                        }}
                        linkColor={() => "#27272a"}
                        linkWidth={1}
                        linkDirectionalArrowLength={2.5}
                        linkDirectionalArrowRelPos={1}
                        linkLabel={(link: GraphLink) => link.type}
                        onNodeClick={(node: any) => onNodeSelected && onNodeSelected(node)}
                        backgroundColor="transparent"
                        d3VelocityDecay={0.4}
                        warmupTicks={100}
                    />
                )}
                {graphData.nodes.length === 0 && (
                    <div className="absolute inset-0 flex flex-col flex-1 items-center justify-center text-[#71717a] gap-3">
                        <Share2 size={32} className="opacity-40" strokeWidth={1.5} />
                        <p className="text-[13px] font-medium">Ready for dataset ingestion.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
