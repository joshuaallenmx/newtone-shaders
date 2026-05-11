import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type PointerEvent as ReactPointerEvent,
    type ReactNode,
} from "react";
import type { Edge, Node } from "@xyflow/react";
import { compileChain, resolveSlotUpstream } from "../compileChain";
import { findAssetByName } from "../assets";
import {
    getAssetDims,
    loadAssetDims,
    subscribeAssetDims,
} from "../assetDims";
import { ChainRenderer, SHADERS } from "../shaders";
import { usePipelineHandle } from "../PipelineHandleContext";
import { PREVIEW_LONG } from "../../playground-next/pipeline/Pipeline";
import { readOutputParams } from "../nodes/OutputNode";
import { NsfwStatusOverlay } from "../NsfwStatusOverlay";
import { NsfwDetectOverlay } from "../NsfwDetectOverlay";

/** IDs kept in sync with the corresponding shader entries in
 *  playground-next/shaders/. */
const NSFW_COMPARE_ID = "nsfwCompare";
const NSFW_DETECT_ID = "nsfwDetect";
import {
    BUTTON_STYLE,
    EMPTY_STYLE,
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";

const STAGE_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    position: "relative",
    overflow: "hidden",
    background: "#0a0a0a",
    touchAction: "none",
    cursor: "grab",
};

const HEADER_ROW_STYLE: CSSProperties = {
    ...PANEL_HEADER_STYLE,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
};

const ZOOM_BADGE_STYLE: CSSProperties = {
    ...BUTTON_STYLE,
    padding: "2px 8px",
    fontSize: 11,
    textTransform: "none",
    letterSpacing: 0,
    fontVariantNumeric: "tabular-nums",
};

const MIN_SCALE = 0.1;
const MAX_SCALE = 20;

/** Cap on snapshot readbacks per rAF tick. With a 20-node graph and
 *  K=4 the whole graph refreshes in ~80 ms after a parameter edit —
 *  visually instant. Each readback is sub-millisecond. */
const MAX_SNAPSHOTS_PER_FRAME = 4;

interface Transform {
    readonly scale: number;
    readonly x: number;
    readonly y: number;
}
const IDENTITY: Transform = { scale: 1, x: 0, y: 0 };

interface PreviewPanelProps {
    readonly nodes: readonly Node[];
    readonly edges: readonly Edge[];
    readonly paramsByNode: Record<string, unknown>;
}

export function PreviewPanel({ nodes, edges, paramsByNode }: PreviewPanelProps) {
    const chain = useMemo(
        () => compileChain(nodes, edges, paramsByNode, SHADERS),
        [nodes, edges, paramsByNode],
    );

    // Resolve the Global Input → its source asset → URL. Trigger an
    // async dim-load when we see a new URL, and re-render when any
    // asset's dims arrive.
    const globalInputAssetUrl = useMemo(() => {
        const gi = nodes.find((n) => n.type === "globalInput");
        if (!gi) return null;
        const upstreamId = resolveSlotUpstream(gi.id, nodes, edges);
        if (!upstreamId) return null;
        const upstream = nodes.find((n) => n.id === upstreamId);
        if (!upstream || upstream.type !== "source") return null;
        const data = upstream.data as { assetName?: string } | undefined;
        const asset = findAssetByName(data?.assetName);
        return asset?.url ?? null;
    }, [nodes, edges]);

    const inputAspect = useGlobalInputAspect(globalInputAssetUrl);

    // Working buffer dimensions. Aspect = Global Input source (1:1
    // fallback when no source is wired); long side = Output node's
    // resolution.
    const buffer = useMemo(() => {
        const outputNode = nodes.find((n) => n.type === "output");
        const params = readOutputParams(
            outputNode ? paramsByNode[outputNode.id] : undefined,
        );
        const aspect = inputAspect ?? 1;
        const longerSide = params.resolution;
        let bufferW: number;
        let bufferH: number;
        if (aspect >= 1) {
            bufferW = longerSide;
            bufferH = Math.max(1, Math.round(longerSide / aspect));
        } else {
            bufferH = longerSide;
            bufferW = Math.max(1, Math.round(longerSide * aspect));
        }
        return { bufferW, bufferH };
    }, [nodes, paramsByNode, inputAspect]);

    // Per-node thumbnail bridge. PipelineRenderer publishes its Pipeline
    // here on mount; we drain a small batch of dirty thumbnails after
    // every render. The chain memo's identity changes only when nodes,
    // edges, or paramsByNode change — so this useEffect fires exactly
    // when the world has actually changed.
    const { setPipeline, setPreviewSize, drain, markAllDirty } =
        usePipelineHandle();

    // Push the current preview canvas size into the context so node
    // thumbnails size their <canvas> elements 1:1 with the readback
    // (no resampling). Mirrors `Pipeline.setOutput`'s computation.
    useEffect(() => {
        const aspect = buffer.bufferW / buffer.bufferH;
        let pw: number;
        let ph: number;
        if (aspect >= 1) {
            pw = PREVIEW_LONG;
            ph = Math.max(1, Math.round(PREVIEW_LONG / aspect));
        } else {
            ph = PREVIEW_LONG;
            pw = Math.max(1, Math.round(PREVIEW_LONG * aspect));
        }
        setPreviewSize({ w: pw, h: ph });
    }, [buffer.bufferW, buffer.bufferH, setPreviewSize]);
    useEffect(() => {
        markAllDirty();
    }, [chain, markAllDirty]);
    const onPostRender = useCallback(() => {
        drain(MAX_SNAPSHOTS_PER_FRAME);
    }, [drain]);

    useEffect(() => {
        // One-line debug aid so the user can see why the preview is empty.
        // Removed by the dev-tools console filter if it's noisy.
        // eslint-disable-next-line no-console
        console.debug(
            "[preview] chain=",
            chain,
            "nodes=",
            nodes.length,
            "edges=",
            edges.length,
        );
    }, [chain, nodes.length, edges.length]);

    const unknownShaderIds = useMemo(() => {
        const known = new Set(SHADERS.map((s) => s.id));
        const found = new Set<string>();
        for (const n of nodes) {
            if (n.type !== "shader") continue;
            const data = n.data as { shaderId?: string } | undefined;
            const id = data?.shaderId;
            if (id && !known.has(id)) found.add(id);
        }
        return Array.from(found);
    }, [nodes]);

    const diagnostics = useMemo(
        () => diagnoseChain(nodes, edges),
        [nodes, edges],
    );

    const [transform, setTransform] = useState<Transform>(IDENTITY);
    const reset = useCallback(() => setTransform(IDENTITY), []);

    return (
        <div style={PANEL_STYLE}>
            <div style={HEADER_ROW_STYLE}>
                <span>Preview</span>
                <button
                    type="button"
                    style={ZOOM_BADGE_STYLE}
                    onClick={reset}
                    title="Reset zoom & pan (or double-click stage)"
                >
                    {Math.round(transform.scale * 100)}%
                </button>
            </div>
            <div style={{ ...PANEL_BODY_STYLE, padding: 0 }}>
                <PanZoomStage transform={transform} setTransform={setTransform}>
                    {chain ? (
                        <CanvasFrame aspectRatio={buffer.bufferW / buffer.bufferH}>
                            <ChainRenderer
                                chain={chain}
                                viewMode="fit"
                                bufferW={buffer.bufferW}
                                bufferH={buffer.bufferH}
                                onPipelineChange={setPipeline}
                                onPostRender={onPostRender}
                            />
                            {chain.entry?.id === NSFW_COMPARE_ID && (
                                <NsfwStatusOverlay chain={chain} />
                            )}
                            {chain.entry?.id === NSFW_DETECT_ID && (
                                <NsfwDetectOverlay chain={chain} />
                            )}
                        </CanvasFrame>
                    ) : unknownShaderIds.length > 0 ? (
                        <div style={EMPTY_STYLE}>
                            <div>
                                This project references shaders that no longer
                                exist:
                            </div>
                            <div
                                style={{
                                    marginTop: 8,
                                    fontFamily: "monospace",
                                    color: "#e2a84a",
                                }}
                            >
                                {unknownShaderIds.join(", ")}
                            </div>
                            <div style={{ marginTop: 8 }}>
                                Delete those nodes from the graph (or pick a
                                different project from the topbar) to recover.
                            </div>
                        </div>
                    ) : diagnostics.length > 0 ? (
                        <div style={EMPTY_STYLE}>
                            <div style={{ marginBottom: 8 }}>
                                Chain isn't ready to render:
                            </div>
                            <ul
                                style={{
                                    margin: 0,
                                    paddingLeft: 20,
                                    color: "#e2a84a",
                                    textAlign: "left",
                                }}
                            >
                                {diagnostics.map((msg, i) => (
                                    <li key={i}>{msg}</li>
                                ))}
                            </ul>
                        </div>
                    ) : (
                        <div style={EMPTY_STYLE}>
                            Drag a file onto the graph and connect it to Output
                        </div>
                    )}
                </PanZoomStage>
            </div>
        </div>
    );
}

interface CanvasFrameProps {
    /** W/H ratio (e.g. `2/3 ≈ 0.667`). The frame sizes itself to the
     *  largest box of this aspect that fits the stage, so the canvas
     *  inside it letterboxes correctly inside the panel. */
    readonly aspectRatio: number;
    readonly children: ReactNode;
}

const CANVAS_FRAME_CENTERING_STYLE: CSSProperties = {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
};

function CanvasFrame({ aspectRatio, children }: CanvasFrameProps) {
    // The inner box uses aspect-ratio + max-width/max-height to size
    // itself to the largest rectangle of the requested ratio that fits
    // the panel.
    const safe = Number.isFinite(aspectRatio) && aspectRatio > 0 ? aspectRatio : 1;
    const innerStyle: CSSProperties = {
        position: "relative",
        width: "100%",
        aspectRatio: `${safe}`,
        maxWidth: "100%",
        maxHeight: "100%",
    };
    return (
        <div style={CANVAS_FRAME_CENTERING_STYLE}>
            <div style={innerStyle}>{children}</div>
        </div>
    );
}

interface PanZoomStageProps {
    readonly transform: Transform;
    readonly setTransform: React.Dispatch<React.SetStateAction<Transform>>;
    readonly children: ReactNode;
}

function PanZoomStage({ transform, setTransform, children }: PanZoomStageProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<{ x: number; y: number } | null>(null);

    // Wheel handler must be non-passive to call preventDefault, so we attach
    // it via addEventListener instead of React's onWheel (which is passive).
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const handler = (e: WheelEvent) => {
            e.preventDefault();
            // ctrlKey covers trackpad pinch (Chrome/Safari/Firefox synthesize
            // ctrlKey on pinch) and mouse ctrl+wheel. Plain wheel pans.
            if (e.ctrlKey || e.metaKey) {
                const rect = el.getBoundingClientRect();
                const cx = e.clientX - rect.left;
                const cy = e.clientY - rect.top;
                const factor = Math.exp(-e.deltaY * 0.01);
                setTransform((prev) => {
                    const next = clampScale(prev.scale * factor);
                    const ratio = next / prev.scale;
                    return {
                        scale: next,
                        x: cx - (cx - prev.x) * ratio,
                        y: cy - (cy - prev.y) * ratio,
                    };
                });
            } else {
                setTransform((prev) => ({
                    scale: prev.scale,
                    x: prev.x - e.deltaX,
                    y: prev.y - e.deltaY,
                }));
            }
        };
        el.addEventListener("wheel", handler, { passive: false });
        return () => el.removeEventListener("wheel", handler);
    }, [setTransform]);

    const onPointerDown = useCallback(
        (e: ReactPointerEvent<HTMLDivElement>) => {
            if (e.button !== 0) return;
            dragRef.current = { x: e.clientX, y: e.clientY };
            e.currentTarget.setPointerCapture(e.pointerId);
        },
        [],
    );

    const onPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLDivElement>) => {
            const drag = dragRef.current;
            if (!drag) return;
            const dx = e.clientX - drag.x;
            const dy = e.clientY - drag.y;
            dragRef.current = { x: e.clientX, y: e.clientY };
            setTransform((prev) => ({
                scale: prev.scale,
                x: prev.x + dx,
                y: prev.y + dy,
            }));
        },
        [setTransform],
    );

    const onPointerUp = useCallback(
        (e: ReactPointerEvent<HTMLDivElement>) => {
            dragRef.current = null;
            try {
                e.currentTarget.releasePointerCapture(e.pointerId);
            } catch {
                // pointer wasn't captured; ignore
            }
        },
        [],
    );

    const onDoubleClick = useCallback(() => {
        setTransform(IDENTITY);
    }, [setTransform]);

    return (
        <div
            ref={containerRef}
            style={STAGE_STYLE}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onDoubleClick={onDoubleClick}
        >
            <div
                style={{
                    position: "absolute",
                    inset: 0,
                    transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
                    transformOrigin: "0 0",
                }}
            >
                {children}
            </div>
        </div>
    );
}

function clampScale(s: number): number {
    return Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
}


/** Walks the graph and lists structural reasons the chain wouldn't compile.
 *  Returns an empty array if everything looks wired up — in that case the
 *  null chain is genuinely an empty/disconnected graph. */
function diagnoseChain(
    nodes: readonly Node[],
    edges: readonly Edge[],
): string[] {
    const issues: string[] = [];

    const outputs = nodes.filter((n) => n.type === "output");
    if (outputs.length === 0) {
        issues.push("Add an Output node and connect a shader to it.");
    } else if (outputs.length > 1) {
        issues.push("Only one Output node is allowed; remove extras.");
    } else {
        const output = outputs[0]!;
        if (!edges.some((e) => e.target === output.id)) {
            issues.push("Output has nothing connected to it.");
        }
    }

    for (const node of nodes) {
        if (node.type === "source") {
            const data = node.data as { assetName?: string } | undefined;
            if (!data?.assetName) {
                issues.push(`Source ${node.id} has no asset selected.`);
            } else if (!findAssetByName(data.assetName)) {
                issues.push(
                    `Source ${node.id} references missing asset ` +
                        `"${data.assetName}" (file may have been removed ` +
                        `from playground/assets/, or the project came from ` +
                        `another machine).`,
                );
            }
            continue;
        }
        if (node.type !== "shader") continue;
        const data = node.data as { shaderId?: string } | undefined;
        const entry = SHADERS.find((s) => s.id === data?.shaderId);
        if (!entry) continue; // unknown shader handled by the other branch
        const declared = entry.inputs ?? [{ id: "in", label: "in" }];
        for (const input of declared) {
            const wired = edges.some(
                (e) =>
                    e.target === node.id &&
                    (e.targetHandle === input.id ||
                        (e.targetHandle == null && input.id === "in")),
            );
            if (!wired) {
                issues.push(
                    `${entry.name} (${node.id}) is missing input "${input.label ?? input.id}".`,
                );
            }
        }
    }

    return issues;
}

/** Resolve a URL → its image's natural width/height ratio. Lazy-loads
 *  the image on first request and re-renders the consumer when the
 *  load resolves (or any other URL's load resolves — the subscriber is
 *  shared, but we filter by URL). Returns `null` while the image is
 *  loading or when no URL is given. */
function useGlobalInputAspect(url: string | null): number | null {
    const [, force] = useState(0);
    useEffect(() => {
        if (!url) return;
        // Kick the load if not already cached. The promise's outcome
        // is propagated via subscribeAssetDims.
        loadAssetDims(url).catch((err) => {
            console.warn("[preview] global input asset failed to load:", err);
        });
        const unsub = subscribeAssetDims(() => force((v) => v + 1));
        return unsub;
    }, [url]);
    if (!url) return null;
    const dims = getAssetDims(url);
    if (!dims) return null;
    return dims.w / Math.max(1, dims.h);
}
