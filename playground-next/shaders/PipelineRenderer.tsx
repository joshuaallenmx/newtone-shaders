import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type PointerEvent as ReactPointerEvent,
} from "react";
import { Pipeline, flattenChain } from "../pipeline/Pipeline";
import type { ChainSpec, PointerFrame, ViewMode } from ".";

interface PipelineRendererProps {
    readonly chain: ChainSpec;
    readonly viewMode: ViewMode;
    /** Working buffer dimensions. Aspect = Global Input source's
     *  aspect; long side = Output node's resolution. The canvas backing
     *  buffer matches these 1:1, and every per-node FBO is sized to
     *  them. */
    readonly bufferW: number;
    readonly bufferH: number;
    /** Called with the live `Pipeline` instance after construction and
     *  with `null` on dispose. Lets the editor publish the instance into
     *  a registry context for per-node thumbnails. */
    readonly onPipelineChange?: (pipeline: Pipeline | null) => void;
    /** Called once per rAF tick, immediately after `pipeline.renderFrame`
     *  returns. Used by the editor to drain a dirty-queue of node
     *  thumbnails — running here guarantees readbacks see the freshest
     *  outputTex contents. */
    readonly onPostRender?: () => void;
}

const HOST_FIT: CSSProperties = {
    position: "relative",
    width: "100%",
    height: "100%",
    background: "#0a0a0a",
};

const HOST_ACTUAL: CSSProperties = {
    position: "relative",
    display: "inline-block",
    background: "#0a0a0a",
};

// Canvas backing buffer is sized by the Output node (via Pipeline.setOutput).
// `object-fit: contain` letterboxes the canvas inside its parent box without
// distorting it — the parent (CanvasFrame) provides the aspect-correct slot.
const CANVAS_FIT: CSSProperties = {
    width: "100%",
    height: "100%",
    display: "block",
    objectFit: "contain",
};

const CANVAS_ACTUAL: CSSProperties = { display: "block" };

const HUD_STYLE: CSSProperties = {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#bdbdbd",
    fontSize: 13,
    fontVariantNumeric: "tabular-nums",
    background: "rgba(0,0,0,0.35)",
    pointerEvents: "none",
};

interface PipelineStatus {
    readonly producersInFlight: number;
    readonly latestProgress: unknown;
    readonly error: string | null;
}

const IDLE_STATUS: PipelineStatus = {
    producersInFlight: 0,
    latestProgress: null,
    error: null,
};

function describeProgress(p: unknown): string {
    if (!p || typeof p !== "object") return "loading…";
    const obj = p as Record<string, unknown>;
    const status = typeof obj.status === "string" ? obj.status : null;
    const file = typeof obj.file === "string" ? obj.file : null;
    const progress =
        typeof obj.progress === "number" ? obj.progress : null;
    if (status && file && progress != null) {
        return `${status} ${file} — ${Math.round(progress)}%`;
    }
    if (status) return status;
    return "loading…";
}

/** Stable string of the chain's structural shape — node ids, entry ids,
 *  input topology, source URLs. Param-only changes leave this alone, which
 *  is what we want: the pipeline rebuilds only when topology changes. */
function structuralDigest(chain: ChainSpec): string {
    const parts: string[] = [];
    const visit = (c: ChainSpec) => {
        if (c.entry === null) {
            parts.push(`src:${c.nodeId ?? ""}:${c.src ?? ""}`);
            return;
        }
        parts.push(
            `node:${c.nodeId ?? ""}:${c.entry.id}:${c.inputs.length}:${c.src ?? ""}`,
        );
        for (const upstream of c.inputs) visit(upstream);
    };
    visit(chain);
    return parts.join("|");
}

function paramsFromChain(chain: ChainSpec): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    const visit = (c: ChainSpec) => {
        if (c.nodeId != null && c.entry !== null) {
            out[c.nodeId] = c.params;
        }
        for (const upstream of c.inputs) visit(upstream);
    };
    visit(chain);
    return out;
}

export function PipelineRenderer({
    chain,
    viewMode,
    bufferW,
    bufferH,
    onPipelineChange,
    onPostRender,
}: PipelineRendererProps) {
    const hostRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const pipelineRef = useRef<Pipeline | null>(null);
    const startMsRef = useRef(performance.now());
    const paramsRef = useRef<Record<string, unknown>>({});
    paramsRef.current = paramsFromChain(chain);

    // Callback refs let the rAF loop and mount effect read the latest
    // callbacks without re-running the mount effect when they change.
    const onPipelineChangeRef = useRef(onPipelineChange);
    const onPostRenderRef = useRef(onPostRender);
    onPipelineChangeRef.current = onPipelineChange;
    onPostRenderRef.current = onPostRender;

    // Pointer state lives in a mutable ref so the rAF tick can read it
    // without re-rendering on every move.
    const pointerRef = useRef<PointerFrame>({
        uv: [0.5, 0.5],
        active: false,
    });

    const digest = useMemo(() => structuralDigest(chain), [chain]);
    const [status, setStatus] = useState<PipelineStatus>(IDLE_STATUS);

    // Mount: create pipeline, start the rAF loop. Sizing comes from
    // Output params (effect below) — no ResizeObserver needed, since
    // panel size doesn't drive rendering anymore.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        try {
            pipelineRef.current = new Pipeline(canvas);
        } catch (e) {
            console.error("[PipelineRenderer] init failed:", e);
            return;
        }
        onPipelineChangeRef.current?.(pipelineRef.current);
        startMsRef.current = performance.now();

        let raf = 0;
        const tick = () => {
            const p = pipelineRef.current;
            if (p) {
                const tNow =
                    (performance.now() - startMsRef.current) / 1000;
                p.renderFrame(tNow, pointerRef.current, paramsRef.current);
                // Drain hook runs immediately after — outputTex contents
                // are fresh, so any per-node readback gets the same
                // frame the terminal blit just showed.
                onPostRenderRef.current?.();
            }
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);

        return () => {
            cancelAnimationFrame(raf);
            onPipelineChangeRef.current?.(null);
            const p = pipelineRef.current;
            if (p) p.dispose();
            pipelineRef.current = null;
        };
    }, []);

    // Push working buffer dimensions into the pipeline whenever they
    // change. Aspect comes from the Global Input source; long side
    // comes from the Output node's resolution.
    useEffect(() => {
        const p = pipelineRef.current;
        if (!p) return;
        p.setOutput({ bufferW, bufferH });
    }, [bufferW, bufferH]);

    // Rebuild only when structural digest changes — param tweaks just feed
    // through paramsRef on the next rAF tick.
    useEffect(() => {
        const p = pipelineRef.current;
        if (!p) return;
        try {
            const plan = flattenChain(chain);
            p.rebuild(plan);
        } catch (e) {
            console.error("[PipelineRenderer] rebuild failed:", e);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [digest]);

    // Loading HUD status — polled at 5 Hz, separate from the rAF loop.
    useEffect(() => {
        const id = window.setInterval(() => {
            const p = pipelineRef.current;
            if (!p) return;
            const next = p.getStatus();
            setStatus((prev) =>
                prev.producersInFlight === next.producersInFlight &&
                prev.latestProgress === next.latestProgress &&
                prev.error === next.error
                    ? prev
                    : next,
            );
        }, 200);
        return () => window.clearInterval(id);
    }, []);

    const hostStyle = viewMode === "fit" ? HOST_FIT : HOST_ACTUAL;
    const canvasStyle = viewMode === "fit" ? CANVAS_FIT : CANVAS_ACTUAL;

    // Pointer UV: map host CSS coords to (0..1) with y flipped — `vUv`
    // runs bottom-up by convention so this matches what shaders see.
    const onPointerMove = useCallback(
        (e: ReactPointerEvent<HTMLDivElement>) => {
            const host = hostRef.current;
            if (!host) return;
            const rect = host.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const x = (e.clientX - rect.left) / rect.width;
            const y = 1 - (e.clientY - rect.top) / rect.height;
            pointerRef.current = {
                uv: [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))],
                active: true,
            };
        },
        [],
    );
    const onPointerLeave = useCallback(() => {
        pointerRef.current = {
            uv: pointerRef.current.uv,
            active: false,
        };
    }, []);

    return (
        <div
            ref={hostRef}
            style={hostStyle}
            onPointerMove={onPointerMove}
            onPointerEnter={onPointerMove}
            onPointerLeave={onPointerLeave}
        >
            <canvas ref={canvasRef} style={canvasStyle} />
            {status.error && (
                <div style={{ ...HUD_STYLE, color: "#ff7777" }}>
                    error: {status.error}
                </div>
            )}
            {!status.error && status.producersInFlight > 0 && (
                <div style={HUD_STYLE}>
                    {describeProgress(status.latestProgress)}
                </div>
            )}
        </div>
    );
}
