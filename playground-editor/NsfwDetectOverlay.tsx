import {
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
} from "react";
import {
    loadDetector,
    type DetectedRegion,
    type NsfwDetector,
    type NudeNetClass,
} from "../src/detect";
import {
    DEFAULT_NSFW_DETECT_PARAMS,
    type NsfwDetectParams,
} from "../playground-next/shaders/NsfwDetect";
import { CAPTURE_H, CAPTURE_W } from "../playground-next/pipeline/Pipeline";
import { usePipelineHandle } from "./PipelineHandleContext";
import type { ChainSpec } from "./shaders";

const DETECT_MODEL_URL = "/nudenet/320n.onnx";

// Lazy-load the detector once per page; the ONNX model is ~10 MB and
// the WASM runtime takes a moment to spin up. Subsequent renders reuse
// the cached promise.
let detectorPromise: Promise<NsfwDetector> | null = null;
function getDetector(): Promise<NsfwDetector> {
    if (!detectorPromise) {
        detectorPromise = loadDetector({ modelUrl: DETECT_MODEL_URL });
    }
    return detectorPromise;
}

interface NsfwDetectOverlayProps {
    /** The terminal ChainSpec — its `entry.id` should be
     *  NSFW_DETECT_ENTRY_ID. We read `chain.inputs[0].nodeId` for the
     *  upstream node whose outputTex to capture. */
    readonly chain: ChainSpec;
}

type Status =
    | { kind: "idle" }
    | { kind: "loading-model" }
    | { kind: "detecting" }
    | { kind: "ready"; regions: readonly DetectedRegion[] }
    | { kind: "error"; message: string };

const OVERLAY_STYLE: CSSProperties = {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    fontFamily:
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
};

const STATUS_BADGE_STYLE: CSSProperties = {
    position: "absolute",
    top: 12,
    left: 12,
    fontSize: 12,
    color: "#bdbdbd",
    background: "rgba(15,15,15,0.78)",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 6,
    padding: "6px 10px",
};

const BOX_BASE_STYLE: CSSProperties = {
    position: "absolute",
    borderStyle: "solid",
    borderWidth: 2,
    borderColor: "#ff6b6b",
    boxShadow: "0 0 0 1px rgba(0,0,0,0.6) inset",
    borderRadius: 2,
};

const LABEL_STYLE: CSSProperties = {
    position: "absolute",
    bottom: "100%",
    left: -2,
    background: "#ff6b6b",
    color: "#0a0a0a",
    fontSize: 10,
    fontWeight: 600,
    padding: "2px 5px",
    borderRadius: "2px 2px 0 0",
    whiteSpace: "nowrap",
    fontVariantNumeric: "tabular-nums",
};

/** Map a fit-captured box (in CAPTURE_W × CAPTURE_H pixel space) to
 *  normalised canvas-content coords [0,1] × [0,1]. The capture used fit
 *  mode, so the source content sits inside a letterbox/pillarbox region
 *  of the capture; we undo that here using the canvas's display
 *  aspect. */
function captureBoxToCanvasNorm(
    box: readonly [number, number, number, number],
    canvasAspect: number,
): { x: number; y: number; w: number; h: number } {
    let srcW: number;
    let srcH: number;
    let lbX: number;
    let lbY: number;
    if (canvasAspect >= 1) {
        srcW = CAPTURE_W;
        srcH = CAPTURE_H / canvasAspect;
        lbX = 0;
        lbY = (CAPTURE_H - srcH) / 2;
    } else {
        srcH = CAPTURE_H;
        srcW = CAPTURE_W * canvasAspect;
        lbX = (CAPTURE_W - srcW) / 2;
        lbY = 0;
    }
    const [bx, by, bw, bh] = box;
    return {
        x: (bx - lbX) / Math.max(srcW, 1e-3),
        y: (by - lbY) / Math.max(srcH, 1e-3),
        w: bw / Math.max(srcW, 1e-3),
        h: bh / Math.max(srcH, 1e-3),
    };
}

export function NsfwDetectOverlay({ chain }: NsfwDetectOverlayProps) {
    const { handle, version } = usePipelineHandle();
    const [status, setStatus] = useState<Status>({ kind: "idle" });
    const hostRef = useRef<HTMLDivElement>(null);

    // Track the overlay host's actual on-screen aspect ratio. The host
    // is mounted inside CanvasFrame, which is already aspect-correct,
    // so this gives us exactly the canvas display aspect.
    const [canvasAspect, setCanvasAspect] = useState(1);
    useEffect(() => {
        const el = hostRef.current;
        if (!el) return;
        const update = () => {
            const r = el.getBoundingClientRect();
            if (r.height > 0) setCanvasAspect(r.width / r.height);
        };
        update();
        const ro = new ResizeObserver(update);
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const inputId = chain.inputs[0]?.nodeId ?? null;
    const params: NsfwDetectParams = {
        ...DEFAULT_NSFW_DETECT_PARAMS,
        ...((chain.params as Partial<NsfwDetectParams> | null) ?? {}),
    };
    const selectedClasses = useMemo(
        () => new Set<NudeNetClass>(params.classes),
        [params.classes],
    );

    // Re-detect on chain change (params/topology) or Pipeline replacement.
    // Class selection alone doesn't trigger inference — we filter the
    // displayed boxes from the cached `status.regions`.
    const generationRef = useRef(0);
    useEffect(() => {
        let cancelled = false;
        generationRef.current += 1;
        const myGen = generationRef.current;

        const run = async () => {
            if (!inputId) {
                setStatus({ kind: "ready", regions: [] });
                return;
            }
            if (!detectorPromise) setStatus({ kind: "loading-model" });
            let detector: NsfwDetector;
            try {
                detector = await getDetector();
            } catch (e) {
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({
                    kind: "error",
                    message: `failed to load NudeNet: ${
                        e instanceof Error ? e.message : String(e)
                    }`,
                });
                return;
            }
            if (cancelled || myGen !== generationRef.current) return;

            setStatus({ kind: "detecting" });
            const pipeline = handle.pipeline;
            if (!pipeline) return;

            // Use FIT capture so box coords map cleanly back to the
            // canvas display rectangle (which also fits the source).
            let img: ImageData | null = null;
            for (let attempt = 0; attempt < 30; attempt++) {
                img = pipeline.captureNodeImageData(inputId, "fit");
                if (img) break;
                await new Promise((r) =>
                    requestAnimationFrame(() => r(null)),
                );
                if (cancelled || myGen !== generationRef.current) return;
            }
            if (!img) {
                setStatus({
                    kind: "error",
                    message: "input never became ready",
                });
                return;
            }

            // Detector consumes a canvas (ort accepts ImageBitmap-likes);
            // ImageData → 2D canvas is the most portable path.
            const off = document.createElement("canvas");
            off.width = img.width;
            off.height = img.height;
            const ctx = off.getContext("2d");
            if (!ctx) {
                setStatus({
                    kind: "error",
                    message: "no 2D context for capture",
                });
                return;
            }
            ctx.putImageData(img, 0, 0);

            try {
                // Pre-NMS threshold stays low so the slider can sweep
                // freely on the JS side without re-running inference.
                const regions = await detector.detect(off, {
                    scoreThreshold: 0.05,
                });
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({ kind: "ready", regions });
            } catch (e) {
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({
                    kind: "error",
                    message: `detect failed: ${
                        e instanceof Error ? e.message : String(e)
                    }`,
                });
            }
        };
        run();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [inputId, chain, version]);

    const visible = useMemo(() => {
        if (status.kind !== "ready") return [];
        return status.regions
            .filter(
                (r) =>
                    selectedClasses.has(r.class as NudeNetClass) &&
                    r.score >= params.minScore,
            )
            .map((r) => ({
                ...r,
                norm: captureBoxToCanvasNorm(r.box, canvasAspect),
            }));
    }, [status, selectedClasses, params.minScore, canvasAspect]);

    return (
        <div
            ref={hostRef}
            className="newtone-nsfw-detect"
            style={OVERLAY_STYLE}
        >
            {(status.kind === "idle" || status.kind === "loading-model") && (
                <div
                    className="newtone-nsfw-detect-status"
                    style={STATUS_BADGE_STYLE}
                >
                    {status.kind === "loading-model"
                        ? "loading NudeNet…"
                        : "preparing…"}
                </div>
            )}
            {status.kind === "detecting" && (
                <div
                    className="newtone-nsfw-detect-status"
                    style={STATUS_BADGE_STYLE}
                >
                    detecting…
                </div>
            )}
            {status.kind === "error" && (
                <div
                    className="newtone-nsfw-detect-status"
                    style={{ ...STATUS_BADGE_STYLE, color: "#ff7777" }}
                >
                    {status.message}
                </div>
            )}
            {visible.map((r, i) => (
                <div
                    key={`${r.class}-${i}`}
                    className={`newtone-nsfw-detect-box newtone-nsfw-detect-box-${r.class.toLowerCase()}`}
                    style={{
                        ...BOX_BASE_STYLE,
                        left: `${r.norm.x * 100}%`,
                        top: `${r.norm.y * 100}%`,
                        width: `${r.norm.w * 100}%`,
                        height: `${r.norm.h * 100}%`,
                    }}
                    data-class={r.class}
                    data-score={r.score.toFixed(3)}
                >
                    {params.showLabels && (
                        <div
                            className="newtone-nsfw-detect-label"
                            style={LABEL_STYLE}
                        >
                            {r.class.toLowerCase()} ·{" "}
                            {(r.score * 100).toFixed(0)}%
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}
