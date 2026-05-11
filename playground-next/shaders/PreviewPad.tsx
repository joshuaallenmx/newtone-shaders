import {
    useCallback,
    useRef,
    type CSSProperties,
    type PointerEvent as ReactPE,
} from "react";
// The thumbnail registry lives in the editor package, but the runtime
// dependency is one-way at module-init time (PipelineHandleContext only
// imports the Pipeline type, never any shader entry), so this import
// doesn't form a cycle. The pad component just needs `useNodeSnapshot`
// to register a canvas for per-frame snapshots.
import {
    useNodeSnapshot,
    usePreviewSize,
} from "../../playground-editor/PipelineHandleContext";

interface PreviewPadProps {
    /** Current value in vUv coords (bottom-up, 0..1). */
    readonly value: { readonly x: number; readonly y: number };
    readonly onChange: (next: { x: number; y: number }) => void;
    /** When provided, the pad shows a live thumbnail of that node as
     *  its backdrop. The thumbnail comes from the same drain path the
     *  graph-node thumbnails use, so it costs no extra readback. */
    readonly nodeId?: string | null;
    /** Visual color of the draggable dot. Defaults to a soft cyan. */
    readonly dotColor?: string;
}

const PAD_OUTER_STYLE: CSSProperties = {
    width: "100%",
    aspectRatio: "1 / 1",
    background:
        "repeating-linear-gradient(45deg, #0d0d0d 0 6px, #111 6px 12px)",
    border: "1px solid #2a2a2a",
    borderRadius: 4,
    position: "relative",
    cursor: "crosshair",
    touchAction: "none",
    marginTop: 2,
    marginBottom: 6,
    overflow: "hidden",
};

const SNAPSHOT_CANVAS_STYLE: CSSProperties = {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
    opacity: 0.9,
};

const DOT_BASE_STYLE: CSSProperties = {
    position: "absolute",
    width: 12,
    height: 12,
    borderRadius: 12,
    transform: "translate(-50%, -50%)",
    pointerEvents: "none",
    boxShadow:
        "0 0 0 2px rgba(255,255,255,0.5), 0 0 0 4px rgba(0,0,0,0.55)",
};

export function PreviewPad({
    value,
    onChange,
    nodeId,
    dotColor = "#7fc7ff",
}: PreviewPadProps) {
    const padRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const draggingRef = useRef(false);
    const { w: previewW, h: previewH } = usePreviewSize();

    useNodeSnapshot(nodeId ?? null, canvasRef);

    const setFromEvent = useCallback(
        (cx: number, cy: number) => {
            const el = padRef.current;
            if (!el) return;
            const rect = el.getBoundingClientRect();
            const x = (cx - rect.left) / rect.width;
            // Pad's top is "y=1" in vUv (bottom-up) so the dot tracks
            // what the user sees on the canvas.
            const y = 1 - (cy - rect.top) / rect.height;
            onChange({
                x: Math.max(0, Math.min(1, x)),
                y: Math.max(0, Math.min(1, y)),
            });
        },
        [onChange],
    );

    const onPointerDown = useCallback(
        (e: ReactPE<HTMLDivElement>) => {
            draggingRef.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            setFromEvent(e.clientX, e.clientY);
        },
        [setFromEvent],
    );

    const onPointerMove = useCallback(
        (e: ReactPE<HTMLDivElement>) => {
            if (!draggingRef.current) return;
            setFromEvent(e.clientX, e.clientY);
        },
        [setFromEvent],
    );

    const onPointerUp = useCallback((e: ReactPE<HTMLDivElement>) => {
        draggingRef.current = false;
        try {
            e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
            // not captured; ignore
        }
    }, []);

    return (
        <div
            ref={padRef}
            style={PAD_OUTER_STYLE}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
        >
            {nodeId ? (
                <canvas
                    ref={canvasRef}
                    width={previewW}
                    height={previewH}
                    style={SNAPSHOT_CANVAS_STYLE}
                />
            ) : null}
            <div
                style={{
                    ...DOT_BASE_STYLE,
                    background: dotColor,
                    left: `${value.x * 100}%`,
                    top: `${(1 - value.y) * 100}%`,
                }}
            />
        </div>
    );
}
