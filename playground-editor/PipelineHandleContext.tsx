import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
    type RefObject,
} from "react";
import type { Pipeline } from "../playground-next/pipeline/Pipeline";

/** Default preview canvas dimensions used until the Pipeline reports
 *  its real `previewW × previewH`. Matches `PREVIEW_LONG` so node
 *  thumbnails reserve a sensible amount of space on first paint. */
const FALLBACK_PREVIEW_W = 128;
const FALLBACK_PREVIEW_H = 128;

// Bridge between the single Pipeline instance owned by PipelineRenderer
// and the per-node `<canvas>` thumbnails rendered inside React Flow's
// node components. PipelineRenderer registers the live Pipeline here on
// mount; node components register their canvas by `nodeId` and the
// dispatch loop (also driven by PipelineRenderer's rAF) drains a small
// dirty queue per frame.
//
// Steady state cost: zero. The dirty set is only populated when params
// or topology change, so a graph at rest doesn't readPixels at all.

interface PipelineHandle {
    pipeline: Pipeline | null;
    /** A node may have several canvases listening (the graph thumbnail
     *  and one or more inspector PreviewPads, etc.). Each gets painted
     *  on drain. */
    readonly registry: Map<string, Set<HTMLCanvasElement>>;
    readonly dirty: Set<string>;
    cursor: number;
}

interface PipelineHandleContextValue {
    readonly handle: PipelineHandle;
    /** Bumped when the Pipeline instance is attached/detached. Hooks
     *  re-register so they survive a Pipeline replacement (project
     *  switch, StrictMode double-mount). */
    readonly version: number;
    /** Per-node thumbnail canvas dimensions. Matches the working buffer
     *  aspect with longest side held at `PREVIEW_LONG`; updates whenever
     *  the Global Input source's aspect changes. Components consuming
     *  this set their `<canvas>` width/height to it so the readback's
     *  `putImageData` lands 1:1 with no resampling. */
    readonly previewSize: { readonly w: number; readonly h: number };
    setPipeline: (p: Pipeline | null) => void;
    setPreviewSize: (size: { readonly w: number; readonly h: number }) => void;
    register: (nodeId: string, canvas: HTMLCanvasElement) => () => void;
    /** Mark every currently-registered node as dirty. */
    markAllDirty: () => void;
    /** Drain up to `max` dirty nodes by snapshotting their canvases.
     *  Returns the number actually snapshotted. Nodes that aren't ready
     *  yet (no outputTex) stay on the queue. */
    drain: (max: number) => number;
}

const PipelineHandleContext = createContext<PipelineHandleContextValue | null>(
    null,
);

interface PipelineHandleProviderProps {
    readonly children: ReactNode;
}

export function PipelineHandleProvider({ children }: PipelineHandleProviderProps) {
    // The handle lives in a ref so per-frame mutation (cursor advance,
    // dirty toggling) doesn't trigger React re-renders. Only `version`
    // is reactive — bumped when the Pipeline instance changes so hooks
    // know to re-register against a fresh instance.
    const handleRef = useRef<PipelineHandle>({
        pipeline: null,
        registry: new Map(),
        dirty: new Set(),
        cursor: 0,
    });
    const [version, setVersion] = useState(0);
    const [previewSize, setPreviewSizeState] = useState<{
        readonly w: number;
        readonly h: number;
    }>({ w: FALLBACK_PREVIEW_W, h: FALLBACK_PREVIEW_H });
    const setPreviewSize = useCallback(
        (size: { readonly w: number; readonly h: number }) => {
            setPreviewSizeState((prev) =>
                prev.w === size.w && prev.h === size.h ? prev : size,
            );
        },
        [],
    );

    const setPipeline = useCallback((p: Pipeline | null) => {
        handleRef.current.pipeline = p;
        // On attach, mark every registered canvas dirty so it picks up
        // a snapshot from the new instance. On detach, just clear.
        if (p) {
            const dirty = handleRef.current.dirty;
            for (const id of handleRef.current.registry.keys()) {
                dirty.add(id);
            }
        } else {
            handleRef.current.dirty.clear();
        }
        setVersion((v) => v + 1);
    }, []);

    const register = useCallback(
        (nodeId: string, canvas: HTMLCanvasElement) => {
            const reg = handleRef.current.registry;
            let set = reg.get(nodeId);
            if (!set) {
                set = new Set();
                reg.set(nodeId, set);
            }
            set.add(canvas);
            handleRef.current.dirty.add(nodeId);
            return () => {
                const s = reg.get(nodeId);
                if (s) {
                    s.delete(canvas);
                    if (s.size === 0) {
                        reg.delete(nodeId);
                        handleRef.current.dirty.delete(nodeId);
                    }
                }
            };
        },
        [],
    );

    const markAllDirty = useCallback(() => {
        const dirty = handleRef.current.dirty;
        for (const id of handleRef.current.registry.keys()) {
            dirty.add(id);
        }
    }, []);

    const drain = useCallback((max: number) => {
        const h = handleRef.current;
        const p = h.pipeline;
        if (!p || h.dirty.size === 0) return 0;

        // Snapshot the dirty set into a stable array so we can advance a
        // cursor through it without invalidating the iteration on
        // delete-and-readd cycles.
        const ids = Array.from(h.dirty);
        let processed = 0;
        let attempts = 0;
        const limit = Math.min(max, ids.length);
        while (processed < limit && attempts < ids.length) {
            const id = ids[h.cursor % ids.length]!;
            h.cursor = (h.cursor + 1) % ids.length;
            attempts++;
            const set = h.registry.get(id);
            if (!set || set.size === 0) {
                // Node unregistered between dirty add and drain — skip.
                h.dirty.delete(id);
                continue;
            }
            // Paint each registered canvas. snapshotNode does its own
            // GPU readback per call; for nodes with multiple listeners
            // (graph thumbnail + inspector pad) the readback runs once
            // per canvas. Sub-millisecond each, fine for a handful.
            let anyOk = false;
            for (const canvas of set) {
                const ctx = canvas.getContext("2d");
                if (!ctx) continue;
                const ok = p.snapshotNode(ctx, id);
                if (ok) anyOk = true;
            }
            if (anyOk) {
                h.dirty.delete(id);
                processed++;
            }
            // If no canvas was ready (node not yet rendered), leave it
            // in the dirty set for next frame.
        }
        return processed;
    }, []);

    const value = useMemo<PipelineHandleContextValue>(
        () => ({
            handle: handleRef.current,
            version,
            previewSize,
            setPipeline,
            setPreviewSize,
            register,
            markAllDirty,
            drain,
        }),
        [
            version,
            previewSize,
            setPipeline,
            setPreviewSize,
            register,
            markAllDirty,
            drain,
        ],
    );

    return (
        <PipelineHandleContext.Provider value={value}>
            {children}
        </PipelineHandleContext.Provider>
    );
}

/** Pipeline-side hook used by PipelineRenderer to publish/clear the
 *  Pipeline reference and pump the dirty drain each frame. */
export function usePipelineHandle(): PipelineHandleContextValue {
    const ctx = useContext(PipelineHandleContext);
    if (!ctx) {
        throw new Error(
            "usePipelineHandle: must be inside <PipelineHandleProvider>",
        );
    }
    return ctx;
}

/** Node-side hook: register a canvas to receive thumbnail snapshots for
 *  the given nodeId. Pass `null` to opt out (e.g. an unwired Slot).
 *  Re-registers when nodeId changes or when the underlying Pipeline is
 *  replaced. */
export function useNodeSnapshot(
    nodeId: string | null,
    canvasRef: RefObject<HTMLCanvasElement | null>,
): void {
    const ctx = useContext(PipelineHandleContext);
    useEffect(() => {
        if (!ctx) return;
        if (!nodeId) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const unregister = ctx.register(nodeId, canvas);
        return unregister;
        // ctx.version triggers re-register after Pipeline replacement;
        // canvasRef is a stable ref so it's intentionally not in deps.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [ctx, nodeId, ctx?.version]);
}

/** Hook returning the current per-node thumbnail dimensions. Updates
 *  whenever the working buffer aspect changes (Global Input source
 *  swap, Output aspect change while no source is wired). Components
 *  use this to size their `<canvas>` elements so the readback's
 *  `putImageData` is 1:1 with no resampling. */
export function usePreviewSize(): { readonly w: number; readonly h: number } {
    const ctx = useContext(PipelineHandleContext);
    if (!ctx) {
        throw new Error(
            "usePreviewSize: must be inside <PipelineHandleProvider>",
        );
    }
    return ctx.previewSize;
}
