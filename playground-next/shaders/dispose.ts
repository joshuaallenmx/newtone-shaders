/**
 * Iterate any object and call `.dispose()` on every member that has one.
 * transformers.js / onnxruntime-web Tensors hold GPU memory that the JS
 * GC can't reclaim — they need explicit dispose to free WebGPU buffers.
 *
 * Safe to pass plain objects: members without `dispose` (e.g. RawImage)
 * are skipped.
 */
export function disposeTensors(obj: unknown): void {
    if (!obj || typeof obj !== "object") return;
    for (const value of Object.values(obj as Record<string, unknown>)) {
        if (!value || typeof value !== "object") continue;
        const maybeDispose = (value as { dispose?: unknown }).dispose;
        if (typeof maybeDispose === "function") {
            try {
                (value as { dispose: () => void }).dispose();
            } catch {
                // best-effort: a tensor may already be disposed
            }
        }
    }
}

/**
 * Force a canvas to release its backing buffer. Setting width/height to 0
 * is the standard JS trick to prompt browsers to drop the GPU allocation
 * without waiting for GC.
 */
export function disposeCanvas(canvas: HTMLCanvasElement | null): void {
    if (!canvas) return;
    canvas.width = 0;
    canvas.height = 0;
}
