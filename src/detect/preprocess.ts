import type { DetectInput } from "./types";

export interface PreprocessResult {
    /** NCHW float32 tensor data, length `3 * inputSize * inputSize`. */
    readonly tensor: Float32Array;
    /** Original (pre-pad) image width in pixels. */
    readonly origWidth: number;
    /** Original (pre-pad) image height in pixels. */
    readonly origHeight: number;
    /**
     * `max(origWidth, origHeight)` — the size of the square that the
     * (top-left aligned) image was padded into before being resized to the
     * model input. Used to scale boxes back to source pixels.
     */
    readonly maxSize: number;
}

function inputDimensions(input: DetectInput): { w: number; h: number } {
    if (input instanceof HTMLVideoElement) {
        return { w: input.videoWidth, h: input.videoHeight };
    }
    if (input instanceof HTMLImageElement) {
        return {
            w: input.naturalWidth || input.width,
            h: input.naturalHeight || input.height,
        };
    }
    return { w: input.width, h: input.height };
}

/**
 * Match NudeNet's reference preprocessing: pad to a square by extending
 * with black on the right and bottom (no centering), resize to
 * `inputSize × inputSize`, divide by 255, emit RGB in NCHW order.
 */
export function preprocess(
    input: DetectInput,
    inputSize: number,
): PreprocessResult {
    const { w: origWidth, h: origHeight } = inputDimensions(input);
    if (!origWidth || !origHeight) {
        throw new Error("preprocess: input has zero width or height");
    }

    const maxSize = Math.max(origWidth, origHeight);
    const scale = inputSize / maxSize;
    const drawW = origWidth * scale;
    const drawH = origHeight * scale;

    const canvas =
        typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(inputSize, inputSize)
            : Object.assign(document.createElement("canvas"), {
                  width: inputSize,
                  height: inputSize,
              });
    const ctx = canvas.getContext("2d", {
        willReadFrequently: true,
    }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
    if (!ctx) throw new Error("preprocess: 2D context unavailable");

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(input as CanvasImageSource, 0, 0, drawW, drawH);

    const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
    const planeStride = inputSize * inputSize;
    const tensor = new Float32Array(3 * planeStride);
    for (let i = 0; i < planeStride; i++) {
        const o = i * 4;
        tensor[i] = data[o] / 255;
        tensor[planeStride + i] = data[o + 1] / 255;
        tensor[planeStride * 2 + i] = data[o + 2] / 255;
    }

    return { tensor, origWidth, origHeight, maxSize };
}
