import type { BinaryMask } from "../pipeline";
import { hairScoreMap } from "./score";
import type { ClassifyHairOptions, HairClassifyInput } from "./types";

function readPixels(input: HairClassifyInput): {
    data: Uint8ClampedArray;
    width: number;
    height: number;
} {
    if (typeof ImageData !== "undefined" && input instanceof ImageData) {
        return { data: input.data, width: input.width, height: input.height };
    }
    const w =
        input instanceof HTMLImageElement
            ? input.naturalWidth || input.width
            : input.width;
    const h =
        input instanceof HTMLImageElement
            ? input.naturalHeight || input.height
            : input.height;
    if (!w || !h) throw new Error("classifyHair: input has zero size");

    const canvas =
        typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(w, h)
            : Object.assign(document.createElement("canvas"), {
                  width: w,
                  height: h,
              });
    const ctx = canvas.getContext("2d", {
        willReadFrequently: true,
    }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
    if (!ctx) throw new Error("classifyHair: 2D context unavailable");
    ctx.drawImage(input as CanvasImageSource, 0, 0);
    const img = ctx.getImageData(0, 0, w, h);
    return { data: img.data, width: w, height: h };
}

/**
 * Local-luminance-variance hair classifier — JS port of
 * `src/shaders/hair-detect/`. Pixels above `threshold` are emitted as 255,
 * others as 0. Same params shape as the GPU shader (shared via
 * `HairDetectParams`).
 */
export function classifyHair(
    input: HairClassifyInput,
    options: ClassifyHairOptions = {},
): BinaryMask {
    const threshold = options.threshold ?? 0.5;
    const pixels = readPixels(input);
    const scores = hairScoreMap(pixels, options.params);
    const out = new Uint8Array(pixels.width * pixels.height);
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] >= threshold) out[i] = 255;
    }
    return { data: out, width: pixels.width, height: pixels.height };
}
