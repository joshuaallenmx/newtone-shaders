import { createDefaultSkinMaskParams } from "../shaders/skin-mask";
import type { BinaryMask } from "../pipeline";
import type { ClassifySkinOptions, SkinClassifyInput } from "./types";

function smoothstep(e0: number, e1: number, x: number): number {
    const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
}

function readPixels(input: SkinClassifyInput): {
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
    if (!w || !h) throw new Error("classifySkin: input has zero size");

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
    if (!ctx) throw new Error("classifySkin: 2D context unavailable");
    ctx.drawImage(input as CanvasImageSource, 0, 0);
    const img = ctx.getImageData(0, 0, w, h);
    return { data: img.data, width: w, height: h };
}

/**
 * BT.601 YCbCr skin classifier ported from `src/shaders/skin-mask/`. For
 * each pixel: convert RGB→YCbCr, score via three smoothstep windows
 * (Y/Cb/Cr) with shared `feather`, multiply, threshold. Output is a
 * source-pixel-sized binary mask (0 outside, 255 inside).
 */
export function classifySkin(
    input: SkinClassifyInput,
    options: ClassifySkinOptions = {},
): BinaryMask {
    const params = options.params ?? createDefaultSkinMaskParams();
    const threshold = options.threshold ?? 0.5;
    const { data, width, height } = readPixels(input);

    const out = new Uint8Array(width * height);

    const yLo0 = params.yMin - params.feather;
    const yLo1 = params.yMin + params.feather;
    const yHi0 = params.yMax - params.feather;
    const yHi1 = params.yMax + params.feather;
    const cbLo0 = params.cbMin - params.feather;
    const cbLo1 = params.cbMin + params.feather;
    const cbHi0 = params.cbMax - params.feather;
    const cbHi1 = params.cbMax + params.feather;
    const crLo0 = params.crMin - params.feather;
    const crLo1 = params.crMin + params.feather;
    const crHi0 = params.crMax - params.feather;
    const crHi1 = params.crMax + params.feather;

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;

        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5;
        const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5;

        const yScore =
            smoothstep(yLo0, yLo1, y) * (1 - smoothstep(yHi0, yHi1, y));
        const cbScore =
            smoothstep(cbLo0, cbLo1, cb) * (1 - smoothstep(cbHi0, cbHi1, cb));
        const crScore =
            smoothstep(crLo0, crLo1, cr) * (1 - smoothstep(crHi0, crHi1, cr));

        const score = yScore * cbScore * crScore;
        if (score >= threshold) out[p] = 255;
    }

    return { data: out, width, height };
}
