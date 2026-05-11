import type { SegmentMask } from "./types";

/**
 * Render a `SegmentMask` onto an `HTMLCanvasElement` ready for upload as a
 * `THREE.CanvasTexture`. White inside the region, transparent outside.
 *
 * The optional `color` parameter lets you tint the inside RGB channels
 * (alpha is still mask-driven). Default is opaque white.
 */
export function maskToCanvas(
    mask: SegmentMask,
    options: {
        readonly canvas?: HTMLCanvasElement;
        readonly color?: readonly [number, number, number];
    } = {},
): HTMLCanvasElement {
    const canvas = options.canvas ?? document.createElement("canvas");
    canvas.width = mask.width;
    canvas.height = mask.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("maskToCanvas: 2D context unavailable");

    const [r, g, b] = options.color ?? [255, 255, 255];
    const image = ctx.createImageData(mask.width, mask.height);
    const out = image.data;
    const src = mask.data;
    for (let i = 0; i < src.length; i++) {
        const o = i * 4;
        out[o] = r;
        out[o + 1] = g;
        out[o + 2] = b;
        out[o + 3] = src[i];
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
}
