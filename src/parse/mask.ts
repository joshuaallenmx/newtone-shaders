import type { BinaryMask, ParseResult } from "./types";

/**
 * Build a binary mask selecting only pixels whose class index is in
 * `classes`. Used by the pipeline filter (Phase B) to clip SAM masks
 * against an allowlist of human-parsing classes (e.g. arms, legs, hair).
 */
export function parseToMask(
    result: ParseResult,
    classes: ReadonlyArray<number>,
): BinaryMask {
    const allow = new Uint8Array(256);
    for (const c of classes) {
        if (c >= 0 && c < allow.length) allow[c] = 1;
    }
    const data = new Uint8Array(result.classMap.length);
    const src = result.classMap;
    for (let i = 0; i < src.length; i++) {
        if (allow[src[i]]) data[i] = 255;
    }
    return { data, width: result.width, height: result.height };
}

// Hand-picked palette indexed by class id. Background is fully transparent
// so the source image shows through cleanly. Other classes use
// semantically motivated hues (skin warm, clothing cool, accessories
// saturated) so the overlay is readable at a glance.
const PALETTE: ReadonlyArray<readonly [number, number, number, number]> = [
    [0, 0, 0, 0], // 0  Background
    [180, 120, 60, 255], // 1  Hat
    [120, 70, 200, 255], // 2  Hair
    [80, 220, 220, 255], // 3  Sunglasses
    [60, 130, 230, 255], // 4  Upper-clothes
    [220, 100, 180, 255], // 5  Skirt
    [40, 80, 200, 255], // 6  Pants
    [200, 60, 200, 255], // 7  Dress
    [240, 200, 60, 255], // 8  Belt
    [200, 60, 70, 255], // 9  Left-shoe
    [220, 90, 100, 255], // 10 Right-shoe
    [255, 200, 170, 255], // 11 Face
    [120, 220, 120, 255], // 12 Left-leg
    [150, 235, 150, 255], // 13 Right-leg
    [240, 190, 150, 255], // 14 Left-arm
    [255, 215, 175, 255], // 15 Right-arm
    [140, 90, 50, 255], // 16 Bag
    [180, 120, 220, 255], // 17 Scarf
];

export interface ParseVisualizationOptions {
    /**
     * Reuse this canvas instead of allocating a new one. Resized to fit.
     */
    readonly canvas?: HTMLCanvasElement;
    /**
     * Scalar applied to every non-background pixel's alpha, 0..1.
     * @default 0.6
     */
    readonly alpha?: number;
}

/**
 * Render a `ParseResult` to a canvas: each pixel painted with its
 * class's palette color. Background pixels stay transparent so the
 * caller can stack this canvas on top of the source image.
 */
export function parseToVisualizationCanvas(
    result: ParseResult,
    options: ParseVisualizationOptions = {},
): HTMLCanvasElement {
    const canvas = options.canvas ?? document.createElement("canvas");
    canvas.width = result.width;
    canvas.height = result.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("parseToVisualizationCanvas: 2D context unavailable");

    const alpha = options.alpha ?? 0.6;
    const image = ctx.createImageData(result.width, result.height);
    const out = image.data;
    const src = result.classMap;
    for (let i = 0; i < src.length; i++) {
        const cls = src[i];
        const c = PALETTE[cls] ?? PALETTE[0];
        const o = i * 4;
        out[o] = c[0];
        out[o + 1] = c[1];
        out[o + 2] = c[2];
        out[o + 3] = Math.round(c[3] * alpha);
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
}

/**
 * RGB triplet for a class id, useful for painting legend swatches.
 */
export function colorForParseClass(classId: number): [number, number, number] {
    const c = PALETTE[classId] ?? PALETTE[0];
    return [c[0], c[1], c[2]];
}
