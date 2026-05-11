import type { RGB } from "./kmeans";

/**
 * Heckbert's median-cut color quantization. Splits the RGB bounding box of
 * the input pixels recursively along its longest axis at the median pixel
 * value, producing `k` axis-aligned color regions whose averages form the
 * palette.
 *
 * Compared to k-means, median cut gives a palette that covers the color
 * *space* more uniformly — dense regions (e.g. skin) don't dominate the
 * palette at the expense of sparser-but-distinct colors (e.g. background,
 * accents). It's also deterministic — same input → same palette.
 */

interface Box {
    indices: Uint32Array;
    rRange: number;
    gRange: number;
    bRange: number;
    longestRange: number;
    longestAxis: 0 | 1 | 2;
}

function computeBox(
    pixels: Uint8Array | Uint8ClampedArray,
    indices: Uint32Array,
): Box {
    let rMin = 255;
    let rMax = 0;
    let gMin = 255;
    let gMax = 0;
    let bMin = 255;
    let bMax = 0;
    for (let i = 0; i < indices.length; i++) {
        const idx = indices[i] * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        if (r < rMin) rMin = r;
        if (r > rMax) rMax = r;
        if (g < gMin) gMin = g;
        if (g > gMax) gMax = g;
        if (b < bMin) bMin = b;
        if (b > bMax) bMax = b;
    }
    const rRange = rMax - rMin;
    const gRange = gMax - gMin;
    const bRange = bMax - bMin;
    let longestAxis: 0 | 1 | 2 = 0;
    let longestRange = rRange;
    if (gRange > longestRange) {
        longestAxis = 1;
        longestRange = gRange;
    }
    if (bRange > longestRange) {
        longestAxis = 2;
        longestRange = bRange;
    }
    return { indices, rRange, gRange, bRange, longestRange, longestAxis };
}

function splitBox(
    pixels: Uint8Array | Uint8ClampedArray,
    box: Box,
): [Box, Box] {
    const axis = box.longestAxis;
    const arr = Array.from(box.indices);
    arr.sort((a, b) => pixels[a * 4 + axis] - pixels[b * 4 + axis]);
    const mid = arr.length >>> 1;
    const left = new Uint32Array(arr.slice(0, mid));
    const right = new Uint32Array(arr.slice(mid));
    return [computeBox(pixels, left), computeBox(pixels, right)];
}

function averageColor(
    pixels: Uint8Array | Uint8ClampedArray,
    indices: Uint32Array,
): RGB {
    let r = 0;
    let g = 0;
    let b = 0;
    for (let i = 0; i < indices.length; i++) {
        const idx = indices[i] * 4;
        r += pixels[idx];
        g += pixels[idx + 1];
        b += pixels[idx + 2];
    }
    const n = Math.max(indices.length, 1);
    return [Math.round(r / n), Math.round(g / n), Math.round(b / n)] as RGB;
}

export interface MedianCutOptions {
    /** Number of palette entries. */
    readonly k: number;
}

/**
 * Quantize RGBA pixel data into `k` palette colors via median cut. Alpha is
 * ignored. Returns palette colors sorted by population (descending) — most
 * common cluster first.
 */
export function medianCutRgb(
    pixels: Uint8Array | Uint8ClampedArray,
    opts: MedianCutOptions,
): RGB[] {
    const k = Math.max(1, Math.min(opts.k, 256));
    const n = Math.floor(pixels.length / 4);
    if (n === 0) return [];

    const initial = new Uint32Array(n);
    for (let i = 0; i < n; i++) initial[i] = i;

    const boxes: Box[] = [computeBox(pixels, initial)];

    while (boxes.length < k) {
        let pickIdx = -1;
        let pickScore = 0;
        for (let i = 0; i < boxes.length; i++) {
            const b = boxes[i];
            if (b.indices.length < 2) continue;
            // Score by longest axis range × population so a "fat" cluster
            // gets split before a tiny isolated one.
            const score = b.longestRange * Math.log2(b.indices.length + 1);
            if (score > pickScore) {
                pickScore = score;
                pickIdx = i;
            }
        }
        if (pickIdx === -1) break;
        const [a, b] = splitBox(pixels, boxes[pickIdx]);
        boxes.splice(pickIdx, 1, a, b);
    }

    return boxes
        .map((box) => ({
            color: averageColor(pixels, box.indices),
            count: box.indices.length,
        }))
        .sort((a, b) => b.count - a.count)
        .map((entry) => entry.color);
}
