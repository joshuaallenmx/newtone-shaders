import * as THREE from "three";
import {
    kmeansRgb,
    medianCutRgb,
    type KMeansOptions,
    type RGB,
} from "../cluster";

/** Drawable element types we can pull pixels out of. */
export type SampleableElement =
    | HTMLImageElement
    | HTMLVideoElement
    | HTMLCanvasElement
    | ImageBitmap;

export interface SamplePixelsOptions {
    /** Square edge length the source is scaled into before reading. @default 64 */
    readonly size?: number;
}

/**
 * Draw a sampleable element onto a small canvas and return its RGBA pixel
 * bytes. The source is scaled to fit `size × size` (preserving aspect, but
 * letterboxed onto an opaque canvas is *not* needed since k-means ignores
 * the alpha channel). Returns `null` if the source isn't ready (e.g. video
 * metadata not loaded).
 */
export function samplePixels(
    el: SampleableElement,
    opts: SamplePixelsOptions = {},
): Uint8ClampedArray | null {
    const size = opts.size ?? 64;
    const w = "videoWidth" in el ? el.videoWidth : el.width;
    const h = "videoHeight" in el ? el.videoHeight : el.height;
    if (!w || !h) return null;

    const aspect = w / h;
    let dw: number;
    let dh: number;
    if (aspect >= 1) {
        dw = size;
        dh = Math.max(1, Math.round(size / aspect));
    } else {
        dh = size;
        dw = Math.max(1, Math.round(size * aspect));
    }

    const canvas = document.createElement("canvas");
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(el as CanvasImageSource, 0, 0, dw, dh);
    return ctx.getImageData(0, 0, dw, dh).data;
}

/**
 * Pull the underlying DOM element out of a `THREE.Texture`. Works for image
 * and video textures (the most common cases); other texture sources return
 * `null` and the caller can fall back.
 */
export function elementFromTexture(
    tex: THREE.Texture,
): SampleableElement | null {
    const src = (tex as { image?: unknown }).image;
    if (!src) return null;
    if (
        src instanceof HTMLImageElement ||
        src instanceof HTMLVideoElement ||
        src instanceof HTMLCanvasElement ||
        (typeof ImageBitmap !== "undefined" && src instanceof ImageBitmap)
    ) {
        return src;
    }
    return null;
}

export interface SampleCornerColorOptions {
    /** Edge length the source is scaled into before sampling. @default 128 */
    readonly sampleSize?: number;
    /** Side length of each corner square (in scaled-canvas pixels). @default 8 */
    readonly cornerSize?: number;
}

/**
 * Average the four corners of a sampleable element into a single byte-RGB
 * tuple. Useful as an automatic background-color pick for chroma-key /
 * silhouette workflows.
 *
 * The element is drawn at a moderate resolution (default 128 on the long
 * edge) so the corner samples include some bilinear blending — small dust /
 * vignetting in the corners doesn't dominate.
 */
export function sampleCornerColor(
    el: SampleableElement,
    opts: SampleCornerColorOptions = {},
): RGB | null {
    const sampleSize = opts.sampleSize ?? 128;
    const cornerSize = opts.cornerSize ?? 8;
    const w = "videoWidth" in el ? el.videoWidth : el.width;
    const h = "videoHeight" in el ? el.videoHeight : el.height;
    if (!w || !h) return null;

    const aspect = w / h;
    const dw =
        aspect >= 1 ? sampleSize : Math.max(1, Math.round(sampleSize * aspect));
    const dh =
        aspect >= 1 ? Math.max(1, Math.round(sampleSize / aspect)) : sampleSize;

    const canvas = document.createElement("canvas");
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(el as CanvasImageSource, 0, 0, dw, dh);

    const c = Math.min(cornerSize, dw, dh);
    const corners: ReadonlyArray<readonly [number, number]> = [
        [0, 0],
        [dw - c, 0],
        [0, dh - c],
        [dw - c, dh - c],
    ];

    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
    for (const [cx, cy] of corners) {
        const data = ctx.getImageData(cx, cy, c, c).data;
        for (let i = 0; i < data.length; i += 4) {
            r += data[i];
            g += data[i + 1];
            b += data[i + 2];
            count++;
        }
    }
    if (count === 0) return null;
    return [
        Math.round(r / count),
        Math.round(g / count),
        Math.round(b / count),
    ] as RGB;
}

export function sampleCornerColorFromTexture(
    tex: THREE.Texture,
    opts: SampleCornerColorOptions = {},
): RGB | null {
    const el = elementFromTexture(tex);
    if (!el) return null;
    return sampleCornerColor(el, opts);
}

export type PaletteMethod = "median-cut" | "kmeans";

export interface ExtractPaletteOptions
    extends Omit<KMeansOptions, "k">,
        SamplePixelsOptions {
    /** Number of palette entries. @default 8 */
    readonly paletteSize?: number;
    /**
     * Quantization algorithm.
     *   - `"median-cut"` (default): splits the RGB bounding box at the
     *      median of its longest axis. Gives uniform color-space coverage —
     *      dense regions don't dominate over sparser-but-distinct colors.
     *      Deterministic.
     *   - `"kmeans"`: density-weighted clustering. Allocates more palette
     *      slots to populous regions (e.g. multiple skin variants on a
     *      portrait). Stochastic; uses the `seed` option.
     */
    readonly method?: PaletteMethod;
}

/**
 * Sample a texture's underlying DOM element and quantize the pixels into a
 * palette of dominant colors. Returns `null` if the source isn't sampleable
 * yet (e.g. video metadata not loaded).
 *
 * Colors are returned in `[0, 255]` byte space, sorted by population
 * descending — so `palette[0]` is the most common cluster.
 */
export function extractPaletteFromTexture(
    tex: THREE.Texture,
    opts: ExtractPaletteOptions = {},
): RGB[] | null {
    const el = elementFromTexture(tex);
    if (!el) return null;
    const pixels = samplePixels(el, { size: opts.size });
    if (!pixels) return null;
    const k = opts.paletteSize ?? 8;
    const method = opts.method ?? "median-cut";
    if (method === "kmeans") {
        return kmeansRgb(pixels, {
            k,
            maxIterations: opts.maxIterations,
            tolerance: opts.tolerance,
            seed: opts.seed,
        });
    }
    return medianCutRgb(pixels, { k });
}
