import { hairScoreMap } from "../hair";
import type { HairDetectParams } from "../hair";
import type { FocusPoint } from "./types";

export type ChromaticFocusInput =
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap
    | OffscreenCanvas;

export interface ChromaticFocusOptions {
    /** Source-pixel box `[x, y, w, h]` to search within. */
    readonly box: readonly [number, number, number, number];
    /**
     * How strongly to bias toward the box center, 0..1.
     *  - 0  : pure chromatic search; argmax wins regardless of position.
     *  - 1  : extreme center bias (rarely useful).
     * @default 0.4
     */
    readonly centerBias?: number;
    /**
     * Shift the Gaussian center horizontally, normalized to box width.
     * 0 = box center, +0.25 = right of center, -0.25 = left of center.
     * @default 0
     */
    readonly centerOffsetX?: number;
    /**
     * Shift the Gaussian center vertically, normalized to box height.
     * 0 = box center, +0.15 = below center, -0.15 = above center. Useful
     * for classes whose landmark sits anatomically below the box center
     * (e.g. introitus inside `FEMALE_GENITALIA_EXPOSED`). @default 0
     */
    readonly centerOffsetY?: number;
    /**
     * Box-blur radius (pixels) applied to the score map. Smooths over
     * single-pixel freckles and JPEG noise. @default 3
     */
    readonly smoothing?: number;
    /** Maximum number of points to return, sorted by score desc. @default 1 */
    readonly maxPoints?: number;
    /** Minimum normalized score (0..1) for a point to be returned. @default 0 */
    readonly minScore?: number;
    /**
     * Suppress hair-textured pixels in the chromatic score map. Pulls the
     * argmax away from pubic / body hair clumps that share the dark+warm
     * signature with the landmark. Score is multiplied by
     * `1 - hairPenalty * hairScore`. 0 = disabled. @default 0
     */
    readonly hairPenalty?: number;
    /**
     * Override the hair-detector parameters when `hairPenalty > 0`.
     * Defaults to the same window the GPU `hair-detect` shader uses.
     */
    readonly hairParams?: HairDetectParams;
    /**
     * Boost pixels lying on coherent vertical edges (Sobel-X magnitude
     * smoothed along the Y axis). Targets the closed-vulva case where the
     * salient feature is a vertical cleft line rather than a chromatic
     * spot. Added to the chromatic score after normalization. 0 =
     * disabled. @default 0
     */
    readonly verticalLineBonus?: number;
}

interface CropPixels {
    readonly data: Uint8ClampedArray;
    readonly width: number;
    readonly height: number;
    readonly offsetX: number;
    readonly offsetY: number;
}

function cropToCanvas(
    image: ChromaticFocusInput,
    box: readonly [number, number, number, number],
): CropPixels {
    const ix = Math.max(0, Math.floor(box[0]));
    const iy = Math.max(0, Math.floor(box[1]));
    const iw = Math.max(1, Math.floor(box[2]));
    const ih = Math.max(1, Math.floor(box[3]));

    const canvas =
        typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(iw, ih)
            : Object.assign(document.createElement("canvas"), {
                  width: iw,
                  height: ih,
              });
    const ctx = canvas.getContext("2d", {
        willReadFrequently: true,
    }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
    if (!ctx) throw new Error("findChromaticFocus: 2D context unavailable");
    ctx.drawImage(
        image as CanvasImageSource,
        ix,
        iy,
        iw,
        ih,
        0,
        0,
        iw,
        ih,
    );
    const id = ctx.getImageData(0, 0, iw, ih);
    return {
        data: id.data,
        width: iw,
        height: ih,
        offsetX: ix,
        offsetY: iy,
    };
}

export interface VerticalLineMap {
    /** Normalized 0..1 line-strength per pixel; length = `width × height`. */
    readonly data: Float32Array;
    readonly width: number;
    readonly height: number;
    /** Box origin in source pixels — lets callers blit the map back into source space. */
    readonly offsetX: number;
    readonly offsetY: number;
}

export interface FindVerticalLineMapOptions {
    /** Source-pixel box `[x, y, w, h]` to search within. */
    readonly box: readonly [number, number, number, number];
    /**
     * Vertical box-blur radius (in pixels) applied to the Sobel-X
     * magnitude. Higher values demand longer coherent vertical edges.
     * @default 5% of box height (min 2)
     */
    readonly blurRadius?: number;
}

/**
 * Sobel-X gradient magnitude over BT.709 luminance — high response on
 * vertical edges (left/right brightness differs across the column).
 * Output is normalized to `[0, 1]` (max kernel response = 4).
 */
function sobelXMagnitudeMap(
    data: Uint8ClampedArray | Uint8Array,
    width: number,
    height: number,
): Float32Array {
    const out = new Float32Array(width * height);
    const lum = (x: number, y: number): number => {
        const sx = x < 0 ? 0 : x >= width ? width - 1 : x;
        const sy = y < 0 ? 0 : y >= height ? height - 1 : y;
        const o = (sy * width + sx) * 4;
        return (
            0.2126 * (data[o] / 255) +
            0.7152 * (data[o + 1] / 255) +
            0.0722 * (data[o + 2] / 255)
        );
    };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const gx =
                -lum(x - 1, y - 1) +
                lum(x + 1, y - 1) +
                -2 * lum(x - 1, y) +
                2 * lum(x + 1, y) +
                -lum(x - 1, y + 1) +
                lum(x + 1, y + 1);
            out[y * width + x] = Math.abs(gx) * 0.25;
        }
    }
    return out;
}

/**
 * Sobel-X magnitude → vertical box-blur → max-normalized to `[0, 1]`.
 * Pixels on coherent vertical edges score high; isolated gradients
 * average out vertically. Pure compute, accepts already-cropped pixels.
 */
function verticalLineScoreMap(
    data: Uint8ClampedArray | Uint8Array,
    width: number,
    height: number,
    blurRadius?: number,
): Float32Array {
    const planeSize = width * height;
    const raw = sobelXMagnitudeMap(data, width, height);
    const blurred = new Float32Array(planeSize);
    const r = blurRadius ?? Math.max(2, Math.floor(height * 0.05));
    boxBlur1D(raw, blurred, width, height, r, false);
    let max = 0;
    for (let i = 0; i < blurred.length; i++) {
        if (blurred[i] > max) max = blurred[i];
    }
    if (max > 0) {
        const inv = 1 / max;
        for (let i = 0; i < blurred.length; i++) {
            blurred[i] *= inv;
        }
    }
    return blurred;
}

function boxBlur1D(
    src: Float32Array,
    dst: Float32Array,
    w: number,
    h: number,
    radius: number,
    horizontal: boolean,
): void {
    if (radius <= 0) {
        dst.set(src);
        return;
    }
    if (horizontal) {
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                const x0 = Math.max(0, x - radius);
                const x1 = Math.min(w - 1, x + radius);
                let sum = 0;
                for (let k = x0; k <= x1; k++) sum += src[row + k];
                dst[row + x] = sum / (x1 - x0 + 1);
            }
        }
    } else {
        for (let x = 0; x < w; x++) {
            for (let y = 0; y < h; y++) {
                const y0 = Math.max(0, y - radius);
                const y1 = Math.min(h - 1, y + radius);
                let sum = 0;
                for (let k = y0; k <= y1; k++) sum += src[k * w + x];
                dst[y * w + x] = sum / (y1 - y0 + 1);
            }
        }
    }
}

/**
 * Find salient "darkness × redness" peaks inside a detection box. Used to
 * refine the SAM 2 point prompt from `boxCenter` toward the actual
 * areola/nipple position. Pure JS, runs on canvas pixels.
 *
 * Score per pixel:
 *   y  = BT.601 luminance (0..1)
 *   cr = BT.601 red chroma  (0..1, neutral at 0.5)
 *   darkness = 1 - y
 *   redness  = max(0, cr - 0.5) × 2
 *   score    = darkness × redness
 *
 * The score map is box-blurred, then multiplied by an optional Gaussian
 * centered on the box (to suppress edge spurious peaks), then scanned
 * for the top-N maxima with non-max suppression.
 */
export function findChromaticFocus(
    image: ChromaticFocusInput,
    options: ChromaticFocusOptions,
): FocusPoint[] {
    const centerBias = options.centerBias ?? 0.4;
    const smoothing = options.smoothing ?? 3;
    const maxPoints = options.maxPoints ?? 1;
    const minScore = options.minScore ?? 0;

    const { data, width, height, offsetX, offsetY } = cropToCanvas(
        image,
        options.box,
    );
    const planeSize = width * height;
    const scores = new Float32Array(planeSize);

    for (let i = 0; i < planeSize; i++) {
        const o = i * 4;
        const r = data[o] / 255;
        const g = data[o + 1] / 255;
        const b = data[o + 2] / 255;
        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5;
        const darkness = 1 - y;
        const redness = Math.max(0, cr - 0.5) * 2;
        scores[i] = darkness * redness;
    }

    if (smoothing > 0) {
        const tmp = new Float32Array(planeSize);
        boxBlur1D(scores, tmp, width, height, smoothing, true);
        boxBlur1D(tmp, scores, width, height, smoothing, false);
    }

    const lineBonus = options.verticalLineBonus ?? 0;
    if (lineBonus > 0) {
        const line = verticalLineScoreMap(data, width, height);
        for (let i = 0; i < planeSize; i++) {
            scores[i] += lineBonus * line[i];
        }
    }

    const hairPenalty = options.hairPenalty ?? 0;
    if (hairPenalty > 0) {
        const hair = hairScoreMap(
            { data, width, height },
            options.hairParams,
        );
        for (let i = 0; i < planeSize; i++) {
            scores[i] *= 1 - hairPenalty * hair[i];
        }
    }

    if (centerBias > 0) {
        const offsetX = options.centerOffsetX ?? 0;
        const offsetY = options.centerOffsetY ?? 0;
        const cx = width * (0.5 + offsetX);
        const cy = height * (0.5 + offsetY);
        const sigma = Math.max(width, height) * 0.35;
        const inv2s2 = 1 / (2 * sigma * sigma);
        for (let py = 0; py < height; py++) {
            for (let px = 0; px < width; px++) {
                const dx = px - cx;
                const dy = py - cy;
                const bias = Math.exp(-(dx * dx + dy * dy) * inv2s2);
                const i = py * width + px;
                scores[i] *= 1 - centerBias + centerBias * bias;
            }
        }
    }

    const out: FocusPoint[] = [];
    const suppressed = new Uint8Array(planeSize);
    const suppRadius = Math.max(
        2,
        Math.floor(Math.min(width, height) * 0.12),
    );
    const suppR2 = suppRadius * suppRadius;

    for (let n = 0; n < maxPoints; n++) {
        let bestScore = minScore;
        let bestIdx = -1;
        for (let i = 0; i < planeSize; i++) {
            if (suppressed[i]) continue;
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) break;

        const py = Math.floor(bestIdx / width);
        const px = bestIdx - py * width;
        out.push({
            x: offsetX + px,
            y: offsetY + py,
            score: bestScore,
        });

        const y0 = Math.max(0, py - suppRadius);
        const y1 = Math.min(height - 1, py + suppRadius);
        const x0 = Math.max(0, px - suppRadius);
        const x1 = Math.min(width - 1, px + suppRadius);
        for (let qy = y0; qy <= y1; qy++) {
            const dy = qy - py;
            for (let qx = x0; qx <= x1; qx++) {
                const dx = qx - px;
                if (dx * dx + dy * dy <= suppR2) {
                    suppressed[qy * width + qx] = 1;
                }
            }
        }
    }

    return out;
}

/**
 * Public wrapper around the same Sobel-X + vertical-blur pipeline that
 * `findChromaticFocus` uses internally for `verticalLineBonus`. Crops to
 * the box, returns a normalized line-strength map plus the box origin so
 * callers can render the result back in source-image space.
 */
export function findVerticalLineMap(
    image: ChromaticFocusInput,
    options: FindVerticalLineMapOptions,
): VerticalLineMap {
    const { data, width, height, offsetX, offsetY } = cropToCanvas(
        image,
        options.box,
    );
    const lineData = verticalLineScoreMap(
        data,
        width,
        height,
        options.blurRadius,
    );
    return { data: lineData, width, height, offsetX, offsetY };
}

export interface FindHairFocusOptions {
    /** Source-pixel box `[x, y, w, h]` to search within. */
    readonly box: readonly [number, number, number, number];
    /** @see ChromaticFocusOptions.centerBias */
    readonly centerBias?: number;
    /** @see ChromaticFocusOptions.centerOffsetX */
    readonly centerOffsetX?: number;
    /** @see ChromaticFocusOptions.centerOffsetY */
    readonly centerOffsetY?: number;
    /** @see ChromaticFocusOptions.smoothing */
    readonly smoothing?: number;
    /** @see ChromaticFocusOptions.maxPoints */
    readonly maxPoints?: number;
    /** @see ChromaticFocusOptions.minScore */
    readonly minScore?: number;
    /** Override hair-detector params; defaults to `createDefaultHairDetectParams()`. */
    readonly hairParams?: HairDetectParams;
}

/**
 * Mirror of `findChromaticFocus` but the score map is `hairScoreMap`
 * (local luminance variance + saturation/luma gates). Argmax lands on
 * the densest hair clump inside the box. Useful for pubic-hair focus
 * inside synthesized regions where the chromatic darkness × redness
 * signal is unhelpful (hair is dark-but-not-red).
 */
export function findHairFocus(
    image: ChromaticFocusInput,
    options: FindHairFocusOptions,
): FocusPoint[] {
    const centerBias = options.centerBias ?? 0.4;
    const smoothing = options.smoothing ?? 3;
    const maxPoints = options.maxPoints ?? 1;
    const minScore = options.minScore ?? 0;

    const { data, width, height, offsetX, offsetY } = cropToCanvas(
        image,
        options.box,
    );
    const planeSize = width * height;
    const scores = hairScoreMap(
        { data, width, height },
        options.hairParams,
    );

    if (smoothing > 0) {
        const tmp = new Float32Array(planeSize);
        boxBlur1D(scores, tmp, width, height, smoothing, true);
        boxBlur1D(tmp, scores, width, height, smoothing, false);
    }

    if (centerBias > 0) {
        const offsetCx = options.centerOffsetX ?? 0;
        const offsetCy = options.centerOffsetY ?? 0;
        const cx = width * (0.5 + offsetCx);
        const cy = height * (0.5 + offsetCy);
        const sigma = Math.max(width, height) * 0.35;
        const inv2s2 = 1 / (2 * sigma * sigma);
        for (let py = 0; py < height; py++) {
            for (let px = 0; px < width; px++) {
                const dx = px - cx;
                const dy = py - cy;
                const bias = Math.exp(-(dx * dx + dy * dy) * inv2s2);
                const i = py * width + px;
                scores[i] *= 1 - centerBias + centerBias * bias;
            }
        }
    }

    const out: FocusPoint[] = [];
    const suppressed = new Uint8Array(planeSize);
    const suppRadius = Math.max(
        2,
        Math.floor(Math.min(width, height) * 0.12),
    );
    const suppR2 = suppRadius * suppRadius;

    for (let n = 0; n < maxPoints; n++) {
        let bestScore = minScore;
        let bestIdx = -1;
        for (let i = 0; i < planeSize; i++) {
            if (suppressed[i]) continue;
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) break;

        const py = Math.floor(bestIdx / width);
        const px = bestIdx - py * width;
        out.push({
            x: offsetX + px,
            y: offsetY + py,
            score: bestScore,
        });

        const y0 = Math.max(0, py - suppRadius);
        const y1 = Math.min(height - 1, py + suppRadius);
        const x0 = Math.max(0, px - suppRadius);
        const x1 = Math.min(width - 1, px + suppRadius);
        for (let qy = y0; qy <= y1; qy++) {
            const dy = qy - py;
            for (let qx = x0; qx <= x1; qx++) {
                const dx = qx - px;
                if (dx * dx + dy * dy <= suppR2) {
                    suppressed[qy * width + qx] = 1;
                }
            }
        }
    }

    return out;
}
