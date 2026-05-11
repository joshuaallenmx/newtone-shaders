/**
 * Local-maximum extraction over a 2D scalar field — used to convert a
 * Hough accumulator (or any score map) into discrete peak positions.
 *
 * Input is a Uint8 RGBA buffer (as returned by `renderer.readRenderTargetPixels`),
 * but only the red channel is read; alpha is ignored. Each cell value is
 * normalized to `[0, 1]` for thresholding.
 *
 * For each cell ≥ `threshold`, we scan a `(2 * suppressionRadius + 1)²`
 * window. The cell is accepted only if it's a strict local maximum (no
 * neighbor exceeds it) — classic non-max suppression. Ties are broken by
 * scan order, so a flat plateau emits one peak (its top-left corner).
 *
 * Output peaks are sorted by score descending and capped at `maxPeaks`.
 */

export interface AccumulatorPeak {
    /** X in accumulator-pixel space (origin top-left). */
    cx: number;
    /** Y in accumulator-pixel space (origin top-left). */
    cy: number;
    /** Score in `[0, 1]`. */
    score: number;
}

export interface FindAccumulatorPeaksOptions {
    /** Width of the accumulator buffer, in pixels. */
    readonly width: number;
    /** Height of the accumulator buffer, in pixels. */
    readonly height: number;
    /** RGBA pixel buffer (length `width * height * 4`). */
    readonly pixels: Uint8Array | Uint8ClampedArray;
    /** Score threshold in `[0, 1]`. Cells below are skipped. */
    readonly threshold: number;
    /** Half-width of the NMS window in accumulator pixels. */
    readonly suppressionRadius: number;
    /** Cap the number of returned peaks. @default 64 */
    readonly maxPeaks?: number;
    /**
     * If `true`, treat the buffer's row 0 as the bottom of the image (WebGL
     * convention) and flip Y on output so peaks are in top-down coords. The
     * input buffer order is unchanged. @default false
     */
    readonly flipY?: boolean;
}

export function findAccumulatorPeaks(
    opts: FindAccumulatorPeaksOptions,
): AccumulatorPeak[] {
    const { width, height, pixels, threshold } = opts;
    const r = Math.max(1, Math.round(opts.suppressionRadius));
    const maxPeaks = opts.maxPeaks ?? 64;
    const flipY = opts.flipY ?? false;
    if (pixels.length < width * height * 4) return [];

    const get = (x: number, y: number): number => pixels[(y * width + x) * 4];
    const minByte = threshold * 255;

    const peaks: AccumulatorPeak[] = [];
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const v = get(x, y);
            if (v < minByte) continue;
            const x0 = Math.max(0, x - r);
            const y0 = Math.max(0, y - r);
            const x1 = Math.min(width - 1, x + r);
            const y1 = Math.min(height - 1, y + r);
            let isMax = true;
            for (let yy = y0; yy <= y1 && isMax; yy++) {
                for (let xx = x0; xx <= x1; xx++) {
                    if (xx === x && yy === y) continue;
                    if (get(xx, yy) > v) {
                        isMax = false;
                        break;
                    }
                }
            }
            if (!isMax) continue;
            peaks.push({
                cx: x,
                cy: flipY ? height - 1 - y : y,
                score: v / 255,
            });
        }
    }
    peaks.sort((a, b) => b.score - a.score);
    if (peaks.length > maxPeaks) peaks.length = maxPeaks;
    return peaks;
}
