/**
 * K-means clustering on RGB pixels — small, dependency-free, deterministic
 * given a seed. Suited to extracting a posterized palette from a sampled
 * frame (a few thousand pixels, k ≤ 16, ≤ 16 iterations).
 */

export type RGB = readonly [number, number, number];

export interface KMeansOptions {
    /** Number of clusters. */
    readonly k: number;
    /** Maximum iterations. @default 12 */
    readonly maxIterations?: number;
    /** Convergence threshold — stop when total centroid drift falls below. @default 0.5 */
    readonly tolerance?: number;
    /** PRNG seed for repeatable initialization. @default `Date.now() & 0xffffffff` */
    readonly seed?: number;
}

/** Mulberry32 — fast 32-bit PRNG, ample for cluster init. */
function makeRng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Cluster RGBA pixel data into `k` RGB centroids. Alpha is ignored. Returns
 * cluster centroids sorted by population descending (most common first).
 */
export function kmeansRgb(
    pixels: Uint8Array | Uint8ClampedArray,
    opts: KMeansOptions,
): RGB[] {
    const k = Math.max(1, Math.min(opts.k, 64));
    const maxIterations = opts.maxIterations ?? 12;
    const tolerance = opts.tolerance ?? 0.5;
    const rng = makeRng(opts.seed ?? (Date.now() & 0xffffffff));
    const n = Math.floor(pixels.length / 4);
    if (n === 0) return [];

    // K-means++ init: first centroid random, subsequent ones biased toward
    // distant points. Better convergence than uniform random.
    const centroids = new Float64Array(k * 3);
    {
        const i0 = Math.floor(rng() * n);
        centroids[0] = pixels[i0 * 4];
        centroids[1] = pixels[i0 * 4 + 1];
        centroids[2] = pixels[i0 * 4 + 2];
    }
    const dists = new Float64Array(n);
    for (let c = 1; c < k; c++) {
        let total = 0;
        for (let i = 0; i < n; i++) {
            const r = pixels[i * 4];
            const g = pixels[i * 4 + 1];
            const b = pixels[i * 4 + 2];
            let best = Infinity;
            for (let j = 0; j < c; j++) {
                const dr = r - centroids[j * 3];
                const dg = g - centroids[j * 3 + 1];
                const db = b - centroids[j * 3 + 2];
                const d = dr * dr + dg * dg + db * db;
                if (d < best) best = d;
            }
            dists[i] = best;
            total += best;
        }
        const target = rng() * total;
        let acc = 0;
        let pick = 0;
        for (let i = 0; i < n; i++) {
            acc += dists[i];
            if (acc >= target) {
                pick = i;
                break;
            }
        }
        centroids[c * 3] = pixels[pick * 4];
        centroids[c * 3 + 1] = pixels[pick * 4 + 1];
        centroids[c * 3 + 2] = pixels[pick * 4 + 2];
    }

    const sums = new Float64Array(k * 3);
    const counts = new Uint32Array(k);

    for (let iter = 0; iter < maxIterations; iter++) {
        sums.fill(0);
        counts.fill(0);

        for (let i = 0; i < n; i++) {
            const r = pixels[i * 4];
            const g = pixels[i * 4 + 1];
            const b = pixels[i * 4 + 2];
            let best = 0;
            let bestDist = Infinity;
            for (let c = 0; c < k; c++) {
                const dr = r - centroids[c * 3];
                const dg = g - centroids[c * 3 + 1];
                const db = b - centroids[c * 3 + 2];
                const d = dr * dr + dg * dg + db * db;
                if (d < bestDist) {
                    bestDist = d;
                    best = c;
                }
            }
            sums[best * 3] += r;
            sums[best * 3 + 1] += g;
            sums[best * 3 + 2] += b;
            counts[best]++;
        }

        let drift = 0;
        for (let c = 0; c < k; c++) {
            if (counts[c] === 0) {
                // Re-seed empty cluster from a random pixel to keep `k`
                // distinct slots active.
                const i = Math.floor(rng() * n);
                const newR = pixels[i * 4];
                const newG = pixels[i * 4 + 1];
                const newB = pixels[i * 4 + 2];
                drift += Math.abs(centroids[c * 3] - newR);
                drift += Math.abs(centroids[c * 3 + 1] - newG);
                drift += Math.abs(centroids[c * 3 + 2] - newB);
                centroids[c * 3] = newR;
                centroids[c * 3 + 1] = newG;
                centroids[c * 3 + 2] = newB;
                continue;
            }
            const newR = sums[c * 3] / counts[c];
            const newG = sums[c * 3 + 1] / counts[c];
            const newB = sums[c * 3 + 2] / counts[c];
            drift += Math.abs(centroids[c * 3] - newR);
            drift += Math.abs(centroids[c * 3 + 1] - newG);
            drift += Math.abs(centroids[c * 3 + 2] - newB);
            centroids[c * 3] = newR;
            centroids[c * 3 + 1] = newG;
            centroids[c * 3 + 2] = newB;
        }

        if (drift < tolerance) break;
    }

    // Sort by population (most common first) — caller-friendly ordering.
    const order = Array.from({ length: k }, (_, i) => i);
    order.sort((a, b) => counts[b] - counts[a]);
    return order.map(
        (c) =>
            [
                Math.round(centroids[c * 3]),
                Math.round(centroids[c * 3 + 1]),
                Math.round(centroids[c * 3 + 2]),
            ] as RGB,
    );
}
