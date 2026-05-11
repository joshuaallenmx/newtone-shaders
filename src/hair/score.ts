import { createDefaultHairDetectParams } from "../shaders/hair-detect";
import type { HairDetectParams } from "../shaders/hair-detect";
import type { HairScoreInput } from "./types";

function smoothstep(e0: number, e1: number, x: number): number {
    if (e1 <= e0) return x >= e1 ? 1 : 0;
    const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
}

/**
 * JS port of `src/shaders/hair-detect/glsl.ts`. For each pixel, samples a
 * 5×5 grid (spacing = `kernelRadius * 0.5` source pixels, so the kernel
 * spans `±kernelRadius` per side) and computes:
 *
 *   meanL, stddev(L)            from BT.709 luminance of each sample
 *   meanRgb                     for HSV saturation
 *   texture = smoothstep(textureFloor, textureCeil, stddev * textureGain)
 *   lumaMask = (lumaMin <= meanL < lumaMax) ? 1 : 0
 *   satMask  = 1 - smoothstep(satMax, satMax + 0.05, sat(meanRgb))
 *   hair     = texture * lumaMask * satMask
 *
 * Returns a flat `Float32Array` of length `width × height`, values 0..1.
 */
export function hairScoreMap(
    input: HairScoreInput,
    params?: HairDetectParams,
): Float32Array {
    const p = params ?? createDefaultHairDetectParams();
    const { data, width, height } = input;
    const out = new Float32Array(width * height);

    const radius = p.kernelRadius;
    const gain = p.textureGain;
    const tFloor = p.textureFloor;
    const tCeil = p.textureCeil;
    const satMax = p.saturationMax;
    const lumaMin = p.lumaMin;
    const lumaMax = p.lumaMax;

    // Spacing between samples in the 5×5 kernel, in pixels. Matches the
    // shader's `(kernelRadius * 0.5) / srcSize` step in normalized UV.
    const stepF = radius * 0.5;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sumL = 0;
            let sumL2 = 0;
            let sumR = 0;
            let sumG = 0;
            let sumB = 0;
            for (let dy = -2; dy <= 2; dy++) {
                const sy0 = Math.round(y + dy * stepF);
                const sy =
                    sy0 < 0 ? 0 : sy0 >= height ? height - 1 : sy0;
                for (let dx = -2; dx <= 2; dx++) {
                    const sx0 = Math.round(x + dx * stepF);
                    const sx =
                        sx0 < 0 ? 0 : sx0 >= width ? width - 1 : sx0;
                    const o = (sy * width + sx) * 4;
                    const r = data[o] / 255;
                    const g = data[o + 1] / 255;
                    const b = data[o + 2] / 255;
                    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    sumL += L;
                    sumL2 += L * L;
                    sumR += r;
                    sumG += g;
                    sumB += b;
                }
            }

            const n = 25;
            const meanL = sumL / n;
            const variance = Math.max(sumL2 / n - meanL * meanL, 0);
            const stddev = Math.sqrt(variance);
            const meanR = sumR / n;
            const meanG = sumG / n;
            const meanB = sumB / n;

            const texture = smoothstep(tFloor, tCeil, stddev * gain);
            const lumaMask =
                meanL >= lumaMin && meanL < lumaMax ? 1 : 0;
            const maxC = Math.max(meanR, meanG, meanB);
            const minC = Math.min(meanR, meanG, meanB);
            const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
            const satMask = 1 - smoothstep(satMax, satMax + 0.05, sat);

            out[y * width + x] = texture * lumaMask * satMask;
        }
    }

    return out;
}
