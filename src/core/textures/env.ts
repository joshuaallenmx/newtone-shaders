import * as THREE from "three";
import { cssColorToRgbBytes } from "../color/css";

const DEFAULT_STOPS: readonly string[] = ["#595294", "#e6de8c"];

/**
 * Vertical RGBA gradient packed into a Uint8Array. Used as a 2D stand-in for
 * environment cubemaps when sampled along a reflection vector.
 *
 * Stops interpolate in sRGB byte space (matches CSS `linear-gradient`); pair
 * with `colorSpace = SRGBColorSpace` on the texture so the GPU converts to
 * linear at sample time.
 */
export function fillEnvData(
    data: Uint8Array,
    size: number,
    stops: readonly string[],
): void {
    const colors = stops.length > 0 ? stops : DEFAULT_STOPS;
    const rgb = colors.map(cssColorToRgbBytes);
    const segCount = Math.max(1, rgb.length - 1);
    for (let y = 0; y < size; y++) {
        const t = size === 1 ? 0 : y / (size - 1);
        const seg = t * segCount;
        const i0 = Math.min(rgb.length - 1, Math.floor(seg));
        const i1 = Math.min(rgb.length - 1, i0 + 1);
        const f = seg - i0;
        const r = Math.round(rgb[i0][0] + (rgb[i1][0] - rgb[i0][0]) * f);
        const g = Math.round(rgb[i0][1] + (rgb[i1][1] - rgb[i0][1]) * f);
        const b = Math.round(rgb[i0][2] + (rgb[i1][2] - rgb[i0][2]) * f);
        for (let x = 0; x < size; x++) {
            const i = (y * size + x) * 4;
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = 255;
        }
    }
}

export interface EnvTextureOptions {
    /** Side length of the (square) gradient texture. @default 256 */
    readonly size?: number;
}

export function makeEnvTexture(
    stops: readonly string[],
    opts: EnvTextureOptions = {},
): THREE.DataTexture {
    const size = opts.size ?? 256;
    const data = new Uint8Array(size * size * 4);
    fillEnvData(data, size, stops);
    const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
}

/** Refill an existing gradient texture's data and flag it dirty. */
export function updateEnvTexture(
    tex: THREE.DataTexture,
    stops: readonly string[],
): void {
    const data = tex.image.data as Uint8Array;
    const size = tex.image.width as number;
    fillEnvData(data, size, stops);
    tex.needsUpdate = true;
}
