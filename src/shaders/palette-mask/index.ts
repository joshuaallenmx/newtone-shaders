import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import type { RGB } from "../../core/cluster";
import { PALETTE_MASK_PASS } from "./glsl";

export { PALETTE_MASK_PASS };

/** Maximum palette entries the shader is compiled for. */
export const PALETTE_MAX = 16;

export type PaletteMaskMode = "posterize" | "mask" | "overlay";

export const PALETTE_MASK_MODE_INDEX: Record<PaletteMaskMode, number> = {
    posterize: 0,
    mask: 1,
    overlay: 2,
};

/**
 * Mutable knobs. The two `Float32Array` buffers are stable references —
 * mutate fields directly each frame; the GPU sees the change on the next
 * render. Use `setPaletteFromRgb` for the common path.
 */
export interface PaletteMaskParams {
    /** Stable buffer of length `PALETTE_MAX * 3`, packed `r,g,b,r,g,b,…` in [0,1]. */
    readonly palette: Float32Array;
    /** Stable buffer of length `PALETTE_MAX`, 0 or 1 per slot. */
    readonly enabled: Float32Array;
    /** Active palette count, `1..PALETTE_MAX`. @default 0 (no entries) */
    paletteSize: number;
    /** Output mode. @default "posterize" */
    mode: PaletteMaskMode;
    /** Brightness multiplier on disabled entries in posterize/overlay. @default 0.15 */
    offMix: number;
}

export function createDefaultPaletteMaskParams(): PaletteMaskParams {
    return {
        palette: new Float32Array(PALETTE_MAX * 3),
        enabled: new Float32Array(PALETTE_MAX),
        paletteSize: 0,
        mode: "posterize",
        offMix: 0.15,
    };
}

/**
 * Write a byte-RGB palette into the stable buffers and update `paletteSize`.
 * `enabledList` is optional; if omitted, every active slot is enabled.
 */
export function setPaletteFromRgb(
    params: PaletteMaskParams,
    rgbList: ReadonlyArray<RGB>,
    enabledList?: ReadonlyArray<boolean>,
): void {
    const n = Math.min(rgbList.length, PALETTE_MAX);
    for (let i = 0; i < n; i++) {
        const c = rgbList[i];
        params.palette[i * 3] = c[0] / 255;
        params.palette[i * 3 + 1] = c[1] / 255;
        params.palette[i * 3 + 2] = c[2] / 255;
    }
    for (let i = n; i < PALETTE_MAX; i++) {
        params.palette[i * 3] = 0;
        params.palette[i * 3 + 1] = 0;
        params.palette[i * 3 + 2] = 0;
    }
    for (let i = 0; i < PALETTE_MAX; i++) {
        if (i >= n) {
            params.enabled[i] = 0;
        } else if (enabledList && i < enabledList.length) {
            params.enabled[i] = enabledList[i] ? 1 : 0;
        } else {
            params.enabled[i] = 1;
        }
    }
    params.paletteSize = n;
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type PaletteMaskSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreatePaletteMaskPipelineOptions {
    readonly id?: string;
    readonly source: PaletteMaskSource;
    readonly params?: PaletteMaskParams;
}

export function createPaletteMaskPipelineConfig(
    opts: CreatePaletteMaskPipelineOptions,
): { config: PipelineConfig; params: PaletteMaskParams } {
    const params = opts.params ?? createDefaultPaletteMaskParams();

    const iResolutionVec = new THREE.Vector3();
    const iResolutionProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => {
            iResolutionVec.set(ctx.target.w, ctx.target.h, 1);
            return iResolutionVec;
        },
    };

    const sourceProvider: UniformProvider = (() => {
        if (typeof opts.source === "function") {
            const fn = opts.source;
            return {
                kind: "computed",
                fn: () => fn() ?? (null as unknown as THREE.Texture),
            };
        }
        return { kind: "texture", ref: { kind: "asset", texture: opts.source } };
    })();

    const config: PipelineConfig = {
        id: opts.id ?? "palette-mask",
        passes: [
            {
                id: "mask",
                fragment: composeFragment(PALETTE_MASK_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uPalette: { kind: "static", value: params.palette },
                    uEnabled: { kind: "static", value: params.enabled },
                    uPaletteSize: {
                        kind: "computed",
                        fn: () => params.paletteSize,
                    },
                    uMode: {
                        kind: "computed",
                        fn: () => PALETTE_MASK_MODE_INDEX[params.mode],
                    },
                    uOffMix: {
                        kind: "computed",
                        fn: () => params.offMix,
                    },
                },
            },
        ],
        outputs: { final: "mask" },
    };

    return { config, params };
}
