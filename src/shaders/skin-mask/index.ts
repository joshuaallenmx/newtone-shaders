import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { SKIN_MASK_PASS } from "./glsl";

export { SKIN_MASK_PASS };

/**
 * Soft thresholds in YCbCr (BT.601, all values in [0,1]). Defaults track the
 * Chai & Ngan 1999 chroma window with the Y window left wide-open.
 */
export interface SkinMaskParams {
    /** Lower luminance bound. @default 0 */
    yMin: number;
    /** Upper luminance bound. @default 1 */
    yMax: number;
    /** Lower blue-chroma bound. @default 0.302 (= 77/255) */
    cbMin: number;
    /** Upper blue-chroma bound. @default 0.498 (= 127/255) */
    cbMax: number;
    /** Lower red-chroma bound. @default 0.522 (= 133/255) */
    crMin: number;
    /** Upper red-chroma bound. @default 0.678 (= 173/255) */
    crMax: number;
    /** Smoothstep half-width on each window edge. @default 0.02 */
    feather: number;
}

export function createDefaultSkinMaskParams(): SkinMaskParams {
    return {
        yMin: 0,
        yMax: 1,
        cbMin: 0.302,
        cbMax: 0.498,
        crMin: 0.522,
        crMax: 0.678,
        feather: 0.02,
    };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type SkinMaskSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateSkinMaskPipelineOptions {
    readonly id?: string;
    /** Source media texture — `THREE.Texture` for images, `THREE.VideoTexture` for video. */
    readonly source: SkinMaskSource;
    /** Mutable knobs; safe to mutate fields each frame. */
    readonly params?: SkinMaskParams;
}

/**
 * Build a `PipelineConfig` for the skin-mask shader. Single screen pass that
 * outputs a grayscale skin-probability mask via a YCbCr chrominance window.
 */
export function createSkinMaskPipelineConfig(
    opts: CreateSkinMaskPipelineOptions,
): { config: PipelineConfig; params: SkinMaskParams } {
    const params = opts.params ?? createDefaultSkinMaskParams();

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

    const knob = (key: keyof SkinMaskParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key],
    });

    const config: PipelineConfig = {
        id: opts.id ?? "skin-mask",
        passes: [
            {
                id: "mask",
                fragment: composeFragment(SKIN_MASK_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uYMin: knob("yMin"),
                    uYMax: knob("yMax"),
                    uCbMin: knob("cbMin"),
                    uCbMax: knob("cbMax"),
                    uCrMin: knob("crMin"),
                    uCrMax: knob("crMax"),
                    uFeather: knob("feather"),
                },
            },
        ],
        outputs: { final: "mask" },
    };

    return { config, params };
}
