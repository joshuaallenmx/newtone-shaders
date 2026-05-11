import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { HAIR_DETECT_PASS } from "./glsl";

export { HAIR_DETECT_PASS };

export interface HairDetectParams {
    /** Half-width of the 5×5 kernel in source pixels. @default 2 */
    kernelRadius: number;
    /** Linear multiplier on σ before threshold. @default 6 */
    textureGain: number;
    /** Smoothstep low edge — texture below this clips. @default 0.20 */
    textureFloor: number;
    /** Smoothstep high edge. @default 1.0 */
    textureCeil: number;
    /** HSV saturation upper bound — pixels above are excluded. @default 0.7 */
    saturationMax: number;
    /** Lower luminance bound. @default 0.0 */
    lumaMin: number;
    /** Upper luminance bound. @default 1.0 */
    lumaMax: number;
}

export function createDefaultHairDetectParams(): HairDetectParams {
    return {
        kernelRadius: 2,
        textureGain: 6,
        textureFloor: 0.2,
        textureCeil: 1.0,
        saturationMax: 0.7,
        lumaMin: 0.0,
        lumaMax: 1.0,
    };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type HairDetectSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateHairDetectPipelineOptions {
    readonly id?: string;
    readonly source: HairDetectSource;
    readonly params?: HairDetectParams;
}

export function createHairDetectPipelineConfig(
    opts: CreateHairDetectPipelineOptions,
): { config: PipelineConfig; params: HairDetectParams } {
    const params = opts.params ?? createDefaultHairDetectParams();

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

    const knob = (key: keyof HairDetectParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key],
    });

    const config: PipelineConfig = {
        id: opts.id ?? "hair-detect",
        passes: [
            {
                id: "hair",
                fragment: composeFragment(HAIR_DETECT_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uKernelRadius: knob("kernelRadius"),
                    uTextureGain: knob("textureGain"),
                    uTextureFloor: knob("textureFloor"),
                    uTextureCeil: knob("textureCeil"),
                    uSaturationMax: knob("saturationMax"),
                    uLumaMin: knob("lumaMin"),
                    uLumaMax: knob("lumaMax"),
                },
            },
        ],
        outputs: { final: "hair" },
    };

    return { config, params };
}
