import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { SILHOUETTE_PASS } from "./glsl";

export { SILHOUETTE_PASS };

export type SilhouetteMode =
    | "outline"
    | "mask"
    | "key"
    | "overlay"
    | "stable";

export const SILHOUETTE_MODE_INDEX: Record<SilhouetteMode, number> = {
    outline: 0,
    mask: 1,
    key: 2,
    overlay: 3,
    stable: 4,
};

export interface SilhouetteParams {
    /** Reference background color, normalized [0,1]. Mutate the components. */
    readonly referenceColor: THREE.Vector3;
    /**
     * Pre-blur radius in source pixels — set higher to suppress fine surface
     * detail (apple skin specks, highlights) and only trace major silhouettes.
     * Set to 0 to disable. @default 4
     */
    smoothRadius: number;
    /** Distance above which a pixel is treated as foreground. @default 0.18 */
    threshold: number;
    /** Smoothstep half-width on the threshold edge. @default 0.04 */
    feather: number;
    /**
     * Half-width of the threshold sweep used in `stable` mode. The shader
     * aggregates outlines across thresholds in
     * `[threshold - thresholdSpread, threshold + thresholdSpread]`. @default 0.1
     */
    thresholdSpread: number;
    /** Gradient sample radius in source pixels — controls outline thickness. @default 1.5 */
    outlineThickness: number;
    /** Output composition. @default "outline" */
    mode: SilhouetteMode;
    /** Background dim factor in overlay mode. @default 0.15 */
    offMix: number;
}

export function createDefaultSilhouetteParams(): SilhouetteParams {
    return {
        referenceColor: new THREE.Vector3(1, 1, 1),
        smoothRadius: 4,
        threshold: 0.18,
        feather: 0.04,
        thresholdSpread: 0.1,
        outlineThickness: 1.5,
        mode: "outline",
        offMix: 0.15,
    };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type SilhouetteSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateSilhouettePipelineOptions {
    readonly id?: string;
    readonly source: SilhouetteSource;
    readonly params?: SilhouetteParams;
}

export function createSilhouettePipelineConfig(
    opts: CreateSilhouettePipelineOptions,
): { config: PipelineConfig; params: SilhouetteParams } {
    const params = opts.params ?? createDefaultSilhouetteParams();

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

    const numKnob = (key: keyof SilhouetteParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key] as number,
    });

    const config: PipelineConfig = {
        id: opts.id ?? "silhouette",
        passes: [
            {
                id: "trace",
                fragment: composeFragment(SILHOUETTE_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uReferenceColor: {
                        kind: "static",
                        value: params.referenceColor,
                    },
                    uSmoothRadius: numKnob("smoothRadius"),
                    uThreshold: numKnob("threshold"),
                    uFeather: numKnob("feather"),
                    uThresholdSpread: numKnob("thresholdSpread"),
                    uOutlineThickness: numKnob("outlineThickness"),
                    uMode: {
                        kind: "computed",
                        fn: () => SILHOUETTE_MODE_INDEX[params.mode],
                    },
                    uOffMix: numKnob("offMix"),
                },
            },
        ],
        outputs: { final: "trace" },
    };

    return { config, params };
}
