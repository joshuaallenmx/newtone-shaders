import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { POSTERIZE_PASS } from "./glsl";

export { POSTERIZE_PASS };

export interface PosterizeParams {
    /** Levels per RGB channel. Total possible colors = `levels³`. @default 4 */
    levels: number;
}

export function createDefaultPosterizeParams(): PosterizeParams {
    return { levels: 4 };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type PosterizeSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreatePosterizePipelineOptions {
    readonly id?: string;
    readonly source: PosterizeSource;
    readonly params?: PosterizeParams;
}

export function createPosterizePipelineConfig(
    opts: CreatePosterizePipelineOptions,
): { config: PipelineConfig; params: PosterizeParams } {
    const params = opts.params ?? createDefaultPosterizeParams();

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
        id: opts.id ?? "posterize",
        passes: [
            {
                id: "main",
                fragment: composeFragment(POSTERIZE_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uLevels: {
                        kind: "computed",
                        fn: () => params.levels,
                    },
                },
            },
        ],
        outputs: { final: "main" },
    };

    return { config, params };
}
