import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { LOW_POLY_PASS } from "./glsl";

export { LOW_POLY_PASS };

export type LowPolyMode = "facets-edges" | "facets" | "wireframe";
export type LowPolyColorMode = "grayscale" | "hsv";

export const LOW_POLY_MODE_INDEX: Record<LowPolyMode, number> = {
    "facets-edges": 0,
    facets: 1,
    wireframe: 2,
};

export const LOW_POLY_COLOR_INDEX: Record<LowPolyColorMode, number> = {
    grayscale: 0,
    hsv: 1,
};

export interface LowPolyParams {
    /** Angular bins, 4..32. @default 12 */
    facets: number;
    /** Gradient sample distance in source pixels — larger → coarser polygons. @default 3 */
    smoothRadius: number;
    /** Gradient magnitude below which no edge is drawn. @default 0.04 */
    edgeThreshold: number;
    /** Smoothstep width past `edgeThreshold`. @default 0.03 */
    edgeWidth: number;
    /** Fake light direction X. @default 0.4 */
    lightX: number;
    /** Fake light direction Y. @default 0.6 */
    lightY: number;
    /** Output composition. @default "facets-edges" */
    mode: LowPolyMode;
    /** How facets get colored. @default "grayscale" */
    colorMode: LowPolyColorMode;
}

export function createDefaultLowPolyParams(): LowPolyParams {
    return {
        facets: 12,
        smoothRadius: 3,
        edgeThreshold: 0.04,
        edgeWidth: 0.03,
        lightX: 0.4,
        lightY: 0.6,
        mode: "facets-edges",
        colorMode: "grayscale",
    };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type LowPolySource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateLowPolyPipelineOptions {
    readonly id?: string;
    readonly source: LowPolySource;
    readonly params?: LowPolyParams;
}

export function createLowPolyPipelineConfig(
    opts: CreateLowPolyPipelineOptions,
): { config: PipelineConfig; params: LowPolyParams } {
    const params = opts.params ?? createDefaultLowPolyParams();

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

    const numKnob = (key: keyof LowPolyParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key] as number,
    });

    const config: PipelineConfig = {
        id: opts.id ?? "low-poly",
        passes: [
            {
                id: "stylize",
                fragment: composeFragment(LOW_POLY_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uFacets: {
                        kind: "computed",
                        fn: () => Math.round(params.facets),
                    },
                    uSmoothRadius: numKnob("smoothRadius"),
                    uEdgeThreshold: numKnob("edgeThreshold"),
                    uEdgeWidth: numKnob("edgeWidth"),
                    uLightX: numKnob("lightX"),
                    uLightY: numKnob("lightY"),
                    uMode: {
                        kind: "computed",
                        fn: () => LOW_POLY_MODE_INDEX[params.mode],
                    },
                    uColorMode: {
                        kind: "computed",
                        fn: () => LOW_POLY_COLOR_INDEX[params.colorMode],
                    },
                },
            },
        ],
        outputs: { final: "stylize" },
    };

    return { config, params };
}
