import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import {
    CIRCLE_EDGE_PASS,
    CIRCLE_HOUGH_PASS,
    CIRCLE_RENDER_PASS,
} from "./glsl";

export { CIRCLE_EDGE_PASS, CIRCLE_HOUGH_PASS, CIRCLE_RENDER_PASS };

/**
 * Pass id for the Hough accumulator. Exposed so the React component can
 * read it back via `pipeline.getPassTarget(CIRCLE_HOUGH_PASS_ID)`.
 */
export const CIRCLE_HOUGH_PASS_ID = "hough";

export type CircleDetectMode =
    | "source"
    | "accumulator"
    | "mask"
    | "overlay"
    | "edges";

export const CIRCLE_DETECT_MODE_INDEX: Record<CircleDetectMode, number> = {
    accumulator: 0,
    mask: 1,
    overlay: 2,
    edges: 3,
    source: 4,
};

export interface CircleDetectParams {
    /** Target circle radius in canvas pixels. @default 60 */
    radius: number;
    /** Half-width of the radius sweep — circles ±this from `radius` still register. @default 10 */
    radiusSpread: number;
    /** Angular samples per radius (1..128). More = stricter / cleaner. @default 48 */
    samples: number;
    /** Threshold for mask / accumulator-derived ring detection. @default 0.35 */
    minScore: number;
    /** Output composition. @default "source" */
    mode: CircleDetectMode;
    /** Source brightness in overlay mode. @default 0.35 */
    offMix: number;
}

export function createDefaultCircleDetectParams(): CircleDetectParams {
    return {
        radius: 60,
        radiusSpread: 10,
        samples: 48,
        minScore: 0.35,
        mode: "source",
        offMix: 0.35,
    };
}

/**
 * A discrete circle extracted from the Hough accumulator, in CSS-pixel
 * (display) coordinates with origin at the top-left of the visible canvas.
 * Suitable for `ctx.arc(circle.cx, circle.cy, circle.r, 0, 2π)`.
 */
export interface DetectedCircle {
    /** Center X in CSS pixels. */
    cx: number;
    /** Center Y in CSS pixels. */
    cy: number;
    /** Radius in CSS pixels. */
    r: number;
    /** Confidence score in `[0, 1]`. */
    score: number;
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type CircleDetectSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateCircleDetectPipelineOptions {
    readonly id?: string;
    readonly source: CircleDetectSource;
    readonly params?: CircleDetectParams;
}

export function createCircleDetectPipelineConfig(
    opts: CreateCircleDetectPipelineOptions,
): { config: PipelineConfig; params: CircleDetectParams } {
    const params = opts.params ?? createDefaultCircleDetectParams();

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

    const numKnob = (key: keyof CircleDetectParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key] as number,
    });

    const config: PipelineConfig = {
        id: opts.id ?? "circle-detect",
        passes: [
            {
                id: "edges",
                fragment: composeFragment(CIRCLE_EDGE_PASS),
                target: { kind: "pingpong", size: "full" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                },
            },
            {
                id: CIRCLE_HOUGH_PASS_ID,
                fragment: composeFragment(CIRCLE_HOUGH_PASS),
                // Half-resolution accumulator: cheap CPU readback for peak
                // extraction without dropping enough fidelity to matter for
                // typical circle radii.
                target: { kind: "pingpong", size: "half" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "edges" },
                    },
                    uRadius: {
                        kind: "computed",
                        // The hough target is half-res so the radius lives in
                        // accumulator-pixel space — half the canvas-pixel
                        // value the user sees.
                        fn: () => params.radius * 0.5,
                    },
                    uRadiusSpread: {
                        kind: "computed",
                        fn: () => params.radiusSpread * 0.5,
                    },
                    uSamples: {
                        kind: "computed",
                        fn: () => Math.round(params.samples),
                    },
                },
            },
            {
                id: "render",
                fragment: composeFragment(CIRCLE_RENDER_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: CIRCLE_HOUGH_PASS_ID },
                    },
                    iChannel1: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "edges" },
                    },
                    iChannel2: sourceProvider,
                    uMinScore: numKnob("minScore"),
                    uMode: {
                        kind: "computed",
                        fn: () => CIRCLE_DETECT_MODE_INDEX[params.mode],
                    },
                    uOffMix: numKnob("offMix"),
                },
            },
        ],
        outputs: { final: "render", accumulator: CIRCLE_HOUGH_PASS_ID },
    };

    return { config, params };
}
