import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import { EDGE_DETECT_PASS } from "./glsl";

export { EDGE_DETECT_PASS };

export interface EdgeDetectParams {
    /** Linear multiplier on Sobel gradient magnitude. @default 1.0 */
    edgeStrength: number;
    /** Smoothstep low-edge — gradients below this are clipped to black. @default 0.0 */
    edgeThreshold: number;
    /** Smoothstep high-edge — gradients above this are full white. @default 1.0 */
    edgeKnee: number;
}

export function createDefaultEdgeDetectParams(): EdgeDetectParams {
    return {
        edgeStrength: 1,
        edgeThreshold: 0,
        edgeKnee: 1,
    };
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type EdgeDetectSource = THREE.Texture | (() => THREE.Texture | null);

export interface CreateEdgeDetectPipelineOptions {
    readonly id?: string;
    /** Source media texture — `THREE.Texture` for images, `THREE.VideoTexture` for video. */
    readonly source: EdgeDetectSource;
    /** Mutable knobs; safe to mutate fields each frame. */
    readonly params?: EdgeDetectParams;
}

/**
 * Build a `PipelineConfig` for the edge-detect shader. Single screen pass
 * that runs Sobel on the source's luminance and writes a grayscale edge map.
 */
export function createEdgeDetectPipelineConfig(
    opts: CreateEdgeDetectPipelineOptions,
): { config: PipelineConfig; params: EdgeDetectParams } {
    const params = opts.params ?? createDefaultEdgeDetectParams();

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

    const knob = (key: keyof EdgeDetectParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key],
    });

    const config: PipelineConfig = {
        id: opts.id ?? "edge-detect",
        passes: [
            {
                id: "edges",
                fragment: composeFragment(EDGE_DETECT_PASS),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: sourceProvider,
                    uEdgeStrength: knob("edgeStrength"),
                    uEdgeThreshold: knob("edgeThreshold"),
                    uEdgeKnee: knob("edgeKnee"),
                },
            },
        ],
        outputs: { final: "edges" },
    };

    return { config, params };
}
