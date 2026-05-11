import * as THREE from "three";
import type { PipelineConfig, UniformProvider } from "../../core/pipeline";
import { composeFragment } from "../../core/pipeline";
import type {
    Signal,
    PointerState,
    ScrollState,
} from "../../core/signals";
import {
    LIQUID_METAL_BUFFER_A,
    LIQUID_METAL_BUFFER_B,
    LIQUID_METAL_IMAGE,
} from "./glsl";
import {
    createDefaultColorUniforms,
    createDefaultParams,
    type LiquidMetalColorUniforms,
    type LiquidMetalParams,
} from "./uniforms";

export { LIQUID_METAL_BUFFER_A, LIQUID_METAL_BUFFER_B, LIQUID_METAL_IMAGE };
export {
    createDefaultColorUniforms,
    createDefaultParams,
    type LiquidMetalColorUniforms,
    type LiquidMetalParams,
} from "./uniforms";

export interface LiquidMetalSignals {
    /** Pointer position drives the fluid motor. */
    readonly pointer: Signal<PointerState>;
    /** Optional scroll source — only stirs the fluid when `params.scrollForce > 0`. */
    readonly scroll?: Signal<ScrollState>;
}

/** Either a stable texture or a getter the pipeline calls each frame. */
export type EnvTextureSource = THREE.Texture | (() => THREE.Texture | null);

export interface LiquidMetalTextures {
    /** RGBA noise — surface drop modulation in BUFFER_A. */
    readonly noise: THREE.Texture;
    /** Env reflection: gradient `THREE.Texture` or a getter to allow swapping. */
    readonly env: EnvTextureSource;
    /** Barrier mask (white = solid) — collision shape in BUFFER_A. */
    readonly mask: THREE.Texture;
}

export interface CreateLiquidMetalPipelineOptions {
    readonly id?: string;
    readonly signals: LiquidMetalSignals;
    readonly textures: LiquidMetalTextures;
    /** Mutable color knobs; safe to mutate fields each frame. */
    readonly colorUniforms?: LiquidMetalColorUniforms;
    /** Mutable fluid-feel knobs; safe to mutate fields each frame. */
    readonly params?: LiquidMetalParams;
}

const projectPointerPosition = (s: PointerState): THREE.Vector4 => s.position;
const projectScrollVelocity = (s: ScrollState): THREE.Vector2 => s.velocity;
const ZERO_VEC2 = Object.freeze(new THREE.Vector2(0, 0));

/**
 * Build a `PipelineConfig` for the LiquidMetal shader. Caller is responsible
 * for instantiating `new Pipeline(renderer, config)` and starting/stopping the
 * supplied signals.
 */
export function createLiquidMetalPipelineConfig(
    opts: CreateLiquidMetalPipelineOptions,
): {
    config: PipelineConfig;
    colorUniforms: LiquidMetalColorUniforms;
    params: LiquidMetalParams;
} {
    const { signals, textures } = opts;
    const colorUniforms = opts.colorUniforms ?? createDefaultColorUniforms();
    const params = opts.params ?? createDefaultParams();

    const iTimeProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => ctx.time,
    };
    const iFrameProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => ctx.frame,
    };
    const iResolutionVec = new THREE.Vector3();
    const iResolutionProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => {
            iResolutionVec.set(ctx.target.w, ctx.target.h, 1);
            return iResolutionVec;
        },
    };
    const iMouseProvider: UniformProvider = {
        kind: "signal",
        signal: signals.pointer,
        project: projectPointerPosition as (s: unknown) => THREE.Vector4,
    };
    const iScrollVelocityProvider: UniformProvider = signals.scroll
        ? {
              kind: "signal",
              signal: signals.scroll,
              project: projectScrollVelocity as (s: unknown) => THREE.Vector2,
          }
        : { kind: "static", value: ZERO_VEC2 as THREE.Vector2 };

    const envProvider: UniformProvider = (() => {
        if (typeof textures.env === "function") {
            const fn = textures.env;
            return {
                kind: "computed",
                fn: () => fn() ?? (null as unknown as THREE.Texture),
            };
        }
        return {
            kind: "texture",
            ref: { kind: "asset", texture: textures.env },
        };
    })();

    const fluidParam = (key: keyof LiquidMetalParams): UniformProvider => ({
        kind: "computed",
        fn: () => params[key] as number,
    });

    const config: PipelineConfig = {
        id: opts.id ?? "liquid-metal",
        passes: [
            {
                id: "bufferB",
                fragment: composeFragment(LIQUID_METAL_BUFFER_B),
                target: { kind: "pingpong", size: [2, 2] },
                uniforms: {
                    iMouse: iMouseProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "bufferB" },
                    },
                },
            },
            {
                id: "bufferA",
                fragment: composeFragment(LIQUID_METAL_BUFFER_A),
                target: { kind: "pingpong", size: "full" },
                uniforms: {
                    iTime: iTimeProvider,
                    iFrame: iFrameProvider,
                    iResolution: iResolutionProvider,
                    iMouse: iMouseProvider,
                    iScrollVelocity: iScrollVelocityProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "bufferA" },
                    },
                    iChannel1: {
                        kind: "texture",
                        ref: { kind: "asset", texture: textures.noise },
                    },
                    iChannel2: {
                        kind: "texture",
                        ref: { kind: "asset", texture: textures.mask },
                    },
                    iChannel3: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "bufferB" },
                    },
                    uViscosity: fluidParam("viscosity"),
                    uAdvectionScale: fluidParam("advectionScale"),
                    uPointerForce: fluidParam("pointerForce"),
                    uAmbientFlow: fluidParam("ambientFlow"),
                    uDropAmplitudeNear: fluidParam("dropAmplitudeNear"),
                    uDropAmplitudeFar: fluidParam("dropAmplitudeFar"),
                    uScrollForce: fluidParam("scrollForce"),
                },
            },
            {
                id: "image",
                fragment: composeFragment(LIQUID_METAL_IMAGE),
                target: { kind: "screen" },
                uniforms: {
                    iResolution: iResolutionProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "bufferA" },
                    },
                    iChannel1: {
                        kind: "texture",
                        ref: { kind: "asset", texture: textures.noise },
                    },
                    iChannel2: envProvider,
                    uTint: { kind: "static", value: colorUniforms.tint },
                    uBaseColor: { kind: "static", value: colorUniforms.baseColor },
                    uContrast: {
                        kind: "computed",
                        fn: () => colorUniforms.contrast,
                    },
                    uSaturation: {
                        kind: "computed",
                        fn: () => colorUniforms.saturation,
                    },
                    uGradientDelta: fluidParam("gradientDelta"),
                    uEnvBrightnessBoost: fluidParam("envBrightnessBoost"),
                },
            },
        ],
        outputs: { final: "image" },
    };

    return { config, colorUniforms, params };
}
