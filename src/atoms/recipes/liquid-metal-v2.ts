import * as THREE from "three";
import { compose } from "../compose";
import type { Recipe } from "../types";
import type { UniformProvider } from "../../core/pipeline";
import type { PointerState, Signal } from "../../core/signals";
import { advect } from "../forces/advect";
import { damp } from "../forces/damp";
import { addAmbientFlow } from "../forces/addAmbientFlow";
import { addPointerForce } from "../forces/addPointerForce";
import { clampEdges } from "../forces/clampEdges";
import { gateInit } from "../forces/gateInit";
import { heightToNormals } from "../renderers/heightToNormals";
import { viewDirFromFragCoord } from "../renderers/viewDirFromFragCoord";
import { sampleEnvironment } from "../renderers/sampleEnvironment";
import { multiplyComposite } from "../renderers/multiplyComposite";
import { debugVelocity } from "../renderers/debugVelocity";

export interface LiquidMetalV2RecipeOptions {
    /** Env reflection gradient texture. */
    readonly envTexture: THREE.Texture;
    /** Pointer signal — required for `addPointerForce`. */
    readonly pointer: Signal<PointerState>;
    /**
     * Sticky flag — `true` once the pointer signal has fired at least
     * once, and never reverts. Drives the `uHasPointer` uniform that
     * gates the ambient-flow atom off after first user interaction.
     */
    readonly hasPointer: { readonly current: boolean };
    /** Final multiplicative tint, default white. */
    readonly tint?: THREE.Vector3;
    /** Additive brightness lift on env samples, mirrors v1 default 0.15. */
    readonly envBrightnessBoost?: number;
    /** Finite-difference step (in canvas-width units). v1 default = 1.4. */
    readonly gradientDelta?: number;
    /** Velocity advection scale. v1 default = 5.0. */
    readonly advectionScale?: number;
    /** Viscosity-mix factor for the velocity update. v1 default = 0.025. */
    readonly viscosity?: number;
    /**
     * Ambient autonomous-stir amplitude. v1 default = 0.003, but the
     * oscillator biases the field toward whatever direction it was
     * pointing at first user interaction, which reads as a directional
     * preference / drift. v2 default is `0` (no autonomous stir — the
     * field is at rest until the user moves the pointer). Set to a
     * positive value to opt back in.
     */
    readonly ambientFlow?: number;
    /**
     * Per-pointermove force scale. With the v2 falloff formula
     * `1/(dot/R² + 1)` the center value is 1 (vs. v1's `20`), so the
     * effective force is ~20× weaker per unit `uPointerForce`. Default
     * bumped to compensate. @default 0.006
     */
    readonly pointerForce?: number;
    /**
     * Minimum falloff radius for the pointer force, in UV units
     * (canvas-width fractions). The force is 1.0 at the cursor and
     * 0.5 at this distance. @default 0.08
     */
    readonly pointerBaseRadius?: number;
    /**
     * How much extra falloff radius is added per render-pixel of
     * pointer speed (`length(iPointerDelta)`). Larger value = the
     * radius scales more aggressively with cursor speed. @default 0.005
     */
    readonly pointerSpeedScale?: number;
    /**
     * Diagnostic mode — replaces the production render chain
     * (`heightToNormals → viewDir → env → multiplyComposite`) with a
     * single `debugVelocity` renderer that paints `state.xy` as RGB.
     * Use to verify the sim is producing symmetric velocity in all
     * four cardinal directions. @default false
     */
    readonly debug?: boolean;
    /**
     * Thickness of the absorbing edge band, in **render pixels**.
     * Same value applies to all four edges regardless of canvas
     * aspect ratio. @default 8
     */
    readonly edgeBandPx?: number;
}

const projectPointerPosition = (s: PointerState): THREE.Vector4 => s.position;
const projectPointerDelta = (s: PointerState): THREE.Vector2 => s.delta;

/**
 * Step 3b recipe — minimum sim + pointer interactivity. Sim atoms now
 * include `addPointerForce`, fed by the CPU-side pointer signal's
 * `position` (→ `iMouse`) and `delta` (→ `iPointerDelta`). No Buffer B
 * needed.
 *
 * Order of sim atoms matches v1: curl → advect → damp → ambient flow →
 * pointer force → init gate. Init gate is last so it overrides the rest
 * for the first 4 frames.
 */
export function buildLiquidMetalV2Recipe(
    opts: LiquidMetalV2RecipeOptions,
): Recipe {
    const tint = opts.tint ?? new THREE.Vector3(1, 1, 1);
    const envBrightnessBoost = opts.envBrightnessBoost ?? 0.15;
    const gradientDelta = opts.gradientDelta ?? 1.4;
    const advectionScale = opts.advectionScale ?? 5.0;
    const viscosity = opts.viscosity ?? 0.025;
    const ambientFlow = opts.ambientFlow ?? 0;
    const pointerForce = opts.pointerForce ?? 0.006;
    const pointerBaseRadius = opts.pointerBaseRadius ?? 0.08;
    const pointerSpeedScale = opts.pointerSpeedScale ?? 0.005;
    const edgeBandPx = opts.edgeBandPx ?? 1;
    const debug = opts.debug ?? false;

    const iResolutionVec = new THREE.Vector3();
    const iResolutionProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => {
            iResolutionVec.set(ctx.target.w, ctx.target.h, 1);
            return iResolutionVec;
        },
    };
    const iTimeProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => ctx.time,
    };
    const iFrameProvider: UniformProvider = {
        kind: "computed",
        fn: (ctx) => ctx.frame,
    };
    const iMouseProvider: UniformProvider = {
        kind: "signal",
        signal: opts.pointer,
        project: projectPointerPosition as (s: unknown) => THREE.Vector4,
    };
    const iPointerDeltaProvider: UniformProvider = {
        kind: "signal",
        signal: opts.pointer,
        project: projectPointerDelta as (s: unknown) => THREE.Vector2,
    };
    const uHasPointerProvider: UniformProvider = {
        kind: "computed",
        fn: () => (opts.hasPointer.current ? 1.0 : 0.0),
    };

    return {
        id: "liquid-metal-v2",
        passes: [
            {
                id: "sim",
                target: { kind: "pingpong", size: "full" },
                atoms: [
                    advect,
                    damp,
                    addAmbientFlow,
                    addPointerForce,
                    clampEdges,
                    gateInit,
                ],
                uniforms: {
                    iTime: iTimeProvider,
                    iFrame: iFrameProvider,
                    iResolution: iResolutionProvider,
                    iMouse: iMouseProvider,
                    iPointerDelta: iPointerDeltaProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "sim" },
                    },
                    uAdvectionScale: { kind: "static", value: advectionScale },
                    uViscosity: { kind: "static", value: viscosity },
                    uAmbientFlow: { kind: "static", value: ambientFlow },
                    uHasPointer: uHasPointerProvider,
                    uPointerForce: { kind: "static", value: pointerForce },
                    uPointerBaseRadius: { kind: "static", value: pointerBaseRadius },
                    uPointerSpeedScale: { kind: "static", value: pointerSpeedScale },
                    uEdgeBandPx: { kind: "static", value: edgeBandPx },
                },
            },
            {
                id: "image",
                target: { kind: "screen" },
                atoms: debug
                    ? [debugVelocity]
                    : [
                          heightToNormals,
                          viewDirFromFragCoord,
                          sampleEnvironment,
                          multiplyComposite,
                      ],
                uniforms: {
                    iResolution: iResolutionProvider,
                    // iMouse must be bound on the image pass too — atoms
                    // like debugVelocity (and anything that needs to know
                    // the cursor) read it. Without this binding the
                    // image pass's iMouse stays at the WebGL default
                    // (0,0,0,0) regardless of what the sim pass sees.
                    iMouse: iMouseProvider,
                    // iPointerDelta also bound on the image pass so
                    // diagnostic atoms can read the raw delta uniform
                    // without going through the sim.
                    iPointerDelta: iPointerDeltaProvider,
                    iChannel0: {
                        kind: "texture",
                        ref: { kind: "pass", passId: "sim" },
                    },
                    iChannel2: {
                        kind: "texture",
                        ref: { kind: "asset", texture: opts.envTexture },
                    },
                    uTint: { kind: "static", value: tint },
                    uEnvBrightnessBoost: {
                        kind: "static",
                        value: envBrightnessBoost,
                    },
                    uGradientDelta: { kind: "static", value: gradientDelta },
                },
            },
        ],
        outputs: { final: "image" },
    };
}

export function createLiquidMetalV2(opts: LiquidMetalV2RecipeOptions) {
    return compose(buildLiquidMetalV2Recipe(opts));
}
