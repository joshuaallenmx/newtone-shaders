import * as THREE from "three";

/**
 * Mutable color/look uniforms shared between the React layer and the shader.
 * The Pipeline reads these every frame via computed providers, so the React
 * layer can mutate fields in place when props change without rebuilding the
 * pipeline.
 */
export interface LiquidMetalColorUniforms {
    /** Linear-RGB tint multiplier applied at the end of the IMAGE pass. */
    readonly tint: THREE.Vector3;
    /** Linear-RGB highlight color (where the sim is calm). */
    readonly baseColor: THREE.Vector3;
    /** 0..1 — how much sim pattern shows through. */
    contrast: number;
    /** 0..1 — 0 desaturates fully toward luminance. */
    saturation: number;
}

export function createDefaultColorUniforms(): LiquidMetalColorUniforms {
    return {
        tint: new THREE.Vector3(1, 1, 1),
        baseColor: new THREE.Vector3(1, 1, 1),
        contrast: 0.35,
        saturation: 1,
    };
}

/**
 * Mutable fluid-feel parameters. Defaults match the original Shadertoy
 * constants, so a fresh `createDefaultParams()` produces the same look as
 * pre-promotion. Mutate fields in place to drive the shader at runtime.
 */
export interface LiquidMetalParams {
    /** Damping coefficient for the velocity field (xy mix factor). */
    viscosity: number;
    /** Velocity advection scale (the `5.0` factor in the shader). */
    advectionScale: number;
    /** How much each pointer-move event pushes the fluid. */
    pointerForce: number;
    /** Ambient stir applied when the pointer hasn't been seen yet. */
    ambientFlow: number;
    /** Surface drop amplitude — large-scale modulation. */
    dropAmplitudeNear: number;
    /** Surface drop amplitude — fine-scale modulation. */
    dropAmplitudeFar: number;
    /** Finite-difference step for the metal-shading normal estimation. */
    gradientDelta: number;
    /** Brightness boost added to env-reflection samples. */
    envBrightnessBoost: number;
    /** Scroll-velocity-Y multiplier added to the fluid each frame. 0 = off. */
    scrollForce: number;
}

export function createDefaultParams(): LiquidMetalParams {
    return {
        viscosity: 0.025,
        advectionScale: 5.0,
        pointerForce: 0.0003,
        ambientFlow: 0.003,
        dropAmplitudeNear: 0.002,
        dropAmplitudeFar: 0.001,
        gradientDelta: 1.4,
        envBrightnessBoost: 0.15,
        scrollForce: 0,
    };
}
