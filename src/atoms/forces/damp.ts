import type { Atom } from "../types";

/**
 * Multiplicative decay of velocity. Each frame, `state.xy` is mixed
 * toward zero by `uViscosity`. With viscosity ≈ 0.025, ~97.5% of last
 * frame's velocity survives — the field decays slowly, a stir lingers
 * for ~3 seconds before fading. Larger viscosity = faster decay,
 * stiffer fluid.
 *
 * Stands in for the `ν∇²v` diffusion term. Not a true Laplacian — a
 * proper viscosity would diffuse the field across neighbours; this just
 * decays it. But for a paint-like sim where the goal is "stir lingers
 * then fades", multiplicative decay matches the visual target with one
 * fragment-shader op instead of N Jacobi iterations.
 *
 * **Removed from v1**: v1's damp blended `state.xy` toward
 * `multiScaleCurl`'s output velocity. Since we no longer run
 * `multiScaleCurl` in the default sim, there's no v to blend toward —
 * decay-to-zero is the right replacement.
 *
 * Writes: `state.xy` (mutates in place)
 * Uniforms: `uViscosity` (float, 0..1)
 *
 * Source (modified): [glsl.ts:64](../../shaders/liquid-metal/glsl.ts#L64).
 */
export const damp: Atom = {
    id: "damp",
    uniforms: `uniform float uViscosity;`,
    body: `state.xy = mix(state.xy, vec2(0.0), uViscosity);`,
};
