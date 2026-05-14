import type { PassTarget, UniformProvider } from "../core/pipeline";

/**
 * A single composable shader operation — one Navier-Stokes term, one input
 * force, one render translation. Atoms contribute three text fragments which
 * the composer concatenates into a single fragment shader:
 *
 * 1. `uniforms` — GLSL `uniform` declarations this atom requires (deduped at
 *    compose time, so two atoms can declare the same uniform safely). Skip
 *    framework-provided uniforms already in the SHADERTOY_PRELUDE
 *    (`iTime`, `iFrame`, `iResolution`, `iMouse`, `iChannel0..3`, etc.).
 * 2. `definitions` — GLSL helper functions or constants placed before
 *    `mainImage`. The atom's own function definition lives here.
 * 3. `body` — one or more GLSL statements inserted at the call site inside
 *    `mainImage`, in recipe order.
 *
 * Convention: atom bodies read and write a shared set of named locals
 * declared at the top of `mainImage`: `pos` (vec2 fragCoord copy),
 * `uv` (vec2 normalized), `state` (vec4 field sample / output accumulator).
 * Force-pass atoms operate on `state`; render-pass atoms produce a `vec3 col`
 * accumulator and a `vec3 n` normal as needed.
 */
export interface Atom {
    readonly id: string;
    readonly uniforms?: string;
    readonly definitions?: string;
    readonly body: string;
}

/**
 * One pass of a recipe — a list of atoms to fuse into one fragment shader,
 * plus the target (screen, pingpong, fixed) and the provider bindings for
 * every uniform any atom in this pass declares.
 */
export interface PassRecipe {
    readonly id: string;
    readonly target: PassTarget;
    readonly atoms: readonly Atom[];
    readonly uniforms?: Readonly<Record<string, UniformProvider>>;
}

/**
 * A full recipe — typically two passes (simulation + render) but the shape
 * supports any number. Maps onto `PipelineConfig` after `compose()`.
 */
export interface Recipe {
    readonly id: string;
    readonly passes: readonly PassRecipe[];
    readonly outputs?: Readonly<Record<string, string>>;
}
