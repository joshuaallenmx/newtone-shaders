import type { Atom } from "../types";

/**
 * Strict-multiplicative composite: `col = col * refl * uTint`.
 *
 * This is *the* defining characteristic of the liquid-metal look — three
 * multiplied terms (base × reflection × tint). Because everything is
 * multiplied, a single dark term collapses the whole image to black; this is
 * why the v1 wrapper hardcodes `uTint` to white even in dark mode.
 *
 * Reads: `vec3 col`, `vec3 refl`
 * Writes: `vec3 col` (mutates in place)
 * Uniforms: `uTint` (vec3)
 *
 * Source: [glsl.ts:142](../../shaders/liquid-metal/glsl.ts#L142).
 */
export const multiplyComposite: Atom = {
    id: "multiplyComposite",
    uniforms: `uniform vec3 uTint;`,
    body: `col = col * refl * uTint;`,
};
