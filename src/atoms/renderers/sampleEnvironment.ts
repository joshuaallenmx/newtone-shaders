import type { Atom } from "../types";

/**
 * Reflects the view direction off the surface normal and samples a 2D
 * environment texture as a poor-man's cubemap. The vertical gradient stored
 * in `iChannel2` becomes a luminance ramp the surface "sees". A uniform
 * brightness boost is added so the env can lift the multiplicative composite
 * out of pure black at the dark stops.
 *
 * The swizzle `R.xzy` (in v1, the `myenv` helper takes the same swizzle)
 * remaps the reflection vector so the gradient's Y axis lines up with the
 * world's "up" in the reflected scene.
 *
 * Reads: `vec3 dir`, `vec3 n`
 * Writes: `vec3 refl`
 * Uniforms: `uEnvBrightnessBoost` (float), `iChannel2` (sampler2D — env)
 *
 * Source: [glsl.ts:115-117 + 135-136](../../shaders/liquid-metal/glsl.ts#L115-L136).
 */
export const sampleEnvironment: Atom = {
    id: "sampleEnvironment",
    uniforms: `uniform float uEnvBrightnessBoost;`,
    definitions: `
        vec3 sampleEnvDir(vec3 reflDir, sampler2D envTex, float boost) {
            return texture(envTex, reflDir.xy * 0.5 + 0.5).xyz + boost;
        }
    `,
    body: `
        vec3 R = reflect(dir, n);
        vec3 refl = sampleEnvDir(R.xzy, iChannel2, uEnvBrightnessBoost);
    `,
};
