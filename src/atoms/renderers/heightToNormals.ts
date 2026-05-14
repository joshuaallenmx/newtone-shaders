import type { Atom } from "../types";

/**
 * Treats the RGB of `iChannel0` as a scalar height field via `length(rgb)`
 * and emits a surface normal computed by central finite differences over the
 * four neighbouring texels. The step size `uGradientDelta / iResolution.x`
 * scales with canvas width so the sampling distance is DPR-aware.
 *
 * The output normal stays mostly `+z` (tangent-space convention) with a
 * small in-plane tilt proportional to the local height gradient. The factor
 * `0.02` is v1's choice and acts as a "bump strength" — larger values make
 * the surface visibly bumpier at the cost of more distortion under
 * reflection. Reproduces v1 exactly so v2 starts visually equivalent.
 *
 * Reads: `vec2 uv`, `sampler2D iChannel0` (height field)
 * Writes: `vec3 n` (overwrites the default flat normal)
 * Uniforms: `uGradientDelta` (float)
 *
 * Source: [glsl.ts:120-131](../../shaders/liquid-metal/glsl.ts#L120-L131).
 */
export const heightToNormals: Atom = {
    id: "heightToNormals",
    uniforms: `uniform float uGradientDelta;`,
    definitions: `
        float heightAt(vec2 uv) {
            return length(texture(iChannel0, uv).xyz);
        }
        vec2 heightGradient(vec2 uv, float delta) {
            vec2 d = vec2(delta, 0.0);
            return vec2(
                heightAt(uv + d.xy) - heightAt(uv - d.xy),
                heightAt(uv + d.yx) - heightAt(uv - d.yx)
            ) / delta;
        }
    `,
    body: `
        n = normalize(vec3(
            -heightGradient(uv, uGradientDelta / iResolution.x) * 0.02,
            1.0
        ));
    `,
};
