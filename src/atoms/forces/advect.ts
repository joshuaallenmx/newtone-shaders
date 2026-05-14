import type { Atom } from "../types";

/**
 * Semi-Lagrangian self-advection. Each pixel asks "where did I come
 * from?" by looking backward along its own velocity vector and copying
 * the value at that offset position. This implements the `−(v·∇)v`
 * advection term of Navier-Stokes; the same operation moves any other
 * field stored in the same texture (e.g. surface height in `state.zw`).
 *
 * **Symmetric convention**: `state.xy` is the physical velocity field
 * directly. A positive `state.x` means the field flows right; a
 * positive `state.y` means the field flows up. No `vec2(-1, 1)` flip
 * (v1 had one — see history in the README).
 *
 * Reads the previous frame's velocity at this pixel by sampling
 * `iChannel0` once, then samples again at the advect offset. The
 * `CLAMP_TO_EDGE` wrap mode on the render target means material can't
 * advect off the canvas — combined with `clampEdges` zeroing the band
 * each frame, the fluid stays contained.
 *
 * Reads: `vec2 pos`, `vec2 uv`, `sampler2D iChannel0`
 * Writes: `vec4 state` (the advected previous frame)
 * Uniforms: `uAdvectionScale` (float)
 *
 * Source (modified): [glsl.ts:63](../../shaders/liquid-metal/glsl.ts#L63).
 */
export const advect: Atom = {
    id: "advect",
    uniforms: `uniform float uAdvectionScale;`,
    body: `
        vec2 advRes = vec2(textureSize(iChannel0, 0));
        vec2 advV = textureLod(iChannel0, uv, 0.0).xy;
        state = textureLod(
            iChannel0,
            (pos - advV * uAdvectionScale * sqrt(advRes.x / 600.0)) / advRes,
            0.0
        );
    `,
};
