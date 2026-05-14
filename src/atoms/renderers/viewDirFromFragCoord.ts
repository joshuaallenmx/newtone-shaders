import type { Atom } from "../types";

/**
 * Constructs a view-direction unit vector pointing from a virtual camera at
 * `(0, 0, 0)` through this pixel, with the canvas plane at `z = -1` and the
 * canvas's wider axis spanning `x ∈ [-0.5, +0.5]` (an "aspect-correct" NDC
 * with the X axis driving scale). Subsequent atoms reflect this vector off
 * the surface normal to look up the environment.
 *
 * Writes: `vec3 dir`
 *
 * Source: [glsl.ts:133-134](../../shaders/liquid-metal/glsl.ts#L133-L134).
 */
export const viewDirFromFragCoord: Atom = {
    id: "viewDirFromFragCoord",
    body: `
        vec2 ndc = (pos - iResolution.xy * 0.5) / iResolution.x;
        vec3 dir = normalize(vec3(ndc, -1.0));
    `,
};
