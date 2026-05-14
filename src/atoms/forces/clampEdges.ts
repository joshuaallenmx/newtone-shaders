import type { Atom } from "../types";

/**
 * Absorbing boundary condition. Inside a narrow band of width
 * `uEdgeBandPx` (**render pixels**) at each canvas edge, all four
 * components of the simulation state are forced to zero — velocity and
 * surface height alike. The visible result is a "calm ring" around the
 * perimeter that contains the fluid without depending on which
 * direction the field happens to be flowing.
 *
 * **Units**: the band width is in render pixels (`gl_FragCoord` pixels
 * after the DPR cap, *not* CSS pixels). This gives a visually consistent
 * thickness on all four edges regardless of canvas aspect ratio. An
 * earlier draft used UV fractions, which made the band twice as thick
 * on the long axis of any non-square canvas.
 *
 * Each axis's pixel width is converted to a UV threshold using that
 * axis's resolution:
 *   • x edges: band = uEdgeBandPx / iResolution.x
 *   • y edges: band = uEdgeBandPx / iResolution.y
 *
 * Reads: `vec2 uv`, `vec4 state`, `vec3 iResolution`
 * Writes: `vec4 state` (zeros within band)
 * Uniforms: `uEdgeBandPx` (float — band width in render pixels)
 */
export const clampEdges: Atom = {
    id: "clampEdges",
    uniforms: `uniform float uEdgeBandPx;`,
    body: `
        vec2 ceBand = vec2(
            uEdgeBandPx / iResolution.x,
            uEdgeBandPx / iResolution.y
        );
        float beL = step(uv.x, ceBand.x);
        float beR = step(1.0 - ceBand.x, uv.x);
        float beB = step(uv.y, ceBand.y);
        float beT = step(1.0 - ceBand.y, uv.y);
        float beAny = max(max(beL, beR), max(beB, beT));
        state = mix(state, vec4(0.0), beAny);
    `,
};
