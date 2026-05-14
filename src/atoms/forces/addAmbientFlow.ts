import type { Atom } from "../types";

/**
 * Time-driven autonomous stir centered on the canvas. A cosine pair with
 * a 90° phase offset (1.57 rad) sweeps a circular force vector around
 * the centre at ~0.3 rad per render frame. The force is shaped by the
 * same inverse-square falloff kernel as `addPointerForce`.
 *
 * **Diverges from v1** in two ways:
 *   1. Uses an explicit `uHasPointer` uniform as the activation guard
 *      instead of `iMouse.x < 1.0`. The v1 guard re-activates the
 *      ambient flow when the pointer leaves the canvas to the left/top
 *      (position.x goes negative). v2's sticky flag stays at 1.0 once
 *      the pointer signal has fired at least once.
 *   2. Drops the `fract()` from the screen-space distance metric —
 *      otherwise the falloff is periodic and the rightmost pixel
 *      receives full-strength force from a center stir.
 *
 * **Known issue retained**: the ambient flow's amplitude has no matched
 * dissipation, so even while it's running it slowly inflates the field
 * energy beyond what viscosity can drain. We're keeping that for v1
 * visual parity in the pre-interaction phase; once the user moves once
 * it's off forever.
 *
 * Reads: `vec2 pos`, `float iTime`, `float uHasPointer`
 * Writes: `state.xy` (mutates in place)
 * Uniforms: `uAmbientFlow` (float), `uHasPointer` (float — 0 or 1)
 *
 * Source: [glsl.ts:66-70](../../shaders/liquid-metal/glsl.ts#L66-L70).
 */
export const addAmbientFlow: Atom = {
    id: "addAmbientFlow",
    uniforms: `
        uniform float uAmbientFlow;
        uniform float uHasPointer;
    `,
    body: `
        if (uHasPointer < 0.5) {
            vec2 afRes = vec2(textureSize(iChannel0, 0));
            vec2 afScr = (pos - afRes * 0.5) / afRes.x;
            state.xy += uAmbientFlow * cos(iTime * 0.3 - vec2(0.0, 1.57))
                / (dot(afScr, afScr) / 0.05 + 0.05);
        }
    `,
};
