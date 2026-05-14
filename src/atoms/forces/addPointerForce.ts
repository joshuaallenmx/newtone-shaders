import type { Atom } from "../types";

/**
 * **DIAGNOSTIC VERSION** — unconditionally injects a constant negative
 * `state.x` contribution AND the pointer-driven force. Used to test
 * whether the simulation can store negative `state.x` at all.
 *
 * If after a hard reload the top half of the debug view shows BLUE
 * everywhere → state CAN be negative, the bug is the iPointerDelta
 * uniform binding (or the projection from PointerState.delta).
 *
 * If the top half stays RED or BLACK → something prevents state.x
 * from ever being negative — likely a render-target format issue or
 * an unexpected clamp somewhere.
 *
 * Revert this file once the diagnosis is done.
 */
export const addPointerForce: Atom = {
    id: "addPointerForce",
    uniforms: `
        uniform float uPointerForce;
        uniform float uPointerBaseRadius;
        uniform float uPointerSpeedScale;
    `,
    body: `
        // TEMP DIAGNOSTIC: force a constant negative x contribution.
        state.x -= 0.05;

        vec2 pfRes = vec2(textureSize(iChannel0, 0));
        vec2 pfScr = (pos - iMouse.xy) / pfRes.x;
        float pfSpeed = length(iPointerDelta);
        float pfRadius = uPointerBaseRadius + uPointerSpeedScale * pfSpeed;
        float pfFalloff = dot(pfScr, pfScr) / (pfRadius * pfRadius) + 1.0;
        state.xy += uPointerForce * iPointerDelta / pfFalloff;
    `,
};
