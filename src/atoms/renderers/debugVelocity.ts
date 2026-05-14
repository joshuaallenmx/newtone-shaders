import type { Atom } from "../types";

/**
 * Two-layer X-axis diagnostic. Splits the canvas horizontally:
 *
 *   • **Top half**: `state.x` (sim buffer's velocity, what the rendering
 *     would normally use). Red = positive, blue = negative.
 *   • **Bottom half**: `iPointerDelta.x` (raw uniform value, no sim
 *     processing). Red = positive, blue = negative.
 *
 * Reading the result during a left-drag:
 *   • Bottom half blue + top half blue → both layers see negative;
 *     the bug is elsewhere (visualization mixing).
 *   • Bottom half blue + top half red → `iPointerDelta.x` is negative
 *     but `state.x` ends up positive — bug in the sim atoms.
 *   • Bottom half red + top half red → `iPointerDelta.x` is positive
 *     even for left drag — CPU-side bug in `onMove`.
 *   • Bottom half black + top half black → events not landing.
 */
export const debugVelocity: Atom = {
    id: "debugVelocity",
    body: `
        state = texture(iChannel0, uv);

        float dvT = 0.001;
        col = vec3(0.0);

        if (uv.y > 0.5) {
            // Top half: state.x sign.
            col.r = step(dvT, state.x);
            col.b = step(dvT, -state.x);
        } else {
            // Bottom half: iPointerDelta.x sign — read directly from
            // the uniform so the sim atoms aren't in the loop.
            col.r = step(dvT, iPointerDelta.x);
            col.b = step(dvT, -iPointerDelta.x);
        }

        // Thin horizontal divider at uv.y = 0.5 so the two halves are
        // clearly demarcated.
        if (abs(uv.y - 0.5) < 0.002) col = vec3(0.3);

        float dvMouseDistPx = distance(pos, iMouse.xy);
        col = mix(col, vec3(1.0), step(dvMouseDistPx, 20.0));

        float dvCenterDistPx = distance(pos, iResolution.xy * 0.5);
        col = mix(col, vec3(1.0, 0.0, 0.0), step(dvCenterDistPx, 40.0));
    `,
};
