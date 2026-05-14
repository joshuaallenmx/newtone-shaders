import type { Atom } from "../types";

/**
 * Clears the sim buffer for the first few frames so whatever garbage was
 * in the render target before allocation doesn't poison the first
 * physical state. `iFrame <= 4` is v1's threshold — generous enough that
 * a momentary stall (e.g. the texture upload landing late) still ends up
 * cleared.
 *
 * Belongs at the END of the sim-pass atom chain because it overrides
 * everything else for the first few frames.
 *
 * Reads: `int iFrame`
 * Writes: `vec4 state`
 *
 * Source: [glsl.ts:83](../../shaders/liquid-metal/glsl.ts#L83).
 */
export const gateInit: Atom = {
    id: "gateInit",
    body: `if (iFrame <= 4) state = vec4(0.0);`,
};
