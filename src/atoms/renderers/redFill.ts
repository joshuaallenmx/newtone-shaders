import type { Atom } from "../types";

/**
 * Plumbing-verification stub: paints opaque red by overwriting the shared
 * `col` accumulator. Compose auto-writes `fragColor = vec4(col, 1.0)` for
 * screen-target passes, so this atom alone produces a red frame.
 *
 * Kept as a regression fixture — useful when refactoring `compose.ts`.
 */
export const redFill: Atom = {
    id: "redFill",
    body: `col = vec3(1.0, 0.0, 0.0);`,
};
