import type { FocusPoint } from "./types";

/**
 * Trivial focus heuristic: the geometric center of a `[x, y, w, h]` box.
 * Used as the first-pass point prompt for SAM 2 nipple targeting before
 * adding chromatic refinement.
 */
export function boxCenter(
    box: readonly [number, number, number, number],
): FocusPoint {
    const [x, y, w, h] = box;
    return { x: x + w / 2, y: y + h / 2, score: 1 };
}
