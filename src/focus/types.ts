/**
 * A salient point inside a detection box — used to drive SAM 2 point
 * prompts (e.g. nipple targeting inside a `FEMALE_BREAST_EXPOSED` box).
 */
export interface FocusPoint {
    readonly x: number;
    readonly y: number;
    /** 0..1 confidence; for `boxCenter` this is always 1. */
    readonly score: number;
}
