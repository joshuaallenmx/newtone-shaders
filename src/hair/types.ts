export {
    type HairDetectParams,
    createDefaultHairDetectParams,
} from "../shaders/hair-detect";

export type HairClassifyInput =
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap
    | OffscreenCanvas
    | ImageData;

export interface ClassifyHairOptions {
    /**
     * Hair-detection parameters. Defaults to
     * `createDefaultHairDetectParams()` — same window the GPU shader uses.
     */
    readonly params?: import("../shaders/hair-detect").HairDetectParams;
    /**
     * Score threshold for binarization, 0..1. @default 0.5
     */
    readonly threshold?: number;
}

/**
 * Pre-decoded RGBA pixel data accepted by `hairScoreMap`. Lets the focus
 * pipeline pass cropped `getImageData` output without a redundant draw.
 */
export interface HairScoreInput {
    readonly data: Uint8ClampedArray | Uint8Array;
    readonly width: number;
    readonly height: number;
}
