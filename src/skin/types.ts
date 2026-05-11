export {
    type SkinMaskParams,
    createDefaultSkinMaskParams,
} from "../shaders/skin-mask";

export type SkinClassifyInput =
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap
    | OffscreenCanvas
    | ImageData;

export interface ClassifySkinOptions {
    /**
     * YCbCr window parameters. Defaults to `createDefaultSkinMaskParams()`
     * — the Chai & Ngan 1999 chroma window with Y left wide-open.
     */
    readonly params?: import("../shaders/skin-mask").SkinMaskParams;
    /**
     * Skin-score threshold for binarization, 0..1. Pixels at or above the
     * threshold are emitted as 255, others as 0. @default 0.5
     */
    readonly threshold?: number;
}
