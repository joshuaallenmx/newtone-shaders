/**
 * Tier 3 of the detection pipeline: pixel-precise segmentation via SAM 2.
 *
 * The image encoder is heavy (~150MB model, hundreds of ms even on WebGPU)
 * but only depends on the source image. The mask decoder is light and
 * depends on a per-prompt box. The `ImageSegmenter` API splits these so
 * the encoder runs once per `setImage` and the decoder runs once per
 * `segmentBox`, which is the exact pattern called out in the project
 * brief.
 */
export type SegmentSource =
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap
    | OffscreenCanvas;

export interface SegmentMask {
    /** Binary mask in source-image coordinates: 0 outside, 255 inside. */
    readonly data: Uint8Array;
    readonly width: number;
    readonly height: number;
    /** Best-mask IoU score from SAM's mask quality predictor. */
    readonly score: number;
}

/**
 * Which mask of the SAM multi-mask output to return.
 *  - "best": highest predicted IoU (default, matches single-mask use).
 *  - "largest" / "medium" / "smallest": sort the 3 masks by pixel area
 *    and pick by rank — useful for hierarchical prompts (e.g. point at
 *    a breast → "smallest" tends to land on the areola).
 */
export type SegmentScale = "best" | "largest" | "medium" | "smallest";

export interface SegmentPointOptions {
    readonly scale?: SegmentScale;
}

/**
 * One element of a multi-point prompt. `positive: true` means "include this
 * region in the mask"; `false` means "exclude this region" — used to carve
 * unwanted areas out of an existing mask via shift-click refinement.
 */
export interface SegmentPoint {
    readonly x: number;
    readonly y: number;
    readonly positive: boolean;
}

export interface ImageSegmenter {
    /**
     * Run the SAM 2 vision encoder on `image` and cache the embeddings.
     * Subsequent `segmentBox` / `segmentPoint` / `segmentPoints` calls
     * reuse them until `setImage` is called again.
     */
    setImage(image: SegmentSource): Promise<void>;
    /**
     * Run the SAM 2 mask decoder for a single box prompt against the
     * currently-set image. Box is `[x, y, w, h]` in source pixels.
     */
    segmentBox(
        box: readonly [number, number, number, number],
    ): Promise<SegmentMask>;
    /**
     * Run the SAM 2 mask decoder for a single positive point prompt.
     * Point is `[x, y]` in source pixels. Returns one of the multi-mask
     * outputs selected by `options.scale` (default "best").
     */
    segmentPoint(
        point: readonly [number, number],
        options?: SegmentPointOptions,
    ): Promise<SegmentMask>;
    /**
     * Run the SAM 2 mask decoder for a multi-point prompt with positive
     * and/or negative labels per point. Use this to add a click and to
     * refine an existing mask with negative carve-out points.
     */
    segmentPoints(
        points: readonly SegmentPoint[],
        options?: SegmentPointOptions,
    ): Promise<SegmentMask>;
    dispose(): void;
}

export type SegmentDevice =
    | "auto"
    | "webgpu"
    | "wasm"
    | "cpu"
    | "webnn"
    | "webnn-gpu"
    | "webnn-cpu";

export type SegmentDtype = "auto" | "fp32" | "fp16" | "q8" | "q4";

export interface SegmenterProgress {
    readonly status: string;
    readonly file?: string;
    readonly progress?: number;
    readonly loaded?: number;
    readonly total?: number;
}

export interface LoadSegmenterOptions {
    /**
     * HuggingFace model id. Defaults to
     * `onnx-community/sam2-hiera-tiny-ONNX` (the transformers.js-compatible
     * port of the smallest SAM 2 image model, with config + processor
     * files alongside the ONNX weights). Other options on the same API:
     * `onnx-community/sam2.1-hiera-tiny-ONNX`,
     * `onnx-community/sam2-hiera-small-ONNX`.
     */
    readonly modelId?: string;
    /** Default: `"webgpu"` if available, else `"wasm"`. */
    readonly device?: SegmentDevice;
    /** Default: `"fp32"` on WebGPU, `"q8"` on WASM. */
    readonly dtype?: SegmentDtype;
    /**
     * Receives transformers.js download / init events. Useful for showing
     * a "downloading model… X%" spinner.
     */
    readonly onProgress?: (event: SegmenterProgress) => void;
}
