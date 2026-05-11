/**
 * Tier 2 of the detection pipeline: macro localization via NudeNet ONNX.
 *
 * A `DetectedRegion` is an axis-aligned bounding box in source-image pixel
 * space, paired with the NudeNet class label and the model's confidence
 * score. Tier 3 (SAM) consumes these boxes as prompts to produce
 * pixel-precise masks within each region.
 */
export interface DetectedRegion {
    readonly class: string;
    readonly box: readonly [number, number, number, number];
    readonly score: number;
}

export type DetectInput =
    | HTMLImageElement
    | HTMLCanvasElement
    | HTMLVideoElement
    | ImageBitmap
    | OffscreenCanvas;

export interface DetectOptions {
    /**
     * Minimum class confidence to keep a candidate before NMS. NudeNet's
     * reference implementation uses 0.2. @default 0.2
     */
    readonly scoreThreshold?: number;
    /**
     * IoU threshold for non-maximum suppression. NudeNet's reference
     * implementation uses 0.45. @default 0.45
     */
    readonly iouThreshold?: number;
    /**
     * Optional class allowlist. When provided, regions whose class is not in
     * the list are dropped before NMS.
     */
    readonly classes?: ReadonlyArray<string>;
}

export type ExecutionProvider = "webgpu" | "wasm" | "webgl" | "cpu";

export interface LoadDetectorOptions {
    /**
     * URL to the NudeNet ONNX file (typically the 320n YOLOv8 detector).
     * For the playground we self-host at `/nudenet/320n.onnx`.
     */
    readonly modelUrl: string;
    /**
     * Square input resolution the model was exported at. NudeNet's 320n
     * uses 320; the larger 640m uses 640. @default 320
     */
    readonly inputSize?: number;
    /**
     * Ordered list of execution providers. ORT picks the first one that
     * loads. @default ["wasm"]
     */
    readonly executionProviders?: ReadonlyArray<ExecutionProvider>;
    /**
     * Override `ort.env.wasm.wasmPaths`. Set this once per page if you're
     * self-hosting the WASM artifacts. When omitted, the loader points ORT
     * at the version-pinned jsdelivr CDN.
     */
    readonly wasmPaths?: string;
}

export interface NsfwDetector {
    detect(
        input: DetectInput,
        options?: DetectOptions,
    ): Promise<DetectedRegion[]>;
    dispose(): void;
}
