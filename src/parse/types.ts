import type { BinaryMask } from "../pipeline";

/**
 * 18-class label set from `mattmdjaga/segformer_b2_clothes` — a SegFormer-B2
 * fine-tuned on the LIP human-parsing dataset. Order matches the model's
 * `id2label` map; index into this tuple == class id emitted by the model.
 */
export const HUMAN_PARSE_CLASSES = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
] as const;

export type HumanParseClass = (typeof HUMAN_PARSE_CLASSES)[number];

export type ParseSource =
    | HTMLImageElement
    | HTMLCanvasElement
    | ImageBitmap
    | OffscreenCanvas;

export interface ParseResult {
    /** Per-pixel class index in source-pixel space, length `width * height`. */
    readonly classMap: Uint8Array;
    readonly width: number;
    readonly height: number;
    /** Class indices that appear at least once in this image. */
    readonly presentClasses: ReadonlyArray<number>;
}

export type ParseDevice =
    | "auto"
    | "webgpu"
    | "wasm"
    | "cpu"
    | "webnn"
    | "webnn-gpu"
    | "webnn-cpu";

export type ParseDtype = "auto" | "fp32" | "fp16" | "q8" | "q4";

export interface ParseProgress {
    readonly status: string;
    readonly file?: string;
    readonly progress?: number;
    readonly loaded?: number;
    readonly total?: number;
}

export interface LoadHumanParserOptions {
    /**
     * HuggingFace model id. Defaults to `mattmdjaga/segformer_b2_clothes` —
     * SegFormer-B2 / LIP, 18 classes, ~100MB ONNX. Other compatible models
     * with the same `id2label` schema may also work via this loader.
     */
    readonly modelId?: string;
    /** Default: `"webgpu"` if available, else `"wasm"`. */
    readonly device?: ParseDevice;
    /** Default: `"fp32"` on WebGPU, `"q8"` on WASM. */
    readonly dtype?: ParseDtype;
    /**
     * Receives transformers.js download / init events. Useful for showing
     * a "downloading model… X%" spinner during the first run.
     */
    readonly onProgress?: (event: ParseProgress) => void;
}

export interface HumanParser {
    parse(image: ParseSource): Promise<ParseResult>;
    dispose(): void;
}

// Re-export `BinaryMask` so callers can type the output of `parseToMask`
// without reaching into `../pipeline`.
export type { BinaryMask };
