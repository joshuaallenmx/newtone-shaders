import {
    RawImage,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    type Tensor,
} from "@huggingface/transformers";
import type {
    HumanParser,
    LoadHumanParserOptions,
    ParseResult,
    ParseSource,
} from "./types";
import { HUMAN_PARSE_CLASSES } from "./types";

const DEFAULT_MODEL_ID = "mattmdjaga/segformer_b2_clothes";

async function pickDevice(
    requested: LoadHumanParserOptions["device"],
): Promise<"webgpu" | "wasm"> {
    if (requested === "wasm") return "wasm";
    if (requested === "webgpu") return "webgpu";
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
        try {
            const adapter = await (
                navigator as Navigator & {
                    gpu: { requestAdapter: () => Promise<unknown> };
                }
            ).gpu.requestAdapter();
            if (adapter) return "webgpu";
        } catch {
            // fall through to wasm
        }
    }
    return "wasm";
}

async function sourceToRawImage(source: ParseSource): Promise<RawImage> {
    const w =
        source instanceof HTMLImageElement
            ? source.naturalWidth || source.width
            : source.width;
    const h =
        source instanceof HTMLImageElement
            ? source.naturalHeight || source.height
            : source.height;
    if (!w || !h) throw new Error("parse: source has zero size");

    const canvas =
        typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(w, h)
            : Object.assign(document.createElement("canvas"), {
                  width: w,
                  height: h,
              });
    const ctx = canvas.getContext("2d") as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D
        | null;
    if (!ctx) throw new Error("parse: 2D context unavailable");
    ctx.drawImage(source as CanvasImageSource, 0, 0);
    return RawImage.fromCanvas(canvas as OffscreenCanvas);
}

function classMapFromTensor(
    segmentation: Tensor,
    width: number,
    height: number,
): Uint8Array {
    const expected = width * height;
    const data = segmentation.data as
        | BigInt64Array
        | Int32Array
        | Float32Array;
    if (data.length !== expected) {
        throw new Error(
            `parse: segmentation length ${data.length} != ${expected} for ${width}x${height}`,
        );
    }
    const out = new Uint8Array(expected);
    if (data instanceof BigInt64Array) {
        for (let i = 0; i < expected; i++) out[i] = Number(data[i]) & 0xff;
    } else {
        for (let i = 0; i < expected; i++) out[i] = data[i] & 0xff;
    }
    return out;
}

function presentClassesFrom(classMap: Uint8Array): number[] {
    const seen = new Uint8Array(HUMAN_PARSE_CLASSES.length);
    for (let i = 0; i < classMap.length; i++) {
        const c = classMap[i];
        if (c < seen.length) seen[c] = 1;
    }
    const out: number[] = [];
    for (let c = 0; c < seen.length; c++) {
        if (seen[c]) out.push(c);
    }
    return out;
}

export async function loadHumanParser(
    options: LoadHumanParserOptions = {},
): Promise<HumanParser> {
    const modelId = options.modelId ?? DEFAULT_MODEL_ID;
    const device = await pickDevice(options.device);
    const dtype =
        options.dtype ?? (device === "webgpu" ? "fp32" : "q8");

    const processor = (await SegformerImageProcessor.from_pretrained(modelId, {
        progress_callback: options.onProgress as
            | ((event: unknown) => void)
            | undefined,
    })) as SegformerImageProcessor;
    const model = (await SegformerForSemanticSegmentation.from_pretrained(
        modelId,
        {
            device,
            dtype,
            progress_callback: options.onProgress as
                | ((event: unknown) => void)
                | undefined,
        },
    )) as SegformerForSemanticSegmentation;

    let disposed = false;

    return {
        async parse(image: ParseSource): Promise<ParseResult> {
            if (disposed) throw new Error("parser disposed");
            const raw = await sourceToRawImage(image);
            const inputs = await processor(raw);
            const outputs = await model(inputs);

            const [post] = processor.post_process_semantic_segmentation(
                outputs,
                [[raw.height, raw.width]],
            );
            const classMap = classMapFromTensor(
                post.segmentation,
                raw.width,
                raw.height,
            );
            const presentClasses =
                Array.isArray(post.labels) && post.labels.length > 0
                    ? [...post.labels].sort((a, b) => a - b)
                    : presentClassesFrom(classMap);

            return {
                classMap,
                width: raw.width,
                height: raw.height,
                presentClasses,
            };
        },
        dispose() {
            disposed = true;
        },
    };
}
