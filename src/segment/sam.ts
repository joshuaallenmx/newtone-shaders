import {
    RawImage,
    Sam2Model,
    Sam2Processor,
    type Tensor,
} from "@huggingface/transformers";
import type {
    ImageSegmenter,
    LoadSegmenterOptions,
    SegmentMask,
    SegmentPoint,
    SegmentPointOptions,
    SegmentScale,
    SegmentSource,
} from "./types";

const DEFAULT_MODEL_ID = "onnx-community/sam2-hiera-tiny-ONNX";

interface ImageState {
    readonly raw: RawImage;
    readonly embeddings: Record<string, Tensor>;
    readonly originalSize: [number, number];
    readonly reshapedSize: [number, number];
}

async function pickDevice(
    requested: LoadSegmenterOptions["device"],
): Promise<"webgpu" | "wasm"> {
    if (requested === "wasm") return "wasm";
    if (requested === "webgpu") return "webgpu";
    // "auto" / undefined: try webgpu, fall back to wasm.
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

async function sourceToRawImage(source: SegmentSource): Promise<RawImage> {
    const w =
        source instanceof HTMLImageElement
            ? source.naturalWidth || source.width
            : source.width;
    const h =
        source instanceof HTMLImageElement
            ? source.naturalHeight || source.height
            : source.height;
    if (!w || !h) throw new Error("segment: source has zero size");

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
    if (!ctx) throw new Error("segment: 2D context unavailable");
    ctx.drawImage(source as CanvasImageSource, 0, 0);
    return RawImage.fromCanvas(canvas as OffscreenCanvas);
}

function argmax(arr: ArrayLike<number>): number {
    let best = -Infinity;
    let idx = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > best) {
            best = arr[i];
            idx = i;
        }
    }
    return idx;
}

/**
 * Pick a mask index from a multi-mask output by `scale`. For "best" we
 * trust SAM's IoU prediction; for size-based scales we count non-zero
 * pixels per mask plane and sort.
 */
function pickMaskIndex(
    maskData: Uint8Array,
    numMasks: number,
    planeSize: number,
    iouScores: Float32Array,
    scale: SegmentScale,
): number {
    if (scale === "best") return argmax(iouScores);
    const areas: { idx: number; area: number }[] = [];
    for (let m = 0; m < numMasks; m++) {
        const offset = m * planeSize;
        let area = 0;
        for (let i = 0; i < planeSize; i++) {
            if (maskData[offset + i]) area++;
        }
        areas.push({ idx: m, area });
    }
    areas.sort((a, b) => b.area - a.area); // largest first
    if (scale === "largest") return areas[0].idx;
    if (scale === "smallest") return areas[areas.length - 1].idx;
    // "medium" — pick the middle by area
    return areas[Math.floor(areas.length / 2)].idx;
}

function extractMask(
    src: Uint8Array,
    bestIdx: number,
    numMasks: number,
    planeSize: number,
): Uint8Array {
    const offset = (bestIdx % numMasks) * planeSize;
    const out = new Uint8Array(planeSize);
    for (let i = 0; i < planeSize; i++) {
        out[i] = src[offset + i] ? 255 : 0;
    }
    return out;
}

export async function loadSegmenter(
    options: LoadSegmenterOptions = {},
): Promise<ImageSegmenter> {
    const modelId = options.modelId ?? DEFAULT_MODEL_ID;
    const device = await pickDevice(options.device);
    const dtype =
        options.dtype ?? (device === "webgpu" ? "fp32" : "q8");

    const processor = (await Sam2Processor.from_pretrained(modelId, {
        progress_callback: options.onProgress as
            | ((event: unknown) => void)
            | undefined,
    })) as Sam2Processor;
    const model = (await Sam2Model.from_pretrained(modelId, {
        device,
        dtype,
        progress_callback: options.onProgress as
            | ((event: unknown) => void)
            | undefined,
    })) as Sam2Model;

    let state: ImageState | null = null;
    let disposed = false;

    const segmentPointsImpl = async (
        points: readonly SegmentPoint[],
        options: SegmentPointOptions = {},
    ): Promise<SegmentMask> => {
        if (disposed) throw new Error("segmenter disposed");
        if (!state) throw new Error("segment: call setImage first");
        if (points.length === 0) {
            throw new Error("segment: at least one point required");
        }
        const scale: SegmentScale = options.scale ?? "best";
        // Sam2Processor expects 4D input_points: [batch, point_batch, n_points, 2].
        const inputPoints = [
            [points.map((p) => [p.x, p.y])],
        ] as unknown as number[][][][];
        const inputLabels = [
            [points.map((p) => (p.positive ? 1 : 0))],
        ] as unknown as number[][][];
        const promptInputs = await processor(state.raw, {
            input_points: inputPoints,
            input_labels: inputLabels,
        });
        const outputs = await model({
            ...state.embeddings,
            input_points: promptInputs.input_points,
            input_labels: promptInputs.input_labels,
        });

        const masksList = await processor.post_process_masks(
            outputs.pred_masks,
            promptInputs.original_sizes,
            promptInputs.reshaped_input_sizes,
        );
        const maskTensor = masksList[0] as Tensor;
        const dims = maskTensor.dims as ReadonlyArray<number>;
        const [height, width] = [
            dims[dims.length - 2],
            dims[dims.length - 1],
        ];
        const numMasks = dims[dims.length - 3];
        const planeSize = height * width;

        const iouTensor = outputs.iou_scores as Tensor;
        const iouData = iouTensor.data as Float32Array;
        const src = maskTensor.data as Uint8Array;

        const idx = pickMaskIndex(
            src,
            numMasks,
            planeSize,
            iouData,
            scale,
        );
        const score = iouData[idx % iouData.length];
        const data = extractMask(src, idx, numMasks, planeSize);
        return { data, width, height, score };
    };

    return {
        async setImage(image: SegmentSource): Promise<void> {
            if (disposed) throw new Error("segmenter disposed");
            const raw = await sourceToRawImage(image);
            const inputs = await processor(raw);
            const embeddings = await model.get_image_embeddings({
                pixel_values: inputs.pixel_values,
            });
            state = {
                raw,
                embeddings,
                originalSize: inputs.original_sizes[0] as [number, number],
                reshapedSize: inputs.reshaped_input_sizes[0] as [
                    number,
                    number,
                ],
            };
        },

        async segmentBox(
            box: readonly [number, number, number, number],
        ): Promise<SegmentMask> {
            if (disposed) throw new Error("segmenter disposed");
            if (!state) throw new Error("segment: call setImage first");
            const [x, y, w, h] = box;
            const promptInputs = await processor(state.raw, {
                input_boxes: [[[x, y, x + w, y + h]]],
            });
            const outputs = await model({
                ...state.embeddings,
                input_boxes: promptInputs.input_boxes,
            });

            const masksList = await processor.post_process_masks(
                outputs.pred_masks,
                promptInputs.original_sizes,
                promptInputs.reshaped_input_sizes,
            );
            const maskTensor = masksList[0] as Tensor;
            const dims = maskTensor.dims as ReadonlyArray<number>;
            // Expected dims: [num_masks, H, W] or [batch, num_masks, H, W].
            const [height, width] = [dims[dims.length - 2], dims[dims.length - 1]];
            const numMasks = dims[dims.length - 3];
            const planeSize = height * width;

            const iouTensor = outputs.iou_scores as Tensor;
            const iouData = iouTensor.data as Float32Array;
            const best = argmax(iouData);
            const score = iouData[best];

            const data = extractMask(
                maskTensor.data as Uint8Array,
                best,
                numMasks,
                planeSize,
            );

            return { data, width, height, score };
        },

        segmentPoint(
            point: readonly [number, number],
            options: SegmentPointOptions = {},
        ): Promise<SegmentMask> {
            return segmentPointsImpl(
                [{ x: point[0], y: point[1], positive: true }],
                options,
            );
        },

        segmentPoints(
            points: readonly SegmentPoint[],
            options: SegmentPointOptions = {},
        ): Promise<SegmentMask> {
            return segmentPointsImpl(points, options);
        },

        dispose() {
            disposed = true;
            state = null;
        },
    };
}
