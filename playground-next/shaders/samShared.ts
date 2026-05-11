import { loadSegmenter } from "../../src/segment";
import type {
    ImageSegmenter,
    SegmenterProgress,
} from "../../src/segment/types";

// Module-scoped SAM 2 segmenter cache.
//
// Used by both `Sam.tsx` and `DescribeMask.tsx` (which runs CLIPSeg →
// SAM internally for high-res keyword masks). Sharing this cache means
// flipping a SAM node's variant or changing a Describe Mask label
// reuses any segmenter already loaded — one ViT in memory per
// variant, not one per node.
//
// `setImage` mutates the segmenter's internal embeddings, so per-
// variant we also serialize image swaps through a single in-flight
// promise. Concurrent calls against the same source reuse the cached
// encoder pass; against different sources the second call queues
// behind the first.

interface VariantState {
    segmenter: Promise<ImageSegmenter> | null;
    currentSrc: string | null;
    currentImage:
        | Promise<{ readonly width: number; readonly height: number }>
        | null;
}

const STATE_BY_MODEL = new Map<string, VariantState>();

function getVariantState(modelId: string): VariantState {
    let s = STATE_BY_MODEL.get(modelId);
    if (!s) {
        s = { segmenter: null, currentSrc: null, currentImage: null };
        STATE_BY_MODEL.set(modelId, s);
    }
    return s;
}

export function getSegmenter(
    modelId: string,
    onProgress: (event: SegmenterProgress) => void,
): Promise<ImageSegmenter> {
    const s = getVariantState(modelId);
    if (!s.segmenter) {
        s.segmenter = loadSegmenter({
            modelId,
            onProgress,
        }).catch((err: unknown) => {
            s.segmenter = null;
            throw err;
        });
    }
    return s.segmenter;
}

function loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = () =>
            reject(new Error(`sam: failed to load image: ${src}`));
        img.src = src;
    });
}

export function ensureImageSet(
    modelId: string,
    seg: ImageSegmenter,
    src: string,
    onProgress: (event: SegmenterProgress) => void,
): Promise<{ readonly width: number; readonly height: number }> {
    const s = getVariantState(modelId);
    if (s.currentSrc === src && s.currentImage) {
        return s.currentImage;
    }
    s.currentSrc = src;
    onProgress({ status: "encoding", file: src, progress: 0 });
    s.currentImage = (async () => {
        const img = await loadImage(src);
        await seg.setImage(img);
        onProgress({ status: "encoded", file: src, progress: 100 });
        return {
            width: img.naturalWidth || img.width,
            height: img.naturalHeight || img.height,
        };
    })().catch((err: unknown) => {
        if (s.currentSrc === src) {
            s.currentSrc = null;
            s.currentImage = null;
        }
        throw err;
    });
    return s.currentImage;
}

// SAM 2.1 family on onnx-community. Encoder is the cost driver:
// tiny ≈ 150MB, small ≈ 200MB, base+ ≈ 320MB. All variants emit
// masks at full source resolution from the lightweight decoder —
// variant choice trades download/encode time for embedding richness,
// which shows up most on cluttered scenes and fine boundary work.
export interface SamVariant {
    readonly id: string;
    readonly modelId: string;
    readonly label: string;
}

export const SAM_VARIANTS: ReadonlyArray<SamVariant> = [
    {
        id: "tiny",
        modelId: "onnx-community/sam2.1-hiera-tiny-ONNX",
        label: "Tiny (fastest)",
    },
    {
        id: "small",
        modelId: "onnx-community/sam2.1-hiera-small-ONNX",
        label: "Small (balanced)",
    },
    {
        id: "base-plus",
        modelId: "onnx-community/sam2.1-hiera-base-plus-ONNX",
        label: "Base+ (best)",
    },
];

export const DEFAULT_SAM_MODEL_ID = SAM_VARIANTS[1]!.modelId; // small
