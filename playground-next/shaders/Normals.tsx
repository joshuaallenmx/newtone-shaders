import { type CSSProperties } from "react";
import {
    AutoImageProcessor,
    AutoModelForNormalEstimation,
    RawImage,
    type Tensor,
} from "@huggingface/transformers";
import type { SegmenterProgress } from "@newtonedev/shaders";
import type {
    ProducerSpec,
    ShaderControlsProps,
    ShaderEntry,
    TextureUpload,
} from ".";
import { disposeTensors } from "./dispose";

type NormalsDtype = "fp16" | "q4f16" | "q8";

interface NormalsParams {
    readonly modelId: string;
    readonly dtype: NormalsDtype;
}

const DTYPES: readonly { readonly id: NormalsDtype; readonly label: string }[] =
    [
        { id: "fp16", label: "fp16 (best)" },
        { id: "q4f16", label: "q4f16 (lightest)" },
        { id: "q8", label: "q8" },
    ];

interface VariantOption {
    readonly id: string;
    readonly label: string;
    readonly modelId: string;
}

const VARIANTS: readonly VariantOption[] = [
    {
        id: "06b",
        label: "0.6B (best)",
        modelId: "onnx-community/sapiens-normal-0.6b",
    },
    {
        id: "03b",
        label: "0.3B (faster)",
        modelId: "onnx-community/sapiens-normal-0.3b",
    },
];

const DEFAULT_PARAMS: NormalsParams = {
    modelId: VARIANTS[0]!.modelId,
    dtype: "fp16",
};

interface NormalsState {
    readonly processor: unknown;
    readonly model: unknown;
}

// Cache by modelId+dtype since changing dtype requires a new ONNX session.
const statePromises = new Map<string, Promise<NormalsState>>();
const stateKey = (modelId: string, dtype: NormalsDtype) =>
    `${modelId}::${dtype}`;

async function pickDevice(): Promise<"webgpu" | "wasm"> {
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
        try {
            const adapter = await (
                navigator as Navigator & {
                    gpu: { requestAdapter: () => Promise<unknown> };
                }
            ).gpu.requestAdapter();
            if (adapter) return "webgpu";
        } catch {
            // fall through
        }
    }
    return "wasm";
}

function getState(
    modelId: string,
    dtype: NormalsDtype,
    onProgress: (event: SegmenterProgress) => void,
): Promise<NormalsState> {
    const key = stateKey(modelId, dtype);
    let p = statePromises.get(key);
    if (!p) {
        console.log("[normals] loading", key);
        p = (async () => {
            const device = await pickDevice();
            const cb = ((event: unknown) => {
                console.log("[normals] progress", event);
                onProgress(event as SegmenterProgress);
            }) as (event: unknown) => void;
            const processor = await AutoImageProcessor.from_pretrained(modelId, {
                progress_callback: cb,
            });
            const model = await AutoModelForNormalEstimation.from_pretrained(
                modelId,
                { device, dtype, progress_callback: cb },
            );
            console.log("[normals] ready", key);
            return { processor, model };
        })().catch((err: unknown) => {
            console.error("[normals] load failed", key, err);
            statePromises.delete(key);
            throw err;
        });
        statePromises.set(key, p);
    }
    return p;
}

interface ProcessorCallable {
    (image: RawImage): Promise<{ pixel_values: Tensor }>;
}

interface ModelCallable {
    (inputs: { pixel_values: Tensor }): Promise<Record<string, Tensor>>;
}

async function inferNormals(
    state: NormalsState,
    src: string,
): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }> {
    const image = await RawImage.fromURL(src);
    const inputs = await (state.processor as unknown as ProcessorCallable)(
        image,
    );
    const outputs = await (state.model as unknown as ModelCallable)({
        pixel_values: inputs.pixel_values,
    });

    try {
        return extractNormals(outputs);
    } finally {
        disposeTensors(inputs);
        disposeTensors(outputs);
    }
}

function extractNormals(
    outputs: Record<string, Tensor>,
): { rgba: Uint8ClampedArray; width: number; height: number } {
    // Pick the normal-map tensor: try common names, else first tensor.
    const candidate =
        outputs.predicted_normal ??
        outputs.normals ??
        outputs.logits ??
        Object.values(outputs)[0];
    if (!candidate) throw new Error("no normal tensor in model output");
    const tensor = candidate;
    const dims = tensor.dims as readonly number[];
    console.log("[normals] output dims", dims);

    // Expected dims: [1, 3, H, W]. Some models give [1, H, W, 3].
    let h: number;
    let w: number;
    let chFirst: boolean;
    if (dims.length === 4 && dims[1] === 3) {
        h = dims[2]!;
        w = dims[3]!;
        chFirst = true;
    } else if (dims.length === 4 && dims[3] === 3) {
        h = dims[1]!;
        w = dims[2]!;
        chFirst = false;
    } else {
        throw new Error(`unexpected normal tensor dims: ${dims.join("x")}`);
    }

    const data = tensor.data as Float32Array;
    const rgba = new Uint8ClampedArray(w * h * 4);
    const planeSize = w * h;

    // Sapiens normals are typically L2-normalized 3-vectors. We don't enforce
    // it here — clamp to [-1, 1] then map to [0, 255]. Y is flipped to match
    // the standard "Y up" normal-map convention (image Y points down).
    if (chFirst) {
        for (let p = 0; p < planeSize; p++) {
            const nx = data[p]!;
            const ny = data[planeSize + p]!;
            const nz = data[2 * planeSize + p]!;
            const o = p * 4;
            rgba[o] = Math.round((Math.max(-1, Math.min(1, nx)) + 1) * 127.5);
            rgba[o + 1] = Math.round(
                (Math.max(-1, Math.min(1, -ny)) + 1) * 127.5,
            );
            rgba[o + 2] = Math.round(
                (Math.max(-1, Math.min(1, nz)) + 1) * 127.5,
            );
            rgba[o + 3] = 255;
        }
    } else {
        for (let p = 0; p < planeSize; p++) {
            const base = p * 3;
            const nx = data[base]!;
            const ny = data[base + 1]!;
            const nz = data[base + 2]!;
            const o = p * 4;
            rgba[o] = Math.round((Math.max(-1, Math.min(1, nx)) + 1) * 127.5);
            rgba[o + 1] = Math.round(
                (Math.max(-1, Math.min(1, -ny)) + 1) * 127.5,
            );
            rgba[o + 2] = Math.round(
                (Math.max(-1, Math.min(1, nz)) + 1) * 127.5,
            );
            rgba[o + 3] = 255;
        }
    }
    return { rgba, width: w, height: h };
}

const CONTROL_LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
};

const CONTROL_SELECT_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 6px",
    fontSize: 12,
    fontFamily: "inherit",
    width: "100%",
    boxSizing: "border-box",
};


function NormalsControls({ params, onChange }: ShaderControlsProps) {
    const current = params as NormalsParams;
    return (
        <div>
            <div style={CONTROL_LABEL_STYLE}>variant</div>
            <select
                style={CONTROL_SELECT_STYLE}
                value={current.modelId}
                onChange={(e) =>
                    onChange({
                        ...current,
                        modelId: e.target.value,
                    } satisfies NormalsParams)
                }
            >
                {VARIANTS.map((v) => (
                    <option key={v.id} value={v.modelId}>
                        {v.label}
                    </option>
                ))}
            </select>
            <div style={CONTROL_LABEL_STYLE}>precision</div>
            <select
                style={CONTROL_SELECT_STYLE}
                value={current.dtype}
                onChange={(e) =>
                    onChange({
                        ...current,
                        dtype: e.target.value as NormalsDtype,
                    } satisfies NormalsParams)
                }
            >
                {DTYPES.map((d) => (
                    <option key={d.id} value={d.id}>
                        {d.label}
                    </option>
                ))}
            </select>
        </div>
    );
}

// ─── Pipeline-native producer ───────────────────────────────────────────
//
// Wraps the Sapiens normal-estimation pipeline as a producer. The model
// runs once per (src, modelId, dtype) tuple; subsequent frames re-use the
// uploaded texture. Output RGBA is painted into an offscreen canvas and
// uploaded as a `canvas` TextureUpload.

const normalsProducerSpec: ProducerSpec = {
    inputKey: (params, upstream) => {
        const p = (params as Partial<NormalsParams> | null) ?? {};
        const src = upstream[0]?.src ?? "";
        return `${src}|${p.modelId ?? DEFAULT_PARAMS.modelId}|${p.dtype ?? DEFAULT_PARAMS.dtype}`;
    },
    run: async (params, ctx) => {
        const p: NormalsParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<NormalsParams>),
        };
        const src = ctx.upstream[0]?.src;
        if (!src) throw new Error("normals producer: no source URL");
        const state = await getState(p.modelId, p.dtype, (event) => {
            ctx.onProgress?.(event);
        });
        if (ctx.signal.aborted) throw new Error("aborted");
        const { rgba, width, height } = await inferNormals(state, src);
        if (ctx.signal.aborted) throw new Error("aborted");
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const cctx = canvas.getContext("2d");
        if (!cctx) throw new Error("normals: 2d context unavailable");
        const imageData = cctx.createImageData(width, height);
        imageData.data.set(rgba);
        cctx.putImageData(imageData, 0, 0);
        const upload: TextureUpload = {
            source: { kind: "canvas", canvas },
            width,
            height,
        };
        return upload;
    },
};

export const normalsEntry: ShaderEntry = {
    id: "normals",
    name: "Normals (Sapiens)",
    defaultParams: DEFAULT_PARAMS,
    Controls: NormalsControls,
    producer: normalsProducerSpec,
};
