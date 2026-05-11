import type { CSSProperties } from "react";
import { pipeline, RawImage } from "@huggingface/transformers";
import type { SegmenterProgress } from "@newtonedev/shaders";
import type {
    GpuPassSpec,
    ProducerSpec,
    ShaderControlsProps,
    ShaderEntry,
    TextureUpload,
} from ".";
import { disposeTensors } from "./dispose";

interface DepthParams {
    readonly modelId: string;
    readonly invert: boolean;
    /** Input resolution fed to the model. 518 is the trained-on size. */
    readonly resolution: number;
}

// Model patch grid is 14×14, so any multiple of 14 is valid. Above ~1400
// quality degrades because the ViT was trained at 518.
const RESOLUTIONS: readonly number[] = [518, 700, 1036, 1400, 1750];

interface VariantOption {
    readonly id: string;
    readonly label: string;
    readonly modelId: string;
}

// v3 is shipped on the hub but transformers.js doesn't yet have a processor
// mapping for it (fails with "this.processor is not a function" at inference
// time). v2 is the documented working version.
const VARIANTS: readonly VariantOption[] = [
    {
        id: "large-v2",
        label: "Large (v2)",
        modelId: "onnx-community/depth-anything-v2-large",
    },
    {
        id: "base-v2",
        label: "Base (v2)",
        modelId: "onnx-community/depth-anything-v2-base",
    },
    {
        id: "small-v2",
        label: "Small (v2)",
        modelId: "onnx-community/depth-anything-v2-small",
    },
];

const DEFAULT_PARAMS: DepthParams = {
    modelId: VARIANTS[0]!.modelId,
    invert: false,
    resolution: 518,
};

interface DepthPipelineFn {
    (input: string | RawImage): Promise<{ depth: RawImage }>;
}

interface ProcessorSize {
    height: number;
    width: number;
}

interface DepthPipelineWithProcessor {
    processor?: {
        image_processor?: { size?: ProcessorSize };
        size?: ProcessorSize;
    };
}

function setInputResolution(dp: DepthPipelineFn, resolution: number): void {
    const proc = (dp as unknown as DepthPipelineWithProcessor).processor;
    const target: ProcessorSize = { height: resolution, width: resolution };
    if (proc?.image_processor && "size" in proc.image_processor) {
        proc.image_processor.size = target;
    } else if (proc && "size" in proc) {
        proc.size = target;
    } else {
        console.warn("[depth] could not locate processor.size to override");
    }
}

const pipelinePromises = new Map<string, Promise<DepthPipelineFn>>();

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

function getPipeline(
    modelId: string,
    onProgress: (event: SegmenterProgress) => void,
): Promise<DepthPipelineFn> {
    let p = pipelinePromises.get(modelId);
    if (!p) {
        p = (async () => {
            const device = await pickDevice();
            const dtype = device === "webgpu" ? "fp32" : "q8";
            const dp = (await pipeline("depth-estimation", modelId, {
                device,
                dtype,
                progress_callback: ((event: unknown) => {
                    onProgress(event as SegmenterProgress);
                }) as (event: unknown) => void,
            })) as unknown as DepthPipelineFn;
            return dp;
        })().catch((err: unknown) => {
            pipelinePromises.delete(modelId);
            throw err;
        });
        pipelinePromises.set(modelId, p);
    }
    return p;
}

// Re-exported helpers — used by other depth-consuming entries that wrap the
// same pipeline. Kept here so model loading is deduplicated.
export function getDepthPipeline(
    modelId: string,
    onProgress: (event: SegmenterProgress) => void,
): Promise<DepthPipelineFn> {
    return getPipeline(modelId, onProgress);
}

export function setDepthInputResolution(
    dp: DepthPipelineFn,
    resolution: number,
): void {
    setInputResolution(dp, resolution);
}

// ─── Producer + gpu specs ───────────────────────────────────────────────

const depthProducerSpec: ProducerSpec = {
    inputKey: (params, upstream) => {
        const p = (params as Partial<DepthParams> | null) ?? {};
        const src = upstream[0]?.src ?? "";
        return `${src}|${p.modelId ?? DEFAULT_PARAMS.modelId}|${p.resolution ?? DEFAULT_PARAMS.resolution}`;
    },
    run: async (params, ctx) => {
        const p: DepthParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<DepthParams>),
        };
        const src = ctx.upstream[0]?.src;
        if (!src) throw new Error("depth producer: no source URL");
        const dp = await getPipeline(p.modelId, (event) => {
            ctx.onProgress?.(event);
        });
        if (ctx.signal.aborted) throw new Error("aborted");
        setInputResolution(dp, p.resolution);
        const result = await dp(src);
        if (ctx.signal.aborted) {
            disposeTensors(result);
            throw new Error("aborted");
        }
        const depth = result.depth;
        const upload: TextureUpload = {
            source: {
                kind: "rawimage",
                image: {
                    data: depth.data as Uint8Array,
                    width: depth.width,
                    height: depth.height,
                    channels: depth.channels,
                },
            },
            width: depth.width,
            height: depth.height,
        };
        disposeTensors(result);
        return upload;
    },
};

const DEPTH_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uDepth;
uniform float uInvert;
void main() {
    float d = texture(uDepth, vUv).r;
    d = mix(d, 1.0 - d, uInvert);
    outColor = vec4(d, d, d, 1.0);
}
`;

const depthGpuSpec: GpuPassSpec = {
    fragSrc: DEPTH_FRAG_SRC,
    samplers: ["uDepth"],
    uniforms: ["uInvert"],
    setUniforms: (gl, locs, params) => {
        const p = (params as Partial<DepthParams> | null) ?? {};
        gl.uniform1f(locs.get("uInvert")!, p.invert ? 1 : 0);
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

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

const CONTROL_CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

function DepthControls({ params, onChange }: ShaderControlsProps) {
    const current: DepthParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<DepthParams>),
    };
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
                    } satisfies DepthParams)
                }
            >
                {VARIANTS.map((v) => (
                    <option key={v.id} value={v.modelId}>
                        {v.label}
                    </option>
                ))}
            </select>
            <div style={CONTROL_LABEL_STYLE}>resolution</div>
            <select
                style={CONTROL_SELECT_STYLE}
                value={current.resolution}
                onChange={(e) =>
                    onChange({
                        ...current,
                        resolution: parseInt(e.target.value, 10),
                    } satisfies DepthParams)
                }
            >
                {RESOLUTIONS.map((r) => (
                    <option key={r} value={r}>
                        {r}px{r === 518 ? " (native)" : ""}
                    </option>
                ))}
            </select>
            <label style={CONTROL_CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={current.invert}
                    onChange={(e) =>
                        onChange({
                            ...current,
                            invert: e.target.checked,
                        } satisfies DepthParams)
                    }
                />
                invert (close = dark)
            </label>
        </div>
    );
}

export const depthEntry: ShaderEntry = {
    id: "depth",
    name: "Depth (Anything v2)",
    defaultParams: DEFAULT_PARAMS,
    Controls: DepthControls,
    producer: depthProducerSpec,
    gpu: depthGpuSpec,
};
