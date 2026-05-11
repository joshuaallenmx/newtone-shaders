import type { CSSProperties } from "react";
import {
    AutoProcessor,
    AutoTokenizer,
    CLIPSegForImageSegmentation,
    RawImage,
    type PreTrainedModel,
    type PreTrainedTokenizer,
    type Processor,
    type Tensor,
} from "@huggingface/transformers";
import type {
    ProducerSpec,
    ShaderControlsProps,
    ShaderEntry,
    TextureUpload,
} from ".";
import type { SegmenterProgress } from "../../src/segment/types";
import { PreviewPad } from "./PreviewPad";
import { findUpstreamId } from "./findUpstream";
import {
    DEFAULT_SAM_MODEL_ID,
    SAM_VARIANTS,
    ensureImageSet,
    getSegmenter,
} from "./samShared";

// Describe Mask — open-vocabulary segmentation, chain CLIPSeg → SAM 2.
//
// Each "region" is a (label, optional focus point) pair. For every
// region we obtain a seed point in source-pixel coordinates and feed
// it to SAM 2 — which gives a binary mask at full source resolution
// with crisp boundaries. The seed point comes from one of two paths:
//
//   • Focus point set → use the user-positioned dot directly. CLIPSeg
//     is skipped for that region (the user already knows where).
//   • No focus → run CLIPSeg with the label, take the argmax of the
//     soft mask (mapped back through the processor's center-crop into
//     source pixels), feed that to SAM as the seed.
//
// All per-region SAM masks are unioned via per-pixel OR, so combining
// unrelated objects ("couch + ball + lamp") into one mask works the
// same way it always did. CLIPSeg only handles "where roughly is the
// thing"; SAM does the actual pixel work — open-vocab keywords + SAM
// edge quality.

const ENTRY_ID = "describe-mask";
const CLIPSEG_MODEL_ID = "Xenova/clipseg-rd64-refined";
const MASK_SIDE = 352;
const MIN_LABEL_CONFIDENCE = 0.2;

interface RegionPoint {
    readonly x: number;
    readonly y: number;
}

interface DescribeRegion {
    /** Free-form descriptor: "ball", "couch", "the red cup". Empty
     *  labels with `useFocus = false` are skipped at run time;
     *  empty labels with a focus point still work (the point alone
     *  drives SAM, no CLIPSeg call needed). */
    readonly label: string;
    /** When true, skip CLIPSeg for this region and seed SAM with the
     *  user-positioned `focus` point. Useful when the keyword would
     *  match in several places and you want to disambiguate, or when
     *  you just want SAM directly without typing anything. */
    readonly useFocus: boolean;
    /** Focus point in vUv (bottom-up 0..1). Drag the pad to position. */
    readonly focus: RegionPoint;
}

interface DescribeParams {
    readonly regions: readonly DescribeRegion[];
    readonly invert: boolean;
    /** SAM 2.1 variant — see SAM_VARIANTS. */
    readonly samModelId: string;
}

const DEFAULT_REGION: DescribeRegion = {
    label: "person",
    useFocus: false,
    focus: { x: 0.5, y: 0.5 },
};

const DEFAULT_PARAMS: DescribeParams = {
    regions: [DEFAULT_REGION],
    invert: false,
    samModelId: DEFAULT_SAM_MODEL_ID,
};

// ─── CLIPSeg cache ──────────────────────────────────────────────────────

interface CLIPSegBundle {
    readonly tokenizer: PreTrainedTokenizer;
    readonly processor: Processor;
    readonly model: PreTrainedModel;
}

let clipSegBundlePromise: Promise<CLIPSegBundle> | null = null;
const clipSegPixelValuesBySource = new Map<string, Promise<Tensor>>();
const clipSegImageDimsBySource = new Map<
    string,
    { readonly width: number; readonly height: number }
>();

interface ProgressCallbackOptions {
    readonly progress_callback?: (event: unknown) => void;
}

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
            // fall through to wasm
        }
    }
    return "wasm";
}

function getClipSegBundle(
    onProgress: (event: SegmenterProgress) => void,
): Promise<CLIPSegBundle> {
    if (!clipSegBundlePromise) {
        clipSegBundlePromise = (async () => {
            const device = await pickDevice();
            const dtype = device === "webgpu" ? "fp32" : "q8";
            const progress_callback = ((event: unknown) => {
                onProgress(event as SegmenterProgress);
            }) as ProgressCallbackOptions["progress_callback"];
            const [tokenizer, processor, model] = await Promise.all([
                AutoTokenizer.from_pretrained(CLIPSEG_MODEL_ID, {
                    progress_callback,
                }),
                AutoProcessor.from_pretrained(CLIPSEG_MODEL_ID, {
                    progress_callback,
                }),
                CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_ID, {
                    device,
                    dtype,
                    progress_callback,
                } as Record<string, unknown>),
            ]);
            return {
                tokenizer: tokenizer as PreTrainedTokenizer,
                processor: processor as Processor,
                model: model as PreTrainedModel,
            };
        })().catch((err: unknown) => {
            clipSegBundlePromise = null;
            throw err;
        });
    }
    return clipSegBundlePromise;
}

function getClipSegPixelValues(
    processor: Processor,
    src: string,
    onProgress: (event: SegmenterProgress) => void,
): Promise<Tensor> {
    let p = clipSegPixelValuesBySource.get(src);
    if (!p) {
        onProgress({ status: "encoding image", file: src, progress: 0 });
        p = (async () => {
            const image = await RawImage.read(src);
            clipSegImageDimsBySource.set(src, {
                width: image.width,
                height: image.height,
            });
            const inputs = (await (
                processor as unknown as (
                    input: RawImage,
                ) => Promise<{ pixel_values: Tensor }>
            )(image)) as { pixel_values: Tensor };
            onProgress({ status: "encoded image", file: src, progress: 100 });
            return inputs.pixel_values;
        })().catch((err: unknown) => {
            clipSegPixelValuesBySource.delete(src);
            throw err;
        });
        clipSegPixelValuesBySource.set(src, p);
    }
    return p;
}

function clamp01(v: number): number {
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.min(1, v));
}

function readRegion(raw: unknown): DescribeRegion {
    if (!raw || typeof raw !== "object") return DEFAULT_REGION;
    const r = raw as Partial<DescribeRegion> & {
        focus?: Partial<RegionPoint>;
    };
    return {
        label: typeof r.label === "string" ? r.label : DEFAULT_REGION.label,
        useFocus: !!r.useFocus,
        focus: {
            x: clamp01(r.focus?.x ?? DEFAULT_REGION.focus.x),
            y: clamp01(r.focus?.y ?? DEFAULT_REGION.focus.y),
        },
    };
}

function readParams(raw: unknown): DescribeParams {
    if (!raw || typeof raw !== "object") return DEFAULT_PARAMS;
    const r = raw as Partial<DescribeParams> & {
        // Legacy shape: a single comma-separated `prompt: string` (and
        // earlier the `regions` shape carried `radius`; we ignore that
        // now since SAM doesn't need a soft Gaussian).
        prompt?: string;
    };
    let regions: DescribeRegion[];
    if (Array.isArray(r.regions) && r.regions.length > 0) {
        regions = r.regions.map(readRegion);
    } else if (typeof r.prompt === "string") {
        const phrases = r.prompt
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s.length > 0);
        regions =
            phrases.length > 0
                ? phrases.map((label) => ({ ...DEFAULT_REGION, label }))
                : [DEFAULT_REGION];
    } else {
        regions = [...DEFAULT_PARAMS.regions];
    }
    const knownVariant = SAM_VARIANTS.some((v) => v.modelId === r.samModelId);
    return {
        regions,
        invert: !!r.invert,
        samModelId: knownVariant ? r.samModelId! : DEFAULT_PARAMS.samModelId,
    };
}

function regionToKey(r: DescribeRegion): string {
    if (!r.useFocus) return `T:${r.label.trim()}`;
    return `F:${r.label.trim()}@${r.focus.x.toFixed(3)},${r.focus.y.toFixed(3)}`;
}

// ─── CLIPSeg coord mapping ──────────────────────────────────────────────
//
// CLIPSegImageProcessor resizes the image's shorter edge to 352, then
// center-crops a 352×352 square. To map a pixel (cx, cy) ∈ [0, 352)²
// of the logits back into source-pixel space we have to invert that:
// rebuild the resized canvas, place the crop, scale to source dims.

interface CLIPSegMapping {
    readonly scale: number;
    readonly offsetX: number;
    readonly offsetY: number;
}

function clipSegMapping(srcW: number, srcH: number): CLIPSegMapping {
    if (srcW >= srcH) {
        const scale = srcH / MASK_SIDE;
        // resized width in source units is srcW; resized height is srcH;
        // crop offset along x in resized-source units is (srcW - srcH) / 2.
        return { scale, offsetX: (srcW - srcH) / 2, offsetY: 0 };
    }
    const scale = srcW / MASK_SIDE;
    return { scale, offsetX: 0, offsetY: (srcH - srcW) / 2 };
}

function maskPixelToSource(
    cx: number,
    cy: number,
    mapping: CLIPSegMapping,
): { x: number; y: number } {
    return {
        x: cx * mapping.scale + mapping.offsetX,
        y: cy * mapping.scale + mapping.offsetY,
    };
}

// ─── Producer ──────────────────────────────────────────────────────────

const describeProducerSpec: ProducerSpec = {
    inputKey: (params, upstream) => {
        const p = readParams(params);
        const src = upstream[0]?.src ?? "";
        const regionKey = p.regions.map(regionToKey).join("|");
        return `${src}|${p.samModelId}|${regionKey}|${p.invert ? 1 : 0}`;
    },
    run: async (params, ctx) => {
        const p = readParams(params);
        const src = ctx.upstream[0]?.src;
        if (!src) throw new Error("describe-mask: no upstream image source");

        // Strip empty rows. A region with no label AND no focus point
        // contributes nothing, so it's skipped.
        const activeRegions = p.regions.filter(
            (r) => r.label.trim().length > 0 || r.useFocus,
        );

        const onProgress = (event: SegmenterProgress) => {
            ctx.onProgress?.(event);
        };

        // Always need SAM — load the segmenter and encode the source.
        const seg = await getSegmenter(p.samModelId, onProgress);
        if (ctx.signal.aborted) throw new Error("aborted");
        const samDims = await ensureImageSet(p.samModelId, seg, src, onProgress);
        if (ctx.signal.aborted) throw new Error("aborted");
        const sourceW = samDims.width;
        const sourceH = samDims.height;

        if (activeRegions.length === 0) {
            return blackUpload(sourceW, sourceH, p.invert);
        }

        // Group regions by whether they need CLIPSeg. A region without
        // a focus point requires CLIPSeg to find the seed; one with a
        // focus skips CLIPSeg entirely.
        const labelRegions: { region: DescribeRegion; index: number }[] = [];
        const focusRegions: { region: DescribeRegion; index: number }[] = [];
        activeRegions.forEach((region, index) => {
            if (region.useFocus) focusRegions.push({ region, index });
            else if (region.label.trim().length > 0)
                labelRegions.push({ region, index });
        });

        // Per-region seed points in source-pixel space.
        const seedPoints: Array<{
            readonly label: string;
            readonly x: number;
            readonly y: number;
            readonly confident: boolean;
        }> = [];

        if (labelRegions.length > 0) {
            const bundle = await getClipSegBundle(onProgress);
            if (ctx.signal.aborted) throw new Error("aborted");
            const pixelValues = await getClipSegPixelValues(
                bundle.processor,
                src,
                onProgress,
            );
            if (ctx.signal.aborted) throw new Error("aborted");

            const phrases = labelRegions.map((lr) => lr.region.label.trim());
            const textInputs = (
                bundle.tokenizer as unknown as (
                    input: string[],
                    opts: { padding: boolean; truncation: boolean },
                ) => { input_ids: Tensor; attention_mask: Tensor }
            )(phrases, { padding: true, truncation: true });

            onProgress({ status: "running CLIPSeg", progress: 0 });
            const outputs = (await (
                bundle.model as unknown as (
                    inputs: Record<string, Tensor>,
                ) => Promise<{ logits: Tensor }>
            )({
                ...textInputs,
                pixel_values: pixelValues,
            })) as { logits: Tensor };
            if (ctx.signal.aborted) throw new Error("aborted");

            const logits = outputs.logits;
            const dims = logits.dims as ReadonlyArray<number>;
            const W = dims[dims.length - 1]!;
            const H = dims[dims.length - 2]!;
            const N = dims.length >= 3 ? dims[dims.length - 3]! : 1;
            const planeSize = W * H;
            const data = logits.data as Float32Array;
            // The processor's center-crop frame of reference. Use the
            // CLIPSeg-loaded image dims if we have them (more accurate
            // for the case where SAM and CLIPSeg disagree on dims —
            // they shouldn't, but a saved image cache lookup might
            // race the SAM image load).
            const dimsForMapping =
                clipSegImageDimsBySource.get(src) ?? {
                    width: sourceW,
                    height: sourceH,
                };
            const mapping = clipSegMapping(
                dimsForMapping.width,
                dimsForMapping.height,
            );

            for (let n = 0; n < labelRegions.length; n++) {
                const planeOffset = (n < N ? n : 0) * planeSize;
                let bestVal = -Infinity;
                let bestIdx = 0;
                for (let i = 0; i < planeSize; i++) {
                    const v = data[planeOffset + i]!;
                    if (v > bestVal) {
                        bestVal = v;
                        bestIdx = i;
                    }
                }
                const cy = Math.floor(bestIdx / W);
                const cx = bestIdx - cy * W;
                const peakProb = 1 / (1 + Math.exp(-bestVal));
                const srcPx = maskPixelToSource(cx, cy, mapping);
                seedPoints.push({
                    label: phrases[n] ?? "",
                    x: clamp(srcPx.x, 0, dimsForMapping.width - 1),
                    y: clamp(srcPx.y, 0, dimsForMapping.height - 1),
                    confident: peakProb >= MIN_LABEL_CONFIDENCE,
                });
            }
        }

        for (const { region } of focusRegions) {
            // Focus is in vUv (bottom-up). SAM wants top-down source px.
            const x = clamp01(region.focus.x) * sourceW;
            const y = (1 - clamp01(region.focus.y)) * sourceH;
            seedPoints.push({
                label: region.label.trim(),
                x: clamp(x, 0, sourceW - 1),
                y: clamp(y, 0, sourceH - 1),
                confident: true,
            });
        }

        // Drop low-confidence CLIPSeg seeds — sometimes a label has no
        // good match in the image and the argmax lands on noise. The
        // user can always set a focus point to force the seed.
        const usableSeeds = seedPoints.filter((s) => s.confident);
        if (usableSeeds.length === 0) {
            return blackUpload(sourceW, sourceH, p.invert);
        }

        // Run SAM 2 once per seed, union the masks.
        const accum = new Uint8Array(sourceW * sourceH);
        for (let i = 0; i < usableSeeds.length; i++) {
            if (ctx.signal.aborted) throw new Error("aborted");
            const seed = usableSeeds[i]!;
            onProgress({
                status: `SAM ${i + 1}/${usableSeeds.length}: ${seed.label || "(focus)"}`,
                progress: Math.round((i / usableSeeds.length) * 100),
            });
            const mask = await seg.segmentPoint([seed.x, seed.y], {
                scale: "best",
            });
            // Defensive: if SAM's output dims differ from source dims
            // (post_process_masks scales back to original), trust the
            // returned dims — only union when shapes match. In practice
            // they always match because the segmenter passes
            // original_sizes through.
            if (
                mask.width === sourceW &&
                mask.height === sourceH
            ) {
                const md = mask.data;
                for (let j = 0; j < md.length; j++) {
                    if (md[j]) accum[j] = 255;
                }
            } else {
                // Fall back to nearest-neighbour rescale if SAM returned
                // a different size — should never happen but better to
                // produce something than throw.
                const md = mask.data;
                const sx = mask.width / sourceW;
                const sy = mask.height / sourceH;
                for (let y = 0; y < sourceH; y++) {
                    const my = Math.floor(y * sy);
                    for (let x = 0; x < sourceW; x++) {
                        const mx = Math.floor(x * sx);
                        if (md[my * mask.width + mx]) {
                            accum[y * sourceW + x] = 255;
                        }
                    }
                }
            }
        }

        let outBytes = accum;
        if (p.invert) {
            const flipped = new Uint8Array(accum.length);
            for (let i = 0; i < accum.length; i++) flipped[i] = 255 - accum[i]!;
            outBytes = flipped;
        }

        const upload: TextureUpload = {
            source: {
                kind: "rawimage",
                image: {
                    data: outBytes,
                    width: sourceW,
                    height: sourceH,
                    channels: 1,
                },
            },
            width: sourceW,
            height: sourceH,
        };
        return upload;
    },
};

function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
}

function blackUpload(width: number, height: number, invert: boolean): TextureUpload {
    const data = new Uint8Array(width * height);
    if (invert) data.fill(255);
    return {
        source: {
            kind: "rawimage",
            image: { data, width, height, channels: 1 },
        },
        width,
        height,
    };
}

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
};

const INPUT_STYLE: CSSProperties = {
    flex: 1,
    minWidth: 0,
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 6px",
    fontSize: 12,
    fontFamily: "inherit",
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 4,
};

const REGION_FRAME_STYLE: CSSProperties = {
    border: "1px solid #2a2a2a",
    borderRadius: 4,
    padding: "8px",
    marginTop: 6,
};

const SMALL_BUTTON_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 8px",
    fontSize: 11,
    fontFamily: "inherit",
    cursor: "pointer",
};

const ADD_BUTTON_STYLE: CSSProperties = {
    ...SMALL_BUTTON_STYLE,
    width: "100%",
    marginTop: 8,
    padding: "6px 8px",
};

const CHECK_INLINE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 4,
    fontSize: 11,
    color: "#bbb",
    cursor: "pointer",
};

const CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
    cursor: "pointer",
};

const HINT_STYLE: CSSProperties = {
    color: "#666",
    fontSize: 11,
    marginTop: 10,
    lineHeight: 1.5,
};

const SELECT_STYLE: CSSProperties = {
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

interface PromptRowProps {
    readonly index: number;
    readonly region: DescribeRegion;
    readonly upstreamId: string | null;
    readonly canRemove: boolean;
    readonly onChange: (next: DescribeRegion) => void;
    readonly onRemove: () => void;
}

function PromptRow({
    index,
    region,
    upstreamId,
    canRemove,
    onChange,
    onRemove,
}: PromptRowProps) {
    const update = (patch: Partial<DescribeRegion>) =>
        onChange({ ...region, ...patch });
    return (
        <div style={REGION_FRAME_STYLE}>
            <div style={ROW_STYLE}>
                <span
                    style={{
                        color: "#666",
                        fontSize: 11,
                        width: 18,
                        textAlign: "center",
                    }}
                    title={`region ${index + 1}`}
                >
                    {index + 1}
                </span>
                <input
                    type="text"
                    value={region.label}
                    placeholder="couch, ball, the red cup…"
                    onChange={(e) => update({ label: e.target.value })}
                    style={INPUT_STYLE}
                    spellCheck={false}
                />
                <button
                    type="button"
                    onClick={onRemove}
                    style={{
                        ...SMALL_BUTTON_STYLE,
                        opacity: canRemove ? 1 : 0.4,
                        cursor: canRemove ? "pointer" : "not-allowed",
                    }}
                    disabled={!canRemove}
                    title={
                        canRemove
                            ? "remove region"
                            : "at least one region required"
                    }
                >
                    ×
                </button>
            </div>
            <label style={CHECK_INLINE}>
                <input
                    type="checkbox"
                    checked={region.useFocus}
                    onChange={(e) => update({ useFocus: e.target.checked })}
                />
                use focus point (skip CLIPSeg, seed SAM directly)
            </label>
            {region.useFocus ? (
                <PreviewPad
                    value={region.focus}
                    onChange={(c) => update({ focus: c })}
                    nodeId={upstreamId}
                    dotColor="#7fc7ff"
                />
            ) : null}
        </div>
    );
}

function DescribeControls({
    params,
    onChange,
    nodes,
    edges,
    nodeId,
}: ShaderControlsProps) {
    const cur = readParams(params);
    const update = (patch: Partial<DescribeParams>) =>
        onChange({ ...cur, ...patch });
    const upstreamId = findUpstreamId(nodes, edges, nodeId);

    const updateRegion = (i: number, next: DescribeRegion) => {
        const regions = cur.regions.map((r, idx) => (idx === i ? next : r));
        update({ regions });
    };
    const addRegion = () => {
        update({ regions: [...cur.regions, { ...DEFAULT_REGION, label: "" }] });
    };
    const removeRegion = (i: number) => {
        if (cur.regions.length <= 1) return;
        update({ regions: cur.regions.filter((_, idx) => idx !== i) });
    };

    return (
        <div>
            <div style={LABEL_STYLE}>
                describe regions (each row is a separate object — they
                get unioned into a single mask)
            </div>
            {cur.regions.map((region, i) => (
                <PromptRow
                    key={i}
                    index={i}
                    region={region}
                    upstreamId={upstreamId}
                    canRemove={cur.regions.length > 1}
                    onChange={(next) => updateRegion(i, next)}
                    onRemove={() => removeRegion(i)}
                />
            ))}
            <button type="button" onClick={addRegion} style={ADD_BUTTON_STYLE}>
                + add region
            </button>

            <div style={LABEL_STYLE}>SAM 2 model variant</div>
            <select
                style={SELECT_STYLE}
                value={cur.samModelId}
                onChange={(e) => update({ samModelId: e.target.value })}
            >
                {SAM_VARIANTS.map((v) => (
                    <option key={v.id} value={v.modelId}>
                        {v.label}
                    </option>
                ))}
            </select>

            <label style={CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={cur.invert}
                    onChange={(e) => update({ invert: e.target.checked })}
                />
                invert mask
            </label>
            <div style={HINT_STYLE}>
                CLIPSeg locates each label (open vocabulary —
                &ldquo;couch&rdquo;, &ldquo;ball&rdquo;, anything) and
                hands its peak to SAM 2, which produces the actual
                mask at the source&apos;s full resolution. Toggle
                &ldquo;use focus point&rdquo; on a region to skip
                CLIPSeg and seed SAM directly with the dot — useful
                for disambiguating multiple matches or working without
                a label at all. Per-region SAM masks are unioned, so
                unrelated objects combine cleanly.
            </div>
        </div>
    );
}

export const describeMaskEntry: ShaderEntry = {
    id: ENTRY_ID,
    name: "Describe Mask",
    defaultParams: DEFAULT_PARAMS,
    Controls: DescribeControls,
    inputs: [{ id: "in", label: "image" }],
    producer: describeProducerSpec,
};

export type { DescribeParams, DescribeRegion };
