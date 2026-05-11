import type { CSSProperties } from "react";
import type {
    SegmentPoint,
    SegmentScale,
    SegmenterProgress,
} from "../../src/segment/types";
import type {
    ProducerSpec,
    ShaderControlsProps,
    ShaderEntry,
    TextureUpload,
} from ".";
import { PreviewPad } from "./PreviewPad";
import { findUpstreamId } from "./findUpstream";
import {
    DEFAULT_SAM_MODEL_ID,
    SAM_VARIANTS,
    ensureImageSet,
    getSegmenter,
} from "./samShared";

// SAM 2 — point-prompt segmentation as a producer node.
//
// Wire any image (Source or anything upstream) into the input. The
// node lazy-loads SAM 2's image encoder on first run, encodes the
// upstream source once, then re-runs the lightweight mask decoder
// each time you change the click-point or scale. Output is a binary
// mask: 1.0 inside the segmented region, 0.0 outside — pipe it into
// MaskMerge, ColorGrade, or wire it as the input to ContourFlow /
// Swarm to drive particles by SAM regions.
//
// The encoder is heavy (~150MB model, hundreds of ms even on WebGPU).
// We cache the loaded segmenter at module scope and the most-recent
// `setImage(src)` so multiple SAM nodes pointing at the same source
// share that work; switching to a different upstream re-encodes lazily.

const ENTRY_ID = "sam";

interface SamUv {
    readonly x: number;
    readonly y: number;
}

interface SamParams {
    /** Click point in vUv (bottom-up 0..1). Drag the pad to move it. */
    readonly point: SamUv;
    /** Which of SAM's three multi-mask outputs to return. */
    readonly scale: SegmentScale;
    /** When true, output is inverted (mask hole instead of mask fill). */
    readonly invert: boolean;
    /** SAM 2.1 model variant — see `VARIANTS`. */
    readonly modelId: string;
}

const DEFAULT_PARAMS: SamParams = {
    point: { x: 0.5, y: 0.5 },
    scale: "best",
    invert: false,
    modelId: DEFAULT_SAM_MODEL_ID,
};

function readSamParams(raw: unknown): SamParams {
    if (!raw || typeof raw !== "object") return DEFAULT_PARAMS;
    const r = raw as Partial<SamParams> & { point?: Partial<SamUv> };
    const knownVariant = SAM_VARIANTS.some((v) => v.modelId === r.modelId);
    return {
        point: {
            x: clamp01(r.point?.x ?? DEFAULT_PARAMS.point.x),
            y: clamp01(r.point?.y ?? DEFAULT_PARAMS.point.y),
        },
        scale:
            r.scale === "largest" ||
            r.scale === "medium" ||
            r.scale === "smallest" ||
            r.scale === "best"
                ? r.scale
                : DEFAULT_PARAMS.scale,
        invert: !!r.invert,
        modelId: knownVariant ? r.modelId! : DEFAULT_PARAMS.modelId,
    };
}

function clamp01(v: number): number {
    if (!Number.isFinite(v)) return 0.5;
    return Math.max(0, Math.min(1, v));
}

// ─── Producer ──────────────────────────────────────────────────────────

const samProducerSpec: ProducerSpec = {
    inputKey: (params, upstream) => {
        const p = readSamParams(params);
        const src = upstream[0]?.src ?? "";
        return `${src}|${p.modelId}|${p.point.x.toFixed(4)},${p.point.y.toFixed(4)}|${p.scale}|${p.invert ? 1 : 0}`;
    },
    run: async (params, ctx) => {
        const p = readSamParams(params);
        const src = ctx.upstream[0]?.src;
        if (!src) {
            throw new Error("sam: no upstream image source");
        }
        const onProgress = (event: SegmenterProgress) => {
            ctx.onProgress?.(event);
        };
        const seg = await getSegmenter(p.modelId, onProgress);
        if (ctx.signal.aborted) throw new Error("aborted");
        const dims = await ensureImageSet(p.modelId, seg, src, onProgress);
        if (ctx.signal.aborted) throw new Error("aborted");

        // SegmentPoint takes coords in source pixel space. Our point
        // lives in vUv (bottom-up); flip Y so the click on the canvas
        // lines up with the image row SAM sees.
        const clickPx: SegmentPoint = {
            x: p.point.x * dims.width,
            y: (1 - p.point.y) * dims.height,
            positive: true,
        };
        const mask = await seg.segmentPoints([clickPx], {
            scale: p.scale,
        });
        if (ctx.signal.aborted) throw new Error("aborted");

        // Optional invert. SAM returns 0/255; flip to 255/0.
        const data = p.invert
            ? new Uint8Array(mask.data.length)
            : (mask.data as Uint8Array);
        if (p.invert) {
            for (let i = 0; i < mask.data.length; i++) {
                data[i] = 255 - mask.data[i]!;
            }
        }

        const upload: TextureUpload = {
            source: {
                kind: "rawimage",
                image: {
                    data,
                    width: mask.width,
                    height: mask.height,
                    channels: 1,
                },
            },
            width: mask.width,
            height: mask.height,
        };
        return upload;
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
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

const CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

const HINT_STYLE: CSSProperties = {
    color: "#666",
    fontSize: 11,
    marginTop: 10,
    lineHeight: 1.5,
};

const SCALES: ReadonlyArray<{ readonly id: SegmentScale; readonly label: string }> = [
    { id: "best", label: "best (highest IoU)" },
    { id: "largest", label: "largest" },
    { id: "medium", label: "medium" },
    { id: "smallest", label: "smallest" },
];

function SamControls({
    params,
    onChange,
    nodes,
    edges,
    nodeId,
}: ShaderControlsProps) {
    const cur = readSamParams(params);
    const update = (patch: Partial<SamParams>) =>
        onChange({ ...cur, ...patch });
    const upstreamId = findUpstreamId(nodes, edges, nodeId);

    return (
        <div>
            <div style={LABEL_STYLE}>click point (drag the dot)</div>
            <PreviewPad
                value={cur.point}
                onChange={(c) => update({ point: c })}
                nodeId={upstreamId}
                dotColor="#7fc7ff"
            />
            <div style={LABEL_STYLE}>model variant</div>
            <select
                style={SELECT_STYLE}
                value={cur.modelId}
                onChange={(e) => update({ modelId: e.target.value })}
            >
                {SAM_VARIANTS.map((v) => (
                    <option key={v.id} value={v.modelId}>
                        {v.label}
                    </option>
                ))}
            </select>
            <div style={LABEL_STYLE}>scale</div>
            <select
                style={SELECT_STYLE}
                value={cur.scale}
                onChange={(e) =>
                    update({ scale: e.target.value as SegmentScale })
                }
            >
                {SCALES.map((s) => (
                    <option key={s.id} value={s.id}>
                        {s.label}
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
                Drag the dot on the pad to point at the region to
                segment. Each variant is downloaded and encoded once
                per source — switching variants spins up a fresh
                encoder side-by-side, so an A/B between tiny and
                base+ is one encode-per-variant rather than one
                encode-per-flip. Subsequent point moves only re-run
                the lightweight decoder. Use <code>scale</code> to
                pick between SAM&apos;s three multi-mask outputs
                (&ldquo;smallest&rdquo; is useful for hierarchical
                refinement, e.g. point at a breast and land on an
                areola).
            </div>
        </div>
    );
}

export const samEntry: ShaderEntry = {
    id: ENTRY_ID,
    name: "SAM 2 Mask",
    defaultParams: DEFAULT_PARAMS,
    Controls: SamControls,
    inputs: [{ id: "in", label: "image" }],
    producer: samProducerSpec,
};

export type { SamParams, SamUv };
