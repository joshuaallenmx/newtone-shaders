import type { CSSProperties } from "react";
import type {
    EditorEdgeLike,
    EditorNodeLike,
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

// Layers — variadic stacking node. Many incoming edges feed a single
// "in" handle; the rendering order plus per-layer opacity and blend
// mode live in `params.layers`. The compiler emits an aligned, healed
// copy of that list so the GPU pass receives one entry per active
// input slot.

export const LAYERS_MAX = 16;
export const LAYERS_ENTRY_ID = "layers";

export type LayerBlend = "normal" | "multiply" | "screen" | "add" | "overlay";

export interface LayerEntry {
    readonly src: string;
    readonly opacity: number;
    readonly blend: LayerBlend;
}

export interface LayersParams {
    readonly layers: readonly LayerEntry[];
}

const DEFAULT_PARAMS: LayersParams = {
    layers: [],
};

/** Read `params.layers`, falling back to legacy `params.order: string[]` so
 *  graphs saved before per-layer controls existed keep rendering. Missing /
 *  out-of-range fields are clamped to safe defaults. */
export function readLayers(raw: unknown): LayerEntry[] {
    if (!raw || typeof raw !== "object") return [];
    const obj = raw as { layers?: readonly unknown[]; order?: readonly unknown[] };
    if (Array.isArray(obj.layers)) {
        return obj.layers
            .map((e): LayerEntry | null => {
                if (!e || typeof e !== "object") return null;
                const r = e as Partial<LayerEntry>;
                if (typeof r.src !== "string") return null;
                return {
                    src: r.src,
                    opacity: clamp01(typeof r.opacity === "number" ? r.opacity : 1),
                    blend: normalizeBlend(r.blend),
                };
            })
            .filter((l): l is LayerEntry => l !== null);
    }
    if (Array.isArray(obj.order)) {
        return obj.order
            .filter((s): s is string => typeof s === "string")
            .map((src) => ({ src, opacity: 1, blend: "normal" as const }));
    }
    return [];
}

function clamp01(v: number): number {
    if (!Number.isFinite(v)) return 1;
    return Math.max(0, Math.min(1, v));
}

function normalizeBlend(v: unknown): LayerBlend {
    return v === "multiply" || v === "screen" || v === "add" || v === "overlay"
        ? v
        : "normal";
}

const BLEND_NORMAL = 0;
const BLEND_MULTIPLY = 1;
const BLEND_SCREEN = 2;
const BLEND_ADD = 3;
const BLEND_OVERLAY = 4;

function blendModeToInt(m: LayerBlend): number {
    switch (m) {
        case "multiply":
            return BLEND_MULTIPLY;
        case "screen":
            return BLEND_SCREEN;
        case "add":
            return BLEND_ADD;
        case "overlay":
            return BLEND_OVERLAY;
        default:
            return BLEND_NORMAL;
    }
}

// Sampler array indexing in WebGL2 GLSL ES 3.00 requires constant
// expressions, so we declare 16 named samplers and dispatch via an
// if-ladder. The loop bound MAX is fixed; the active count comes from
// `uLayerCount`.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

const int MAX_LAYERS = ${LAYERS_MAX};

uniform int uLayerCount;
uniform float uOpacity[MAX_LAYERS];
uniform int uBlend[MAX_LAYERS];
uniform sampler2D uLayer0;
uniform sampler2D uLayer1;
uniform sampler2D uLayer2;
uniform sampler2D uLayer3;
uniform sampler2D uLayer4;
uniform sampler2D uLayer5;
uniform sampler2D uLayer6;
uniform sampler2D uLayer7;
uniform sampler2D uLayer8;
uniform sampler2D uLayer9;
uniform sampler2D uLayer10;
uniform sampler2D uLayer11;
uniform sampler2D uLayer12;
uniform sampler2D uLayer13;
uniform sampler2D uLayer14;
uniform sampler2D uLayer15;

vec4 sampleLayer(int i, vec2 uv) {
    if (i ==  0) return texture(uLayer0,  uv);
    if (i ==  1) return texture(uLayer1,  uv);
    if (i ==  2) return texture(uLayer2,  uv);
    if (i ==  3) return texture(uLayer3,  uv);
    if (i ==  4) return texture(uLayer4,  uv);
    if (i ==  5) return texture(uLayer5,  uv);
    if (i ==  6) return texture(uLayer6,  uv);
    if (i ==  7) return texture(uLayer7,  uv);
    if (i ==  8) return texture(uLayer8,  uv);
    if (i ==  9) return texture(uLayer9,  uv);
    if (i == 10) return texture(uLayer10, uv);
    if (i == 11) return texture(uLayer11, uv);
    if (i == 12) return texture(uLayer12, uv);
    if (i == 13) return texture(uLayer13, uv);
    if (i == 14) return texture(uLayer14, uv);
    return texture(uLayer15, uv);
}

vec3 blendRgb(int mode, vec3 base, vec3 src) {
    if (mode == 1) return base * src;                              // multiply
    if (mode == 2) return 1.0 - (1.0 - base) * (1.0 - src);        // screen
    if (mode == 3) return min(vec3(1.0), base + src);              // add
    if (mode == 4) {
        // overlay: 2·base·src where base<0.5, else 1 − 2·(1−base)·(1−src)
        return mix(
            2.0 * base * src,
            1.0 - 2.0 * (1.0 - base) * (1.0 - src),
            step(0.5, base)
        );
    }
    return src;                                                    // normal
}

void main() {
    // Standard "src over dest" composite, bottom-to-top, with per-layer
    // opacity scaling src.a and per-layer blend mode applied to the
    // *unpremultiplied* base before re-compositing. When the accumulator
    // is empty (first layer or transparent below) the blend formula
    // would give a black halo, so we fall back to plain src.rgb in that
    // case — preserving the old shader's behavior at uOpacity=1,
    // uBlend=normal exactly.
    vec4 acc = vec4(0.0);
    for (int i = 0; i < MAX_LAYERS; i++) {
        if (i >= uLayerCount) break;
        vec4 src = sampleLayer(i, vUv);
        float a = src.a * clamp(uOpacity[i], 0.0, 1.0);
        int mode = uBlend[i];
        vec3 srcRgb = src.rgb;
        if (mode != 0 && acc.a > 0.0) {
            vec3 base = acc.rgb / acc.a;
            srcRgb = blendRgb(mode, base, src.rgb);
        }
        float oneMinusA = 1.0 - a;
        acc = vec4(
            srcRgb * a + acc.rgb * oneMinusA,
            a + acc.a * oneMinusA
        );
    }
    outColor = acc;
}
`;

const SAMPLERS: readonly string[] = Array.from(
    { length: LAYERS_MAX },
    (_, i) => `uLayer${i}`,
);

const layersGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: SAMPLERS,
    uniforms: ["uLayerCount", "uOpacity[0]", "uBlend[0]"],
    setUniforms: (gl, locs, params, frame) => {
        // The active count = how many of `inputsPresent` are true. The
        // compiler builds the chain so present slots come first, but
        // we double-check by counting `true`s here.
        let count = 0;
        for (const present of frame.inputsPresent) {
            if (present) count++;
        }
        gl.uniform1i(locs.get("uLayerCount")!, count);

        // The compiler emits `params.layers` aligned 1:1 with the active
        // input slots. Slots beyond `count` (bound to the placeholder
        // texture) get default values that won't disturb anything because
        // the loop short-circuits at `i >= uLayerCount`.
        const layers = (params as LayersParams | undefined)?.layers ?? [];
        const opacity = new Float32Array(LAYERS_MAX);
        const blend = new Int32Array(LAYERS_MAX);
        for (let i = 0; i < LAYERS_MAX; i++) {
            const layer = layers[i];
            opacity[i] = layer ? clamp01(layer.opacity) : 1;
            blend[i] = layer ? blendModeToInt(layer.blend) : BLEND_NORMAL;
        }
        gl.uniform1fv(locs.get("uOpacity[0]")!, opacity);
        gl.uniform1iv(locs.get("uBlend[0]")!, blend);
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const SECTION_LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 4,
    marginBottom: 8,
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 4,
    padding: "6px 8px",
    marginBottom: 6,
    fontSize: 12,
};

const ROW_HEADER_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
};

const ROW_CONTROLS_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
};

const LABEL_STYLE: CSSProperties = {
    flex: 1,
    color: "#f0f0f0",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
};

const ICON_BTN_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    width: 22,
    height: 22,
    fontSize: 11,
    fontFamily: "inherit",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
};

const HINT_STYLE: CSSProperties = {
    color: "#666",
    fontSize: 11,
    marginTop: 8,
    lineHeight: 1.5,
};

const SLIDER_STYLE: CSSProperties = {
    flex: 1,
    minWidth: 0,
    accentColor: "#7aa2ff",
};

const NUMERIC_STYLE: CSSProperties = {
    width: 40,
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 3,
    padding: "2px 4px",
    fontFamily: "inherit",
    fontSize: 11,
    fontVariantNumeric: "tabular-nums",
    textAlign: "right",
};

const SELECT_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 3,
    padding: "2px 4px",
    fontSize: 11,
    fontFamily: "inherit",
};

const BLEND_OPTIONS: ReadonlyArray<{ value: LayerBlend; label: string }> = [
    { value: "normal", label: "Normal" },
    { value: "multiply", label: "Multiply" },
    { value: "screen", label: "Screen" },
    { value: "add", label: "Add" },
    { value: "overlay", label: "Overlay" },
];

function nodeLabel(node: EditorNodeLike | undefined): string {
    if (!node) return "(missing)";
    if (node.type === "source") {
        const name = (node.data as { assetName?: string } | undefined)
            ?.assetName;
        return name ? `Source · ${name}` : "Source · (none)";
    }
    if (node.type === "shader") {
        const id = (node.data as { shaderId?: string } | undefined)
            ?.shaderId;
        return id ? `${id} · ${node.id}` : node.id;
    }
    if (node.type === "slot") return `Slot · ${node.id}`;
    return node.id;
}

function LayersControls({
    params,
    onChange,
    nodes,
    edges,
    nodeId,
}: ShaderControlsProps) {
    if (!nodes || !edges || !nodeId) {
        return (
            <div style={HINT_STYLE}>
                Layer ordering needs the editor's workspace context — open
                this from inside the editor to manage layer order.
            </div>
        );
    }

    const stored = readLayers(params);

    // Self-heal: drop entries whose edge no longer exists; append any new
    // upstream sources that appeared since the last render. Keeps the
    // displayed list aligned with reality even if onConnect/onEdgesChange
    // missed an update.
    const incoming = edges.filter((e) => e.target === nodeId);
    const incomingSources = new Set(incoming.map((e) => e.source));
    const seenSrc = new Set<string>();
    const layers: LayerEntry[] = [];
    for (const entry of stored) {
        if (incomingSources.has(entry.src) && !seenSrc.has(entry.src)) {
            layers.push(entry);
            seenSrc.add(entry.src);
        }
    }
    for (const e of incoming) {
        if (!seenSrc.has(e.source)) {
            layers.push({ src: e.source, opacity: 1, blend: "normal" });
            seenSrc.add(e.source);
        }
    }

    const commit = (next: LayerEntry[]) =>
        onChange({ layers: next } satisfies LayersParams);

    const move = (from: number, dir: -1 | 1) => {
        const to = from + dir;
        if (to < 0 || to >= layers.length) return;
        const next = layers.slice();
        const [m] = next.splice(from, 1);
        next.splice(to, 0, m!);
        commit(next);
    };

    const setOpacity = (i: number, value: number) => {
        const next = layers.slice();
        next[i] = { ...next[i]!, opacity: clamp01(value) };
        commit(next);
    };

    const setBlend = (i: number, value: LayerBlend) => {
        const next = layers.slice();
        next[i] = { ...next[i]!, blend: value };
        commit(next);
    };

    if (layers.length === 0) {
        return (
            <div style={HINT_STYLE}>
                Wire one or more upstream nodes into this Layers' input. Each
                connection becomes a layer; reorder them here. Rendering goes
                bottom-to-top — the last entry sits on top of the stack.
            </div>
        );
    }

    return (
        <div>
            <div style={SECTION_LABEL_STYLE}>
                layers (bottom → top)
            </div>
            {layers.map((layer, i) => {
                const node = nodes.find((n) => n.id === layer.src);
                const opacityPct = Math.round(layer.opacity * 100);
                return (
                    <div key={layer.src} style={ROW_STYLE}>
                        <div style={ROW_HEADER_STYLE}>
                            <span
                                style={{
                                    color: "#666",
                                    fontSize: 10,
                                    fontVariantNumeric: "tabular-nums",
                                    width: 18,
                                }}
                            >
                                {String(i + 1).padStart(2, "0")}
                            </span>
                            <span style={LABEL_STYLE} title={layer.src}>
                                {nodeLabel(node)}
                            </span>
                            <button
                                type="button"
                                style={ICON_BTN_STYLE}
                                onClick={() => move(i, -1)}
                                disabled={i === 0}
                                title="move down (toward bottom)"
                            >
                                ↓
                            </button>
                            <button
                                type="button"
                                style={ICON_BTN_STYLE}
                                onClick={() => move(i, 1)}
                                disabled={i === layers.length - 1}
                                title="move up (toward top)"
                            >
                                ↑
                            </button>
                        </div>
                        <div style={ROW_CONTROLS_STYLE}>
                            <span style={{ color: "#888", fontSize: 11, width: 50 }}>
                                opacity
                            </span>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.01}
                                value={layer.opacity}
                                onChange={(e) =>
                                    setOpacity(i, Number(e.target.value))
                                }
                                style={SLIDER_STYLE}
                            />
                            <input
                                type="number"
                                min={0}
                                max={100}
                                step={1}
                                value={opacityPct}
                                onChange={(e) =>
                                    setOpacity(i, Number(e.target.value) / 100)
                                }
                                style={NUMERIC_STYLE}
                            />
                        </div>
                        <div style={ROW_CONTROLS_STYLE}>
                            <span style={{ color: "#888", fontSize: 11, width: 50 }}>
                                blend
                            </span>
                            <select
                                value={layer.blend}
                                onChange={(e) =>
                                    setBlend(i, e.target.value as LayerBlend)
                                }
                                style={{ ...SELECT_STYLE, flex: 1 }}
                            >
                                {BLEND_OPTIONS.map((opt) => (
                                    <option key={opt.value} value={opt.value}>
                                        {opt.label}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                );
            })}
            <div style={HINT_STYLE}>
                Top of the list draws on top. Opacity scales the layer's
                contribution; blend mode operates on the accumulated stack
                below. Cap: {LAYERS_MAX} layers.
            </div>
        </div>
    );
}

export const layersEntry: ShaderEntry = {
    id: LAYERS_ENTRY_ID,
    name: "Layers (stack)",
    defaultParams: DEFAULT_PARAMS,
    Controls: LayersControls,
    inputs: [{ id: "in", label: "layers" }],
    variadic: true,
    gpu: layersGpuSpec,
};

// Re-exported for the editor types that don't want to drag in @xyflow.
export type { EditorEdgeLike, EditorNodeLike };
