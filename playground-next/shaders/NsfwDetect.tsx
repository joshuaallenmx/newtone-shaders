import type { CSSProperties } from "react";
import {
    NUDENET_CLASSES,
    type NudeNetClass,
} from "../../src/detect";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

// NSFW Detect — runs the NudeNet macro-localization model on whatever's
// wired into its single input and overlays bounding boxes for the
// selected classes. The fragment shader is a pure passthrough of the
// input; all the work happens in the editor's `NsfwDetectOverlay`,
// which detects this entry id at the chain root, captures the node's
// outputTex via `captureNodeImageData(... , "fit")`, runs the ONNX
// detector, and renders DOM boxes positioned relative to the canvas
// content area.

export const NSFW_DETECT_ENTRY_ID = "nsfwDetect";

export interface NsfwDetectParams {
    /** Class allowlist — boxes are drawn only for classes in this set.
     *  An empty array means "show none". */
    readonly classes: readonly NudeNetClass[];
    /** Confidence threshold below which detections are dropped. The
     *  detector applies its own pre-NMS threshold; this filter runs on
     *  the post-NMS results so the slider feels live. */
    readonly minScore: number;
    /** Show the class label + score above each box. */
    readonly showLabels: boolean;
}

const EXPOSED_CLASSES: NudeNetClass[] = [
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
    "FEET_EXPOSED",
];

const COVERED_CLASSES: NudeNetClass[] = [
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "ANUS_COVERED",
    "BELLY_COVERED",
    "ARMPITS_COVERED",
    "FEET_COVERED",
];

const FACE_CLASSES: NudeNetClass[] = ["FACE_FEMALE", "FACE_MALE"];

export const DEFAULT_NSFW_DETECT_PARAMS: NsfwDetectParams = {
    classes: EXPOSED_CLASSES,
    minScore: 0.3,
    showLabels: true,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
void main() {
    outColor = texture(uSrc, vUv);
}
`;

const nsfwDetectGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSrc"],
    uniforms: [],
    setUniforms: () => {
        // No uniforms — pure passthrough.
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const SECTION_LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginTop: 12,
    marginBottom: 6,
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 4,
    fontSize: 12,
    color: "#bdbdbd",
};

const QUICK_BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 8px",
    fontSize: 11,
    cursor: "pointer",
    fontFamily: "inherit",
};

const SLIDER_LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 8,
    marginBottom: 2,
    display: "flex",
    justifyContent: "space-between",
};

function NsfwDetectControls({ params, onChange }: ShaderControlsProps) {
    const cur: NsfwDetectParams = {
        ...DEFAULT_NSFW_DETECT_PARAMS,
        ...(params as Partial<NsfwDetectParams>),
    };
    const selected = new Set<NudeNetClass>(cur.classes);

    const set = (next: Set<NudeNetClass>) =>
        onChange({
            ...cur,
            classes: Array.from(next),
        } satisfies NsfwDetectParams);

    const toggle = (c: NudeNetClass) => {
        const next = new Set(selected);
        if (next.has(c)) next.delete(c);
        else next.add(c);
        set(next);
    };

    const setAll = (cs: readonly NudeNetClass[]) =>
        set(new Set(cs));

    return (
        <div>
            <div style={SECTION_LABEL_STYLE}>quick selection</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                <button
                    type="button"
                    style={QUICK_BUTTON_STYLE}
                    onClick={() => setAll(NUDENET_CLASSES as unknown as NudeNetClass[])}
                >
                    all
                </button>
                <button
                    type="button"
                    style={QUICK_BUTTON_STYLE}
                    onClick={() => setAll([])}
                >
                    none
                </button>
                <button
                    type="button"
                    style={QUICK_BUTTON_STYLE}
                    onClick={() => setAll(EXPOSED_CLASSES)}
                >
                    exposed
                </button>
                <button
                    type="button"
                    style={QUICK_BUTTON_STYLE}
                    onClick={() => setAll(COVERED_CLASSES)}
                >
                    covered
                </button>
                <button
                    type="button"
                    style={QUICK_BUTTON_STYLE}
                    onClick={() => setAll(FACE_CLASSES)}
                >
                    faces
                </button>
            </div>

            <div style={SECTION_LABEL_STYLE}>exposed</div>
            {EXPOSED_CLASSES.map((c) => (
                <ClassRow
                    key={c}
                    label={c}
                    checked={selected.has(c)}
                    onToggle={() => toggle(c)}
                />
            ))}

            <div style={SECTION_LABEL_STYLE}>covered</div>
            {COVERED_CLASSES.map((c) => (
                <ClassRow
                    key={c}
                    label={c}
                    checked={selected.has(c)}
                    onToggle={() => toggle(c)}
                />
            ))}

            <div style={SECTION_LABEL_STYLE}>faces</div>
            {FACE_CLASSES.map((c) => (
                <ClassRow
                    key={c}
                    label={c}
                    checked={selected.has(c)}
                    onToggle={() => toggle(c)}
                />
            ))}

            <div style={SLIDER_LABEL_STYLE}>
                <span>min score</span>
                <span style={{ color: "#bdbdbd" }}>
                    {cur.minScore.toFixed(2)}
                </span>
            </div>
            <input
                type="range"
                min={0.05}
                max={1}
                step={0.01}
                value={cur.minScore}
                onChange={(e) =>
                    onChange({
                        ...cur,
                        minScore: parseFloat(e.target.value),
                    } satisfies NsfwDetectParams)
                }
                style={{ width: "100%" }}
            />

            <label
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    marginTop: 8,
                    fontSize: 12,
                    color: "#bdbdbd",
                }}
            >
                <input
                    type="checkbox"
                    checked={cur.showLabels}
                    onChange={(e) =>
                        onChange({
                            ...cur,
                            showLabels: e.target.checked,
                        } satisfies NsfwDetectParams)
                    }
                />
                show class labels
            </label>
        </div>
    );
}

function ClassRow({
    label,
    checked,
    onToggle,
}: {
    label: string;
    checked: boolean;
    onToggle: () => void;
}) {
    return (
        <label style={ROW_STYLE}>
            <input
                type="checkbox"
                checked={checked}
                onChange={onToggle}
            />
            <span
                style={{
                    fontFamily:
                        "ui-monospace, SFMono-Regular, Menlo, monospace",
                    fontSize: 11,
                }}
            >
                {label}
            </span>
        </label>
    );
}

export const nsfwDetectEntry: ShaderEntry = {
    id: NSFW_DETECT_ENTRY_ID,
    name: "NSFW Detect (bounding boxes)",
    defaultParams: DEFAULT_NSFW_DETECT_PARAMS,
    Controls: NsfwDetectControls,
    inputs: [{ id: "in", label: "image" }],
    gpu: nsfwDetectGpuSpec,
};
