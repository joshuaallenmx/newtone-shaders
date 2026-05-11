import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface MaskMergeParams {
    /** Mask luminance at which the mix is exactly 50/50. */
    readonly threshold: number;
    /** Width of the transition band around the threshold (0 = hard edge). */
    readonly softness: number;
    /** When true, swap which channel each tone reveals. */
    readonly invert: boolean;
    /** Gamma applied to mask luminance before remapping — <1 lifts mids
     *  toward white, >1 pushes toward black. */
    readonly maskGamma: number;
}

const DEFAULT_PARAMS: MaskMergeParams = {
    threshold: 0.5,
    softness: 0.2,
    invert: false,
    maskGamma: 1,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uA;
uniform sampler2D uB;
uniform sampler2D uMask;

uniform float uThreshold;
uniform float uSoftness;
uniform float uInvert;
uniform float uMaskGamma;

const vec3 LUM_W = vec3(0.2126, 0.7152, 0.0722);

void main() {
    vec4 a = texture(uA, vUv);
    vec4 b = texture(uB, vUv);
    vec3 maskRgb = texture(uMask, vUv).rgb;

    float m = clamp(dot(maskRgb, LUM_W), 0.0, 1.0);
    m = pow(m, max(uMaskGamma, 1e-3));
    m = mix(m, 1.0 - m, uInvert);

    // Smoothstep across [threshold - soft/2, threshold + soft/2]. Soft=0
    // collapses to a hard step; we fall back to step() to avoid the
    // smoothstep degenerate case.
    float halfSoft = max(uSoftness, 0.0) * 0.5;
    float lo = clamp(uThreshold - halfSoft, 0.0, 1.0);
    float hi = clamp(uThreshold + halfSoft, 0.0, 1.0);
    float w = halfSoft < 1e-4
        ? step(uThreshold, m)
        : smoothstep(lo, hi, m);

    vec3 rgb = mix(b.rgb, a.rgb, w);
    float alpha = mix(b.a, a.a, w);
    outColor = vec4(rgb, alpha);
}
`;

const maskMergeGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uA", "uB", "uMask"],
    uniforms: ["uThreshold", "uSoftness", "uInvert", "uMaskGamma"],
    setUniforms: (gl, locs, params) => {
        const p: MaskMergeParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<MaskMergeParams>),
        };
        gl.uniform1f(locs.get("uThreshold")!, p.threshold);
        gl.uniform1f(locs.get("uSoftness")!, p.softness);
        gl.uniform1f(locs.get("uInvert")!, p.invert ? 1 : 0);
        gl.uniform1f(locs.get("uMaskGamma")!, p.maskGamma);
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 2,
    display: "flex",
    justifyContent: "space-between",
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
};

const CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    cursor: "pointer",
    fontFamily: "inherit",
};

interface SliderRowProps {
    readonly label: string;
    readonly value: number;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly defaultValue: number;
    readonly format?: (v: number) => string;
    readonly onChange: (v: number) => void;
}

function SliderRow({
    label,
    value,
    min,
    max,
    step,
    defaultValue,
    format,
    onChange,
}: SliderRowProps) {
    const display = format ? format(value) : value.toFixed(2);
    return (
        <>
            <div style={LABEL_STYLE}>
                <span>{label}</span>
                <span style={{ color: "#bdbdbd" }}>{display}</span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() => onChange(defaultValue)}
                    title="reset"
                >
                    ↺
                </button>
            </div>
        </>
    );
}

function MaskMergeControls({ params, onChange }: ShaderControlsProps) {
    const cur: MaskMergeParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<MaskMergeParams>),
    };
    const update = (patch: Partial<MaskMergeParams>) =>
        onChange({ ...cur, ...patch } satisfies MaskMergeParams);

    return (
        <div>
            <SliderRow
                label="threshold (white ↔ black pivot)"
                value={cur.threshold}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.threshold}
                onChange={(v) => update({ threshold: v })}
            />
            <SliderRow
                label="softness"
                value={cur.softness}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.softness}
                onChange={(v) => update({ softness: v })}
            />
            <SliderRow
                label="mask gamma"
                value={cur.maskGamma}
                min={0.2}
                max={3}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.maskGamma}
                onChange={(v) => update({ maskGamma: v })}
            />
            <label style={CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={cur.invert}
                    onChange={(e) => update({ invert: e.target.checked })}
                />
                invert mask (swap which tone reveals which channel)
            </label>
        </div>
    );
}

export const maskMergeEntry: ShaderEntry = {
    id: "maskMerge",
    name: "Mask Merge (white = A, black = B)",
    defaultParams: DEFAULT_PARAMS,
    Controls: MaskMergeControls,
    inputs: [
        { id: "white", label: "A · revealed by white" },
        { id: "black", label: "B · revealed by black" },
        { id: "mask", label: "mask (luminance)" },
    ],
    gpu: maskMergeGpuSpec,
};
