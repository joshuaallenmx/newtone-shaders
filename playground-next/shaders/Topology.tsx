import { type CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface TopologyParams {
    /** Pre-blur sigma in pixels. 0 = off. Smooths luminance before
     *  gradient so noise/grain doesn't dominate the field. */
    readonly blur: number;
    /** Gamma curve on the normalized magnitude. >1 = punchier (only the
     *  strongest edges stay bright). <1 = lifts the mids. */
    readonly contrast: number;
    /** Floor — magnitudes below this fraction (0..1) of the peak are
     *  clamped to black. Useful for killing background noise. */
    readonly threshold: number;
    /** When true, edges are dark on a bright background. */
    readonly invert: boolean;
}

const DEFAULT_PARAMS: TopologyParams = {
    blur: 1,
    contrast: 1.4,
    threshold: 0.05,
    invert: false,
};

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
};

const RANGE_STYLE: CSSProperties = { width: "100%" };

const CHECK_ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

interface SliderProps {
    readonly label: string;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly value: number;
    readonly onChange: (v: number) => void;
}

function Slider({ label, min, max, step, value, onChange }: SliderProps) {
    return (
        <div>
            <div style={LABEL_STYLE}>
                {label}: {value}
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                style={RANGE_STYLE}
            />
        </div>
    );
}

function TopologyControls({ params, onChange }: ShaderControlsProps) {
    const cur = params as TopologyParams;
    const safe: TopologyParams = {
        blur: cur.blur ?? DEFAULT_PARAMS.blur,
        contrast: cur.contrast ?? DEFAULT_PARAMS.contrast,
        threshold: cur.threshold ?? DEFAULT_PARAMS.threshold,
        invert: cur.invert ?? DEFAULT_PARAMS.invert,
    };
    return (
        <div>
            <Slider
                label="blur (px)"
                min={0}
                max={8}
                step={0.5}
                value={safe.blur}
                onChange={(n) => onChange({ ...safe, blur: n })}
            />
            <Slider
                label="contrast"
                min={0.5}
                max={4}
                step={0.05}
                value={safe.contrast}
                onChange={(n) => onChange({ ...safe, contrast: n })}
            />
            <Slider
                label="threshold"
                min={0}
                max={0.5}
                step={0.01}
                value={safe.threshold}
                onChange={(n) => onChange({ ...safe, threshold: n })}
            />
            <label style={CHECK_ROW_STYLE}>
                <input
                    type="checkbox"
                    checked={safe.invert}
                    onChange={(e) =>
                        onChange({ ...safe, invert: e.target.checked })
                    }
                />
                invert (dark = edges)
            </label>
        </div>
    );
}

// ─── Pipeline-native gpu pass ───────────────────────────────────────────
//
// Sobel-magnitude edge density. The legacy CPU path normalized by the
// per-frame max magnitude — that requires a reduction pass which doesn't
// fit a single-shader pipeline. Here we use a fixed normalization factor
// (max possible Sobel magnitude on luminance ∈ [0,1]) and lean on the
// `contrast` and `threshold` params to do the same shaping.
//
// The legacy `blur` pre-pass is not implemented here. If you want pre-blur,
// chain through a dedicated blur shader upstream (none in the library yet).

const SOBEL_MAX = 5.65685; // sqrt(2 * 4^2) — bound for unit-luminance edges

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;
uniform float uNorm;
uniform float uContrast;
uniform float uThreshold;
uniform float uInvert;

float lum(vec2 uv) {
    vec3 c = texture(uSource, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 t = 1.0 / vec2(textureSize(uSource, 0));

    float tl = lum(vUv + vec2(-t.x,  t.y));
    float tc = lum(vUv + vec2( 0.0,  t.y));
    float tr = lum(vUv + vec2( t.x,  t.y));
    float ml = lum(vUv + vec2(-t.x,  0.0));
    float mr = lum(vUv + vec2( t.x,  0.0));
    float bl = lum(vUv + vec2(-t.x, -t.y));
    float bc = lum(vUv + vec2( 0.0, -t.y));
    float br = lum(vUv + vec2( t.x, -t.y));

    float dx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    float dy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

    float mag = length(vec2(dx, dy)) * uNorm; // 0..~1 typical

    float floor_ = clamp(uThreshold, 0.0, 1.0);
    float v;
    if (mag < floor_) {
        v = 0.0;
    } else {
        v = (mag - floor_) / max(1e-6, 1.0 - floor_);
    }
    v = pow(clamp(v, 0.0, 1.0), 1.0 / max(0.01, uContrast));
    v = mix(v, 1.0 - v, uInvert);

    outColor = vec4(vec3(v), 1.0);
}
`;

const topologyGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: ["uNorm", "uContrast", "uThreshold", "uInvert"],
    setUniforms: (gl, locs, params) => {
        const p: TopologyParams = {
            blur: DEFAULT_PARAMS.blur,
            contrast: DEFAULT_PARAMS.contrast,
            threshold: DEFAULT_PARAMS.threshold,
            invert: DEFAULT_PARAMS.invert,
            ...(params as Partial<TopologyParams>),
        };
        gl.uniform1f(locs.get("uNorm")!, 1 / SOBEL_MAX);
        gl.uniform1f(locs.get("uContrast")!, p.contrast);
        gl.uniform1f(locs.get("uThreshold")!, p.threshold);
        gl.uniform1f(locs.get("uInvert")!, p.invert ? 1 : 0);
    },
};

export const topologyEntry: ShaderEntry = {
    id: "topology",
    name: "Topology (edge density)",
    defaultParams: DEFAULT_PARAMS,
    Controls: TopologyControls,
    gpu: topologyGpuSpec,
};
