import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface CompositeParams {
    /** Focus point in normalized image coords. 0,0 = top-left, 1,1 = bottom-right. */
    readonly focusU: number;
    readonly focusV: number;
    /** Max blur radius in source-pixel units. */
    readonly maxBlurRadius: number;
    /** Depth tolerance around the focus, 0..1. Pixels within ±focusRange of
     *  focusDepth are kept sharp. */
    readonly focusRange: number;
    /** Falloff exponent applied to the (out-of-focus) distance term. */
    readonly falloffPower: number;
    /** Sample count on the bokeh ring (clamped to 64 in shader). */
    readonly samples: number;
}

const DEFAULT_PARAMS: CompositeParams = {
    focusU: 0.5,
    focusV: 0.5,
    maxBlurRadius: 8,
    focusRange: 0.05,
    falloffPower: 2,
    samples: 16,
};

// ─── Pipeline-native gpu pass ───────────────────────────────────────────
//
// Variable-radius circular blur driven by depth distance from a focus
// point. Two inputs:
//   inputs[0]="image" → uSource (RGB color)
//   inputs[1]="depth" → uDepth  (grayscale, depth in .r)
//
// `focusU`/`focusV` are exposed in image-y-down terms (intuitive for the
// user) and converted to vUv-up before sampling the depth texture.

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;
uniform sampler2D uDepth;
uniform vec2 uFocusUv;
uniform float uMaxBlurRadius;
uniform float uFocusRange;
uniform float uFalloffPower;
uniform int uSamples;

const int MAX_SAMPLES = 64;

void main() {
    float depth = texture(uDepth, vUv).r;
    float focusDepth = texture(uDepth, uFocusUv).r;
    float diff = abs(depth - focusDepth);
    float beyond = max(0.0, diff - uFocusRange);
    float denom = max(1e-4, 1.0 - uFocusRange);
    float t = clamp(beyond / denom, 0.0, 1.0);
    float coc = pow(t, uFalloffPower) * uMaxBlurRadius;

    if (coc < 0.5) {
        outColor = texture(uSource, vUv);
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(uSource, 0));
    vec3 sum = texture(uSource, vUv).rgb;
    float weightSum = 1.0;

    int n = uSamples > MAX_SAMPLES ? MAX_SAMPLES : uSamples;
    if (n < 1) n = 1;
    float invN = 1.0 / float(n);

    // Two concentric rings (radius * 0.55 and radius * 1.0) for slightly
    // more uniform bokeh than a single ring.
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (i >= n) break;
        float angle = float(i) * invN * 6.2831853;
        vec2 dir = vec2(cos(angle), sin(angle));
        sum += texture(uSource, vUv + dir * coc * 0.55 * texel).rgb;
        sum += texture(uSource, vUv + dir * coc * texel).rgb;
        weightSum += 2.0;
    }

    outColor = vec4(sum / weightSum, 1.0);
}
`;

const compositeGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource", "uDepth"],
    uniforms: [
        "uFocusUv",
        "uMaxBlurRadius",
        "uFocusRange",
        "uFalloffPower",
        "uSamples",
    ],
    setUniforms: (gl, locs, params) => {
        const p: CompositeParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<CompositeParams>),
        };
        // Convert image-y-down focus → vUv-up for sampling.
        gl.uniform2f(
            locs.get("uFocusUv")!,
            clamp01(p.focusU),
            1 - clamp01(p.focusV),
        );
        gl.uniform1f(locs.get("uMaxBlurRadius")!, p.maxBlurRadius);
        gl.uniform1f(locs.get("uFocusRange")!, p.focusRange);
        gl.uniform1f(locs.get("uFalloffPower")!, p.falloffPower);
        gl.uniform1i(locs.get("uSamples")!, Math.round(p.samples));
    },
};

function clamp01(x: number): number {
    return Math.max(0, Math.min(1, x));
}

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
};

function CompositeControls({ params, onChange }: ShaderControlsProps) {
    const cur: CompositeParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<CompositeParams>),
    };
    const update = (patch: Partial<CompositeParams>) =>
        onChange({ ...cur, ...patch } satisfies CompositeParams);

    return (
        <div>
            <div style={LABEL_STYLE}>
                focus X (left → right): {cur.focusU.toFixed(2)}
            </div>
            <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={cur.focusU}
                onChange={(e) =>
                    update({ focusU: parseFloat(e.target.value) })
                }
                style={{ width: "100%" }}
            />
            <div style={LABEL_STYLE}>
                focus Y (top → bottom): {cur.focusV.toFixed(2)}
            </div>
            <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={cur.focusV}
                onChange={(e) =>
                    update({ focusV: parseFloat(e.target.value) })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>
                max blur radius: {cur.maxBlurRadius.toFixed(0)}px
            </div>
            <input
                type="range"
                min={0}
                max={64}
                step={1}
                value={cur.maxBlurRadius}
                onChange={(e) =>
                    update({
                        maxBlurRadius: parseFloat(e.target.value),
                    })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>
                focus range: {cur.focusRange.toFixed(3)}
            </div>
            <input
                type="range"
                min={0}
                max={0.5}
                step={0.005}
                value={cur.focusRange}
                onChange={(e) =>
                    update({ focusRange: parseFloat(e.target.value) })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>
                falloff power: {cur.falloffPower.toFixed(2)}
            </div>
            <input
                type="range"
                min={0.5}
                max={6}
                step={0.05}
                value={cur.falloffPower}
                onChange={(e) =>
                    update({
                        falloffPower: parseFloat(e.target.value),
                    })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>samples: {cur.samples}</div>
            <input
                type="range"
                min={1}
                max={64}
                step={1}
                value={cur.samples}
                onChange={(e) =>
                    update({ samples: parseInt(e.target.value, 10) })
                }
                style={{ width: "100%" }}
            />
        </div>
    );
}

export const compositeShaderEntry: ShaderEntry = {
    id: "composite-fx",
    name: "Composite (DoF)",
    defaultParams: DEFAULT_PARAMS,
    Controls: CompositeControls,
    inputs: [
        { id: "image", label: "image" },
        { id: "depth", label: "depth (grayscale)" },
    ],
    gpu: compositeGpuSpec,
};
