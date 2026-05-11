import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface DepthRampParams {
    /** Invert depth before sampling (close = high). */
    readonly invert: boolean;
}

const DEFAULT_PARAMS: DepthRampParams = {
    invert: false,
};

// ─── Pipeline-native gpu pass ───────────────────────────────────────────
//
// LUT lookup: depth value (0..1) → x-coordinate in the ramp texture →
// output color. The ramp's middle row is sampled, so wire a horizontal
// (→) gradient for the cleanest banding. Vertical / diagonal gradients
// still work but their full variation isn't captured by the mid-row.
//
// Inputs: handle id "depth" → uDepth sampler, "ramp" → uRamp sampler.

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uDepth;
uniform sampler2D uRamp;
uniform float uInvert;
void main() {
    float d = texture(uDepth, vUv).r;
    d = mix(d, 1.0 - d, uInvert);
    vec3 c = texture(uRamp, vec2(d, 0.5)).rgb;
    outColor = vec4(c, 1.0);
}
`;

const depthRampGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uDepth", "uRamp"],
    uniforms: ["uInvert"],
    setUniforms: (gl, locs, params) => {
        const p = (params as Partial<DepthRampParams> | null) ?? {};
        gl.uniform1f(locs.get("uInvert")!, p.invert ? 1 : 0);
    },
};

const CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

function DepthRampControls({ params, onChange }: ShaderControlsProps) {
    const cur: DepthRampParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<DepthRampParams>),
    };
    return (
        <div>
            <label style={CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={cur.invert}
                    onChange={(e) =>
                        onChange({
                            invert: e.target.checked,
                        } satisfies DepthRampParams)
                    }
                />
                invert depth (close = high)
            </label>
        </div>
    );
}

export const depthRampEntry: ShaderEntry = {
    id: "depth-ramp",
    name: "Depth + Ramp",
    defaultParams: DEFAULT_PARAMS,
    Controls: DepthRampControls,
    inputs: [
        { id: "depth", label: "depth (grayscale)" },
        { id: "ramp", label: "ramp (gradient)" },
    ],
    gpu: depthRampGpuSpec,
};
