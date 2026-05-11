import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface BumpParams {
    readonly strength: number;
    readonly invert: boolean;
}

const DEFAULT_PARAMS: BumpParams = { strength: 3, invert: false };

// Sobel + tangent-space normal assembly. Per-fragment Sobel on luminance,
// then build a normal vector with z=1 anchor; strength controls relief
// depth. Y is mirrored in the output to match the standard "Y up"
// normal-map convention.

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;
uniform float uStrength;
uniform float uSign;          // 1 or -1 (invert)

float lum(vec2 uv) {
    vec3 c = texture(uSource, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 t = 1.0 / vec2(textureSize(uSource, 0));

    // Source uploaded with UNPACK_FLIP_Y_WEBGL=true: image y-down
    // corresponds to higher vUv.y, so "top" rows are at +t.y.
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

    float nx = uSign * -dx * uStrength;
    float ny = uSign * -dy * uStrength;
    vec3 n = normalize(vec3(nx, ny, 1.0));
    outColor = vec4(
        (n.x + 1.0) * 0.5,
        (-n.y + 1.0) * 0.5,
        (n.z + 1.0) * 0.5,
        1.0
    );
}
`;

const bumpNormalsGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: ["uStrength", "uSign"],
    setUniforms: (gl, locs, params) => {
        const p: BumpParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<BumpParams>),
        };
        gl.uniform1f(locs.get("uStrength")!, p.strength);
        gl.uniform1f(locs.get("uSign")!, p.invert ? -1 : 1);
    },
};

const CONTROL_LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
};

const CONTROL_INPUT_STYLE: CSSProperties = {
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

function BumpControls({ params, onChange }: ShaderControlsProps) {
    const current: BumpParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<BumpParams>),
    };
    return (
        <div>
            <div style={CONTROL_LABEL_STYLE}>strength</div>
            <input
                type="number"
                min={0.1}
                max={20}
                step={0.1}
                value={current.strength}
                style={CONTROL_INPUT_STYLE}
                onChange={(e) => {
                    const n = parseFloat(e.target.value);
                    if (Number.isFinite(n))
                        onChange({ ...current, strength: n });
                }}
            />
            <label style={CONTROL_CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={current.invert}
                    onChange={(e) =>
                        onChange({ ...current, invert: e.target.checked })
                    }
                />
                invert
            </label>
        </div>
    );
}

export const bumpNormalsEntry: ShaderEntry = {
    id: "bump-normals",
    name: "Normals (gradient)",
    defaultParams: DEFAULT_PARAMS,
    Controls: BumpControls,
    gpu: bumpNormalsGpuSpec,
};
