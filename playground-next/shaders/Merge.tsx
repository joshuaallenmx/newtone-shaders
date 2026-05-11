import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

type BlendMode = "alpha" | "multiply" | "screen" | "min" | "max" | "mask";

interface MergeParams {
    readonly mode: BlendMode;
    /** Inverts the output alpha — what was opaque becomes transparent. */
    readonly invertAlpha: boolean;
}

const DEFAULT_PARAMS: MergeParams = {
    mode: "alpha",
    invertAlpha: false,
};

const BLEND_INDEX: Record<BlendMode, number> = {
    alpha: 0,
    multiply: 1,
    screen: 2,
    min: 3,
    max: 4,
    mask: 5,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uA;
uniform sampler2D uB;
uniform int uMode;
uniform float uInvertAlpha;

float lum(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec4 a = texture(uA, vUv);
    vec4 b = texture(uB, vUv);
    float la = lum(a.rgb);
    float lb = lum(b.rgb);

    if (uMode == 0) {
        // alpha: A as grayscale value, B as alpha mask.
        outColor = vec4(la, la, la, lb * b.a);
    } else if (uMode == 1) {
        // multiply
        float v = la * lb;
        outColor = vec4(v, v, v, max(a.a, b.a));
    } else if (uMode == 2) {
        // screen
        float v = 1.0 - (1.0 - la) * (1.0 - lb);
        outColor = vec4(v, v, v, max(a.a, b.a));
    } else if (uMode == 3) {
        // min (darken)
        float v = min(la, lb);
        outColor = vec4(v, v, v, max(a.a, b.a));
    } else if (uMode == 4) {
        // max (lighten)
        float v = max(la, lb);
        outColor = vec4(v, v, v, max(a.a, b.a));
    } else {
        // mask: pass A's full colour, write B's luminance × alpha into
        // the alpha channel. No blending — A is kept as-is, B simply
        // gates where it's visible downstream.
        outColor = vec4(a.rgb, a.a * lb * b.a);
    }

    // Optional alpha inversion — applied uniformly so it works for any
    // mode (most useful with alpha/mask, where the alpha channel is a
    // real mask).
    outColor.a = mix(outColor.a, 1.0 - outColor.a, uInvertAlpha);
}
`;

const mergeGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uA", "uB"],
    uniforms: ["uMode", "uInvertAlpha"],
    setUniforms: (gl, locs, params) => {
        const p: MergeParams = {
            ...DEFAULT_PARAMS,
            ...((params as Partial<MergeParams> | null) ?? {}),
        };
        gl.uniform1i(locs.get("uMode")!, BLEND_INDEX[p.mode]);
        gl.uniform1f(locs.get("uInvertAlpha")!, p.invertAlpha ? 1 : 0);
    },
};

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
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

function MergeControls({ params, onChange }: ShaderControlsProps) {
    const cur = (params ?? DEFAULT_PARAMS) as Partial<MergeParams>;
    const safe: MergeParams = {
        mode: (cur.mode ?? DEFAULT_PARAMS.mode) as BlendMode,
        invertAlpha: cur.invertAlpha ?? DEFAULT_PARAMS.invertAlpha,
    };
    return (
        <div>
            <div style={LABEL_STYLE}>blend mode</div>
            <select
                style={SELECT_STYLE}
                value={safe.mode}
                onChange={(e) =>
                    onChange({ ...safe, mode: e.target.value as BlendMode })
                }
            >
                <option value="alpha">A as value, B as alpha</option>
                <option value="multiply">multiply</option>
                <option value="screen">screen</option>
                <option value="min">min (darken)</option>
                <option value="max">max (lighten)</option>
                <option value="mask">mask only (A's color, B's luminance → alpha)</option>
            </select>
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
                    checked={safe.invertAlpha}
                    onChange={(e) =>
                        onChange({ ...safe, invertAlpha: e.target.checked })
                    }
                />
                invert alpha
            </label>
        </div>
    );
}

export const mergeEntry: ShaderEntry = {
    id: "merge",
    name: "Merge (2 inputs)",
    defaultParams: DEFAULT_PARAMS,
    Controls: MergeControls,
    inputs: [
        { id: "value", label: "value (rgb)" },
        { id: "alpha", label: "alpha (mask)" },
    ],
    gpu: mergeGpuSpec,
};
