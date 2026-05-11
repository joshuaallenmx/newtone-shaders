import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface IlluminationParams {
    readonly brightness: number;
    readonly contrast: number;
}

const DEFAULT_PARAMS: IlluminationParams = { brightness: 1, contrast: 1 };

// Same effect as the previous Canvas2D path: grayscale + brightness +
// contrast. CSS `brightness(b)` filter multiplies; `contrast(c)` scales
// around 0.5. Both are linear, perfect for a fragment shader.

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSource;
uniform float uBrightness;
uniform float uContrast;
void main() {
    vec3 c = texture(uSource, vUv).rgb;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    vec3 g = vec3(lum) * uBrightness;
    g = (g - 0.5) * uContrast + 0.5;
    outColor = vec4(clamp(g, 0.0, 1.0), 1.0);
}
`;

const illuminationGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: ["uBrightness", "uContrast"],
    setUniforms: (gl, locs, params) => {
        const p: IlluminationParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<IlluminationParams>),
        };
        gl.uniform1f(locs.get("uBrightness")!, p.brightness);
        gl.uniform1f(locs.get("uContrast")!, p.contrast);
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

function NumInput(props: {
    readonly value: number;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly onChange: (n: number) => void;
}) {
    return (
        <input
            type="number"
            min={props.min}
            max={props.max}
            step={props.step}
            value={props.value}
            style={CONTROL_INPUT_STYLE}
            onChange={(e) => {
                const n = parseFloat(e.target.value);
                if (Number.isFinite(n)) props.onChange(n);
            }}
        />
    );
}

function IlluminationControls({ params, onChange }: ShaderControlsProps) {
    const current: IlluminationParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<IlluminationParams>),
    };
    return (
        <div>
            <div style={CONTROL_LABEL_STYLE}>brightness</div>
            <NumInput
                value={current.brightness}
                min={0.5}
                max={2}
                step={0.05}
                onChange={(n) => onChange({ ...current, brightness: n })}
            />
            <div style={CONTROL_LABEL_STYLE}>contrast</div>
            <NumInput
                value={current.contrast}
                min={0.5}
                max={3}
                step={0.05}
                onChange={(n) => onChange({ ...current, contrast: n })}
            />
        </div>
    );
}

export const illuminationEntry: ShaderEntry = {
    id: "illumination",
    name: "Sunlight (extracted)",
    defaultParams: DEFAULT_PARAMS,
    Controls: IlluminationControls,
    gpu: illuminationGpuSpec,
};
