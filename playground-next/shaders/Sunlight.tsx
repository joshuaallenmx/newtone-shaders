import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface SunlightParams {
    readonly azimuth: number;
    readonly elevation: number;
    readonly ambient: number;
    readonly strength: number;
}

const DEFAULT_PARAMS: SunlightParams = {
    azimuth: 45,
    elevation: 45,
    ambient: 0.1,
    strength: 5,
};

// Per-fragment Sobel-derived normal dotted with a light direction.
// Compass-bearing azimuth: 0° = sun directly above the subject, 90° =
// right, 180° = below, 270° = left. Elevation 0° = grazing, 90° = frontal.

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;
uniform vec3 uLightDir;
uniform float uAmbient;
uniform float uStrength;

float lum(vec2 uv) {
    vec3 c = texture(uSource, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 t = 1.0 / vec2(textureSize(uSource, 0));

    // Source uploaded with UNPACK_FLIP_Y_WEBGL=true: image y-down corresponds
    // to higher vUv.y, so "top" rows are at +t.y.
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

    vec3 n = normalize(
        vec3(-dx * uStrength, -dy * uStrength, 1.0)
    );
    float diff = max(0.0, dot(n, uLightDir));
    float ambient = clamp(uAmbient, 0.0, 1.0);
    float shade = clamp(ambient + (1.0 - ambient) * diff, 0.0, 1.0);
    outColor = vec4(vec3(shade), 1.0);
}
`;

const sunlightGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: ["uLightDir", "uAmbient", "uStrength"],
    setUniforms: (gl, locs, params) => {
        const p: SunlightParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<SunlightParams>),
        };
        const azRad = (p.azimuth * Math.PI) / 180;
        const elRad = (p.elevation * Math.PI) / 180;
        const Lx = Math.sin(azRad) * Math.cos(elRad);
        // Negate y because compass 0° = top in image-y-down terms.
        const Ly = -Math.cos(azRad) * Math.cos(elRad);
        const Lz = Math.sin(elRad);
        gl.uniform3f(locs.get("uLightDir")!, Lx, Ly, Lz);
        gl.uniform1f(locs.get("uAmbient")!, p.ambient);
        gl.uniform1f(locs.get("uStrength")!, p.strength);
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

function SunlightControls({ params, onChange }: ShaderControlsProps) {
    const current: SunlightParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<SunlightParams>),
    };
    return (
        <div>
            <div style={CONTROL_LABEL_STYLE}>azimuth (°)</div>
            <NumInput
                value={current.azimuth}
                min={0}
                max={360}
                step={1}
                onChange={(n) => onChange({ ...current, azimuth: n })}
            />
            <div style={CONTROL_LABEL_STYLE}>elevation (°)</div>
            <NumInput
                value={current.elevation}
                min={0}
                max={90}
                step={1}
                onChange={(n) => onChange({ ...current, elevation: n })}
            />
            <div style={CONTROL_LABEL_STYLE}>ambient</div>
            <NumInput
                value={current.ambient}
                min={0}
                max={1}
                step={0.05}
                onChange={(n) => onChange({ ...current, ambient: n })}
            />
            <div style={CONTROL_LABEL_STYLE}>strength</div>
            <NumInput
                value={current.strength}
                min={1}
                max={20}
                step={0.5}
                onChange={(n) => onChange({ ...current, strength: n })}
            />
        </div>
    );
}

export const sunlightEntry: ShaderEntry = {
    id: "sunlight",
    name: "Sunlight (synthetic)",
    defaultParams: DEFAULT_PARAMS,
    Controls: SunlightControls,
    gpu: sunlightGpuSpec,
};
