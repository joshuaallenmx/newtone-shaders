import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

// Tile the canvas with a square grid of circles. Each circle samples a
// small N×N grid of taps inside its cell and averages them, so the dot's
// colour is the local mean of the source rather than a single texel.
// The space between circles is the (editable) solid background.
//
// Aspect handling mirrors Particles: gridX cells horizontally, and
// gridY = gridX / aspect, which keeps cells (and therefore the dots)
// square in pixels regardless of canvas aspect ratio.

interface CircleGridParams {
    /** Cells per shorter axis. */
    readonly density: number;
    /** Circle radius in cell units. 1 = touching adjacent cells, >1 = overlapping. */
    readonly circleSize: number;
    /** Soft-edge width in cell units. */
    readonly softness: number;
    /** N along each axis: total taps per circle = N×N. */
    readonly sampleCount: number;
    /** Hex color of the gap fill. */
    readonly background: string;
    /** 0 = pass source through, 1 = full dot effect. */
    readonly mix: number;
}

const DEFAULT_PARAMS: CircleGridParams = {
    density: 60,
    circleSize: 0.92,
    softness: 0.04,
    sampleCount: 4,
    background: "#0a0a0a",
    mix: 1,
};

// Cap the per-circle tap count at compile time so the loop unrolls
// predictably across drivers; the sampleCount uniform indexes inside
// this fixed bound via an `if (i >= N) break;` guard.
const MAX_SAMPLE_AXIS = 8;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;

uniform float uDensity;
uniform float uAspect;
uniform float uCircleSize;
uniform float uSoftness;
uniform int   uSampleCount;
uniform vec3  uBackground;
uniform float uMix;

const int MAX_N = ${MAX_SAMPLE_AXIS};

void main() {
    vec2 grid = vec2(uDensity, max(1.0, uDensity / max(uAspect, 1e-3)));
    vec2 cellSize = 1.0 / grid;
    vec2 cellId = floor(vUv * grid);
    vec2 cellCenter = (cellId + 0.5) * cellSize;

    // Position in the cell, in cell units. Range is -0.5..0.5 on both
    // axes regardless of aspect, because we divided by cellSize.
    vec2 inCell = (vUv - cellCenter) / cellSize;
    float dist = length(inCell);
    float radius = uCircleSize * 0.5;
    float soft = max(uSoftness, 1e-4);

    vec3 src = texture(uSource, vUv).rgb;

    // Outside the soft edge — short-circuit to background. Saves the
    // multi-tap average for the >90% of pixels in the gaps.
    if (dist > radius + soft) {
        outColor = vec4(mix(src, uBackground, uMix), 1.0);
        return;
    }

    // Average source colour across an N×N grid of taps inside the cell.
    vec4 avg = vec4(0.0);
    float invN = 1.0 / float(uSampleCount);
    int taps = 0;
    for (int j = 0; j < MAX_N; j++) {
        if (j >= uSampleCount) break;
        for (int i = 0; i < MAX_N; i++) {
            if (i >= uSampleCount) break;
            vec2 t = (vec2(float(i), float(j)) + 0.5) * invN;
            vec2 sampleUv = cellCenter + (t - 0.5) * cellSize;
            sampleUv = clamp(sampleUv, vec2(0.0), vec2(1.0));
            avg += texture(uSource, sampleUv);
            taps++;
        }
    }
    avg /= max(float(taps), 1.0);

    float circleAlpha = 1.0 - smoothstep(radius - soft, radius, dist);
    vec3 dotted = mix(uBackground, avg.rgb, circleAlpha);
    outColor = vec4(mix(src, dotted, uMix), 1.0);
}
`;

interface RGB {
    readonly r: number;
    readonly g: number;
    readonly b: number;
}

function parseHexColor(hex: string): RGB {
    const cleaned = hex.replace(/^#/, "");
    if (cleaned.length === 3) {
        return {
            r: parseInt(cleaned[0]! + cleaned[0]!, 16),
            g: parseInt(cleaned[1]! + cleaned[1]!, 16),
            b: parseInt(cleaned[2]! + cleaned[2]!, 16),
        };
    }
    if (cleaned.length === 6) {
        return {
            r: parseInt(cleaned.slice(0, 2), 16),
            g: parseInt(cleaned.slice(2, 4), 16),
            b: parseInt(cleaned.slice(4, 6), 16),
        };
    }
    return { r: 10, g: 10, b: 10 };
}

const circleGridGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: [
        "uDensity",
        "uAspect",
        "uCircleSize",
        "uSoftness",
        "uSampleCount",
        "uBackground",
        "uMix",
    ],
    setUniforms: (gl, locs, params) => {
        const p: CircleGridParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<CircleGridParams>),
        };
        const drawW = gl.drawingBufferWidth;
        const drawH = gl.drawingBufferHeight;
        const aspect = drawW / Math.max(1, drawH);
        const bg = parseHexColor(p.background);
        const samples = Math.max(
            1,
            Math.min(MAX_SAMPLE_AXIS, Math.round(p.sampleCount)),
        );
        gl.uniform1f(locs.get("uDensity")!, p.density);
        gl.uniform1f(locs.get("uAspect")!, aspect);
        gl.uniform1f(locs.get("uCircleSize")!, p.circleSize);
        gl.uniform1f(locs.get("uSoftness")!, p.softness);
        gl.uniform1i(locs.get("uSampleCount")!, samples);
        gl.uniform3f(
            locs.get("uBackground")!,
            bg.r / 255,
            bg.g / 255,
            bg.b / 255,
        );
        gl.uniform1f(locs.get("uMix")!, p.mix);
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

const COLOR_INPUT_STYLE: CSSProperties = {
    width: 32,
    height: 22,
    padding: 0,
    border: "1px solid #333",
    borderRadius: 4,
    background: "transparent",
    cursor: "pointer",
    flexShrink: 0,
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

function CircleGridControls({ params, onChange }: ShaderControlsProps) {
    const cur: CircleGridParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<CircleGridParams>),
    };
    const update = (patch: Partial<CircleGridParams>) =>
        onChange({ ...cur, ...patch } satisfies CircleGridParams);

    return (
        <div>
            <div style={ROW_STYLE}>
                <button
                    type="button"
                    style={{ ...BUTTON_STYLE, marginLeft: "auto" }}
                    onClick={() => onChange(DEFAULT_PARAMS)}
                    title="reset all parameters to defaults"
                >
                    reset all
                </button>
            </div>

            <SliderRow
                label="density (cells/axis)"
                value={cur.density}
                min={4}
                max={300}
                step={1}
                defaultValue={DEFAULT_PARAMS.density}
                format={(v) => v.toFixed(0)}
                onChange={(v) => update({ density: v })}
            />
            <SliderRow
                label="circle size (1 = touching)"
                value={cur.circleSize}
                min={0.05}
                max={1.5}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.circleSize}
                onChange={(v) => update({ circleSize: v })}
            />
            <SliderRow
                label="edge softness"
                value={cur.softness}
                min={0}
                max={0.3}
                step={0.005}
                defaultValue={DEFAULT_PARAMS.softness}
                onChange={(v) => update({ softness: v })}
            />
            <SliderRow
                label={`sample count (${cur.sampleCount}×${cur.sampleCount} taps)`}
                value={cur.sampleCount}
                min={1}
                max={MAX_SAMPLE_AXIS}
                step={1}
                defaultValue={DEFAULT_PARAMS.sampleCount}
                format={(v) => v.toFixed(0)}
                onChange={(v) => update({ sampleCount: Math.round(v) })}
            />

            <div style={LABEL_STYLE}>
                <span>background</span>
                <span style={{ color: "#bdbdbd" }}>{cur.background}</span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="color"
                    style={COLOR_INPUT_STYLE}
                    value={cur.background}
                    onChange={(e) =>
                        update({ background: e.target.value })
                    }
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() =>
                        update({ background: DEFAULT_PARAMS.background })
                    }
                    title="reset"
                >
                    ↺
                </button>
            </div>

            <SliderRow
                label="mix (0 = source, 1 = dots)"
                value={cur.mix}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.mix}
                onChange={(v) => update({ mix: v })}
            />
        </div>
    );
}

export const circleGridEntry: ShaderEntry = {
    id: "circleGrid",
    name: "Circle Grid (averaged dots)",
    defaultParams: DEFAULT_PARAMS,
    Controls: CircleGridControls,
    inputs: [{ id: "in", label: "image" }],
    gpu: circleGridGpuSpec,
};
