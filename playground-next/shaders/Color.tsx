import type { CSSProperties } from "react";
import type { GpuPassSpec, ShaderControlsProps, ShaderEntry } from ".";

interface ColorParams {
    /** Hex color (`#rrggbb` or `#rgb`). */
    readonly color: string;
    /** Output alpha, 0..1. */
    readonly alpha: number;
}

const DEFAULT_PARAMS: ColorParams = {
    color: "#ffffff",
    alpha: 1,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform vec3 uColor;
uniform float uAlpha;
void main() {
    outColor = vec4(uColor, uAlpha);
}
`;

function parseHexColor(hex: string): { r: number; g: number; b: number } {
    const cleaned = hex.startsWith("#") ? hex.slice(1) : hex;
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
    return { r: 0, g: 0, b: 0 };
}

const colorGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: [],
    uniforms: ["uColor", "uAlpha"],
    setUniforms: (gl, locs, params) => {
        const p: ColorParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<ColorParams>),
        };
        const { r, g, b } = parseHexColor(p.color);
        gl.uniform3f(locs.get("uColor")!, r / 255, g / 255, b / 255);
        gl.uniform1f(locs.get("uAlpha")!, Math.max(0, Math.min(1, p.alpha)));
    },
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 6,
};

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    width: 50,
};

const NUMBER_INPUT_STYLE: CSSProperties = {
    flex: 1,
    minWidth: 0,
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
    fontFamily: "inherit",
    fontVariantNumeric: "tabular-nums",
};

const COLOR_INPUT_STYLE: CSSProperties = {
    flex: "0 0 36px",
    height: 24,
    padding: 0,
    background: "transparent",
    border: "1px solid #333",
    borderRadius: 4,
    cursor: "pointer",
};

function ColorControls({ params, onChange }: ShaderControlsProps) {
    const p: ColorParams = {
        ...DEFAULT_PARAMS,
        ...((params ?? {}) as Partial<ColorParams>),
    };
    const set = (patch: Partial<ColorParams>) => onChange({ ...p, ...patch });
    return (
        <>
            <div style={ROW_STYLE}>
                <span style={LABEL_STYLE}>color</span>
                <input
                    type="color"
                    value={p.color}
                    onChange={(e) => set({ color: e.target.value })}
                    style={COLOR_INPUT_STYLE}
                />
                <input
                    type="text"
                    value={p.color}
                    onChange={(e) => set({ color: e.target.value })}
                    style={NUMBER_INPUT_STYLE}
                />
            </div>
            <div style={ROW_STYLE}>
                <span style={LABEL_STYLE}>alpha</span>
                <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.001}
                    value={p.alpha}
                    onChange={(e) => set({ alpha: Number(e.target.value) })}
                    style={{ flex: 1, minWidth: 0, accentColor: "#7aa2ff" }}
                />
                <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={Number(p.alpha.toFixed(3))}
                    onChange={(e) => set({ alpha: Number(e.target.value) })}
                    style={{ ...NUMBER_INPUT_STYLE, flex: "0 0 64px" }}
                />
            </div>
        </>
    );
}

export const colorEntry: ShaderEntry = {
    id: "color",
    name: "Color",
    defaultParams: DEFAULT_PARAMS,
    Controls: ColorControls,
    inputs: [],
    gpu: colorGpuSpec,
};
