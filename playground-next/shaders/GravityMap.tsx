import type { CSSProperties } from "react";
import type { GpuPassSpec, ShaderControlsProps, ShaderEntry } from ".";
import { PreviewPad } from "./PreviewPad";

// GravityMap — a topology-aware "pull" field over a normal-mapped surface.
//
// For each pixel, we compute the straight-line distance to the gravity
// center and modulate it by the local surface tilt: where the normal map
// says the surface tilts so that "moving toward the center" is uphill,
// the effective distance grows (less pull); where it tilts downhill
// toward the center, distance shrinks (more pull). Output is a scalar
// field in 0..1 written to RGB — feed it into Layers, ColorGrade, or a
// downstream physics shader (e.g. Swarm).
//
// The center can either follow the live pointer ("pointer" mode, good
// for standalone visualization) or sit at a fixed param-controlled
// position you drag in the inspector ("fixed" mode, what you want when
// composing with Swarm — Swarm uses the pointer for repulsion, so the
// gravity peak needs its own fixed position to not fight with it).
//
// Single-pass, first-order: catches local surface bias along the line
// to center but doesn't path-trace around ridges. Real geodesic distance
// would need an iterative Eikonal solver via the renderOverride hatch.

export type CenterMode = "pointer" | "fixed";

interface GravityMapParams {
    /** Where the gravity center sits. "pointer" makes it follow the
     *  live cursor; "fixed" pins it to `center` (drag the pad to move). */
    readonly centerMode: CenterMode;
    /** Center position in vUv coords (bottom-up 0..1). Used when
     *  `centerMode === "fixed"`. */
    readonly center: { readonly x: number; readonly y: number };
    /** Falloff radius in UV units (0..~1.5). Pixels outside read 0. */
    readonly radius: number;
    /** Power curve applied to the final intensity. >1 sharpens the
     *  pull near the center; <1 widens the soft falloff. */
    readonly falloff: number;
    /** How much the normal map distorts the metric. 0 = pure radial
     *  distance, 1 = full topology bias, negative inverts. */
    readonly topo: number;
    /** Most authoring tools (and our BumpNormals shader) flip Y on
     *  encode. Toggle if your input map uses the opposite convention. */
    readonly flipY: boolean;
    /** Invert the output (black at center, white outside). */
    readonly invert: boolean;
}

const DEFAULT_PARAMS: GravityMapParams = {
    centerMode: "fixed",
    center: { x: 0.5, y: 0.5 },
    radius: 0.5,
    falloff: 1.5,
    topo: 1.0,
    flipY: true,
    invert: false,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uNormals;
uniform vec2 uCenter;
uniform float uRadius;
uniform float uFalloff;
uniform float uTopo;
uniform int uFlipY;
uniform int uInvert;

void main() {
    // Decode RGB → [-1, 1]. Most authoring pipelines (and our
    // BumpNormals shader) bake in a Y flip on encode, so undo it when
    // uFlipY is set so the slope vector points "uphill" in the same
    // sense as the surface visually rises.
    vec3 nRaw = texture(uNormals, vUv).rgb * 2.0 - 1.0;
    vec2 slope = vec2(nRaw.x, uFlipY == 1 ? -nRaw.y : nRaw.y);

    vec2 d = uCenter - vUv;
    float r = length(d);
    vec2 dir = r > 1e-6 ? d / r : vec2(0.0);

    // climb > 0  →  surface uphill points toward the center, so the
    //               path from this pixel to the center is uphill →
    //               effective distance grows, pull weakens.
    // climb < 0  →  uphill points away, the path is downhill →
    //               effective distance shrinks, pull strengthens.
    float climb = dot(slope, dir);
    float effective = r * (1.0 + uTopo * climb);

    float radius = max(uRadius, 1e-6);
    float intensity = clamp(1.0 - effective / radius, 0.0, 1.0);
    intensity = pow(intensity, max(uFalloff, 0.001));

    if (uInvert == 1) intensity = 1.0 - intensity;

    outColor = vec4(vec3(intensity), 1.0);
}
`;

const gravityMapGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uNormals"],
    uniforms: [
        "uCenter",
        "uRadius",
        "uFalloff",
        "uTopo",
        "uFlipY",
        "uInvert",
    ],
    setUniforms: (gl, locs, params, frame) => {
        const p: GravityMapParams = readParams(params);
        const cx =
            p.centerMode === "fixed" ? p.center.x : frame.pointer.uv[0];
        const cy =
            p.centerMode === "fixed" ? p.center.y : frame.pointer.uv[1];
        gl.uniform2f(locs.get("uCenter")!, cx, cy);
        gl.uniform1f(locs.get("uRadius")!, p.radius);
        gl.uniform1f(locs.get("uFalloff")!, p.falloff);
        gl.uniform1f(locs.get("uTopo")!, p.topo);
        gl.uniform1i(locs.get("uFlipY")!, p.flipY ? 1 : 0);
        gl.uniform1i(locs.get("uInvert")!, p.invert ? 1 : 0);
    },
};

function readParams(raw: unknown): GravityMapParams {
    const r = raw as Partial<GravityMapParams> | undefined;
    return {
        ...DEFAULT_PARAMS,
        ...(r ?? {}),
        center: { ...DEFAULT_PARAMS.center, ...(r?.center ?? {}) },
        centerMode:
            r?.centerMode === "pointer" || r?.centerMode === "fixed"
                ? r.centerMode
                : DEFAULT_PARAMS.centerMode,
    };
}

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 4,
};

const SLIDER_STYLE: CSSProperties = {
    flex: 1,
    minWidth: 0,
    accentColor: "#7aa2ff",
};

const NUM_INPUT_STYLE: CSSProperties = {
    flex: "0 0 64px",
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 11,
    fontFamily: "inherit",
    fontVariantNumeric: "tabular-nums",
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

const TOGGLE_LABEL_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    color: "#bbb",
    fontSize: 12,
    marginTop: 8,
    cursor: "pointer",
};

const HINT_STYLE: CSSProperties = {
    color: "#666",
    fontSize: 11,
    marginTop: 10,
    lineHeight: 1.5,
};

interface SliderProps {
    readonly value: number;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly onChange: (n: number) => void;
}

function Slider({ value, min, max, step, onChange }: SliderProps) {
    return (
        <div style={ROW_STYLE}>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                style={SLIDER_STYLE}
            />
            <input
                type="number"
                min={min}
                max={max}
                step={step}
                value={Number(value.toFixed(3))}
                onChange={(e) => {
                    const n = Number(e.target.value);
                    if (Number.isFinite(n)) onChange(n);
                }}
                style={NUM_INPUT_STYLE}
            />
        </div>
    );
}

function GravityMapControls({
    params,
    onChange,
    nodeId,
}: ShaderControlsProps) {
    const cur = readParams(params);
    const update = (patch: Partial<GravityMapParams>) =>
        onChange({ ...cur, ...patch });
    return (
        <div>
            <div style={LABEL_STYLE}>center mode</div>
            <select
                style={SELECT_STYLE}
                value={cur.centerMode}
                onChange={(e) =>
                    update({ centerMode: e.target.value as CenterMode })
                }
            >
                <option value="fixed">fixed (drag below)</option>
                <option value="pointer">follow pointer</option>
            </select>
            {cur.centerMode === "fixed" ? (
                <>
                    <div style={LABEL_STYLE}>center (drag over field)</div>
                    <PreviewPad
                        value={cur.center}
                        onChange={(c) => update({ center: c })}
                        nodeId={nodeId ?? null}
                        dotColor="#ffd86b"
                    />
                </>
            ) : null}
            <div style={LABEL_STYLE}>radius</div>
            <Slider
                value={cur.radius}
                min={0.01}
                max={1.5}
                step={0.01}
                onChange={(n) => update({ radius: n })}
            />
            <div style={LABEL_STYLE}>falloff curve</div>
            <Slider
                value={cur.falloff}
                min={0.1}
                max={4}
                step={0.05}
                onChange={(n) => update({ falloff: n })}
            />
            <div style={LABEL_STYLE}>topology influence</div>
            <Slider
                value={cur.topo}
                min={-2}
                max={2}
                step={0.05}
                onChange={(n) => update({ topo: n })}
            />
            <label style={TOGGLE_LABEL_STYLE}>
                <input
                    type="checkbox"
                    checked={cur.flipY}
                    onChange={(e) => update({ flipY: e.target.checked })}
                />
                flip Y (match BumpNormals encoding)
            </label>
            <label style={TOGGLE_LABEL_STYLE}>
                <input
                    type="checkbox"
                    checked={cur.invert}
                    onChange={(e) => update({ invert: e.target.checked })}
                />
                invert output
            </label>
            <div style={HINT_STYLE}>
                In <em>fixed</em> mode the pad shows this node's output as
                a backdrop so you can drag the center over a feature you
                care about. Topology influence at 0 = plain radial
                falloff; positive = uphill pixels (relative to the
                center) feel weaker pull; negative reverses the bias.
            </div>
        </div>
    );
}

export const gravityMapEntry: ShaderEntry = {
    id: "gravityMap",
    name: "Gravity map",
    defaultParams: DEFAULT_PARAMS,
    Controls: GravityMapControls,
    inputs: [{ id: "in", label: "normals" }],
    gpu: gravityMapGpuSpec,
};
