import type { CSSProperties } from "react";
import type { GpuPassSpec, ShaderControlsProps, ShaderEntry } from ".";

// Marbles — a simple 2D particle simulation that reads a scalar gravity
// field as input and rolls toward higher field values. Physics runs in
// JS each frame, sampling the captured input image's gradient at every
// marble position; the GLSL pass just renders the resulting positions
// as smoothed circles over (optionally) the field backdrop.
//
// Wire a Gravity Map (or any single-channel intensity field) into the
// "field" handle; marbles fall toward the brightest regions. Drop the
// shader into a Layers stack with `show field = false` to overlay them
// on something else.

export const MARBLES_MAX = 64;

interface MarblesParams {
    /** How many marbles to simulate (capped at MARBLES_MAX). */
    readonly count: number;
    /** Marble radius in vUv units (0..0.05 looks reasonable). */
    readonly radius: number;
    /** Acceleration magnitude scale: F = grad(field) * gravity. */
    readonly gravity: number;
    /** Per-second velocity damping (0 = none, 1 = instant stop). */
    readonly damping: number;
    /** Velocity multiplier on edge bounce (0 = stick, 1 = elastic). */
    readonly bounce: number;
    /** Marble fill color, "#rrggbb". */
    readonly color: string;
    /** When true, the field backdrop shows through; when false, the
     *  output is marbles on transparent so they layer cleanly. */
    readonly showField: boolean;
    /** PRNG seed for initial marble layout. Editing this re-seeds the
     *  positions. */
    readonly seed: number;
}

const DEFAULT_PARAMS: MarblesParams = {
    count: 24,
    radius: 0.014,
    gravity: 3,
    damping: 0.6,
    bounce: 0.5,
    color: "#ffd86b",
    showField: true,
    seed: 1,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

const int MAX_MARBLES = ${MARBLES_MAX};

uniform sampler2D uField;
uniform vec2 uMarblePos[MAX_MARBLES];
uniform int uMarbleCount;
uniform float uRadius;
uniform vec3 uColor;
uniform int uShowField;
uniform float uAA;

void main() {
    float minD = 1e9;
    for (int i = 0; i < MAX_MARBLES; i++) {
        if (i >= uMarbleCount) break;
        float d = distance(vUv, uMarblePos[i]);
        if (d < minD) minD = d;
    }
    float marbleAlpha =
        1.0 - smoothstep(uRadius - uAA, uRadius + uAA, minD);

    if (uShowField == 1) {
        vec4 fld = texture(uField, vUv);
        vec3 rgb = mix(fld.rgb, uColor, marbleAlpha);
        outColor = vec4(rgb, max(fld.a, marbleAlpha));
    } else {
        outColor = vec4(uColor, marbleAlpha);
    }
}
`;

// ─── Per-node JS state ──────────────────────────────────────────────────
//
// Module-scoped: keyed by the editor node id from FrameContext. The
// pipeline doesn't expose a per-node lifecycle hook for purely-GPU
// shaders, so stale entries linger until the page reloads — fine for
// this scale. State persists across chain rebuilds with the same id.

interface MarblesState {
    positions: Float32Array; // length 2 * MARBLES_MAX (extra slack stays zero)
    velocities: Float32Array;
    count: number;
    seed: number;
    lastTNow: number;
}

const STATE_BY_NODE = new Map<string, MarblesState>();

function mulberry32(seed: number): () => number {
    let s = seed | 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function ensureState(
    nodeId: string,
    count: number,
    seed: number,
): MarblesState {
    let state = STATE_BY_NODE.get(nodeId);
    if (
        !state ||
        state.count !== count ||
        state.seed !== seed
    ) {
        const rnd = mulberry32(seed * 2654435761 + count);
        const positions = new Float32Array(MARBLES_MAX * 2);
        const velocities = new Float32Array(MARBLES_MAX * 2);
        for (let i = 0; i < count; i++) {
            positions[i * 2] = 0.1 + rnd() * 0.8;
            positions[i * 2 + 1] = 0.1 + rnd() * 0.8;
        }
        state = {
            positions,
            velocities,
            count,
            seed,
            lastTNow: 0,
        };
        STATE_BY_NODE.set(nodeId, state);
    }
    return state;
}

function parseHexColor(hex: string): [number, number, number] {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
    if (!m) return [1, 0.85, 0.42];
    const v = parseInt(m[1]!, 16);
    return [
        ((v >> 16) & 0xff) / 255,
        ((v >> 8) & 0xff) / 255,
        (v & 0xff) / 255,
    ];
}

function clampInt(v: number, lo: number, hi: number): number {
    return v < lo ? lo : v > hi ? hi : v;
}

const marblesGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uField"],
    uniforms: [
        "uMarblePos[0]",
        "uMarbleCount",
        "uRadius",
        "uColor",
        "uShowField",
        "uAA",
    ],
    capturedInputSlots: [0],
    setUniforms: (gl, locs, params, frame) => {
        const p: MarblesParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<MarblesParams>),
        };
        const count = clampInt(Math.round(p.count), 0, MARBLES_MAX);
        const state = ensureState(frame.nodeId, count, p.seed);

        // dt: clamp to a plausible single-frame interval. Big jumps (tab
        // suspended, debugger paused) would otherwise launch marbles
        // off-screen on the first frame after resume.
        const dt =
            state.lastTNow > 0
                ? Math.max(0, Math.min(0.05, frame.tNow - state.lastTNow))
                : 0;
        state.lastTNow = frame.tNow;

        const field = frame.capturedInputs[0];
        if (field && dt > 0 && count > 0) {
            const W = field.width;
            const H = field.height;
            const data = field.data;
            const sample = (u: number, v: number): number => {
                // ImageData rows run top-down; vUv runs bottom-up.
                const ix = clampInt(Math.floor(u * W), 0, W - 1);
                const iy = clampInt(Math.floor((1 - v) * H), 0, H - 1);
                return data[(iy * W + ix) * 4]! / 255;
            };
            const eps = 2 / W;
            const dampPerFrame = Math.pow(
                Math.max(0, 1 - p.damping),
                dt,
            );

            for (let i = 0; i < count; i++) {
                const xi = i * 2;
                const yi = xi + 1;
                const x = state.positions[xi]!;
                const y = state.positions[yi]!;
                // Central differences → gradient pointing toward higher
                // field intensity (i.e., toward the cursor, modulated
                // by topology).
                const gx =
                    (sample(x + eps, y) - sample(x - eps, y)) / (2 * eps);
                const gy =
                    (sample(x, y + eps) - sample(x, y - eps)) / (2 * eps);
                state.velocities[xi] = state.velocities[xi]! + gx * p.gravity * dt;
                state.velocities[yi] = state.velocities[yi]! + gy * p.gravity * dt;
                state.velocities[xi] = state.velocities[xi]! * dampPerFrame;
                state.velocities[yi] = state.velocities[yi]! * dampPerFrame;
                let nx = x + state.velocities[xi]! * dt;
                let ny = y + state.velocities[yi]! * dt;
                if (nx < 0) {
                    nx = 0;
                    state.velocities[xi] = -state.velocities[xi]! * p.bounce;
                } else if (nx > 1) {
                    nx = 1;
                    state.velocities[xi] = -state.velocities[xi]! * p.bounce;
                }
                if (ny < 0) {
                    ny = 0;
                    state.velocities[yi] = -state.velocities[yi]! * p.bounce;
                } else if (ny > 1) {
                    ny = 1;
                    state.velocities[yi] = -state.velocities[yi]! * p.bounce;
                }
                state.positions[xi] = nx;
                state.positions[yi] = ny;
            }
        }

        gl.uniform2fv(locs.get("uMarblePos[0]")!, state.positions);
        gl.uniform1i(locs.get("uMarbleCount")!, count);
        gl.uniform1f(locs.get("uRadius")!, Math.max(0.0005, p.radius));
        const [r, g, b] = parseHexColor(p.color);
        gl.uniform3f(locs.get("uColor")!, r, g, b);
        gl.uniform1i(locs.get("uShowField")!, p.showField ? 1 : 0);
        // ~1.5 pixels of AA, in vUv units. Field's size is the only
        // resolution we know cheaply; close enough.
        const aaPx =
            field && field.width > 0 ? 1.5 / field.width : 1.5 / 1024;
        gl.uniform1f(locs.get("uAA")!, aaPx);
    },
};

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

const NUM_STYLE: CSSProperties = {
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

const COLOR_INPUT_STYLE: CSSProperties = {
    background: "#0a0a0a",
    border: "1px solid #333",
    borderRadius: 4,
    width: 40,
    height: 24,
    padding: 0,
    cursor: "pointer",
};

const TOGGLE_LABEL: CSSProperties = {
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

const BUTTON_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 10px",
    fontSize: 11,
    fontFamily: "inherit",
    cursor: "pointer",
    marginTop: 6,
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
                style={NUM_STYLE}
            />
        </div>
    );
}

function MarblesControls({ params, onChange }: ShaderControlsProps) {
    const cur: MarblesParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<MarblesParams>),
    };
    const update = (patch: Partial<MarblesParams>) =>
        onChange({ ...cur, ...patch });
    return (
        <div>
            <div style={LABEL_STYLE}>count</div>
            <Slider
                value={cur.count}
                min={1}
                max={MARBLES_MAX}
                step={1}
                onChange={(n) => update({ count: n })}
            />
            <div style={LABEL_STYLE}>radius</div>
            <Slider
                value={cur.radius}
                min={0.002}
                max={0.05}
                step={0.001}
                onChange={(n) => update({ radius: n })}
            />
            <div style={LABEL_STYLE}>gravity</div>
            <Slider
                value={cur.gravity}
                min={0}
                max={10}
                step={0.05}
                onChange={(n) => update({ gravity: n })}
            />
            <div style={LABEL_STYLE}>damping (per second)</div>
            <Slider
                value={cur.damping}
                min={0}
                max={1}
                step={0.01}
                onChange={(n) => update({ damping: n })}
            />
            <div style={LABEL_STYLE}>edge bounce</div>
            <Slider
                value={cur.bounce}
                min={0}
                max={1}
                step={0.01}
                onChange={(n) => update({ bounce: n })}
            />
            <div style={LABEL_STYLE}>color</div>
            <div style={ROW_STYLE}>
                <input
                    type="color"
                    value={cur.color}
                    onChange={(e) => update({ color: e.target.value })}
                    style={COLOR_INPUT_STYLE}
                />
                <span
                    style={{
                        color: "#888",
                        fontSize: 11,
                        fontVariantNumeric: "tabular-nums",
                    }}
                >
                    {cur.color}
                </span>
            </div>
            <label style={TOGGLE_LABEL}>
                <input
                    type="checkbox"
                    checked={cur.showField}
                    onChange={(e) => update({ showField: e.target.checked })}
                />
                show field backdrop
            </label>
            <div style={LABEL_STYLE}>seed</div>
            <Slider
                value={cur.seed}
                min={0}
                max={9999}
                step={1}
                onChange={(n) => update({ seed: Math.round(n) })}
            />
            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={() => update({ seed: cur.seed + 1 })}
                title="Increment seed to re-roll initial positions"
            >
                shake
            </button>
            <div style={HINT_STYLE}>
                Wire a Gravity Map into the field handle. Marbles roll up the
                gradient toward brighter regions; pointer-driven gravity in
                the upstream shader is what they're chasing. Higher damping
                makes them settle faster; lower lets them oscillate.
            </div>
        </div>
    );
}

export const marblesEntry: ShaderEntry = {
    id: "marbles",
    name: "Marbles",
    defaultParams: DEFAULT_PARAMS,
    Controls: MarblesControls,
    inputs: [{ id: "in", label: "field" }],
    gpu: marblesGpuSpec,
};
