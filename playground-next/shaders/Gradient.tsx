import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

type WaveShape = "sine" | "triangle" | "sawtooth" | "square";

interface GradientParams {
    /** Direction in degrees. 0 = left→right, 90 = top→bottom (display y). */
    readonly angleDeg: number;
    /** Number of full cycles along the direction axis. Integer = seamless.
     *  Used when no modulator input is wired (or the pointer is outside
     *  the canvas). */
    readonly frequency: number;
    /** Lower bound of frequency when the modulator is active. White (lum=1)
     *  at the pointer's UV maps here. */
    readonly frequencyMin: number;
    /** Upper bound of frequency when the modulator is active. Black (lum=0)
     *  at the pointer's UV maps here. */
    readonly frequencyMax: number;
    /** Phase offset, 0..1 of one cycle. */
    readonly phase: number;
    readonly wave: WaveShape;
    readonly invert: boolean;
    /** Edge softness for square waves, 0..1 of one cycle width. */
    readonly softness: number;
    /** Bias the wave's mapping toward one colour. [-1, 1].
     *  +1 widens the `colorHigh` area, -1 widens the `colorLow` area,
     *  0 leaves the wave untouched. */
    readonly balance: number;
    readonly colorLow: string;
    readonly colorHigh: string;
    /** When true, phase advances by speedHz × tNow each frame. */
    readonly playing: boolean;
    /** Cycles per second. Negative reverses direction. */
    readonly speedHz: number;
}

const DEFAULT_PARAMS: GradientParams = {
    angleDeg: 0,
    frequency: 2,
    frequencyMin: 1,
    frequencyMax: 16,
    phase: 0,
    wave: "sine",
    invert: false,
    softness: 0.05,
    balance: 0,
    colorLow: "#000000",
    colorHigh: "#ffffff",
    playing: false,
    speedHz: 0.25,
};

const WAVE_INDEX: Record<WaveShape, number> = {
    sine: 0,
    triangle: 1,
    sawtooth: 2,
    square: 3,
};

// vUv runs bottom-up by convention. We project the direction against
// (vUv.x, 1.0 - vUv.y) so the user-facing angle reads with display y-down:
// 0° = →, 90° = ↓ on screen.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform float uAngleRad;
uniform float uFrequency;
uniform float uFrequencyMin;
uniform float uFrequencyMax;
uniform float uPhase;
uniform int uWave;
uniform float uSoftness;
uniform float uInvert;
uniform float uBalance;
uniform vec3 uColorLow;
uniform vec3 uColorHigh;

// Frequency modulation: when a modulator chain is wired AND the pointer
// is over the canvas, the luminance of uModulator at uPointerUv picks a
// frequency from [uFrequencyMin (white), uFrequencyMax (black)]. When
// either gate is false, uFrequency is used.
uniform sampler2D uModulator;
uniform vec2 uPointerUv;
uniform float uPointerActive;
uniform float uModulatorActive;

float sampleWave(float t) {
    float f = fract(t);
    if (uWave == 0) {
        return (sin(f * 6.2831853) + 1.0) * 0.5;
    } else if (uWave == 1) {
        return f < 0.5 ? f * 2.0 : 2.0 - f * 2.0;
    } else if (uWave == 2) {
        return f;
    }
    float s = clamp(uSoftness, 0.0001, 0.5);
    float rise = smoothstep(0.25 - s, 0.25 + s, f);
    float fall = smoothstep(0.75 - s, 0.75 + s, f);
    return rise - fall;
}

void main() {
    vec2 dir = vec2(cos(uAngleRad), sin(uAngleRad));
    float span = abs(dir.x) + abs(dir.y);
    vec2 nDir = dir / max(span, 1e-6);
    // (vUv.x, 1.0 - vUv.y) puts (0,0) at top-left and (1,1) at bottom-right
    // — display-y-down — so angle semantics match Canvas2D.
    float t = vUv.x * nDir.x + (1.0 - vUv.y) * nDir.y;

    float modGate = uModulatorActive * uPointerActive;
    vec3 modSample = texture(uModulator, uPointerUv).rgb;
    float modLum = dot(modSample, vec3(0.299, 0.587, 0.114));
    float modFreq = mix(uFrequencyMax, uFrequencyMin, modLum);
    float freq = mix(uFrequency, modFreq, modGate);

    float g = sampleWave(t * freq + uPhase);
    g = mix(g, 1.0 - g, uInvert);

    // Balance: a symmetric power curve in log space biases the output
    // toward colorHigh (uBalance > 0) or colorLow (uBalance < 0). 0 is
    // identity. The k=3 multiplier gives the [-1, 1] slider a useful
    // range — exponent runs across 1/8..1..8.
    float biasExp = pow(2.0, -uBalance * 3.0);
    g = pow(clamp(g, 0.0, 1.0), biasExp);

    outColor = vec4(mix(uColorLow, uColorHigh, g), 1.0);
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
    return { r: 0, g: 0, b: 0 };
}

const gradientGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uModulator"],
    uniforms: [
        "uAngleRad",
        "uFrequency",
        "uFrequencyMin",
        "uFrequencyMax",
        "uPhase",
        "uWave",
        "uSoftness",
        "uInvert",
        "uBalance",
        "uColorLow",
        "uColorHigh",
        "uPointerUv",
        "uPointerActive",
        "uModulatorActive",
    ],
    setUniforms: (gl, locs, params, frame) => {
        const p: GradientParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<GradientParams>),
        };
        const lo = parseHexColor(p.colorLow);
        const hi = parseHexColor(p.colorHigh);
        const angleRad = (p.angleDeg * Math.PI) / 180;
        const phase =
            p.phase + (p.playing ? p.speedHz * frame.tNow : 0);
        gl.uniform1f(locs.get("uAngleRad")!, angleRad);
        gl.uniform1f(locs.get("uFrequency")!, p.frequency);
        gl.uniform1f(locs.get("uFrequencyMin")!, p.frequencyMin);
        gl.uniform1f(locs.get("uFrequencyMax")!, p.frequencyMax);
        gl.uniform1f(locs.get("uPhase")!, phase);
        gl.uniform1i(locs.get("uWave")!, WAVE_INDEX[p.wave]);
        gl.uniform1f(locs.get("uSoftness")!, p.softness);
        gl.uniform1f(locs.get("uInvert")!, p.invert ? 1 : 0);
        gl.uniform1f(locs.get("uBalance")!, p.balance);
        gl.uniform3f(
            locs.get("uColorLow")!,
            lo.r / 255,
            lo.g / 255,
            lo.b / 255,
        );
        gl.uniform3f(
            locs.get("uColorHigh")!,
            hi.r / 255,
            hi.g / 255,
            hi.b / 255,
        );
        gl.uniform2f(
            locs.get("uPointerUv")!,
            frame.pointer.uv[0],
            frame.pointer.uv[1],
        );
        gl.uniform1f(
            locs.get("uPointerActive")!,
            frame.pointer.active ? 1 : 0,
        );
        gl.uniform1f(
            locs.get("uModulatorActive")!,
            frame.inputsPresent[0] ? 1 : 0,
        );
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 4,
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

const CHECK_ROW: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginTop: 8,
    fontSize: 12,
    color: "#bdbdbd",
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
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

const COLOR_INPUT_STYLE: CSSProperties = {
    width: 32,
    height: 22,
    padding: 0,
    border: "1px solid #333",
    borderRadius: 4,
    background: "transparent",
    cursor: "pointer",
};

const PRESET_ANGLES: ReadonlyArray<{ label: string; deg: number }> = [
    { label: "→", deg: 0 },
    { label: "↘", deg: 45 },
    { label: "↓", deg: 90 },
    { label: "↙", deg: 135 },
    { label: "←", deg: 180 },
    { label: "↖", deg: 225 },
    { label: "↑", deg: 270 },
    { label: "↗", deg: 315 },
];

const SPEED_PRESETS = [0.1, 0.25, 0.5, 1, 2, 4] as const;

function GradientControls({ params, onChange }: ShaderControlsProps) {
    const cur: GradientParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<GradientParams>),
    };
    const update = (patch: Partial<GradientParams>) =>
        onChange({ ...cur, ...patch } satisfies GradientParams);

    return (
        <div>
            <div style={LABEL_STYLE}>direction</div>
            <div style={{ ...ROW_STYLE, flexWrap: "wrap", gap: 4 }}>
                {PRESET_ANGLES.map((a) => (
                    <button
                        key={a.deg}
                        type="button"
                        style={{
                            ...BUTTON_STYLE,
                            background:
                                cur.angleDeg === a.deg ? "#1f2a3a" : "#1a1a1a",
                            border:
                                cur.angleDeg === a.deg
                                    ? "1px solid #4a90e2"
                                    : "1px solid #333",
                        }}
                        onClick={() => update({ angleDeg: a.deg })}
                        title={`${a.deg}°`}
                    >
                        {a.label}
                    </button>
                ))}
            </div>
            <div style={LABEL_STYLE}>angle: {cur.angleDeg.toFixed(0)}°</div>
            <input
                type="range"
                min={0}
                max={359}
                step={1}
                value={cur.angleDeg}
                onChange={(e) =>
                    update({ angleDeg: parseInt(e.target.value, 10) })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>
                frequency: {cur.frequency}{" "}
                {Number.isInteger(cur.frequency) ? "(seamless)" : "(non-tiling)"}
            </div>
            <input
                type="range"
                min={1}
                max={32}
                step={1}
                value={cur.frequency}
                onChange={(e) =>
                    update({ frequency: parseInt(e.target.value, 10) })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>
                modulator range (white → min, black → max){" "}
                <span style={{ color: "#666" }}>
                    — wire any B/W chain into the input handle
                </span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="range"
                    min={1}
                    max={64}
                    step={1}
                    value={cur.frequencyMin}
                    onChange={(e) =>
                        update({ frequencyMin: parseInt(e.target.value, 10) })
                    }
                    style={{ flex: 1 }}
                    title={`min ${cur.frequencyMin}`}
                />
                <input
                    type="range"
                    min={1}
                    max={64}
                    step={1}
                    value={cur.frequencyMax}
                    onChange={(e) =>
                        update({ frequencyMax: parseInt(e.target.value, 10) })
                    }
                    style={{ flex: 1 }}
                    title={`max ${cur.frequencyMax}`}
                />
            </div>
            <div
                style={{
                    color: "#888",
                    fontSize: 11,
                    display: "flex",
                    justifyContent: "space-between",
                }}
            >
                <span>min: {cur.frequencyMin}</span>
                <span>max: {cur.frequencyMax}</span>
            </div>

            <div style={LABEL_STYLE}>phase: {cur.phase.toFixed(2)}</div>
            <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={cur.phase}
                onChange={(e) => update({ phase: parseFloat(e.target.value) })}
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>animation</div>
            <div style={ROW_STYLE}>
                <button
                    type="button"
                    style={{
                        ...BUTTON_STYLE,
                        background: cur.playing ? "#1f2a3a" : "#1a1a1a",
                        border: cur.playing
                            ? "1px solid #4a90e2"
                            : "1px solid #333",
                        minWidth: 60,
                    }}
                    onClick={() => update({ playing: !cur.playing })}
                    title={cur.playing ? "pause" : "play"}
                >
                    {cur.playing ? "⏸ pause" : "▶ play"}
                </button>
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() => update({ speedHz: -cur.speedHz })}
                    title="reverse direction"
                >
                    ⇄ reverse
                </button>
            </div>

            <div style={LABEL_STYLE}>
                speed: {cur.speedHz.toFixed(2)} cycles/sec
                {cur.speedHz < 0 ? " (reverse)" : ""}
            </div>
            <input
                type="range"
                min={-8}
                max={8}
                step={0.05}
                value={cur.speedHz}
                onChange={(e) =>
                    update({ speedHz: parseFloat(e.target.value) })
                }
                style={{ width: "100%" }}
            />
            <div style={{ ...ROW_STYLE, flexWrap: "wrap", gap: 4 }}>
                {SPEED_PRESETS.map((s) => {
                    const active = Math.abs(cur.speedHz - s) < 0.001;
                    return (
                        <button
                            key={s}
                            type="button"
                            style={{
                                ...BUTTON_STYLE,
                                background: active ? "#1f2a3a" : "#1a1a1a",
                                border: active
                                    ? "1px solid #4a90e2"
                                    : "1px solid #333",
                            }}
                            onClick={() => update({ speedHz: s })}
                            title={`${s} cycles/sec`}
                        >
                            {s}×
                        </button>
                    );
                })}
            </div>

            <div style={LABEL_STYLE}>wave</div>
            <select
                style={SELECT_STYLE}
                value={cur.wave}
                onChange={(e) => update({ wave: e.target.value as WaveShape })}
            >
                <option value="sine">sine (smooth)</option>
                <option value="triangle">triangle</option>
                <option value="sawtooth">sawtooth</option>
                <option value="square">square</option>
            </select>

            {cur.wave === "square" && (
                <>
                    <div style={LABEL_STYLE}>
                        softness: {cur.softness.toFixed(2)}
                    </div>
                    <input
                        type="range"
                        min={0.001}
                        max={0.5}
                        step={0.005}
                        value={cur.softness}
                        onChange={(e) =>
                            update({ softness: parseFloat(e.target.value) })
                        }
                        style={{ width: "100%" }}
                    />
                </>
            )}

            <div style={LABEL_STYLE}>
                balance: {cur.balance.toFixed(2)}{" "}
                {cur.balance > 0
                    ? "(more high)"
                    : cur.balance < 0
                        ? "(more low)"
                        : "(neutral)"}
            </div>
            <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={cur.balance}
                onChange={(e) =>
                    update({ balance: parseFloat(e.target.value) })
                }
                style={{ width: "100%" }}
            />

            <div style={LABEL_STYLE}>colors (low → high)</div>
            <div style={ROW_STYLE}>
                <input
                    type="color"
                    style={COLOR_INPUT_STYLE}
                    value={cur.colorLow}
                    onChange={(e) => update({ colorLow: e.target.value })}
                    title="color at wave low (value 0)"
                />
                <input
                    type="color"
                    style={COLOR_INPUT_STYLE}
                    value={cur.colorHigh}
                    onChange={(e) => update({ colorHigh: e.target.value })}
                    title="color at wave high (value 1)"
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() =>
                        update({
                            colorLow: cur.colorHigh,
                            colorHigh: cur.colorLow,
                        })
                    }
                    title="swap colors"
                >
                    swap
                </button>
            </div>

            <label style={CHECK_ROW}>
                <input
                    type="checkbox"
                    checked={cur.invert}
                    onChange={(e) => update({ invert: e.target.checked })}
                />
                invert wave
            </label>
        </div>
    );
}

export const gradientEntry: ShaderEntry = {
    id: "gradient",
    name: "Gradient (oscillating)",
    defaultParams: DEFAULT_PARAMS,
    Controls: GradientControls,
    inputs: [
        {
            id: "modulator",
            label: "frequency map (B/W, sampled at pointer)",
            optional: true,
        },
    ],
    gpu: gradientGpuSpec,
};
