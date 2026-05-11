import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

// Procedural single-pass "particle" pass.
//
// We don't run a real particle simulation (no ping-pong state textures, no
// transform feedback) — instead, every output pixel evaluates a small
// neighborhood of deterministic cells. Each cell hashes to a single
// particle with:
//   • a jittered spawn position inside the cell,
//   • a phase offset so particles are out-of-sync,
//   • emission gated by source brightness at the spawn position (whites
//     emit),
//   • death gated by source brightness at the current position (blacks
//     absorb),
//   • drift in a user-set flow direction over its lifetime.
//
// Checking the 3×3 cell neighborhood around each pixel costs nine texture
// fetches per particle position plus the dot-mask math; cheap on any GPU
// that runs the rest of the pipeline. We clamp particle drift to one cell
// of travel so the 3×3 window always catches every contributor.

interface ParticleParams {
    /** Cells per shorter axis. The other axis scales by aspect to keep
     *  cells square in pixels. */
    readonly density: number;
    /** Particle radius, fraction of a cell. */
    readonly size: number;
    /** Source luminance below this never spawns a particle. */
    readonly emitThreshold: number;
    /** Source luminance below this absorbs an in-flight particle. */
    readonly absorbThreshold: number;
    /** Flow direction, degrees. 0 = right, 90 = down (display y-down),
     *  270 = up. */
    readonly flowAngle: number;
    /** Drift in cells per second. Capped at ~1 cell-per-lifetime by the
     *  shader so the 3×3 search window stays correct. */
    readonly flowSpeed: number;
    /** Particle lifetime in seconds. */
    readonly lifetime: number;
    /** Particle base color. */
    readonly particleColor: string;
    /** 0 = particles drawn over solid black; 1 = drawn over the source. */
    readonly backgroundMix: number;
    /** 0 = constant particle color; 1 = particle color × source luminance
     *  at spawn (so brighter regions emit brighter particles). */
    readonly brightnessTint: number;
    /** Additive contribution multiplier (lets you make the glow stronger
     *  without picking a super-bright color). */
    readonly intensity: number;
}

const DEFAULT_PARAMS: ParticleParams = {
    density: 60,
    size: 0.35,
    emitThreshold: 0.55,
    absorbThreshold: 0.25,
    flowAngle: 270,
    flowSpeed: 0.6,
    lifetime: 3,
    particleColor: "#ffffff",
    backgroundMix: 1,
    brightnessTint: 1,
    intensity: 1.2,
};

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;

uniform float uTime;
uniform float uDensity;
uniform float uAspect;
uniform float uSize;
uniform float uEmitThreshold;
uniform float uAbsorbThreshold;
uniform vec2 uFlowDir;
uniform float uFlowSpeed;
uniform float uLifetime;
uniform vec3 uParticleColor;
uniform float uBackgroundMix;
uniform float uBrightnessTint;
uniform float uIntensity;

const vec3 LUM_W = vec3(0.2126, 0.7152, 0.0722);

float hash11(float x) {
    return fract(sin(x * 127.1) * 43758.5453);
}
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
vec2 hash22(vec2 p) {
    return vec2(
        hash21(p),
        hash21(p + vec2(31.7, 17.3))
    );
}

float lumOf(vec2 uv) {
    return dot(texture(uSource, clamp(uv, 0.0, 1.0)).rgb, LUM_W);
}

void main() {
    vec3 src = texture(uSource, vUv).rgb;
    vec3 bg = src * uBackgroundMix;

    // Square cells in pixels: gridX = density, gridY = density / aspect.
    vec2 grid = vec2(uDensity, max(1.0, uDensity / max(uAspect, 1e-3)));
    vec2 cellSize = 1.0 / grid;
    float cellMin = min(cellSize.x, cellSize.y);

    // Cap drift so the 3×3 search window covers every contributor.
    float maxTravelCells = 0.95;
    float effSpeed = min(uFlowSpeed, maxTravelCells / max(uLifetime, 1e-3));

    vec2 myCell = floor(vUv * grid);
    vec3 accum = vec3(0.0);

    for (float dy = -1.0; dy <= 1.0; dy += 1.0) {
        for (float dx = -1.0; dx <= 1.0; dx += 1.0) {
            vec2 cell = myCell + vec2(dx, dy);
            // Keep particle bookkeeping in cell space; when the cell is
            // off-grid, hashing still gives a deterministic answer but
            // its spawn UV will be outside [0,1] and emit luminance ≈ 0.
            vec2 jitter = hash22(cell) * 0.7 + 0.15;
            float phase0 = hash21(cell + 0.123);

            vec2 spawnUv = (cell + jitter) * cellSize;
            if (spawnUv.x < 0.0 || spawnUv.x > 1.0 ||
                spawnUv.y < 0.0 || spawnUv.y > 1.0) continue;

            float spawnLum = lumOf(spawnUv);
            float emit = smoothstep(
                uEmitThreshold * 0.85,
                uEmitThreshold,
                spawnLum
            );
            if (emit < 1e-3) continue;

            // Phase loops over [0, lifetime), offset per cell.
            float t = mod(uTime + phase0 * uLifetime, uLifetime);
            float phaseFrac = t / max(uLifetime, 1e-3);

            // Drift across cells; uFlowDir is unit-length in UV space
            // (set by setUniforms with the aspect ratio baked in).
            vec2 curUv = spawnUv + uFlowDir * cellMin * effSpeed * t;

            float curLum = lumOf(curUv);
            float alive = smoothstep(
                uAbsorbThreshold * 0.5,
                uAbsorbThreshold,
                curLum
            );

            // Birth/death taper at the lifetime endpoints.
            float lifeFade = smoothstep(0.0, 0.08, phaseFrac)
                           * (1.0 - smoothstep(0.85, 1.0, phaseFrac));

            // Distance to the particle, soft circular falloff.
            float radius = uSize * cellMin;
            // Aspect-correct distance so circles look round in pixels.
            vec2 d = (vUv - curUv) * vec2(uAspect, 1.0);
            float dist = length(d);
            float dot1 = 1.0 - smoothstep(radius * 0.6, radius, dist);

            // Optional brightness tint: brighter spawns emit hotter dots.
            float tintFactor = mix(1.0, spawnLum, uBrightnessTint);

            accum += uParticleColor
                   * dot1
                   * emit
                   * alive
                   * lifeFade
                   * tintFactor
                   * uIntensity;
        }
    }

    outColor = vec4(bg + accum, 1.0);
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
    return { r: 255, g: 255, b: 255 };
}

const particlesGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: [
        "uTime",
        "uDensity",
        "uAspect",
        "uSize",
        "uEmitThreshold",
        "uAbsorbThreshold",
        "uFlowDir",
        "uFlowSpeed",
        "uLifetime",
        "uParticleColor",
        "uBackgroundMix",
        "uBrightnessTint",
        "uIntensity",
    ],
    setUniforms: (gl, locs, params, frame) => {
        const p: ParticleParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<ParticleParams>),
        };
        // Aspect from the bound draw buffer — same metric the FBO holds.
        const drawW = gl.drawingBufferWidth;
        const drawH = gl.drawingBufferHeight;
        const aspect = drawW / Math.max(1, drawH);
        const angleRad = (p.flowAngle * Math.PI) / 180;
        // UV-space flow direction, normalized so cellMin units in the
        // shader correspond to roughly square steps regardless of aspect.
        const flowX = Math.cos(angleRad);
        const flowY = Math.sin(angleRad);
        const c = parseHexColor(p.particleColor);

        gl.uniform1f(locs.get("uTime")!, frame.tNow);
        gl.uniform1f(locs.get("uDensity")!, p.density);
        gl.uniform1f(locs.get("uAspect")!, aspect);
        gl.uniform1f(locs.get("uSize")!, p.size);
        gl.uniform1f(locs.get("uEmitThreshold")!, p.emitThreshold);
        gl.uniform1f(locs.get("uAbsorbThreshold")!, p.absorbThreshold);
        gl.uniform2f(locs.get("uFlowDir")!, flowX, flowY);
        gl.uniform1f(locs.get("uFlowSpeed")!, p.flowSpeed);
        gl.uniform1f(locs.get("uLifetime")!, Math.max(0.1, p.lifetime));
        gl.uniform3f(
            locs.get("uParticleColor")!,
            c.r / 255,
            c.g / 255,
            c.b / 255,
        );
        gl.uniform1f(locs.get("uBackgroundMix")!, p.backgroundMix);
        gl.uniform1f(locs.get("uBrightnessTint")!, p.brightnessTint);
        gl.uniform1f(locs.get("uIntensity")!, p.intensity);
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const SECTION_TITLE_STYLE: CSSProperties = {
    color: "#bdbdbd",
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginTop: 12,
    marginBottom: 4,
    paddingTop: 8,
    borderTop: "1px solid #2a2a2a",
};

const FIRST_SECTION_TITLE_STYLE: CSSProperties = {
    ...SECTION_TITLE_STYLE,
    marginTop: 0,
    paddingTop: 0,
    borderTop: "none",
};

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

function ParticleControls({ params, onChange }: ShaderControlsProps) {
    const cur: ParticleParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<ParticleParams>),
    };
    const update = (patch: Partial<ParticleParams>) =>
        onChange({ ...cur, ...patch } satisfies ParticleParams);

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

            <div style={FIRST_SECTION_TITLE_STYLE}>field</div>
            <SliderRow
                label="density (cells/axis)"
                value={cur.density}
                min={10}
                max={250}
                step={1}
                defaultValue={DEFAULT_PARAMS.density}
                format={(v) => v.toFixed(0)}
                onChange={(v) => update({ density: v })}
            />
            <SliderRow
                label="particle size"
                value={cur.size}
                min={0.05}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.size}
                onChange={(v) => update({ size: v })}
            />

            <div style={SECTION_TITLE_STYLE}>emission · absorption</div>
            <SliderRow
                label="emit threshold (white spawns)"
                value={cur.emitThreshold}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.emitThreshold}
                onChange={(v) => update({ emitThreshold: v })}
            />
            <SliderRow
                label="absorb threshold (black kills)"
                value={cur.absorbThreshold}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.absorbThreshold}
                onChange={(v) => update({ absorbThreshold: v })}
            />

            <div style={SECTION_TITLE_STYLE}>motion</div>
            <SliderRow
                label="flow angle (°)"
                value={cur.flowAngle}
                min={0}
                max={360}
                step={1}
                defaultValue={DEFAULT_PARAMS.flowAngle}
                format={(v) => v.toFixed(0)}
                onChange={(v) => update({ flowAngle: v })}
            />
            <SliderRow
                label="flow speed (cells/sec)"
                value={cur.flowSpeed}
                min={0}
                max={3}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.flowSpeed}
                onChange={(v) => update({ flowSpeed: v })}
            />
            <SliderRow
                label="lifetime (sec)"
                value={cur.lifetime}
                min={0.5}
                max={10}
                step={0.1}
                defaultValue={DEFAULT_PARAMS.lifetime}
                format={(v) => v.toFixed(1)}
                onChange={(v) => update({ lifetime: v })}
            />

            <div style={SECTION_TITLE_STYLE}>look</div>
            <div style={LABEL_STYLE}>
                <span>particle color</span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="color"
                    style={COLOR_INPUT_STYLE}
                    value={cur.particleColor}
                    onChange={(e) =>
                        update({ particleColor: e.target.value })
                    }
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() =>
                        update({
                            particleColor: DEFAULT_PARAMS.particleColor,
                        })
                    }
                    title="reset"
                >
                    ↺
                </button>
            </div>
            <SliderRow
                label="intensity"
                value={cur.intensity}
                min={0}
                max={4}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.intensity}
                onChange={(v) => update({ intensity: v })}
            />
            <SliderRow
                label="brightness tint (0=flat, 1=source-lit)"
                value={cur.brightnessTint}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.brightnessTint}
                onChange={(v) => update({ brightnessTint: v })}
            />
            <SliderRow
                label="background mix (0=black, 1=source)"
                value={cur.backgroundMix}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.backgroundMix}
                onChange={(v) => update({ backgroundMix: v })}
            />
        </div>
    );
}

export const particlesEntry: ShaderEntry = {
    id: "particles",
    name: "Particles (whites emit, blacks absorb)",
    defaultParams: DEFAULT_PARAMS,
    Controls: ParticleControls,
    inputs: [{ id: "in", label: "field (B/W)" }],
    gpu: particlesGpuSpec,
};
