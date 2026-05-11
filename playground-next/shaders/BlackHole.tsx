import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

// Procedural "black hole" particle field.
//
// We can't run a real N-body simulation in a single fragment pass, so we
// fake it with deterministic particles: each output pixel iterates a
// fixed list of N particles, each with a hash-based spawn UV and phase
// offset. Per particle we compute, from the current global time:
//
//   • spawn position in the source's UV space (sampled across the
//     canvas — every part of the source contributes),
//   • current polar (r, θ) around the black-hole centre, where r decays
//     toward 0 over the particle's lifetime and θ accelerates as r
//     shrinks (closer = faster orbit, like a Keplerian fall),
//   • current size, scaled by r so particles shrink as they spiral in,
//   • colour, sampled from the source at the *spawn* position so each
//     particle carries a slice of the image into the void.
//
// Particles outside the event horizon contribute additively. Everything
// inside the horizon is force-darkened.

interface BlackHoleParams {
    /** Hole center in UV space. */
    readonly centerX: number;
    readonly centerY: number;
    /** Event horizon radius in aspect-corrected UV units (0..0.5). */
    readonly horizonRadius: number;
    /** Soft-edge width for the horizon, same units. */
    readonly horizonSoftness: number;
    /** Radial falloff exponent: how aggressive the inward pull is.
     *  1 = linear shrink, 2 = ease-in (gentle then fast), 4 = sharp dive. */
    readonly pullPower: number;
    /** Particle lifetime in seconds. */
    readonly lifetime: number;
    /** Base orbital angular velocity, radians per lifetime. */
    readonly orbitSpeed: number;
    /** Extra angular acceleration as r shrinks; 0 = constant orbit,
     *  higher = pronounced "winding up" near the centre. */
    readonly orbitAccel: number;
    /** How many particles in the loop (compile-time max is MAX_N). */
    readonly particleCount: number;
    /** Particle radius at spawn, in aspect-corrected UV units. */
    readonly baseSize: number;
    /** Exponent on the size-vs-radius scaling. 1 = linear. <1 keeps
     *  particles bigger longer. */
    readonly sizeFalloff: number;
    /** Multiplier on the particle accumulation. */
    readonly intensity: number;
    /** 0 = particles drawn over solid black, 1 = drawn over the source. */
    readonly backgroundMix: number;
}

const DEFAULT_PARAMS: BlackHoleParams = {
    centerX: 0.5,
    centerY: 0.5,
    horizonRadius: 0.05,
    horizonSoftness: 0.02,
    pullPower: 2.5,
    lifetime: 6,
    orbitSpeed: 6.0,
    orbitAccel: 4.0,
    particleCount: 200,
    baseSize: 0.012,
    sizeFalloff: 1.0,
    intensity: 1.5,
    backgroundMix: 0.15,
};

// Compile-time max particle count. The uniform `uParticleCount` indexes
// inside this fixed bound via an `if (i >= N) break;` guard, so drivers
// treat the loop as dynamic-bounded rather than fully unrolling. The
// per-particle squared-distance early-out keeps the cost sublinear in N
// for any reasonable particle size — bumping this is much cheaper than
// it looks on paper.
const MAX_N = 1024;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;

uniform float uTime;
uniform float uAspect;
uniform vec2  uCenter;
uniform float uHorizonRadius;
uniform float uHorizonSoftness;
uniform float uPullPower;
uniform float uLifetime;
uniform float uOrbitSpeed;
uniform float uOrbitAccel;
uniform int   uParticleCount;
uniform float uBaseSize;
uniform float uSizeFalloff;
uniform float uIntensity;
uniform float uBackgroundMix;

const int MAX_N = ${MAX_N};
const float TAU = 6.28318530718;

float hash11(float x) {
    return fract(sin(x * 127.1) * 43758.5453);
}
vec2 hash21(float x) {
    return vec2(
        fract(sin(x * 127.1) * 43758.5453),
        fract(sin(x * 311.7) * 96485.6951)
    );
}

void main() {
    vec3 src = texture(uSource, vUv).rgb;
    vec3 bg = src * uBackgroundMix;

    // Aspect-corrected coords ensure the orbit and horizon look round on
    // non-square canvases (we measure radial distance in "pixel-square"
    // units rather than UV units).
    vec2 ar = vec2(uAspect, 1.0);
    vec2 pos = vUv * ar;
    vec2 ctr = uCenter * ar;

    vec3 accum = vec3(0.0);

    for (int i = 0; i < MAX_N; i++) {
        if (i >= uParticleCount) break;
        float fi = float(i) + 1.0;

        // Per-particle deterministic spawn position (in UV) and phase.
        vec2 spawnUv = hash21(fi * 0.7193);
        float phaseOffset = hash11(fi * 1.3331);

        // Spawn polar around the black hole, in aspect-corrected coords.
        vec2 fromCenter = spawnUv * ar - ctr;
        float spawnR = length(fromCenter);
        // Skip particles that spawn already inside the horizon.
        if (spawnR < uHorizonRadius) continue;
        float spawnTheta = atan(fromCenter.y, fromCenter.x);

        // Lifetime fraction t ∈ [0,1).
        float t = fract(uTime / max(uLifetime, 1e-3) + phaseOffset);

        // Radial decay — pullPower controls how aggressive the dive is.
        // pow(1-t, k): k=1 → linear, k=2 → ease-in (slow then fast), etc.
        float currentR = spawnR * pow(1.0 - t, max(uPullPower, 1.0));

        // Particle dies once it crosses the event horizon.
        if (currentR < uHorizonRadius * 0.6) continue;

        // Orbital theta — accelerates as r shrinks, modelling
        // angular-momentum conservation (cheap fake of dθ/dt ∝ 1/r²).
        float currentTheta = spawnTheta
            + uOrbitSpeed * t
            + uOrbitAccel * t * t;

        // Reproject to UV.
        vec2 currentPosAR = ctr + currentR * vec2(cos(currentTheta), sin(currentTheta));
        vec2 currentUv = currentPosAR / ar;

        // Size shrinks with r.
        float rRatio = currentR / max(spawnR, 1e-3);
        float radius = uBaseSize * pow(max(rRatio, 1e-3), max(uSizeFalloff, 1e-3));

        // Birth/death taper at the lifetime endpoints, plus an extra
        // fade as r approaches the horizon.
        float lifeFade = smoothstep(0.0, 0.05, t)
                       * (1.0 - smoothstep(0.95, 1.0, t));
        float horizonFade = smoothstep(
            uHorizonRadius * 0.6,
            uHorizonRadius * 1.2,
            currentR
        );
        float fade = lifeFade * horizonFade;
        if (fade < 1e-3) continue;

        // Distance test — squared first to skip far-away particles cheaply.
        vec2 d = pos - currentPosAR;
        float distSq = dot(d, d);
        float r2 = radius * radius;
        if (distSq > r2 * 4.0) continue;

        float dist = sqrt(distSq);
        float dot1 = 1.0 - smoothstep(radius * 0.6, radius, dist);
        if (dot1 < 1e-3) continue;

        // Colour comes from the source at the spawn point — the
        // particle "carries" that pixel as it falls in.
        vec3 color = texture(uSource, spawnUv).rgb;
        accum += color * dot1 * fade;
    }

    vec3 final = bg + accum * uIntensity;

    // Horizon mask: hard black inside, smooth edge to full pass outside.
    float distToCenter = length(pos - ctr);
    float horizonGate = smoothstep(
        max(uHorizonRadius - uHorizonSoftness, 0.0),
        uHorizonRadius,
        distToCenter
    );
    final *= horizonGate;

    outColor = vec4(final, 1.0);
}
`;

const blackHoleGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: [
        "uTime",
        "uAspect",
        "uCenter",
        "uHorizonRadius",
        "uHorizonSoftness",
        "uPullPower",
        "uLifetime",
        "uOrbitSpeed",
        "uOrbitAccel",
        "uParticleCount",
        "uBaseSize",
        "uSizeFalloff",
        "uIntensity",
        "uBackgroundMix",
    ],
    setUniforms: (gl, locs, params, frame) => {
        const p: BlackHoleParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<BlackHoleParams>),
        };
        const drawW = gl.drawingBufferWidth;
        const drawH = gl.drawingBufferHeight;
        const aspect = drawW / Math.max(1, drawH);
        const count = Math.max(
            1,
            Math.min(MAX_N, Math.round(p.particleCount)),
        );
        gl.uniform1f(locs.get("uTime")!, frame.tNow);
        gl.uniform1f(locs.get("uAspect")!, aspect);
        gl.uniform2f(locs.get("uCenter")!, p.centerX, p.centerY);
        gl.uniform1f(locs.get("uHorizonRadius")!, p.horizonRadius);
        gl.uniform1f(locs.get("uHorizonSoftness")!, p.horizonSoftness);
        gl.uniform1f(locs.get("uPullPower")!, p.pullPower);
        gl.uniform1f(locs.get("uLifetime")!, p.lifetime);
        gl.uniform1f(locs.get("uOrbitSpeed")!, p.orbitSpeed);
        gl.uniform1f(locs.get("uOrbitAccel")!, p.orbitAccel);
        gl.uniform1i(locs.get("uParticleCount")!, count);
        gl.uniform1f(locs.get("uBaseSize")!, p.baseSize);
        gl.uniform1f(locs.get("uSizeFalloff")!, p.sizeFalloff);
        gl.uniform1f(locs.get("uIntensity")!, p.intensity);
        gl.uniform1f(locs.get("uBackgroundMix")!, p.backgroundMix);
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

function BlackHoleControls({ params, onChange }: ShaderControlsProps) {
    const cur: BlackHoleParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<BlackHoleParams>),
    };
    const update = (patch: Partial<BlackHoleParams>) =>
        onChange({ ...cur, ...patch } satisfies BlackHoleParams);

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

            <div style={FIRST_SECTION_TITLE_STYLE}>position · horizon</div>
            <SliderRow
                label="center x"
                value={cur.centerX}
                min={0}
                max={1}
                step={0.005}
                defaultValue={DEFAULT_PARAMS.centerX}
                onChange={(v) => update({ centerX: v })}
            />
            <SliderRow
                label="center y"
                value={cur.centerY}
                min={0}
                max={1}
                step={0.005}
                defaultValue={DEFAULT_PARAMS.centerY}
                onChange={(v) => update({ centerY: v })}
            />
            <SliderRow
                label="horizon radius"
                value={cur.horizonRadius}
                min={0}
                max={0.5}
                step={0.001}
                defaultValue={DEFAULT_PARAMS.horizonRadius}
                onChange={(v) => update({ horizonRadius: v })}
            />
            <SliderRow
                label="horizon edge softness"
                value={cur.horizonSoftness}
                min={0}
                max={0.2}
                step={0.001}
                defaultValue={DEFAULT_PARAMS.horizonSoftness}
                onChange={(v) => update({ horizonSoftness: v })}
            />

            <div style={SECTION_TITLE_STYLE}>gravity</div>
            <SliderRow
                label="pull power (curve of inward fall)"
                value={cur.pullPower}
                min={1}
                max={6}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.pullPower}
                onChange={(v) => update({ pullPower: v })}
            />
            <SliderRow
                label="lifetime (sec from spawn to event horizon)"
                value={cur.lifetime}
                min={0.5}
                max={20}
                step={0.1}
                defaultValue={DEFAULT_PARAMS.lifetime}
                format={(v) => v.toFixed(1)}
                onChange={(v) => update({ lifetime: v })}
            />

            <div style={SECTION_TITLE_STYLE}>orbital motion</div>
            <SliderRow
                label="orbit speed (rad / lifetime)"
                value={cur.orbitSpeed}
                min={0}
                max={20}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.orbitSpeed}
                onChange={(v) => update({ orbitSpeed: v })}
            />
            <SliderRow
                label="orbit acceleration (winding near center)"
                value={cur.orbitAccel}
                min={0}
                max={20}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.orbitAccel}
                onChange={(v) => update({ orbitAccel: v })}
            />

            <div style={SECTION_TITLE_STYLE}>particles</div>
            <SliderRow
                label="count"
                value={cur.particleCount}
                min={8}
                max={MAX_N}
                step={1}
                defaultValue={DEFAULT_PARAMS.particleCount}
                format={(v) => v.toFixed(0)}
                onChange={(v) =>
                    update({ particleCount: Math.round(v) })
                }
            />
            <SliderRow
                label="base size"
                value={cur.baseSize}
                min={0.001}
                max={0.05}
                step={0.0005}
                defaultValue={DEFAULT_PARAMS.baseSize}
                format={(v) => v.toFixed(3)}
                onChange={(v) => update({ baseSize: v })}
            />
            <SliderRow
                label="size falloff (1 = linear, <1 keeps big longer)"
                value={cur.sizeFalloff}
                min={0.2}
                max={3}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.sizeFalloff}
                onChange={(v) => update({ sizeFalloff: v })}
            />
            <SliderRow
                label="intensity"
                value={cur.intensity}
                min={0}
                max={5}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.intensity}
                onChange={(v) => update({ intensity: v })}
            />

            <div style={SECTION_TITLE_STYLE}>background</div>
            <SliderRow
                label="background mix (0 = black, 1 = source)"
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

export const blackHoleEntry: ShaderEntry = {
    id: "blackHole",
    name: "Black Hole (orbital particles)",
    defaultParams: DEFAULT_PARAMS,
    Controls: BlackHoleControls,
    inputs: [{ id: "in", label: "image" }],
    gpu: blackHoleGpuSpec,
};
