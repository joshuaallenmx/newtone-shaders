import type {
    RenderOverrideContext,
    RenderOverrideSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";
import type { CSSProperties } from "react";
import { PreviewPad } from "./PreviewPad";
import { findUpstreamId } from "./findUpstream";

// Swarm — GPU particle system that settles toward a fixed `center` and
// is repelled by the live pointer. Uses the renderOverride escape hatch:
// state lives in a ping-pong RGBA32F texture (one pixel per particle =
// pos.xy + vel.xy), trails accumulate in a ping-pong RGBA8 buffer with
// per-frame multiplicative decay + additive splatting, output is the
// trail blit. ~4096+ particles run comfortably; the JS side does no
// physics — every per-particle integration happens in the simulate
// fragment shader.

const SWARM_ENTRY_ID = "swarm";

interface SwarmCenter {
    readonly x: number;
    readonly y: number;
}

interface SwarmParams {
    readonly count: number;
    readonly center: SwarmCenter;
    /** Pull toward the center (units of acceleration in world coords).
     *  Combined with `viscosity`, the terminal velocity at distance `d`
     *  is `gravity × d / viscosity` — so high viscosity + moderate
     *  gravity gives the "fall slow" feel rather than free-fall. */
    readonly gravity: number;
    readonly repel: number;
    readonly repelRadius: number;
    /** Damping rate. Acts as a terminal-velocity cap: at higher values
     *  the spring force is balanced almost instantly, so particles drift
     *  toward the centre at a soft, near-constant speed. */
    readonly viscosity: number;
    readonly pointSize: number;
    readonly trailDecay: number;
    readonly color: string;
    readonly seed: number;
    /** Time-dilation floor when the wired B/W field is fully white.
     *  Local effective dt = `dt × mix(1.0, slowFactor, fieldLum)`, so
     *  `slowFactor = 0` means white pixels freeze the simulation there
     *  and `slowFactor = 1` disables dilation. Black areas always run
     *  at full speed. Ignored when the field input isn't wired. */
    readonly slowFactor: number;
    /** When true, particles cycle in and out of existence on a
     *  deterministic per-particle period. Each particle's lifetime is
     *  picked from `[lifeMin, lifeMax]` based on a hash of its index;
     *  birth offsets are staggered so the population is always mixed
     *  across the lifetime curve. When the cycle wraps, the particle
     *  respawns at a fresh random position with zero velocity. */
    readonly lifeEnabled: boolean;
    readonly lifeMin: number;
    readonly lifeMax: number;
    /** Fade-in / fade-out durations (seconds), used by the renderer to
     *  taper alpha at the start and end of each lifetime so the cycle
     *  reads as gentle in/out rather than pop-in pop-out. */
    readonly fadeIn: number;
    readonly fadeOut: number;
    /** Where respawned particles appear. `"center"` makes the center
     *  param a continuous emission source — particles are born there
     *  with a random outward velocity. `"random"` scatters them across
     *  the canvas. */
    readonly spawnMode: "center" | "random";
    /** Initial outward speed for spawn-at-center mode (vUv units per
     *  second). Higher = wider fountain, lower = tight cluster. */
    readonly spawnSpeed: number;
}

const DEFAULT_PARAMS: SwarmParams = {
    count: 4096,
    center: { x: 0.5, y: 0.5 },
    // Soft terminal-velocity defaults: low gravity + high viscosity →
    // particles ease toward the centre at a steady, gentle speed
    // instead of accelerating into a tight orbit / overshooting.
    gravity: 0.5,
    repel: 1.4,
    repelRadius: 0.18,
    viscosity: 3.5,
    pointSize: 1.6,
    trailDecay: 0.92,
    color: "#7fc7ff",
    seed: 1,
    slowFactor: 0.05,
    lifeEnabled: true,
    lifeMin: 2.5,
    lifeMax: 6,
    fadeIn: 0.5,
    fadeOut: 0.7,
    spawnMode: "center",
    spawnSpeed: 0.4,
};

const COUNT_MIN = 256;
const COUNT_MAX = 16384;

// ─── Per-node state ─────────────────────────────────────────────────────
//
// Module-scoped Map keyed by the editor node id from FrameContext —
// matches the pattern Marbles uses, with explicit `dispose` here so the
// pipeline frees GL resources when the node is removed.

interface SwarmState {
    stateA: WebGLTexture;
    stateB: WebGLTexture;
    fboA: WebGLFramebuffer;
    fboB: WebGLFramebuffer;
    trailA: WebGLTexture;
    trailB: WebGLTexture;
    trailFboA: WebGLFramebuffer;
    trailFboB: WebGLFramebuffer;
    indexVbo: WebGLBuffer;
    /** Number of particles populated in the index VBO. */
    indexCount: number;
    stateW: number;
    stateH: number;
    /** Trail textures track the pipeline's working buffer (rectangular,
     *  matches the Global Input aspect). */
    trailW: number;
    trailH: number;
    /** 0 → A is current (most recently written); 1 → B is current. */
    statePing: 0 | 1;
    trailPing: 0 | 1;
    /** Latest seed/count we initialized state for; bumping either re-inits. */
    initSeed: number;
    initCount: number;
    initStateW: number;
    initStateH: number;
    seededOnce: boolean;
    lastTNow: number;
}

const STATE_BY_NODE = new Map<string, SwarmState>();

// ─── Shaders ────────────────────────────────────────────────────────────

const STATE_INIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outState;
uniform float uSeed;
uniform vec2 uTexSize;
uniform vec2 uCenter;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec2 idx = floor(vUv * uTexSize);
    float r1 = hash21(idx + uSeed);
    float r2 = hash21(idx + uSeed + 17.0);
    float r3 = hash21(idx + uSeed + 91.0);
    // Initial: scattered around the canvas with a small bias toward the
    // center so the initial frame looks intentional rather than uniform
    // noise. Velocity is zeroed; gravity will pull them in.
    vec2 jitter = vec2(r1, r2) - 0.5;
    vec2 pos = mix(vec2(r1, r2), uCenter + jitter * 0.4, 0.35);
    pos = clamp(pos, vec2(0.0), vec2(1.0));
    // Tiny tangential push so they don't all collapse straight in.
    vec2 toC = uCenter - pos;
    vec2 tangent = vec2(-toC.y, toC.x);
    vec2 vel = tangent * (r3 - 0.5) * 0.6;
    outState = vec4(pos, vel);
}
`;

const SIM_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outState;

uniform sampler2D uState;
uniform sampler2D uField;       // B/W: luminance → local time dilation.
                                // White ≈ slow time, black ≈ full speed.
uniform int uHasField;
uniform float uSlowFactor;      // effDt = dt × mix(1.0, uSlowFactor, lum)
                                // when field is wired. uSlowFactor = 0
                                // freezes the sim in fully-white pixels.
uniform vec2 uCenter;
uniform vec2 uPointer;
uniform int uPointerActive;
uniform float uGravity;
uniform float uRepel;
uniform float uRepelRadius;
uniform float uViscosity;
uniform float uDt;
uniform vec2 uTexSize;
uniform float uTNow;
uniform int uLifeEnabled;
uniform float uLifeMin;
uniform float uLifeMax;
uniform float uSeedJitter;
uniform int uSpawnMode;     // 0 = random, 1 = center
uniform float uSpawnSpeed;
uniform float uAspect;      // bufferW / bufferH — works in "world" coords
                            // where world.x ∈ [0, aspect], world.y ∈ [0, 1]
                            // so distance / direction math is isotropic
                            // (a circle in world space is a circle on
                            // screen, regardless of buffer aspect).

float hash11(float x) {
    return fract(sin(x * 12.9898 + 78.233 + uSeedJitter) * 43758.5453);
}

void main() {
    vec2 idx = floor(vUv * uTexSize);
    float i = idx.y * uTexSize.x + idx.x;

    vec4 prev = texture(uState, vUv);
    // Storage stays in vUv (0..1 × 0..1) so the renderer can map
    // particle positions directly to clip space without knowing aspect.
    // Physics, however, runs in *world* space where x is scaled by
    // aspect — that way distances, springs, and repulsion radii are
    // visually circular regardless of buffer shape.
    vec2 posUv = prev.xy;
    vec2 velUv = prev.zw;
    vec2 posW = vec2(posUv.x * uAspect, posUv.y);
    vec2 velW = vec2(velUv.x * uAspect, velUv.y);

    // Respawn check (in world coords for spawn-at-center fountain).
    if (uLifeEnabled == 1 && uDt > 0.0) {
        float period = uLifeMin
            + hash11(i + 31.0) * max(0.001, uLifeMax - uLifeMin);
        float birthOffset = hash11(i + 53.0) * period;
        float age = mod(uTNow - birthOffset, period);
        float prevAge = mod((uTNow - uDt) - birthOffset, period);
        if (age < prevAge) {
            float epoch = floor((uTNow - birthOffset) / period);
            if (uSpawnMode == 1) {
                // Born at the gravity center with a random outward
                // velocity in world space. The center param is in
                // vUv; promote to world.
                posW = vec2(uCenter.x * uAspect, uCenter.y);
                float angle =
                    hash11(i + epoch * 13.7 + 23.0) * 6.2831853;
                velW = vec2(cos(angle), sin(angle)) * uSpawnSpeed;
            } else {
                // Random scatter across the visible canvas.
                posW = vec2(
                    hash11(i + epoch * 13.7 + 7.0) * uAspect,
                    hash11(i + epoch * 13.7 + 19.0)
                );
                velW = vec2(0.0);
            }
        }
    }

    // Sample the field at the particle's vUv position. The wired input
    // is now interpreted as a B/W mask: luminance ∈ [0, 1] → local
    // time-dilation factor. White (1.0) ≈ uSlowFactor × dt (frozen at
    // 0), black (0.0) ≈ full dt. Acceleration / damping / advance all
    // scale by the same effective dt, so motion truly slows in white
    // regions instead of just travelling shorter distances per frame.
    vec2 currentUv = vec2(posW.x / uAspect, posW.y);
    float fieldLum = 0.0;
    if (uHasField == 1) {
        vec3 fc = texture(uField, currentUv).rgb;
        fieldLum = clamp(dot(fc, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
    }
    float effDt = uDt * mix(1.0, clamp(uSlowFactor, 0.0, 1.0), fieldLum);

    // Spring toward the center, in world. Always active — the field no
    // longer redirects attraction, only modulates how fast the local
    // physics evolves.
    vec2 centerW = vec2(uCenter.x * uAspect, uCenter.y);
    vec2 springW = centerW - posW;
    vec2 accelW = springW * uGravity;

    if (uPointerActive == 1) {
        vec2 pointerW = vec2(uPointer.x * uAspect, uPointer.y);
        vec2 from = posW - pointerW;
        float d = length(from);
        if (d > 1e-5 && d < uRepelRadius) {
            float t = 1.0 - d / uRepelRadius;
            accelW += normalize(from) * (t * t) * uRepel;
        }
    }

    velW += accelW * effDt;
    velW *= exp(-uViscosity * effDt);
    posW += velW * effDt;

    // Soft elastic bounce on the rectangular world bounds:
    // [0, aspect] × [0, 1].
    if (posW.x < 0.0) { posW.x = 0.0; velW.x = abs(velW.x) * 0.4; }
    if (posW.x > uAspect) { posW.x = uAspect; velW.x = -abs(velW.x) * 0.4; }
    if (posW.y < 0.0) { posW.y = 0.0; velW.y = abs(velW.y) * 0.4; }
    if (posW.y > 1.0) { posW.y = 1.0; velW.y = -abs(velW.y) * 0.4; }

    // Convert back to vUv for storage / rendering.
    outState = vec4(
        posW.x / uAspect,
        posW.y,
        velW.x / uAspect,
        velW.y
    );
}
`;

const TRAIL_DECAY_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTrail;
uniform float uDecay;
void main() {
    outColor = texture(uTrail, vUv) * uDecay;
}
`;

const POINT_VERT = `#version 300 es
in float aIndex;
uniform sampler2D uState;
uniform vec2 uStateSize;
uniform float uPointSize;
uniform float uTNow;
uniform int uLifeEnabled;
uniform float uLifeMin;
uniform float uLifeMax;
uniform float uFadeIn;
uniform float uFadeOut;
uniform float uSeedJitter;

out float vAlpha;

float hash11(float x) {
    return fract(sin(x * 12.9898 + 78.233 + uSeedJitter) * 43758.5453);
}

void main() {
    float W = uStateSize.x;
    float ix = mod(aIndex, W);
    float iy = floor(aIndex / W);
    vec2 stateUv = (vec2(ix, iy) + 0.5) / uStateSize;
    vec4 state = texture(uState, stateUv);
    vec2 pos = state.xy;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = uPointSize;

    // Same period / offset as the simulate pass so fade aligns with
    // respawn boundaries.
    float alpha = 1.0;
    if (uLifeEnabled == 1) {
        float period = uLifeMin
            + hash11(aIndex + 31.0) * max(0.001, uLifeMax - uLifeMin);
        float birthOffset = hash11(aIndex + 53.0) * period;
        float age = mod(uTNow - birthOffset, period);
        alpha = smoothstep(0.0, max(0.001, uFadeIn), age)
              * (1.0 - smoothstep(period - max(0.001, uFadeOut), period, age));
    }
    vAlpha = alpha;
}
`;

const POINT_FRAG = `#version 300 es
precision highp float;
in float vAlpha;
out vec4 outColor;
uniform vec3 uColor;
uniform float uAlpha;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    // Soft round splat. Edge AA via smoothstep across ~0.1 of the point.
    float a = smoothstep(0.5, 0.32, r);
    float finalA = a * uAlpha * vAlpha;
    outColor = vec4(uColor * finalA, finalA);
}
`;

const BLIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTex;
void main() {
    outColor = texture(uTex, vUv);
}
`;

// ─── Helpers ────────────────────────────────────────────────────────────

function pickStateSize(count: number): { w: number; h: number } {
    const c = Math.max(COUNT_MIN, Math.min(COUNT_MAX, count));
    const side = Math.ceil(Math.sqrt(c));
    return { w: side, h: side };
}

function parseHexColor(hex: string): [number, number, number] {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
    if (!m) return [0.5, 0.78, 1];
    const v = parseInt(m[1]!, 16);
    return [
        ((v >> 16) & 0xff) / 255,
        ((v >> 8) & 0xff) / 255,
        (v & 0xff) / 255,
    ];
}

function createFloatStateTex(
    gl: WebGL2RenderingContext,
    w: number,
    h: number,
): WebGLTexture {
    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture failed");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // NEAREST: each particle reads its own state texel exactly; no
    // interpolation between particles.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        w,
        h,
        0,
        gl.RGBA,
        gl.FLOAT,
        null,
    );
    return tex;
}

function createTrailTex(
    gl: WebGL2RenderingContext,
    w: number,
    h: number,
): WebGLTexture {
    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture failed");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA8,
        w,
        h,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
    );
    return tex;
}

function attachFbo(
    gl: WebGL2RenderingContext,
    tex: WebGLTexture,
): WebGLFramebuffer {
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error("createFramebuffer failed");
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        tex,
        0,
    );
    return fbo;
}

function createIndexVbo(
    gl: WebGL2RenderingContext,
    count: number,
): WebGLBuffer {
    const vbo = gl.createBuffer();
    if (!vbo) throw new Error("createBuffer failed");
    const data = new Float32Array(count);
    for (let i = 0; i < count; i++) data[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return vbo;
}

function ensureState(
    ctx: RenderOverrideContext,
    p: SwarmParams,
): SwarmState {
    const gl = ctx.gl;
    let state = STATE_BY_NODE.get(ctx.nodeId);
    const wantedCount = Math.max(
        COUNT_MIN,
        Math.min(COUNT_MAX, Math.round(p.count)),
    );
    const { w, h } = pickStateSize(wantedCount);

    if (!state) {
        // First-time allocation. Enable float-rendering once per context;
        // WebGL2 ships RGBA32F as renderable only via this extension.
        gl.getExtension("EXT_color_buffer_float");

        const stateA = createFloatStateTex(gl, w, h);
        const stateB = createFloatStateTex(gl, w, h);
        const fboA = attachFbo(gl, stateA);
        const fboB = attachFbo(gl, stateB);
        const trailW = ctx.bufferW;
        const trailH = ctx.bufferH;
        const trailA = createTrailTex(gl, trailW, trailH);
        const trailB = createTrailTex(gl, trailW, trailH);
        const trailFboA = attachFbo(gl, trailA);
        const trailFboB = attachFbo(gl, trailB);
        // Clear initial trails to black.
        gl.bindFramebuffer(gl.FRAMEBUFFER, trailFboA);
        gl.viewport(0, 0, trailW, trailH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindFramebuffer(gl.FRAMEBUFFER, trailFboB);
        gl.clear(gl.COLOR_BUFFER_BIT);
        const indexVbo = createIndexVbo(gl, w * h);
        state = {
            stateA,
            stateB,
            fboA,
            fboB,
            trailA,
            trailB,
            trailFboA,
            trailFboB,
            indexVbo,
            indexCount: w * h,
            stateW: w,
            stateH: h,
            trailW,
            trailH,
            statePing: 0,
            trailPing: 0,
            initSeed: NaN,
            initCount: 0,
            initStateW: w,
            initStateH: h,
            seededOnce: false,
            lastTNow: 0,
        };
        STATE_BY_NODE.set(ctx.nodeId, state);
    }

    // Resize state textures if count crossed a sqrt-bucket boundary.
    if (state.stateW !== w || state.stateH !== h) {
        gl.deleteTexture(state.stateA);
        gl.deleteTexture(state.stateB);
        gl.deleteFramebuffer(state.fboA);
        gl.deleteFramebuffer(state.fboB);
        gl.deleteBuffer(state.indexVbo);
        const stateA = createFloatStateTex(gl, w, h);
        const stateB = createFloatStateTex(gl, w, h);
        state = {
            ...state,
            stateA,
            stateB,
            fboA: attachFbo(gl, stateA),
            fboB: attachFbo(gl, stateB),
            indexVbo: createIndexVbo(gl, w * h),
            indexCount: w * h,
            stateW: w,
            stateH: h,
            statePing: 0,
            seededOnce: false,
        };
        STATE_BY_NODE.set(ctx.nodeId, state);
    }

    // Resize trail textures if the pipeline's working buffer changed.
    if (state.trailW !== ctx.bufferW || state.trailH !== ctx.bufferH) {
        gl.deleteTexture(state.trailA);
        gl.deleteTexture(state.trailB);
        gl.deleteFramebuffer(state.trailFboA);
        gl.deleteFramebuffer(state.trailFboB);
        const trailW = ctx.bufferW;
        const trailH = ctx.bufferH;
        const trailA = createTrailTex(gl, trailW, trailH);
        const trailB = createTrailTex(gl, trailW, trailH);
        const trailFboA = attachFbo(gl, trailA);
        const trailFboB = attachFbo(gl, trailB);
        gl.bindFramebuffer(gl.FRAMEBUFFER, trailFboA);
        gl.viewport(0, 0, trailW, trailH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindFramebuffer(gl.FRAMEBUFFER, trailFboB);
        gl.clear(gl.COLOR_BUFFER_BIT);
        state = {
            ...state,
            trailA,
            trailB,
            trailFboA,
            trailFboB,
            trailW,
            trailH,
            trailPing: 0,
        };
        STATE_BY_NODE.set(ctx.nodeId, state);
    }

    return state;
}

function disposeState(gl: WebGL2RenderingContext, nodeId: string): void {
    const state = STATE_BY_NODE.get(nodeId);
    if (!state) return;
    gl.deleteTexture(state.stateA);
    gl.deleteTexture(state.stateB);
    gl.deleteFramebuffer(state.fboA);
    gl.deleteFramebuffer(state.fboB);
    gl.deleteTexture(state.trailA);
    gl.deleteTexture(state.trailB);
    gl.deleteFramebuffer(state.trailFboA);
    gl.deleteFramebuffer(state.trailFboB);
    gl.deleteBuffer(state.indexVbo);
    STATE_BY_NODE.delete(nodeId);
}

function bindIndexAttribute(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    vbo: WebGLBuffer,
): void {
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const loc = gl.getAttribLocation(program, "aIndex");
    if (loc < 0) return;
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 0);
}

function disableAttribute(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    name: string,
): void {
    const loc = gl.getAttribLocation(program, name);
    if (loc >= 0) gl.disableVertexAttribArray(loc);
}

// ─── Render override ────────────────────────────────────────────────────

function render(ctx: RenderOverrideContext): void {
    const params: SwarmParams = {
        ...DEFAULT_PARAMS,
        ...(ctx.params as Partial<SwarmParams>),
    };
    const gl = ctx.gl;
    const state = ensureState(ctx, params);

    const initProg = ctx.compileProgram(
        "swarm.init",
        STATE_INIT_FRAG,
        [],
        ["uSeed", "uTexSize", "uCenter"],
    );
    const simProg = ctx.compileProgram(
        "swarm.sim",
        SIM_FRAG,
        ["uState", "uField"],
        [
            "uHasField",
            "uSlowFactor",
            "uCenter",
            "uPointer",
            "uPointerActive",
            "uGravity",
            "uRepel",
            "uRepelRadius",
            "uViscosity",
            "uDt",
            "uTexSize",
            "uTNow",
            "uLifeEnabled",
            "uLifeMin",
            "uLifeMax",
            "uSeedJitter",
            "uSpawnMode",
            "uSpawnSpeed",
            "uAspect",
        ],
    );
    const decayProg = ctx.compileProgram(
        "swarm.decay",
        TRAIL_DECAY_FRAG,
        ["uTrail"],
        ["uDecay"],
    );
    const pointProg = ctx.compileProgram(
        "swarm.point",
        POINT_FRAG,
        [],
        [
            "uState",
            "uStateSize",
            "uPointSize",
            "uColor",
            "uAlpha",
            "uTNow",
            "uLifeEnabled",
            "uLifeMin",
            "uLifeMax",
            "uFadeIn",
            "uFadeOut",
            "uSeedJitter",
        ],
        POINT_VERT,
    );
    const blitProg = ctx.compileProgram(
        "swarm.blit",
        BLIT_FRAG,
        ["uTex"],
        [],
    );

    const wantedCount = Math.max(
        COUNT_MIN,
        Math.min(COUNT_MAX, Math.round(params.count)),
    );
    const drawCount = Math.min(wantedCount, state.indexCount);
    const dt =
        state.lastTNow > 0
            ? Math.max(0, Math.min(0.05, ctx.tNow - state.lastTNow))
            : 0;
    state.lastTNow = ctx.tNow;

    // ─── Phase 0: (re)initialize state on first run / seed change ───
    const needsInit =
        !state.seededOnce ||
        state.initSeed !== params.seed ||
        state.initCount !== wantedCount;
    if (needsInit) {
        const writeFbo = state.fboA;
        gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
        gl.viewport(0, 0, state.stateW, state.stateH);
        gl.disable(gl.BLEND);
        gl.useProgram(initProg.program);
        ctx.bindQuadAttribute(initProg.program);
        gl.uniform1f(
            initProg.uniformLocs.get("uSeed")!,
            params.seed * 0.137 + 1,
        );
        gl.uniform2f(
            initProg.uniformLocs.get("uTexSize")!,
            state.stateW,
            state.stateH,
        );
        gl.uniform2f(
            initProg.uniformLocs.get("uCenter")!,
            params.center.x,
            params.center.y,
        );
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        state.statePing = 0;
        state.seededOnce = true;
        state.initSeed = params.seed;
        state.initCount = wantedCount;
    }

    // ─── Phase 1: simulate ────────────────────────────────────────────
    const readState = state.statePing === 0 ? state.stateA : state.stateB;
    const writeFbo = state.statePing === 0 ? state.fboB : state.fboA;
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
    gl.viewport(0, 0, state.stateW, state.stateH);
    gl.disable(gl.BLEND);
    gl.useProgram(simProg.program);
    ctx.bindQuadAttribute(simProg.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readState);
    gl.uniform1i(simProg.samplerLocs[0]!, 0);
    // uField goes to TEXTURE1. When the input isn't wired, bind any
    // valid texture as a placeholder (the read state tex is fine; the
    // sample result gets multiplied by `uHasField = 0` in the shader).
    const fieldTex = ctx.inputTextures[0];
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, fieldTex ?? readState);
    gl.uniform1i(simProg.samplerLocs[1]!, 1);
    gl.uniform1i(
        simProg.uniformLocs.get("uHasField")!,
        fieldTex ? 1 : 0,
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uSlowFactor")!,
        Math.max(0, Math.min(1, params.slowFactor)),
    );
    gl.uniform2f(
        simProg.uniformLocs.get("uCenter")!,
        params.center.x,
        params.center.y,
    );
    gl.uniform2f(
        simProg.uniformLocs.get("uPointer")!,
        ctx.pointer.uv[0],
        ctx.pointer.uv[1],
    );
    gl.uniform1i(
        simProg.uniformLocs.get("uPointerActive")!,
        ctx.pointer.active ? 1 : 0,
    );
    gl.uniform1f(simProg.uniformLocs.get("uGravity")!, params.gravity);
    gl.uniform1f(simProg.uniformLocs.get("uRepel")!, params.repel);
    gl.uniform1f(
        simProg.uniformLocs.get("uRepelRadius")!,
        Math.max(0.001, params.repelRadius),
    );
    gl.uniform1f(simProg.uniformLocs.get("uViscosity")!, params.viscosity);
    gl.uniform1f(simProg.uniformLocs.get("uDt")!, dt);
    gl.uniform2f(
        simProg.uniformLocs.get("uTexSize")!,
        state.stateW,
        state.stateH,
    );
    gl.uniform1f(simProg.uniformLocs.get("uTNow")!, ctx.tNow);
    gl.uniform1i(
        simProg.uniformLocs.get("uLifeEnabled")!,
        params.lifeEnabled ? 1 : 0,
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uLifeMin")!,
        Math.max(0.05, params.lifeMin),
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uLifeMax")!,
        Math.max(params.lifeMin + 0.01, params.lifeMax),
    );
    // Mix the user seed into hash11 so re-rolling produces a different
    // population layout. Same value also fed to the renderer so the
    // fade curves stay in lockstep with the respawn boundaries.
    const seedJitter = (params.seed * 0.7193) % 6.2832;
    gl.uniform1f(simProg.uniformLocs.get("uSeedJitter")!, seedJitter);
    gl.uniform1i(
        simProg.uniformLocs.get("uSpawnMode")!,
        params.spawnMode === "center" ? 1 : 0,
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uSpawnSpeed")!,
        Math.max(0, params.spawnSpeed),
    );
    const aspect = ctx.bufferW / Math.max(1, ctx.bufferH);
    gl.uniform1f(simProg.uniformLocs.get("uAspect")!, aspect);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    state.statePing = state.statePing === 0 ? 1 : 0;
    const currentState = state.statePing === 0 ? state.stateA : state.stateB;

    // ─── Phase 2: decay trail (read prev → write next) ───────────────
    const trailRead = state.trailPing === 0 ? state.trailA : state.trailB;
    const trailWriteFbo =
        state.trailPing === 0 ? state.trailFboB : state.trailFboA;
    gl.bindFramebuffer(gl.FRAMEBUFFER, trailWriteFbo);
    gl.viewport(0, 0, state.trailW, state.trailH);
    gl.disable(gl.BLEND);
    gl.useProgram(decayProg.program);
    ctx.bindQuadAttribute(decayProg.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, trailRead);
    gl.uniform1i(decayProg.samplerLocs[0]!, 0);
    gl.uniform1f(
        decayProg.uniformLocs.get("uDecay")!,
        Math.max(0, Math.min(0.999, params.trailDecay)),
    );
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // ─── Phase 3: splat points (additive blend onto trail) ──────────
    // Same FBO is bound from phase 2.
    gl.useProgram(pointProg.program);
    bindIndexAttribute(gl, pointProg.program, state.indexVbo);
    // Disable aPosition so the previous program's attribute doesn't
    // try to read from the index VBO with stride/offset that don't
    // match.
    disableAttribute(gl, pointProg.program, "aPosition");
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentState);
    gl.uniform1i(pointProg.uniformLocs.get("uState")!, 0);
    gl.uniform2f(
        pointProg.uniformLocs.get("uStateSize")!,
        state.stateW,
        state.stateH,
    );
    gl.uniform1f(
        pointProg.uniformLocs.get("uPointSize")!,
        Math.max(0.5, params.pointSize),
    );
    const [r, g, b] = parseHexColor(params.color);
    gl.uniform3f(pointProg.uniformLocs.get("uColor")!, r, g, b);
    gl.uniform1f(pointProg.uniformLocs.get("uAlpha")!, 1.0);
    gl.uniform1f(pointProg.uniformLocs.get("uTNow")!, ctx.tNow);
    gl.uniform1i(
        pointProg.uniformLocs.get("uLifeEnabled")!,
        params.lifeEnabled ? 1 : 0,
    );
    gl.uniform1f(
        pointProg.uniformLocs.get("uLifeMin")!,
        Math.max(0.05, params.lifeMin),
    );
    gl.uniform1f(
        pointProg.uniformLocs.get("uLifeMax")!,
        Math.max(params.lifeMin + 0.01, params.lifeMax),
    );
    gl.uniform1f(
        pointProg.uniformLocs.get("uFadeIn")!,
        Math.max(0.001, params.fadeIn),
    );
    gl.uniform1f(
        pointProg.uniformLocs.get("uFadeOut")!,
        Math.max(0.001, params.fadeOut),
    );
    gl.uniform1f(pointProg.uniformLocs.get("uSeedJitter")!, seedJitter);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, drawCount);
    gl.disable(gl.BLEND);
    state.trailPing = state.trailPing === 0 ? 1 : 0;
    const currentTrail =
        state.trailPing === 0 ? state.trailA : state.trailB;

    // ─── Phase 4: blit trail to outputFbo ───────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.outputFbo);
    gl.viewport(0, 0, ctx.bufferW, ctx.bufferH);
    gl.useProgram(blitProg.program);
    ctx.bindQuadAttribute(blitProg.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentTrail);
    gl.uniform1i(blitProg.samplerLocs[0]!, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

const swarmRenderOverride: RenderOverrideSpec = {
    render,
    dispose: disposeState,
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

function SwarmControls({
    params,
    onChange,
    nodes,
    edges,
    nodeId,
}: ShaderControlsProps) {
    const cur: SwarmParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<SwarmParams>),
        center: {
            ...DEFAULT_PARAMS.center,
            ...((params as Partial<SwarmParams>)?.center ?? {}),
        },
    };
    const update = (patch: Partial<SwarmParams>) =>
        onChange({ ...cur, ...patch });
    const upstreamId = findUpstreamId(nodes, edges, nodeId);

    return (
        <div>
            <div style={LABEL_STYLE}>
                center (drag — used when no field is wired)
            </div>
            <PreviewPad
                value={cur.center}
                onChange={(c) => update({ center: c })}
                nodeId={upstreamId}
                dotColor="#ffd86b"
            />
            <div style={LABEL_STYLE}>count</div>
            <Slider
                value={cur.count}
                min={COUNT_MIN}
                max={COUNT_MAX}
                step={64}
                onChange={(n) => update({ count: Math.round(n) })}
            />
            <div style={LABEL_STYLE}>gravity (pull toward center)</div>
            <Slider
                value={cur.gravity}
                min={0}
                max={6}
                step={0.05}
                onChange={(n) => update({ gravity: n })}
            />
            <div style={LABEL_STYLE}>pointer repel strength</div>
            <Slider
                value={cur.repel}
                min={0}
                max={6}
                step={0.05}
                onChange={(n) => update({ repel: n })}
            />
            <div style={LABEL_STYLE}>repel radius</div>
            <Slider
                value={cur.repelRadius}
                min={0.01}
                max={0.6}
                step={0.005}
                onChange={(n) => update({ repelRadius: n })}
            />
            <div style={LABEL_STYLE}>viscosity (settling speed)</div>
            <Slider
                value={cur.viscosity}
                min={0}
                max={8}
                step={0.05}
                onChange={(n) => update({ viscosity: n })}
            />
            <div style={LABEL_STYLE}>
                slow factor (white = freeze, black = full speed)
            </div>
            <Slider
                value={cur.slowFactor}
                min={0}
                max={1}
                step={0.005}
                onChange={(n) => update({ slowFactor: n })}
            />
            <div style={LABEL_STYLE}>point size (px)</div>
            <Slider
                value={cur.pointSize}
                min={0.5}
                max={6}
                step={0.1}
                onChange={(n) => update({ pointSize: n })}
            />
            <div style={LABEL_STYLE}>trail decay (per frame)</div>
            <Slider
                value={cur.trailDecay}
                min={0}
                max={0.999}
                step={0.001}
                onChange={(n) => update({ trailDecay: n })}
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

            <label
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    color: "#bbb",
                    fontSize: 12,
                    marginTop: 12,
                    cursor: "pointer",
                }}
            >
                <input
                    type="checkbox"
                    checked={cur.lifeEnabled}
                    onChange={(e) =>
                        update({ lifeEnabled: e.target.checked })
                    }
                />
                life cycle (respawn + fade)
            </label>
            {cur.lifeEnabled ? (
                <>
                    <div style={LABEL_STYLE}>spawn from</div>
                    <select
                        style={{
                            background: "#0a0a0a",
                            color: "#f0f0f0",
                            border: "1px solid #333",
                            borderRadius: 4,
                            padding: "4px 6px",
                            fontSize: 12,
                            fontFamily: "inherit",
                            width: "100%",
                            boxSizing: "border-box",
                        }}
                        value={cur.spawnMode}
                        onChange={(e) =>
                            update({
                                spawnMode: e.target.value as
                                    | "center"
                                    | "random",
                            })
                        }
                    >
                        <option value="center">center (fountain)</option>
                        <option value="random">random scatter</option>
                    </select>
                    {cur.spawnMode === "center" ? (
                        <>
                            <div style={LABEL_STYLE}>
                                spawn speed (initial outward push)
                            </div>
                            <Slider
                                value={cur.spawnSpeed}
                                min={0}
                                max={2}
                                step={0.01}
                                onChange={(n) =>
                                    update({ spawnSpeed: n })
                                }
                            />
                        </>
                    ) : null}
                    <div style={LABEL_STYLE}>life min (s)</div>
                    <Slider
                        value={cur.lifeMin}
                        min={0.1}
                        max={Math.max(0.5, cur.lifeMax - 0.1)}
                        step={0.05}
                        onChange={(n) => update({ lifeMin: n })}
                    />
                    <div style={LABEL_STYLE}>life max (s)</div>
                    <Slider
                        value={cur.lifeMax}
                        min={Math.max(0.2, cur.lifeMin + 0.1)}
                        max={20}
                        step={0.05}
                        onChange={(n) => update({ lifeMax: n })}
                    />
                    <div style={LABEL_STYLE}>fade in (s)</div>
                    <Slider
                        value={cur.fadeIn}
                        min={0.001}
                        max={3}
                        step={0.01}
                        onChange={(n) => update({ fadeIn: n })}
                    />
                    <div style={LABEL_STYLE}>fade out (s)</div>
                    <Slider
                        value={cur.fadeOut}
                        min={0.001}
                        max={3}
                        step={0.01}
                        onChange={(n) => update({ fadeOut: n })}
                    />
                </>
            ) : null}
            <div style={HINT_STYLE}>
                Particles always feel a soft pull toward the centre.
                Wire a B/W mask into the field input to slow them down
                in the white regions — useful for keeping clouds of
                particles parked over highlights, mattes, or any
                hand-drawn region.
                <br />
                <br />
                Life cycle staggers per-particle lifetimes; combined with
                trail decay it gives a continuous breathing flow as new
                particles fade in at random spots and old ones fade out.
            </div>
        </div>
    );
}

export const swarmEntry: ShaderEntry = {
    id: SWARM_ENTRY_ID,
    name: "Swarm",
    defaultParams: DEFAULT_PARAMS,
    Controls: SwarmControls,
    inputs: [{ id: "in", label: "speed map (B/W)", optional: true }],
    renderOverride: swarmRenderOverride,
};

export type { SwarmParams, SwarmCenter };
