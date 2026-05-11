import type {
    RenderOverrideContext,
    RenderOverrideSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";
import type { CSSProperties } from "react";
import { PreviewPad } from "./PreviewPad";
import { findUpstreamId } from "./findUpstream";

// ContourFlow — particles ride the iso-lines of a scalar input field
// (typically a depth map, but anything 0..1 works: luminance, gravity,
// hand-drawn masks). Visually equivalent to extracting contours via
// marching squares and orbiting particles along them, but the contours
// are *implicit*: each particle's velocity each frame is the **tangent
// to the local gradient** — perpendicular to ∇field — which by
// definition follows the level set of `field`. A weak perpendicular
// "bind" force corrects drift back to the particle's assigned target
// depth, so each particle stays on its band even after many laps.
//
// Storage: one RGBA32F per particle — `(posUv.xy, targetDepth, dir)`.
// `dir = ±1` picks orbit handedness (some clockwise, some not, so the
// flow doesn't all run in lockstep). No velocity is integrated; the
// flow is direct displacement along the tangent each frame, which is
// stable and never overshoots like spring/damper systems do.

const ENTRY_ID = "contour-flow";

interface FlowCenter {
    readonly x: number;
    readonly y: number;
}

interface ContourFlowParams {
    readonly count: number;
    /** Reference point shown on the inspector pad — purely cosmetic
     *  (helps you visualize where you're working when no field is
     *  wired). Particles don't anchor to this. */
    readonly reference: FlowCenter;
    /** Tangent flow speed in vUv units per second. */
    readonly flowSpeed: number;
    /** How strongly the perpendicular bind force pulls a wandering
     *  particle back to its target depth. 0 = no correction (drifts
     *  freely), high = sharp clamp to the contour. */
    readonly bindStrength: number;
    /** Sampling step (pixels) for the gradient finite-difference. 1 =
     *  per-pixel; larger smooths over noise in the depth map. */
    readonly gradientStep: number;
    /** Pointer repulsion within `repelRadius`, pushes particles
     *  outward (vUv per second). */
    readonly repel: number;
    readonly repelRadius: number;
    /** Particle splat size in pixels. */
    readonly pointSize: number;
    /** Per-frame trail multiplicative decay. */
    readonly trailDecay: number;
    readonly color: string;
    readonly seed: number;
    /** Lifecycle: when enabled, particles cycle in/out on per-particle
     *  random periods and respawn at fresh random scatter positions
     *  with a fresh target-depth lock. */
    readonly lifeEnabled: boolean;
    readonly lifeMin: number;
    readonly lifeMax: number;
    readonly fadeIn: number;
    readonly fadeOut: number;
    /** How wide a band around the target depth a particle should treat
     *  as "on contour" — used by the bind force to scale the pull. */
    readonly bandWidth: number;
}

const DEFAULT_PARAMS: ContourFlowParams = {
    count: 4096,
    reference: { x: 0.5, y: 0.5 },
    flowSpeed: 0.08,
    bindStrength: 1.5,
    gradientStep: 1.5,
    repel: 1.4,
    repelRadius: 0.18,
    pointSize: 1.6,
    trailDecay: 0.92,
    color: "#7fc7ff",
    seed: 1,
    lifeEnabled: true,
    lifeMin: 4,
    lifeMax: 9,
    fadeIn: 0.6,
    fadeOut: 0.9,
    bandWidth: 0.05,
};

const COUNT_MIN = 256;
const COUNT_MAX = 16384;

interface FlowState {
    stateA: WebGLTexture;
    stateB: WebGLTexture;
    fboA: WebGLFramebuffer;
    fboB: WebGLFramebuffer;
    trailA: WebGLTexture;
    trailB: WebGLTexture;
    trailFboA: WebGLFramebuffer;
    trailFboB: WebGLFramebuffer;
    indexVbo: WebGLBuffer;
    indexCount: number;
    stateW: number;
    stateH: number;
    trailW: number;
    trailH: number;
    statePing: 0 | 1;
    trailPing: 0 | 1;
    initSeed: number;
    initCount: number;
    seededOnce: boolean;
    /** Set to false on init / respawn-all. The init pass needs the field
     *  texture to sample target depths; if the field is the missing-
     *  input placeholder (1×1 black) on the very first frame, every
     *  particle locks to depth = 0 and they all collapse onto whatever
     *  contour is at field == 0. We re-init once a real field arrives. */
    seedFieldKey: WebGLTexture | null;
    lastTNow: number;
}

const STATE_BY_NODE = new Map<string, FlowState>();

const STATE_INIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outState;
uniform float uSeed;
uniform vec2 uTexSize;
uniform sampler2D uField;

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
    // Random scatter across the canvas. Sample the field at the spawn
    // point — that's the iso-line this particle will follow forever.
    vec2 pos = vec2(r1, r2);
    vec3 fc = texture(uField, pos).rgb;
    float depth = clamp(dot(fc, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
    float dir = r3 < 0.5 ? -1.0 : 1.0;
    outState = vec4(pos, depth, dir);
}
`;

const SIM_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outState;

uniform sampler2D uState;
uniform sampler2D uField;
uniform int uHasField;
uniform vec2 uPointer;
uniform int uPointerActive;
uniform float uFlowSpeed;
uniform float uBindStrength;
uniform float uBandWidth;
uniform float uGradientStep;
uniform float uRepel;
uniform float uRepelRadius;
uniform float uDt;
uniform vec2 uTexSize;
uniform float uTNow;
uniform int uLifeEnabled;
uniform float uLifeMin;
uniform float uLifeMax;
uniform float uSeedJitter;
uniform float uAspect;

float hash11(float x) {
    return fract(sin(x * 12.9898 + 78.233 + uSeedJitter) * 43758.5453);
}

float sampleDepth(vec2 uv) {
    vec3 c = texture(uField, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
    return clamp(dot(c, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
}

void main() {
    vec2 idx = floor(vUv * uTexSize);
    float i = idx.y * uTexSize.x + idx.x;

    vec4 prev = texture(uState, vUv);
    vec2 pos = prev.xy;
    float targetDepth = prev.z;
    float dir = prev.w;

    // ─── Lifecycle respawn ────────────────────────────────────────────
    if (uLifeEnabled == 1 && uDt > 0.0) {
        float period = uLifeMin
            + hash11(i + 31.0) * max(0.001, uLifeMax - uLifeMin);
        float birthOffset = hash11(i + 53.0) * period;
        float age = mod(uTNow - birthOffset, period);
        float prevAge = mod((uTNow - uDt) - birthOffset, period);
        if (age < prevAge) {
            float epoch = floor((uTNow - birthOffset) / period);
            // Fresh random scatter + re-lock to whatever depth lives
            // there. Each respawn shuffles the contour assignment.
            pos = vec2(
                hash11(i + epoch * 13.7 + 7.0),
                hash11(i + epoch * 13.7 + 19.0)
            );
            float r3 = hash11(i + epoch * 13.7 + 41.0);
            dir = r3 < 0.5 ? -1.0 : 1.0;
            if (uHasField == 1) {
                targetDepth = sampleDepth(pos);
            }
        }
    }

    if (uHasField == 1) {
        // Finite-difference gradient. Step is in pixel space so the
        // gradient stays meaningful regardless of buffer resolution.
        vec2 step = uGradientStep / vec2(textureSize(uField, 0));
        float dxR = sampleDepth(pos + vec2(step.x, 0.0));
        float dxL = sampleDepth(pos - vec2(step.x, 0.0));
        float dyU = sampleDepth(pos + vec2(0.0, step.y));
        float dyD = sampleDepth(pos - vec2(0.0, step.y));
        // ∇field in vUv units. We work in vUv-with-aspect so distances
        // and tangents look isotropic on screen.
        vec2 gradUv = vec2(dxR - dxL, dyU - dyD) * 0.5 / step;
        vec2 gradAspect = vec2(gradUv.x / uAspect, gradUv.y);
        float gradMag = length(gradAspect);

        vec2 flow = vec2(0.0);
        if (gradMag > 1e-4) {
            // Tangent to grad-field is the iso-line direction. The
            // 90deg flip gives the natural contour-following motion;
            // multiplying by 'dir' lets half the population run the
            // other way so contours don't read as a single conveyor.
            vec2 tangentAspect =
                vec2(-gradAspect.y, gradAspect.x) / gradMag;
            // Bring back to vUv space (tangent's x was in aspect units).
            vec2 tangentUv = vec2(tangentAspect.x * uAspect, tangentAspect.y);
            flow += tangentUv * uFlowSpeed * dir;

            // Bind force: pull perpendicular to the contour back toward
            // targetDepth. Sign of (current - target) tells which side
            // of the contour the particle is on; ∇field points "uphill"
            // so subtract that direction when current > target.
            float currentDepth = sampleDepth(pos);
            float depthError = currentDepth - targetDepth;
            float bindFactor =
                clamp(depthError / max(1e-4, uBandWidth), -1.0, 1.0);
            vec2 bindDirAspect = -gradAspect / gradMag;
            vec2 bindDirUv =
                vec2(bindDirAspect.x * uAspect, bindDirAspect.y);
            flow += bindDirUv * bindFactor * uBindStrength * uFlowSpeed;
        }
        // Flat zones (|∇| ≈ 0) → no flow, particle parks. Combined with
        // lifecycle respawn this prevents permanent stalls.

        // Pointer repel — purely a viewer-driven nudge, in vUv-aspect.
        if (uPointerActive == 1) {
            vec2 fromAspect = vec2(
                (pos.x - uPointer.x) * uAspect,
                pos.y - uPointer.y
            );
            float d = length(fromAspect);
            if (d > 1e-5 && d < uRepelRadius) {
                float t = 1.0 - d / uRepelRadius;
                vec2 repelAspect = fromAspect / d * (t * t) * uRepel;
                vec2 repelUv =
                    vec2(repelAspect.x / uAspect, repelAspect.y);
                flow += repelUv;
            }
        }

        pos += flow * uDt;
    }

    // Soft wrap on the canvas edges. Hard clamp would pile particles up
    // on the borders and the bind force would never recover — wrapping
    // gives them a clean re-entry on the opposite side.
    pos = fract(pos);

    outState = vec4(pos, targetDepth, dir);
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
out float vDepth;

float hash11(float x) {
    return fract(sin(x * 12.9898 + 78.233 + uSeedJitter) * 43758.5453);
}

void main() {
    float W = uStateSize.x;
    float ix = mod(aIndex, W);
    float iy = floor(aIndex / W);
    vec2 stateUv = (vec2(ix, iy) + 0.5) / uStateSize;
    vec4 state = texture(uState, stateUv);
    gl_Position = vec4(state.xy * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = uPointSize;
    vDepth = state.z;

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
in float vDepth;
out vec4 outColor;
uniform vec3 uColor;
uniform float uAlpha;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    float a = smoothstep(0.5, 0.32, r);
    // Modulate brightness by depth so different bands read as different
    // shades of the chosen colour — gives the iso-line orbit a sense of
    // depth without needing per-band hue control. Keeps the high end
    // close to the picked colour and gently darkens toward 0.
    float depthShade = mix(0.55, 1.0, vDepth);
    float finalA = a * uAlpha * vAlpha;
    outColor = vec4(uColor * depthShade * finalA, finalA);
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
    p: ContourFlowParams,
): FlowState {
    const gl = ctx.gl;
    let state = STATE_BY_NODE.get(ctx.nodeId);
    const wantedCount = Math.max(
        COUNT_MIN,
        Math.min(COUNT_MAX, Math.round(p.count)),
    );
    const { w, h } = pickStateSize(wantedCount);

    if (!state) {
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
            seededOnce: false,
            seedFieldKey: null,
            lastTNow: 0,
        };
        STATE_BY_NODE.set(ctx.nodeId, state);
    }

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
            seedFieldKey: null,
        };
        STATE_BY_NODE.set(ctx.nodeId, state);
    }

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

function render(ctx: RenderOverrideContext): void {
    const params: ContourFlowParams = {
        ...DEFAULT_PARAMS,
        ...(ctx.params as Partial<ContourFlowParams>),
    };
    const gl = ctx.gl;
    const state = ensureState(ctx, params);
    const fieldTex = ctx.inputTextures[0] ?? null;

    const initProg = ctx.compileProgram(
        "contourFlow.init",
        STATE_INIT_FRAG,
        ["uField"],
        ["uSeed", "uTexSize"],
    );
    const simProg = ctx.compileProgram(
        "contourFlow.sim",
        SIM_FRAG,
        ["uState", "uField"],
        [
            "uHasField",
            "uPointer",
            "uPointerActive",
            "uFlowSpeed",
            "uBindStrength",
            "uBandWidth",
            "uGradientStep",
            "uRepel",
            "uRepelRadius",
            "uDt",
            "uTexSize",
            "uTNow",
            "uLifeEnabled",
            "uLifeMin",
            "uLifeMax",
            "uSeedJitter",
            "uAspect",
        ],
    );
    const decayProg = ctx.compileProgram(
        "contourFlow.decay",
        TRAIL_DECAY_FRAG,
        ["uTrail"],
        ["uDecay"],
    );
    const pointProg = ctx.compileProgram(
        "contourFlow.point",
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
        "contourFlow.blit",
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

    // ─── Phase 0: (re)init when seed/count/field-arrival changes ────
    const fieldChanged =
        fieldTex !== null && fieldTex !== state.seedFieldKey;
    const needsInit =
        !state.seededOnce ||
        state.initSeed !== params.seed ||
        state.initCount !== wantedCount ||
        fieldChanged;
    if (needsInit) {
        const writeFbo = state.fboA;
        gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
        gl.viewport(0, 0, state.stateW, state.stateH);
        gl.disable(gl.BLEND);
        gl.useProgram(initProg.program);
        ctx.bindQuadAttribute(initProg.program);
        gl.activeTexture(gl.TEXTURE0);
        // Bind the placeholder if the field isn't wired yet — every
        // particle ends up on `depth = 0`. We re-init the moment a real
        // field becomes available.
        gl.bindTexture(gl.TEXTURE_2D, fieldTex ?? state.stateA);
        gl.uniform1i(initProg.samplerLocs[0]!, 0);
        gl.uniform1f(
            initProg.uniformLocs.get("uSeed")!,
            params.seed * 0.137 + 1,
        );
        gl.uniform2f(
            initProg.uniformLocs.get("uTexSize")!,
            state.stateW,
            state.stateH,
        );
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        state.statePing = 0;
        state.seededOnce = true;
        state.initSeed = params.seed;
        state.initCount = wantedCount;
        state.seedFieldKey = fieldTex;
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
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, fieldTex ?? readState);
    gl.uniform1i(simProg.samplerLocs[1]!, 1);
    gl.uniform1i(
        simProg.uniformLocs.get("uHasField")!,
        fieldTex ? 1 : 0,
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
    gl.uniform1f(simProg.uniformLocs.get("uFlowSpeed")!, params.flowSpeed);
    gl.uniform1f(
        simProg.uniformLocs.get("uBindStrength")!,
        params.bindStrength,
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uBandWidth")!,
        Math.max(0.001, params.bandWidth),
    );
    gl.uniform1f(
        simProg.uniformLocs.get("uGradientStep")!,
        Math.max(0.5, params.gradientStep),
    );
    gl.uniform1f(simProg.uniformLocs.get("uRepel")!, params.repel);
    gl.uniform1f(
        simProg.uniformLocs.get("uRepelRadius")!,
        Math.max(0.001, params.repelRadius),
    );
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
    const seedJitter = (params.seed * 0.7193) % 6.2832;
    gl.uniform1f(simProg.uniformLocs.get("uSeedJitter")!, seedJitter);
    const aspect = ctx.bufferW / Math.max(1, ctx.bufferH);
    gl.uniform1f(simProg.uniformLocs.get("uAspect")!, aspect);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    state.statePing = state.statePing === 0 ? 1 : 0;
    const currentState = state.statePing === 0 ? state.stateA : state.stateB;

    // ─── Phase 2: trail decay ─────────────────────────────────────────
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

    // ─── Phase 3: splat points ────────────────────────────────────────
    gl.useProgram(pointProg.program);
    bindIndexAttribute(gl, pointProg.program, state.indexVbo);
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

const contourFlowRenderOverride: RenderOverrideSpec = {
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

function ContourFlowControls({
    params,
    onChange,
    nodes,
    edges,
    nodeId,
}: ShaderControlsProps) {
    const cur: ContourFlowParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<ContourFlowParams>),
        reference: {
            ...DEFAULT_PARAMS.reference,
            ...((params as Partial<ContourFlowParams>)?.reference ?? {}),
        },
    };
    const update = (patch: Partial<ContourFlowParams>) =>
        onChange({ ...cur, ...patch });
    const upstreamId = findUpstreamId(nodes, edges, nodeId);

    return (
        <div>
            <div style={LABEL_STYLE}>field preview</div>
            <PreviewPad
                value={cur.reference}
                onChange={(c) => update({ reference: c })}
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
            <div style={LABEL_STYLE}>flow speed (along contour)</div>
            <Slider
                value={cur.flowSpeed}
                min={0}
                max={0.5}
                step={0.001}
                onChange={(n) => update({ flowSpeed: n })}
            />
            <div style={LABEL_STYLE}>bind strength (stay on contour)</div>
            <Slider
                value={cur.bindStrength}
                min={0}
                max={6}
                step={0.05}
                onChange={(n) => update({ bindStrength: n })}
            />
            <div style={LABEL_STYLE}>band width</div>
            <Slider
                value={cur.bandWidth}
                min={0.005}
                max={0.5}
                step={0.005}
                onChange={(n) => update({ bandWidth: n })}
            />
            <div style={LABEL_STYLE}>gradient step (px)</div>
            <Slider
                value={cur.gradientStep}
                min={0.5}
                max={6}
                step={0.1}
                onChange={(n) => update({ gradientStep: n })}
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
                title="Increment seed to redistribute particles across new contours"
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
                Wire any scalar map (depth, luminance, gravity) into the
                input. Each particle locks to its spawn-point's value
                and orbits along that iso-line — no SVG extraction
                needed; the contours emerge from the gradient field.
                Bind strength controls how tightly the particle hugs
                its contour vs. wandering across bands.
            </div>
        </div>
    );
}

export const contourFlowEntry: ShaderEntry = {
    id: ENTRY_ID,
    name: "Contour Flow",
    defaultParams: DEFAULT_PARAMS,
    Controls: ContourFlowControls,
    inputs: [{ id: "in", label: "field (scalar)", optional: true }],
    renderOverride: contourFlowRenderOverride,
};

export type { ContourFlowParams, FlowCenter };
