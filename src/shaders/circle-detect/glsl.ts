import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Pass 1 — Sobel edge magnitude on Rec. 709 luminance, written into the
 * intermediate render target so the Hough pass can sample it. The contain UV
 * is applied here, so the edge texture is in canvas-space with letterbox
 * regions clamped to 0.
 */
export const CIRCLE_EDGE_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);

float lum(vec2 uv) {
    return dot(texture(iChannel0, uv).rgb, LUMA_709);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 srcSize = vec2(textureSize(iChannel0, 0));
    if (srcSize.x < 1.0 || srcSize.y < 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 uv = containUv(fragCoord, srcSize, iResolution.xy);
    if (isLetterbox(uv)) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 px = 1.0 / srcSize;

    float tl = lum(uv + vec2(-px.x, -px.y));
    float t  = lum(uv + vec2(   0.0, -px.y));
    float tr = lum(uv + vec2( px.x, -px.y));
    float l  = lum(uv + vec2(-px.x,    0.0));
    float r  = lum(uv + vec2( px.x,    0.0));
    float bl = lum(uv + vec2(-px.x,  px.y));
    float b  = lum(uv + vec2(   0.0,  px.y));
    float br = lum(uv + vec2( px.x,  px.y));

    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
    float g = sqrt(gx * gx + gy * gy);
    fragColor = vec4(vec3(g), 1.0);
}
`;

/**
 * Pass 2 — Circular Hough accumulator. For each output pixel we sample the
 * edge texture at `uSamples` evenly-spaced points on a circle of radius
 * `uRadius` (in canvas pixels). Pixels surrounded by strong edges at that
 * radius get a high score — i.e. they're plausible circle centers.
 *
 * Multiple radii are swept across `[uRadius - uRadiusSpread, uRadius +
 * uRadiusSpread]` and the best score is kept, so circles slightly off the
 * target radius still register.
 *
 * Output: grayscale accumulator score, no mode switching. The render pass
 * handles all visualization.
 *
 * Inputs:
 *   - `iChannel0`: edge magnitude from pass 1 (canvas resolution).
 *
 * Uniforms:
 *   - `uRadius`: target circle radius in canvas pixels. @default 60
 *   - `uRadiusSpread`: half-width of the radius sweep. @default 10
 *   - `uSamples`: angular samples per radius (1..MAX_SAMPLES). @default 48
 */
export const CIRCLE_HOUGH_PASS = /* glsl */ `
uniform float uRadius;
uniform float uRadiusSpread;
uniform int uSamples;

const int MAX_SAMPLES = 128;
const int RADIUS_STEPS = 4;
const float TWO_PI = 6.28318530718;

float circleScore(vec2 fragCoord, vec2 dstSize, float radius, int samples) {
    float sum = 0.0;
    int counted = 0;
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (i >= samples) break;
        float theta = float(i) * TWO_PI / float(samples);
        vec2 offset = vec2(cos(theta), sin(theta)) * radius;
        vec2 sampleUv = (fragCoord + offset) / dstSize;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 ||
            sampleUv.y < 0.0 || sampleUv.y > 1.0) continue;
        sum += texture(iChannel0, sampleUv).r;
        counted++;
    }
    return counted == 0 ? 0.0 : sum / float(counted);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 dstSize = iResolution.xy;
    float bestScore = 0.0;
    if (uRadiusSpread < 0.5) {
        bestScore = circleScore(fragCoord, dstSize, uRadius, uSamples);
    } else {
        for (int i = 0; i < RADIUS_STEPS; i++) {
            float frac = float(i) / float(RADIUS_STEPS - 1);
            float r = uRadius + (frac * 2.0 - 1.0) * uRadiusSpread;
            r = max(r, 1.0);
            float s = circleScore(fragCoord, dstSize, r, uSamples);
            bestScore = max(bestScore, s);
        }
    }
    fragColor = vec4(vec3(bestScore), 1.0);
}
`;

/**
 * Pass 3 — Final composite. Reads the Hough accumulator (iChannel0), the
 * edge texture (iChannel1, debug), and the source media (iChannel2). Emits
 * one of several visualizations selected by `uMode`.
 *
 * Vector-quality circle strokes are *not* rendered here — they're drawn on
 * a 2D-canvas overlay above the WebGL canvas after a CPU readback of the
 * accumulator. This pass just provides the underlying picture (or
 * diagnostics).
 *
 * Output modes:
 *   - 0 (accumulator): grayscale Hough heatmap.
 *   - 1 (mask): binary — white where score > `uMinScore`.
 *   - 2 (overlay): source dimmed by `uOffMix`, accumulator added on top.
 *   - 3 (edges): pass 1 output — for tuning.
 *   - 4 (source): unmodified source media. Use this when the 2D overlay
 *      should sit over a clean image.
 *
 * Uniforms:
 *   - `uMinScore`: threshold for mask mode.
 *   - `uMode`: 0..4 as above.
 *   - `uOffMix`: source brightness in overlay mode.
 */
export const CIRCLE_RENDER_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform float uMinScore;
uniform int uMode;
uniform float uOffMix;

vec3 sourceAt(vec2 fragCoord, vec2 dstSize) {
    vec2 srcSize = vec2(textureSize(iChannel2, 0));
    vec2 sourceUv = containUv(fragCoord, srcSize, dstSize);
    if (isLetterbox(sourceUv)) return vec3(0.0);
    return texture(iChannel2, sourceUv).rgb;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 dstSize = iResolution.xy;
    vec2 uv = fragCoord / dstSize;

    if (uMode == 3) {
        fragColor = vec4(vec3(texture(iChannel1, uv).r), 1.0);
        return;
    }
    if (uMode == 4) {
        fragColor = vec4(sourceAt(fragCoord, dstSize), 1.0);
        return;
    }
    if (uMode == 0) {
        fragColor = vec4(vec3(texture(iChannel0, uv).r), 1.0);
        return;
    }
    if (uMode == 1) {
        float v = texture(iChannel0, uv).r;
        fragColor = vec4(vec3(step(uMinScore, v)), 1.0);
        return;
    }
    // uMode == 2 (overlay)
    vec3 src = sourceAt(fragCoord, dstSize);
    float v = texture(iChannel0, uv).r;
    fragColor = vec4(clamp(src * uOffMix + vec3(v), 0.0, 1.0), 1.0);
}
`;
