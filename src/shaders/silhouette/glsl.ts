import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Silhouette tracer — background subtraction by reference-color distance,
 * then outlines the resulting foreground/background boundary.
 *
 * The source is pre-blurred over a 5×5 kernel (radius `uSmoothRadius` in
 * source pixels) before distance-from-reference is measured, so small
 * surface features (highlights, skin texture, specks) don't create internal
 * edges. RGB distance from `uReferenceColor` is then run through a
 * smoothstep to produce a soft mask (1 = foreground, 0 = background). The
 * silhouette is the *gradient* of that mask — non-zero only where the mask
 * transitions, i.e. the subject's major outline.
 *
 * The "stable" mode aggregates outlines across `STABILITY_SAMPLES` threshold
 * values within `±uThresholdSpread` of `uThreshold` and returns the mean.
 * Real boundaries score high (they read as edges across many thresholds);
 * texture noise scores low (it shifts position with each threshold). This is
 * threshold-stability used as an edge-confidence measure — see Witkin (1983)
 * on scale-space stability.
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: depends on `uMode`:
 *   - 0 (outline): black bg, white silhouette line at the chosen threshold.
 *   - 1 (mask): white where foreground, black where background.
 *   - 2 (key): source RGB on foreground, black background.
 *   - 3 (overlay): source RGB on foreground, dimmed source on background,
 *      with the silhouette drawn over both.
 *   - 4 (stable): outline averaged across `STABILITY_SAMPLES` thresholds in
 *      `[uThreshold - uThresholdSpread, uThreshold + uThresholdSpread]`.
 *      Wired as a confidence map — bright = edge survived all thresholds.
 *
 * Uniforms:
 *   - `uReferenceColor`: vec3 in [0,1] — the assumed background color.
 *   - `uSmoothRadius`: pre-blur radius in source pixels — set higher to
 *      suppress fine surface detail. @default 4
 *   - `uThreshold`: distance above which a pixel is "foreground". @default 0.18
 *   - `uFeather`: smoothstep half-width on the threshold edge. @default 0.04
 *   - `uThresholdSpread`: half-width of the threshold sweep in stable mode. @default 0.1
 *   - `uOutlineThickness`: gradient sample radius in source pixels. @default 1.5
 *   - `uMode`: 0/1/2/3/4 as above.
 *   - `uOffMix`: brightness multiplier on the background in overlay mode. @default 0.15
 */
export const SILHOUETTE_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform vec3 uReferenceColor;
uniform float uSmoothRadius;
uniform float uThreshold;
uniform float uFeather;
uniform float uThresholdSpread;
uniform float uOutlineThickness;
uniform int uMode;
uniform float uOffMix;

const int STABILITY_SAMPLES = 8;

vec3 blurredSourceAt(vec2 uv, vec2 srcSize) {
    if (uSmoothRadius < 0.001) {
        return texture(iChannel0, uv).rgb;
    }
    // 5x5 box-ish blur, samples spaced at uSmoothRadius/2 source pixels.
    vec2 d = (uSmoothRadius * 0.5) / srcSize;
    vec3 sum = vec3(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            sum += texture(iChannel0, uv + vec2(float(dx), float(dy)) * d).rgb;
        }
    }
    return sum / 25.0;
}

float maskAt(vec2 uv, vec2 srcSize) {
    vec3 c = blurredSourceAt(uv, srcSize);
    float dist = length(c - uReferenceColor);
    return smoothstep(uThreshold - uFeather, uThreshold + uFeather, dist);
}

float distanceAt(vec2 uv, vec2 srcSize) {
    vec3 c = blurredSourceAt(uv, srcSize);
    return length(c - uReferenceColor);
}

float stableOutline(vec2 uv, vec2 srcSize, vec2 thickStep) {
    // Compute the blurred-source distance ONCE at each gradient sample
    // position, then evaluate the mask at multiple thresholds cheaply.
    float distL = distanceAt(uv + vec2(-thickStep.x, 0.0), srcSize);
    float distR = distanceAt(uv + vec2( thickStep.x, 0.0), srcSize);
    float distT = distanceAt(uv + vec2(0.0, -thickStep.y), srcSize);
    float distB = distanceAt(uv + vec2(0.0,  thickStep.y), srcSize);

    float sum = 0.0;
    for (int i = 0; i < STABILITY_SAMPLES; i++) {
        float frac = float(i) / float(STABILITY_SAMPLES - 1);
        float t = uThreshold + (frac * 2.0 - 1.0) * uThresholdSpread;
        float mL = smoothstep(t - uFeather, t + uFeather, distL);
        float mR = smoothstep(t - uFeather, t + uFeather, distR);
        float mT = smoothstep(t - uFeather, t + uFeather, distT);
        float mB = smoothstep(t - uFeather, t + uFeather, distB);
        vec2 grad = vec2(mR - mL, mB - mT);
        sum += clamp(length(grad), 0.0, 1.0);
    }
    return sum / float(STABILITY_SAMPLES);
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
    vec2 d = uOutlineThickness * px;

    if (uMode == 4) {
        float stability = stableOutline(uv, srcSize, d);
        fragColor = vec4(vec3(stability), 1.0);
        return;
    }

    float center = maskAt(uv, srcSize);
    float ml = maskAt(uv + vec2(-d.x, 0.0), srcSize);
    float mr = maskAt(uv + vec2( d.x, 0.0), srcSize);
    float mt = maskAt(uv + vec2(0.0, -d.y), srcSize);
    float mb = maskAt(uv + vec2(0.0,  d.y), srcSize);
    vec2 grad = vec2(mr - ml, mb - mt);
    float outline = clamp(length(grad), 0.0, 1.0);

    if (uMode == 1) {
        fragColor = vec4(vec3(center), 1.0);
        return;
    }

    vec3 rgb = texture(iChannel0, uv).rgb;

    if (uMode == 2) {
        fragColor = vec4(rgb * center, 1.0);
        return;
    }

    if (uMode == 3) {
        // Overlay: foreground at full brightness, background dimmed, with
        // silhouette burned in white on top.
        vec3 base = mix(rgb * uOffMix, rgb, center);
        vec3 col = mix(base, vec3(1.0), outline);
        fragColor = vec4(col, 1.0);
        return;
    }

    // Default — outline only on black.
    fragColor = vec4(vec3(outline), 1.0);
}
`;
