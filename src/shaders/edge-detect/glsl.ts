import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Edge-detection layer — Sobel operator on Rec. 709 luminance.
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: grayscale edge intensity in `outColor.rgb`, alpha = 1.0. The source
 * is rendered with a `contain` ("fit") aspect — letterbox regions are black.
 *
 * Uniforms:
 *   - `uEdgeStrength`: linear multiplier on the gradient magnitude. @default 1.0
 *   - `uEdgeThreshold`: smoothstep low-edge for noise rejection. @default 0.0
 *   - `uEdgeKnee`: smoothstep high-edge — full intensity above this. @default 1.0
 */
export const EDGE_DETECT_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform float uEdgeStrength;
uniform float uEdgeThreshold;
uniform float uEdgeKnee;

const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);

float luma(vec2 uv) {
    return dot(texture(iChannel0, uv).rgb, LUMA_709);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 srcSize = vec2(textureSize(iChannel0, 0));
    if (srcSize.x < 1.0 || srcSize.y < 1.0) {
        // Source isn't ready yet (e.g. video metadata not loaded).
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 uv = containUv(fragCoord, srcSize, iResolution.xy);
    if (isLetterbox(uv)) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 px = 1.0 / srcSize;

    // 3x3 neighborhood luminance.
    float tl = luma(uv + vec2(-px.x, -px.y));
    float t  = luma(uv + vec2(   0.0, -px.y));
    float tr = luma(uv + vec2( px.x, -px.y));
    float l  = luma(uv + vec2(-px.x,    0.0));
    float r  = luma(uv + vec2( px.x,    0.0));
    float bl = luma(uv + vec2(-px.x,  px.y));
    float b  = luma(uv + vec2(   0.0,  px.y));
    float br = luma(uv + vec2( px.x,  px.y));

    // Sobel kernels.
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
    float g = sqrt(gx * gx + gy * gy) * uEdgeStrength;

    float edge = smoothstep(uEdgeThreshold, uEdgeKnee, g);
    fragColor = vec4(vec3(edge), 1.0);
}
`;
