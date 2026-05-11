import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Skin-tone mask — chrominance-window classifier in YCbCr (BT.601).
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: grayscale skin probability in `outColor.rgb`, alpha = 1.0 (white =
 * skin, black = not skin). Two soft windows on Cb and Cr select the canonical
 * skin chroma cluster (Chai & Ngan 1999); a Y window optionally rejects very
 * dark or blown-out pixels. The source is rendered with a `contain` ("fit")
 * aspect — letterbox regions are black.
 *
 * Uniforms:
 *   - `uYMin` / `uYMax`: luminance window. @default [0, 1] (no filter)
 *   - `uCbMin` / `uCbMax`: blue-chroma window. @default [0.302, 0.498]
 *   - `uCrMin` / `uCrMax`: red-chroma window. @default [0.522, 0.678]
 *   - `uFeather`: smoothstep half-width on each window edge. @default 0.02
 */
export const SKIN_MASK_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform float uYMin;
uniform float uYMax;
uniform float uCbMin;
uniform float uCbMax;
uniform float uCrMin;
uniform float uCrMax;
uniform float uFeather;

vec3 rgbToYCbCr(vec3 rgb) {
    float y  =  0.299    * rgb.r + 0.587    * rgb.g + 0.114    * rgb.b;
    float cb = -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5      * rgb.b + 0.5;
    float cr =  0.5      * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5;
    return vec3(y, cb, cr);
}

float window(float v, float lo, float hi, float feather) {
    return smoothstep(lo - feather, lo + feather, v) *
           (1.0 - smoothstep(hi - feather, hi + feather, v));
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
    vec3 rgb = texture(iChannel0, uv).rgb;
    vec3 ycc = rgbToYCbCr(rgb);

    float yMask  = window(ycc.x, uYMin,  uYMax,  uFeather);
    float cbMask = window(ycc.y, uCbMin, uCbMax, uFeather);
    float crMask = window(ycc.z, uCrMin, uCrMax, uFeather);

    float mask = yMask * cbMask * crMask;
    fragColor = vec4(vec3(mask), 1.0);
}
`;
