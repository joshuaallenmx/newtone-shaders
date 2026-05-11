import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Per-channel posterization — the classical "Posterize" filter.
 *
 * For each pixel, each RGB channel is independently rounded to one of
 * `uLevels` values evenly distributed in `[0, 1]`. Total possible output
 * colors: `uLevels^3` (e.g. 4 → 64, 6 → 216, 8 → 512). Color regions can't
 * be "lost" because there's no palette selection — every value snaps to the
 * nearest cell of the regular grid.
 *
 * Quantization happens in **sRGB (display) space**, not linear, so bands are
 * perceptually evenly spaced. The renderer's `outputColorSpace = SRGBColorSpace`
 * gives us a linear-RGB sample on `texture()`, so we transform: linear →
 * sRGB → quantize → linear → output (which the renderer re-encodes for
 * display).
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: posterized RGB with alpha = 1.0. Letterbox regions are black.
 *
 * Uniforms:
 *   - `uLevels`: levels per channel, expected `>= 2`. @default 4
 */
export const POSTERIZE_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform float uLevels;

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

    vec3 rgbLinear = texture(iChannel0, uv).rgb;
    vec3 rgbSrgb = pow(max(rgbLinear, 0.0), vec3(1.0 / 2.2));
    float steps = max(uLevels - 1.0, 1.0);
    vec3 quantSrgb = clamp(round(rgbSrgb * steps) / steps, 0.0, 1.0);
    vec3 outLinear = pow(quantSrgb, vec3(2.2));
    fragColor = vec4(outLinear, 1.0);
}
`;
