import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Palette mask — quantize each pixel to its nearest entry in a CPU-supplied
 * palette and either show the posterized result or a binary mask of the
 * "enabled" entries.
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: depends on `uMode`:
 *   - 0 (posterize): pixel = its nearest palette color, dimmed if disabled.
 *   - 1 (mask): white if the pixel's nearest palette color is enabled, else black.
 *   - 2 (overlay): source RGB if enabled, else darkened to `uOffMix * source`.
 *
 * Source is rendered with `contain` ("fit") aspect — letterbox regions are
 * always black regardless of mode.
 *
 * Uniforms:
 *   - `uPalette[16]`: palette colors in linear-RGB-ish [0,1]. Slots beyond
 *      `uPaletteSize` are ignored.
 *   - `uEnabled[16]`: 1.0 / 0.0 per palette slot.
 *   - `uPaletteSize`: active count, 1..16.
 *   - `uMode`: 0/1/2 as above.
 *   - `uOffMix`: brightness multiplier on the "off" portion in posterize/overlay modes. @default 0.15
 */
export const PALETTE_MASK_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

const int PALETTE_MAX = 16;

uniform vec3 uPalette[PALETTE_MAX];
uniform float uEnabled[PALETTE_MAX];
uniform int uPaletteSize;
uniform int uMode;
uniform float uOffMix;

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

    int best = 0;
    float bestDist = 1e10;
    for (int i = 0; i < PALETTE_MAX; i++) {
        if (i >= uPaletteSize) break;
        vec3 d = rgb - uPalette[i];
        float dist = dot(d, d);
        if (dist < bestDist) {
            bestDist = dist;
            best = i;
        }
    }

    float enabled = uEnabled[best];

    if (uMode == 1) {
        // Mask
        fragColor = vec4(vec3(enabled), 1.0);
    } else if (uMode == 2) {
        // Overlay: source where enabled, dimmed source elsewhere.
        vec3 col = mix(rgb * uOffMix, rgb, enabled);
        fragColor = vec4(col, 1.0);
    } else {
        // Posterize: snapped palette color, dimmed when disabled.
        vec3 col = uPalette[best] * mix(uOffMix, 1.0, enabled);
        fragColor = vec4(col, 1.0);
    }
}
`;
