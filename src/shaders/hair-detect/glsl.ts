import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Hair-detection layer — local luminance-variance detector with luma and
 * saturation gates. Hair has high-frequency directional structure and
 * relatively low saturation; skin and plain backgrounds are smooth. We
 * compute the standard deviation of luminance over a 5×5 kernel sized in
 * source pixels, smoothstep it into a probability, then multiply by a luma
 * window and a saturation cap.
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: grayscale "hair-likeness" — white where likely hair, black else.
 * Letterbox regions are always black.
 *
 * Uniforms:
 *   - `uKernelRadius`: half-width of the 5×5 kernel in source pixels. Larger
 *      values capture coarser texture (good for low-res / blurry hair). @default 2
 *   - `uTextureGain`: linear multiplier on σ before threshold. @default 6
 *   - `uTextureFloor`: smoothstep low edge — gradients below this clip. @default 0.20
 *   - `uTextureCeil`: smoothstep high edge — full intensity above. @default 1.0
 *   - `uSaturationMax`: HSV saturation upper bound; pixels above are
 *      excluded (rejects bright accents like lips). @default 0.7
 *   - `uLumaMin` / `uLumaMax`: brightness window; rejects shadows / blown
 *      highlights. @default 0.0 / 1.0
 */
export const HAIR_DETECT_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform float uKernelRadius;
uniform float uTextureGain;
uniform float uTextureFloor;
uniform float uTextureCeil;
uniform float uSaturationMax;
uniform float uLumaMin;
uniform float uLumaMax;

const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);

float lum(vec3 rgb) {
    return dot(rgb, LUMA_709);
}

float hsvSat(vec3 rgb) {
    float maxC = max(max(rgb.r, rgb.g), rgb.b);
    float minC = min(min(rgb.r, rgb.g), rgb.b);
    return (maxC > 0.0) ? (maxC - minC) / maxC : 0.0;
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

    vec2 px = (uKernelRadius * 0.5) / srcSize;

    // 5x5 sample grid spans uKernelRadius source pixels on each side of the
    // center. dx,dy in {-2,-1,0,1,2}; step length = uKernelRadius * 0.5 / srcSize.
    float sumL  = 0.0;
    float sumL2 = 0.0;
    vec3  sumRgb = vec3(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 offset = vec2(float(dx), float(dy)) * px;
            vec3 c = texture(iChannel0, uv + offset).rgb;
            float L = lum(c);
            sumL  += L;
            sumL2 += L * L;
            sumRgb += c;
        }
    }

    const float n = 25.0;
    float meanL = sumL / n;
    float variance = max((sumL2 / n) - meanL * meanL, 0.0);
    float stddev = sqrt(variance);
    vec3 meanRgb = sumRgb / n;

    float texture = smoothstep(uTextureFloor, uTextureCeil, stddev * uTextureGain);

    float lumaMask = step(uLumaMin, meanL) * (1.0 - step(uLumaMax, meanL));
    float satMask = 1.0 - smoothstep(uSaturationMax, uSaturationMax + 0.05, hsvSat(meanRgb));

    float hair = texture * lumaMask * satMask;
    fragColor = vec4(vec3(hair), 1.0);
}
`;
