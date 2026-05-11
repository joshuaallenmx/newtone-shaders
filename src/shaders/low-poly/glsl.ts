import { CONTAIN_UV_GLSL } from "../../core/pipeline";

/**
 * Low-poly stylization — shape-from-shading approximation that turns smooth
 * regions into faceted polygons.
 *
 * For each pixel we compute a smoothed luminance gradient (from samples
 * `uSmoothRadius` source pixels away), treat its direction as a fake surface
 * normal in the XY plane, and quantize that direction into one of `uFacets`
 * angular bins. Pixels in the same bin share the same shading, producing
 * connected "facets". A Sobel-equivalent edge magnitude is overlaid so the
 * facet boundaries read as a wireframe.
 *
 * Inputs:
 *   - `iChannel0`: source media (image or video).
 *
 * Output: stylized RGB depending on `uMode` and `uColorMode`.
 *
 * Uniforms:
 *   - `uFacets`: angular bins, 4..32. @default 12
 *   - `uSmoothRadius`: gradient sample distance in source pixels. Larger →
 *      coarser polygons. @default 3
 *   - `uEdgeThreshold`: gradient magnitude below which no edge is drawn.
 *      @default 0.04
 *   - `uEdgeWidth`: smoothstep width past the threshold. @default 0.03
 *   - `uLightX` / `uLightY`: fake light direction in image space. @default 0.4 / 0.6
 *   - `uMode`: 0 facets+edges, 1 facets only, 2 wireframe (white bg, black lines).
 *   - `uColorMode`: 0 grayscale Lambertian, 1 HSV color wheel by bin.
 */
export const LOW_POLY_PASS = /* glsl */ `
${CONTAIN_UV_GLSL}

uniform int uFacets;
uniform float uSmoothRadius;
uniform float uEdgeThreshold;
uniform float uEdgeWidth;
uniform float uLightX;
uniform float uLightY;
uniform int uMode;
uniform int uColorMode;

const vec3 LUMA_709 = vec3(0.2126, 0.7152, 0.0722);
const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;

float lum(vec2 uv) {
    return dot(texture(iChannel0, uv).rgb, LUMA_709);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
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
    vec2 d = uSmoothRadius * px;

    // 4-tap central-difference gradient at radius d. Wider taps act as a
    // built-in low-pass filter so the direction quantization is stable.
    float lL = lum(uv + vec2(-d.x, 0.0));
    float lR = lum(uv + vec2( d.x, 0.0));
    float lT = lum(uv + vec2(0.0, -d.y));
    float lB = lum(uv + vec2(0.0,  d.y));
    vec2 grad = vec2(lR - lL, lB - lT);
    float mag = length(grad);

    // Quantize gradient direction. In low-magnitude (flat) regions the
    // angle is meaningless — pin to a fixed bin so flat areas don't shimmer.
    float binIndex;
    if (mag < uEdgeThreshold * 0.5) {
        binIndex = 0.0;
    } else {
        float theta = atan(grad.y, grad.x);
        float binWidth = TWO_PI / float(uFacets);
        binIndex = floor((theta + PI) / binWidth);
    }
    float binWidth = TWO_PI / float(uFacets);
    float qTheta = binIndex * binWidth - PI + binWidth * 0.5;
    vec2 qDir = vec2(cos(qTheta), sin(qTheta));

    // Lambertian fake-shade. Treat qDir as the XY of a normal, with a fixed
    // Z lean so flat-on facets aren't pure black.
    vec3 normal = normalize(vec3(qDir, 0.5));
    vec3 light = normalize(vec3(uLightX, uLightY, 1.0));
    float lambert = clamp(dot(normal, light), 0.0, 1.0);
    float shade = 0.15 + 0.85 * lambert;

    // Edge mask from gradient magnitude.
    float edge = smoothstep(uEdgeThreshold, uEdgeThreshold + uEdgeWidth, mag);

    // Per-bin color.
    vec3 facetColor;
    if (uColorMode == 1) {
        float hue = (binIndex + 0.5) / float(uFacets);
        facetColor = hsv2rgb(vec3(hue, 0.6, shade));
    } else {
        facetColor = vec3(shade);
    }

    vec3 col;
    if (uMode == 2) {
        // Wireframe: black edges on white.
        col = vec3(1.0 - edge);
    } else if (uMode == 1) {
        // Facets only — no overlay.
        col = facetColor;
    } else {
        // Facets with edge overlay.
        col = facetColor * (1.0 - edge);
    }
    fragColor = vec4(col, 1.0);
}
`;
