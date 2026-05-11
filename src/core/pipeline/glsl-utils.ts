/**
 * Aspect-correct contain ("fit") UV. Fits the source fully into the
 * destination while preserving aspect; the parts of the destination not
 * covered by the source map to UVs outside `[0,1]` and should be treated as
 * letterbox by the caller (early-out to black, transparent, etc.).
 *
 * Provides:
 *   `vec2 containUv(vec2 fragCoord, vec2 srcSize, vec2 dstSize)`
 */
export const CONTAIN_UV_GLSL = /* glsl */ `
vec2 containUv(vec2 fragCoord, vec2 srcSize, vec2 dstSize) {
    vec2 dstUv = fragCoord / dstSize;
    float srcAspect = srcSize.x / srcSize.y;
    float dstAspect = dstSize.x / dstSize.y;
    if (srcAspect > dstAspect) {
        // Source wider — fit width, letterbox top/bottom.
        float scale = srcAspect / dstAspect;
        return vec2(dstUv.x, (dstUv.y - 0.5) * scale + 0.5);
    }
    // Source taller — fit height, letterbox sides.
    float scale = dstAspect / srcAspect;
    return vec2((dstUv.x - 0.5) * scale + 0.5, dstUv.y);
}

bool isLetterbox(vec2 uv) {
    return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}
`;
