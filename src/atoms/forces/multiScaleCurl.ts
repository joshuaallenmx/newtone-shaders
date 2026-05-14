import type { Atom } from "../types";

/**
 * The heart of the flockaroo single-pass CFD. Samples the previous frame's
 * stored velocity at 5 angular positions around expanding octaves
 * (`b *= 2.0` per octave, up to 20 octaves capped by canvas size), takes a
 * discrete curl by dotting each sample with a perpendicular offset, and
 * accumulates the result into a new velocity `v`.
 *
 * Three small but important details:
 *   • Each octave's curl contribution is divided by `dot(b,b)` (squared
 *     radius) so far-out octaves don't dominate — they contribute scaled
 *     down by their reach.
 *   • The half-angle rotation `mh` (36°) decorrelates inner-ring samples
 *     from outer-ring samples so the resulting field has no rotational
 *     artifact at multiples of 72°.
 *   • The initial `b = cos(iFrame*.3 - vec2(0, 1.57))` rotates with frame
 *     count, so the curl-sampling pattern itself drifts over time — even
 *     a static state produces evolving velocity.
 *
 * Reads: `sampler2D iChannel0` (previous frame velocity in `.xy`)
 * Writes: `vec2 v` (this frame's velocity estimate)
 *
 * Source: [glsl.ts:31-59](../../shaders/liquid-metal/glsl.ts#L31-L59).
 */
export const multiScaleCurl: Atom = {
    id: "multiScaleCurl",
    definitions: `
        #ifndef MSC_PI2
        #define MSC_PI2 6.283185
        #endif
        #ifndef MSC_RotNum
        #define MSC_RotNum 5
        #endif
        const float msc_ang = MSC_PI2 / float(MSC_RotNum);
        const mat2 msc_m = mat2(cos(msc_ang), sin(msc_ang), -sin(msc_ang), cos(msc_ang));
        const mat2 msc_mh = mat2(cos(msc_ang * 0.5), sin(msc_ang * 0.5), -sin(msc_ang * 0.5), cos(msc_ang * 0.5));

        float msc_getRot(vec2 pos, vec2 b, vec2 res) {
            vec2 p = b;
            float rot = 0.0;
            for (int i = 0; i < MSC_RotNum; i++) {
                rot += dot(
                    textureLod(iChannel0, (pos + p) / res, 0.0).xy - vec2(0.5),
                    p.yx * vec2(1.0, -1.0)
                );
                p = msc_m * p;
            }
            return rot / float(MSC_RotNum) / dot(b, b);
        }
    `,
    body: `
        vec2 res = vec2(textureSize(iChannel0, 0));
        vec2 b = cos(float(iFrame) * 0.3 - vec2(0.0, 1.57));
        vec2 v = vec2(0.0);
        float bbMax = 0.5 * res.y; bbMax *= bbMax;
        for (int l = 0; l < 20; l++) {
            if (dot(b, b) > bbMax) break;
            vec2 p = b;
            for (int i = 0; i < MSC_RotNum; i++) {
                v += p.yx * msc_getRot(pos + p, -msc_mh * b, res);
                p = msc_m * p;
            }
            b *= 2.0;
        }
    `,
};
