// ─────────────────────────────────────────────────────────────────────────────
// LICENSE NOTE — adapted from "Single pass CFD" by Florian Berger ("flockaroo"),
// 2019. Original work licensed under Creative Commons Attribution-NonCommercial-
// ShareAlike 3.0 Unported (CC BY-NC-SA 3.0):
//     https://creativecommons.org/licenses/by-nc-sa/3.0/
// This adaptation retains the NC attribution header until the project's own
// rewrite of BUFFER_A supersedes the original solver. See README.md.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Buffer A — fluid simulation. Reads its own previous frame (iChannel0), an
 * RGBA noise texture (iChannel1), a barrier mask (iChannel2 — white = solid),
 * and the mouse-delta tracker (iChannel3, see BUFFER_B).
 */
export const LIQUID_METAL_BUFFER_A = /* glsl */ `
#define PI2 6.283185
#define Res0 vec2(textureSize(iChannel0,0))
#define Res1 vec2(textureSize(iChannel1,0))
#define RotNum 5

#define maskTex iChannel2

uniform float uViscosity;
uniform float uAdvectionScale;
uniform float uPointerForce;
uniform float uAmbientFlow;
uniform float uDropAmplitudeNear;
uniform float uDropAmplitudeFar;
uniform float uScrollForce;

const float ang = PI2/float(RotNum);
const mat2 m = mat2(cos(ang),sin(ang),-sin(ang),cos(ang));
const mat2 mh = mat2(cos(ang*0.5),sin(ang*0.5),-sin(ang*0.5),cos(ang*0.5));

float getRot(vec2 pos, vec2 b) {
    float l = log2(dot(b,b))*sqrt(.125)*.0;
    vec2 p = b;
    float rot = 0.0;
    for (int i = 0; i < RotNum; i++) {
        rot += dot(textureLod(iChannel0,(pos+p)/Res0.xy,l).xy-vec2(0.5), p.yx*vec2(1,-1));
        p = m*p;
    }
    return rot/float(RotNum)/dot(b,b);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 pos = fragCoord;
    vec2 b = cos(float(iFrame)*.3 - vec2(0,1.57));
    vec2 v = vec2(0);
    float bbMax = .5*Res0.y; bbMax *= bbMax;
    for (int l = 0; l < 20; l++) {
        if (dot(b,b) > bbMax) break;
        vec2 p = b;
        for (int i = 0; i < RotNum; i++) {
            v += p.yx * getRot(pos+p, -mh*b);
            p = m*p;
        }
        b *= 2.0;
    }

    // Advect: drop fract() so iChannel0's CLAMP_TO_EDGE wrap mode turns the
    // canvas borders into hard edges instead of wrapping around.
    fragColor = textureLod(iChannel0, (pos - v*vec2(-1,1)*uAdvectionScale*sqrt(Res0.x/600.))/Res0.xy, 0.);
    fragColor.xy = mix(fragColor.xy, v*vec2(-1,1)*sqrt(.125)*.9, uViscosity);

    vec2 c = iMouse.xy;
    vec2 dmouse = texelFetch(iChannel3, ivec2(0), 0).zw;
    if (iMouse.x < 1.) c = Res0*.5;
    vec2 scr = fract((fragCoord.xy - c)/Res0.x + .5) - .5;
    if (iMouse.x < 1.) fragColor.xy += uAmbientFlow*cos(iTime*.3 - vec2(0,1.57)) / (dot(scr,scr)/0.05 + .05);
    fragColor.xy += uPointerForce*dmouse / (dot(scr,scr)/0.05 + .05);
    fragColor.xy += uScrollForce*vec2(0.0, iScrollVelocity.y) / (dot(scr,scr)/0.05 + .05);

    fragColor.zw += (texture(iChannel1, fragCoord/Res1*.35).zw - .5)*uDropAmplitudeNear;
    fragColor.zw += (texture(iChannel1, fragCoord/Res1*.7).zw - .5)*uDropAmplitudeFar;

    // Visual collision: zero velocity and clear surface inside barrier shapes.
    float mask = texture(maskTex, fragCoord.xy / iResolution.xy).r;
    float barrier = smoothstep(0.05, 0.6, mask);
    fragColor.xy = mix(fragColor.xy, vec2(0.0), barrier);
    fragColor.zw = mix(fragColor.zw, vec2(0.0), barrier);

    if (iFrame <= 4) fragColor = vec4(0);
}
`;

/**
 * Buffer B — stores current mouse pos in xy and per-frame delta in zw at
 * texel (0,0). Sampled by Buffer A as a sub-frame mouse motion source.
 */
export const LIQUID_METAL_BUFFER_B = /* glsl */ `
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec4 c = texelFetch(iChannel0, ivec2(0), 0);
    vec2 m = iMouse.xy;
    vec2 d = vec2(0);
    if (iMouse.xy != iMouse.zw) { d = iMouse.xy - c.xy; }
    fragColor = vec4(m, d);
}
`;

/**
 * Image — shades Buffer A as bismuth-ish liquid metal. iChannel2 is sampled
 * along the reflection vector as a 2D stand-in for a cubemap.
 */
export const LIQUID_METAL_IMAGE = /* glsl */ `
#define Res (iResolution.xy)

uniform vec3 uTint;
uniform vec3 uBaseColor;
uniform float uContrast;
uniform float uSaturation;
uniform float uGradientDelta;
uniform float uEnvBrightnessBoost;

vec4 myenv(vec3 dir) {
    return texture(iChannel2, dir.xy*0.5 + 0.5) + uEnvBrightnessBoost;
}

vec4 getCol(vec2 uv) { return texture(iChannel0, uv); }
float getVal(vec2 uv) { return length(getCol(uv).xyz); }

vec2 getGrad(vec2 uv, float delta) {
    vec2 d = vec2(delta, 0);
    return vec2(getVal(uv+d.xy)-getVal(uv-d.xy),
                getVal(uv+d.yx)-getVal(uv-d.yx)) / delta;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 n = vec3(-getGrad(uv, uGradientDelta/iResolution.x)*.02, 1.);
    n = normalize(n);

    vec2 sc = (fragCoord - Res*.5)/Res.x;
    vec3 dir = normalize(vec3(sc, -1.));
    vec3 R = reflect(dir, n);
    vec3 refl = myenv(R.xzy).xyz;

    vec3 col = getCol(uv).xyz + .5;
    col = mix(uBaseColor, col, uContrast);
    col *= .95 + -.05*n;

    vec3 outRgb = col * refl * uTint;
    // Desaturate toward Rec.709 luminance when uSaturation < 1.
    float luma = dot(outRgb, vec3(0.2126, 0.7152, 0.0722));
    fragColor.xyz = mix(vec3(luma), outRgb, uSaturation);
    fragColor.w = 1.;
}
`;
